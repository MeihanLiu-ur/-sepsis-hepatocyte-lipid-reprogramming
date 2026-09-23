#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
20_zonation_firstlook.py -- 区带 × 疾病状态:第一版结果(快跑版)

定位
  这是**预跑**,目的是尽快回答核心假说,不追求最终出图质量。
  打分用简化 module score(基因集内 log 表达的均值),AUCell 版见后续脚本。
  规划文档 §2 已定:主打分用 AUCell(基于秩,不受基因集大小与稀疏度影响),
  简化 score 仅作一致性交叉验证 —— 本脚本输出的就是这个交叉验证的雏形。

做什么
  1. 从 merged.h5ad 取肝细胞
  2. 算 portal_score / central_score / zonation index = central - portal
  3. 在 **Sham 组**按分位数标定 zone1/2/3,把同一套切点套到 CLP 组
     (关键:两组共用切点,否则 collapse 会被切点自身的移动掩盖)
  4. 比较两组 zonation index 的**分布形态** —— collapse 的判据是
     方差坍缩 / 双峰消失,而不是三个硬分类的比例变化
  5. 8 个脂质程序按 zone × 分组打分

用法: python scripts/20_zonation_firstlook.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
import scanpy as sc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_sepsis import ZONE, LIPID_PROGRAMS, PROC_DIR, FIG_DIR  # noqa: E402

IN = os.path.join(PROC_DIR, "merged.h5ad")


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def mscore(A, genes):
    """简化 module score: 基因集内 log 表达的均值(忽略未匹配基因)"""
    present = [g for g in genes if g in set(A.var_names)]
    if not present:
        return np.zeros(A.n_obs), []
    idx = [list(A.var_names).index(g) for g in present]
    s = np.asarray(A.X[:, idx].mean(axis=1)).ravel()
    return np.asarray(s).ravel(), present


def main():
    log("读取", IN)
    A = sc.read_h5ad(IN)
    hep = A[A.obs["celltype"] == "Hepatocyte"].copy()
    log(f"肝细胞: {hep.n_obs:,}  "
        f"(CLP {int((hep.obs['group']=='CLP').sum()):,} / "
        f"Sham {int((hep.obs['group']=='Sham').sum()):,})")

    # ---------- 1. 区带打分 ----------
    log("计算 portal / central 打分")
    portal, gp = mscore(hep, ZONE["periportal"])
    central, gc = mscore(hep, ZONE["pericentral"])
    log(f"  periportal 命中 {len(gp)}/{len(ZONE['periportal'])}; "
        f"pericentral 命中 {len(gc)}/{len(ZONE['pericentral'])}")

    hep.obs["portal_score"] = portal
    hep.obs["central_score"] = central
    hep.obs["zonation_index"] = central - portal

    zi = hep.obs["zonation_index"].values
    grp = hep.obs["group"].values

    # ---------- 2. 分布形态比较(核心:collapse 判据) ----------
    log()
    log("=" * 68)
    log("核心结果 1:zonation index 分布形态 (collapse 判据)")
    log("=" * 68)
    rows = []
    for g in ["Sham", "CLP"]:
        v = zi[grp == g]
        rows.append(dict(group=g, n=len(v), mean=v.mean(), sd=v.std(),
                         iqr=np.subtract(*np.percentile(v, [75, 25])),
                         q05=np.percentile(v, 5), q25=np.percentile(v, 25),
                         median=np.median(v), q75=np.percentile(v, 75),
                         q95=np.percentile(v, 95)))
    df = pd.DataFrame(rows).set_index("group")
    for line in df.round(3).to_string().split("\n"):
        log("  " + line)

    sd_sham = zi[grp == "Sham"].std()
    sd_clp = zi[grp == "CLP"].std()
    log()
    log(f"  标准差比 CLP/Sham = {sd_clp/sd_sham:.3f}  "
        f"(<1 表示 CLP 区带分布**收窄** = collapse)")
    log(f"  IQR   比 CLP/Sham = {df.loc['CLP','iqr']/df.loc['Sham','iqr']:.3f}")
    log(f"  均值差 CLP-Sham   = {df.loc['CLP','mean']-df.loc['Sham','mean']:+.3f}")

    # 样本层(n=3 vs 3),不把细胞数当 n
    log()
    log("  样本层均值(每个样本一个值, 这才是统计单位):")
    per_sample = hep.obs.groupby(["group", "sample"])["zonation_index"].agg(
        ["count", "mean", "std"]).round(3)
    for line in per_sample.to_string().split("\n"):
        log("    " + line)
    sh = per_sample.loc["Sham", "mean"].values
    cl = per_sample.loc["CLP", "mean"].values
    log(f"    Sham 样本均值: {np.round(sh,3).tolist()}  mean={sh.mean():.3f} sd={sh.std(ddof=1):.3f}")
    log(f"    CLP  样本均值: {np.round(cl,3).tolist()}  mean={cl.mean():.3f} sd={cl.std(ddof=1):.3f}")
    log(f"    样本层差异 = {cl.mean()-sh.mean():+.3f}  (n=3 vs 3, 未做检验, 仅看方向)")

    # ---------- 3. 在 Sham 标定 zone,套用到 CLP ----------
    log()
    log("=" * 68)
    log("核心结果 2:在 Sham 组按三分位标定 zone,套用到 CLP 组")
    log("=" * 68)
    sham_zi = zi[grp == "Sham"]
    c1, c2 = np.percentile(sham_zi, [33.3, 66.7])
    log(f"  Sham 分位切点: {c1:.3f} / {c2:.3f}   (zone1 = 最负 = 门静脉周; zone3 = 最正 = 中央静脉周)")

    def zonify(v):
        return np.where(v < c1, "zone1", np.where(v < c2, "zone2", "zone3"))
    hep.obs["zone"] = zonify(zi)

    tab = pd.crosstab(hep.obs["zone"], hep.obs["group"])
    tab_pct = (tab / tab.sum() * 100).round(1)
    log()
    log("  各 zone 细胞数:")
    for line in tab.to_string().split("\n"):
        log("    " + line)
    log("  各组内占比 (%):")
    for line in tab_pct.to_string().split("\n"):
        log("    " + line)

    # ---------- 4. 脂质程序 × zone × 分组 ----------
    log()
    log("=" * 68)
    log("核心结果 3:脂质程序打分 (zone × 分组)")
    log("=" * 68)
    for prog, gl in LIPID_PROGRAMS.items():
        s, hit = mscore(hep, gl)
        hep.obs[f"LP_{prog}"] = s

    lp_cols = [f"LP_{p}" for p in LIPID_PROGRAMS]
    tab_lp = hep.obs.groupby(["zone", "group"])[lp_cols].mean()
    log()
    log("  平均 program score (行 = zone × 分组):")
    for line in tab_lp.round(3).to_string().split("\n"):
        log("    " + line)

    # zone3 vs zone1 的落差,分组比较 —— 这是"zone3 最脆弱"假说的直接检验
    log()
    log("  zone3 - zone1 落差(负值越大 = zone3 掉得越狠):")
    delta = {}
    for g in ["Sham", "CLP"]:
        d = tab_lp.loc[("zone3", g)] - tab_lp.loc[("zone1", g)]
        delta[g] = d
    cmp = pd.DataFrame(delta)
    cmp["差(CLP-Sham)"] = cmp["CLP"] - cmp["Sham"]
    for line in cmp.round(3).to_string().split("\n"):
        log("    " + line)

    out = os.path.join(PROC_DIR, "hep_zonation.h5ad")
    hep.write(out)
    log()
    log(f"[save] -> {out}")

    csv = os.path.join(PROC_DIR, "lipid_by_zone_group.csv")
    tab_lp.round(4).to_csv(csv)
    log(f"[save] -> {csv}")


if __name__ == "__main__":
    main()
