#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
40_zonation_aucell.py -- 区带 × 疾病状态:正式版(向量化 AUCell + 深度校正)

相对 20_zonation_firstlook.py 的三处升级
  1. **AUCell 取代简化 module score**。AUCell 基于细胞内表达**秩**,
     不受基因集大小与数据稀疏度影响,且**天然对测序深度不敏感**
     (深度只改变计数尺度,不改变细胞内排序)——这正好回应"CLP 组更深"的混杂。
     实现为全向量化版本:每个细胞的秩只算一次,10 个基因集复用。
  2. **深度校正敏感性分析**:把 counts 二项降采样到统一深度后重跑,
     看结论是否翻转。
  3. **只纳入肝细胞 + 已剔除 liver_22**(见 30_exclude_and_recluster.py)。

统计口径
  - 一切差异的统计单位是**样本**,不是细胞。Sham 组 n=2、CLP 组 n=3。
  - 核心假说"zone3 最脆弱"的主检验是**样本内对比**(zone3 vs zone1 的落差),
    每个样本独立可算;组间对比(落差在 CLP 下是否变大)才是 n=2 vs 3 的弱项。

用法: python scripts/40_zonation_aucell.py [--ds 1500]
"""
from __future__ import annotations

import os
import sys
import time
import argparse

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_sepsis import ZONE, LIPID_PROGRAMS, PROC_DIR  # noqa: E402


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ---------------------------------------------------------------- AUCell ---
def cellwise_ranks(X):
    """
    对每个细胞的非零基因算降序秩(0 = 表达最高),返回与 X.data 对齐的扁平数组。
    零表达基因并列排在最后(秩 = 该细胞非零基因数),对 AUC 无贡献(会被 clip 掉或很小)。
    """
    X = X.tocsr()
    indptr, data = X.indptr, X.data
    ranks = np.empty(data.shape[0], dtype=np.int64)
    for i in range(X.shape[0]):
        s, e = indptr[i], indptr[i + 1]
        k = e - s
        if k == 0:
            continue
        v = data[s:e]
        order = np.argsort(-v, kind="stable")
        r = np.empty(k, dtype=np.int64)
        r[order] = np.arange(k)
        ranks[s:e] = r
    return ranks


def aucell(X, X_genes, gene_list, auc_max, ranks, cell_of_entry, n_cells):
    """单个基因集的 AUCell"""
    pos = {}
    for i, g in enumerate(X_genes):
        pos.setdefault(g, i)
    idx = np.array(sorted({pos[g] for g in gene_list if g in pos}), dtype=int)
    if idx.size == 0:
        return np.zeros(n_cells, dtype=np.float32)
    gs = np.zeros(len(X_genes), dtype=bool)
    gs[idx] = True
    sel = gs[X.indices]
    if not sel.any():
        return np.zeros(n_cells, dtype=np.float32)
    r = ranks[sel]
    c = cell_of_entry[sel]
    contrib = np.clip(auc_max - r, 0, None).astype(np.float64)
    auc = np.bincount(c, weights=contrib, minlength=n_cells)
    return (auc / (auc_max * len(idx))).astype(np.float32)


def run_all(X, genes, obs_names, tag, auc_frac=0.05):
    """
    X: 细胞 x 基因 CSR(已 log 归一化); genes: 基因名列表; obs_names: 细胞名
    (注: 不接收 AnnData —— anndata 0.13 的 .X 赋值与 scipy 新版稀疏矩阵
     存在兼容问题, 直接传矩阵更稳)
    """
    X = X.tocsr()
    n_genes = X.shape[1]
    auc_max = int(max(10, auc_frac * n_genes))
    log(f"  [{tag}] AUCell: n_genes={n_genes}, auc_max={auc_max} (top {auc_frac:.0%})")

    t0 = time.time()
    ranks = cellwise_ranks(X)
    cell_of_entry = np.repeat(np.arange(X.shape[0]), np.diff(X.indptr))
    log(f"  [{tag}] 细胞内秩计算完成 ({time.time()-t0:.1f}s)")

    res = {}
    res["central"] = aucell(X, genes, ZONE["pericentral"], auc_max,
                            ranks, cell_of_entry, X.shape[0])
    res["portal"] = aucell(X, genes, ZONE["periportal"], auc_max,
                           ranks, cell_of_entry, X.shape[0])
    for p, gl in LIPID_PROGRAMS.items():
        res[p] = aucell(X, genes, gl, auc_max, ranks, cell_of_entry, X.shape[0])
    res["zonation_index"] = res["central"] - res["portal"]
    return pd.DataFrame(res, index=obs_names)


def downsample_to(hep, target):
    """把 counts 二项降采样到每个细胞 target 条 UMI,再 CP10K + log1p"""
    C = hep.layers["counts"].tocsr()      # 细胞 x 基因(注意: 不要转置)
    C = C.astype(np.int64)
    rng = np.random.default_rng(0)
    indptr, indices, data = C.indptr, C.indices, C.data
    tot = np.asarray(C.sum(1)).ravel()
    p = np.minimum(1.0, target / np.maximum(tot, 1))
    per_cell_p = p[np.repeat(np.arange(C.shape[0]), np.diff(indptr))]
    new_data = rng.binomial(data, per_cell_p)
    D = sp.csr_matrix((new_data, indices, indptr), shape=C.shape)
    D = sp.csr_matrix(D)          # scipy>=1.15 会返回 csr_array, anndata 只认 csr_matrix
    D.eliminate_zeros()
    tot2 = np.asarray(D.sum(1)).ravel()
    tot2[tot2 == 0] = 1
    D = sp.csr_matrix(sp.diags(1e4 / tot2) @ D)
    D.data = np.log1p(D.data)
    return D.tocsr()


# ---------------------------------------------------------------- 主流程 ---
def report(df, hep, tag):
    zi = df["zonation_index"].values
    grp = hep.obs["group"].values
    samp = hep.obs["sample"].values

    log()
    log("=" * 70)
    log(f"[{tag}] 核心结果 1:zonation index 分布形态(collapse 判据)")
    log("=" * 70)
    rows = []
    for g in ["Sham", "CLP"]:
        v = zi[grp == g]
        rows.append(dict(group=g, n=len(v), mean=v.mean(), sd=v.std(),
                         iqr=np.subtract(*np.percentile(v, [75, 25]))))
    d = pd.DataFrame(rows).set_index("group")
    for line in d.round(3).to_string().split("\n"):
        log("  " + line)
    log(f"  SD 比 CLP/Sham  = {d.loc['CLP','sd']/d.loc['Sham','sd']:.3f}")
    log(f"  IQR 比 CLP/Sham = {d.loc['CLP','iqr']/d.loc['Sham','iqr']:.3f}")

    log()
    log(f"[{tag}] 样本层(统计单位 = 样本, Sham n=2 / CLP n=3):")
    ps = pd.DataFrame({"sample": samp, "group": grp, "zi": zi}) \
        .groupby(["group", "sample"])["zi"].agg(["count", "mean", "std"])
    for line in ps.round(3).to_string().split("\n"):
        log("    " + line)
    for g in ["Sham", "CLP"]:
        m = ps.loc[g, "mean"].values
        log(f"    {g}: 各样本均值 {np.round(m,3).tolist()}  "
            f"mean={m.mean():.3f} sd={m.std(ddof=1) if len(m)>1 else float('nan'):.3f}")

    # ---- 在 Sham 标定 zone, 套用到 CLP ----
    sham_zi = zi[grp == "Sham"]
    c1, c2 = np.percentile(sham_zi, [33.3, 66.7])
    zone = np.where(zi < c1, "zone1", np.where(zi < c2, "zone2", "zone3"))
    df2 = df.copy()
    df2["zone"] = zone
    df2["group"] = grp
    df2["sample"] = samp

    log()
    log("=" * 70)
    log(f"[{tag}] 核心结果 2:zone × 分组 的脂质程序(AUCell)")
    log("=" * 70)
    cols = list(LIPID_PROGRAMS)
    tab = df2.groupby(["zone", "group"])[cols].mean()
    for line in tab.round(4).to_string().split("\n"):
        log("  " + line)

    log()
    log(f"[{tag}] 核心假说检验:zone3 - zone1 落差(负 = zone3 更低)")
    delta = {}
    for g in ["Sham", "CLP"]:
        delta[g] = tab.loc[("zone3", g)] - tab.loc[("zone1", g)]
    cmp = pd.DataFrame(delta)
    cmp["CLP - Sham"] = cmp["CLP"] - cmp["Sham"]
    for line in cmp.round(4).to_string().split("\n"):
        log("  " + line)

    # 每个样本内部的 zone3-zone1 落差(这才是主检验, 组内对比, 不依赖组间)
    log()
    log(f"[{tag}] 每个样本内部的 zone3-zone1 落差(主检验: 组内对比)")
    per = []
    for s in sorted(set(samp)):
        m = (samp == s)
        g = grp[m][0]
        z3 = zi[m & (zone == "zone3")]
        z1 = zi[m & (zone == "zone1")]
        row = dict(sample=s, group=g, n_z1=len(z1), n_z3=len(z3))
        for c in cols:
            v3 = df2.loc[m & (df2["zone"] == "zone3"), c].mean()
            v1 = df2.loc[m & (df2["zone"] == "zone1"), c].mean()
            row[c] = v3 - v1
        per.append(row)
    pert = pd.DataFrame(per).set_index("sample")
    for line in pert.round(4).to_string().split("\n"):
        log("  " + line)

    return df2, tab, cmp, pert


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default=os.path.join(PROC_DIR, "hep_only.h5ad"))
    ap.add_argument("--ds", type=int, default=1500,
                    help="深度校正敏感性分析:降采样到该 UMI 数,0=跳过")
    args = ap.parse_args()

    log("读取", args.inp)
    hep = sc.read_h5ad(args.inp)
    log(f"  肝细胞 {hep.n_obs:,};  "
        f"CLP {int((hep.obs['group']=='CLP').sum()):,} / "
        f"Sham {int((hep.obs['group']=='Sham').sum()):,}")

    # ---- 主分析 ----
    log()
    log(">>> 主分析:CP10K + log1p -> AUCell")
    D = run_all(hep.X.tocsr(), list(hep.var_names), hep.obs_names, "main")
    df2, tab, cmp, pert = report(D, hep, "main")
    D.to_csv(os.path.join(PROC_DIR, "aucell_main.csv"))
    tab.round(5).to_csv(os.path.join(PROC_DIR, "lipid_zone_group_aucell.csv"))
    pert.round(5).to_csv(os.path.join(PROC_DIR, "per_sample_zone3_minus_zone1.csv"))

    # ---- 深度校正敏感性分析 ----
    if args.ds > 0:
        log()
        log("=" * 70)
        log(f">>> 敏感性分析:counts 降采样到 {args.ds} UMI/细胞 后重跑")
        log("=" * 70)
        Xd = downsample_to(hep, args.ds)
        Dd = run_all(Xd, list(hep.var_names), hep.obs_names, f"ds{args.ds}")
        df2d, tabd, cmpd, pertd = report(Dd, hep, f"ds{args.ds}")
        tabd.round(5).to_csv(
            os.path.join(PROC_DIR, f"lipid_zone_group_aucell_ds{args.ds}.csv"))
        pertd.round(5).to_csv(
            os.path.join(PROC_DIR, f"per_sample_zone3_minus_zone1_ds{args.ds}.csv"))

        # 一致性:两个版本的相关性
        log()
        log("=" * 70)
        log("主分析 vs 降采样版 的一致性")
        log("=" * 70)
        common = list(LIPID_PROGRAMS) + ["zonation_index"]
        cors = {}
        for c in common:
            cors[c] = float(np.corrcoef(D[c].values, Dd[c].values)[0, 1])
        cs = pd.Series(cors).round(3)
        for line in cs.to_string().split("\n"):
            log("  " + line)
        log(f"  最低相关 = {cs.min():.3f}  "
            f"({'一致' if cs.min() > 0.7 else '不一致, 深度混杂确实存在'})")


if __name__ == "__main__":
    main()
