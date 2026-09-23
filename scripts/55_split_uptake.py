#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
55_split_uptake.py -- 把 Uptake 程序按**基因家族**拆成三条通路(v3)

背景
  原 Uptake 程序(8 基因)内部方向打架:Cd36 上调、Ldlr / Scarb1 / Lipc 下调。
  根因是它把三条**调控通路完全不同**的基因家族塞进了一个基因集:
    - LDLR 基因家族(Ldlr / Vldlr / Lrp1 + Ldlrap1 / Lrpap1):受 SREBP / PCSK9
      调控,负责受体介导的 LDL / 残粒颗粒清除
    - B 类清道夫受体(Scarb1 = SR-B1, Cd36 = SR-B2):受 PPARα / LXR 与炎症
      信号调控,负责修饰脂蛋白 / oxLDL 的应激性清除
    - Slc27(FATP)溶质载体家族 + Fabp1:受 PPARα 调控,负责游离脂肪酸的跨膜
      转运与胞内运输
  合并打分会把方向相反的信号相互抵消,出一个"≈0、p=0.63"的假阴性。

做法
  1. 按 lib_sepsis.LIPID_PROGRAMS (v3, 10 个程序) **重算全部 AUCell**。
     (只重算秩一次,10 个程序复用,成本可控;不复用旧 Uptake 列)
  2. 程序层统计:每样本点值、% 变化、完全分离判定、样本层置换 p。
  3. 独立 bulk 队列(GSE311736, n=4 vs 4)上做程序层竞争性基因集检验,
     —— 这是唯一能给出 p<0.05 的一层。
  4. 单基因层面的方向一致性检查:每个程序内部有多少基因方向一致。
     内部不一致的程序要如实标注,不能只报均值。
  5. Lipc 已移出程序(分泌型酶,不属摄取 / 转运机器),但仍单独报告。

输出
  data/proc/aucell_v2.csv                 per-cell 10 程序 AUCell
  data/proc/fig3_program_effect_v2.csv    程序层(每样本点值 + 分离度 + 置换 p)
  data/proc/bulk_program_geneset_v2.csv   独立队列程序层基因集检验
  data/proc/bulk_receptor_split.csv       拆分后三条通路的单基因 bulk 结果

用法: python scripts/55_split_uptake.py
"""
from __future__ import annotations

import os
import sys
import time
import itertools

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_sepsis import LIPID_PROGRAMS, PROC_DIR  # noqa: E402

# 拆分后要重点报告的通路(按基因家族 / 外排目的地)
SPLIT = ["LDLR_clearance", "Scavenger_SR_B", "FA_transport",
         "HDL_efflux", "Biliary_excretion", "Bile_acid_synth"]
# 已移出程序但需单独报告的基因(定义正确但不适用,或归类错误)
REPORT_ONLY = {
    "Lipc": "肝脂肪酶:分泌型,窦周间隙水解脂蛋白 TG,不属肝细胞摄取机器",
    "Abcg1": "肝细胞几乎不表达(snRNA 检出 2.3%,bulk baseMean 4),log2FC 不可靠",
    "Nr1h3": "LXRα 是转录因子,不是外排执行者(v4 已移出,方向 −0.894 padj 1.1e-4)",
}


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ------------------------------------------------------------------ AUCell --
def cellwise_ranks(X):
    """每个细胞的非零基因算降序秩(0 = 表达最高),返回与 X.data 对齐的扁平数组。"""
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


# ------------------------------------------------------------------- 统计 ---
def perm_p(vals, groups, grp_a="CLP", grp_b="Sham"):
    """样本层置换检验,同时返回该设计的理论最小 p。"""
    v = np.asarray(vals, dtype=float)
    g = np.asarray(groups)
    n = len(v)
    k = int((g == grp_a).sum())
    obs = v[g == grp_a].mean() - v[g == grp_b].mean()
    stat = []
    for comb in itertools.combinations(range(n), k):
        m = np.zeros(n, dtype=bool)
        m[list(comb)] = True
        stat.append(v[m].mean() - v[~m].mean())
    stat = np.asarray(stat)
    p_two = float((np.abs(stat) >= abs(obs) - 1e-12).mean())
    return obs, p_two, 1.0 / len(stat), len(stat)


def separation(clp_vals, sham_vals):
    lo_s, hi_s = min(sham_vals), max(sham_vals)
    lo_c, hi_c = min(clp_vals), max(clp_vals)
    if hi_c < lo_s:
        return "完全分离(CLP 全低于 Sham)"
    if lo_c > hi_s:
        return "完全分离(CLP 全高于 Sham)"
    return "有重叠"


def main():
    import argparse as _ap
    _p = _ap.ArgumentParser()
    _p.add_argument("--inp", default=os.path.join(PROC_DIR, "hep_consensus.h5ad"),
                    help="肝细胞集。默认 B_consensus(注释 ∩ 判据, 59,121 个);"
                         "传 data/proc/hep_only.h5ad 可复现宽松集"
                         "(A_loose, 67,145)的结果")
    _a = _p.parse_args()

    t_start = time.time()
    hep = sc.read_h5ad(_a.inp)
    grp = hep.obs["group"].values
    samp = hep.obs["sample"].values
    log(f"肝细胞 {hep.n_obs:,}  基因 {hep.n_vars:,}   [来自 {_a.inp}]")

    # ---------------- 1. 重算 9 个程序 ----------------
    X = hep.X.tocsr()
    genes = list(hep.var_names)
    n_genes = X.shape[1]
    auc_max = int(max(10, 0.05 * n_genes))
    log(f"AUCell: auc_max = {auc_max} (top 5%)")

    # 先报告基因检出情况,避免"程序里一半基因没测到"的暗坑
    log("程序基因检出率检查:")
    detect = {}
    for p, gl in LIPID_PROGRAMS.items():
        hit = [g for g in gl if g in set(genes)]
        det = []
        for g in hit:
            i = genes.index(g)
            det.append(float((X[:, i] > 0).sum()) / X.shape[0])
        detect[p] = (len(hit), len(gl), np.mean(det) if det else 0.0)
        flag = "" if len(hit) == len(gl) else "  ⚠️ 有基因未匹配"
        log(f"  {p:<22s} {len(hit):>2d}/{len(gl):<2d} 匹配  "
            f"平均检出率 {np.mean(det) if det else 0:6.1%}{flag}")

    t0 = time.time()
    ranks = cellwise_ranks(X)
    cell_of_entry = np.repeat(np.arange(X.shape[0]), np.diff(X.indptr))
    log(f"细胞内秩计算完成 ({time.time()-t0:.1f}s)")

    res = {}
    for p, gl in LIPID_PROGRAMS.items():
        res[p] = aucell(X, genes, gl, auc_max, ranks, cell_of_entry, X.shape[0])
    D = pd.DataFrame(res, index=hep.obs_names)
    D.to_csv(os.path.join(PROC_DIR, "aucell_v2.csv"))
    log(f"[save] {PROC_DIR}/aucell_v2.csv  ({len(D.columns)} 个程序)")
    del ranks, cell_of_entry

    # ---------------- 2. 程序层统计 ----------------
    log()
    log("=" * 96)
    log("程序层(snRNA, 9 个程序):每样本点值 / 变化幅度 / 分离度 / 样本层置换 p")
    log("=" * 96)
    samp_of = {s: ("CLP" if s in set(samp[grp == "CLP"]) else "Sham") for s in set(samp)}
    rows = []
    for p in LIPID_PROGRAMS:
        v = D[p].values
        per = pd.Series(v).groupby(samp).mean()
        clp = [per[s] for s in per.index if samp_of[s] == "CLP"]
        sham = [per[s] for s in per.index if samp_of[s] == "Sham"]
        obs, pp, pmin, nperm = perm_p(per.values, [samp_of[s] for s in per.index])
        pct = (np.mean(clp) - np.mean(sham)) / np.mean(sham) * 100
        u, _ = mannwhitneyu(v[grp == "CLP"], v[grp == "Sham"], alternative="two-sided")
        n1, n2 = int((grp == "CLP").sum()), int((grp == "Sham").sum())
        rows.append(dict(program=p,
                         sham_mean=float(np.mean(sham)),
                         clp_mean=float(np.mean(clp)),
                         pct_change=round(float(pct), 2),
                         **{f"CLP_{i+1}": round(float(x), 4) for i, x in enumerate(clp)},
                         **{f"Sham_{i+1}": round(float(x), 4) for i, x in enumerate(sham)},
                         separation=separation(clp, sham),
                         perm_p=round(float(pp), 3),
                         perm_min_p=round(float(pmin), 3),
                         cell_AUC=round(float(u / (n1 * n2)), 3)))
    T = pd.DataFrame(rows).set_index("program")
    for line in T.round(4).to_string().split("\n"):
        log("  " + line)
    T.round(5).to_csv(os.path.join(PROC_DIR, "fig3_program_effect_v2.csv"))

    # ---------------- 3. 独立 bulk 队列程序层检验 ----------------
    log()
    log("=" * 96)
    log("独立 bulk 队列(GSE311736, n=4 vs 4):程序层竞争性基因集检验")
    log("=" * 96)
    bulk = pd.read_csv(os.path.join(PROC_DIR, "bulk_HEP_CLP_vs_Sham.csv"),
                       index_col=0)
    bg = bulk["log2FoldChange"].dropna()
    prows = []
    for p, gl in LIPID_PROGRAMS.items():
        gs = bulk.loc[[g for g in gl if g in bulk.index], "log2FoldChange"].dropna()
        if len(gs) < 3:
            prows.append(dict(program=p, n=len(gs), mean_log2FC=np.nan,
                              p_gene_set=np.nan))
            continue
        u, pv = mannwhitneyu(gs, bg, alternative="two-sided")
        prows.append(dict(program=p, n=len(gs),
                          mean_log2FC=round(float(gs.mean()), 3),
                          median_log2FC=round(float(gs.median()), 3),
                          bg_median=round(float(bg.median()), 3),
                          p_gene_set=float(pv)))
    P = pd.DataFrame(prows)
    for line in P.to_string(index=False).split("\n"):
        log("  " + line)
    P.to_csv(os.path.join(PROC_DIR, "bulk_program_geneset_v2.csv"), index=False)

    # ---------------- 4. 拆分后两条通路的单基因明细 ----------------
    log()
    log("=" * 96)
    log("拆分后两条摄取通路的单基因 bulk 结果(GSE311736)")
    log("=" * 96)
    urows = []
    for p in SPLIT:
        for g in LIPID_PROGRAMS[p]:
            if g in bulk.index:
                r = bulk.loc[g]
                urows.append(dict(program=p, gene=g,
                                  baseMean=round(float(r["baseMean"]), 1),
                                  log2FC=round(float(r["log2FoldChange"]), 3),
                                  padj=float(r["padj"]),
                                  direction="↑" if r["log2FoldChange"] > 0.1 else
                                            ("↓" if r["log2FoldChange"] < -0.1 else "≈")))
            else:
                urows.append(dict(program=p, gene=g, baseMean=np.nan,
                                  log2FC=np.nan, padj=np.nan,
                                  direction="未检出"))
    U = pd.DataFrame(urows)
    # 移出程序的基因:仍如实报告,不隐藏
    for g, why in REPORT_ONLY.items():
        if g in bulk.index:
            r = bulk.loc[g]
            U.loc[len(U)] = dict(
                program="(已移出程序)", gene=g,
                baseMean=round(float(r["baseMean"]), 1),
                log2FC=round(float(r["log2FoldChange"]), 3),
                padj=float(r["padj"]),
                direction="↑" if r["log2FoldChange"] > 0.1 else
                          ("↓" if r["log2FoldChange"] < -0.1 else "≈"))
            log(f"  [移出] {g}: log2FC={r['log2FoldChange']:+.3f} "
                f"padj={r['padj']:.3g}  —— {why}")
    for line in U.to_string(index=False).split("\n"):
        log("  " + line)
    U.to_csv(os.path.join(PROC_DIR, "bulk_receptor_split.csv"), index=False)

    # ---------------- 5. 内部方向一致性(不能只报均值) ----------------
    log()
    log("=" * 96)
    log("程序内部方向一致性检查(bulk):均值掩盖异质性的地方必须标出来")
    log("=" * 96)
    for p, gl in LIPID_PROGRAMS.items():
        gs = bulk.loc[[g for g in gl if g in bulk.index], "log2FoldChange"].dropna()
        if len(gs) < 3:
            continue
        up = int((gs > 0.1).sum())
        dn = int((gs < -0.1).sum())
        flat = len(gs) - up - dn
        consist = max(up, dn) / len(gs)
        mark = "" if consist >= 0.7 else "   ⚠️ 内部方向不一致,均值不可解释"
        log(f"  {p:<22s} n={len(gs):>2d}  上调 {up:>2d} / 下调 {dn:>2d} / 平 {flat:>2d}"
            f"   一致率 {consist:5.0%}{mark}")
        if p in SPLIT or consist < 0.7:
            for g, v in gs.items():
                pa = bulk.loc[g, "padj"]
                st = "***" if pa < 0.001 else ("**" if pa < 0.01 else
                                              ("*" if pa < 0.05 else "ns"))
                log(f"        {g:<10s} {v:+7.3f}  padj={pa:.3g} {st}")

    log()
    log("=" * 96)
    log(f"全部完成,用时 {time.time()-t_start:.0f}s")
    log("=" * 96)


if __name__ == "__main__":
    main()
