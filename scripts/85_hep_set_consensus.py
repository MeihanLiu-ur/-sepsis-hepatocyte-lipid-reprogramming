#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
85_hep_set_consensus.py -- 肝细胞集的三套定义与稳健性对比

为什么需要三套
  83 号发现宽松判据(npos>=3 AND neg<=1)得到的 67,145 个细胞里,
  有 8,024 个被聚类注释为非肝类型(11.95%);84 号的 UMAP 证据显示这些
  细胞 96%+ 不在肝细胞区域 —— 判据确实偏松。
  但 84 号的严格判据(neg==0)又会误杀真肝细胞:它排除了 21,692 个,
  远超实际污染的 8,024 个。

三套定义
  A loose      判据 alone                       npos>=3 AND neg<=1
  B consensus  聚类注释 ∩ 判据  ← **推荐主分析**  celltype==Hepatocyte AND 判据
  C strict     严格判据 + 注释                    celltype==Hepatocyte AND npos>=4 AND neg==0

  B 的概念最清晰:"既通过 cluster 层注释(9 类 marker 的 z-score,比单基因
  判据稳健),又通过 per-cell 双条件判据(挡 ambient 空液滴)"。两套独立
  证据的交集,可信度最高。

判据
  三套下分别重算 12 条程序的 CLP vs Sham 方向,**方向全部一致**才说明
  细胞集定义不影响主结论。这个"多重定义稳健性"本身就是方法学加分项。

输出
  data/proc/hep_set_sensitivity.csv     三套 × 12 程序的方向对照
  data/proc/hep_set_definitions.csv     三套的细胞数与构成

用法: python scripts/85_hep_set_consensus.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import scanpy as sc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_sepsis import PROC_DIR  # noqa: E402

HEP_POS = ["Alb", "Tat", "Cps1", "Apoa1", "Ttr", "Serpina1a"]
HEP_NEG = {
    "immune":   ["Ptprc", "Cd3e", "Cd79a", "S100a8"],
    "endo":     ["Pecam1", "Cdh5"],
    "mesench":  ["Col1a1", "Dcn"],
    "cholangio": ["Krt19", "Epcam"],
}
NEG_FLAT = [g for gl in HEP_NEG.values() for g in gl]


def log(*a, **kw):
    print(*a, flush=True, **kw)


def main():
    A = sc.read_h5ad(os.path.join(PROC_DIR, "merged_no22.h5ad"))
    v = list(A.var_names)
    X = A.X.tocsr()
    ct = A.obs["celltype"].values
    grp_all = A.obs["group"].values
    samp_all = A.obs["sample"].values

    npos = np.zeros(A.n_obs, dtype=int)
    for g in HEP_POS:
        if g in set(v):
            npos += (np.asarray(X[:, v.index(g)].todense()).ravel() > 0).astype(int)
    neg = np.zeros(A.n_obs, dtype=int)
    for g in NEG_FLAT:
        if g in set(v):
            neg += (np.asarray(X[:, v.index(g)].todense()).ravel() > 0).astype(int)

    loose = (npos >= 3) & (neg <= 1)
    is_hc = ct == "Hepatocyte"
    strict_d = (npos >= 4) & (neg == 0)

    SETS = {
        "A_loose":     loose,
        "B_consensus": is_hc & loose,
        "C_strict":    is_hc & strict_d,
    }
    log("=" * 92)
    log("三套肝细胞集的定义与规模")
    log("=" * 92)
    drows = []
    for name, m in SETS.items():
        n = int(m.sum())
        # 注释为肝细胞的比例(= 注释一致性)
        purity = float((ct[m] == "Hepatocyte").mean())
        drows.append(dict(set=name, n=n, annot_purity=round(purity, 4),
                          n_CLP=int((m & (grp_all == "CLP")).sum()),
                          n_Sham=int((m & (grp_all == "Sham")).sum())))
        log(f"  {name:<14s} n={n:>7,}   注释一致性 {purity:>6.1%}   "
            f"CLP {(m & (grp_all=='CLP')).sum():>6,} / "
            f"Sham {(m & (grp_all=='Sham')).sum():>6,}")
    D = pd.DataFrame(drows)
    D.to_csv(os.path.join(PROC_DIR, "hep_set_definitions.csv"), index=False)
    log(f"\n[save] {PROC_DIR}/hep_set_definitions.csv")

    # 各样本的肝细胞数(最少的那个决定统计能力)
    log()
    log("  各样本肝细胞数(⚠ Sham 组最少的那个决定统计能力):")
    for name, m in SETS.items():
        per = pd.Series(m).groupby(samp_all).sum()
        txt = "  ".join(f"{s}={int(per[s]):,}" for s in sorted(per.index))
        sham = [int(per[s]) for s in per.index
                if s in set(samp_all[grp_all == "Sham"])]
        log(f"    {name:<14s} {txt}   Sham 最少 {min(sham):,}")

    # ---------------- 12 条程序在三套下的方向 ----------------
    hep = sc.read_h5ad(os.path.join(PROC_DIR, "hep_only.h5ad"))
    S = pd.read_csv(os.path.join(PROC_DIR, "aucell_v2.csv"), index_col=0)
    S = S.loc[hep.obs_names]
    grp = hep.obs["group"].values
    samp = hep.obs["sample"].values
    clp_set = set(samp[grp == "CLP"])
    # hep_only 对应 merged_no22 中 loose 为真的细胞(按原顺序)
    loose_idx = np.where(loose)[0]
    log(f"\nhep_only {hep.n_obs:,}  vs  loose 索引 {len(loose_idx):,}"
        f"  (一致? {hep.n_obs == len(loose_idx)})")
    # 用样本+顺序对齐:hep_only 的 obs_names 应能在 merged 里找到
    merged_names = A.obs_names.values
    hep_names = hep.obs_names.values
    pos_in_merged = {n: i for i, n in enumerate(merged_names)}
    hep_row = np.array([pos_in_merged[n] for n in hep_names])
    log(f"  按 obs_names 对齐成功 {len(hep_row):,}")

    sub = {name: m[hep_row] for name, m in SETS.items()}
    log()
    log("=" * 92)
    log("12 条程序在三套肝细胞集下的 CLP vs Sham 方向")
    log("=" * 92)
    log(f"  {'program':<24s}" + "".join(f"{n:>14s}" for n in SETS) + "   一致?")
    rows = []
    for p in S.columns:
        vall = S[p].values
        vals = {}
        for name in SETS:
            m = sub[name]
            per = pd.Series(vall[m]).groupby(samp[m]).mean()
            clp = [per[s] for s in per.index if s in clp_set]
            sham = [per[s] for s in per.index if s not in clp_set]
            vals[name] = ((np.mean(clp) - np.mean(sham)) / np.mean(sham) * 100
                          if clp and sham else np.nan)
        signs = {np.sign(x) for x in vals.values() if np.isfinite(x)}
        ok = "✓" if len(signs) == 1 else "✗ 翻转"
        rows.append(dict(program=p, **{k: round(float(x), 2) for k, x in vals.items()},
                         consistent=ok))
        log(f"  {p:<24s}" + "".join(f"{vals[n]:>14.1f}" for n in SETS)
            + f"   {ok}")
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(PROC_DIR, "hep_set_sensitivity.csv"), index=False)
    log(f"\n[save] {PROC_DIR}/hep_set_sensitivity.csv")

    n_ok = int((R["consistent"] == "✓").sum())
    log()
    log("=" * 92)
    log("判定")
    log("=" * 92)
    log(f"  {n_ok}/{len(R)} 条程序在三套定义下方向一致")
    if n_ok == len(R):
        log("  ✅ **全部一致** —— 细胞集定义不影响任何一条程序的方向。")
        log("     建议主分析用 **B_consensus**(注释 ∩ 判据,两套独立证据的交集),")
        log("     并在 Methods 报告三套定义的稳健性对比 + 各自的污染率。")
    else:
        log("  ⚠️ 有程序随定义翻转,必须固定一套定义并说明理由。")

    log()
    log("  推荐 B_consensus 的理由:")
    log("   1. 概念清晰:cluster 层注释(9 类 marker z-score)+ per-cell 双条件判据,")
    log("      两套**独立**证据的交集,互为交叉验证")
    log("   2. 排除了 83 号发现的 8,024 个被判入的非肝细胞(UMAP 证据显示")
    log("      它们 96%+ 不在肝细胞区域)")
    log("   3. 不像 C_strict(neg==0)那样误杀真肝细胞 —— ambient 污染普遍存在时,")
    log("      neg==0 会把吸附了少量 ambient 信号的真肝细胞一并排除")


if __name__ == "__main__":
    main()
