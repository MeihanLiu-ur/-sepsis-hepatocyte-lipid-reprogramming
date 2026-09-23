#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
86_make_consensus.py -- 生成主分析用的肝细胞集(B_consensus)

背景
  83/84/85 号脚本发现:宽松判据得到的 67,145 个"肝细胞"里,
  有 8,024 个被聚类注释为非肝类型(11.95%),且 UMAP 证据显示这些细胞
  96%+ 不在肝细胞区域 —— 是**真·非实质细胞污染**,不是 ambient。

  B_consensus = 聚类注释(celltype == Hepatocyte) ∩ per-cell 双条件判据
              = 两套**独立**证据的交集
              = 59,121 个细胞,注释一致性 100%

  三套定义(A_loose 67,145 / B_consensus 59,121 / C_strict 43,462)下,
  12 条程序的 CLP vs Sham 方向**全部一致**(见 hep_set_sensitivity.csv),
  说明细胞集定义不影响主结论。本脚本产出主分析集,
  宽松集保留为敏感性分析。

用法: python scripts/86_make_consensus.py
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
    grp = A.obs["group"].values
    samp = A.obs["sample"].values

    npos = np.zeros(A.n_obs, dtype=int)
    for g in HEP_POS:
        if g in set(v):
            npos += (np.asarray(X[:, v.index(g)].todense()).ravel() > 0).astype(int)
    neg = np.zeros(A.n_obs, dtype=int)
    for g in NEG_FLAT:
        if g in set(v):
            neg += (np.asarray(X[:, v.index(g)].todense()).ravel() > 0).astype(int)

    loose = (npos >= 3) & (neg <= 1)
    consensus = (ct == "Hepatocyte") & loose

    A.obs["n_hep_pos"] = npos
    A.obs["n_nonhep_marker"] = neg
    A.obs["is_hep_loose"] = loose
    A.obs["is_hep_consensus"] = consensus

    log("=" * 88)
    log("肝细胞集:宽松 vs 共识")
    log("=" * 88)
    log(f"  A_loose     (判据)                    {int(loose.sum()):>7,}")
    log(f"  B_consensus (注释 ∩ 判据)  ← 主分析   {int(consensus.sum()):>7,}")
    log(f"  被共识集排除的(注释非肝但通过判据)   "
        f"{int((loose & ~consensus).sum()):>7,}")

    B = A[consensus].copy()
    log()
    log(f"  新对象 {B.n_obs:,} x {B.n_vars:,}")
    log("  各样本 / 分组:")
    tab = pd.crosstab(B.obs["sample"], B.obs["group"])
    for s in sorted(B.obs["sample"].unique()):
        m = (B.obs["sample"] == s).values
        g = B.obs.loc[m, "group"].iloc[0]
        log(f"    {s:<10s} {g:<5s} {int(m.sum()):>7,}")
    # ⚠️ 只对 Sham 组样本取最小值 —— 遍历所有样本会把 CLP 样本算成 0
    sham_samples = [s for s in B.obs["sample"].unique()
                    if (B.obs["sample"] == s).values.any()
                    and B.obs.loc[B.obs["sample"] == s, "group"].iloc[0] == "Sham"]
    sham_counts = [int((B.obs["sample"] == s).sum()) for s in sham_samples]
    sham_min = min(sham_counts) if sham_counts else 0
    log(f"  Sham 组各样本: " +
        ", ".join(f"{s}={c:,}" for s, c in zip(sham_samples, sham_counts)))
    log(f"  ⚠ Sham 组最少的样本仍有 {sham_min:,} 个肝细胞"
        f"  (目标 >= 3,000 → {'PASS' if sham_min >= 3000 else 'FAIL'})")

    out = os.path.join(PROC_DIR, "hep_consensus.h5ad")
    B.write(out)
    log(f"\n[save] {out}")
    # 同时回写 obs 到 merged_no22,便于后续脚本直接取用
    A.write(os.path.join(PROC_DIR, "merged_no22.h5ad"))
    log(f"[save] {os.path.join(PROC_DIR, 'merged_no22.h5ad')} (obs 已加判据列)")


if __name__ == "__main__":
    main()
