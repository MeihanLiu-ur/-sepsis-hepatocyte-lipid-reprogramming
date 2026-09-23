#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
84_ambient_vs_true.py -- 肝细胞集污染:是 ambient 还是真·非实质细胞?

背景
  83 号脚本发现:67,145 个"肝细胞"里有 8,024 个被聚类注释为非肝类型(11.95%),
  其中内皮 58.9%、HSC 70.3%、胆管细胞 88.6% 被判入。

  两种互斥解释,处置完全相反:
    H1 判据太松 —— 真·非实质细胞混进来了 → 必须收紧判据、重跑全部下游
    H2 注释太差 —— 这些其实是带 ambient RNA 的肝细胞 → 判据没错,是
       cluster 层的 marker 注释被 ambient 带偏了(§4.0b 已记录
       "marker argmax 不可信"这个坑)

怎么区分
  ambient RNA 的特征是"**检出但低表达**":污染分子只占该细胞总 UMI 的极小
  一部分。真·非实质细胞则是"**检出且高表达**"。
  → 比较同一注释类型内 is_hep=True 与 False 两组细胞的谱系 marker
    **CP10K 表达水平**,而不只是检出率。

  另加两条独立证据:
    - UMAP 位置:若这些细胞落在肝细胞群内,它们就是肝细胞
    - **收紧判据的敏感性分析**:用严格判据(neg_hit == 0)重新筛一遍,
      重算 12 条程序方向,看 Fig3 的结论是否改变。
      **结论不变 → 12% 污染不影响主结论**,这是最有力的回答。

用法: python scripts/84_ambient_vs_true.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import scanpy as sc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_sepsis import PROC_DIR  # noqa: E402

# 与 30 号脚本保持一致
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


def cp10k(A, gene):
    """返回某基因的 CP10K 表达向量"""
    v = list(A.var_names)
    if gene not in set(v):
        return None
    return np.asarray(A.X[:, v.index(gene)].todense()).ravel()


def main():
    A = sc.read_h5ad(os.path.join(PROC_DIR, "merged_no22.h5ad"))
    v = list(A.var_names)
    X = A.X.tocsr()
    ct = A.obs["celltype"].values
    is_hep = A.obs["is_hep"].values.astype(bool)
    log(f"细胞 {A.n_obs:,}   判据判为肝细胞 {is_hep.sum():,}")

    # 重算判据的两个分量(与 30 号一致)
    npos = np.zeros(A.n_obs, dtype=int)
    for g in HEP_POS:
        if g in set(v):
            npos += (np.asarray(X[:, v.index(g)].todense()).ravel() > 0).astype(int)
    neg = np.zeros(A.n_obs, dtype=int)
    for g in NEG_FLAT:
        if g in set(v):
            neg += (np.asarray(X[:, v.index(g)].todense()).ravel() > 0).astype(int)
    hep = (npos >= 3) & (neg <= 1)
    log(f"重算判据得到 {hep.sum():,}(与 obs 一致? {(hep == is_hep).all()})")

    # ---------- 证据 1:表达量对比(ambient vs 真谱系) ----------
    log()
    log("=" * 100)
    log("证据 1  同一注释类型内,is_hep=True vs False 的谱系 marker **表达水平**")
    log("  若 True 组的 marker 表达接近 0 → ambient(判据对,注释错)")
    log("  若 True 组的 marker 表达也很高 → 真·非实质细胞(判据太松,需收紧)")
    log("=" * 100)
    pairs = {
        "Endothelial":  ["Pecam1", "Cdh5"],
        "HSC":          ["Col1a1", "Dcn"],
        "Cholangiocyte": ["Krt19", "Epcam"],
        "Neutrophil":   ["S100a8", "Ptprc"],
        "Kupffer":      ["Ptprc", "Cd3e"],
        "T_NK":         ["Cd3e", "Ptprc"],
        "Hepatocyte":   ["Alb", "Tat"],
    }
    log(f"  {'celltype':<15s}{'gene':<8s}{'True 组中位':>13s}"
        f"{'False 组中位':>13s}{'检出率T':>9s}{'检出率F':>9s}")
    verdict = {}
    for c, gl in pairs.items():
        m = ct == c
        if m.sum() < 30:
            continue
        mt = m & is_hep          # 被判入的
        mf = m & ~is_hep         # 被排除的
        if mt.sum() < 20 or mf.sum() < 20:
            continue
        for g in gl:
            x = cp10k(A, g)
            if x is None:
                continue
            xt, xf = x[mt], x[mf]
            log(f"  {c:<15s}{g:<8s}{np.median(xt):>13.2f}{np.median(xf):>13.2f}"
                f"{float((xt>0).mean()):>9.1%}{float((xf>0).mean()):>9.1%}")
            ratio = (np.median(xf) + 0.01) / (np.median(xt) + 0.01)
            verdict.setdefault(c, []).append((g, np.median(xt), np.median(xf), ratio))

    log()
    log("  判读:")
    for c, items in verdict.items():
        if c == "Hepatocyte":
            continue
        med_ratio = np.median([it[3] for it in items])
        tag = ("判据对/注释错(被判入的是带 ambient 的肝细胞)"
               if med_ratio > 5 else
               ("判据太松(真·非实质细胞混入)" if med_ratio < 2 else "介于两者之间"))
        log(f"    {c:<15s} 被排除组/被判入组 表达倍数 = {med_ratio:>6.1f}x   → {tag}")

    # ---------- 证据 2:UMAP 位置 ----------
    log()
    log("=" * 100)
    log("证据 2  被判入的非肝细胞在 UMAP 上落在哪里?")
    log("=" * 100)
    U = A.obsm["X_umap"]
    hc = U[ct == "Hepatocyte"]
    ctr = np.median(hc, axis=0)
    # 以肝细胞群的 90 分位半径作为"肝细胞领地"
    d_hc = np.linalg.norm(hc - ctr, axis=1)
    r90 = np.percentile(d_hc, 90)
    for c in pairs:
        m = (ct == c) & is_hep
        if m.sum() < 30 or c == "Hepatocyte":
            continue
        d = np.linalg.norm(U[m] - ctr, axis=1)
        log(f"  {c:<15s} 被判入 {m.sum():>6,} 个,"
            f"落在肝细胞 90% 半径内的比例 = {float((d <= r90).mean()):.1%}")

    # ---------- 证据 3:收紧判据的敏感性分析 ----------
    log()
    log("=" * 100)
    log("证据 3  **收紧判据的敏感性分析** —— 12 条程序方向是否改变?")
    log("  严格判据: npos >= 4  AND  neg == 0")
    log("=" * 100)
    strict = (npos >= 4) & (neg == 0)
    log(f"  宽松判据 {is_hep.sum():,}  →  严格判据 {strict.sum():,} "
        f"({strict.sum()/max(is_hep.sum(),1):.1%})")
    keep_idx = np.where(strict)[0]
    log(f"  两者交集 {int((is_hep & strict).sum()):,};"
        f"严格判据新排除的(原判据下的肝细胞) = {int((is_hep & ~strict).sum()):,}")

    hep_only = sc.read_h5ad(os.path.join(PROC_DIR, "hep_only.h5ad"))
    S = pd.read_csv(os.path.join(PROC_DIR, "aucell_v2.csv"), index_col=0)
    S = S.loc[hep_only.obs_names]
    grp = hep_only.obs["group"].values
    samp = hep_only.obs["sample"].values
    # obs 顺序:hep_only 是 merged_no22 中 is_hep 的子集,按原顺序
    hep_mask_in_merged = is_hep
    strict_in_hep = strict[hep_mask_in_merged]
    log(f"  在 hep_only 的 {hep_only.n_obs:,} 个细胞里,"
        f"严格判据保留 {int(strict_in_hep.sum()):,} ({strict_in_hep.mean():.1%})")

    clp_set = set(samp[grp == "CLP"])
    log()
    log(f"  {'program':<24s}{'宽松 %Δ':>11s}{'严格 %Δ':>11s}{'方向一致?':>11s}")
    rows = []
    for p in S.columns:
        v = S[p].values
        def pct(mask):
            per = pd.Series(v[mask]).groupby(samp[mask]).mean()
            clp = [per[s] for s in per.index if s in clp_set]
            sham = [per[s] for s in per.index if s not in clp_set]
            if not clp or not sham:
                return np.nan
            return (np.mean(clp) - np.mean(sham)) / np.mean(sham) * 100
        a = pct(np.ones(len(v), dtype=bool))    # 全部(宽松)
        b = pct(strict_in_hep)                   # 严格
        same = "✓" if (np.isfinite(a) and np.isfinite(b) and (a > 0) == (b > 0)) \
            else "✗ 翻转"
        rows.append(dict(program=p, loose=round(float(a), 2),
                         strict=round(float(b), 2), same=same))
        log(f"  {p:<24s}{a:>11.1f}{b:>11.1f}{same:>11s}")
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(PROC_DIR, "fig2_strict_sensitivity.csv"), index=False)
    log(f"\n[save] {PROC_DIR}/fig2_strict_sensitivity.csv")
    n_same = int((R["same"] == "✓").sum())
    log()
    log(f"  方向一致 {n_same}/{len(R)} 条程序")
    if n_same == len(R):
        log("  → **全部方向一致**:12% 的污染不改变任何一条程序的方向,")
        log("    主结论稳健。仍应在 Methods 中披露污染率与敏感性分析。")
    else:
        log("  → 有程序方向翻转,**必须**收紧判据后重跑下游。")


if __name__ == "__main__":
    main()
