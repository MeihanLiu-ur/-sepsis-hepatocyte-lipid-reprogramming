#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
30_exclude_and_recluster.py -- 剔除 liver_22 后重新整合聚类

背景(2026-08-29 实测)
  六个样本里 liver_22 (GSM8482483, GEO 标注 "liver, Sham, rep3") 是异类:
    原始未过滤矩阵中 Alb>0 & UMI>=200 的细胞只有 1,124 个
      (其余 5 个样本为 33,598 ~ 37,541, 差 30 倍)
    Tat>0 只有 464 个(其余 49,159 ~ 81,333, 差 100 倍以上)
    Fabp4>0 却有 56,329 个(其余仅 1,155 ~ 6,544)
    高表达谱 = Fabp4 / Cfd / Lpl / Cd36 / Ebf1 / Car3 / Ghr -> 脂肪/间质谱系
  GEO 元数据明确写 tissue: liver, 但数据内容不是肝细胞。
  与 QC 阈值无关(已用未过滤矩阵验证), 剔除。

本脚本
  1. 剔除 liver_22, 重跑 HVG -> PCA -> Harmony -> leiden
  2. 报告**批次混合度**(验收指标: cluster 最大样本占比中位数 < 0.6)
  3. 用 per-cell 判据挑肝细胞(不再依赖全局聚类的注释, 因为非实质细胞注释不可信)
     判据: 肝细胞标志高 **且** 非肝谱系标志低(排除 ambient 空液滴)
  4. 报告各样本肝细胞数是否均衡(验收: Sham 两个样本各 >= 3,000)

用法: python scripts/30_exclude_and_recluster.py [--res 0.8]
"""
from __future__ import annotations

import os
import sys
import time
import argparse

import numpy as np
import pandas as pd
import scanpy as sc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_sepsis import PROC_DIR  # noqa: E402

EXCLUDE = ["liver_22"]

# 肝细胞身份基因(高表达要求)—— 只用最稳健、最难被 ambient 抹掉的几个
HEP_POS = ["Alb", "Tat", "Cps1", "Apoa1", "Ttr", "Serpina1a"]
# 非肝谱系标志(必须低)—— 用来排除 ambient 空液滴与真·非实质细胞
HEP_NEG = {
    "immune":   ["Ptprc", "Cd3e", "Cd79a", "S100a8"],
    "endo":     ["Pecam1", "Cdh5"],
    "mesench":  ["Col1a1", "Dcn"],
    "cholangio": ["Krt19", "Epcam"],
}


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def frac_pos(A, gene, mask=None):
    """某基因在给定细胞子集上的阳性比例"""
    v = list(A.var_names)
    if gene not in set(v):
        return None
    i = v.index(gene)
    x = A.X[:, i]
    if mask is not None:
        x = x[mask]
    x = np.asarray(x.todense()).ravel() if hasattr(x, "todense") else np.asarray(x).ravel()
    return float((x > 0).mean())


def mean_expr(A, gene, mask=None):
    v = list(A.var_names)
    if gene not in set(v):
        return np.nan
    i = v.index(gene)
    x = A.X[:, i]
    if mask is not None:
        x = x[mask]
    return float(np.asarray(x.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=float, default=0.8)
    ap.add_argument("--inp", default=os.path.join(PROC_DIR, "merged.h5ad"))
    ap.add_argument("--out", default=os.path.join(PROC_DIR, "merged_no22.h5ad"))
    args = ap.parse_args()

    log("读取", args.inp)
    A = sc.read_h5ad(args.inp)
    log(f"  原始: {A.n_obs:,} 细胞")

    # ---------- 1. 剔除 ----------
    keep = ~A.obs["sample"].isin(EXCLUDE).values
    log(f"  剔除 {EXCLUDE}: 去掉 {int((~keep).sum()):,} 个细胞, 剩 {int(keep.sum()):,}")
    A = A[keep].copy()
    log(f"  剩余样本: {sorted(A.obs['sample'].unique())}")

    # ---------- 2. 重新整合聚类 ----------
    log()
    log("重跑 HVG -> PCA -> Harmony -> leiden")
    A.layers["counts"] = A.layers["counts"] if "counts" in A.layers else A.X.copy()
    sc.pp.normalize_total(A, target_sum=1e4)
    sc.pp.log1p(A)
    sc.pp.highly_variable_genes(A, n_top_genes=2500, batch_key="batch")
    Asub = A[:, A.var["highly_variable"]].copy()
    sc.pp.scale(Asub, max_value=10)
    sc.tl.pca(Asub, n_comps=50, svd_solver="arpack", random_state=0)
    # 注: scanpy 的 harmony_integrate 与新版 anndata 有 shape 兼容问题
    #     (harmony_out.Z_corr.T 会得到一维 (50,) 而非 (n_cells, 50)),
    #     故此处直接调 harmonypy 并自行纠正转置。
    import harmonypy as hm
    ho = hm.run_harmony(np.asarray(Asub.obsm["X_pca"]), Asub.obs, "batch",
                        random_state=0)
    Z = np.asarray(ho.Z_corr)
    if Z.shape[0] != Asub.n_obs:
        Z = Z.T
    assert Z.shape == (Asub.n_obs, Asub.obsm["X_pca"].shape[1]), \
        f"harmony 输出形状异常: {Z.shape}"
    Asub.obsm["X_pca_harmony"] = Z
    sc.pp.neighbors(Asub, use_rep="X_pca_harmony", n_neighbors=15, random_state=0)
    sc.tl.leiden(Asub, resolution=args.res, key_added="leiden",
                 flavor="igraph", n_iterations=2, directed=False)
    sc.tl.umap(Asub, random_state=0)
    A.obs["leiden"] = Asub.obs["leiden"].values
    A.obsm["X_pca_harmony"] = Asub.obsm["X_pca_harmony"]
    A.obsm["X_umap"] = Asub.obsm["X_umap"]
    log(f"  -> {A.obs['leiden'].nunique()} 个 cluster")

    # ---------- 3. 批次混合度(核心验收指标) ----------
    ct = pd.crosstab(A.obs["leiden"], A.obs["sample"])
    frac = ct.div(ct.sum(1), axis=0).max(1)
    log()
    log("=" * 68)
    log("验收 1: 批次混合度")
    log("=" * 68)
    log(f"  cluster 最大样本占比: 中位 {frac.median():.2f}  均值 {frac.mean():.2f}")
    log(f"  >0.9 的单一来源 cluster: {(frac>0.9).sum()}/{len(frac)}   "
        f"(剔除前为 19/36, 中位 0.96)")
    log(f"  目标: 中位数 < 0.6")
    ok_mix = frac.median() < 0.6
    log(f"  判定: {'PASS' if ok_mix else 'FAIL'}")

    big = ct.sum(1).sort_values(ascending=False).head(8).index
    log()
    log("  最大的 8 个 cluster 的样本构成:")
    for line in ct.loc[big].to_string().split("\n"):
        log("    " + line)

    # ---------- 4. per-cell 肝细胞判据 ----------
    log()
    log("=" * 68)
    log("验收 2: per-cell 肝细胞判据(不依赖聚类注释)")
    log("=" * 68)
    log("  判据: 肝细胞身份基因阳性数 >= 阈值  **且**  非肝谱系标志全为阴性")

    v = list(A.var_names)
    pidxs = {g: v.index(g) for g in HEP_POS if g in set(v)}
    npos = np.zeros(A.n_obs, dtype=int)
    for g, i in pidxs.items():
        x = A.X[:, i]
        x = np.asarray(x.todense()).ravel() if hasattr(x, "todense") else np.asarray(x).ravel()
        npos += (x > 0).astype(int)

    neg_hit = np.zeros(A.n_obs, dtype=int)
    for grp, gl in HEP_NEG.items():
        for g in gl:
            if g not in set(v):
                continue
            i = v.index(g)
            x = A.X[:, i]
            x = np.asarray(x.todense()).ravel() if hasattr(x, "todense") else np.asarray(x).ravel()
            neg_hit += (x > 0).astype(int)

    hep = (npos >= 3) & (neg_hit <= 1)
    A.obs["is_hep"] = hep
    log(f"  身份基因命中 {len(pidxs)}/{len(HEP_POS)}; "
        f"满足 '>=3 个阳性 且 <=1 个非肝标志' 的细胞: {int(hep.sum()):,} "
        f"({hep.mean()*100:.1f}%)")

    tab = pd.crosstab(A.obs["sample"], A.obs["group"])
    log()
    log("  各样本肝细胞数:")
    for s in sorted(A.obs["sample"].unique()):
        m = (A.obs["sample"] == s).values
        g = A.obs.loc[m, "group"].iloc[0]
        nh = int((hep & m).sum())
        log(f"    {s} ({g:4s}): {nh:>7,} / {int(m.sum()):>6,}  ({nh/m.sum()*100:5.1f}%)")

    sham = A.obs["group"] == "Sham"
    cnt = {s: int((hep & (A.obs["sample"] == s).values).sum())
           for s in sorted(A.obs["sample"].unique())}
    sham_min = min([c for s, c in cnt.items() if s.startswith("liver_") and
                    A.obs.loc[A.obs["sample"] == s, "group"].iloc[0] == "Sham"])
    log()
    log(f"  Sham 组最少样本的肝细胞数 = {sham_min:,}  (目标 >= 3,000)")
    ok_hep = sham_min >= 3000
    log(f"  判定: {'PASS' if ok_hep else 'FAIL'}")

    # ---------- 5. 保存 ----------
    A.write(args.out)
    log()
    log(f"[save] -> {args.out}  ({A.n_obs:,} 细胞, {int(hep.sum()):,} 肝细胞)")

    # 顺带把肝细胞子集也存一份
    H = A[hep].copy()
    ho = os.path.join(PROC_DIR, "hep_only.h5ad")
    H.write(ho)
    log(f"[save] -> {ho}  ({H.n_obs:,} 肝细胞)")

    log()
    log("=" * 68)
    log(f"总判定  批次混合={'PASS' if ok_mix else 'FAIL'}   "
        f"肝细胞均衡={'PASS' if ok_hep else 'FAIL'}")
    log("=" * 68)


if __name__ == "__main__":
    main()
