#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10_preprocess.py -- C1 主流程(一):取数 -> cell calling -> QC -> 整合 -> 聚类 -> 注释

为什么需要它:
  01_gate_probe.py 用 per-cell marker argmax 判细胞类型, 得到"99.6% 肝细胞",
  这个数字不可信。原因有二:
    ① 按 UMI 取前 N 会**系统性富集多倍体肝细胞核**(肝细胞 RNA 含量远高于
       非实质细胞), 实测 UMI 排名曲线呈双峰:
          rank 10,000 -> UMI 3,103 | rank 20,000 -> 383 | rank 50,000 -> 85
       非实质细胞全部落在 UMI 300-1000 的低丰度尾巴里。
    ② snRNA 存在**ambient RNA 污染**,Alb/Apoa1/Apoa2 是最丰余的转录本,
       几乎每个核都能捡到几条, 直接算 marker 均值会把所有核拉向 hepatocyte。
  因此改为: 先**聚类**, 再在**cluster 层面**用 marker 打分注释。

流程:
  load -> cell calling -> per-sample QC -> scrublet 双细胞
        -> concat -> CP10K+log1p -> HVG -> scale -> PCA -> Harmony -> leiden
        -> cluster 层 marker 注释 -> 保存 h5ad + 报告

用法:
  python scripts/10_preprocess.py [--min-umi 500] [--min-genes 200]
                                  [--n-hvg 2500] [--res 0.8] [--noscrublet]
"""
from __future__ import annotations

import os
import sys
import time
import gzip
import argparse

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp
import scanpy as sc
import anndata as ad

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_sepsis import SAMPLES, RAW_DIR, PROC_DIR, FIG_DIR, MARKERS  # noqa: E402

sc.settings.verbosity = 2
sc.settings.n_jobs = 8


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def read_sample(gsm, suffix, group, raw_dir):
    """读一个样本的 Cell Ranger 三件套, 返回 AnnData(细胞 x 基因, 未过滤)"""
    bc_f = os.path.join(raw_dir, f"{gsm}_barcodes_{suffix}.tsv.gz")
    ft_f = os.path.join(raw_dir, f"{gsm}_features_{suffix}.tsv.gz")
    mt_f = os.path.join(raw_dir, f"{gsm}_matrix_{suffix}.mtx.gz")

    with gzip.open(ft_f, "rt") as fh:
        genes = [ln.rstrip("\n").split("\t")[1] for ln in fh if ln.strip()]
    with gzip.open(bc_f, "rt") as fh:
        bcs = [ln.rstrip("\n").split("\t")[0] for ln in fh if ln.strip()]

    X = sio.mmread(mt_f).tocsc()              # 基因 x 细胞
    A = ad.AnnData(X.T.tocsr().astype(np.float32))   # 细胞 x 基因
    A.var_names = pd.Index(genes)
    A.obs_names = pd.Index([f"{suffix}_{b.split('-')[0]}" for b in bcs])
    A.obs["sample"] = suffix
    A.obs["gsm"] = gsm
    A.obs["group"] = group
    A.var_names_make_unique()
    return A


def cellcall_diagnostics(A):
    """打印 UMI 排名曲线 + 各阈值细胞数, 用于选定 cell calling 阈值"""
    tot = np.asarray(A.X.sum(1)).ravel()
    ng = np.asarray((A.X > 0).sum(1)).ravel()
    o = np.argsort(-tot)
    log("  cell calling 诊断 (UMI 排名曲线):")
    for r in [5000, 10000, 20000, 50000, 100000]:
        if r <= len(o):
            log(f"      rank {r:>7,}: UMI = {tot[o[r-1]]:>9,.0f}")
    for t in [1000, 500, 300]:
        log(f"      UMI>={t:>4} 且 n_genes>=200: "
            f"{int(((tot>=t)&(ng>=200)).sum()):>8,} 个")
    return tot, ng


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-umi", type=int, default=500)
    ap.add_argument("--min-genes", type=int, default=200)
    ap.add_argument("--max-genes", type=int, default=6000)
    ap.add_argument("--max-mt", type=float, default=10.0)
    ap.add_argument("--n-hvg", type=int, default=2500)
    ap.add_argument("--res", type=float, default=0.8)
    ap.add_argument("--noscrublet", action="store_true")
    ap.add_argument("--raw-dir", default=RAW_DIR)
    ap.add_argument("--out", default=os.path.join(PROC_DIR, "merged.h5ad"))
    args = ap.parse_args()

    os.makedirs(PROC_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    log("=" * 70)
    log("C1 主流程 (一): 取数 -> QC -> 整合 -> 聚类 -> 注释")
    log(f"参数: min_umi={args.min_umi} min_genes={args.min_genes} "
        f"max_genes={args.max_genes} max_mt={args.max_mt} "
        f"n_hvg={args.n_hvg} res={args.res}")
    log("=" * 70)

    # ---------------- 1. 逐样本读取 + cell calling + QC ----------------
    adatas, rows = [], []
    for gsm, suffix, group in SAMPLES:
        log(f"[load] {gsm} ({suffix}, {group})")
        A = read_sample(gsm, suffix, group, args.raw_dir)
        n_raw = A.n_obs

        if gsm == SAMPLES[0][0]:
            tot, ng = cellcall_diagnostics(A)

        A.obs["total_counts"] = np.asarray(A.X.sum(1)).ravel()
        A.obs["n_genes"] = np.asarray((A.X > 0).sum(1)).ravel()
        mt = A.var_names.str.startswith(("mt-", "MT-"))
        A.obs["pct_mt"] = (np.asarray(A.X[:, mt].sum(1)).ravel() /
                           np.maximum(A.obs["total_counts"].values, 1) * 100) \
            if mt.any() else 0.0

        keep = ((A.obs["total_counts"] >= args.min_umi) &
                (A.obs["n_genes"] >= args.min_genes) &
                (A.obs["n_genes"] <= args.max_genes) &
                (A.obs["pct_mt"] <= args.max_mt)).values
        n_flt = int(keep.sum())
        A = A[keep].copy()
        log(f"      raw {n_raw:,} -> QC 后 {n_flt:,} ({n_flt/n_raw*100:.2f}%)")

        # ---------------- 2. 双细胞 ----------------
        if not args.noscrublet:
            try:
                sc.pp.scrublet(A, random_state=0)
                nd = int(A.obs["predicted_doublet"].sum())
                A = A[~A.obs["predicted_doublet"].values].copy()
                log(f"      scrublet 剔除双细胞 {nd:,} -> 剩 {A.n_obs:,}")
            except Exception as e:
                log(f"      [warn] scrublet 失败({type(e).__name__}: {e}), 跳过")

        rows.append(dict(gsm=gsm, sample=suffix, group=group,
                         raw=n_raw, after_qc=n_flt, final=A.n_obs,
                         med_umi=float(np.median(A.obs["total_counts"])),
                         med_gene=float(np.median(A.obs["n_genes"]))))
        adatas.append(A)

    pd.DataFrame(rows).to_csv(os.path.join(PROC_DIR, "qc_per_sample.csv"),
                              index=False)
    log("[qc] 逐样本汇总已写 data/proc/qc_per_sample.csv")

    # ---------------- 3. 合并 ----------------
    log("[merge] 合并 6 个样本")
    A = ad.concat(adatas, join="outer", label="batch",
                  keys=[s for _, s, _ in SAMPLES], index_unique="-")
    log(f"        合并后: {A.n_obs:,} 细胞 x {A.n_vars:,} 基因")

    # ---------------- 4. 归一化 / HVG / PCA ----------------
    A.layers["counts"] = A.X.copy()
    sc.pp.normalize_total(A, target_sum=1e4)
    sc.pp.log1p(A)
    A.raw = A

    sc.pp.highly_variable_genes(A, n_top_genes=args.n_hvg, batch_key="batch")
    Asub = A[:, A.var["highly_variable"]].copy()
    sc.pp.scale(Asub, max_value=10)
    sc.tl.pca(Asub, n_comps=50, svd_solver="arpack", random_state=0)
    log(f"[pca] done  {Asub.shape}")

    # ---------------- 5. Harmony 整合 ----------------
    try:
        sc.external.pp.harmony_integrate(Asub, key="batch", basis="X_pca",
                                         adjusted_basis="X_pca_harmony")
        rep = "X_pca_harmony"
        log("[harmony] 批次校正 done")
    except Exception as e:
        rep = "X_pca"
        log(f"[warn] harmony 失败({type(e).__name__}: {e}), 用未校正 PCA")

    # ---------------- 6. 聚类 ----------------
    sc.pp.neighbors(Asub, use_rep=rep, n_neighbors=15, random_state=0)
    sc.tl.leiden(Asub, resolution=args.res, key_added="leiden",
                 flavor="igraph", n_iterations=2, directed=False)
    log(f"[cluster] leiden res={args.res} -> {Asub.obs['leiden'].nunique()} 个 cluster")

    # ---------------- 7. cluster 层注释 ----------------
    A.obs["leiden"] = Asub.obs["leiden"].values
    A.obsm["X_pca"] = Asub.obsm["X_pca"]
    A.obsm[rep] = Asub.obsm[rep]
    A.obsm["X_umap"] = None

    gidx = {g: i for i, g in enumerate(A.var_names)}
    # 每个 cluster 的 marker 平均 log 表达矩阵 (cluster x celltype)
    clusters = sorted(A.obs["leiden"].unique(), key=lambda x: int(x))
    Xl = A[:, :].X
    prof = np.zeros((len(clusters), len(MARKERS)))
    cts = list(MARKERS)
    for ci, c in enumerate(clusters):
        m = (A.obs["leiden"] == c).values
        sub = Xl[m]
        for ti, ct in enumerate(cts):
            ids = [gidx[g] for g in MARKERS[ct] if g in gidx]
            prof[ci, ti] = float(np.asarray(sub[:, ids].mean()).mean())
    # cluster 内 z-score, 消除细胞类型间 marker 表达量级差异(如 Alb 极高)
    z = (prof - prof.mean(0)) / (prof.std(0) + 1e-9)
    assign = {c: cts[int(np.argmax(z[i]))] for i, c in enumerate(clusters)}
    A.obs["celltype"] = A.obs["leiden"].map(assign).astype(str)

    log()
    log("[annot] cluster -> 细胞类型 (cluster 层 marker z-score 注释)")
    comp = pd.crosstab(A.obs["leiden"], A.obs["celltype"])
    for c in clusters:
        n = int((A.obs["leiden"] == c).sum())
        log(f"      {c:>3s} -> {assign[c]:<14s} n={n:>7,} "
            f"({n/A.n_obs*100:5.1f}%)")

    log()
    log("[annot] 细胞类型总览")
    for ct, n in A.obs["celltype"].value_counts().items():
        log(f"      {ct:<14s} {n:>8,}  ({n/A.n_obs*100:5.1f}%)")

    log()
    log("[annot] 分组构成 (行=细胞类型, 列=CLP/Sham)")
    ct_tab = pd.crosstab(A.obs["celltype"], A.obs["group"])
    for line in ct_tab.to_string().split("\n"):
        log("      " + line)

    # ---------------- 8. UMAP + 保存 ----------------
    sc.tl.umap(Asub, random_state=0)
    A.obsm["X_umap"] = Asub.obsm["X_umap"]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    A.write(args.out)
    log()
    log(f"[save] -> {args.out}   ({A.n_obs:,} 细胞 x {A.n_vars:,} 基因)")

    # ---- 肝细胞闸门复核(G1 的诚实版本) ----
    n_hep = int((A.obs["celltype"] == "Hepatocyte").sum())
    sham_hep = int(((A.obs["celltype"] == "Hepatocyte") &
                    (A.obs["group"] == "Sham")).sum())
    log()
    log("=" * 70)
    log(f"[G1 复核] 聚类注释后肝细胞数 = {n_hep:,}")
    log(f"          Sham 组肝细胞 = {sham_hep:,}  "
        f"(门槛 1,500) -> {'PASS' if sham_hep>=1500 else 'FAIL'}")
    log(f"          每 zone 约 = {sham_hep//3:,}  (门槛 500) "
        f"-> {'PASS' if sham_hep//3>=500 else 'FAIL'}")
    log("=" * 70)


if __name__ == "__main__":
    main()
