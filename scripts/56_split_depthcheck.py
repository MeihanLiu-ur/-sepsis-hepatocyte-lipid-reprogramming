#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
56_split_depthcheck.py -- 拆分后三条通路的深度校正敏感性分析

为什么需要这一步
  55 号脚本发现:LDLR_clearance 在 snRNA 层是 **+30.3%(升高)**,
  但在独立 bulk 队列里是 **−1.097(降低)**,两层方向相反。
  一个可能的解释是**测序深度混杂** —— CLP 组每个核的 UMI 更多,而
  Ldlr / Lrp1 这类高表达管家型基因在"细胞内排名"上会随深度系统性移动,
  AUCell 是基于秩的,理论上对深度不敏感,但**低检出率基因**
  (Vldlr 0.4%、Lrpap1 7.0%、Ldlrap1 8.8%)在深度变化时秩会剧烈跳动。

做法
  把 counts 二项降采样到每细胞统一 UMI,再 CP10K + log1p,重跑 AUCell,
  看三条拆分通路的方向在原生深度与统一深度下是否一致。

判读
  - 若降采样后 LDLR_clearance 由正转负 → 原信号是深度伪影,以 bulk 为准。
  - 若降采样后仍为正        → 是 snRNA 与 bulk 的真实分歧,必须在正文披露。

用法: python scripts/56_split_depthcheck.py [--ds 1500]
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
from lib_sepsis import LIPID_PROGRAMS, PROC_DIR  # noqa: E402
from _aucell_common import score_all  # noqa: E402


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def downsample_to(hep, target):
    """counts 二项降采样到每细胞 target 条 UMI,再 CP10K + log1p"""
    C = hep.layers["counts"].tocsr()
    C = C.astype(np.int64)
    rng = np.random.default_rng(0)
    indptr, indices, data = C.indptr, C.indices, C.data
    tot = np.asarray(C.sum(1)).ravel()
    p = np.minimum(1.0, target / np.maximum(tot, 1))
    per_cell_p = p[np.repeat(np.arange(C.shape[0]), np.diff(indptr))]
    new_data = rng.binomial(data, per_cell_p)
    D = sp.csr_matrix((new_data, indices, indptr), shape=C.shape)
    D = sp.csr_matrix(D)          # scipy>=1.15 返回 csr_array, anndata 不认
    D.eliminate_zeros()
    tot2 = np.asarray(D.sum(1)).ravel()
    tot2[tot2 == 0] = 1
    D = sp.csr_matrix(sp.diags(1e4 / tot2) @ D)
    D.data = np.log1p(D.data)
    return D.tocsr()


def score(X, genes, auc_frac=0.05):
    """返回 (各程序 per-cell 打分, auc_max)"""
    X = X.tocsr()
    auc_max = int(max(10, auc_frac * X.shape[1]))
    t0 = time.time()
    out, am = score_all(X, genes, LIPID_PROGRAMS, auc_frac)
    log(f"  打分完成 {time.time()-t0:.1f}s (auc_max={am})")
    return out


def per_sample_pct(D, samp, clp_set, prog):
    per = pd.Series(D[prog]).groupby(samp).mean()
    clp = [per[s] for s in per.index if s in clp_set]
    sham = [per[s] for s in per.index if s not in clp_set]
    return (np.mean(clp) - np.mean(sham)) / np.mean(sham) * 100, clp, sham


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", type=int, default=1500)
    ap.add_argument("--inp", default=os.path.join(PROC_DIR, "hep_consensus.h5ad"),
                    help="肝细胞集。默认 B_consensus(注释 ∩ 判据, 59,121 个);"
                         "传 hep_only.h5ad 可复现宽松集(A_loose, 67,145)的结果")
    args = ap.parse_args()

    hep = sc.read_h5ad(args.inp)
    grp = hep.obs["group"].values
    samp = hep.obs["sample"].values
    clp_set = set(samp[grp == "CLP"])
    genes = list(hep.var_names)
    log(f"肝细胞 {hep.n_obs:,}  基因 {hep.n_vars:,}")

    # 深度现状
    tot = np.asarray(hep.layers["counts"].sum(1)).ravel()
    log(f"原生深度: 中位 {np.median(tot):,.0f} UMI/核")
    for s in sorted(set(samp)):
        m = samp == s
        log(f"    {s:<10s} {'CLP' if s in clp_set else 'Sham':<5s} "
            f"中位 {np.median(tot[m]):>8,.0f}  n={m.sum():>6,}")

    log()
    log("原生深度打分 ...")
    nat = score(hep.X.tocsr(), genes)
    log(f"降采样到 {args.ds:,} UMI 后打分 ...")
    dwn = score(downsample_to(hep, args.ds), genes)

    log()
    log("=" * 88)
    log(f"{'program':<22s}{'原生 %Δ':>10s}{f'降采样 {args.ds} %Δ':>16s}   方向一致?")
    log("=" * 88)
    rows = []
    for p in LIPID_PROGRAMS:
        a, cl, sh = per_sample_pct(nat, samp, clp_set, p)
        b, _, _ = per_sample_pct(dwn, samp, clp_set, p)
        same = (a > 0) == (b > 0)
        rows.append(dict(program=p, native_pct=round(float(a), 2),
                         ds_pct=round(float(b), 2),
                         concordant="✓" if same else "✗ 翻转"))
        log(f"  {p:<22s}{a:>10.1f}{b:>16.1f}   {'✓' if same else '✗ 翻转'}")
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(PROC_DIR, "split_depthcheck.csv"), index=False)
    log(f"\n[save] {PROC_DIR}/split_depthcheck.csv")

    log()
    log("=" * 88)
    log("判读")
    log("=" * 88)
    for _, r in R.iterrows():
        if r["concordant"] != "✓":
            log(f"  ⚠️ {r['program']}: 原生 {r['native_pct']:+.1f}% → "
                f"降采样 {r['ds_pct']:+.1f}%,方向翻转,"
                f"说明**原生结果是深度伪影**,应以 bulk 为准。")
    log(f"  方向一致 {int((R['concordant']=='✓').sum())}/{len(R)} 条程序")


if __name__ == "__main__":
    main()
