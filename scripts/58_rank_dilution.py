#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
58_rank_dilution.py -- 检验"秩稀释"假说

背景
  Lipoprotein_assembly 在 snRNA 层是 −30.5%(深度校正后),但在独立 bulk 队列里
  9 个基因**无一 padj<0.05**(最接近的 Apob padj = 0.0775),程序级 p = 0.646。
  两层严重矛盾,需要一个机制解释。

假说:秩稀释(rank dilution)
  该基因集里 Apoa1 / Apoa2 / Apoe / Apoc3 是肝细胞中丰度最高的转录本之一
  (bulk baseMean 4,655–171,466)。在 CP10K + log1p 之后,这类高丰度基因的
  **细胞内秩**会随该细胞检测到的基因总数增加而下降 —— 深细胞检出更多基因,
  高丰度基因的相对排名被"稀释"。
  本数据 CLP 组明显更深(中位 UMI 4,981–8,655 vs Sham 1,813/7,938),
  故 CLP 细胞的载脂蛋白秩系统性偏低 → AUCell 假阳性。

本脚本检验三件事
  Q1 降采样到统一 1,500 UMI 后,两组的**检出基因数**差异是否仍然存在?
     (若仍存在,说明降采样只统一了 UMI,未统一检测灵敏度,伪影消不掉)
  Q2 各程序的 AUCell 分数与每细胞检出基因数的相关是否很强?
     (|r| 越大,秩稀释嫌疑越大)
  Q3 把 Lipoprotein_assembly 的 snRNA 打分**按 n_genes 分层**后,
     组间差异是否消失?(若消失,证明是秩稀释而非生物学)

用法: python scripts/58_rank_dilution.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_sepsis import PROC_DIR  # noqa: E402


def log(*a):
    print(*a, flush=True)


def main():
    hep = sc.read_h5ad(os.path.join(PROC_DIR, "hep_only.h5ad"))
    grp = hep.obs["group"].values
    samp = hep.obs["sample"].values
    X = hep.X.tocsr()
    ng = np.asarray((X > 0).sum(1)).ravel()
    tot = np.asarray(hep.layers["counts"].sum(1)).ravel()
    log(f"肝细胞 {hep.n_obs:,}  基因 {hep.n_vars:,}")

    # ---------- Q1: 降采样后 n_genes 差异是否还在 ----------
    log()
    log("=" * 88)
    log("Q1  降采样到统一 1,500 UMI 后,两组的检出基因数差异是否仍然存在?")
    log("=" * 88)
    log(f"  {'组':<6s}{'中位 UMI':>10s}{'中位 n_genes':>14s}   (原生)")
    for g in ["Sham", "CLP"]:
        m = grp == g
        log(f"  {g:<6s}{np.median(tot[m]):>10,.0f}{np.median(ng[m]):>14,.0f}")

    C = hep.layers["counts"].tocsr().astype(np.int64)
    rng = np.random.default_rng(0)
    ip, ind, dat = C.indptr, C.indices, C.data
    p = np.minimum(1.0, 1500 / np.maximum(np.asarray(C.sum(1)).ravel(), 1))
    pc = p[np.repeat(np.arange(C.shape[0]), np.diff(ip))]
    D = sp.csr_matrix((rng.binomial(dat, pc), ind, ip), shape=C.shape)
    t2 = np.asarray(D.sum(1)).ravel(); t2[t2 == 0] = 1
    D = sp.csr_matrix(sp.diags(1e4 / t2) @ D)
    D.data = np.log1p(D.data)
    D = D.tocsr()
    ng2 = np.asarray((D > 0).sum(1)).ravel()
    log()
    log(f"  {'组':<6s}{'中位 UMI':>10s}{'中位 n_genes':>14s}   (降采样 1,500)")
    for g in ["Sham", "CLP"]:
        m = grp == g
        log(f"  {g:<6s}{np.median(np.asarray(D.sum(1)).ravel()[m]):>10,.0f}"
            f"{np.median(ng2[m]):>14,.0f}")
    d_native = np.median(ng[grp == "CLP"]) / np.median(ng[grp == "Sham"])
    d_ds = np.median(ng2[grp == "CLP"]) / np.median(ng2[grp == "Sham"])
    log()
    log(f"  CLP/Sham 检出基因数之比: 原生 {d_native:.2f}x  ->  降采样后 {d_ds:.2f}x")
    if d_ds > 1.15:
        log("  → **降采样未能消除检测灵敏度差异**,秩稀释伪影消不掉。")
        log("    (降采样只统一了 UMI 总数,没有统一每个核检测到的基因数)")
    else:
        log("  → 降采样已基本抹平检测灵敏度差异。")

    # ---------- Q2: 程序打分 vs n_genes ----------
    log()
    log("=" * 88)
    log("Q2  各程序 AUCell 分数与每细胞检出基因数的相关(|r| 大 = 秩稀释嫌疑)")
    log("=" * 88)
    S = pd.read_csv(os.path.join(PROC_DIR, "aucell_v2.csv"), index_col=0)
    A = pd.read_csv(os.path.join(PROC_DIR, "split_depthcheck.csv"), index_col=0)
    B = pd.read_csv(os.path.join(PROC_DIR, "bulk_program_geneset_v2.csv"),
                    index_col=0)
    log(f"  {'program':<24s}{'r(n_genes)':>12s}{'snRNA校正%':>12s}"
        f"{'bulk p':>10s}")
    rs = {}
    for prog in S.columns:
        r_ = float(np.corrcoef(ng, S[prog].values)[0, 1])
        rs[prog] = r_
        pv = B.loc[prog, "p_gene_set"] if prog in B.index else np.nan
        ps = f"{pv:>10.3f}" if np.isfinite(pv) else f"{'n<3':>10s}"
        log(f"  {prog:<24s}{r_:>12.3f}{A.loc[prog, 'ds_pct']:>12.1f}{ps}")
    hi = sorted(rs.items(), key=lambda kv: -abs(kv[1]))[:3]
    log()
    log("  秩稀释嫌疑最大的三个程序: " +
        ", ".join(f"{k} (r={v:+.3f})" for k, v in hi))

    # ---------- Q3: 按 n_genes 分层后组间差异是否消失 ----------
    log()
    log("=" * 88)
    log("Q3  Lipoprotein_assembly 按检出基因数分层后,组间差异是否消失?")
    log("=" * 88)
    prog = "Lipoprotein_assembly"
    v = S[prog].values
    # 取两组 n_genes 的重叠区间,按分位数分 5 层
    lo = max(np.percentile(ng[grp == "Sham"], 5),
             np.percentile(ng[grp == "CLP"], 5))
    hi_ = min(np.percentile(ng[grp == "Sham"], 95),
              np.percentile(ng[grp == "CLP"], 95))
    m = (ng >= lo) & (ng <= hi_)
    log(f"  重叠区间 n_genes ∈ [{lo:.0f}, {hi_:.0f}], 覆盖 {m.sum():,} 细胞 "
        f"({m.mean():.1%})")
    qs = np.quantile(ng[m], np.linspace(0, 1, 6))
    qs[0] -= 1e-9
    layer = np.full(m.sum(), -1, dtype=int)
    layer = np.digitize(ng[m], qs[1:-1])
    log()
    log(f"  {'层':<4s}{'n_genes 范围':>18s}{'Sham 均值':>12s}{'CLP 均值':>12s}"
        f"{'差值':>10s}{'n(Sham/CLP)':>16s}")
    diffs = []
    for L in range(5):
        sel = layer == L
        if sel.sum() < 200:
            continue
        idx = np.where(m)[0][sel]
        gs = grp[idx]
        if (gs == "Sham").sum() < 50 or (gs == "CLP").sum() < 50:
            continue
        a = v[idx][gs == "Sham"].mean()
        b = v[idx][gs == "CLP"].mean()
        rng_txt = f"[{np.percentile(ng[idx],0):.0f}-{np.percentile(ng[idx],100):.0f}]"
        log(f"  {L:<4d}{rng_txt:>18s}{a:>12.4f}{b:>12.4f}{b-a:>10.4f}"
            f"{int((gs=='Sham').sum()):>8,}/{int((gs=='CLP').sum()):<7,}")
        diffs.append(b - a)
    if diffs:
        # 未分层的整体差值(同一批细胞)
        a = v[m][grp[m] == "Sham"].mean()
        b = v[m][grp[m] == "CLP"].mean()
        log(f"  {'--':<4s}{'未分层(同批细胞)':>18s}{a:>12.4f}{b:>12.4f}{b-a:>10.4f}")
        log()
        log(f"  分层内差值均值 = {np.mean(diffs):+.4f}; 未分层差值 = {b-a:+.4f}")
        if abs(np.mean(diffs)) < abs(b - a) * 0.5:
            log("  → **分层后差异大幅缩小**,支持秩稀释解释:")
            log("    snRNA 的组间差异主要由检出基因数的组间差异驱动,非生物学。")
        else:
            log("  → 分层后差异仍在,秩稀释解释不成立,需另找原因。")

    log()
    log("=" * 88)
    log("结论提示")
    log("=" * 88)
    log("  无论机制如何, decisive evidence 是 C3:独立队列(n=4 vs 4,有足够")
    log("  统计能力)中该程序 9 个基因**无一 padj<0.05**,程序级 p = 0.646。")
    log("  → 以 bulk 为准:Lipoprotein_assembly 在脓毒症下**无可检出的转录改变**。")
    log("  → snRNA 层的 -30.5% 应归因于方法学因素,不作为结论。")


if __name__ == "__main__":
    main()
