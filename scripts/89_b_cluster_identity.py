#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
89_b_cluster_identity.py -- B cluster(9,266 细胞)身份深挖

背景
  Fig1 面板 a 把 9,266 个细胞标为 "B",但 87 号核查发现其自身 marker
  (Cd79a/Cd79b/Ms4a1)检出率仅 0.4%,肝 marker(Alb/Apoa1)也仅 2.7% ——
  既不是 B 细胞也不是肝细胞。本脚本深挖它到底是什么。

查什么
  1. QC 三件套:total_counts / n_genes / pct_mt 的分布(B vs 其他类型)
     —— 判断是不是低复杂度裸核(濒死/破损核)
  2. B vs 其余细胞的 top 差异基因 —— 定性其身份
  3. 谱系 marker 全景扫描(9 类 + 免疫/红细胞/血小板/应激/线粒体)
  4. 样本分布 —— 已是已知:73% liver_10 + 26% liver_18

用法: python scripts/89_b_cluster_identity.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import scanpy as sc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_sepsis import MARKERS, PROC_DIR  # noqa: E402

# 额外谱系/低复杂度 marker(超出 lib_sepsis.MARKERS 的补充)
EXTRA = {
    "Pan-immune":       ["Ptprc", "Cd52", "Cd74"],
    "B-specific":       ["Cd79a", "Cd79b", "Ms4a1", "Cd19", "Pax5", "Ighm", "Igkc"],
    "T/NK":             ["Cd3e", "Cd3d", "Nkg7", "Klrb1c", "Trbc2"],
    "Myeloid":          ["Csf1r", "Adgre1", "Lyz2", "Itgam", "S100a8"],
    "Erythroid":        ["Hbb-bs", "Hba-a1", "Hba-a2", "Alas2", "Gypa"],
    "Platelet/megakary": ["Pf4", "Itga2b", "Ppbp"],
    "Proliferating":    ["Mki67", "Top2a", "Pcna", "Cenpa"],
    "Mitochondrial":    ["mt-Co1", "mt-Co3", "mt-Nd1", "mt-Atp6"],
    "Ribosomal":        ["Rps27a", "Rplp0", "Rps6", "Rpl3"],
    "Stress/HS":        ["Hsp90aa1", "Hsp90ab1", "Hspa1b", "Dnajb1", "Junb", "Fos"],
    "Hepatocyte":       ["Alb", "Ttr", "Apoa1", "Apoa2", "Cyp2e1", "Mup3"],
    "Cholangiocyte":    ["Krt19", "Krt7", "Epcam", "Spp1"],
    "HSC/fibroblast":   ["Col1a1", "Dcn", "Acta2", "Rgs5", "Pdgfrb"],
    "Endothelial":      ["Pecam1", "Cdh5", "Stab2", "Lyve1"],
}


def log(*a, **kw):
    print(*a, flush=True, **kw)


def detect_rate(A, gene, mask):
    v = list(A.var_names)
    if gene not in set(v):
        return np.nan
    x = np.asarray(A.X[mask][:, v.index(gene)].todense()).ravel()
    return float((x > 0).mean())


def main():
    A = sc.read_h5ad(os.path.join(PROC_DIR, "merged_no22.h5ad"))
    ct = A.obs["celltype"].values
    bmask = np.where(ct == "B")[0]
    other = np.where(ct != "B")[0]
    log(f"全核 {A.n_obs:,}  B cluster {len(bmask):,}")

    # ---------------- 1. QC 三件套 ----------------
    log()
    log("=" * 92)
    log("1. QC 指标:B cluster vs 其他类型")
    log("=" * 92)
    qrows = []
    for nm, mask in [("B cluster", bmask), ("Hepatocyte", np.where(ct == "Hepatocyte")[0]),
                     ("其他非肝", np.where((ct != "B") & (ct != "Hepatocyte"))[0])]:
        tc = A.obs["total_counts"].values[mask]
        ng = A.obs["n_genes"].values[mask]
        mt = A.obs["pct_mt"].values[mask]
        qrows.append(dict(group=nm, n=len(mask),
                          median_UMI=float(np.median(tc)),
                          median_n_genes=float(np.median(ng)),
                          mean_pct_mt=float(np.mean(mt)),
                          pct_mt_p50=float(np.median(mt))))
        log(f"  {nm:<16s} n={len(mask):>7,}  中位 UMI {np.median(tc):>8.0f}  "
            f"中位基因数 {np.median(ng):>6.0f}  平均 pct_mt {np.mean(mt):.1f}%  "
            f"(中位 {np.median(mt):.1f}%)")
    Q = pd.DataFrame(qrows)
    Q.to_csv(os.path.join(PROC_DIR, "fig1_b_cluster_qc.csv"), index=False)

    # ---------------- 2. top 差异基因 ----------------
    log()
    log("=" * 92)
    log("2. B cluster 的 top 差异基因(vs 其余细胞,按 mean log-norm 表达差)")
    log("=" * 92)
    X = A.X.tocsr()
    # 用均值表达(log-normalized 已存),算 log2FC 与检出率差
    mean_b = np.asarray(X[bmask].mean(axis=0)).ravel()
    mean_o = np.asarray(X[other].mean(axis=0)).ravel()
    det_b = np.asarray((X[bmask] > 0).mean(axis=0)).ravel()
    det_o = np.asarray((X[other] > 0).mean(axis=0)).ravel()
    l2fc = np.log2((mean_b + 1e-6) / (mean_o + 1e-6))
    genes = list(A.var_names)

    df = pd.DataFrame({"gene": genes, "mean_B": mean_b, "mean_other": mean_o,
                       "det_B": det_b, "det_other": det_o, "log2FC": l2fc})
    # 只在 B 里检出的基因(B 特异):检出率差大
    df["det_diff"] = df["det_B"] - df["det_other"]
    # top 上调(B 富集)
    up = df[(df["det_B"] > 0.05) & (df["log2FC"] > 0.5)].sort_values("log2FC", ascending=False)
    log("  —— B 富集(top 40,按 log2FC)——")
    for _, r in up.head(40).iterrows():
        log(f"    {r['gene']:<14s} log2FC {r['log2FC']:+.2f}  "
            f"检出 B {r['det_B']:.1%} vs 其他 {r['det_other']:.1%}")
    # 其他细胞富集(B 缺失)
    dn = df[(df["det_other"] > 0.05) & (df["log2FC"] < -2)].sort_values("log2FC")
    log("  —— B 缺失(top 20,其他细胞有而 B 没有)——")
    for _, r in dn.head(20).iterrows():
        log(f"    {r['gene']:<14s} log2FC {r['log2FC']:+.2f}  "
            f"检出 B {r['det_B']:.1%} vs 其他 {r['det_other']:.1%}")
    df.round(4).to_csv(os.path.join(PROC_DIR, "fig1_b_cluster_degs.csv"), index=False)

    # ---------------- 3. 谱系 marker 全景扫描 ----------------
    log()
    log("=" * 92)
    log("3. 谱系 marker 全景扫描(B cluster 检出率)")
    log("=" * 92)
    for cat, gl in EXTRA.items():
        rates = []
        for g in gl:
            d = detect_rate(A, g, bmask)
            if np.isfinite(d):
                rates.append((g, d))
        if rates:
            top = sorted(rates, key=lambda x: -x[1])
            line = "  ".join(f"{g} {d:.1%}" for g, d in top)
            log(f"  {cat:<18s} " + line)

    log()
    log("=" * 92)
    log("结论提示(供 Fig1 面板 a 标注决策)")
    log("=" * 92)


if __name__ == "__main__":
    main()
