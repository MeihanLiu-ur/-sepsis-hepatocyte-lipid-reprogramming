#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
87_niche_comment.py -- 非实质细胞注释可信度核查(CellChat 是否可做?)

背景
  蓝图 Fig4 处有一条旧建议:"CellChat 面板建议整体移除或大幅降权",
  理由之一是"该数据的非实质细胞注释不可信(Ptprc/CD45 在各'免疫'类型中
  均接近 0 而 Alb 很高,是 ambient 空液滴)"。
  这条理由当时是**推断**,今天要用数据核实。

查什么
  对每个被注释为非肝的细胞类型,看它**自己的 marker** 检出率有多高。
  若某类型的自身 marker 检出率很低(< 30%),而肝细胞 marker 检出率很高,
  说明这个 cluster 里混了大量带 ambient 的肝细胞 → 注释不可信。

  对照:肝细胞类型的自身 marker 检出率应接近 100%;
       B 细胞在 Fig2 面板 b 里 Alb+ 只有 2.7%,是"干净免疫细胞"的参照。

判据
  CellChat 的前提是**细胞类型身份可靠**(配体来自 A 类型、受体来自 B 类型)。
  若非肝类型的身份普遍不可信,则 CellChat 推断出的互作就是假的 ——
  审稿人只要看一眼 marker 检出率就能推翻。

用法: python scripts/87_niche_comment.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import scanpy as sc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_sepsis import MARKERS, PROC_DIR  # noqa: E402

# 每个类型最特异的 2–3 个 marker(用于"自身身份"打分)
SELF = {
    "Hepatocyte":    ["Alb", "Tat", "Cps1"],
    "Kupffer":       ["Clec4f", "Vsig4", "Timd4"],
    "Endothelial":   ["Pecam1", "Cdh5", "Stab2"],
    "Neutrophil":    ["S100a8", "S100a9", "Retnlg"],
    "T_NK":          ["Cd3e", "Cd3d", "Nkg7"],
    "B":             ["Cd79a", "Cd79b", "Ms4a1"],
    "HSC":           ["Col1a1", "Dcn", "Rgs5"],
    "Cholangiocyte": ["Krt19", "Krt7", "Epcam"],
    "Macrophage":    ["Adgre1", "Csf1r", "Lyz2"],
}
HEP_AMBIENT = ["Alb", "Apoa1", "Apoa2", "Ttr"]


def log(*a, **kw):
    print(*a, flush=True, **kw)


def detect(A, gene, mask):
    v = list(A.var_names)
    if gene not in set(v):
        return np.nan, np.nan
    x = np.asarray(A.X[mask][:, v.index(gene)].todense()).ravel()
    return float((x > 0).mean()), float(np.median(x))


def main():
    A = sc.read_h5ad(os.path.join(PROC_DIR, "merged_no22.h5ad"))
    ct = A.obs["celltype"].values
    order = pd.Series(ct).value_counts().index.tolist()
    log(f"细胞 {A.n_obs:,}")

    log()
    log("=" * 104)
    log("各类型:自身 marker 检出率  vs  肝细胞 ambient marker 检出率")
    log("=" * 104)
    log(f"  {'celltype':<16s}{'n':>8s}{'自身marker':>12s}{'肝ambient':>12s}"
        f"{'差值':>10s}   判读")
    rows = []
    for c in order:
        m = np.where(ct == c)[0]
        if len(m) < 5:
            continue
        self_d, self_e = [], []
        for g in SELF.get(c, []):
            d, e = detect(A, g, m)
            if np.isfinite(d):
                self_d.append(d); self_e.append(e)
        amb_d = []
        for g in HEP_AMBIENT:
            d, _ = detect(A, g, m)
            if np.isfinite(d):
                amb_d.append(d)
        sd = float(np.mean(self_d)) if self_d else np.nan
        ad = float(np.mean(amb_d)) if amb_d else np.nan
        gap = sd - ad
        if c == "Hepatocyte":
            tag = "参照(自身 = ambient 同源)"
        elif sd >= 0.6:
            tag = "✅ 身份可信"
        elif sd >= 0.3:
            tag = "🟡 身份可疑"
        else:
            tag = "❌ 身份不可信 —— 多为带 ambient 的肝细胞"
        rows.append(dict(celltype=c, n=len(m), self_marker=round(sd, 4),
                         hep_ambient=round(ad, 4), gap=round(gap, 4),
                         verdict=tag))
        log(f"  {c:<16s}{len(m):>8,}{sd:>12.1%}{ad:>12.1%}{gap:>10.1%}   {tag}")
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(PROC_DIR, "fig4_niche_credibility.csv"), index=False)
    log(f"\n[save] {PROC_DIR}/fig4_niche_credibility.csv")

    log()
    log("=" * 104)
    log("逐基因明细(自身 marker)")
    log("=" * 104)
    for c in order:
        if c == "Hepatocyte" or c not in SELF:
            continue
        m = np.where(ct == c)[0]
        if len(m) < 5:
            continue
        txt = []
        for g in SELF[c]:
            d, e = detect(A, g, m)
            txt.append(f"{g} {d:5.1%}(中位{e:.2f})")
        log(f"  {c:<16s} " + "   ".join(txt))

    # ---------- 结论 ----------
    nonhep = [r for r in rows if r["celltype"] != "Hepatocyte"]
    ok = [r for r in nonhep if r["self_marker"] >= 0.6]
    mid = [r for r in nonhep if 0.3 <= r["self_marker"] < 0.6]
    bad = [r for r in nonhep if r["self_marker"] < 0.3]
    log()
    log("=" * 104)
    log("对 CellChat 的判定")
    log("=" * 104)
    log(f"  非肝类型 {len(nonhep)} 个中:")
    log(f"    身份可信(自身 marker >= 60%) : {len(ok)} 个  "
        + ", ".join(r["celltype"] for r in ok))
    log(f"    身份可疑(30% – 60%)          : {len(mid)} 个  "
        + ", ".join(r["celltype"] for r in mid))
    log(f"    身份不可信(< 30%)            : {len(bad)} 个  "
        + ", ".join(r["celltype"] for r in bad))
    log()
    if len(ok) + len(mid) < 4:
        log("  → **可用于 CellChat 的可靠类型不足 4 个**,建议**移除** CellChat 面板。")
        log("    细胞间互作推断需要收发双方身份都可靠;身份不可信时,")
        log("    配体-受体分析只会把 ambient 噪音当成生物学信号。")
    else:
        log("  → 有足够多的可靠类型,CellChat 可做,但应只保留可信类型并显式说明。")
    log()
    log("  另一条独立理由(与数据无关):Adv Sci 2025(PMID 40411399)**已在")
    log("  同一批肝脏数据上做过 CellChat**。重复做既不新颖,也给他们审稿时")
    log("  直接对比的机会 —— 我们的注释若不如他们干净,反而减分。")


if __name__ == "__main__":
    main()
