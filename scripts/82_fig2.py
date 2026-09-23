#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
82_fig2.py -- Figure 2:肝细胞识别、程序打分概览与区带维度

与 Fig1 的分工(避免重复)
  Fig1 面板 e 画的是**判据规则本身**(散点 + 阈值线)
  Fig2 画的是**规则的效果**与**程序/区带两个维度的可用性**

版面(2 行 × 3 列)
  a  dot plot:9 类细胞 × marker —— 证明聚类注释可信
  b  **ambient RNA 核查**:肝细胞高丰度基因(Alb/Apoa1/Apoa2/Ttr/Mup3)
     在各细胞类型中的检出率。snRNA 里这些基因几乎每个核都能捡到,
     若非肝类型也高检出 → ambient 污染,per-cell 单看 marker 均值会系统性
     偏向 hepatocyte。这正是我们必须用**双条件判据**的原因。
  c  12 条脂质程序 AUCell 打分的整体分布(小提琴)
  d  区带 landmark 沿 zonation index 的表达热图(证明区带梯度确实存在)
  e  肝细胞 UMAP 按 zonation index 着色
  f  各样本 zonation index 分布 —— ⚠️ 诚实展示 Sham 组内异质性
     (liver_14 几乎无梯度、liver_18 梯度明显),这是"区带结论站不住"的原因

依赖
  data/proc/merged_no22.h5ad   84,619 核(含非肝细胞)—— 面板 a/b/e
  data/proc/hep_only.h5ad      67,145 肝细胞      —— 面板 c/d/f
  data/proc/aucell_v2.csv      12 条程序 per-cell 打分(v4 定义)
  data/proc/aucell_main.csv    zonation_index(v3 未重算,仍在此文件)

用法: python scripts/82_fig2.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
try:
    import scanpy as sc
except ModuleNotFoundError:
    import anndata as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_sepsis import MARKERS, PROC_DIR, FIG_DIR, LIPID_PROGRAMS  # noqa: E402

SHAM = "#6C7A89"
CLP = "#E17055"
ACCENT = "#2D3436"
CTYPE_COLORS = {
    "Hepatocyte": "#E17055", "Endothelial": "#3498DB", "HSC": "#9B59B6",
    "Kupffer": "#16A085", "Macrophage": "#1ABC9C", "Neutrophil": "#E67E22",
    "T_NK": "#34495E", "Neuron": "#E84393", "Cholangiocyte": "#27AE60",
}
# ambient RNA 核查用的肝细胞高丰度基因(snRNA 中最容易被"捡到"的)
AMBIENT = ["Alb", "Apoa1", "Apoa2", "Ttr", "Mup3", "Serpina1a"]

N_BIN = 12

plt.rcParams.update({
    "font.size": 9.5, "axes.labelsize": 10, "axes.titlesize": 11,
    "axes.linewidth": 0.8, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "DejaVu Sans",
})


def log(*a, **kw):
    print(*a, flush=True, **kw)


def main():
    import argparse as _ap
    _p = _ap.ArgumentParser()
    _p.add_argument("--inp", default=os.path.join(PROC_DIR, "hep_consensus.h5ad"),
                    help="肝细胞集。默认 B_consensus(注释 ∩ 判据, 59,121 个)")
    _p.add_argument("--all", default=os.path.join(PROC_DIR, "merged_no22.h5ad"),
                    help="全核对象(含非肝细胞),用于面板 a/b")
    _a = _p.parse_args()
    os.makedirs(FIG_DIR, exist_ok=True)
    A = sc.read_h5ad(_a.all)
    hep = sc.read_h5ad(_a.inp)
    S12 = pd.read_csv(os.path.join(PROC_DIR, "aucell_v2.csv"), index_col=0)
    Z = pd.read_csv(os.path.join(PROC_DIR, "aucell_main.csv"), index_col=0)
    zi = Z.loc[hep.obs_names, "zonation_index"].values
    log(f"全核 {A.n_obs:,} / 肝细胞 {hep.n_obs:,} / 程序 {S12.shape[1]} 条")

    ct = A.obs["celltype"].values
    # 2026-08-29:B → Neuron(神经元污染,见 lib_sepsis.MARKERS 注与 89 号脚本)
    ct = np.where(ct == "B", "Neuron", ct)
    grp = A.obs["group"].values
    samp = A.obs["sample"].values
    hgrp = hep.obs["group"].values
    hsamp = hep.obs["sample"].values
    vnames = list(A.var_names)

    fig, axes = plt.subplots(2, 3, figsize=(15.6, 8.8))
    ax = axes.ravel()

    # ---------- a. dot plot:细胞类型 × marker ----------
    a0 = ax[0]
    order = pd.Series(ct).value_counts().index.tolist()
    genes = [g for gl in MARKERS.values() for g in gl]
    genes = [g for g in genes if g in set(vnames)]
    X = A.X.tocsr()
    gi = [vnames.index(g) for g in genes]
    det = np.zeros((len(order), len(genes)))
    exp = np.zeros((len(order), len(genes)))
    for i, c in enumerate(order):
        m = np.where(ct == c)[0]
        sub = X[m][:, gi]
        det[i] = np.asarray((sub > 0).mean(axis=0)).ravel()
        # 该类型内的相对表达(z-score across cell types 的行内标准化在后面做)
        exp[i] = np.asarray(sub.mean(axis=0)).ravel()
    # 行内标准化,让小细胞类型也看得见
    exp_n = (exp - exp.mean(axis=0)) / (exp.std(axis=0) + 1e-9)
    exp_n = np.clip(exp_n, -2.5, 2.5)
    yy, xx = np.mgrid[0:len(order), 0:len(genes)]
    sc0 = a0.scatter(xx.ravel(), yy.ravel(),
                     s=(det.ravel() * 130 + 4), c=exp_n.ravel(),
                     cmap="RdBu_r", vmin=-2.5, vmax=2.5,
                     edgecolors="white", linewidths=0.4)
    a0.set_yticks(range(len(order)))
    a0.set_yticklabels(order, fontsize=7.2)
    a0.set_xticks(range(len(genes)))
    a0.set_xticklabels(genes, rotation=90, fontsize=5.6)
    a0.set_title("a  Cluster annotation: marker dot plot", loc="left",
                 fontweight="bold")
    a0.grid(lw=0.2, color="#ECF0F1"); a0.set_axisbelow(True)
    plt.colorbar(sc0, ax=a0, fraction=0.03, pad=0.02, label="rel. expr (z)")
    for i, c in enumerate(order):
        a0.scatter([], [], s=0)   # 占位保持布局
    # 2026-08-30:注释原在右上角(0.99, 0.98),会压在第一行(Macrophage)最右侧的
    # 大点上。移到右下角 —— 对应最后一行细胞类型 + 末端 marker,恒为空区。
    a0.text(0.99, 0.02, "dot size = detection rate",
            transform=a0.transAxes, ha="right", va="bottom", fontsize=6.2,
            color="#7F8C8D",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", lw=0))

    # ---------- b. ambient RNA 核查 ----------
    b0 = ax[1]
    amb = [g for g in AMBIENT if g in set(vnames)]
    det_amb = np.zeros((len(order), len(amb)))
    for i, c in enumerate(order):
        m = np.where(ct == c)[0]
        for j, g in enumerate(amb):
            det_amb[i, j] = float((X[m][:, vnames.index(g)] > 0).mean())
    xs = np.arange(len(amb))
    w = 0.8 / len(order)
    for i, c in enumerate(order):
        b0.bar(xs + (i - len(order) / 2) * w, det_amb[i], width=w,
               color=CTYPE_COLORS.get(c, "#95A5A6"), label=c,
               edgecolor="white", linewidth=0.3)
    b0.set_xticks(xs)
    b0.set_xticklabels(amb, rotation=30, ha="right", fontsize=7)
    b0.set_ylabel("detection rate")
    b0.set_ylim(0, 1.5)   # 顶部留白,给说明文本框腾空间,不压 bar
    b0.set_title("b  ⚠ Ambient RNA: hepatocyte genes leak everywhere",
                 loc="left", fontweight="bold")
    b0.legend(fontsize=5.8, frameon=False, ncol=3)
    b0.grid(axis="y", lw=0.3, color="#ECF0F1"); b0.set_axisbelow(True)
    # 标出非肝类型里最高的 Alb 检出率
    nonhep = [i for i, c in enumerate(order) if c != "Hepatocyte"]
    if nonhep:
        j_alb = amb.index("Alb") if "Alb" in amb else 0
        worst = max(nonhep, key=lambda i: det_amb[i, j_alb])
        b0.annotate(f"{order[worst]}: {det_amb[worst, j_alb]:.0%}",
                    xy=(j_alb + (worst - len(order) / 2) * w, det_amb[worst, j_alb]),
                    textcoords="offset points", xytext=(0, 6), fontsize=6,
                    color=BAD if det_amb[worst, j_alb] > 0.3 else ACCENT,
                    ha="center")
    # 说明精简为单行,放顶部留白(data y=1.12),不压 bar
    b0.text(0.0, 1.12,
            "ambient RNA biases per-cell means → consensus set "
            "(dual criterion): %s nuclei"
            % f"{hep.n_obs:,}",
            fontsize=6.2, color=ACCENT, va="bottom", ha="left")

    # ---------- c. 12 条程序打分分布 ----------
    c0 = ax[2]
    progs = list(S12.columns)
    data = [S12[p].values for p in progs]
    rng = np.random.default_rng(0)
    data_s = [rng.choice(v, size=min(len(v), 6000), replace=False) for v in data]
    parts = c0.violinplot(data_s, positions=range(len(progs)),
                          widths=0.8, showextrema=False)
    for bd in parts["bodies"]:
        bd.set_facecolor("#8E44AD"); bd.set_alpha(0.75)
        bd.set_edgecolor("white"); bd.set_linewidth(0.5)
    for i, v in enumerate(data_s):
        c0.scatter([i], [np.median(v)], s=12, color="white",
                   edgecolors=ACCENT, linewidths=0.8, zorder=5)
    c0.set_xticks(range(len(progs)))
    c0.set_xticklabels([p.replace("_", " ") for p in progs],
                       rotation=90, ha="right", fontsize=6.2)
    c0.set_ylabel("AUCell score")
    c0.set_title("c  12 lipid programs are all scoreable", loc="left",
                 fontweight="bold")
    c0.grid(axis="y", lw=0.3, color="#ECF0F1"); c0.set_axisbelow(True)

    # ---------- d. 区带 landmark 热图 ----------
    d0 = ax[3]
    hv = list(hep.var_names)
    Xh = hep.X.tocsr()
    bins = np.full(len(zi), -1, dtype=int)
    q = np.quantile(zi, np.linspace(0, 1, N_BIN + 1))
    q[0] -= 1e-9
    bins = np.digitize(zi, q[1:-1])
    zone_genes = ([g for g in ZONE_PP if g in set(hv)] +
                  [g for g in ZONE_PC if g in set(hv)])
    M = np.zeros((len(zone_genes), N_BIN))
    for i, g in enumerate(zone_genes):
        x = np.asarray(Xh[:, hv.index(g)].todense()).ravel()
        for b in range(N_BIN):
            m = bins == b
            M[i, b] = x[m].mean() if m.sum() > 50 else np.nan
    # 行内 z-score
    Mn = (M - np.nanmean(M, axis=1, keepdims=True)) / (
        np.nanstd(M, axis=1, keepdims=True) + 1e-9)
    im = d0.imshow(Mn, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    d0.set_yticks(range(len(zone_genes)))
    d0.set_yticklabels(zone_genes, fontsize=6.0)
    d0.set_xticks([0, N_BIN // 2, N_BIN - 1])
    d0.set_xticklabels(["portal", "mid", "central"], fontsize=7)
    d0.set_title("d  Zonation landmarks form a gradient", loc="left",
                 fontweight="bold")
    # 分隔线:前 n_pp 行是 periportal
    n_pp = len([g for g in ZONE_PP if g in set(hv)])
    if 0 < n_pp < len(zone_genes):
        d0.axhline(n_pp - 0.5, color=ACCENT, lw=1.0, ls="--")
        d0.text(-1.8, n_pp / 2, "periportal", rotation=90, va="center",
                ha="right", fontsize=6, color=ACCENT)
        d0.text(-1.8, (n_pp + len(zone_genes)) / 2, "pericentral", rotation=90,
                va="center", ha="right", fontsize=6, color=ACCENT)
    plt.colorbar(im, ax=d0, fraction=0.03, pad=0.02, label="z")

    # ---------- e. UMAP 按 zonation index ----------
    e0 = ax[4]
    U = hep.obsm["X_umap"]
    idx = rng.choice(len(zi), size=min(len(zi), 30000), replace=False)
    sc1 = e0.scatter(U[idx, 0], U[idx, 1], c=zi[idx], s=2.0,
                     cmap="RdBu_r", vmin=np.percentile(zi, 5),
                     vmax=np.percentile(zi, 95), edgecolors="none")
    e0.set_xlabel("UMAP 1"); e0.set_ylabel("UMAP 2")
    e0.set_title("e  Hepatocytes coloured by zonation index", loc="left",
                 fontweight="bold")
    e0.set_xticks([]); e0.set_yticks([])
    plt.colorbar(sc1, ax=e0, fraction=0.03, pad=0.02,
                 label="central − portal")

    # ---------- f. 各样本 zonation index 分布 ----------
    f0 = ax[5]
    labels = sorted(set(hsamp))
    pos = np.arange(len(labels))
    parts = f0.violinplot([zi[hsamp == s] for s in labels], positions=pos,
                          widths=0.8, showextrema=False)
    for i, s in enumerate(labels):
        col = CLP if s in set(hsamp[hgrp == "CLP"]) else SHAM
        parts["bodies"][i].set_facecolor(col)
        parts["bodies"][i].set_alpha(0.75)
        parts["bodies"][i].set_edgecolor("white")
        parts["bodies"][i].set_linewidth(0.5)
    for i, s in enumerate(labels):
        v = zi[hsamp == s]
        f0.scatter([i], [np.median(v)], s=14, color="white",
                   edgecolors=ACCENT, linewidths=0.8, zorder=5)
    f0.set_xticks(pos)
    f0.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    f0.set_ylabel("zonation index")
    f0.set_title("f  ⚠ Sham within-group heterogeneity", loc="left",
                 fontweight="bold")
    f0.grid(axis="y", lw=0.3, color="#ECF0F1"); f0.set_axisbelow(True)
    sds = {s: float(zi[hsamp == s].std()) for s in labels}
    iqrs = {s: float(np.subtract(*np.percentile(zi[hsamp == s], [75, 25])))
            for s in labels}
    sham_samps = [s for s in labels if s not in set(hsamp[hgrp == "CLP"])]
    if len(sham_samps) >= 2:
        sd_ratio = max(sds[s] for s in sham_samps) / \
                   min(sds[s] for s in sham_samps)
        iqr_ratio = max(iqrs[s] for s in sham_samps) / \
                    min(iqrs[s] for s in sham_samps)
    else:
        sd_ratio = iqr_ratio = float("nan")
    worst_s = min(sds, key=sds.get)
    best_s = max(sds, key=sds.get)
    f0.text(0.02, 0.95,
            f"Sham SD ratio = {sd_ratio:.2f}  |  IQR ratio = {iqr_ratio:.2f}\n"
            f"({worst_s} SD {sds[worst_s]:.3f} vs {best_s} SD {sds[best_s]:.3f})\n"
            "one sham liver is nearly flat → zonation\n"
            "comparisons are underpowered (Fig S zonation)",
            transform=f0.transAxes, va="top", fontsize=6.2, color=ACCENT,
            bbox=dict(boxstyle="round,pad=0.3", fc="#FFF9E6",
                      ec="#BDC3C7", lw=0.6))

    fig.suptitle("Figure 2  Hepatocyte identification, lipid-program scoring, "
                 "and the zonation dimension\n"
                 "(GSE275689; %s nuclei, %s hepatocytes after dual-criterion "
                 "selection; 12 lipid programs)"
                 % (f"{A.n_obs:,}", f"{hep.n_obs:,}"),
                 fontsize=10.5, fontweight="bold", y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.935])
    for ext in ("pdf", "png"):
        p = os.path.join(FIG_DIR, f"Fig2_hepatocyte_programs.{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight")
        log(f"[save] {p}")
    plt.close(fig)

    # 控制台归账
    log()
    log("=" * 88)
    log("ambient RNA 核查(面板 b 的数字,写进 Methods):")
    log("=" * 88)
    log(f"  {'celltype':<16s}" + "".join(f"{g:>10s}" for g in amb))
    for i, c in enumerate(order):
        log(f"  {c:<16s}" + "".join(f"{det_amb[i,j]:>10.1%}"
                                    for j in range(len(amb))))
    log()
    log("  判读:若非肝类型的 Alb/Apoa1 检出率仍很高,则 per-cell marker 均值")
    log("        会系统性偏向 hepatocyte —— 这正是双条件判据要挡住的问题。")
    log()
    log("=" * 88)
    log("各样本 zonation index 离散度(面板 f):")
    log("=" * 88)
    for s in labels:
        v = zi[hsamp == s]
        log(f"  {s:<10s} {'CLP' if s in set(hsamp[hgrp=='CLP']) else 'Sham':<5s} "
            f"n={len(v):>6,}  SD={v.std():.3f}  "
            f"IQR={np.subtract(*np.percentile(v,[75,25])):.3f}")


# 区带 landmark(与 lib_sepsis.ZONE 一致,避免循环 import 顺序问题)
ZONE_PP = ["Ass1", "Arg1", "Cps1", "Otc", "Gls2", "Sds", "Hal",
           "Tat", "Hpd", "Pck1", "Cyp2f2", "Aldob"]
ZONE_PC = ["Glul", "Cyp2e1", "Cyp1a2", "Cyp2c29", "Oat", "Slc1a2",
           "Axin2", "Rgn", "Tbx3", "Nnmt", "Gulo"]

BAD = "#C0392B"

if __name__ == "__main__":
    main()
