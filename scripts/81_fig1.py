#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
81_fig1.py -- Figure 1:数据概览与质控(含 liver_22 排除证据)

版面(2 行 × 3 列)
  a  UMAP 全细胞图谱(按细胞类型着色)
  b  UMAP 按分组着色 —— 证明 CLP 与 Sham 细胞充分混合(无批次主导)
  c  QC 指标:每样本的 UMI / 检出基因 / 线粒体比例
  d  **liver_22 排除证据**(本文方法学加分项)
  e  肝细胞提取:per-cell 双条件判据散点
  f  每样本细胞数与肝细胞数

⚠️ 关于 liver_22 —— 必须讲清楚,否则审稿人会误解
  GEO 元数据写 GSM8482483 = "liver, Sham, rep3, tissue: liver"。
  但**标准 QC 之后它仍有 9,571 个细胞**,看起来"完全正常"。
  问题是这些细胞**不是肝细胞**:
    在**未过滤原始矩阵**(1,861,147 barcodes)上统计 marker 阳性数,
    Alb+ 仅 ~1,124(其余样本 33,598–37,541,差 30 倍)、Tat+ 仅 464,
    而脂肪标志物 Fabp4+ 高达 56,329。
  其高表达谱为 Fabp4 / Cfd / Lpl / Cd36 / Ebf1 / Car3 / Ghr —— **脂肪(WAT)谱系**。
  Adv Sci 2025 同批做了肝/肾/WAT/脑四器官,疑为提交者样本混淆。

  → 因此本图的 d 面板跑在**未过滤矩阵**上,用 marker 阳性 barcode 数说话,
    而不是用 QC 后的细胞数。这两件事的区别必须让读者一眼看懂。

用法: python scripts/81_fig1.py
"""
from __future__ import annotations

import os
import sys
import gzip

import numpy as np
import pandas as pd
try:
    import scanpy as sc
except ModuleNotFoundError:
    import anndata as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_sepsis import SAMPLES, RAW_DIR, PROC_DIR, FIG_DIR  # noqa: E402

SHAM = "#6C7A89"
CLP = "#E17055"
ACCENT = "#2D3436"
BAD = "#C0392B"
OK = "#27AE60"

# 未过滤矩阵的 marker 探针:UMI >= 200 的 barcode 才算"一个核"
MIN_UMI = 200
PROBE = {
    "Hepatocyte":  ["Alb", "Tat", "Cps1", "Apoa1", "Ttr", "Serpina1a"],
    "Adipose_WAT": ["Fabp4", "Cfd", "Lpl", "Car3", "Ghr", "Ebf1"],
    "Endothelial": ["Pecam1", "Cdh5", "Stab2"],
    "Immune":      ["Ptprc", "Cd68", "Lyz2"],
}

# UMAP 细胞类型配色(与 MARKERS 顺序无关,按出现频次手工定)
CTYPE_COLORS = {
    "Hepatocyte":    "#E17055",
    "Endothelial":   "#3498DB",
    "HSC":           "#9B59B6",
    "Kupffer":       "#16A085",
    "Macrophage":    "#1ABC9C",
    "Neutrophil":    "#E67E22",
    "T_NK":          "#34495E",
    "Neuron":        "#E84393",
    "Cholangiocyte": "#27AE60",
}

plt.rcParams.update({
    "font.size": 9.5, "axes.labelsize": 10, "axes.titlesize": 11,
    "axes.linewidth": 0.8, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "DejaVu Sans",
})


def log(*a, **kw):
    print(*a, flush=True, **kw)


def probe_raw(gsm, suffix):
    """
    在未过滤原始矩阵上统计 marker 阳性 barcode 数。

    只流式读取关心的基因行,不把整个矩阵读进内存 ——
    单样本矩阵 nnz 可达数千万,全读会爆内存且极慢。
    """
    fpath = os.path.join(RAW_DIR, f"{gsm}_features_{suffix}.tsv.gz")
    mpath = os.path.join(RAW_DIR, f"{gsm}_matrix_{suffix}.mtx.gz")
    if not (os.path.exists(fpath) and os.path.exists(mpath)):
        return None
    names = []
    with gzip.open(fpath, "rt") as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            names.append(p[1] if len(p) > 1 else p[0])
    want = {}
    for grp, gl in PROBE.items():
        for g in gl:
            if g in set(names):
                want.setdefault(names.index(g) + 1, (grp, g))   # mtx 是 1-based
    if not want:
        return None

    want = {}
    for grp, gl in PROBE.items():
        for g in gl:
            if g in set(names):
                want.setdefault(names.index(g) + 1, (grp, g))   # mtx 是 1-based
    if not want:
        return None

    # 先探明注释行数与维度(mtx 的注释行数量不固定)
    n_skip = 0
    with gzip.open(mpath, "rt") as fh:
        for line in fh:
            if line.startswith("%"):
                n_skip += 1
            else:
                n_gene, n_bar, nnz = (int(x) for x in line.split())
                n_skip += 1
                break

    # ⚠️ 关键:总 UMI 必须累加**所有基因**,不能只累加 marker 基因。
    #    否则 "UMI >= 200" 会退化成 "marker 基因 UMI 之和 >= 200",
    #    导致高表达 marker 的样本被系统性选入 —— 判据就失效了。
    kw = dict(sep=r"\s+", comment="%", skiprows=n_skip, header=None,
              names=["g", "b", "v"], engine="c")
    tot = np.zeros(n_bar + 1, dtype=np.float64)
    for ch in pd.read_csv(mpath, chunksize=4_000_000, **kw):
        tot += np.bincount(ch["b"].values.astype(np.int64),
                           weights=ch["v"].values.astype(np.float64),
                           minlength=n_bar + 1)
    n_umi200 = int((tot >= MIN_UMI).sum())

    # 第二遍:在 UMI >= MIN_UMI 的 barcode 上统计 marker 阳性数
    keep = tot >= MIN_UMI
    wlist = np.array(sorted(want))
    pos = {gi: 0 for gi in wlist}
    for ch in pd.read_csv(mpath, chunksize=4_000_000, **kw):
        b = ch["b"].values.astype(np.int64)
        g = ch["g"].values.astype(np.int64)
        m = keep[b] & np.isin(g, wlist)
        if not m.any():
            continue
        ug, uc = np.unique(g[m], return_counts=True)
        for gi, c in zip(ug, uc):
            pos[int(gi)] += int(c)

    out = {"n_barcodes_umi200": n_umi200, "n_barcodes_total": n_bar}
    for gi, (grp, g) in want.items():
        out[f"{grp}:{g}"] = pos.get(gi, 0)
    return out


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    # ---------------- 1. 未过滤矩阵 marker 探针 ----------------
    cache = os.path.join(PROC_DIR, "fig1_raw_probe.csv")
    if os.path.exists(cache):
        PR = pd.read_csv(cache, index_col=0)
        log(f"[cache] {cache}")
    else:
        log("=" * 88)
        log("未过滤原始矩阵 marker 探针(UMI >= 200 的 barcode 才计数)")
        log("=" * 88)
        rows = {}
        for gsm, suffix, group in SAMPLES:
            log(f"  {suffix} ({group}) ... ", end="")
            r = probe_raw(gsm, suffix)
            if r is None:
                log("缺文件,跳过")
                continue
            rows[suffix] = r
            log(f"UMI>={MIN_UMI} 的 barcode {r['n_barcodes_umi200']:,}")
        PR = pd.DataFrame(rows).T
        PR.index.name = "sample"
        PR.to_csv(cache)
        log(f"\n[save] {cache}")

    log()
    log("=" * 96)
    log("liver_22 排除证据:肝细胞 vs 脂肪 marker 阳性 barcode 数")
    log("=" * 96)
    hep_cols = [c for c in PR.columns if c.startswith("Hepatocyte:")]
    wat_cols = [c for c in PR.columns if c.startswith("Adipose_WAT:")]
    show = pd.DataFrame({
        "肝 Alb": PR.get("Hepatocyte:Alb"),
        "肝 Tat": PR.get("Hepatocyte:Tat"),
        "肝 Cps1": PR.get("Hepatocyte:Cps1"),
        "脂 Fabp4": PR.get("Adipose_WAT:Fabp4"),
        "脂 Cfd": PR.get("Adipose_WAT:Cfd"),
        "脂 Lpl": PR.get("Adipose_WAT:Lpl"),
    })
    for line in show.to_string().split("\n"):
        log("  " + line)

    # ---------------- 2. 载入整合数据 ----------------
    A = sc.read_h5ad(os.path.join(PROC_DIR, "merged_no22.h5ad"))
    log(f"\n整合数据 {A.n_obs:,} 细胞 / {A.n_vars:,} 基因")
    grp = A.obs["group"].values
    samp = A.obs["sample"].values
    ct = A.obs["celltype"].values
    # 2026-08-29:B cluster 经 89 号脚本定性为神经元污染(见 lib_sepsis.MARKERS 注),
    # 改标 Neuron(只改呈现层,不动 h5ad;肝细胞主分析不受影响)
    ct = np.where(ct == "B", "Neuron", ct)
    is_hep = A.obs["is_hep"].values

    fig, axes = plt.subplots(2, 3, figsize=(15.6, 9.0))
    ax = axes.ravel()
    U = A.obsm["X_umap"]
    rng = np.random.default_rng(0)

    # ---------- a. UMAP 按细胞类型 ----------
    a0 = ax[0]
    order = pd.Series(ct).value_counts().index.tolist()
    for c in order:
        m = ct == c
        a0.scatter(U[m, 0], U[m, 1], s=2.2, color=CTYPE_COLORS.get(c, "#95A5A6"),
                   label=c, alpha=0.75, edgecolors="none")
    a0.set_xlabel("UMAP 1"); a0.set_ylabel("UMAP 2")
    a0.set_title("a  All nuclei by cell type (n = %s)" % f"{A.n_obs:,}",
                 loc="left", fontweight="bold")
    a0.set_xticks([]); a0.set_yticks([])
    # 图例放左上空白区(y>12 几乎无核;右上角是肝细胞大簇会重叠)。
    # 紧凑字号/间距使 9 项图例高度压到 ~25%,底部停在 Neuron 簇(y 上缘 11)之上
    a0.legend(markerscale=2.6, frameon=True, facecolor="white", framealpha=0.88,
              edgecolor="#BDC3C7", fontsize=5.6, loc="upper left",
              borderpad=0.28, labelspacing=0.18, handlelength=1.1,
              handletextpad=0.5)

    # ---------- b. UMAP 按分组 ----------
    b0 = ax[1]
    for g, col in [("Sham", SHAM), ("CLP", CLP)]:
        m = grp == g
        b0.scatter(U[m, 0], U[m, 1], s=2.2, color=col, label=g,
                   alpha=0.75, edgecolors="none")
    b0.set_xlabel("UMAP 1"); b0.set_ylabel("UMAP 2")
    b0.set_title("b  By condition — no batch domination", loc="left",
                 fontweight="bold")
    b0.set_xticks([]); b0.set_yticks([])
    b0.legend(markerscale=3, frameon=True, facecolor="white", framealpha=0.82,
              edgecolor="#BDC3C7", fontsize=7.5, loc="upper left")

    # ---------- c. QC 指标 ----------
    c0 = ax[2]
    qc = pd.read_csv(os.path.join(PROC_DIR, "qc_per_sample.csv"))
    qc = qc[qc["sample"].isin([s for _, s, _ in SAMPLES if s != "liver_22"])]
    labels = qc["sample"].tolist()
    xs = np.arange(len(labels))
    colors = [CLP if g == "CLP" else SHAM for g in qc["group"]]
    med_umi = qc["med_umi"].values.astype(float)
    c0.bar(xs, med_umi, color=colors, edgecolor="white", linewidth=0.6)
    c0.set_xticks(xs)
    c0.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    c0.set_ylabel("median UMI per nucleus")
    # 2026-08-30:原标题 "(⚠ CLP deeper)" 与稿件改后的措辞冲突。本面板画的是
    # **全保留细胞核**(84,619)的 med_umi,与 Methods 引用的**肝细胞共识集**
    # (59,121)口径不同 —— 两口径的深浅排序并不一致(肝细胞集里最深的库是
    # Sham liver_18 = 9,773;全核口径下最深的库是 CLP liver_06 = 8,446)。
    # 故标题改为中性、只陈述口径,并补画组级中位参考线让事实自陈,不做因果断言。
    c0.set_title("c  Sequencing depth per sample (all retained nuclei)",
                 loc="left", fontweight="bold")
    for g, col in [("CLP", CLP), ("Sham", SHAM)]:
        m = (qc["group"] == g).values
        if m.sum() == 0:
            continue
        gm = float(np.median(med_umi[m]))
        c0.axhline(gm, color=col, ls="--", lw=1.0, alpha=0.9, zorder=1)
        c0.text(0.985, gm, f" {g} median {gm:,.0f} ",
                transform=c0.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=6.2, color=col,
                fontweight="bold")
    # 顶部留白 30%:容纳柱顶数值标注与两条组级中位标签,避免压线
    c0.set_ylim(0, float(med_umi.max()) * 1.30)
    for i, v in enumerate(med_umi):
        c0.text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=6.5,
                color=ACCENT)
    c0.grid(axis="y", lw=0.3, color="#ECF0F1"); c0.set_axisbelow(True)

    # ---------- d. liver_22 排除证据 ----------
    d0 = ax[3]
    samp_order = [s for _, s, _ in SAMPLES]
    h = PR.loc[samp_order, "Hepatocyte:Alb"].values.astype(float)
    w = PR.loc[samp_order, "Adipose_WAT:Fabp4"].values.astype(float)
    xs = np.arange(len(samp_order))
    wc = [BAD if s == "liver_22" else "#BDC3C7" for s in samp_order]
    d0.bar(xs - 0.2, h, width=0.4, color="#34495E", label="Alb+ (hepatocyte)",
           edgecolor="white", linewidth=0.5)
    d0.bar(xs + 0.2, w, width=0.4, color="#E67E22", label="Fabp4+ (adipocyte)",
           edgecolor="white", linewidth=0.5)
    d0.set_xticks(xs)
    d0.set_xticklabels([("liver_22\n⚠ EXCLUDED" if s == "liver_22" else s)
                        for s in samp_order], rotation=30, ha="right", fontsize=6.5)
    d0.set_yscale("log")
    d0.set_ylim(100, 500_000)   # 顶部留白:图例与标注放空白区,不压 bar
    d0.set_ylabel("marker-positive barcodes (log scale)")
    d0.set_title("d  Sample-level QC failure: liver_22 is not liver",
                 loc="left", fontweight="bold")
    d0.legend(fontsize=7, frameon=True, facecolor="white", framealpha=0.85,
              edgecolor="#BDC3C7", loc="upper left")
    d0.grid(axis="y", lw=0.3, color="#ECF0F1"); d0.set_axisbelow(True)
    i22 = samp_order.index("liver_22")
    others = [i for i in range(len(samp_order)) if i != i22]
    ratio = np.median(h[others]) / max(h[i22], 1)
    # 文字放在 liver_22 的 Alb bar 正上方空白区,箭头垂直向下,不压 bar、不越界
    d0.annotate(f"Alb+ {ratio:.0f}x lower\nthan other samples",
                xy=(i22 - 0.2, h[i22]), xytext=(i22 - 0.2, 220_000),
                ha="center", fontsize=7, color=BAD, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=BAD, lw=1.0))

    # ---------- e. 肝细胞提取:per-cell 双条件判据 ----------
    e0 = ax[4]
    # 2026-08-30:原硬编码清单(7 个身份基因含 Apoa2/Cyp2e1/Mup3、6 个非肝标志含
    # Clec4f)与 30_exclude / 86_make_consensus 实际使用的清单不一致 —— 导致本面板
    # 算出 71,039,而稿件 Methods 与 panel f / suptitle 用的是 67,145(且下游主
    # 分析共识集 59,121 也源自真实清单)。改为与流水线完全一致的清单,使全图自洽、
    # 读者按稿件 Methods 复现可得到同一数字。
    idg = ["Alb", "Tat", "Cps1", "Apoa1", "Ttr", "Serpina1a"]
    nonhep = ["Ptprc", "Cd3e", "Cd79a", "S100a8", "Pecam1", "Cdh5",
              "Col1a1", "Dcn", "Krt19", "Epcam"]
    vnames = list(A.var_names)
    X = A.X.tocsr()
    idn = np.zeros(A.n_obs, dtype=int)
    for g in idg:
        if g in vnames:
            idn += (np.asarray(X[:, vnames.index(g)].todense()).ravel() > 0).astype(int)
    nhn = np.zeros(A.n_obs, dtype=int)
    for g in nonhep:
        if g in vnames:
            nhn += (np.asarray(X[:, vnames.index(g)].todense()).ravel() > 0).astype(int)
    sel = (idn >= 3) & (nhn <= 1)
    sel_h = sel & (nhn == 0)
    sub = rng.choice(A.n_obs, size=min(A.n_obs, 12000), replace=False)
    e0.scatter(idn[sub] + rng.normal(0, .12, len(sub)),
               nhn[sub] + rng.normal(0, .12, len(sub)),
               s=2.2, color="#BDC3C7", alpha=0.35, edgecolors="none")
    ms = sel[sub]
    e0.scatter(idn[sub][ms] + rng.normal(0, .12, ms.sum()),
               nhn[sub][ms] + rng.normal(0, .12, ms.sum()),
               s=2.6, color=CLP, alpha=0.5, edgecolors="none")
    e0.axvline(2.5, color=ACCENT, lw=1.0, ls="--")
    e0.axhline(1.5, color=ACCENT, lw=1.0, ls="--")
    e0.set_xlabel("hepatocyte identity genes detected (of %d)" % len(idg))
    e0.set_ylabel("non-hepatocyte lineage markers detected")
    e0.set_title(f"e  Per-nucleus dual criterion  (→ {int(sel.sum()):,} hepatocytes)",
                 loc="left", fontweight="bold")
    e0.text(0.98, 0.95, "kept: identity >= 3  AND  lineage <= 1",
            transform=e0.transAxes, ha="right", va="top", fontsize=6.8,
            color=ACCENT,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#BDC3C7", lw=0.6))

    # ---------- f. 每样本细胞数与肝细胞数 ----------
    f0 = ax[5]
    tab = pd.DataFrame({"sample": samp, "is_hep": is_hep})
    agg = tab.groupby("sample")["is_hep"].agg(["sum", "count"])
    agg = agg.loc[[s for _, s, _ in SAMPLES if s in agg.index]]
    xs = np.arange(len(agg))
    f0.bar(xs, agg["count"].values, color="#BDC3C7", label="all nuclei",
           edgecolor="white", linewidth=0.5)
    f0.bar(xs, agg["sum"].values, color=CLP, label="hepatocytes",
           edgecolor="white", linewidth=0.5)
    f0.set_xticks(xs)
    f0.set_xticklabels(agg.index.tolist(), rotation=30, ha="right", fontsize=7)
    f0.set_ylabel("nuclei")
    f0.set_title("f  Nuclei and hepatocytes per sample", loc="left",
                 fontweight="bold")
    f0.legend(fontsize=7, frameon=False)
    for i, (s, c) in enumerate(zip(agg["sum"].values, agg["count"].values)):
        f0.text(i, c, f"{s/c:.0%}", ha="center", va="bottom", fontsize=6.3,
                color=ACCENT)
    f0.grid(axis="y", lw=0.3, color="#ECF0F1"); f0.set_axisbelow(True)

    fig.suptitle("Figure 1  Single-nucleus atlas of the septic mouse liver: "
                 "data overview, quality control, and hepatocyte identification\n"
                 "(GSE275689; 5 samples after excluding one mislabeled sample — "
                 "panel d; %s nuclei after QC; %s criterion-set hepatocytes "
                 "(59,121 consensus, Fig. 2))"
                 % (f"{A.n_obs:,}", f"{int(is_hep.sum()):,}"),
                 fontsize=10.5, fontweight="bold", y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.935])
    for ext in ("pdf", "png"):
        p = os.path.join(FIG_DIR, f"Fig1_overview_QC.{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight")
        log(f"[save] {p}")
    plt.close(fig)

    log()
    log("=" * 88)
    log("完成。panel d 是本文方法学加分项:主动披露样本级数据事故,")
    log("比被审稿人发现强得多。注意 d 用的是**未过滤矩阵**的 marker 阳性")
    log("barcode 数,而不是 QC 后的细胞数 —— liver_22 标准 QC 后有 9,571 细胞,")
    log("看起来完全正常,问题在于这些细胞不是肝细胞。")
    log("=" * 88)


if __name__ == "__main__":
    main()
