#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
70_fig3.py -- Figure 3:肝细胞脂质重编程(方向、幅度与独立验证)

2026-08-29 v4 定稿版面。相对 v3 的三处变更:
  1. **Efflux 程序已按外排目的地一拆二 + 单列胆汁酸合成**,程序数 10 → **12**。
     依据 `scripts/57_efflux_anatomy.py` 的实测核查:Abca1 与 Abcg8 在单细胞层面
     **几乎不共表达**(r = −0.026,同时检出 20.5%,44.5% 的细胞只表达 Abca1),
     且两者目的地完全不同 —— ABCA1 把胆固醇交给**血浆 HDL 池**(留在体内),
     ABCG5/G8 把固醇泵入**胆汁**(离开体内)。合并打分出无意义的均值。
  2. **版面从 2×3 改为左侧纵向森林图 + 右侧 2×3**,共 7 个面板。
     12 条程序纵向排才放得下,且新增胆汁轴与胆固醇出口重定向两个面板。
  3. **深度校正抓出一处翻转**:`Biliary_excretion` 原生 +3.2% → 降采样 −11.7%,
     与独立 bulk 的 −0.711 同号。12 条程序中 **1 条**原生结果被判定为深度伪影。
     (2026-08-30 更正:原写"第二处翻转 / +4.3%→−11.8% / 2 条"来自 16:47 旧版
      log。最终共识集结果见 data/proc/split_depthcheck.csv —— `LDLR_clearance`
     原生 +36.33% → 降采样 +2.88%,**方向未变、并未翻转**;仅 Biliary_excretion
     翻转。稿件已按最终 CSV 改。)

版面
  a  左侧纵向:snRNA 12 条程序效应方向(每样本一个点,CLP ▲ / Sham ●)+ 层间一致性列
  b  独立 bulk 队列验证(GSE311736, n=4 vs 4)关键基因 log2FC + padj ← **唯一统计来源**
  c  深度校正敏感性:原生 vs 降采样 1,500 UMI(12 条,2 条翻转)
  d  FAO 小提琴(Cpt1a / Acadm / Hadha)
  e  脂质新生小提琴(Fasn / Acaca / Elovl6)
  f  胆汁分泌轴小提琴(Cyp7a1 / Cyp8b1 / Abcb4 / Abcc2)
  g  胆固醇出口重定向:Abca1(↑ 至 HDL)vs Abcg8(↓ 至胆汁)

另出两张补充图
  FigS_zonation_negative.pdf  区带阴性结果(如实呈现)
  FigS_efflux_anatomy.pdf     Abca1 / Abcg8 的解剖学证据(由 57 号脚本产出)

依赖数据
  data/proc/fig3_program_effect_v2.csv    snRNA 程序层(每样本点值)
  data/proc/split_depthcheck.csv          深度校正(原生 vs 1,500 UMI)
  data/proc/bulk_HEP_CLP_vs_Sham.csv      独立 bulk 差异表
  data/proc/bulk_program_geneset_v2.csv   独立队列程序层基因集检验
  data/proc/hep_only.h5ad                 snRNA per-cell 表达

用法: python scripts/70_fig3.py
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
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_sepsis import PROC_DIR, FIG_DIR, LIPID_PROGRAMS  # noqa: E402

SHAM = "#6C7A89"     # 蓝灰
CLP = "#E17055"      # 橙红
ACCENT = "#2D3436"
OK = "#27AE60"       # 层间一致
BAD = "#C0392B"      # 层间冲突(方向相反)
NULLC = "#95A5A6"    # 独立队列无可检出改变(阴性,与"冲突"是两回事)
WARN = "#E67E22"     # 翻转(深度伪影)
HDL = "#2980B9"      # HDL 外排侧
BILE = "#8E44AD"     # 胆汁排泄侧

plt.rcParams.update({
    "font.size": 9.5,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "DejaVu Sans",
})

#  programmes whose snRNA and independent-bulk directions disagree -> out of main line
DISCORDANT = {"Cholesterol_synth", "Lipoprotein_assembly", "Lipid_peroxidation"}
#  2 基因程序:基因集检验做不了(n<3),结论靠单基因
TWO_GENE = {"Scavenger_SR_B", "HDL_efflux"}

# panel b 的关键基因(按 log2FC 从低到高排,作图时自动排序)
BULK_GENES = [
    "Cyp7a1", "Vldlr", "Cyp8b1", "Abcc2", "Ldlrap1", "Slc27a4",
    "Abcb4", "Abcg8", "Slc27a2", "Cpt1a", "Acadm", "Hadha", "Ldlr",
    "Scarb1", "Cd36", "Acly", "Acaca", "Elovl6", "Fasn", "Hmox1", "Abca1",
]


def log(*a):
    print(*a, flush=True)


def stars(p):
    return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))


def load(inp=None):
    inp = inp or os.path.join(PROC_DIR, "hep_consensus.h5ad")
    prog = pd.read_csv(os.path.join(PROC_DIR, "fig3_program_effect_v2.csv"),
                       index_col=0)
    # 2026-08-30:Fig3a 一律用**降采样**口径。
    # 稿件 Methods 写了 "All program-level conclusions use depth-corrected
    # values",但原先 Fig3a 读的是原生表 —— 散点、pct_change、separation
    # 全是原生,与正文数字(降采样)不同源。后果不只是数字对不上:
    # 原生 6/12 完全分离 vs 降采样 9/12,正文因此少报了 3 条
    # (FAO / Lipoprotein_assembly / Lipid_peroxidation)。
    # 若只换 separation 而散点仍画原生,读者会看到"散点明明重叠却标着完全
    # 分离"的自相矛盾,故三者必须同源。原生信息由 Fig3c 单独呈现。
    # 生成脚本: scripts/58_program_effect_ds1500.py
    prog_ds = pd.read_csv(
        os.path.join(PROC_DIR, "fig3_program_effect_ds1500.csv"), index_col=0)
    bulk = pd.read_csv(os.path.join(PROC_DIR, "bulk_HEP_CLP_vs_Sham.csv"),
                       index_col=0)
    dep = pd.read_csv(os.path.join(PROC_DIR, "split_depthcheck.csv")) \
            .set_index("program")
    gs = pd.read_csv(os.path.join(PROC_DIR, "bulk_program_geneset_v2.csv")) \
           .set_index("program")
    hep = sc.read_h5ad(inp)
    return prog, prog_ds, bulk, dep, gs, hep


def concordance(prog, dep, bulk, gs):
    """
    每条程序:独立 bulk 上的平均 log2FC / 显著基因数 / 基因集 p / 层间一致性判定

    三种标注要区分开(混为一谈会误导读者):
      ✗  方向冲突  —— snRNA 与 bulk 方向相反,说明其中一层错了
      ∅  无可检出改变 —— bulk 这一层确有统计能力(n=4 vs 4;12 条程序中可
                        计算的 10 条基因集 p 最小为 0.038 = FAO,即该层能把
                        FAO / Lipogenesis / Bile_acid_synth 判到 p<0.05),
                        但该程序所有基因 padj>0.05。这是**阴性结果**,
                        不等于"两层打架",只说明证据不足以断言改变。
                        (2026-08-30:原写"最小 p=0.0143"是旧版数字,已按
                        data/proc/bulk_program_geneset_v2.csv 更正为 0.038。
                        注意 Scavenger_SR_B / HDL_efflux 各仅 2 个基因,基因集
                        检验无法计算 → p 为 NaN,不属于"阴性"这一类。)
      ✓/✓✓ 一致
    """
    bulk_mean, bulk_p, bulk_nsig = {}, {}, {}
    for p, gl in LIPID_PROGRAMS.items():
        hit = [g for g in gl if g in bulk.index]
        v = bulk.loc[hit, "log2FoldChange"].dropna()
        bulk_mean[p] = float(v.mean()) if len(v) else np.nan
        bulk_nsig[p] = int((bulk.loc[hit, "padj"] < 0.05).sum())
        bulk_p[p] = float(gs.loc[p, "p_gene_set"]) if p in gs.index else np.nan
    conc = {}
    for p in prog.index:
        sn = float(dep.loc[p, "ds_pct"]) if p in dep.index else np.nan
        bk = bulk_mean.get(p, np.nan)
        if not np.isfinite(sn) or not np.isfinite(bk):
            conc[p] = ("?", "grey"); continue
        # 先判"无可检出改变":bulk 一个显著基因都没有时,其均值只是噪音,
        # 谈不上"方向相反"。判据见 gene-set-split-audit 技能的 C3。
        if bulk_nsig.get(p, 0) == 0:
            conc[p] = ("∅", NULLC); continue
        if (sn > 0) != (bk > 0):
            conc[p] = ("✗", BAD); continue
        pv = bulk_p.get(p, np.nan)
        conc[p] = ("✓✓", OK) if (np.isfinite(pv) and pv < 0.05) else ("✓", OK)
    return bulk_mean, bulk_p, bulk_nsig, conc


def violins(Ax, X, vnames, grp, gl, ttl, rng):
    data, pos, cols = [], [], []
    for g in gl:
        if g not in set(vnames):
            continue
        i = vnames.index(g)
        x = np.asarray(X[:, i].todense()).ravel()
        for cond, col in [("Sham", SHAM), ("CLP", CLP)]:
            v = x[grp == cond]
            v = rng.choice(v, size=min(len(v), 4000), replace=False)
            data.append(v); pos.append(len(data) - 1); cols.append(col)
    parts = Ax.violinplot(data, positions=pos, widths=0.82, showextrema=False)
    for b, c in zip(parts["bodies"], cols):
        b.set_facecolor(c); b.set_alpha(0.80)
        b.set_edgecolor("white"); b.set_linewidth(0.5)
    for d, p_ in zip(data, pos):
        Ax.scatter([p_], [np.median(d)], s=13, color="white",
                   edgecolors=ACCENT, linewidths=0.8, zorder=5)
    centers = [2 * j + 0.5 for j in range(len(gl))]
    Ax.set_xticks(centers)
    Ax.set_xticklabels(gl, fontsize=7.2)
    Ax.set_xlim(-0.6, 2 * len(gl) - 0.4)
    Ax.set_ylabel("log-normalized expression")
    Ax.set_title(ttl, loc="left", fontweight="bold", fontsize=8.8)
    Ax.grid(axis="y", lw=0.3, color="#ECF0F1", zorder=0)
    Ax.set_axisbelow(True)


def main():
    import argparse as _ap
    _p = _ap.ArgumentParser()
    _p.add_argument("--inp", default=None,
                    help="肝细胞集。默认 B_consensus (59,121);"
                         "传 data/proc/hep_only.h5ad 可复现宽松集 (67,145)")
    _a = _p.parse_args()
    os.makedirs(FIG_DIR, exist_ok=True)
    prog, prog_ds, bulk, dep, gs, hep = load(_a.inp)
    log(f"肝细胞 {hep.n_obs:,}; bulk 差异表 {bulk.shape[0]:,} 基因; "
        f"程序 {len(prog)} 条")
    # concordance 只用 prog.index(程序名)与 dep 的 ds_pct,故传原表即可
    bulk_mean, bulk_p, bulk_nsig, conc = concordance(prog, dep, bulk, gs)

    fig = plt.figure(figsize=(18.6, 9.8))
    gsp = GridSpec(2, 4, figure=fig, width_ratios=[1.20, 1, 1, 1],
                   wspace=0.48, hspace=0.46)
    A = fig.add_subplot(gsp[:, 0])
    B = fig.add_subplot(gsp[0, 1])
    C = fig.add_subplot(gsp[0, 2])
    D = fig.add_subplot(gsp[0, 3])
    E = fig.add_subplot(gsp[1, 1])
    F = fig.add_subplot(gsp[1, 2])
    G = fig.add_subplot(gsp[1, 3])

    # ================= a. 12 条程序(左侧纵向) =================
    # 2026-08-30:改用降采样表,与正文所有程序层数字同源(见 load() 注释)。
    # 排序依据也随之变为 depth-corrected 的 %Δ,与正文"ordered by the
    # CLP-minus-sham difference"的图注一致。
    P = prog_ds.sort_values("pct_change")
    clp_cols = [c for c in P.columns if c.startswith("CLP_")]
    sh_cols = [c for c in P.columns if c.startswith("Sham_")]
    y = np.arange(len(P))
    for i, (name, r) in enumerate(P.iterrows()):
        cv = [r[c] for c in clp_cols if pd.notna(r[c])]
        sv = [r[c] for c in sh_cols if pd.notna(r[c])]
        A.scatter(sv, [i] * len(sv), s=34, color=SHAM, marker="o",
                  zorder=3, edgecolors="white", linewidths=0.5)
        A.scatter(cv, [i] * len(cv), s=44, color=CLP, marker="^",
                  zorder=3, edgecolors="white", linewidths=0.5)
        A.plot([np.mean(sv), np.mean(cv)], [i, i], color=ACCENT,
               lw=1.0, zorder=2, alpha=0.6)
    A.axvline(0, color="#BDC3C7", lw=0.8, zorder=1)
    A.set_yticks(y)
    A.set_yticklabels([n.replace("_", " ") for n in P.index], fontsize=7.6)
    A.set_xlabel("AUCell program score (depth-corrected, 1,500 UMI)")
    A.set_title("a  snRNA: 12 lipid programs, depth-corrected "
                "(n = 3 CLP / 2 Sham)",
                loc="left", fontweight="bold", fontsize=10)
    xmax = A.get_xlim()[1]
    for i, (name, r) in enumerate(P.iterrows()):
        # 2026-08-30:原用 "*",与 Fig3b 的 DESeq2 显著性星号同形,读者会误
        # 把"完全分离"读成统计显著。而这里双尾置换 p 最小只能到 0.100,
        # 12 条程序无一 <0.05,根本不显著。改用菱形 + 脚注写明 p 值下界。
        if str(r["separation"]).startswith("完全分离"):
            A.text(xmax, i, " ◆", va="center", ha="left", fontsize=9,
                   color=CLP)
        sym, col = conc[name]
        A.text(1.02, i, sym, transform=A.get_yaxis_transform(),
               va="center", ha="left", fontsize=9, color=col, fontweight="bold")
    A.legend(handles=[Line2D([], [], marker="o", ls="", color=SHAM, label="Sham (n=2)"),
                      Line2D([], [], marker="^", ls="", color=CLP, label="CLP (n=3)")],
             loc="lower right", frameon=False, fontsize=7)
    # 2026-08-30:星号语义与 Fig3b 冲突的问题(见上方 ◆ 处注释)。
    # 这里必须写死 p 值下界,否则读者会把 ◆ 当成显著性标记。
    A.text(0.0, -0.105,
           "◆ = complete sample separation: all three CLP samples fall "
           "outside the sham range. With n = 3 vs 2 the\n"
           "smallest attainable two-sided permutation p is 0.100, so ◆ is "
           "descriptive only and does NOT denote significance.\n"
           "Right column = depth-corrected snRNA vs independent bulk: "
           "✓✓ concordant & gene-set p<0.05     ✓ concordant     "
           "✗ direction discordant     "
           "∅ no gene with padj<0.05 in bulk (null, not discordant)",
           transform=A.transAxes, fontsize=6.2, color="#7F8C8D")

    # ================= b. 独立 bulk 验证 =================
    rows = [(g, float(bulk.loc[g, "log2FoldChange"]), float(bulk.loc[g, "padj"]))
            for g in BULK_GENES if g in bulk.index]
    dfb = pd.DataFrame(rows, columns=["gene", "l2fc", "padj"]).sort_values("l2fc")
    yb = np.arange(len(dfb))
    B.barh(yb, dfb["l2fc"],
           color=[CLP if v > 0 else SHAM for v in dfb["l2fc"]],
           height=0.66, edgecolor="white", linewidth=0.5)
    B.axvline(0, color=ACCENT, lw=0.9)
    B.set_yticks(yb); B.set_yticklabels(dfb["gene"], fontsize=6.8)
    B.set_xlabel("log2 fold change (CLP vs Sham)")
    # 2026-08-30:原名 "Independent bulk validation"。两队列时间点(8h vs 24h)、
    # 结扎比例(75% vs 30%)、抗生素方案均不同,严格讲这不是同一效应的"验证",
    # 而是不同条件下的一致性检验。稿件已统一改为平行证据线框架,此处同步。
    B.set_title("b  Independent bulk cohort (GSE311736, n = 4 vs 4)",
                loc="left", fontweight="bold", fontsize=10)
    for i, (g, v, p) in enumerate(zip(dfb["gene"], dfb["l2fc"], dfb["padj"])):
        off = 0.10 if v >= 0 else -0.10
        B.text(v + off, i, stars(p), va="center",
               ha="left" if v >= 0 else "right", fontsize=6.2, color=ACCENT)
    B.text(0.02, -0.22, "* padj<0.05   ** <0.01   *** <0.001   ns = not significant",
           transform=B.transAxes, fontsize=6.3, color="#7F8C8D")

    # ================= c. 深度校正敏感性 =================
    xs = dep["native_pct"].values.astype(float)
    ys = dep["ds_pct"].values.astype(float)
    labs = dep.index.tolist()
    for i in range(len(labs)):
        flip = (xs[i] > 0) != (ys[i] > 0)
        C.scatter([xs[i]], [ys[i]], s=46 if flip else 30,
                  color=WARN if flip else CLP,
                  edgecolors=ACCENT if flip else "white",
                  linewidths=0.9 if flip else 0.5,
                  zorder=4 if flip else 3)
        if flip:
            C.annotate(labs[i].replace("_", " "), (xs[i], ys[i]),
                       textcoords="offset points", xytext=(6, -2),
                       fontsize=6.0, color=WARN, fontweight="bold")
    lim = [min(xs.min(), ys.min()) - 15, max(xs.max(), ys.max()) + 25]
    C.plot(lim, lim, color=ACCENT, lw=0.9, ls="--", alpha=0.7)
    C.plot([0, 0], lim, color="#BDC3C7", lw=0.7)
    C.plot(lim, [0, 0], color="#BDC3C7", lw=0.7)
    C.set_xlim(lim); C.set_ylim(lim)
    C.set_xlabel("Native depth: % change")
    C.set_ylabel("Downsampled 1,500 UMI: % change")
    C.set_title("c  Depth-correction sensitivity (12 programs)",
                loc="left", fontweight="bold")
    C.text(0.03, 0.05,
           f"concordant: {int(((xs<0)==(ys<0)).sum())}/{len(xs)}\n"
           f"flipped (depth artifact): {int(((xs<0)!=(ys<0)).sum())}",
           transform=C.transAxes, fontsize=7.0, color=ACCENT,
           bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#BDC3C7", lw=0.6))
    C.text(0.03, 0.78,
           "median UMI/nucleus (hepatocyte consensus set)\n"
           "  Sham: 1,812 (liver_14) / 9,773 (liver_18)\n"
           "  CLP : 5,012 (liver_02) / 9,367 (liver_06) / 7,275 (liver_10)",
           transform=C.transAxes, fontsize=6.0, color="#7F8C8D", va="top")

    # ================= d/e/f/g 小提琴 =================
    vnames = list(hep.var_names)
    grp = hep.obs["group"].values
    X = hep.X.tocsr()
    rng = np.random.default_rng(0)
    violins(D, X, vnames, grp, ["Cpt1a", "Acadm", "Hadha"],
            "d  Fatty acid oxidation (snRNA)", rng)
    violins(E, X, vnames, grp, ["Fasn", "Acaca", "Elovl6"],
            "e  Lipogenesis (snRNA)", rng)
    violins(F, X, vnames, grp, ["Cyp7a1", "Cyp8b1", "Abcb4", "Abcc2"],
            "f  Biliary secretory axis (snRNA)", rng)
    violins(G, X, vnames, grp, ["Abca1", "Abcg8"],
            "g  Cholesterol exit rewiring (snRNA)", rng)
    G.text(0.5, 0.94, "to HDL ↑   |   to bile ↓", transform=G.transAxes,
           ha="center", va="top", fontsize=7.5, color=ACCENT, fontweight="bold")

    fig.legend(handles=[Line2D([], [], marker="s", ls="", color=SHAM,
                               markersize=8, label="Sham"),
                        Line2D([], [], marker="s", ls="", color=CLP,
                               markersize=8, label="CLP")],
               loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.004))

    fig.suptitle("Figure 3  Hepatocyte lipid reprogramming in sepsis: exogenous import "
                 "and biliary excretion shut down,\n"
                 "while lipogenesis and ABCA1-mediated HDL efflux are induced",
                 fontsize=11.5, fontweight="bold", y=0.985)
    fig.subplots_adjust(left=0.055, right=0.975, top=0.905, bottom=0.055)

    for ext in ("pdf", "png"):
        p = os.path.join(FIG_DIR, f"Fig3_lipid_reprogramming.{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight")
        log(f"[save] {p}")
    plt.close(fig)

    # ================= FigS: 区带阴性结果 =================
    fig2, axs = plt.subplots(1, 2, figsize=(9.2, 3.7))
    z = pd.read_csv(os.path.join(PROC_DIR, "aucell_main.csv"), index_col=0)
    zi = z.loc[hep.obs_names, "zonation_index"]
    g, s = hep.obs["group"].values, hep.obs["sample"].values
    A1 = axs[0]
    A1.hist([zi[g == "Sham"], zi[g == "CLP"]], bins=60, density=True,
            color=[SHAM, CLP], alpha=0.72, label=["Sham", "CLP"])
    A1.set_xlabel("zonation index  (AUCell central − portal)")
    A1.set_ylabel("density")
    A1.set_title("Zonation index distribution", loc="left", fontweight="bold")
    A1.legend(frameon=False)
    # 2026-08-30:增大 ylim 顶部留白,避免左上角文字框与直方图峰值柱子重叠
    A1.set_ylim(0, A1.get_ylim()[1] * 1.40)
    A1.text(0.02, 0.92,
            f"SD ratio CLP/Sham = {zi[g=='CLP'].std()/zi[g=='Sham'].std():.3f}\n"
            f"IQR ratio = {np.subtract(*np.percentile(zi[g=='CLP'],[75,25]))/np.subtract(*np.percentile(zi[g=='Sham'],[75,25])):.3f}\n"
            "(>1 = wider, NOT collapse)",
            transform=A1.transAxes, fontsize=7.5, va="top", color=ACCENT,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#BDC3C7", lw=0.6))
    A2 = axs[1]
    per = pd.DataFrame({"sample": s, "group": g, "zi": zi.values}) \
        .groupby(["group", "sample"])["zi"].mean().reset_index()
    for cond, col in [("Sham", SHAM), ("CLP", CLP)]:
        d = per[per["group"] == cond]
        A2.scatter([cond] * len(d), d["zi"], s=58, color=col,
                   edgecolors="white", linewidths=0.8, zorder=3)
    A2.set_ylabel("mean zonation index")
    A2.set_title("Per-sample means — no group difference", loc="left",
                 fontweight="bold")
    A2.grid(axis="y", lw=0.3, color="#ECF0F1"); A2.set_axisbelow(True)
    for cond in ["Sham", "CLP"]:
        A2.plot([cond], [per[per["group"] == cond]["zi"].mean()],
                marker="_", ms=22, color=ACCENT, zorder=4)
    fig2.suptitle("Supplementary: no detectable collapse of zonation in CLP "
                  "(not detected, not absent)",
                  fontsize=10, fontweight="bold", y=0.97)
    fig2.subplots_adjust(left=0.08, right=0.97, top=0.84, bottom=0.14, wspace=0.28)
    for ext in ("pdf", "png"):
        p = os.path.join(FIG_DIR, f"FigS_zonation_negative.{ext}")
        fig2.savefig(p, dpi=300, bbox_inches="tight")
        log(f"[save] {p}")
    plt.close(fig2)

    # ---------------- 控制台归账 ----------------
    log()
    log("=" * 92)
    log("出图完成。12 条程序层间一致性归账:")
    log("=" * 92)
    log(f"  {'program':<22s}{'snRNA(校正)':>12s}{'bulk mean':>11s}"
        f"{'显著基因':>9s}{'geneset p':>11s}   判定")
    for p in prog.index:
        sn = float(dep.loc[p, "ds_pct"]) if p in dep.index else np.nan
        bk = bulk_mean.get(p, np.nan)
        pv = bulk_p.get(p, np.nan)
        ps = f"{pv:>11.3f}" if np.isfinite(pv) else f"{'n<3':>11s}"
        log(f"  {p:<22s}{sn:>12.1f}{bk:>11.3f}{bulk_nsig.get(p,0):>9d}{ps}"
            f"   {conc[p][0]}")
    log()
    log("  主轴 1:外源进口关闭(LDLR 通路 + FATP) + FAO 受抑 → 脂质新生代偿激活")
    log("  主轴 2:胆汁分泌轴关闭(胆汁酸合成 + 毛细胆管转运) ↔ ABCA1 介导的 HDL")
    log("          外排极度激活 → 胆固醇出口重定向,对应脓毒症相关胆汁淤积")
    log(f"  冲突项(方向相反)与阴性项(独立队列无可检出改变)均不进主线:")
    for p in prog.index:
        if conc[p][0] == "✗":
            log(f"    ✗ {p}: snRNA {float(dep.loc[p,'ds_pct']):+.1f}% vs "
                f"bulk {bulk_mean.get(p, float('nan')):+.3f}  —— 方向相反")
        elif conc[p][0] == "∅":
            log(f"    ∅ {p}: snRNA {float(dep.loc[p,'ds_pct']):+.1f}% vs "
                f"bulk {bulk_mean.get(p, float('nan')):+.3f}  —— "
                f"独立队列 {bulk_nsig.get(p,0)} 个基因 padj<0.05,未能复现")
    log("=" * 92)


if __name__ == "__main__":
    main()
