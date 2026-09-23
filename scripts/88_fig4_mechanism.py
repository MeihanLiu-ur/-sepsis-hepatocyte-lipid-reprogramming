#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
88_fig4_mechanism.py -- Figure 4: 机制衔接(上游转录因子 → 双轴重编程)

定位
  Fig3 是"效应层"(12 条程序朝哪个方向走、幅度多大),Fig4 是"机制层"
  (谁在调控这些改变)。**纯细胞内,不碰细胞间互作**(CellChat 已正式移除)。

四个面板(于哥 2026-08-29 拍板"4 面板全做")
  a  PPARα 轴受抑   —— Ppara 本身 bulk −2.45 (padj 8.8e-14),其 FAO / 脂肪酸
                       转运靶基因整组下调;Cd36/Lpl 反向(炎症诱导,解耦)
  b  FXR/LXR 轴     —— Nr1h4(FXR)/Nr1h3(LXRα)/Rxra 核受体整体关闭 → 胆汁轴
                       靶基因(Cyp7a1/Cyp8b1/Abcc2/Abcb4/Slc10a1)整条关闭;
                       **Abca1 反向 +2.999** —— LXRα 关但 ABCA1 开,说明其诱导
                       是炎症驱动,不依赖 LXR(胆固醇出口重定向的机制基础)
  c  SREBP 轴       —— Srebf1 本身 bulk 不显著(−0.614, ns),但 SREBP-1c 靶
                       基因(Fasn/Acaca/Acly/Elovl6)全线上调 → 脂质新生激活
                       非 SREBP-1c 转录上调驱动(转录后/炎症驱动);
                       SREBP-2 靶(胆固醇合成)方向混乱 → 对应 Fig3 的 ✗ 判定
  d  TF–程序相关性热图 —— 12 个上游 TF × 12 条脂质程序的 per-cell Spearman ρ

关键发现(写进正文/图注,Fig3 完全没有的新增量)
  1. 上游核受体开关整体关闭:Ppara/Nr1h4/Nr1h3/Rxra/Nr1i3 在 bulk 层全部
     显著下调(padj 均 <1e-4),共同伴侣 RXRα 也关 → 两条主轴的上游"总开关"
  2. ABCA1 的炎症解耦:LXRα 关(−0.89)但 ABCA1 开(+3.00),说明 HDL 外排的
     激活不依赖 LXR,而是炎症信号驱动 —— 这正是"胆汁轴关、HDL 外排开"
     两条相反方向能同时成立的机制
  3. 脂质新生激活非 SREBP-1c 转录依赖:Srebf1 ns 但靶基因全上调

依赖数据
  data/proc/hep_consensus.h5ad   主分析集 59,121 肝细胞(log-normalized X)
  data/proc/aucell_v2.csv        12 程序 per-cell AUCell(共识集)
  data/proc/bulk_HEP_CLP_vs_Sham.csv   独立 bulk(DESeq2, n=4 vs 4)← 唯一统计来源

用法: python scripts/88_fig4_mechanism.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_sepsis import PROC_DIR, FIG_DIR, LIPID_PROGRAMS  # noqa: E402

SHAM = "#6C7A89"     # 蓝灰
CLP = "#E17055"      # 橙红
ACCENT = "#2D3436"   # 深灰(核受体/TF)
OK = "#27AE60"
BAD = "#C0392B"
NULLC = "#95A5A6"
HDL = "#2980B9"      # HDL 外排侧
BILE = "#8E44AD"     # 胆汁排泄侧
INFL = "#D68910"     # 炎症诱导(解耦)

plt.rcParams.update({
    "font.size": 9.5,
    "axes.labelsize": 10,
    "axes.titlesize": 10.5,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "DejaVu Sans",
})

# --------------------------------------------------------------------------
# 先验基因清单(不按结果定义,避免循环论证)
# --------------------------------------------------------------------------
# 面板 a:PPARα 轴
PPARA_TF = ["Ppara"]
PPARA_FAO = ["Cpt1a", "Cpt2", "Acadm", "Acadl", "Acadvl", "Hadha", "Hadhb",
             "Crat", "Decr2", "Acox1", "Ehhadh", "Hsd17b4", "Scp2"]
PPARA_FAT = ["Slc27a1", "Slc27a2", "Slc27a4", "Slc27a5", "Fabp1"]
PPARA_INFL = ["Cd36", "Lpl"]          # PPARα 靶但方向反向(炎症诱导)

# 面板 b:FXR / LXR 轴
NR_B = ["Nr1h4", "Nr1h3", "Rxra"]     # FXR / LXRα / RXRα(共同伴侣)
FXR_TARGET = ["Cyp7a1", "Cyp8b1", "Cyp27a1", "Nr0b2", "Abcb11", "Abcc2",
              "Abcb4", "Atp8b1", "Slc10a1", "Slco1a1"]
LXR_HDL = ["Abca1"]                   # LXR 靶 → 血浆 HDL(反向)
LXR_BILE = ["Abcg5", "Abcg8", "Npc1"]  # LXR 靶 → 胆汁

# 面板 c:SREBP 轴
SREBF_TF = ["Srebf1", "Srebf2"]       # SREBP-1c / SREBP-2
SREBF1C_TARGET = ["Fasn", "Acaca", "Acly", "Elovl6", "Scd1", "Me1", "Gpam",
                  "Acacb"]
SREBF2_TARGET = ["Hmgcs1", "Hmgcr", "Sqle", "Mvd", "Fdps", "Lss", "Dhcr7",
                 "Dhcr24", "Msmo1", "Cyp51", "Ldlr", "Pcsk9"]

# 面板 d:TF × 程序相关性热图的 TF 清单(检出率 >30% 者,见探查)
CORR_TF = ["Ppara", "Nr1h4", "Nr1h3", "Rxra", "Srebf1", "Srebf2", "Mlxipl",
           "Hnf4a", "Ppargc1a", "Nfe2l2", "Nr1i3", "Thrb"]


def log(*a):
    print(*a, flush=True)


def stars(p):
    if p is None or not np.isfinite(p):
        return "—"
    return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))


# --------------------------------------------------------------------------
def build_rows(bulk, groups):
    """
    把 (label → 基因列表) 的 groups 转成出图用 DataFrame。
    每个基因带:log2FC / padj / 颜色 / 是否 TF。
    """
    rows = []
    for grp_name, (genes, color, is_tf) in groups.items():
        for g in genes:
            if g in bulk.index:
                r = bulk.loc[g]
                rows.append(dict(gene=g, group=grp_name,
                                 l2fc=float(r["log2FoldChange"]),
                                 padj=float(r["padj"]),
                                 baseMean=float(r["baseMean"]),
                                 color=color, is_tf=is_tf))
            else:
                rows.append(dict(gene=g, group=grp_name, l2fc=np.nan,
                                 padj=np.nan, baseMean=np.nan,
                                 color=color, is_tf=is_tf))
                log(f"    [缺] {g} 不在 bulk 差异表,跳过")
    return pd.DataFrame(rows)


def draw_grouped_bars(ax, df, title):
    """
    横向条形图:log2FC 瀑布,TF 加粗深色,靶基因按功能组着色,
    每行尾标 padj 星号,行标签里 TF 加粗。
    """
    # 排序:TF 置顶,其余按 l2fc 升序
    tf_rows = df[df["is_tf"]].sort_values("l2fc")
    other = df[~df["is_tf"]].sort_values("l2fc")
    dd = pd.concat([tf_rows, other]).reset_index(drop=True)

    y = np.arange(len(dd))
    vals = dd["l2fc"].values
    # 缺失基因(l2fc NaN)画 0 并标灰
    plot_vals = np.where(np.isfinite(vals), vals, 0.0)
    colors = []
    for _, r in dd.iterrows():
        if not np.isfinite(r["l2fc"]):
            colors.append("#E5E8E8")
        elif r["is_tf"]:
            colors.append(ACCENT)
        else:
            colors.append(r["color"])
    ax.barh(y, plot_vals, height=0.68, color=colors,
            edgecolor="white", linewidth=0.5)
    # TF 加粗边框
    for i, r in dd.iterrows():
        if r["is_tf"] and np.isfinite(r["l2fc"]):
            ax.barh(i, r["l2fc"], height=0.68, color=ACCENT,
                    edgecolor="#F1C40F", linewidth=1.3)
    ax.axvline(0, color=ACCENT, lw=0.9)
    ax.set_yticks(y)
    labels = []
    for _, r in dd.iterrows():
        nm = r["gene"]
        labels.append(nm)
    ax.set_yticklabels(labels, fontsize=7.2)
    # TF 标签加粗
    for i, r in dd.iterrows():
        if r["is_tf"]:
            ax.get_yticklabels()[i].set_fontweight("bold")
            ax.get_yticklabels()[i].set_color(ACCENT)
    ax.set_xlabel("log2 fold change (CLP vs Sham), bulk n = 4 vs 4")
    ax.set_title(title, loc="left", fontweight="bold", fontsize=9.0)
    # 星号
    for i, r in dd.iterrows():
        if np.isfinite(r["l2fc"]):
            v = r["l2fc"]
            off = 0.12 if v >= 0 else -0.12
            ax.text(v + off, i, stars(r["padj"]), va="center",
                    ha="left" if v >= 0 else "right", fontsize=6.2, color=ACCENT)
    ax.grid(axis="x", lw=0.3, color="#ECF0F1")
    ax.set_axisbelow(True)
    return dd


def main():
    import argparse as _ap
    _p = _ap.ArgumentParser()
    _p.add_argument("--inp", default=os.path.join(PROC_DIR, "hep_consensus.h5ad"),
                    help="肝细胞集,默认 B_consensus (59,121)")
    _a = _p.parse_args()
    os.makedirs(FIG_DIR, exist_ok=True)

    bulk = pd.read_csv(os.path.join(PROC_DIR, "bulk_HEP_CLP_vs_Sham.csv"),
                       index_col=0)
    hep = sc.read_h5ad(_a.inp)
    auc = pd.read_csv(os.path.join(PROC_DIR, "aucell_v2.csv"), index_col=0)
    log(f"肝细胞 {hep.n_obs:,}  bulk 差异表 {bulk.shape[0]:,} 基因  "
        f"AUCell {auc.shape[1]} 程序")

    # ---------------- 面板 a/b/c 数据 ----------------
    ga = build_rows(bulk, {
        "PPARα (TF)":            (PPARA_TF, ACCENT, True),
        "PPARα targets (FAO)":   (PPARA_FAO, SHAM, False),
        "PPARα targets (FATP)":  (PPARA_FAT, HDL, False),
        "inflammatory (uncoupled)": (PPARA_INFL, INFL, False),
    })
    gb = build_rows(bulk, {
        "Nuclear receptors":     (NR_B, ACCENT, True),
        "FXR targets (biliary)": (FXR_TARGET, BILE, False),
        "LXR target → HDL":      (LXR_HDL, HDL, False),
        "LXR targets → bile":    (LXR_BILE, SHAM, False),
    })
    gc = build_rows(bulk, {
        "SREBP (TF)":            (SREBF_TF, ACCENT, True),
        "SREBP-1c targets (lipogenic)": (SREBF1C_TARGET, CLP, False),
        "SREBP-2 targets (cholesterol)": (SREBF2_TARGET, SHAM, False),
    })
    # 缓存明细(追溯用)
    pd.concat([ga, gb, gc], keys=["PPARA", "FXRLXR", "SREBP"]) \
        .reset_index(level=0).rename(columns={"level_0": "panel"}) \
        .to_csv(os.path.join(PROC_DIR, "fig4_tf_targets.csv"), index=False)

    # ---------------- 面板 d:TF × 程序 Spearman ----------------
    vnames = list(hep.var_names)
    X = hep.X.tocsr()
    g2i = {g: i for i, g in enumerate(vnames)}
    tf_present = [g for g in CORR_TF if g in g2i]
    missing_tf = [g for g in CORR_TF if g not in g2i]
    if missing_tf:
        log(f"    [缺] 相关性 TF 未匹配: {missing_tf}")

    # TF 表达(稀疏 → 稠密,只取需要的列)
    tf_expr = np.zeros((hep.n_obs, len(tf_present)), dtype=np.float32)
    for j, g in enumerate(tf_present):
        tf_expr[:, j] = np.asarray(X[:, g2i[g]].todense()).ravel()

    # 程序 AUCell 对齐 obs_names
    auc = auc.loc[hep.obs_names]
    prog_names = [p for p in LIPID_PROGRAMS if p in auc.columns]
    auc_mat = auc[prog_names].values.astype(np.float64)

    # per-cell Spearman = rank 变换后 Pearson
    M = np.hstack([tf_expr, auc_mat]).astype(np.float64)
    Mr = np.apply_along_axis(rankdata, 0, M)   # 每列 rank
    R = np.corrcoef(Mr, rowvar=False)          # (nTF+nProg) x (nTF+nProg)
    corr = R[:len(tf_present), len(tf_present):]  # TF x program

    corr_df = pd.DataFrame(corr, index=tf_present, columns=prog_names)
    corr_df.to_csv(os.path.join(PROC_DIR, "fig4_tf_program_corr.csv"))
    log("TF × 程序 Spearman ρ (per-cell, rank 后 Pearson):")
    log(corr_df.round(2).to_string())

    # ---------------- 出图 ----------------
    fig = plt.figure(figsize=(17.6, 11.2))
    gsp = GridSpec(2, 2, figure=fig, hspace=0.52, wspace=0.30)
    A = fig.add_subplot(gsp[0, 0])
    B = fig.add_subplot(gsp[0, 1])
    C = fig.add_subplot(gsp[1, 0])
    D = fig.add_subplot(gsp[1, 1])

    da = draw_grouped_bars(A, ga,
        "a  PPARα axis suppressed: Ppara −2.45 (padj 8.8e-14), "
        "FAO & FATP targets down")
    db = draw_grouped_bars(B, gb,
        "b  FXR/LXR axis: biliary route shut down; Abca1 dissociated "
        "from LXRα")
    dc = draw_grouped_bars(C, gc,
        "c  SREBP axis: lipogenesis up despite Srebf1 −0.61 (ns) "
        "(post-translational activation);\n"
        "   cholesterol targets discordant")

    # 图例移到 Figure 底部横排(不再占面板 b 右下角,避免与解耦标注重叠)
    fig.legend(handles=[
        Line2D([], [], marker="s", ls="", color=ACCENT, markersize=9,
               label="nuclear receptor / TF"),
        Line2D([], [], marker="s", ls="", color=BILE, markersize=9,
               label="FXR targets (biliary)"),
        Line2D([], [], marker="s", ls="", color=HDL, markersize=9,
               label="LXR target → HDL / FATP"),
        Line2D([], [], marker="s", ls="", color=SHAM, markersize=9,
               label="FAO / cholesterol / LXR→bile"),
        Line2D([], [], marker="s", ls="", color=CLP, markersize=9,
               label="SREBP-1c targets (lipogenic)"),
        Line2D([], [], marker="s", ls="", color=INFL, markersize=9,
               label="inflammatory (uncoupled)"),
    ], loc="lower center", ncol=3, frameon=False, fontsize=6.8,
       bbox_to_anchor=(0.5, 0.005), columnspacing=1.2)

    # 星号图例
    B.text(0.02, -0.10, "* padj<0.05   ** <0.01   *** <0.001   ns = not significant",
           transform=B.transAxes, fontsize=6.3, color="#7F8C8D")

    # 面板 b 的解耦标注(放左上角空白区:顶部 TF 为向左的负 bar,左侧 x<0.55 为空)
    # 2026-08-30:原措辞 "NOT LXR-driven → inflammatory" 与稿件冲突 —— 稿件已把
    # 炎症驱动降级为"possibly inflammatory ... remains to be tested"。图上也必须
    # 同步降级为"dissociated(解离)"+"driver not identified, not tested here",
    # 否则读者会把未检验的假设当成已证实的结论。配色由红(BAD=冲突)改琥珀
    # (INFL=待检验),视觉语气与措辞一致。
    B.text(0.02, 0.97,
           "Abca1 +3.00 (padj 3e-109) while LXRα −0.89 (padj 1.1e-4):\n"
           "HDL-efflux induction is dissociated from LXRα transcript "
           "abundance\n"
           "(LXR-independent; driver not identified, not tested here)",
           transform=B.transAxes, ha="left", va="top", fontsize=6.4,
           color="#7E5109", fontweight="bold",
           bbox=dict(boxstyle="round,pad=0.35", fc="#FEF9E7", ec=INFL, lw=0.7))

    # 面板 c:关键信息已并入标题,不再放文本框(避免与 bar 重叠)

    # 面板 d 热图
    im = D.imshow(corr, cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
    D.set_xticks(np.arange(len(prog_names)))
    D.set_xticklabels([p.replace("_", "\n") for p in prog_names],
                      fontsize=6.3, rotation=0)
    D.set_yticks(np.arange(len(tf_present)))
    D.set_yticklabels(tf_present, fontsize=7.4)
    # 数值
    for i in range(len(tf_present)):
        for j in range(len(prog_names)):
            v = corr[i, j]
            D.text(j, i, f"{v:+.2f}", ha="center", va="center",
                   fontsize=5.6,
                   color="white" if abs(v) > 0.35 else ACCENT)
    cb = plt.colorbar(im, ax=D, fraction=0.046, pad=0.03)
    cb.set_label("per-cell Spearman ρ (TF × program AUCell)", fontsize=7.5)
    cb.ax.tick_params(labelsize=6.8)
    D.set_title("d  TF–program correlation (snRNA, per-cell rank ρ)",
                loc="left", fontweight="bold", fontsize=9.0)

    fig.suptitle(
        "Figure 4  Upstream transcription-factor rewiring links the two "
        "reprogramming axes\n"
        "Nuclear receptors (Ppara / Nr1h4-FXR / Nr1h3-LXRα / Rxra) are globally "
        "suppressed, while ABCA1 induction is dissociated from LXRα transcript "
        "abundance",
        fontsize=11.5, fontweight="bold", y=0.99)
    fig.subplots_adjust(left=0.055, right=0.975, top=0.915, bottom=0.09)

    for ext in ("pdf", "png"):
        p = os.path.join(FIG_DIR, f"Fig4_mechanism.{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight")
        log(f"[save] {p}")
    plt.close(fig)

    # ---------------- 控制台归账 ----------------
    log()
    log("=" * 92)
    log("Fig4 机制衔接完成。上游 TF 在 bulk 层的归账(唯一统计来源):")
    log("=" * 92)
    for nm, g in [("Ppara (PPARα)", "Ppara"), ("Nr1h4 (FXR)", "Nr1h4"),
                  ("Nr1h3 (LXRα)", "Nr1h3"), ("Rxra (RXRα)", "Rxra"),
                  ("Srebf1 (SREBP-1c)", "Srebf1"), ("Srebf2 (SREBP-2)", "Srebf2"),
                  ("Nr1i3 (CAR)", "Nr1i3"), ("Hnf4a", "Hnf4a")]:
        if g in bulk.index:
            r = bulk.loc[g]
            log(f"  {nm:<20s} log2FC {float(r['log2FoldChange']):+.3f}  "
                f"padj {float(r['padj']):.3g}  {stars(float(r['padj']))}")
    log("=" * 92)


if __name__ == "__main__":
    main()
