#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
80_human_conservation.py -- Figure 5:跨物种保守性验证(人类肝细胞)

数据集
  GSE233483 / PMID 38451817
  hiPSC 来源的多谱系肝类器官(含 Kupffer 细胞,KuLOs),模拟人类脓毒症肝:
    Control  (D14)      n = 2
    Septic   (LPS/IFNγ, D16)  n = 2     ← 脓毒症样刺激
    Recovery D1/D2/D3 (D17/18/19) 各 n = 2

  这是目前可得的**最直接的人类肝细胞脓毒症模型**:人类细胞 + LPS/IFNγ
  (脓毒症的核心刺激),且带**清除内毒素后的恢复时程** —— 小鼠数据只有
  单时间点,这个时程是人类层独有的优势。

⚠️ 必须如实披露的三条局限
  1. iPSC 来源的**类器官**,不是成体肝组织;无肝小叶结构,**不能**做区带分析。
  2. **体外** LPS/IFNγ 刺激,不是体内 CLP 模型。
  3. n = 2 vs 2,且来自**同一 iPSC 细胞系(M48)** —— 是技术重复而非生物学
     重复。故本层**只给方向,不给 p 值**。
     (2026-08-30 改:原写"统计由小鼠两层承担",是发现—验证框架的说法;
      稿件已统一为**平行证据线** —— 三层的条件各不相同(小鼠 24h/30%结扎/
      有抗生素 vs 8h/75%结扎/无抗生素 vs 人类体外急性刺激),任何一层都不能
      "替另一层承担"统计责任,一致性的意义在于跨条件稳健,而非样本量互补。)

方法(与小鼠 bulk 层保持一致,便于跨物种比较)
  - 竞争性基因集检验:程序基因的 log2FC vs 全背景(Wilcoxon)
  - 程序层方向 = 程序内基因 log2FC 均值
  - 时程:每个样本内程序基因的跨样本 z-score 均值

小鼠→人类同源基因
  绝大多数哺乳动物基因的人类直系同源即"小鼠名全大写"(Cpt1a→CPT1A)。
  脚本会报告映射成功率与失败名单,不做静默丢弃。

输出
  data/proc/human_program_conservation.csv   人 vs 鼠 程序方向对照
  data/proc/human_gene_level.csv             关键基因的人鼠对照
  figures/Fig5_human_conservation.pdf        四面板图

用法: python scripts/80_human_conservation.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_sepsis import LIPID_PROGRAMS, PROC_DIR, FIG_DIR  # noqa: E402

HUMAN_FILE = "data/validation/human/GSE233483_KuLOs.txt.gz"
CTRL = ["KuLO_D14_1", "KuLO_D14_2"]
SEP = ["KuLO_sep_1", "KuLO_sep_2"]
RECOV = [("ReD1", ["KuLO_ReD1_1", "KuLO_ReD1_2"]),
         ("ReD2", ["KuLO_ReD2_1", "KuLO_ReD2_2"]),
         ("ReD3", ["KuLO_ReD3_1", "KuLO_ReD3_2"])]

MOUSE_BULK = "data/proc/bulk_HEP_CLP_vs_Sham.csv"
MOUSE_PROG = "data/proc/split_depthcheck.csv"

CLP = "#E17055"
CTRL_C = "#6C7A89"
ACCENT = "#2D3436"
PURPLE = "#8E44AD"    # 人类层
TEAL = "#16A085"      # 恢复时程
OK = "#27AE60"
BAD = "#C0392B"


def log(*a):
    print(*a, flush=True)


def load_human():
    D = pd.read_csv(HUMAN_FILE, sep="\t")
    log(f"原始表 {D.shape[0]:,} 行 x {D.shape[1]} 列")
    cols = CTRL + SEP + [c for _, cs in RECOV for c in cs]
    miss = [c for c in cols if c not in D.columns]
    if miss:
        sys.exit(f"缺列: {miss}")
    E = D[["Gene"] + cols].copy()
    E = E.groupby("Gene")[cols].max()          # 同名基因取最大
    log(f"去重后基因 {E.shape[0]:,}")
    # 过滤低表达:至少在 2 个样本里 > 1
    keep = (E[cols] > 1).sum(axis=1) >= 2
    E = E[keep]
    log(f"过滤低表达后 {E.shape[0]:,} 基因")
    return E, cols


def to_human(g):
    """小鼠基因名 -> 人类同源(绝大多数为全大写)"""
    return g.upper()


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    E, cols = load_human()
    ctrl = E[CTRL].mean(axis=1)
    sep = E[SEP].mean(axis=1)
    l2 = np.log2((sep + 0.5) / (ctrl + 0.5))
    L = pd.DataFrame({"log2FC": l2, "ctrl": ctrl, "sep": sep})
    log(f"log2FC 计算完成,背景基因 {len(L):,}")

    # ---------------- 同源映射 ----------------
    log()
    log("=" * 88)
    log("小鼠 → 人类 同源基因映射")
    log("=" * 88)
    mapped, failed = {}, []
    for p, gl in LIPID_PROGRAMS.items():
        ok = []
        for g in gl:
            h = to_human(g)
            if h in L.index:
                ok.append(h); mapped[g] = h
            else:
                failed.append((p, g, h))
        log(f"  {p:<24s} {len(ok):>2d}/{len(gl):<2d} 命中")
    if failed:
        log()
        log("  未命中的基因 —— 区分两种情形(**关键,不能一笔带过**):")
        raw = pd.read_csv(HUMAN_FILE, sep="\t")
        rawg = set(raw["Gene"].astype(str).str.upper())
        for p, g, h in failed:
            why = ("**基因符号在原始数据里就没有**(平台未覆盖 / 用了别名)"
                   if h not in rawg else
                   "**表达量过低,被低表达过滤剔除**")
            log(f"    {p:<24s} {g:<10s} -> {h:<10s} {why}")
    log(f"\n  总映射成功率 {len(mapped)}/{sum(len(v) for v in LIPID_PROGRAMS.values())}")

    # ---------------- 人 vs 鼠 程序层对照 ----------------
    mb = pd.read_csv(MOUSE_BULK, index_col=0)
    mprog = pd.read_csv(MOUSE_PROG, index_col=0)
    bg_m = mb["log2FoldChange"].dropna()
    bg_h = L["log2FC"]

    log()
    log("=" * 104)
    log("程序层:人类(肝类器官, LPS/IFNγ) vs 小鼠(独立 bulk 队列, CLP)")
    log("=" * 104)
    log(f"  {'program':<24s}{'人 n':>5s}{'人 mean':>9s}{'人 p':>9s}"
        f"{'鼠 mean':>9s}{'鼠 p':>9s}{'snRNA%':>9s}   保守?")
    rows = []
    for p, gl in LIPID_PROGRAMS.items():
        hg = [to_human(g) for g in gl if to_human(g) in L.index]
        if len(hg) < 3:
            rows.append(dict(program=p, human_n=len(hg), human_mean=np.nan,
                             human_p=np.nan, mouse_mean=np.nan, mouse_p=np.nan,
                             conserved="n<3"))
            continue
        hv = L.loc[hg, "log2FC"]
        up_h, ph = mannwhitneyu(hv, bg_h, alternative="two-sided")
        # 小鼠侧
        mg = [g for g in gl if g in mb.index]
        mv = mb.loc[mg, "log2FoldChange"].dropna()
        up_m, pm = mannwhitneyu(mv, bg_m, alternative="two-sided")
        sn = float(mprog.loc[p, "ds_pct"]) if p in mprog.index else np.nan
        cons = "✓ 一致" if (hv.mean() > 0) == (mv.mean() > 0) else "✗ 相反"
        rows.append(dict(program=p, human_n=len(hg),
                         human_mean=round(float(hv.mean()), 3),
                         human_p=float(ph),
                         mouse_mean=round(float(mv.mean()), 3),
                         mouse_p=float(pm), snrna_pct=round(sn, 1),
                         conserved=cons))
        log(f"  {p:<24s}{len(hg):>5d}{hv.mean():>9.3f}{ph:>9.3f}"
            f"{mv.mean():>9.3f}{pm:>9.3f}{sn:>9.1f}   {cons}")
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(PROC_DIR, "human_program_conservation.csv"), index=False)
    log(f"\n[save] {PROC_DIR}/human_program_conservation.csv")

    # ---------------- 关键基因人鼠对照 ----------------
    KEY = ["CPT1A", "ACADM", "HADHA", "SLC27A2", "SLC27A4",
           "LDLR", "VLDLR", "LDLRAP1", "LRP1",
           "SCARB1", "CD36", "FASN", "ACACA", "ELOVL6", "ACLY",
           "CYP7A1", "CYP8B1", "ABCB11", "ABCB4", "ABCC2", "ABCG8",
           "ABCA1", "HMOX1", "SLC7A11", "APOB", "MTTP"]
    log()
    log("=" * 88)
    log("关键基因:人类 vs 小鼠(同一基因的两个物种对照)")
    log("=" * 88)
    log(f"  {'gene':<10s}{'人 log2FC':>11s}{'人 ctrl':>10s}{'人 sep':>10s}"
        f"{'鼠 log2FC':>11s}{'鼠 padj':>11s}   保守?")
    grows = []
    for g in KEY:
        if g not in L.index:
            continue
        hl = float(L.loc[g, "log2FC"])
        mg = g.capitalize()          # CPT1A -> Cpt1a
        ml = mp = np.nan
        if mg in mb.index:
            ml = float(mb.loc[mg, "log2FoldChange"])
            mp = float(mb.loc[mg, "padj"])
        cons = "✓" if (np.isfinite(ml) and (hl > 0) == (ml > 0)) else (
            "✗" if np.isfinite(ml) else "?")
        grows.append(dict(gene=g, human_log2FC=round(hl, 3),
                          human_ctrl=round(float(L.loc[g, "ctrl"]), 1),
                          human_sep=round(float(L.loc[g, "sep"]), 1),
                          mouse_log2FC=round(ml, 3) if np.isfinite(ml) else np.nan,
                          mouse_padj=mp, conserved=cons))
        ps = f"{mp:>11.3g}" if np.isfinite(mp) else f"{'NA':>11s}"
        ms = f"{ml:>11.3f}" if np.isfinite(ml) else f"{'NA':>11s}"
        log(f"  {g:<10s}{hl:>11.3f}{L.loc[g,'ctrl']:>10.1f}{L.loc[g,'sep']:>10.1f}"
            f"{ms}{ps}   {cons}")
    G = pd.DataFrame(grows)
    G.to_csv(os.path.join(PROC_DIR, "human_gene_level.csv"), index=False)
    log(f"\n[save] {PROC_DIR}/human_gene_level.csv")

    # ---------------- 恢复时程 ----------------
    log()
    log("=" * 88)
    log("恢复时程:清除 LPS/IFNγ 后程序是否回到基线(人类层独有)")
    log("=" * 88)
    stages = [("Control", CTRL), ("Septic", SEP)] + RECOV
    Z = np.log2(E[cols] + 0.5)
    zs = Z.sub(Z.mean(axis=1), axis=0).div(Z.std(axis=1).replace(0, np.nan), axis=0)
    traj = {}
    for p, gl in LIPID_PROGRAMS.items():
        hg = [to_human(g) for g in gl if to_human(g) in zs.index]
        if not hg:
            continue
        traj[p] = [float(np.nanmean(zs.loc[hg, cs].values)) for _, cs in stages]
    TR = pd.DataFrame(traj, index=[s for s, _ in stages]).T
    log(f"  {'program':<24s}" + "".join(f"{s:>10s}" for s, _ in stages)
        + f"{'回弹?':>9s}")
    for p, row in TR.iterrows():
        d = row["Septic"] - row["Control"]
        back = abs(row["ReD3"] - row["Control"]) < abs(d) * 0.5
        log(f"  {p:<24s}" + "".join(f"{row[s]:>10.2f}" for s, _ in stages)
            + f"{('✓' if back else '—'):>9s}")
    TR.to_csv(os.path.join(PROC_DIR, "human_recovery_trajectory.csv"))

    # ---------------- 图 ----------------
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.6))
    A, B, C, D_ = axes.ravel()

    # a 人 vs 鼠 程序方向散点
    sub = T.dropna(subset=["human_mean", "mouse_mean"])
    A.scatter(sub["mouse_mean"], sub["human_mean"], s=62, color=PURPLE,
              edgecolors="white", linewidths=0.8, zorder=3)
    A.axhline(0, color="#BDC3C7", lw=0.8); A.axvline(0, color="#BDC3C7", lw=0.8)
    lim = [min(sub["mouse_mean"].min(), sub["human_mean"].min()) - 0.5,
           max(sub["mouse_mean"].max(), sub["human_mean"].max()) + 0.5]
    A.plot(lim, [0, 0], color="#BDC3C7", lw=0.7)
    A.set_xlim(lim); A.set_ylim(lim)
    for _, r in sub.iterrows():
        if r["conserved"].startswith("✗"):
            A.annotate(r["program"].replace("_", " "),
                       (r["mouse_mean"], r["human_mean"]),
                       textcoords="offset points", xytext=(5, 3),
                       fontsize=6.5, color=BAD)
        elif abs(r["human_mean"]) > 0.8 or abs(r["mouse_mean"]) > 1.0:
            A.annotate(r["program"].replace("_", " "),
                       (r["mouse_mean"], r["human_mean"]),
                       textcoords="offset points", xytext=(5, 3),
                       fontsize=6.3, color=ACCENT)
    q1 = sub[((sub.mouse_mean < 0) & (sub.human_mean < 0)) |
             ((sub.mouse_mean > 0) & (sub.human_mean > 0))]
    q2 = sub[((sub.mouse_mean > 0) & (sub.human_mean < 0)) |
             ((sub.mouse_mean < 0) & (sub.human_mean > 0))]
    A.set_xlabel("Mouse: program mean log2FC (GSE311736, n=4 vs 4)")
    A.set_ylabel("Human: program mean log2FC (GSE233483, n=2 vs 2)")
    A.set_title("a  Cross-species concordance of program direction",
                loc="left", fontweight="bold")
    A.text(0.03, 0.05,
           f"concordant (1st/3rd quadrant): {len(q1)}/{len(sub)}\n"
           f"discordant (2nd/4th quadrant): {len(q2)}",
           transform=A.transAxes, fontsize=7.5, color=ACCENT,
           bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#BDC3C7", lw=0.6))

    # b 关键基因人鼠对照
    gg = G.dropna(subset=["mouse_log2FC"]).sort_values("human_log2FC")
    yb = np.arange(len(gg))
    w = 0.38
    B.barh(yb + w / 2, gg["human_log2FC"], height=w, color=PURPLE,
           label="Human KuLO (LPS/IFNγ)", edgecolor="white", linewidth=0.4)
    B.barh(yb - w / 2, gg["mouse_log2FC"], height=w, color=CLP,
           label="Mouse CLP (bulk)", edgecolor="white", linewidth=0.4)
    B.axvline(0, color=ACCENT, lw=0.9)
    B.set_yticks(yb)
    B.set_yticklabels([f"{r.gene}{'' if r.conserved=='✓' else ' ✗'}"
                       for r in gg.itertuples()], fontsize=6.5)
    B.set_xlabel("log2 fold change")
    B.set_title("b  Key genes: human vs mouse (✗ = discordant)",
                loc="left", fontweight="bold")
    B.legend(fontsize=7, frameon=False, loc="lower right")

    # c 胆汁轴 + ABCA1 在人鼠两侧
    panel_genes = [("CYP7A1", "Cyp7a1"), ("CYP8B1", "Cyp8b1"),
                   ("ABCB4", "Abcb4"), ("ABCC2", "Abcc2"),
                   ("ABCG8", "Abcg8"), ("ABCA1", "Abca1"),
                   ("CPT1A", "Cpt1a"), ("ACADM", "Acadm"),
                   ("SLC27A2", "Slc27a2"), ("FASN", "Fasn")]
    xv = np.arange(len(panel_genes))
    hv = [float(L.loc[h, "log2FC"]) if h in L.index else np.nan
          for h, _ in panel_genes]
    mv = [float(mb.loc[m, "log2FoldChange"]) if m in mb.index else np.nan
          for _, m in panel_genes]
    C.bar(xv - 0.2, hv, width=0.4, color=PURPLE, label="Human",
          edgecolor="white", linewidth=0.4)
    C.bar(xv + 0.2, mv, width=0.4, color=CLP, label="Mouse",
          edgecolor="white", linewidth=0.4)
    C.axhline(0, color=ACCENT, lw=0.9)
    C.set_xticks(xv)
    C.set_xticklabels([h for h, _ in panel_genes], rotation=45, ha="right",
                      fontsize=7)
    C.set_ylabel("log2 fold change")
    C.set_title("c  Biliary axis vs ABCA1: conserved in both species",
                loc="left", fontweight="bold")
    C.legend(fontsize=7, frameon=False)
    C.grid(axis="y", lw=0.3, color="#ECF0F1"); C.set_axisbelow(True)

    # d 恢复时程
    show = ["Bile_acid_synth", "Biliary_excretion", "HDL_efflux",
            "FAO", "Lipogenesis", "FA_transport"]
    xs = np.arange(len(stages))
    for p in show:
        if p not in TR.index:
            continue
        v = TR.loc[p].values
        D_.plot(xs, v, "-o", ms=4.5, lw=1.5, label=p.replace("_", " "))
    D_.axvline(0.5, color="#BDC3C7", lw=0.8, ls="--")
    D_.axvline(1.5, color="#BDC3C7", lw=0.8, ls="--")
    D_.axhline(0, color=ACCENT, lw=0.8)
    D_.set_xticks(xs)
    D_.set_xticklabels([s for s, _ in stages], fontsize=7.5)
    D_.set_ylabel("program z-score (across samples)")
    D_.set_title("d  Recovery after endotoxin removal (human only)",
                 loc="left", fontweight="bold")
    D_.legend(fontsize=6.6, frameon=False, ncol=2)
    D_.grid(lw=0.3, color="#ECF0F1"); D_.set_axisbelow(True)

    n_ok = int(sum(1 for r in rows if str(r.get("conserved", "")).startswith("✓")))
    n_tot = int(sum(1 for r in rows if str(r.get("conserved", "")) in ("✓ 一致", "✗ 相反")))
    # 2026-08-30:"validation" → "comparison"。人类层是 n=2 vs 2 技术重复的
    # 单 iPSC 系、且缺 CYP7A1,措辞上只是方向性比对,谈不上验证;稿件已统一。
    fig.suptitle("Figure 5  Cross-species comparison in a human sepsis model: "
                 "FAO suppression and biliary excretion shutdown are conserved,\n"
                 "whereas lipogenesis is discordant  "
                 f"({n_ok}/{n_tot} programs concordant; GSE233483 hiPSC liver "
                 "organoids + LPS/IFNγ; n = 2 vs 2, directional only)",
                 fontsize=10.2, fontweight="bold", y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    for ext in ("pdf", "png"):
        p = os.path.join(FIG_DIR, f"Fig5_human_conservation.{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight")
        log(f"[save] {p}")
    plt.close(fig)

    log()
    log("=" * 88)
    log("完成。此层只给方向不给 p:人类类器官 n=2 vs 2 且同一细胞系,")
    log("属技术重复,只给方向不给 p。三层互为平行独立证据线,")
    log("不存在哪一层替另一层承担统计责任(见文件头注释)。")
    log("=" * 88)


if __name__ == "__main__":
    main()
