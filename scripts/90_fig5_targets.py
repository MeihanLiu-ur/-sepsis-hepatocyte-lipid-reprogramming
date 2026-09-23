#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
90_fig5_targets.py -- Figure 5 补充:候选靶点表(程序级干预靶提名)

定位
  蓝图 Fig5 处 🟡 待补项。基于双轴重编程,提名**程序级**干预靶(而非原
  "区带特异"靶)。每个靶点给出:① 本研究证据(bulk log2FC/padj)
  ② 跨物种保守性 ③ 干预策略。

五个候选靶(覆盖双轴核心程序)
  1. PPARa(核受体)   —— 主轴1 下调侧总开关,激动剂恢复脂肪酸氧化
  2. FXR(核受体)     —— 主轴2 胆汁轴总开关,激动剂恢复胆汁分泌(抗胆汁淤积)
  3. ABCA1(执行基因) —— HDL 外排极度激活,增强促 LPS 清除(保护性)
  4. FASN(执行基因)  —— 脂质新生代偿激活,但人鼠方向相反 → 干预**暂缓**
  5. CD36(执行基因)  —— 清道夫受体上调,但跨物种不保守 → 干预**暂缓**
     (2026-08-30:原写"抑制需谨慎"暗示方向已定;稿件已降级为 deferred,
      因两靶的人鼠方向分歧未解决,任何靶向都需时程研究。图与文同步。)

证据来源
  data/proc/bulk_HEP_CLP_vs_Sham.csv      独立 bulk(n=4 vs 4)← 唯一统计来源
  data/proc/human_program_conservation.csv 人类层保守性(80 号产物)

用法: python scripts/90_fig5_targets.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_sepsis import PROC_DIR, FIG_DIR  # noqa: E402

CLP = "#E17055"
ACCENT = "#2D3436"
NR = "#8E44AD"      # 核受体层
EFF = "#2980B9"     # 执行基因层
BAD = "#C0392B"


def log(*a):
    print(*a, flush=True)


def fmt_l2fc(v, p):
    """log2FC + padj 简洁标注"""
    st = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    return f"{v:+.2f} ({st})"


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    bulk = pd.read_csv(os.path.join(PROC_DIR, "bulk_HEP_CLP_vs_Sham.csv"),
                       index_col=0)
    human = pd.read_csv(os.path.join(PROC_DIR, "human_program_conservation.csv"))

    # 跨物种保守性(从人类层产物取,失败则保守标注)
    cons = {}
    if not human.empty:
        for _, r in human.iterrows():
            cons[r["program"]] = r.get("conserved", "?")

    def ev(g):
        """从 bulk 取 (log2FC, padj) 标注"""
        if g in bulk.index:
            return fmt_l2fc(float(bulk.loc[g, "log2FoldChange"]),
                            float(bulk.loc[g, "padj"]))
        return "n/a"

    # 每个靶:(靶点, 层级, 重编程程序, 证据, 跨物种, 干预策略)
    targets = [
        ("PPARa",  "nuclear receptor", "FAO / FA-transport (down)",
         f"Ppara {ev('Ppara')}",
         "FAO conserved" if cons.get("FAO", "?").startswith("✓") else "FAO: ?",
         "agonist (restore oxidation) — but worsened survival in CLP [16]"),
        ("FXR",    "nuclear receptor", "biliary axis (shut down)",
         f"Nr1h4 {ev('Nr1h4')}",
         "biliary conserved" if cons.get("Biliary_excretion", "?").startswith("✓") else "biliary: ?",
         "agonist (anti-cholestatic)"),
        ("ABCA1",  "effector", "HDL efflux (induced)",
         f"Abca1 {ev('Abca1')}",
         "only program not rebounding",
         "enhance (LPS clearance)"),
        ("FASN",   "effector", "lipogenesis (induced)",
         f"Fasn {ev('Fasn')}",
         "discordant (human vs mouse)",
         "deferred (time-resolved study needed)"),
        ("CD36",   "effector", "scavenger SR-B (induced)",
         f"Cd36 {ev('Cd36')}",
         "not conserved",
         "deferred (direction not conserved)"),
    ]
    cols = ["Candidate target", "Level", "Rewired program",
            "Evidence (bulk n=4 vs 4)", "Cross-species", "Intervention"]

    log("候选靶点表:")
    for t in targets:
        log(f"  {t[0]:<8s} {t[1]:<16s} {t[2]:<28s} {t[3]:<24s} {t[4]:<30s} {t[5]}")

    # 出图(横向表格,指定列宽避免最后一列超出)
    fig, ax = plt.subplots(figsize=(16.0, 4.1))
    ax.axis("off")
    cell = [ [t[0], t[1], t[2], t[3], t[4], t[5]] for t in targets ]
    tab = ax.table(cellText=cell, colLabels=cols, cellLoc="left",
                   colLoc="left", loc="center",
                   colWidths=[0.10, 0.13, 0.21, 0.17, 0.19, 0.20])
    tab.auto_set_font_size(False)
    tab.set_fontsize(9)
    tab.scale(1, 1.75)
    # 表头
    for j in range(len(cols)):
        c = tab[0, j]
        c.set_facecolor(ACCENT)
        c.set_text_props(color="white", fontweight="bold")
    # 行着色:核受体淡紫,执行基因淡蓝;证据/策略列突出
    for i in range(1, len(targets) + 1):
        lvl = targets[i - 1][1]
        base = "#F4F0FA" if lvl == "nuclear receptor" else "#EBF4FB"
        for j in range(len(cols)):
            tab[i, j].set_facecolor(base)
    # 加粗第一列(靶点名)
    for i in range(1, len(targets) + 1):
        tab[i, 0].set_text_props(fontweight="bold", color=ACCENT)

    ax.set_title(
        "Figure S3  Candidate program-level targets from the "
        "two-axis reprogramming",
        loc="left", fontweight="bold", fontsize=11.5)
    ax.text(0.0, -0.06,
            "Evidence = independent purified-hepatocyte bulk RNA-seq "
            "(GSE311736, n = 4 vs 4, DESeq2); *, **, *** = padj < 0.05/0.01/0.001; "
            "ns = not significant.\n"
            "PPARa/FXR are the two suppressed upstream nuclear receptors; "
            "ABCA1 induction is dissociated from LXRα transcript abundance "
            "(Fig 4); the upstream driver is not identified and was not "
            "tested here.",
            transform=ax.transAxes, fontsize=7.8, color="#5F5E5A", va="top")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = os.path.join(FIG_DIR, f"Fig5_candidate_targets.{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight")
        log(f"[save] {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
