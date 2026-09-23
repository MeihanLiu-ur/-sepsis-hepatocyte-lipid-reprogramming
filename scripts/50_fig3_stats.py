#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
50_fig3_stats.py -- Figure 3 主面板数据:每条脂质程序的方向与幅度

按 2026-08-29 拍板的**方案 A** 产出:
  主线 = 肝细胞脂质程序重编程;区带降为次要维度。

输出
  1. fig3_program_effect.csv   每条程序:组均值、相对变化、**每个样本的点值**、
                               完全分离判定、样本层置换 p
  2. fig3_key_genes.csv        关键基因:每样本均值、log2FC、方向
  3. 控制台报告 + 一段"统计能力"的诚实说明

统计口径(重要)
  本数据 Sham 组 n=2、CLP 组 n=3。把 5 个样本分成 3 vs 2 只有 C(5,2)=10 种分法,
  **样本层置换检验能取到的最小 p 值 = 1/10 = 0.10**,数学上不可能 <0.05。
  因此本脚本同时报告:
    - 每样本点值(主展示, 让读者自己看分离度)
    - 完全分离判定(比 p 值更有信息量)
    - 置换 p(带最小值说明)
    - 细胞层效应量(仅作描述, **不是推断检验** —— 不得当 p 值用)
  在补第三个 Sham 样本之前,本文的组间比较只能作为**描述性/假说生成**证据。

用法: python scripts/50_fig3_stats.py [--ds 1500]
"""
from __future__ import annotations

import os
import sys
import itertools
import argparse

import numpy as np
import pandas as pd
import scanpy as sc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_sepsis import LIPID_PROGRAMS, PROC_DIR  # noqa: E402

# Fig3 小提琴图用:每条程序挑 2-3 个读者认得的关键基因
KEY_GENES = {
    "FAO":                   ["Cpt1a", "Acadm", "Hadha"],
    "Peroxisome":            ["Acox1", "Scp2", "Hsd17b4"],
    "Lipogenesis":           ["Fasn", "Acaca", "Scd1"],
    "Cholesterol_synth":     ["Hmgcr", "Hmgcs1", "Sqle"],
    "Lipoprotein_assembly":  ["Apoa2", "Apob", "Mttp"],
    "Uptake":                ["Ldlr", "Scarb1", "Cd36"],
    "Efflux":                ["Abca1", "Abcg8", "Apoe"],
    "Lipid_peroxidation":    ["Gpx4", "Acsl4", "Hmox1"],
}


def log(*a):
    print(f"[{''.join(a[:0])}]", *a, flush=True) if False else print(*a, flush=True)


def perm_p(vals, groups, grp_a="CLP", grp_b="Sham"):
    """
    样本层单侧置换检验:真实差值在全部 C(n, k) 种分组中的分位。
    同时返回该设计下**理论最小 p 值**。
    """
    v = np.asarray(vals, dtype=float)
    g = np.asarray(groups)
    n = len(v)
    k = int((g == grp_a).sum())
    obs = v[g == grp_a].mean() - v[g == grp_b].mean()
    stat = []
    for comb in itertools.combinations(range(n), k):
        m = np.zeros(n, dtype=bool)
        m[list(comb)] = True
        stat.append(v[m].mean() - v[~m].mean())
    stat = np.asarray(stat)
    p_two = float((np.abs(stat) >= abs(obs) - 1e-12).mean())
    p_min = 1.0 / len(stat)
    return obs, p_two, p_min, len(stat)


def separation(clp_vals, sham_vals):
    """完全分离:CLP 的样本值全部落在 Sham 的同一侧(不重叠)"""
    lo_s, hi_s = min(sham_vals), max(sham_vals)
    lo_c, hi_c = min(clp_vals), max(clp_vals)
    if hi_c < lo_s:
        return "完全分离(CLP 全低于 Sham)"
    if lo_c > hi_s:
        return "完全分离(CLP 全高于 Sham)"
    return "有重叠"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default=os.path.join(PROC_DIR, "hep_only.h5ad"))
    ap.add_argument("--ds", type=int, default=1500)
    args = ap.parse_args()

    hep = sc.read_h5ad(args.inp)
    # 直接复用 40 号脚本的打分结果(避免重复计算)
    main_csv = os.path.join(PROC_DIR, "aucell_main.csv")
    if not os.path.exists(main_csv):
        sys.exit(f"缺少 {main_csv},请先跑 40_zonation_aucell.py")
    D = pd.read_csv(main_csv, index_col=0)
    assert len(D) == hep.n_obs, "打分行数与肝细胞数不符"

    grp = hep.obs["group"].values
    samp = hep.obs["sample"].values
    progs = list(LIPID_PROGRAMS)

    print("=" * 78)
    print("Figure 3 主面板:8 条脂质程序的方向与幅度(AUCell,原生深度)")
    print("=" * 78)

    rows = []
    for p in progs:
        v = D[p].values
        per = pd.Series(v).groupby(samp).mean()
        clp = [per[s] for s in per.index if s in
               set(samp[grp == "CLP"])]
        sham = [per[s] for s in per.index if s in
                set(samp[grp == "Sham"])]
        obs, pperm, pmin, nperm = perm_p(per.values,
                                         ["CLP" if s in set(samp[grp == "CLP"])
                                          else "Sham" for s in per.index])
        pct = (np.mean(clp) - np.mean(sham)) / np.mean(sham) * 100
        # 细胞层 Wilcoxon 效应量(描述用, 非推断)
        try:
            from scipy.stats import mannwhitneyu
            u, pw = mannwhitneyu(v[grp == "CLP"], v[grp == "Sham"],
                                  alternative="two-sided")
            n1, n2 = int((grp == "CLP").sum()), int((grp == "Sham").sum())
            auc = u / (n1 * n2)
        except Exception:
            auc, pw = np.nan, np.nan
        rows.append(dict(program=p,
                         sham_mean=np.mean(sham), clp_mean=np.mean(clp),
                         pct_change=pct,
                         **{f"CLP_{i+1}": round(x, 4) for i, x in enumerate(clp)},
                         **{f"Sham_{i+1}": round(x, 4) for i, x in enumerate(sham)},
                         separation=separation(clp, sham),
                         perm_p=round(pperm, 3), perm_min_p=round(pmin, 3),
                         cell_AUC=round(auc, 3)))
    T = pd.DataFrame(rows).set_index("program")
    for line in T.round(4).to_string().split("\n"):
        print("  " + line)
    T.round(5).to_csv(os.path.join(PROC_DIR, "fig3_program_effect.csv"))

    print()
    print("=" * 78)
    print("关键基因:每样本均值与方向(Fig3 小提琴图数据)")
    print("=" * 78)
    v = list(hep.var_names)
    X = hep.X.tocsr()
    grows = []
    for prog, gl in KEY_GENES.items():
        for g in gl:
            if g not in set(v):
                continue
            i = v.index(g)
            x = np.asarray(X[:, i].todense()).ravel()
            per = pd.Series(x).groupby(samp).mean()
            clp = [per[s] for s in per.index if s in set(samp[grp == "CLP"])]
            sham = [per[s] for s in per.index if s in set(samp[grp == "Sham"])]
            l2 = np.log2((np.mean(clp) + 1e-9) / (np.mean(sham) + 1e-9))
            grows.append(dict(program=prog, gene=g,
                              sham=round(np.mean(sham), 3),
                              clp=round(np.mean(clp), 3),
                              log2FC=round(l2, 3),
                              direction="↑" if l2 > 0.1 else
                                        ("↓" if l2 < -0.1 else "≈"),
                              sep=separation(clp, sham)))
    G = pd.DataFrame(grows)
    for line in G.to_string(index=False).split("\n"):
        print("  " + line)
    G.to_csv(os.path.join(PROC_DIR, "fig3_key_genes.csv"), index=False)

    print()
    print("=" * 78)
    print("统计能力的诚实说明(写进 Methods 与 Discussion)")
    print("=" * 78)
    nclp = len(set(samp[grp == "CLP"]))
    nsham = len(set(samp[grp == "Sham"]))
    ncomb = len(list(itertools.combinations(range(nclp + nsham), nclp)))
    print(f"  当前设计: CLP n={nclp}  vs  Sham n={nsham}")
    print(f"  样本层置换检验的全部可能分组数 = C({nclp+nsham},{nclp}) = {ncomb}")
    print(f"  → **最小可达 p 值 = 1/{ncomb} = {1/ncomb:.3f}**,数学上不可能 < 0.05")
    print()
    print("  因此组间比较只能作为**描述性 / 假说生成**证据,须:")
    print("    (a) 主图展示**每个样本的点值**而非只有均值,让读者看分离度;")
    print("    (b) 报告**完全分离**判定(部分程序已达到);")
    print("    (c) 细胞层统计量只当效应量,**不得**当作 p 值使用;")
    print("    (d) 或按规划 §5.0b 选项 C 补第三个 Sham 样本(届时 C(6,3)=20,最小 p=0.05)。")
    print()
    print(f"  降采样敏感性分析另见:lipid_zone_group_aucell_ds{args.ds}.csv")
    print("=" * 78)


if __name__ == "__main__":
    main()
