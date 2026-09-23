#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
57_efflux_anatomy.py -- Efflux 程序的解剖学核查:这些基因的外排目的地是同一个吗?

为什么做这一步
  `Efflux` 程序现在混着方向相反的信号:
    Abca1  +3.00 (padj 3e-109, 极强上调)
    Abcg8  -1.04 (padj 9.5e-3,  显著下调)
  如果只是"噪音",那拆不拆无所谓;但如果**外排目的地本来就不同**,
  那么方向相反恰恰是**真实的差异化调控**,拆开才能讲出机制。

关键生物学事实(先验, 不是从本项目结果反推)
  ABCA1  : 把游离胆固醇/磷脂外排给**贫脂的 apoA-I**,生成 nascent HDL。
           这是胆固醇逆向转运(RCT)的**第一步、限速步**。
           表达:近乎广谱(除红细胞外几乎所有细胞)。
           亚细胞定位:质膜 + 晚期内体。
  ABCG1  : 把胆固醇外排给**成熟的 HDL 颗粒**(不是贫脂 apoA-I)。广谱表达。
  ABCG5/G8: 形成**专性异二聚体**,定位于**毛细胆管膜(canalicular membrane)**,
           把**胆固醇与植物固醇泵入胆汁** → 随粪便排出体外。
           表达:**肝与肠高度特异**。突变致谷固醇血症(sitosterolemia)。
  → 前两者是"把胆固醇交给血浆脂蛋白池(**留在体内**)",
    后者是"把固醇排进胆汁(**离开体内**)"。这是两个完全不同的出口。

本脚本要回答的三个问题
  Q1 在肝细胞里,这些基因的**表达量与检出率**是否可比?(会不会有基因根本没测到)
  Q2 它们沿**门静脉-中央静脉轴(zonation index)**的分布是否不同?
     —— 这是"组织精细分布"的核心证据。胆汁侧转运体应偏向靠近毛细胆管的区域,
        而 ABCA1 若偏 zone1 则说明它服务的是经血液侧入肝的胆固醇。
  Q3 它们在**同一批细胞**里共表达吗?若 Abca1 与 Abcg8 互不共表达,
    说明它们活跃在不同的肝细胞亚群,合并打分在细胞层面就不成立。

输出
  data/proc/efflux_anatomy.csv      每基因的检出率 / 均值 / 区带梯度 / 相关系数
  data/proc/efflux_zonation_bin.csv 每基因 × 10 个区带 bin 的平均表达
  figures/FigS_efflux_anatomy.pdf   区带分布折线 + 共表达散点

用法: python scripts/57_efflux_anatomy.py
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_sepsis import PROC_DIR, FIG_DIR  # noqa: E402

SHAM = "#6C7A89"
CLP = "#E17055"
ACCENT = "#2D3436"
C_HDL = "#2980B9"     # 血浆 HDL 外排侧
C_BILE = "#8E44AD"    # 胆汁排泄侧
C_REG = "#7F8C8D"     # 调控层 / 参照

# 候选基因,按功能目的地分组(先验)
GROUPS = {
    "HDL_efflux": ["Abca1", "Abcg1", "Pltp"],
    "Biliary_sterol": ["Abcg5", "Abcg8", "Abcb11", "Abcb4", "Abcc2", "Atp8b1"],
    "Bile_acid_synth": ["Cyp7a1", "Cyp8b1", "Cyp27a1"],
    "Nuclear_receptor": ["Nr1h3", "Nr1h2", "Nr1h4"],
    "Zone_ref": ["Glul", "Cyp2e1", "Ass1", "Cps1", "Pck1"],
    "Apo_ref": ["Apoe", "Apob", "Apoa1"],
}
COLOR = {"HDL_efflux": C_HDL, "Biliary_sterol": C_BILE,
         "Bile_acid_synth": C_REG, "Nuclear_receptor": C_REG,
         "Zone_ref": "#16A085", "Apo_ref": "#95A5A6"}

N_BIN = 10


def log(*a):
    print(*a, flush=True)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    hep = sc.read_h5ad(os.path.join(PROC_DIR, "hep_only.h5ad"))
    genes = list(hep.var_names)
    X = hep.X.tocsr()
    grp = hep.obs["group"].values
    samp = hep.obs["sample"].values
    log(f"肝细胞 {hep.n_obs:,}  基因 {hep.n_vars:,}")

    # zonation index 放在旧文件里(v3 的 aucell_v2.csv 只重算了 10 个程序)
    zi = pd.read_csv(os.path.join(PROC_DIR, "aucell_main.csv"), index_col=0)
    zi = zi.loc[hep.obs_names, "zonation_index"].values
    bulk = pd.read_csv(os.path.join(PROC_DIR, "bulk_HEP_CLP_vs_Sham.csv"),
                       index_col=0)
    log(f"zonation index 载入 {len(zi):,}")

    # ---------- Q1/Q2: 检出率 + 区带梯度 ----------
    # 区带 bin:按 zonation index 分位数切, Sham 与 CLP 各自切(保证每组都有细胞)
    bins = np.full(len(zi), -1, dtype=int)
    for g in ["Sham", "CLP"]:
        m = grp == g
        q = np.quantile(zi[m], np.linspace(0, 1, N_BIN + 1))
        q[0] -= 1e-9
        bins[m] = np.digitize(zi[m], q[1:-1])

    rows, bin_rows = [], []
    for grp_name, gl in GROUPS.items():
        for g in gl:
            if g not in set(genes):
                rows.append(dict(group=grp_name, gene=g, detect=np.nan,
                                 mean=np.nan, rho=np.nan,
                                 log2FC=np.nan, padj=np.nan,
                                 note="snRNA 未匹配"))
                continue
            i = genes.index(g)
            x = np.asarray(X[:, i].todense()).ravel()
            det = float((x > 0).mean())
            rho = float(np.corrcoef(zi, x)[0, 1]) if x.std() > 0 else np.nan
            # 每样本均值 -> 组均值 -> log2FC
            per = pd.Series(x).groupby(samp).mean()
            clp_set = set(samp[grp == "CLP"])
            clp = [per[s] for s in per.index if s in clp_set]
            sham = [per[s] for s in per.index if s not in clp_set]
            l2 = float(np.log2((np.mean(clp) + 1e-9) / (np.mean(sham) + 1e-9)))
            if g in bulk.index:
                b_l2 = float(bulk.loc[g, "log2FoldChange"])
                b_p = float(bulk.loc[g, "padj"])
                bm = float(bulk.loc[g, "baseMean"])
            else:
                b_l2 = b_p = bm = np.nan
            rows.append(dict(group=grp_name, gene=g, detect=round(det, 4),
                             mean_expr=round(float(x.mean()), 4),
                             rho_zonation=round(rho, 4),
                             snRNA_log2FC=round(l2, 3),
                             bulk_baseMean=round(bm, 1),
                             bulk_log2FC=round(b_l2, 3) if np.isfinite(b_l2) else np.nan,
                             bulk_padj=b_p,
                             note=""))
            # 区带 bin 表达谱(按 CLP / Sham 分开)
            for g2 in ["Sham", "CLP"]:
                for b in range(N_BIN):
                    m = (bins == b) & (grp == g2)
                    if m.sum() < 50:
                        continue
                    bin_rows.append(dict(gene=g, group=g2, bin=b,
                                         n=int(m.sum()),
                                         mean=float(x[m].mean()),
                                         zc=float(np.median(zi[m]))))
    T = pd.DataFrame(rows)
    B = pd.DataFrame(bin_rows)

    log()
    log("=" * 104)
    log("Q1/Q2  Efflux 相关基因的检出率、区带梯度与两层差异方向")
    log("=" * 104)
    hdr = (f"{'group':<18s}{'gene':<10s}{'检出率':>8s}{'ρ(zone)':>9s}"
           f"{'snRNA':>8s}{'bulk':>8s}{'bulk padj':>11s}{'baseMean':>10s}")
    log(hdr)
    for _, r in T.iterrows():
        bp = f"{r['bulk_padj']:>11.3g}" if np.isfinite(r["bulk_padj"]) else f"{'NA':>11s}"
        bm = f"{r['bulk_baseMean']:>10.0f}" if np.isfinite(r["bulk_baseMean"]) else f"{'NA':>10s}"
        bl = f"{r['bulk_log2FC']:>8.3f}" if np.isfinite(r["bulk_log2FC"]) else f"{'NA':>8s}"
        sn = f"{r['snRNA_log2FC']:>8.3f}" if np.isfinite(r["snRNA_log2FC"]) else f"{'NA':>8s}"
        dt = f"{r['detect']:>8.1%}" if np.isfinite(r["detect"]) else f"{'NA':>8s}"
        rh = f"{r['rho_zonation']:>9.3f}" if np.isfinite(r["rho_zonation"]) else f"{'NA':>9s}"
        log(f"{r['group']:<18s}{r['gene']:<10s}{dt}{rh}{sn}{bl}{bp}{bm}"
            f"  {r['note']}")
    T.to_csv(os.path.join(PROC_DIR, "efflux_anatomy.csv"), index=False)
    B.to_csv(os.path.join(PROC_DIR, "efflux_zonation_bin.csv"), index=False)
    log(f"\n[save] {PROC_DIR}/efflux_anatomy.csv")
    log(f"[save] {PROC_DIR}/efflux_zonation_bin.csv")

    # ---------- Q3: Abca1 与 Abcg8 是否共表达 ----------
    def vec(g):
        return np.asarray(X[:, genes.index(g)].todense()).ravel() \
            if g in set(genes) else None

    a1, g8 = vec("Abca1"), vec("Abcg8")
    rho_ab = float(np.corrcoef(a1, g8)[0, 1])
    # 单细胞层面的共表达(非零比例)
    co = float(((a1 > 0) & (g8 > 0)).mean())
    log()
    log("=" * 104)
    log("Q3  Abca1 与 Abcg8 在同一批细胞里共表达吗?")
    log("=" * 104)
    log(f"  细胞层 Pearson r(Abca1, Abcg8) = {rho_ab:+.4f}")
    log(f"  同时检出比例                  = {co:.1%}")
    log(f"  仅 Abca1+                     = {float(((a1>0)&(g8==0)).mean()):.1%}")
    log(f"  仅 Abcg8+                     = {float(((a1==0)&(g8>0)).mean()):.1%}")
    a1b, g8b = vec("Abca1"), vec("Abcg8")
    for g2 in ["Sham", "CLP"]:
        m = grp == g2
        log(f"    {g2}: r = {np.corrcoef(a1b[m], g8b[m])[0,1]:+.4f}   "
            f"共检出 {((a1b[m]>0)&(g8b[m]>0)).mean():.1%}")

    # ---------- 图 ----------
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.3))
    # (1) 区带梯度:选 HDL 侧 / 胆汁侧 / 区带参照
    A = axes[0]
    show = [("Abca1", "HDL_efflux"), ("Abcg1", "HDL_efflux"),
            ("Pltp", "HDL_efflux"),
            ("Abcg5", "Biliary_sterol"), ("Abcg8", "Biliary_sterol"),
            ("Abcb11", "Biliary_sterol"),
            ("Cyp2e1", "Zone_ref"), ("Ass1", "Zone_ref")]
    for g, gname in show:
        sub = B[(B["gene"] == g) & (B["group"] == "Sham")].sort_values("bin")
        if len(sub) < 3:
            continue
        v = sub["mean"].values
        v = v / (np.max(np.abs(v)) + 1e-9)     # 各自归一化到 [-1,1] 便于比形状
        A.plot(sub["bin"].values, v, "-o", ms=3.5, lw=1.4,
               color=COLOR[gname], label=g)
    A.axhline(0, color="#BDC3C7", lw=0.7)
    A.set_xlabel("zonation bin (0 = portal  →  9 = central)")
    A.set_ylabel("expression (per-gene normalized)")
    A.set_title("Zonation profile — HDL efflux vs biliary", loc="left",
                fontweight="bold")
    A.legend(fontsize=6.8, ncol=2, frameon=False)
    A.grid(lw=0.3, color="#ECF0F1")

    # (2) 相关系数条形图
    Ax = axes[1]
    sub = T.dropna(subset=["rho_zonation"]).copy()
    sub = sub[sub["group"].isin(["HDL_efflux", "Biliary_sterol", "Zone_ref",
                                 "Bile_acid_synth", "Nuclear_receptor"])]
    sub = sub.sort_values("rho_zonation")
    Ax.barh(np.arange(len(sub)), sub["rho_zonation"],
            color=[COLOR[g] for g in sub["group"]], height=0.65,
            edgecolor="white", linewidth=0.5)
    Ax.set_yticks(np.arange(len(sub)))
    Ax.set_yticklabels(sub["gene"], fontsize=7.5)
    Ax.axvline(0, color=ACCENT, lw=0.9)
    Ax.set_xlabel("Pearson r with zonation index")
    Ax.set_title("Portal (r<0) ←→ Central (r>0)", loc="left", fontweight="bold")
    Ax.grid(axis="x", lw=0.3, color="#ECF0F1")
    Ax.set_axisbelow(True)

    # (3) Abca1 vs Abcg8 共表达散点
    C3 = axes[2]
    rng = np.random.default_rng(0)
    n_samp = min(len(a1), 15000)
    idx = rng.choice(len(a1), size=n_samp, replace=False)
    # 对零值加微小 jitter,避免原点堆叠造成"空图"观感(审查 P2-3)
    jr = np.random.default_rng(1)
    jx = np.where(a1[idx] == 0, jr.normal(0, 0.02, size=n_samp), 0)
    jy = np.where(g8[idx] == 0, jr.normal(0, 0.02, size=n_samp), 0)
    for g2, col in [("Sham", SHAM), ("CLP", CLP)]:
        m = grp[idx] == g2
        C3.scatter(a1[idx][m] + jx[m], g8[idx][m] + jy[m], s=3, color=col,
                   alpha=0.2, label=g2, edgecolors="none")
    C3.set_xlabel("Abca1 expression")
    C3.set_ylabel("Abcg8 expression")
    C3.set_title(f"Co-expression: r = {rho_ab:+.3f}", loc="left",
                 fontweight="bold")
    C3.legend(fontsize=7, frameon=False, markerscale=3)
    C3.grid(lw=0.3, color="#ECF0F1")

    handles = [plt.Line2D([], [], marker="s", ls="", color=C_HDL, label="HDL efflux"),
               plt.Line2D([], [], marker="s", ls="", color=C_BILE, label="Biliary"),
               plt.Line2D([], [], marker="s", ls="", color="#16A085", label="Zone ref"),
               plt.Line2D([], [], marker="s", ls="", color=C_REG, label="Regulator")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Supplementary: Abca1 and Abcg8 are different exits — "
                 "HDL efflux vs biliary excretion",
                 fontsize=10.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = os.path.join(FIG_DIR, f"FigS_efflux_anatomy.{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight")
        log(f"[save] {p}")
    plt.close(fig)

    log()
    log("=" * 104)
    log("判读提示(看完上表再决定怎么拆)")
    log("=" * 104)
    log("  若 HDL 侧与胆汁侧的 rho_zonation 明显不同号 / 不同量级,")
    log("  或 Abca1 与 Abcg8 共表达 r 很低,则**合并打分在细胞层面不成立**,必须拆。")


if __name__ == "__main__":
    main()
