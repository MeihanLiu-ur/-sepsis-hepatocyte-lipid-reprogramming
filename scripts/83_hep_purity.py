#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
83_hep_purity.py -- 肝细胞集纯度核查:双条件判据真的挡住 ambient 污染了吗?

为什么必须查这一项
  Fig2 面板 b 显示:非肝细胞类型的 Alb 检出率高达 96%–100%(Kupffer 100%、
  Neutrophil 97.7%、Endothelial 97.4%、HSC 95.7%),与肝细胞的 99.8% 无异。
  这意味着**只看 marker 均值会把它们全判成肝细胞**。

  我们的防线是双条件判据(identity >= 3 AND lineage <= 1)。
  但它挡得住吗?如果非肝类型大量通过判据,那么 67,145 个"肝细胞"里
  混着大量污染,Fig3 的所有程序层结论都会被稀释甚至带偏。
  **这一项不查,整篇论文的细胞身份基础就是空的。**

查什么
  Q1 各聚类注释类型里有多少通过了判据(is_hep)?污染率是多少?
  Q2 被判为肝细胞但聚类注释不是肝细胞的那些细胞,是什么类型?
  Q3 反过来:聚类注释为肝细胞但**没**通过判据的,又是什么?
  Q4 关键对照:被判入的"非肝"细胞,其非肝谱系标志命中数是多少?
     (若 lineage 命中数很高却仍被判入,说明判据阈值有问题)

用法: python scripts/83_hep_purity.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import scanpy as sc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_sepsis import MARKERS, PROC_DIR  # noqa: E402

ID_GENES = ["Alb", "Ttr", "Apoa1", "Apoa2", "Serpina1a", "Cyp2e1", "Mup3"]
NONHEP = ["Ptprc", "Pecam1", "Col1a1", "Krt19", "Clec4f", "Cd79a",
          "S100a8", "Ly6g", "Cd3e", "Acta2", "Dcn", "Adgre1"]


def log(*a, **kw):
    print(*a, flush=True, **kw)


def main():
    A = sc.read_h5ad(os.path.join(PROC_DIR, "merged_no22.h5ad"))
    vnames = list(A.var_names)
    X = A.X.tocsr()
    ct = A.obs["celltype"].values
    is_hep = A.obs["is_hep"].values.astype(bool)
    log(f"细胞 {A.n_obs:,}   判据判为肝细胞 {is_hep.sum():,} ({is_hep.mean():.1%})")

    idn = np.zeros(A.n_obs, dtype=int)
    for g in ID_GENES:
        if g in vnames:
            idn += (np.asarray(X[:, vnames.index(g)].todense()).ravel() > 0).astype(int)
    nhn = np.zeros(A.n_obs, dtype=int)
    for g in NONHEP:
        if g in vnames:
            nhn += (np.asarray(X[:, vnames.index(g)].todense()).ravel() > 0).astype(int)

    order = pd.Series(ct).value_counts().index.tolist()

    log()
    log("=" * 100)
    log("Q1/Q2  各注释类型里有多少通过了双条件判据?(污染率)")
    log("=" * 100)
    log(f"  {'celltype':<16s}{'n':>8s}{'is_hep':>9s}{'污染率':>9s}"
        f"{'identity均值':>13s}{'lineage均值':>13s}")
    rows = []
    for c in order:
        m = ct == c
        n = int(m.sum())
        k = int(is_hep[m].sum())
        rows.append(dict(celltype=c, n=n, is_hep=k, pct=k / n,
                         id_mean=round(float(idn[m].mean()), 2),
                         lin_mean=round(float(nhn[m].mean()), 2)))
        log(f"  {c:<16s}{n:>8,}{k:>9,}{k/n:>9.1%}"
            f"{idn[m].mean():>13.2f}{nhn[m].mean():>13.2f}")
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(PROC_DIR, "fig2_hep_purity.csv"), index=False)
    log(f"\n[save] {PROC_DIR}/fig2_hep_purity.csv")

    non_hep_types = [c for c in order if c != "Hepatocyte"]
    tot_non = int(sum(r["n"] for r in rows if r["celltype"] != "Hepatocyte"))
    tot_leak = int(sum(r["is_hep"] for r in rows if r["celltype"] != "Hepatocyte"))
    log()
    log(f"  非肝类型总计 {tot_non:,} 个细胞,其中 {tot_leak:,} 个被判为肝细胞"
        f"  → 判据污染率 {tot_leak/tot_non:.1%}")
    log(f"  占最终肝细胞集 {is_hep.sum():,} 的 {tot_leak/is_hep.sum():.2%}")

    log()
    log("=" * 100)
    log("Q3  注释为肝细胞但**未**通过判据的细胞(漏检)")
    log("=" * 100)
    mh = (ct == "Hepatocyte") & (~is_hep)
    log(f"  {mh.sum():,} 个 ({mh.sum()/max((ct=='Hepatocyte').sum(),1):.1%})")
    if mh.sum():
        log(f"    identity 命中均值 {idn[mh].mean():.2f} / {len(ID_GENES)}")
        log(f"    lineage  命中均值 {nhn[mh].mean():.2f} / {len(NONHEP)}")
        log("    → 这些细胞多半是低深度核(identity 基因没测够)或双细胞")

    log()
    log("=" * 100)
    log("Q4  被判入的非肝细胞:lineage 命中数分布(判据是否失效?)")
    log("=" * 100)
    leak_m = is_hep & (ct != "Hepatocyte")
    if leak_m.sum():
        vc = pd.Series(nhn[leak_m]).value_counts().sort_index()
        for k, v in vc.items():
            log(f"    lineage 命中 {k} 个: {v:>6,} ({v/leak_m.sum():.1%})")
        log()
        log("    判据要求 lineage <= 1,故命中数应全部 <= 1。")
        log("    若出现 > 1,说明 obs['is_hep'] 与此处重算的基因集不一致,需核对。")

    log()
    log("=" * 100)
    log("结论判读")
    log("=" * 100)
    if tot_leak / is_hep.sum() < 0.05:
        log(f"  ✅ 判据污染率 {tot_leak/is_hep.sum():.2%} < 5%,肝细胞集可用。")
    else:
        log(f"  ⚠️ 判据污染率 {tot_leak/is_hep.sum():.2%} >= 5%,需收紧判据后重跑。")
    log()
    log("  注意:即便污染率低,Fig2 面板 b 的 ambient 证据仍然重要 ——")
    log("  它说明**为什么不能用 marker 均值/argmax 注释**,而必须用双条件判据。")


if __name__ == "__main__":
    main()
