#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""临时探针:候选基因在 snRNA 与独立 bulk 队列中的表达与方向"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, scanpy as sc

hep = sc.read_h5ad("data/proc/hep_only.h5ad")
genes = list(hep.var_names)
X = hep.X.tocsr()
bulk = pd.read_csv("data/proc/bulk_HEP_CLP_vs_Sham.csv", index_col=0)
cand = ["Olr1", "Msr1", "Marco", "Pcsk9", "Mylip", "Stab1", "Stab2", "Cd68", "Scarf1",
        "Lrp1", "Ldlrap1", "Lrpap1", "Lrp4", "Lrp2", "Vldlr", "Ldlr", "Sort1", "Sorl1",
        "Lpl", "Lipg", "Lipc", "Angptl3", "Angptl4", "Angptl8", "Apoa5", "Apoc2",
        "Apoe", "Apob", "Cd36", "Scarb1", "Slc27a2", "Slc27a4", "Fabp1", "Slc27a5"]
print(f"{'gene':<10s}{'snRNA检出':>10s}{'baseMean':>11s}{'log2FC':>9s}{'padj':>11s}")
for g in cand:
    d = np.nan
    if g in set(genes):
        d = float((X[:, genes.index(g)] > 0).sum()) / X.shape[0]
    if g in bulk.index:
        r = bulk.loc[g]
        pd_ = r["padj"]
        s = f"{pd_:>11.3g}" if np.isfinite(pd_) else f"{'NA':>11s}"
        print(f"{g:<10s}{d:>10.1%}{r['baseMean']:>11.1f}{r['log2FoldChange']:>9.3f}{s}")
    else:
        print(f"{g:<10s}{d:>10.1%}{'-- bulk 未检出 --':>31s}")
