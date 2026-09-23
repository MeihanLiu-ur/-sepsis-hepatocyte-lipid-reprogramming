#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""临时探查:Fig1 需要的字段在不在"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scanpy as sc, pandas as pd, numpy as np

for f in ["data/proc/merged_no22.h5ad", "data/proc/hep_only.h5ad"]:
    A = sc.read_h5ad(f, backed="r")
    print(f"=== {f} ===")
    print(f"  shape {A.n_obs:,} x {A.n_vars:,}")
    print(f"  obs 列 : {list(A.obs.columns)}")
    print(f"  obsm   : {list(A.obsm.keys())}")
    print(f"  layers : {list(A.layers.keys())}")
    for k in ["leiden", "cell_type", "annotation", "celltype"]:
        if k in A.obs.columns:
            print(f"  {k} ({A.obs[k].nunique()} 类):")
            print(A.obs[k].value_counts().to_string())
    if "sample" in A.obs.columns:
        print("  sample:")
        print(A.obs["sample"].value_counts().to_string())
    A.file.close()
    print()

print("=== qc_per_sample.csv ===")
print(pd.read_csv("data/proc/qc_per_sample.csv").to_string())

print("\n=== data/qc/ 目录 ===")
for root, _, fs in os.walk("data/qc"):
    for x in fs:
        p = os.path.join(root, x)
        print(f"  {p}  {os.path.getsize(p):,} B")
