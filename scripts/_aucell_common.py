#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_aucell_common.py -- AUCell 的共享实现(55 / 56 号脚本共用)

单独抽出来的原因:避免 55(拆分统计)与 56(深度校正)各写一份,
两份实现一旦漂移,敏感性分析就不可比。

AUCell 要点
  对每个细胞,把基因按表达降序排名,计算基因集成员在排名前 auc_max 位内的
  恢复量。基于**秩**,因此不受基因集大小与数据稀疏度影响,且对测序深度
  理论不敏感 —— 但低检出率基因(<10%)在深度变化时秩会剧烈跳动,
  这正是 56 号脚本要检验的。
"""
from __future__ import annotations

import numpy as np


def cellwise_ranks(X):
    """
    每个细胞的非零基因算降序秩(0 = 表达最高),返回与 X.data 对齐的扁平数组。
    零表达基因不出现在稀疏结构中,对 AUC 无贡献。
    """
    X = X.tocsr()
    indptr, data = X.indptr, X.data
    ranks = np.empty(data.shape[0], dtype=np.int64)
    for i in range(X.shape[0]):
        s, e = indptr[i], indptr[i + 1]
        k = e - s
        if k == 0:
            continue
        v = data[s:e]
        order = np.argsort(-v, kind="stable")
        r = np.empty(k, dtype=np.int64)
        r[order] = np.arange(k)
        ranks[s:e] = r
    return ranks


def aucell(X, X_genes, gene_list, auc_max, ranks, cell_of_entry, n_cells):
    """单个基因集的 AUCell"""
    pos = {}
    for i, g in enumerate(X_genes):
        pos.setdefault(g, i)
    idx = np.array(sorted({pos[g] for g in gene_list if g in pos}), dtype=int)
    if idx.size == 0:
        return np.zeros(n_cells, dtype=np.float32)
    gs = np.zeros(len(X_genes), dtype=bool)
    gs[idx] = True
    sel = gs[X.indices]
    if not sel.any():
        return np.zeros(n_cells, dtype=np.float32)
    r = ranks[sel]
    c = cell_of_entry[sel]
    contrib = np.clip(auc_max - r, 0, None).astype(np.float64)
    auc = np.bincount(c, weights=contrib, minlength=n_cells)
    return (auc / (auc_max * len(idx))).astype(np.float32)


def score_all(X, genes, programs, auc_frac=0.05):
    """一次性算多个程序(秩只算一次,程序间复用)"""
    X = X.tocsr()
    n_genes = X.shape[1]
    auc_max = int(max(10, auc_frac * n_genes))
    ranks = cellwise_ranks(X)
    cell_of_entry = np.repeat(np.arange(X.shape[0]), np.diff(X.indptr))
    out = {p: aucell(X, genes, gl, auc_max, ranks, cell_of_entry, X.shape[0])
           for p, gl in programs.items()}
    return out, auc_max
