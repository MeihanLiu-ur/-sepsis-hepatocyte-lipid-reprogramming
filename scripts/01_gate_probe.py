#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_gate_probe.py -- G1/G2/G3 生死闸门探针(带 cell calling)

为什么需要这一版:
  00_probe_lite.py 直接跑在 GEO 的**未过滤矩阵**上(单样本 2,567,689 条
  barcode, 中位 UMI = 1),结果全部失真:
      - 99.7% 的 barcode 被判为"肝细胞"(空液滴的伪影: hepatocyte marker
        数量最多, argmax 在近零向量上系统性偏向它)
      - 所有基因检出率 ~0.1-3%(分母是 256 万个空 barcode)
  所以**必须先做 cell calling, 再判定闸门**。这是本脚本唯一的新增逻辑。

判定门槛(规划文档 §5.1):
  G1 肝细胞数量   全数据(6 样本)肝细胞 >= 3,000;Sham 组 >= 1,500;
                  分 3 个 zone 后每 zone >= 500
  G2 区带 landmark 检出率 >30% 的 landmark >= 8 个(两端各 >= 4)
  G3 脂质程序      >= 5 个程序各有 >= 4 个检出率 >20% 的基因

用法:
  python scripts/01_gate_probe.py [GSM] [raw_dir]
"""
from __future__ import annotations

import os
import sys
import time
import gzip

import numpy as np
import scipy.io as sio
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_sepsis import (  # noqa: E402
    SAMPLES, RAW_DIR, MARKERS, ZONE, LIPID_PROGRAMS,
    knee_call, detect_rate, sample_paths,
)

GSM = sys.argv[1] if len(sys.argv) > 1 else "GSM8482478"
RAW = sys.argv[2] if len(sys.argv) > 2 else RAW_DIR

suffix = dict((g, s) for g, s, _ in SAMPLES)[GSM]
grp = dict((g, c) for g, _, c in SAMPLES)[GSM]


def say(*a):
    print(*a, flush=True)


def read_10x(bc_f, ft_f, mt_f):
    with gzip.open(ft_f, "rt") as fh:
        genes = [ln.rstrip("\n").split("\t")[1] for ln in fh if ln.strip()]
    with gzip.open(bc_f, "rt") as fh:
        bcs = [ln.rstrip("\n").split("\t")[0] for ln in fh if ln.strip()]
    X = sio.mmread(mt_f).tocsc()          # 基因 x 细胞
    return X, np.array(genes), np.array(bcs)


def norm_log(X):
    """CP10K + log1p -> 返回 细胞 x 基因 的 csr"""
    Xc = X.T.tocsr().astype(np.float32)
    tot = np.asarray(Xc.sum(axis=1)).ravel()
    tot[tot == 0] = 1
    Xc = sp.diags(1e4 / tot) @ Xc
    Xc.data = np.log1p(Xc.data)
    return Xc.tocsr()


def score_simple(Xc_log, genes, glist, gidx):
    """简化 module score: 基因集内 log 表达的均值(仅用于细胞类型粗判)"""
    idx = [gidx[g] for g in glist if g in gidx]
    if not idx:
        return None, []
    s = np.asarray(Xc_log[:, idx].mean(axis=1)).ravel()
    return s, [g for g in glist if g in gidx]


def main():
    t0 = time.time()
    bc_f, ft_f, mt_f = sample_paths(GSM, suffix, RAW)
    say("=" * 74)
    say(f"GATE PROBE   {GSM} ({suffix}, {grp})")
    say("=" * 74)

    say(f"[load] 读取 {os.path.basename(mt_f)} ...")
    X, genes, bcs = read_10x(bc_f, ft_f, mt_f)
    say(f"[load] {X.shape[0]} 基因 x {X.shape[1]:,} barcode, nnz={X.nnz:,} "
        f"({time.time()-t0:.1f}s)")

    # ---------- 1. cell calling ----------
    Xc = X.T.tocsr()
    tot = np.asarray(Xc.sum(axis=1)).ravel()
    ngene = np.asarray((Xc > 0).sum(axis=1)).ravel()

    thr, n_knee = knee_call(tot)
    say()
    say("[call] cell calling (knee / inflection point)")
    say(f"       knee 阈值 UMI >= {thr:.0f}  ->  {n_knee:,} 个候选细胞 "
        f"({n_knee/X.shape[1]*100:.2}% of barcodes)")
    order = np.argsort(-tot)
    top10k = tot[order[:10000]]
    say(f"       参考: Adv Sci 2025 用 --force-cells=10000, 其第 10,000 名 UMI = {top10k[-1]:.0f}")
    say(f"       参考: UMI>=500 的 barcode 数 = {int((tot>=500).sum()):,}; "
        f"n_genes>=200 的 = {int((ngene>=200).sum()):,}")

    # 采用: knee 与 force-cells 取较大者, 再叠加 n_genes>=200
    thr_use = min(thr, max(top10k[-1], 100)) if n_knee > 0 else 500
    cell_mask = (tot >= thr_use) & (ngene >= 200)
    n_cells = int(cell_mask.sum())
    say(f"       >>> 采用阈值 UMI>={thr_use:.0f} 且 n_genes>=200 -> {n_cells:,} 个细胞")

    if n_cells < 100:
        say("!! 细胞数过少, 中止"); return

    idx_cells = np.where(cell_mask)[0]
    Xc = Xc[idx_cells]
    tot, ngene = tot[idx_cells], ngene[idx_cells]

    mt = np.array([g.startswith(("mt-", "MT-")) for g in genes])
    pct_mt = (np.asarray(Xc[:, mt].sum(axis=1)).ravel() /
              np.maximum(tot, 1) * 100) if mt.any() else np.zeros(n_cells)

    say()
    say("[qc] 过滤后分布")
    for nm, v in [("total_counts", tot), ("n_genes", ngene), ("pct_mt", pct_mt)]:
        say(f"     {nm:14s} median={np.median(v):9.1f}  q05={np.percentile(v,5):8.1f} "
            f" q95={np.percentile(v,95):9.1f}")

    # ---------- 2. G1: 肝细胞数量 ----------
    Xc_log = norm_log(Xc.T)
    gidx = {g: i for i, g in enumerate(genes)}

    say()
    say("[G1] 细胞类型粗判 (marker score argmax, 仅用于数量级估计)")
    scores = {}
    for ct, gl in MARKERS.items():
        s, found = score_simple(Xc_log, genes, gl, gidx)
        if s is None:
            continue
        scores[ct] = s
    M = np.stack([scores[c] for c in scores], axis=1)
    assign = np.array(list(scores.keys()))[M.argmax(axis=1)]
    # 只有当最高分显著高于次高分时才接受, 否则标为 Ambiguous
    srt = np.sort(M, axis=1)
    amb = (srt[:, -1] - srt[:, -2]) < 0.05
    assign[amb] = "Ambiguous"

    for ct in scores:
        n = int((assign == ct).sum())
        say(f"     {ct:14s} {n:8,}  ({n/n_cells*100:5.1f}%)")
    say(f"     {'Ambiguous':14s} {int(amb.sum()):8,}  ({amb.mean()*100:5.1f}%)")

    hep = assign == "Hepatocyte"
    n_hep = int(hep.sum())
    say()
    say(f">>> G1 肝细胞数 = {n_hep:,} / {n_cells:,} ({n_hep/n_cells*100:.1f}%)")
    say(f"    外推到 6 样本: ~{n_hep*6:,}; Sham 组(3 样本): ~{n_hep*3:,}; "
        f"每 zone: ~{n_hep*3//3:,}")
    g1 = (n_hep * 3) >= 1500 and (n_hep * 3 / 3) >= 500
    say(f"    判定: {'PASS' if g1 else 'FAIL'}  (门槛: Sham 组>=1500, 每 zone>=500)")

    if n_hep < 50:
        say("!! 肝细胞太少, 后续闸门无意义, 中止"); return

    # ---------- 3. G2: 区带 landmark ----------
    Xh = Xc[hep]
    Xh_log = Xc_log[hep]

    say()
    say(f"[G2] 区带 landmark 检出率 (肝细胞子集 n={n_hep:,})")
    n_ok_zone = 0
    per_side = {}
    for side, gl in ZONE.items():
        rows = []
        for g in gl:
            if g not in gidx:
                rows.append((g, None)); continue
            r = detect_rate(Xh, gidx[g])
            rows.append((g, r))
        good = [r for _, r in rows if r is not None and r > 0.30]
        per_side[side] = len(good)
        say(f"   -- {side}  检出>30% 的基因: {len(good)}/{len(gl)}")
        shown = sorted([(g, r) for g, r in rows if r is not None],
                       key=lambda x: -x[1])[:6]
        for g, r in shown:
            flag = "OK " if r > 0.30 else "low"
            say(f"        {g:10s} {r*100:6.2f}%  {flag}")
    n_ok_zone = sum(per_side.values())
    g2 = n_ok_zone >= 8 and per_side.get("periportal", 0) >= 4 and per_side.get("pericentral", 0) >= 4
    say(f">>> G2 检出率>30% 的 landmark 数 = {n_ok_zone} (门槛: 共>=8, 两端各>=4)")
    say(f"    判定: {'PASS' if g2 else 'FAIL'}")

    # ---------- 4. G3: 脂质程序 ----------
    say()
    say(f"[G3] 脂质程序基因检出率 (肝细胞子集 n={n_hep:,})")
    n_prog_ok = 0
    for prog, gl in LIPID_PROGRAMS.items():
        rates = []
        for g in gl:
            if g in gidx:
                rates.append((g, detect_rate(Xh, gidx[g])))
        good = [(g, r) for g, r in rates if r > 0.20]
        ok = len(good) >= 4
        n_prog_ok += int(ok)
        say(f"   -- {prog:22s} 匹配 {len(rates):2d}/{len(gl):2d}, "
            f"检出>20% 的 {len(good):2d} 个, 平均 {np.mean([r for _,r in rates])*100 if rates else 0:5.2f}%"
            f"   {'PASS' if ok else 'FAIL'}")
        for g, r in sorted(rates, key=lambda x: -x[1])[:5]:
            say(f"        {g:10s} {r*100:6.2f}%")
    g3 = n_prog_ok >= 5
    say(f">>> G3 达标程序数 = {n_prog_ok}/8 (门槛 >=5)")
    say(f"    判定: {'PASS' if g3 else 'FAIL'}")

    say()
    say("=" * 74)
    say(f"总判定  G1={'PASS' if g1 else 'FAIL'}  G2={'PASS' if g2 else 'FAIL'}  "
        f"G3={'PASS' if g3 else 'FAIL'}   ({time.time()-t0:.0f}s)")
    say("=" * 74)


if __name__ == "__main__":
    main()
