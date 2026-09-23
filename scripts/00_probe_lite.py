#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
00_probe_lite.py -- 零依赖版 Go/No-Go 探针 (只依赖 numpy / scipy, 无需 scanpy)

用途: 在投入完整分析管线之前, 用一个样本快速回答三个生死问题
    Q1  肝细胞的绝对数量够不够做区带分析?
    Q2  区带 landmark 基因的检出率够不够做 zonation 打分?
    Q3  脂质程序关键基因的检出率够不够做程序打分?

输入: Cell Ranger 三件套所在目录
用法: python3 scripts/00_probe_lite.py data/raw/GSM8482478_liver_02 [outdir]

兼容: Python 3.9+   (本机系统 python3 = 3.9.6, 自带 numpy/scipy)
"""
import sys
import os
import gzip
import time

import numpy as np
from scipy.io import mmread
from scipy.sparse import csc_matrix, csr_matrix

REPORT = []


def say(s=""):
    print(s)
    REPORT.append(str(s))


# ----------------------------------------------------------------------
def read_10x(d, prefix):
    """读 Cell Ranger 三件套, 返回 (X_csc, genes, barcodes)
    X 形状 = 基因 x 细胞"""
    def find(keyword, exts):
        cands = [f for f in os.listdir(d)
                 if f.startswith(prefix) and ("_" + keyword + "_") in f
                 and f.endswith(exts)]
        if not cands:
            raise FileNotFoundError(
                f"在 {d} 下找不到 {prefix}_*{keyword}_*{exts}")
        return os.path.join(d, sorted(cands)[0])

    def open_maybe_gz(p):
        return gzip.open(p, "rt") if p.endswith(".gz") else open(p, "rt")

    files = os.listdir(d)
    fp_bar = find("barcodes", ".tsv.gz" if any(
        f.startswith(prefix) and "_barcodes_" in f and f.endswith(".tsv.gz")
        for f in files) else ".tsv")
    fp_feat = find("features", ".tsv.gz" if any(
        f.startswith(prefix) and "_features_" in f and f.endswith(".tsv.gz")
        for f in files) else ".tsv")
    fp_mat = find("matrix", ".mtx.gz" if any(
        f.startswith(prefix) and "_matrix_" in f and f.endswith(".mtx.gz")
        for f in files) else ".mtx")

    with open_maybe_gz(fp_bar) as fh:
        barcodes = [ln.strip() for ln in fh if ln.strip()]
    with open_maybe_gz(fp_feat) as fh:
        rows = [ln.rstrip("\n").split("\t") for ln in fh if ln.strip()]
    # features.tsv 可能是 2 列(旧版 genes.tsv)或 3 列
    genes = [r[1] if len(r) >= 2 else r[0] for r in rows]

    t0 = time.time()
    with open_maybe_gz(fp_mat) as fh:
        X = mmread(fh).tocsc().astype(np.float32)   # 基因 x 细胞
    say(f"[load] {prefix}: {X.shape[0]} 基因 x {X.shape[1]} 细胞, "
        f"nnz={X.nnz}, 耗时 {time.time()-t0:.1f}s")
    return X, np.array(genes), np.array(barcodes), fp_mat


# ----------------------------------------------------------------------
def normalize_log(X):
    """CP10K + log1p, 输入 基因x细胞 的 csc, 返回 同样稀疏度的 csr(细胞x基因)"""
    Xc = X.T.tocsr()                      # 细胞 x 基因
    tot = np.asarray(Xc.sum(axis=1)).ravel()
    tot[tot == 0] = 1.0
    scaling = csr_matrix((1e4 / tot).astype(np.float32)).T if False else None
    # 手動縮放, 避免構造大對角陣
    Xc = Xc.astype(np.float32)
    Xc = csr_matrix((Xc.data, Xc.indices, Xc.indptr), shape=Xc.shape)
    rowscale = (1e4 / tot).astype(np.float32)
    Xc.data = Xc.data * np.repeat(rowscale, np.diff(Xc.indptr))
    Xc.data = np.log1p(Xc.data)
    return Xc, tot


# ----------------------------------------------------------------------
def score_genes(Xc_log, genes, gene_list, gidx):
    """简化版 module score: 对给定基因集的 log 表达取均值 (细胞 x 1)"""
    cols = [gidx[g] for g in gene_list if g in gidx]
    if not cols:
        return None, []
    sub = Xc_log[:, cols]
    # 用 toarray 前先判断规模
    m = np.asarray(sub.mean(axis=1)).ravel()
    return m, [gene_list[i] for i, g in enumerate(gene_list) if g in gidx]


# ----------------------------------------------------------------------
def detect_rate(X, genes, gidx, gene, mask=None):
    """某基因在给定细胞子集上的检出率 (非零比例)"""
    if gene not in gidx:
        return None
    v = X[gidx[gene], :]
    v = np.asarray(v.todense()).ravel()
    if mask is not None:
        v = v[mask]
    if v.size == 0:
        return None
    return float((v > 0).sum()) / v.size


# ----------------------------------------------------------------------
def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    prefix = sys.argv[2] if len(sys.argv) > 2 else "GSM8482478"
    outdir = sys.argv[3] if len(sys.argv) > 3 else "data/qc"
    os.makedirs(outdir, exist_ok=True)

    say("=" * 74)
    say(f"PROBE (lite)   dir={d}   prefix={prefix}")
    say("=" * 74)

    X, genes, barcodes, fmat = read_10x(d, prefix)
    gidx = {g: i for i, g in enumerate(genes)}
    n_cells = X.shape[1]

    # ---- 2. 基础 QC 分布 (只看分布, 不硬过滤) ----
    Xc = X.T.tocsr()
    tot = np.asarray(Xc.sum(axis=1)).ravel()
    ngene = np.asarray((Xc > 0).sum(axis=1)).ravel()

    mt = np.array([g.startswith(("mt-", "MT-")) for g in genes])
    pct_mt = np.zeros(n_cells)
    if mt.any():
        pct_mt = np.asarray(Xc[:, mt].sum(axis=1)).ravel() / np.maximum(tot, 1) * 100

    say()
    say("[qc] 分布 (未过滤)")
    for name, v in [("total_counts", tot),
                    ("n_genes", ngene),
                    ("pct_mt", pct_mt)]:
        say(f"     {name:14s} median={np.median(v):10.1f}  "
            f"q05={np.percentile(v,5):9.1f}  q95={np.percentile(v,95):10.1f}")
    say("[note] snRNA 的 pct_mt 天然接近 0, 不能作为主过滤阈值")

    # ---- 3. 细胞类型粗判 ----
    Xc_log, _ = normalize_log(X)

    MARKERS = {
        "Hepatocyte":    ["Alb", "Ttr", "Apoa1", "Apoa2", "Serpina1a", "Cyp2e1", "Mup3"],
        "Endothelial":   ["Pecam1", "Cdh5", "Clec4g", "Oit3", "Stab2", "Lyve1"],
        "Kupffer":       ["Clec4f", "Vsig4", "Timd4", "Cd5l"],
        "Macrophage":    ["Adgre1", "Csf1r", "Lyz2", "Cd68", "C1qa"],
        "Neutrophil":    ["S100a8", "S100a9", "Retnlg", "Ly6g", "Il1b"],
        "T_NK":          ["Cd3e", "Cd3d", "Nkg7", "Klrb1c"],
        "B":             ["Cd79a", "Cd79b", "Ms4a1"],
        "HSC":           ["Col1a1", "Dcn", "Acta2", "Rgs5"],
        "Cholangiocyte": ["Krt19", "Krt7", "Epcam", "Spp1"],
    }

    say()
    say("[celltype] marker 打分 argmax 粗判 (仅供数量级估计)")
    scores = {}
    for ct, gl in MARKERS.items():
        s, found = score_genes(Xc_log, genes, gl, gidx)
        if s is None:
            say(f"     {ct:14s} 无 marker 匹配 -- 检查基因名体系")
            continue
        scores[ct] = s
        say(f"     {ct:14s} 匹配 {len(found)}/{len(gl)} 个 marker")

    names = list(scores.keys())
    M = np.vstack([scores[c] for c in names])
    assign = np.array(names)[M.argmax(axis=0)]

    uniq, cnt = np.unique(assign, return_counts=True)
    order = np.argsort(-cnt)
    for i in order:
        say(f"     {uniq[i]:14s} {cnt[i]:8d}  ({100*cnt[i]/n_cells:5.1f}%)")

    hep_mask = (assign == "Hepatocyte")
    n_hep = int(hep_mask.sum())
    say()
    say(f">>> Q1  肝细胞绝对数 = {n_hep} / {n_cells}")
    say(f"     区带分析门槛: 每 zone ~150-300 细胞 -> 单样本需 >=500-1000")
    say(f"     判定: {'PASS' if n_hep >= 500 else 'FAIL'}")

    # ---- 4. 区带 landmark 检出率 ----
    ZONE = {
        "periportal(z1)": ["Ass1", "Arg1", "Cps1", "Otc", "Gls2", "Sds", "Hal",
                           "Tat", "Hpd", "Pck1", "Cyp2f2"],
        "pericentral(z3)": ["Glul", "Cyp2e1", "Cyp1a2", "Oat", "Slc1a2", "Axin2",
                            "Rgn", "Cyp2c29", "Nnmt", "Tbx3"],
    }
    say()
    say(f"[zonation] landmark 基因在肝细胞子集中的检出率 (n={n_hep})")
    n_strong = 0
    rows = []
    for zone, gl in ZONE.items():
        say(f"   -- {zone}")
        for g in gl:
            r = detect_rate(X, genes, gidx, g, hep_mask)
            if r is None:
                say(f"        {g:10s} 未匹配")
                continue
            flag = "OK  " if r > 0.30 else ("weak" if r > 0.10 else "LOW ")
            if r > 0.30:
                n_strong += 1
            say(f"        {g:10s} {r*100:6.2f}%  {flag}")
            rows.append((zone, g, r))
    say()
    say(f">>> Q2  检出率 >30% 的 landmark 数 = {n_strong} (门槛: 两端各>=4, 共>=8)")
    say(f"     判定: {'PASS 区带打分可行' if n_strong >= 8 else 'FAIL landmark 检出不足'}")

    # ---- 5. 脂质程序基因检出率 ----
    LIPID = {
        "FAO":                 ["Cpt1a", "Cpt2", "Acox1", "Acadl", "Acadm", "Hadha",
                                "Hadhb", "Ehhadh", "Ppara", "Hmgcs2", "Slc25a20"],
        "Peroxisome":          ["Pex5", "Pex14", "Abcd1", "Abcd3", "Acox1", "Scp2",
                                "Acot1", "Acot2"],
        "Lipogenesis":         ["Fasn", "Scd1", "Acaca", "Acacb", "Elovl6", "Srebf1",
                                "Mlxipl", "Dgat1", "Dgat2"],
        "Cholesterol_synth":   ["Hmgcr", "Hmgcs1", "Sqle", "Ldlr", "Dhcr7", "Fdft1",
                                "Fdps", "Idi1", "Mvd", "Lss", "Srebf2"],
        "Lipoprotein_assembly":["Mttp", "Apoa1", "Apoa2", "Apob", "Apoe", "Sar1b", "P4hb"],
        "Uptake":              ["Cd36", "Fabp1", "Slc27a1", "Slc27a2", "Ldlr",
                                "Scarb1", "Lrp1", "Olr1"],
        "Efflux":              ["Abca1", "Abcg1", "Abcg5", "Abcg8", "Nr1h3", "Nr1h2"],
        "Lipid_peroxidation":  ["Gpx4", "Slc7a11", "Aifm2", "Acsf2", "Alox5",
                                "Alox15", "Ptgs2", "Nfe2l2", "Hmox1"],
    }
    say()
    say("[lipid] 脂质程序基因在肝细胞子集中的检出率")
    n_prog_ok = 0
    prog_rows = []
    for prog, gl in LIPID.items():
        fr = []
        for g in gl:
            r = detect_rate(X, genes, gidx, g, hep_mask)
            if r is not None:
                fr.append((g, r))
        if not fr:
            say(f"   -- {prog:22s}: 无基因匹配")
            continue
        n_ok = sum(1 for _, r in fr if r > 0.20)
        mean_r = float(np.mean([r for _, r in fr]))
        if n_ok >= 4:
            n_prog_ok += 1
        say(f"   -- {prog:22s} 匹配 {len(fr):2d}/{len(gl):2d}, "
            f"检出>20% 的 {n_ok:2d} 个, 平均 {mean_r*100:5.1f}%")
        prog_rows.append((prog, len(fr), n_ok, mean_r))
        for g, r in sorted(fr, key=lambda x: -x[1])[:5]:
            say(f"          {g:12s} {r*100:6.2f}%")

    say()
    say(f">>> Q3  具备 >=4 个可用基因的脂质程序数 = {n_prog_ok} / {len(prog_rows)} (门槛 >=5)")
    say(f"     判定: {'PASS 程序打分可行' if n_prog_ok >= 5 else 'FAIL 程序基因覆盖不足'}")

    # ---- 6. 落盘 ----
    with open(os.path.join(outdir, f"probe_lite_{prefix}.txt"), "w") as fh:
        fh.write("\n".join(REPORT))

    import csv
    with open(os.path.join(outdir, f"zone_detect_{prefix}.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["zone", "gene", "detect_rate"])
        w.writerows(rows)
    with open(os.path.join(outdir, f"lipid_detect_{prefix}.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["program", "n_matched", "n_above20pct", "mean_detect"])
        w.writerows(prog_rows)

    say()
    say("=" * 74)
    say(f"报告 -> {outdir}/probe_lite_{prefix}.txt")
    say("=" * 74)


if __name__ == "__main__":
    main()
