#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
60_validate_bulk.py -- 解决 Sham n=2 的统计死局:用独立 bulk 队列做正式统计

背景
  主数据 GSE275689 (snRNA) 在剔除 liver_22 后只剩 CLP n=3 vs Sham n=2。
  样本层置换分组数 C(5,3) = 10,**最小可达 p = 0.100**,数学上不可能 <0.05。
  因此 snRNA 只能作**发现层(假说生成)**,不能承载正式统计。

解决方案:两层证据设计
  Tier 1 发现层  GSE275689 snRNA,            n=3 vs 2   -> 方向与效应量
  Tier 2 验证层  GSE311736 纯化肝细胞 bulk,  **n=4 vs 4** -> DESeq2 正式统计
                 (VIB-UGent, 比利时;与主数据的 Leibniz-HKI 德国完全独立)
                 n=4 vs 4 时置换分组数 C(8,4)=70,最小可达 p = 1/70 = 0.014

  这是标准的 discovery + independent validation 结构,
  且 bulk 是 **FACS 纯化肝细胞**,无需 deconvolution,正好对应核心细胞类型。

做什么
  1. 从 snRNA 的 features.tsv 本地构建 Ensembl -> symbol 映射(不依赖网络)
  2. 取 HEP_CLP-1..4 / HEP_Sham-1..4 八列
  3. pydeseq2 跑 CLP vs Sham
  4. 检验 snRNA 发现的方向是否在独立队列中复现:
     - 关键基因逐个看 log2FC / p / padj
     - 8 条脂质程序做**竞争性基因集检验**(程序基因 log2FC vs 背景, Wilcoxon)

用法: python scripts/60_validate_bulk.py
"""
from __future__ import annotations

import os
import sys
import gzip
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_sepsis import LIPID_PROGRAMS, PROC_DIR, RAW_DIR  # noqa: E402

VAL = "data/validation"
COUNTS = os.path.join(VAL, "GSE311736_counts.txt.gz")

# snRNA 里完全分离的基因(见 fig3_key_genes.csv),这里做独立复现检验
SNRNA_KEY = {
    "Cholesterol_synth":    ["Sqle", "Hmgcs1", "Hmgcr", "Msmo1", "Fdps"],
    "Lipogenesis":          ["Acaca", "Fasn", "Scd1", "Acly", "Elovl6"],
    "FAO":                  ["Cpt1a", "Acadm", "Acadl", "Hadha", "Echs1"],
    "Fatty_acid_uptake":    ["Cd36", "Slc27a1", "Slc27a2", "Fabp1"],
    "Lipoprotein_uptake":   ["Ldlr", "Scarb1", "Lrp1"],
    "Lipoprotein_assembly": ["Apoa1", "Apoa2", "Apob", "Mttp"],
    "Efflux":               ["Abca1", "Abcg5", "Abcg8", "Apoe"],
    "Lipid_peroxidation":   ["Gpx4", "Acsl4", "Hmox1", "Alox5", "Tfrc"],
}


def log(*a):
    print(*a, flush=True)


def build_symbol_map(bulk_ids, cache="data/validation/ensembl2symbol.tsv"):
    """
    构建 Ensembl -> symbol 映射。

    注意: 本项目的 snRNA features.tsv **两列都是 symbol**(Cell Ranger 用的
    参考注释没带 Ensembl ID),所以无法本地建表,必须走外部 ID 转换。
    这里用 mygene.info,结果**缓存到本地**,只查一次。
    """
    if os.path.exists(cache):
        m = {}
        with open(cache) as fh:
            next(fh)
            for line in fh:
                p = line.rstrip("\n").split("\t")
                if len(p) >= 2 and p[1]:
                    m[p[0]] = p[1]
        log(f"[map] 读本地缓存 {cache}: {len(m):,} 条")
        return m

    import urllib.request
    import urllib.parse
    import json

    m = {}
    ids = [i for i in bulk_ids]
    B = 900
    log(f"[map] 走 mygene.info 转换 {len(ids):,} 个 Ensembl ID (批大小 {B})")
    for s in range(0, len(ids), B):
        chunk = ids[s:s + B]
        data = urllib.parse.urlencode({
            "q": ",".join(chunk),
            "scopes": "ensembl.gene",
            "fields": "symbol",
            "species": "mouse",
        }).encode()
        try:
            with urllib.request.urlopen(
                    urllib.request.Request("https://mygene.info/v3/query", data=data),
                    timeout=120) as r:
                for hit in json.loads(r.read().decode()):
                    if hit.get("symbol"):
                        m[hit["query"]] = hit["symbol"]
        except Exception as e:
            log(f"[map] 批次 {s//B} 失败: {type(e).__name__}: {e}")
        if (s // B) % 10 == 0:
            log(f"      已处理 {min(s+B, len(ids)):,}/{len(ids):,}")

    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with open(cache, "w") as fh:
        fh.write("ensembl\tsymbol\n")
        for k, v in m.items():
            fh.write(f"{k}\t{v}\n")
    log(f"[map] 完成并缓存: {len(m):,} 条 -> {cache}")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-counts", type=int, default=10,
                    help="基因保留阈值: 8 个样本的总计数 >= 该值")
    args = ap.parse_args()

    if not os.path.exists(COUNTS):
        sys.exit(f"缺少 {COUNTS},请先下载(1.3 MB)")

    # ---------- 1. 读取 + 取肝细胞列 ----------
    log("=" * 74)
    log("Tier 2 验证层: GSE311736 纯化肝细胞 bulk (n=4 CLP vs 4 Sham)")
    log("=" * 74)
    df = pd.read_csv(COUNTS, sep="\t", index_col=0)
    log(f"[load] 完整矩阵: {df.shape[0]:,} 基因 x {df.shape[1]} 样本")

    hep_cols = [c for c in df.columns if c.startswith("HEP_")]
    log(f"[subset] 肝细胞列 ({len(hep_cols)}): {hep_cols}")
    df = df[hep_cols].copy()
    df.index.name = "ensembl"

    # ---------- 2. Ensembl -> symbol ----------
    smap = build_symbol_map(list(df.index))
    df.index = [smap.get(i, i) for i in df.index]
    # 合并同名 symbol(取和)
    df = df.groupby(level=0).sum()
    log(f"[map] 转换后: {df.shape[0]:,} 个 symbol")

    # ---------- 3. 过滤低表达 ----------
    keep = df.sum(axis=1) >= args.min_counts
    log(f"[filter] 总计数 >= {args.min_counts}: 保留 {int(keep.sum()):,} / {len(keep):,}")
    df = df[keep]

    # ---------- 4. DESeq2 ----------
    cond = ["CLP" if "CLP" in c else "Sham" for c in df.columns]
    meta = pd.DataFrame({"condition": cond}, index=df.columns)
    log(f"[design] {dict(pd.Series(cond).value_counts())}")

    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

    dds = DeseqDataSet(counts=df.T.astype(int), metadata=meta,
                       design="~condition", refit_cooks=True, quiet=True)
    dds.deseq2()
    ds = DeseqStats(dds, contrast=["condition", "CLP", "Sham"], quiet=True)
    ds.summary()
    res = ds.results_df.copy()
    res = res.sort_values("padj", na_position="last")
    out = os.path.join(PROC_DIR, "bulk_HEP_CLP_vs_Sham.csv")
    res.to_csv(out)
    log(f"\n[save] 全基因差异表 -> {out}")

    # ---------- 5. 关键基因复现检验 ----------
    log()
    log("=" * 74)
    log("关键基因:snRNA 发现的方向是否在独立 bulk 队列中复现")
    log("=" * 74)
    log(f"{'program':<22s}{'gene':<10s}{'log2FC':>8s}{'pvalue':>11s}{'padj':>11s}  方向")
    rows = []
    for prog, gl in SNRNA_KEY.items():
        for g in gl:
            if g not in res.index:
                log(f"{prog:<22s}{g:<10s}{'-- 未检出 --':>30s}")
                continue
            r = res.loc[g]
            d = "↑" if r["log2FoldChange"] > 0.1 else ("↓" if r["log2FoldChange"] < -0.1 else "≈")
            log(f"{prog:<22s}{g:<10s}{r['log2FoldChange']:>8.3f}"
                f"{r['pvalue']:>11.2e}{r['padj']:>11.2e}  {d}")
            rows.append(dict(program=prog, gene=g,
                             log2FC=r["log2FoldChange"],
                             pvalue=r["pvalue"], padj=r["padj"],
                             direction=d) if False else
                        dict(program=prog, gene=g,
                             log2FC=round(float(r["log2FoldChange"]), 4),
                             pvalue=float(r["pvalue"]), padj=float(r["padj"]),
                             direction=d))
    K = pd.DataFrame(rows)
    K.to_csv(os.path.join(PROC_DIR, "bulk_key_genes.csv"), index=False)

    # ---------- 6. 程序层竞争性基因集检验 ----------
    log()
    log("=" * 74)
    log("程序层:竞争性基因集检验(程序基因 log2FC vs 全背景, Wilcoxon)")
    log("=" * 74)
    from scipy.stats import mannwhitneyu
    bg = res["log2FoldChange"].dropna()
    prows = []
    for prog, gl in LIPID_PROGRAMS.items():
        gs = res.loc[[g for g in gl if g in res.index], "log2FoldChange"].dropna()
        if len(gs) < 3:
            continue
        u, p = mannwhitneyu(gs, bg, alternative="two-sided")
        prows.append(dict(program=prog, n=len(gs),
                          mean_log2FC=round(float(gs.mean()), 3),
                          median_log2FC=round(float(gs.median()), 3),
                          bg_median=round(float(bg.median()), 3),
                          p_gene_set=float(p)))
    # 同时测拆分后的摄取程序
    for prog, gl in [("Fatty_acid_uptake", SNRNA_KEY["Fatty_acid_uptake"]),
                     ("Lipoprotein_uptake", SNRNA_KEY["Lipoprotein_uptake"])]:
        gs = res.loc[[g for g in gl if g in res.index], "log2FoldChange"].dropna()
        if len(gs) < 3:
            continue
        u, p = mannwhitneyu(gs, bg, alternative="two-sided")
        prows.append(dict(program=prog + " [拆分]", n=len(gs),
                          mean_log2FC=round(float(gs.mean()), 3),
                          median_log2FC=round(float(gs.median()), 3),
                          bg_median=round(float(bg.median()), 3),
                          p_gene_set=float(p)))
    P = pd.DataFrame(prows)
    for line in P.to_string(index=False).split("\n"):
        log("  " + line)
    P.to_csv(os.path.join(PROC_DIR, "bulk_program_geneset.csv"), index=False)

    log()
    log("=" * 74)
    log("结论提示")
    log("=" * 74)
    nsig = int((res["padj"] < 0.05).sum())
    log(f"  padj < 0.05 的基因数: {nsig:,} / {len(res):,}")
    log(f"  统计设计: n=4 vs 4 -> 置换分组数 C(8,4) = 70,最小可达 p = {1/70:.4f}")
    log("  这个队列**可以**承载正式统计,snRNA 的 n=2 限制由此解除。")


if __name__ == "__main__":
    main()
