#!/usr/bin/env python3
"""
00_probe_snrna.py -- C0/C1 边界的 Go / No-Go 探针脚本
用途:在投入完整管线之前,用一个样本快速回答三个生死问题
    Q1 肝细胞的绝对数量够不够做区带分析?
    Q2 区带 landmark 基因的检出率够不够做 zonation 打分?
    Q3 脂质程序关键基因的检出率够不够做程序打分?
输入: Cell Ranger 输出的 10X 三件套目录 (barcodes / features / matrix)
输出: 控制台报告 + data/qc/probe_report.txt
作者: WB / 于杨课题组
"""
import sys
import os
import gzip
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

sc.settings.verbosity = 1
sc.settings.n_jobs = 8

# ----------------------------------------------------------------------
# 0. 参数
# ----------------------------------------------------------------------
SAMPLE_DIR = sys.argv[1] if len(sys.argv) > 1 else "data/raw/GSM8482478"
OUT_DIR = "data/qc"
os.makedirs(OUT_DIR, exist_ok=True)

REPORT = []

def say(s=""):
    print(s)
    REPORT.append(str(s))

# ----------------------------------------------------------------------
# 1. 读入 (snRNA: 注意 features 可能是 3 列,genes 可能是 symbol 或 ensembl)
# ----------------------------------------------------------------------
say("=" * 72)
say(f"PROBE  {SAMPLE_DIR}")
say("=" * 72)

a = sc.read_10x_mtx(SAMPLE_DIR, var_names="gene_symbols", cache=True)
a.var_names_make_unique()
say(f"[load] 原始矩阵 shape = {a.shape[0]} 细胞 x {a.shape[1]} 基因")

# ----------------------------------------------------------------------
# 2. 基础 QC 分布 (snRNA 与 scRNA 阈值不同, 先只看分布不硬过滤)
# ----------------------------------------------------------------------
a.var["mt"] = a.var_names.str.startswith(("mt-", "MT-"))
a.var["ribo"] = a.var_names.str.match(r"^Rp[sl]")
sc.pp.calculate_qc_metrics(a, qc_vars=["mt", "ribo"], percent_top=None,
                           log1p=False, inplace=True)

for k in ["n_genes_by_counts", "total_counts", "pct_counts_mt"]:
    v = a.obs[k].values
    say(f"[qc] {k:22s} median={np.median(v):10.1f}  "
        f"q05={np.percentile(v,5):10.1f}  q95={np.percentile(v,95):10.1f}")

say("[note] snRNA 的 pct_counts_mt 天然接近 0, 不能用线粒体比例做过滤阈值; "
    "snRNA 的质量指标应改用 n_genes 下限 + 核内/核仁基因比例 + 双细胞检测")

# ----------------------------------------------------------------------
# 3. 细胞类型粗判: 用 marker 打分而非聚类 (快, 且不受批次影响)
# ----------------------------------------------------------------------
MARKERS = {
    "Hepatocyte": ["Alb", "Ttr", "Apoa1", "Apoa2", "Serpina1a", "Cyp2e1", "Mup3"],
    "Endothelial": ["Pecam1", "Cdh5", "Clec4g", "Oit3", "Stab2"],
    "Kupffer": ["Clec4f", "Vsig4", "Timd4", "Cd5l"],
    "Macrophage": ["Adgre1", "Csf1r", "Lyz2", "Cd68"],
    "Neutrophil": ["S100a8", "S100a9", "Retnlg", "Ly6g"],
    "T_NK": ["Cd3e", "Cd3d", "Nkg7", "Klrb1c"],
    "B": ["Cd79a", "Cd79b", "Ms4a1"],
    "HSC": ["Col1a1", "Dcn", "Acta2", "Rgs5"],
    "Cholangiocyte": ["Krt19", "Krt7", "Epcam", "Spp1"],
}

# score_genes 需要 log 后的数据
a.layers["counts"] = a.X.copy()
sc.pp.normalize_total(a, target_sum=1e4)
sc.pp.log1p(a)

for ct, genes in MARKERS.items():
    g = [x for x in genes if x in a.var_names]
    if len(g) == 0:
        say(f"[marker] {ct:14s} 一个 marker 都没匹配到 -- 检查基因名体系")
        continue
    sc.tl.score_genes(a, g, score_name=f"sc_{ct}")
    a.obs[f"nfound_{ct}"] = len(g)

sc_cols = [f"sc_{ct}" for ct in MARKERS if f"sc_{ct}" in a.obs.columns]
a.obs["ct_guess"] = a.obs[sc_cols].idxmax(axis=1).str.replace("sc_", "", regex=False)

say()
say("[celltype] 粗判细胞类型构成 (marker 打分 argmax, 仅供数量级估计)")
vc = a.obs["ct_guess"].value_counts()
for ct, n in vc.items():
    say(f"    {ct:14s} {n:8d}  ({100*n/a.n_obs:5.1f}%)")

n_hep = int(vc.get("Hepatocyte", 0))
say()
say(f">>> Q1 肝细胞绝对数 = {n_hep}")
say(f"    区带分析经验门槛: 每个 zone 至少 150-300 个肝细胞 -> 需要 >=500-1000")
say(f"    判定: {'PASS 可以继续' if n_hep >= 500 else 'FAIL 肝细胞不足, 需换数据集或换策略'}")

# ----------------------------------------------------------------------
# 4. 区带 landmark 基因检出率
# ----------------------------------------------------------------------
ZONE_LANDMARK = {
    # Halpern et al. 2017 Nature; Ben-Moshe & Itzkovitz 2019 Nat Metab
    "periportal(z1)": ["Ass1", "Arg1", "Cps1", "Otc", "Gls2", "Sds", "Hal",
                       "Tat", "Hpd", "Pck1", "Cyp2f2", "Alb"],
    "pericentral(z3)": ["Glul", "Cyp2e1", "Cyp1a2", "Oat", "Slc1a2", "Axin2",
                        "Rgn", "Wnt2", "Tbx3", "Lgr5", "Cyp2c29", "Nnmt"],
}

hep_mask = a.obs["ct_guess"] == "Hepatocyte"
sub = a[hep_mask]

say()
say("[zonation] 区带 landmark 基因在肝细胞中的检出率")
say(f"    (肝细胞子集 n = {sub.n_obs})")
all_found = []
for zone, genes in ZONE_LANDMARK.items():
    found = []
    for g in genes:
        if g in sub.var_names:
            frac = float((sub[:, g].X > 0).sum()) / sub.n_obs
            found.append((g, frac))
            all_found.append((zone, g, frac))
    say(f"  -- {zone}")
    for g, frac in found:
        flag = "OK " if frac > 0.30 else ("weak" if frac > 0.10 else "LOW ")
        say(f"       {g:10s} {frac*100:6.2f}%  {flag}")

df_zone = pd.DataFrame(all_found, columns=["zone", "gene", "frac"])
n_strong = int((df_zone["frac"] > 0.30).sum())
say()
say(f">>> Q2 检出率 >30% 的区带 landmark 基因数 = {n_strong}")
say(f"    区带打分经验门槛: 两端各 >=4 个可靠 landmark (共 >=8)")
say(f"    判定: {'PASS 区带打分可行' if n_strong >= 8 else 'FAIL landmark 检出不足'}")

# ----------------------------------------------------------------------
# 5. 脂质程序关键基因检出率
# ----------------------------------------------------------------------
LIPID_PROGRAMS = {
    "FAO": ["Cpt1a", "Cpt2", "Acox1", "Acox2", "Acadm", "Acadl", "Hadha",
            "Hadhb", "Ehhadh", "Ppara", "Hmgcs2", "Slc25a20"],
    "Peroxisome": ["Pex5", "Pex14", "Abcd1", "Abcd3", "Acox1", "Ehhadh",
                   "Scp2", "Acot1", "Acot2"],
    "Lipogenesis": ["Fasn", "Scd1", "Acaca", "Acacb", "Elovl6", "Srebf1",
                    "Mlxipl", "Thrsp", "Dgat1", "Dgat2"],
    "Cholesterol_synth": ["Hmgcr", "Hmgcs1", "Sqle", "Ldlr", "Dhcr7",
                          "Fdft1", "Fdps", "Idi1", "Mvd", "Lss", "Srebf2"],
    "Lipoprotein_assembly": ["Mttp", "Apoa1", "Apoa2", "Apob", "Apoe",
                             "Sar1b", "P4hb", "Pdia2"],
    "Uptake": ["Cd36", "Fabp1", "Fabp2", "Slc27a1", "Slc27a2", "Ldlr",
               "Scarb1", "Lrp1", "Olr1"],
    "Efflux": ["Abca1", "Abcg1", "Abcg5", "Abcg8", "Nr1h3", "Nr1h2", "Cpt1a"],
    "Lipid_peroxidation": ["Gpx4", "Slc7a11", "Aifm2", "Acsf2", "Alox5",
                           "Alox15", "Ptgs2", "Nfe2l2", "Hmox1"],
}

say()
say("[lipid] 脂质程序关键基因在肝细胞中的检出率")
prog_summary = {}
for prog, genes in LIPID_PROGRAMS.items():
    rows = []
    for g in genes:
        if g in sub.var_names:
            frac = float((sub[:, g].X > 0).sum()) / sub.n_obs
            rows.append((g, frac))
    if not rows:
        say(f"  -- {prog:20s}: 无基因匹配")
        continue
    n_ok = sum(1 for _, f in rows if f > 0.20)
    mean_frac = float(np.mean([f for _, f in rows]))
    prog_summary[prog] = (len(rows), n_ok, mean_frac)
    say(f"  -- {prog:20s} 匹配 {len(rows):2d}/{len(genes):2d} 基因, "
        f"检出>20% 的 {n_ok:2d} 个, 平均检出率 {mean_frac*100:5.1f}%")

n_prog_ok = sum(1 for p, (_, nok, _) in prog_summary.items() if nok >= 4)
say()
say(f">>> Q3 具备 >=4 个可用基因的脂质程序数 = {n_prog_ok} / {len(prog_summary)}")
say(f"    判定: {'PASS 程序打分可行' if n_prog_ok >= 5 else 'FAIL 程序基因覆盖不足'}")

# ----------------------------------------------------------------------
# 6. 落盘
# ----------------------------------------------------------------------
with open(os.path.join(OUT_DIR, "probe_report.txt"), "w") as fh:
    fh.write("\n".join(REPORT))

df_zone.to_csv(os.path.join(OUT_DIR, "zone_landmark_detection.csv"), index=False)
pd.DataFrame([(p, *v) for p, v in prog_summary.items()],
             columns=["program", "n_matched", "n_above20pct", "mean_detect"]).to_csv(
    os.path.join(OUT_DIR, "lipid_program_detection.csv"), index=False)

a.write(os.path.join(OUT_DIR, "probe_annotated.h5ad"))

say()
say("=" * 72)
say(f"报告 -> {OUT_DIR}/probe_report.txt")
say(f"对象 -> {OUT_DIR}/probe_annotated.h5ad  (已含 ct_guess + 各 marker 打分)")
say("=" * 72)
