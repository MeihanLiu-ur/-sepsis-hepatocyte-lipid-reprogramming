#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lib_sepsis.py -- 脓毒症肝细胞区带脂质重编程项目:共享配置与方法

内容
  1. 样本台账 SAMPLES(6 个样本, 3 CLP vs 3 Sham)
  2. 细胞类型 marker MARKERS(9 类, 对齐 Adv Sci 2025 的 8 类命名)
  3. 区带 landmark ZONE(periportal / pericentral)
  4. 脂质程序 LIPID_PROGRAMS(8 个程序)
  5. aucell() -- AUCell 的 numpy 实现(不依赖 decoupler)
  6. 若干 IO / QC 小工具

设计要点(来自规划文档 §2 / §5)
  - 区带用**连续 zonation index**,不做硬三分。zone2 没有可靠 marker,
    硬聚类会在连续谱上切出假边界,把 zone2 变成"垃圾桶"。
  - zonation index = AUCell(pericentral) - AUCell(periportal),
    在 Sham 组分位数标定后套用到 CLP 组。
  - 统计一律**样本层**(n=3 vs 3),不得把细胞数当 n。
"""
from __future__ import annotations

import os
import numpy as np
import scipy.sparse as sp

# --------------------------------------------------------------------------
# 1. 样本台账
#    GSE275689 / BioProject PRJNA1152574
#    平台 GPL24247 (Illumina NovaSeq 6000), Cell Ranger v7.2.0, mm39
#    模型: C57BL/6 雄鼠, CLP(30% 结扎 + 23G 双穿刺), 术后 24 h 取样
#          术后 6 h 给 Imipenem/Cilastatin(模型偏轻, 14 天死亡率 30%)
#    注意: GEO 存放的是**未过滤矩阵**(单样本 barcode 达 2,567,689 条),
#          必须自己做 cell calling, 不能直接沿用论文的细胞数。
# --------------------------------------------------------------------------
SAMPLES = [
    # gsm,          文件名后缀,   分组
    ("GSM8482478", "liver_02", "CLP"),
    ("GSM8482479", "liver_06", "CLP"),
    ("GSM8482480", "liver_10", "CLP"),
    ("GSM8482481", "liver_14", "Sham"),
    ("GSM8482482", "liver_18", "Sham"),
    ("GSM8482483", "liver_22", "Sham"),
]

RAW_DIR = "data/raw"
PROC_DIR = "data/proc"
FIG_DIR = "figures"

# --------------------------------------------------------------------------
# 2. 细胞类型 marker
# --------------------------------------------------------------------------
MARKERS = {
    "Hepatocyte":    ["Alb", "Ttr", "Apoa1", "Apoa2", "Serpina1a", "Cyp2e1", "Mup3"],
    "Endothelial":   ["Pecam1", "Cdh5", "Clec4g", "Oit3", "Stab2", "Lyve1"],
    "Kupffer":       ["Clec4f", "Vsig4", "Timd4", "Cd5l"],
    "Macrophage":    ["Adgre1", "Csf1r", "Lyz2", "Cd68", "C1qa"],
    "Neutrophil":    ["S100a8", "S100a9", "Retnlg", "Ly6g", "Il1b"],
    "T_NK":          ["Cd3e", "Cd3d", "Nkg7", "Klrb1c"],
    # ⚠️ 2026-08-29:原 "B" 注释经 89 号脚本深挖定性为**神经元污染**,不是 B 细胞。
    #   9,266 个核自身 B marker(Cd79a/Ighm/Cd19)检出率均 <2.5%,而神经元 marker
    #   (Slc32a1/Slc17a6/Sv2b/Gap43)检出率 12–36%、log2FC +9~+12;73% 来自 liver_10、
    #   26% 来自 liver_18(脑组织混入,与 liver_22 为 WAT 同属样本混淆)。已改标 Neuron。
    "Neuron":        ["Slc32a1", "Slc17a6", "Sv2b", "Gap43"],
    "HSC":           ["Col1a1", "Dcn", "Acta2", "Rgs5"],
    "Cholangiocyte": ["Krt19", "Krt7", "Epcam", "Spp1"],
}

# --------------------------------------------------------------------------
# 3. 区带 landmark
#    ⚠️ 纠错记录: 早期材料把 Cyp1a2 写成 zone2 —— **错**。
#       Cyp1a2 属中央静脉周(zone3)富集基因。写错会稀释 zone3 信号。
#    ⚠️ 纠错记录: Glul 与 GS 是同一基因的基因名/蛋白名,不是两个 marker,
#       原路线的"Glul/GS 双打分"不增加任何独立信息。
#    参考: Halpern et al. 2017 Nature; Ben-Moshe & Itzkovitz 2019 Nat Metab
# --------------------------------------------------------------------------
ZONE = {
    # zone 1 门静脉周: 糖异生、尿素循环、氨基酸分解
    "periportal": [
        "Ass1", "Arg1", "Cps1", "Otc", "Gls2", "Sds", "Hal",
        "Tat", "Hpd", "Pck1", "Cyp2f2", "Aldob",
    ],
    # zone 3 中央静脉周: 谷氨酰胺合成、糖酵解/脂质新生、外源物代谢
    "pericentral": [
        "Glul", "Cyp2e1", "Cyp1a2", "Cyp2c29", "Oat", "Slc1a2",
        "Axin2", "Rgn", "Tbx3", "Nnmt", "Gulo",
    ],
}

# --------------------------------------------------------------------------
# 4. 脂质程序(12 个)
#    与 Adv Sci 2025 的差异化: 他们只覆盖甘油磷脂 / 甘油三酯 / 丝氨酸 / 赖氨酸,
#    未涉及 FAO 与胆固醇合成 —— 这两块是净土。
#
#    ⚠️ v3 变更(2026-08-29): 原 "Uptake" (8 基因) 按**基因家族**一分为三。
#       拆分依据是**先验的基因家族归属**(教科书级),不是本项目的差异表达结果:
#
#         LDLR_clearance   LDLR 基因家族 + 内吞必需辅因子
#                          —— 受体介导的 LDL / 残粒颗粒清除
#         Scavenger_SR_B   B 类清道夫受体(SR-B1 = Scarb1 / SR-B2 = Cd36)
#                          —— 修饰脂蛋白与 oxLDL 的应激性清除
#         FA_transport     Slc27(FATP)溶质载体家族 + 胞内脂肪酸结合蛋白
#                          —— 游离脂肪酸的跨膜转运与胞内运输
#
#       为什么必须拆:三个家族的调控通路完全不同(LDLR 受 SREBP / PCSK9 调控,
#       SR-B 受 PPARα / LXR 与炎症信号调控,FATP 受 PPARα 调控),合并打分会把
#       方向相反的信号相互抵消,得出"≈0、p=0.63"的假阴性 —— 这正是 v1 的结局。
#
#    ⚠️ 移除的基因: Lipc(肝脂肪酶)。
#       它是**分泌型酶**,在窦周间隙水解脂蛋白 TG,不属于肝细胞的摄取 / 转运 /
#       合成机器,放进 "Uptake" 是 v1 的定义错误。其 bulk 结果仍在
#       data/proc/bulk_receptor_split.csv 中如实保留,只是不参与程序打分。
#
#    ⚠️ v4 变更(2026-08-29): 原 "Efflux" (7 基因) 按**外排目的地**一分为二,
#       并单列胆汁酸合成。依据见 scripts/57_efflux_anatomy.py 的实测核查。
#
#       **为什么要拆 —— 三条实测证据(不是从结果反推)**:
#       ① **Abca1 与 Abcg8 在单细胞层面几乎不共表达**:Pearson r = −0.026,
#          同时检出仅 20.5%,而 44.5% 的细胞**只**表达 Abca1。
#          → 两者活跃在不同的肝细胞亚群,合并打分在细胞层面就不成立。
#       ② **外排目的地根本不同(先验生物学)**:
#            ABCA1  → 把游离胆固醇外排给**贫脂 apoA-I**,生成 nascent HDL
#                     (胆固醇逆向转运的第一步、限速步;胆固醇**留在体内**)
#            ABCG5/G8 → 专性异二聚体,定位于**毛细胆管膜**,把胆固醇与
#                     植物固醇**泵入胆汁**(胆固醇**离开体内**)
#       ③ **方向相反且都显著**:Abca1 +3.00 (padj 3e-109) vs
#          Abcg8 −1.04 (padj 9.5e-3)。合并后得到 −0.633 的无意义均值。
#
#       ⚠️ 区带梯度**不支持**拆分:Abca1 ρ = −0.073、Abcg8 ρ = −0.053,
#          两者沿门静脉-中央静脉轴分布几乎一致。这一项如实报告,不夸大。
#
#       **顺带发现的更大信号:整条胆汁分泌轴系统性关闭**
#          Cyp7a1 −7.76 (padj 3.3e-8) / Cyp8b1 −2.29 (padj 1.7e-13)
#          Nr1h4(FXR) −1.68 (padj 2e-12) / Nr1h3(LXRα) −0.89 (padj 1.1e-4)
#          Abcc2 −1.38 (padj 6.7e-11) / Abcb4 −0.95 (padj 8.1e-6)
#       这直接对应**脓毒症相关胆汁淤积(sepsis-associated cholestasis)**,
#       是有临床对应表型的独立发现,故单列 Bile_acid_synth 一条程序。
#
#    ⚠️ v4 移除的基因(定义正确,但不适用于肝细胞,结果仍如实保留):
#       - Abcg1 : 肝细胞几乎不表达(snRNA 检出率 2.3%,bulk baseMean 4),
#                 其 log2FC −5.15 是低表达噪音,保留会把 HDL_efflux 带向错误方向
#       - Nr1h3 : LXRα 是**转录因子**,不是外排执行者,放进转运体基因集是定义错误
#       - Apoe  : 多功能载脂蛋白(既是外排受体配体,也是脂蛋白结构蛋白),
#                 归 Lipoprotein_assembly 更合适
#
#    ⚠️ 已知不确定性(必须在正文披露):
#       - Scavenger_SR_B 与 HDL_efflux 各只有 2 个基因,AUCell 分辨率低,
#         结论主要靠单基因证据(且 HDL_efflux 实际由 Abca1 主导)。
#       - Cd36 兼具炎症诱导的模式识别受体功能,其上调**不能**解释为
#         "脂肪酸摄取需求增加",只能解释为应激性清道夫反应。
# --------------------------------------------------------------------------
LIPID_PROGRAMS = {
    "FAO": [
        "Cpt1a", "Cpt2", "Acadm", "Acadl", "Acadvl", "Hadha", "Hadhb",
        "Etfa", "Etfb", "Slc25a20", "Acaa2", "Echs1",
    ],
    "Peroxisome": [
        "Acox1", "Acox2", "Acox3", "Hsd17b4", "Scp2", "Ehhadh",
        "Abcd1", "Abcd2", "Abcd3", "Decr2", "Crat",
    ],
    "Lipogenesis": [
        "Acly", "Acaca", "Acacb", "Fasn", "Scd1", "Scd2",
        "Elovl6", "Me1", "Gpam",
    ],
    "Cholesterol_synth": [
        "Hmgcs1", "Hmgcr", "Mvd", "Mvk", "Fdps", "Fdft1", "Sqle", "Lss",
        "Cyp51", "Msmo1", "Nsdhl", "Sc5d", "Dhcr7", "Dhcr24", "Idi1",
    ],
    "Lipoprotein_assembly": [
        # Apoe 于 v4 从 Efflux 移入此处:它是脂蛋白**结构蛋白**兼受体配体,
        # 不是外排转运体,归装配更合适
        "Apob", "Mttp", "Tm6sf2", "Sar1b", "Apoa1", "Apoa2", "Apoa4", "Apoc3",
        "Apoe",
    ],
    # ▼ v3 拆分:原 Uptake 按基因家族一分为三,互不重叠
    "LDLR_clearance": [
        # LDLR 基因家族 + 内吞必需辅因子:受体介导的 LDL / 残粒颗粒清除
        # Ldlrap1 = LDLR 内吞必需的接头蛋白;Lrpap1 = LRP1 的分子伴侣
        "Ldlr", "Vldlr", "Lrp1", "Ldlrap1", "Lrpap1",
    ],
    "Scavenger_SR_B": [
        # B 类清道夫受体:SR-B1 = Scarb1, SR-B2 = Cd36
        "Scarb1", "Cd36",
    ],
    "FA_transport": [
        # Slc27(FATP)溶质载体家族 + 胞内脂肪酸结合蛋白
        "Slc27a1", "Slc27a2", "Slc27a4", "Slc27a5", "Fabp1",
    ],
    # ▼ v4 拆分:原 Efflux 按**外排目的地**一分为二 + 单列胆汁酸合成
    "HDL_efflux": [
        # 胆固醇外排至**血浆 HDL 池**(胆固醇留在体内)
        # Abca1: 外排给贫脂 apoA-I → 生成 nascent HDL,逆向转运限速步
        # Pltp : 磷脂转运蛋白,介导 HDL 颗粒重塑
        # ⚠️ Abcg1 已移除(肝细胞检出率仅 2.3%,见文件头说明)
        "Abca1", "Pltp",
    ],
    "Biliary_excretion": [
        # 毛细胆管膜(canalicular)转运体:把固醇/胆盐/磷脂**泵入胆汁**
        # Abcg5 + Abcg8 为专性异二聚体,负责胆固醇与植物固醇排泄
        # Abcb11(BSEP 胆盐) / Abcb4(MDR3 磷脂) / Abcc2(MRP2 有机阴离子)
        # Atp8b1(FIC1 磷脂翻转酶,维持毛细胆管膜脂不对称性)
        "Abcg5", "Abcg8", "Abcb11", "Abcb4", "Abcc2", "Atp8b1",
    ],
    "Bile_acid_synth": [
        # 胆汁酸合成:Cyp7a1 = 经典途径限速酶, Cyp8b1 = 胆酸/鹅脱氧胆酸分支,
        # Cyp27a1 = 替代途径。与 Biliary_excretion 同属"胆汁分泌轴"的上游
        "Cyp7a1", "Cyp8b1", "Cyp27a1",
    ],
    "Lipid_peroxidation": [
        "Gpx4", "Slc7a11", "Acsl4", "Alox5", "Alox15",
        "Hmox1", "Fth1", "Ftl1", "Tfrc", "Nqo1",
    ],
}

# --------------------------------------------------------------------------
# 5. AUCell 的 numpy 实现
# --------------------------------------------------------------------------
def aucell(X, gene_set_idx, auc_max=None, seed=0):
    """
    AUCell: 对每个细胞, 计算基因集在"该细胞基因表达排名"上的恢复曲线下面积。

    与 AddModuleScore 的区别: AUCell 基于**秩**,不受基因集大小与数据稀疏度
    影响;这也是选它做主打分方法的原因(AddModuleScore 仅作一致性交叉验证)。

    参数
      X             : 细胞 x 基因的 log-normalized 矩阵 (np.ndarray 或 sparse)
      gene_set_idx  : 基因集在矩阵列上的整数索引 (np.ndarray of int)
      auc_max       : 恢复曲线截断位置(默认取基因集大小的 5%, AUCell 惯例
                      top 5% 排名;这里默认 = max(10, 0.05*n_genes) )

    返回
      auc : shape (n_cells,) 的 float32 数组, 取值 0~1
    """
    X = sp.csr_matrix(X) if sp.issparse(X) else np.asarray(X)
    n_cells, n_genes = X.shape
    gs = np.asarray(gene_set_idx, dtype=int)
    gs = gs[(gs >= 0) & (gs < n_genes)]
    if gs.size == 0:
        return np.zeros(n_cells, dtype=np.float32)

    if auc_max is None:
        auc_max = int(max(10, 0.05 * n_genes))
    auc_max = int(min(auc_max, n_genes))

    gs_set = set(gs.tolist())
    auc = np.zeros(n_cells, dtype=np.float32)

    # 逐细胞求秩(按表达从高到低)。
    # 对 6 万细胞 x 3 万基因, 这个循环是主要耗时点;用 argsort 一次性搞定。
    if sp.issparse(X):
        Xd = X.toarray()
    else:
        Xd = X

    # 用 argpartition 加速: 只需要知道每个基因集成员排在前 auc_max 的数量
    # —— 但 AUC 需要**每个**基因集成员的秩(不只是前 auc_max),故直接 argsort。
    order = np.argsort(-Xd, axis=1, kind="stable")   # 降序: 表达高 -> 秩小
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.tile(np.arange(n_genes), (n_cells, 1)), axis=1)

    r = rank[:, gs]                       # (n_cells, n_gs) 各成员的秩
    r_sorted = np.sort(r, axis=1)         # 成员按秩升序
    k = np.arange(1, auc_max + 1)         # 恢复曲线横轴
    # 截断到 auc_max 以内的成员数(向量化)
    n_within = (r_sorted < auc_max).sum(axis=1)
    # AUC = 1/(auc_max * n_gs) * sum_{i} (auc_max - rank_i)_+
    contrib = np.clip(auc_max - r_sorted, 0, None).sum(axis=1)
    auc = (contrib / (auc_max * len(gs))).astype(np.float32)
    del order, rank, Xd
    return auc


def zonation_index(X, gidx, auc_max=None):
    """
    连续区带指数 = AUCell(pericentral) - AUCell(periportal)

    正值偏向中央静脉周(zone3 端),负值偏向门静脉周(zone1 端)。
    在 Sham 组按分位数标定 zone1/2/3 后,把同样的切点套用到 CLP 组
    (保证两组可比;若各自标定,collapse 会被切点移动掩盖)。
    """
    pp = [gidx[g] for g in ZONE["periportal"] if g in gidx]
    pc = [gidx[g] for g in ZONE["pericentral"] if g in gidx]
    if not pp or not pc:
        raise ValueError("区带 landmark 基因全部未匹配, 检查基因名体系")
    return aucell(X, pc, auc_max=auc_max) - aucell(X, pp, auc_max=auc_max), len(pp), len(pc)


# --------------------------------------------------------------------------
# 6. 小工具
# --------------------------------------------------------------------------
def sample_paths(gsm, suffix, raw_dir=RAW_DIR):
    """返回 (barcodes, features, matrix) 三件套路径"""
    return (
        os.path.join(raw_dir, f"{gsm}_barcodes_{suffix}.tsv.gz"),
        os.path.join(raw_dir, f"{gsm}_features_{suffix}.tsv.gz"),
        os.path.join(raw_dir, f"{gsm}_matrix_{suffix}.mtx.gz"),
    )


def knee_call(counts, n_cand=20000, n_bins=200):
    """
    简易 knee / inflection point 细胞识别。

    counts: 每个 barcode 的 UMI 总数 (np.ndarray, 已按降序排好或未排均可)
    做法: 取前 n_cand 个 barcode,在 log(rank) - log(count) 空间上,
          找离"首尾连线"最远的点(Drop-seq / EmptyDrops 的 knee 思想)。

    返回: (阈值 count, 估计细胞数)
    """
    c = np.sort(np.asarray(counts, dtype=float))[::-1]
    c = c[c > 0]
    if c.size == 0:
        return 0.0, 0
    n = min(n_cand, c.size)
    x = np.log10(np.arange(1, n + 1))
    y = np.log10(c[:n] + 1)
    # 首尾连线
    p1, p2 = np.array([x[0], y[0]]), np.array([x[-1], y[-1]])
    v = p2 - p1
    v = v / (np.linalg.norm(v) + 1e-12)
    pts = np.stack([x, y], axis=1) - p1
    # 点到直线距离
    proj = pts @ v
    perp = pts - np.outer(proj, v)
    dist = np.linalg.norm(perp, axis=1)
    k = int(np.argmax(dist))
    thr = float(c[k])
    ncells = int((np.asarray(counts) >= thr).sum())
    return thr, ncells


def detect_rate(X, col):
    """某基因(列索引 col)在细胞上的检出率(非零比例)"""
    if sp.issparse(X):
        v = X[:, col]
        return float((v > 0).sum()) / X.shape[0]
    return float((X[:, col] > 0).sum()) / X.shape[0]
