"""
58_program_effect_ds1500.py
============================
生成 **降采样口径** 的程序层效应表，供 Fig 3a 使用。

为什么需要这个文件
-------------------
稿件 Methods 明确写了 "All program-level conclusions use depth-corrected
values."。但 Fig 3a 原本读的是 `fig3_program_effect_v2.csv` —— 那是**原生
深度**的结果，包括其中的 `separation` 列。于是出现口径错配：

    正文数字  → 降采样 (ds_pct)
    图上散点  → 原生 (CLP_1..3 / Sham_1..2)
    图上星号  → 原生 (separation)

后果不只是数字对不上 —— 原生 6/12 完全分离、降采样 **9/12** 完全分离，
正文因此少报了 3 条（FAO / Lipoprotein_assembly / Lipid_peroxidation）。
更糟的是，若只把标记换成降采样而散点仍画原生，读者会看到"散点明明重叠
却标着完全分离"的自相矛盾。

因此本脚本产出与 `fig3_program_effect_v2.csv` **同结构** 的降采样版本，
让 Fig 3a 的散点、数字、分离标记三者同源（均为 depth-corrected）。
原生深度的对应信息由 Fig 3c 单独呈现（原生 vs 降采样散点对比），
信息不丢失。

输出
----
data/proc/fig3_program_effect_ds1500.csv
    列结构与 v2 完全一致:
    sham_mean, clp_mean, pct_change, CLP_1..3, Sham_1..2,
    separation, perm_p(双尾), perm_min_p, cell_AUC

data/proc/fig3_program_perm_ds1500.csv  (若不存在则写)
    原生 vs 降采样 的并排对照, 供正文/回复审稿人时核对。
"""
import sys
import types
import os

# --- scanpy shim: 本环境无 scanpy, 只用其 read_h5ad -----------------------
_m = types.ModuleType('scanpy')
from anndata import read_h5ad           # noqa: E402
_m.read_h5ad = read_h5ad
_m.__version__ = "0.0-shim"
sys.modules['scanpy'] = _m

import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402
from itertools import combinations       # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from lib_sepsis import LIPID_PROGRAMS, PROC_DIR   # noqa: E402
from _aucell_common import score_all              # noqa: E402

import importlib.util                             # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "ds56", os.path.join(ROOT, 'scripts', '56_split_depthcheck.py'))
ds56 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ds56)          # 只加载定义, 不执行 main()

TARGET = 1500


def sep_label(c, sh):
    """完全分离的两种方向。"""
    if c[0] > sh[-1]:
        return "完全分离(CLP 全高于 Sham)"
    if c[-1] < sh[0]:
        return "完全分离(CLP 全低于 Sham)"
    return "有重叠"


def perm_two_sided(vals, clp_idx, sham_idx):
    """C(5,3)=10 种分组的置换检验, 返回 (obs_diff, 双尾p, 最小可达p)。"""
    obs = vals[clp_idx].mean() - vals[sham_idx].mean()
    diffs = []
    for combo in combinations(range(len(vals)), len(clp_idx)):
        g1 = [vals[i] for i in combo]
        g2 = [vals[i] for i in range(len(vals)) if i not in combo]
        diffs.append(np.mean(g1) - np.mean(g2))
    diffs = np.array(diffs)
    p_two = float((np.abs(diffs) >= abs(obs) - 1e-12).sum() / len(diffs))
    return float(obs), p_two, 1.0 / len(diffs)


def per_sample_table(D, samp, clp_s, sham_s):
    """把细胞级程序打分压成 每样本均值 -> 与 v2 同结构的效应表。"""
    per = pd.DataFrame(D).groupby(samp).mean()
    samples = list(clp_s) + list(sham_s)
    clp_idx = np.arange(len(clp_s))
    sham_idx = np.arange(len(clp_s), len(samples))
    rows = {}
    for p in LIPID_PROGRAMS:
        vals = np.array([float(per.loc[s, p]) for s in samples])
        obs, p_two, p_min = perm_two_sided(vals, clp_idx, sham_idx)
        c = sorted(vals[clp_idx])
        sh = sorted(vals[sham_idx])
        sh_mean = float(vals[sham_idx].mean())
        cl_mean = float(vals[clp_idx].mean())
        row = {
            "sham_mean": sh_mean,
            "clp_mean": cl_mean,
            "pct_change": obs / sh_mean * 100.0,
            "separation": sep_label(c, sh),
            "perm_p": p_two,
            "perm_min_p": p_min,
            "cell_AUC": float(np.asarray(D[p]).mean()),
        }
        for k, s in enumerate(clp_s, 1):
            row[f"CLP_{k}"] = float(per.loc[s, p])
        for k, s in enumerate(sham_s, 1):
            row[f"Sham_{k}"] = float(per.loc[s, p])
        rows[p] = row
    T = pd.DataFrame(rows).T
    cols = (["sham_mean", "clp_mean", "pct_change"]
            + [f"CLP_{k}" for k in range(1, len(clp_s) + 1)]
            + [f"Sham_{k}" for k in range(1, len(sham_s) + 1)]
            + ["separation", "perm_p", "perm_min_p", "cell_AUC"])
    return T[cols].astype(float, errors="ignore")


def main():
    hep = read_h5ad(os.path.join(PROC_DIR, 'hep_consensus.h5ad'))
    grp = hep.obs['group'].values
    samp = hep.obs['sample'].values
    clp_s = sorted(set(samp[grp == 'CLP']))
    sham_s = sorted(set(samp[grp != 'CLP']))
    genes = list(hep.var_names)
    print(f"肝细胞 {hep.n_obs:,}  基因 {hep.n_vars:,}")
    print(f"CLP : {clp_s}")
    print(f"Sham: {sham_s}")

    print("原生深度打分 ...", flush=True)
    nat = ds56.score(hep.X.tocsr(), genes)
    print(f"降采样到 {TARGET} UMI 后打分 ...", flush=True)
    dwn = ds56.score(ds56.downsample_to(hep, TARGET), genes)

    T_nat = per_sample_table(nat, samp, clp_s, sham_s)
    T_dwn = per_sample_table(dwn, samp, clp_s, sham_s)

    # --- 并排对照(供正文/审稿回复核对) --------------------------------
    cmp = pd.DataFrame({
        "program": T_nat.index,
        "native_pct": T_nat["pct_change"].round(2).values,
        "native_sep": T_nat["separation"].values,
        "native_p_two": T_nat["perm_p"].values,
        "ds_pct": T_dwn["pct_change"].round(2).values,
        "ds_sep": T_dwn["separation"].values,
        "ds_p_two": T_dwn["perm_p"].values,
    })
    pd.set_option('display.width', 260)
    print()
    print("=" * 118)
    print(f"原生 vs 降采样 {TARGET} UMI —— 分离状态与双尾置换 p (C(5,3)=10 种分组)")
    print("=" * 118)
    print(cmp.to_string(index=False))
    n_nat = int(T_nat["separation"].str.startswith("完全").sum())
    n_dwn = int(T_dwn["separation"].str.startswith("完全").sum())
    print()
    print(f"原生   完全分离: {n_nat}/12   (双尾 p 最小 {T_nat['perm_p'].min():.3f})")
    print(f"降采样 完全分离: {n_dwn}/12   (双尾 p 最小 {T_dwn['perm_p'].min():.3f})")
    flip = [p for p in T_nat.index
            if T_nat.loc[p, "separation"].startswith("完全")
            != T_dwn.loc[p, "separation"].startswith("完全")]
    if flip:
        print(f"口径切换后分离状态改变的程序: {flip}")

    out1 = os.path.join(PROC_DIR, 'fig3_program_effect_ds1500.csv')
    out2 = os.path.join(PROC_DIR, 'fig3_program_perm_ds1500.csv')
    T_dwn.round(6).to_csv(out1)
    cmp.to_csv(out2, index=False)
    print(f"\n[save] {out1}")
    print(f"[save] {out2}")


if __name__ == "__main__":
    main()
