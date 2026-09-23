#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 01_download.sh -- C1/S1  主数据下载 (GSE275689, snRNA, 6 样本)
#
# 文件名后缀已逐样本从 GEO SOFT 核实, 勿臆造:
#   CLP  rep1-3 : GSM8482478 liver_02 | GSM8482479 liver_06 | GSM8482480 liver_10
#   Sham rep1-3 : GSM8482481 liver_14 | GSM8482482 liver_18 | GSM8482483 liver_22
#
# 用法:  bash scripts/01_download.sh             # 下载全部
#        bash scripts/01_download.sh sham        # 只下 Sham 三个
#        bash scripts/01_download.sh clp         # 只下 CLP 三个
#
# 断点续传: 使用 curl -C -, 中断后重跑即可, 不会重复下载已完成部分
# ---------------------------------------------------------------------------
set -euo pipefail

BASE="https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8482nnn"
OUT="data/raw"
mkdir -p "$OUT"
cd "$OUT"

TARGET="${1:-all}"

declare -a SAMPLES=(
  "GSM8482478:liver_02:clp"
  "GSM8482479:liver_06:clp"
  "GSM8482480:liver_10:clp"
  "GSM8482481:liver_14:sham"
  "GSM8482482:liver_18:sham"
  "GSM8482483:liver_22:sham"
)

for item in "${SAMPLES[@]}"; do
  IFS=':' read -r gsm suffix group <<< "$item"

  if [[ "$TARGET" != "all" && "$TARGET" != "$group" ]]; then
    continue
  fi

  echo "------------------------------------------------------------"
  echo "[download] $gsm  ($group)  suffix=$suffix"

  # Cell Ranger 三件套. 注意 matrix 用的是 .mtx.gz 而非 .tsv.gz
  curl -C - --retry 3 --retry-delay 5 -# -L \
    -o "${gsm}_barcodes_${suffix}.tsv.gz" \
    "$BASE/$gsm/suppl/${gsm}_barcodes_${suffix}.tsv.gz"

  curl -C - --retry 3 --retry-delay 5 -# -L \
    -o "${gsm}_features_${suffix}.tsv.gz" \
    "$BASE/$gsm/suppl/${gsm}_features_${suffix}.tsv.gz"

  curl -C - --retry 3 --retry-delay 5 -# -L \
    -o "${gsm}_matrix_${suffix}.mtx.gz" \
    "$BASE/$gsm/suppl/${gsm}_matrix_${suffix}.mtx.gz"
done

echo "------------------------------------------------------------"
echo "[done] 文件清单:"
ls -lh *.tsv.gz *.mtx.gz 2>/dev/null || echo "(无文件, 检查网络)"

# ---------------------------------------------------------------------------
# 校验: 三个文件的行数/gzip 完整性
# ---------------------------------------------------------------------------
echo
echo "[verify] gzip 完整性检查"
for f in *.tsv.gz *.mtx.gz; do
  [[ -f "$f" ]] || continue
  if gzip -t "$f" 2>/dev/null; then
    echo "  OK   $f"
  else
    echo "  BAD  $f  <-- 下载不完整, 请重跑本脚本 (curl -C - 会续传)"
  fi
done
