#!/usr/bin/env bash
# ============================================================================
# 03_fetch_all.sh —— GSE275689 全量取数(barcodes + features + matrix × 6 样本)
#
# 网络实测(2026-08-28 ~ 08-29):
#   - 白天/晚间 NCBI 对单 IP 限流: 单连接 ~12 KB/s;4–8 路 Range 突发会直接
#     触发惩罚(跌到 20–100 B/s),需退避数十分钟才恢复。
#   - 解除限流后可达 ~10 MB/s(GSM8482478 剩余 203 MB 用 21 秒下完)。
#   策略: 单连接顺序下载 + 断点续传 + 卡死退避重试。不赌并发。
#
# 用法:
#   bash 03_fetch_all.sh            # 全部 6 个样本
#   bash 03_fetch_all.sh GSM8482479 # 只下指定样本
#
# 校验: 每个文件下完跑 gzip -t,通过才打 .ok 标记。
#       矩阵额外校验解压后的 nnz 行数与 mtx 头部声明一致。
# ============================================================================
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
RAW="data/raw"
LOG="data/raw/_fetch_all.log"
mkdir -p "$RAW"

BASE="https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8482nnn"

# GSM : 文件名后缀 : 分组
declare -a SAMPLES=(
  "GSM8482478:liver_02:clp"
  "GSM8482479:liver_06:clp"
  "GSM8482480:liver_10:clp"
  "GSM8482481:liver_14:sham"
  "GSM8482482:liver_18:sham"
  "GSM8482483:liver_22:sham"
)

# 每个样本需要的文件: 样式 -> 文件名中间关键字
KINDS=("barcodes:tsv.gz" "features:tsv.gz" "matrix:mtx.gz")

ONLY="${1:-}"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

size_of() { curl -sI "$1" | tr -d '\r' | awk 'tolower($1)=="content-length:"{print $2}' | tail -1; }

# 通用续传: 低速(2KB/s 持续 120s)判卡死 -> 退避 -> 再续传
fetch_file() {
  local url="$1" out="$2"
  local ok="$out.ok"
  [ -f "$ok" ] && { log "  SKIP $(basename "$out") (已就绪)"; return 0; }

  local expect; expect=$(size_of "$url"); [ -z "$expect" ] && expect=0
  local backoff=20 attempt=0

  while [ ! -f "$ok" ]; do
    attempt=$((attempt+1))
    local before after
    before=$(stat -f%z "$out" 2>/dev/null || echo 0)

    curl -sS -L -C - --speed-limit 2000 --speed-time 120 --max-time 3600 \
         -o "$out" "$url"
    after=$(stat -f%z "$out" 2>/dev/null || echo 0)

    if [ "$expect" -gt 0 ] && [ "$after" -ge "$expect" ]; then
      if gzip -t "$out" 2>/dev/null; then
        : > "$ok"; return 0
      fi
      log "  WARN 大小够但 gzip -t 失败(坏块),删除重来: $(basename "$out")"
      rm -f "$out"; backoff=20; continue
    fi

    local pct="?"
    [ "$expect" -gt 0 ] && pct="$(awk -v a="$after" -v b="$expect" 'BEGIN{printf "%.1f",a*100/b}')%"

    if [ "$after" -le "$before" ]; then
      log "  STALL $(basename "$out") 无进展($after B,rc=$?) -> 退避 ${backoff}s"
      sleep "$backoff"
      backoff=$(( backoff*2 )); [ "$backoff" -gt 600 ] && backoff=600
    else
      log "  PART $(basename "$out") ${pct} (+$((after-before)) B)"
      backoff=20; sleep 3
    fi
  done
}

# 矩阵专用深度校验: 解压后数据行数是否等于头部声明的 nnz
verify_matrix() {
  local out="$1"
  local declared actual
  declared=$(gunzip -c "$out" | sed -n '3p' | awk '{print $3}')
  actual=$(gunzip -c "$out" | grep -vc '^%')
  actual=$((actual - 1))   # 减去 dims 行
  if [ "$declared" = "$actual" ]; then
    log "  VERIFY OK nnz: 声明 $declared = 实测 $actual"
  else
    log "  VERIFY FAIL nnz: 声明 $declared != 实测 $actual —— 文件不完整,删除 .ok 重下"
    rm -f "$out.ok"
  fi
}

log "===== 全量取数开始 ====="
for s in "${SAMPLES[@]}"; do
  IFS=: read -r gsm suffix grp <<< "$s"
  [ -n "$ONLY" ] && [ "$ONLY" != "$gsm" ] && continue
  log "--- $gsm ($suffix, $grp) ---"
  for k in "${KINDS[@]}"; do
    kind="${k%%:*}"; ext="${k##*:}"
    url="$BASE/$gsm/suppl/${gsm}_${kind}_${suffix}.${ext}"
    out="$RAW/${gsm}_${kind}_${suffix}.${ext}"
    fetch_file "$url" "$out"
    if [ "$kind" = "matrix" ] && [ -f "$out.ok" ]; then verify_matrix "$out"; fi
  done
done
log "===== 全部完成 ====="
echo; echo "=== 最终清单 ==="; ls -la "$RAW" | grep -vE "^total|_fetch"
