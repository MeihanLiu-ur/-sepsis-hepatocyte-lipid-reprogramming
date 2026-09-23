#!/usr/bin/env bash
# ============================================================================
# 02_fetch_polite.sh —— GSE275689 礼貌型断点续传下载器
#
# 背景: NCBI ftp 对单 IP 有速率配额。实测:
#   - 单连接顺序下载   ~12 KB/s
#   - 4 路 Range 突发  首轮 ~110 KB/s,但立刻触发限流,之后跌到 ~40 B/s
#   - 8 路 Range 突发  直接被掐到 ~100 B/s
#   结论: 不要贪并发。走单连接 + 断点续传 + 卡死后退避重试,靠时间取胜。
#
# 用法:
#   bash 02_fetch_polite.sh            # 默认只下第一个样本(先过 G1 闸门)
#   bash 02_fetch_polite.sh all        # 下全部 6 个样本
#
# 产物: data/raw/<GSM>_matrix_<suffix>.mtx.gz,通过 gzip -t 校验后打 .ok 标记
# ============================================================================
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
RAW="data/raw"
LOG="data/raw/_fetch.log"
mkdir -p "$RAW"

BASE="https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8482nnn"

# GSM : 文件名后缀 : 分组 : 预期大小(字节)
declare -a SAMPLES=(
  "GSM8482478:liver_02:clp:251911028"
  "GSM8482479:liver_06:clp:0"
  "GSM8482480:liver_10:clp:0"
  "GSM8482481:liver_14:sham:0"
  "GSM8482482:liver_18:sham:0"
  "GSM8482483:liver_22:sham:0"
)

MODE="${1:-first}"
[ "$MODE" = "all" ] || SAMPLES=( "${SAMPLES[0]}" )

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# 单个样本的续传循环
# 策略: 低速(低于 2 KB/s 持续 120s)判定为卡死 -> 掐断 -> 退避 -> 再续传
fetch_one() {
  local gsm="$1" suffix="$2" grp="$3" expect="$4"
  local url="$BASE/$gsm/suppl/${gsm}_matrix_${suffix}.mtx.gz"
  local out="$RAW/${gsm}_matrix_${suffix}.mtx.gz"
  local ok="$out.ok"

  [ -f "$ok" ] && { log "SKIP $gsm (已完成)"; return 0; }

  # 抓真实大小(用于进度百分比)
  if [ "$expect" = "0" ]; then
    expect=$(curl -sI "$url" | tr -d '\r' | awk 'tolower($1)=="content-length:"{print $2}' | tail -1)
    [ -z "$expect" ] && expect=0
  fi

  local backoff=20
  local attempt=0
  while [ ! -f "$ok" ]; do
    attempt=$((attempt+1))
    local before after
    before=$(stat -f%z "$out" 2>/dev/null || echo 0)

    # --speed-limit/--speed-time: 连续 120s 低于 2KB/s 就放弃本次连接(不是放弃任务)
    curl -sS -L -C - \
         --speed-limit 2000 --speed-time 120 \
         --max-time 1800 \
         -o "$out" "$url"
    rc=$?

    after=$(stat -f%z "$out" 2>/dev/null || echo 0)

    if [ "$expect" -gt 0 ] && [ "$after" -ge "$expect" ]; then
      if gzip -t "$out" 2>/dev/null; then
        : > "$ok"
        log "DONE $gsm ($grp) ${after}/${expect} bytes, gzip OK, 第 $attempt 次连接"
        return 0
      else
        log "WARN $gsm 大小已够但 gzip -t 失败,判定为坏块,删除重来"
        rm -f "$out"
        backoff=20
        continue
      fi
    fi

    local pct="?"
    [ "$expect" -gt 0 ] && pct="$(awk -v a="$after" -v b="$expect" 'BEGIN{printf "%.1f", a*100/b}')%"

    if [ "$after" -le "$before" ]; then
      log "STALL $gsm 无进展 before=$before after=$after (rc=$rc) -> 退避 ${backoff}s"
      sleep "$backoff"
      backoff=$(( backoff * 2 )); [ "$backoff" -gt 600 ] && backoff=600
    else
      log "PART $gsm ${after}/${expect} (${pct}) 本轮 +$((after-before)) bytes -> 稍候 5s 续传"
      backoff=20
      sleep 5
    fi
  done
}

log "===== 开始下载 (mode=$MODE) ====="
for s in "${SAMPLES[@]}"; do
  IFS=: read -r gsm suffix grp expect <<< "$s"
  fetch_one "$gsm" "$suffix" "$grp" "$expect"
done
log "===== 全部完成 ====="
