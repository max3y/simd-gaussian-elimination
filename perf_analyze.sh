#!/usr/bin/env bash
###############################################################################
# perf_analyze.sh  —  SIMD 高斯消元性能深度分析脚本
###############################################################################
# 功能:
#   1. 批量运行全部实现并收集 perf stat 硬件计数器
#   2. 生成 CSV 格式的汇总报告 (可直接导入 Python/R/Excel)
#   3. 可选生成逐函数的 perf record + 火焰图数据
#   4. 支持 Linux perf 和 Apple Instruments (macOS) 两种后端
#
# 使用:
#   chmod +x perf_analyze.sh
#   ./perf_analyze.sh              # 完整分析
#   ./perf_analyze.sh --quick      # 快速模式 (仅 cycles,instructions)
#   ./perf_analyze.sh --flamegraph # 生成火焰图 (需 FlameGraph 脚本)
#   ./perf_analyze.sh --target avx_aligned  # 单目标深度分析
###############################################################################

set -euo pipefail

BIN_DIR="./bin"
OUT_DIR="./perf_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_LOG="${OUT_DIR}/perf_log_${TIMESTAMP}.txt"
CSV_OUT="${OUT_DIR}/perf_summary_${TIMESTAMP}.csv"

# ---- 硬件计数器事件集 ----
EVENTS_BASIC="cycles,instructions,cache-references,cache-misses"
EVENTS_ADVANCED="L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses,\
dTLB-loads,dTLB-load-misses,branches,branch-misses"
EVENTS_SIMD="fp_arith_inst_retired.128b_packed_single,\
fp_arith_inst_retired.256b_packed_single,\
fp_arith_inst_retired.512b_packed_single"
EVENTS_FULL="${EVENTS_BASIC},${EVENTS_ADVANCED},${EVENTS_SIMD}"

# ---- 待测目标列表 ----
TARGETS=(
    "serial_naive"
    "serial_cache_opt"
    "sse_aligned"
    "sse_unaligned"
    "avx_aligned"
    "avx_unaligned"
    "avx512_aligned"
    "avx512_unaligned"
    "expanded_test"
)

# ---- 颜色输出 ----
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log_info()  { echo -e "${GREEN}[INFO]${NC}  $*" | tee -a "$RUN_LOG"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*" | tee -a "$RUN_LOG"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" | tee -a "$RUN_LOG"; }
log_perf()  { echo -e "${CYAN}[PERF]${NC} $*" | tee -a "$RUN_LOG"; }

###############################################################################
# 环境检测
###############################################################################
detect_env() {
    log_info "Detecting execution environment..."

    mkdir -p "$OUT_DIR"

    if command -v perf &>/dev/null; then
        log_info "perf: $(perf --version | head -1)"
        PERF_BACKEND="linux-perf"
    elif command -v xcrun &>/dev/null && xcrun xctrace version &>/dev/null 2>&1; then
        log_warn "Linux perf not found, using macOS xctrace (limited events)"
        PERF_BACKEND="macos-xctrace"
    else
        log_error "No supported performance analysis backend found!"
        log_error "Install linux-tools-common (APT) or perf (YUM)"
        exit 1
    fi

    # CPU 信息
    if [ -f /proc/cpuinfo ]; then
        log_info "CPU: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)"
        log_info "Cores: $(grep -c processor /proc/cpuinfo)"
    elif command -v sysctl &>/dev/null; then
        log_info "CPU: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'Unknown')"
    fi

    echo "Timestamp: ${TIMESTAMP}" > "$RUN_LOG"
    echo "Backend: ${PERF_BACKEND}" >> "$RUN_LOG"
    echo "" >> "$RUN_LOG"
}

###############################################################################
# 单目标 perf stat
###############################################################################
run_perf_stat() {
    local exe_name="$1"
    local exe_path="${BIN_DIR}/${exe_name}"

    if [ ! -f "$exe_path" ] && [ ! -f "${exe_path}.exe" ]; then
        # 尝试 .exe 后缀 (Windows cross)
        if [ -f "${exe_path}.exe" ]; then
            exe_path="${exe_path}.exe"
        else
            log_warn "Skipping ${exe_name}: binary not found"
            echo "${exe_name},N/A,N/A,N/A,N/A,N/A,N/A" >> "$CSV_OUT"
            return
        fi
    fi

    log_perf "Profiling: ${exe_name}"

    if [ "$PERF_BACKEND" = "linux-perf" ]; then
        # Linux perf stat — 解析输出
        local perf_out
        perf_out=$(perf stat -e "$EVENTS_FULL" "$exe_path" 2>&1) || true

        # 提取关键指标
        local cycles=$(echo "$perf_out" | grep -oP '[\d,]+(?=\s+cycles)' | head -1 | tr -d ',')
        local instrs=$(echo "$perf_out" | grep -oP '[\d,]+(?=\s+instructions)' | head -1 | tr -d ',')
        local cache_ref=$(echo "$perf_out" | grep -oP '[\d,]+(?=\s+cache-references)' | head -1 | tr -d ',')
        local cache_miss=$(echo "$perf_out" | grep -oP '[\d,]+(?=\s+cache-misses)' | head -1 | tr -d ',')
        local branches=$(echo "$perf_out" | grep -oP '[\d,]+(?=\s+branches)' | head -1 | tr -d ',')
        local br_miss=$(echo "$perf_out" | grep -oP '[\d,]+(?=\s+branch-misses)' | head -1 | tr -d ',')

        cycles=${cycles:-0}
        instrs=${instrs:-0}
        cache_ref=${cache_ref:-0}
        cache_miss=${cache_miss:-0}
        branches=${branches:-0}
        br_miss=${br_miss:-0}

        # 计算衍生指标
        local ipc=0
        local cache_miss_pct=0
        local br_miss_pct=0
        [ "$cycles" -gt 0 ] && ipc=$(echo "scale=4; $instrs / $cycles" | bc)
        [ "$cache_ref" -gt 0 ] && cache_miss_pct=$(echo "scale=2; 100 * $cache_miss / $cache_ref" | bc)
        [ "$branches" -gt 0 ] && br_miss_pct=$(echo "scale=2; 100 * $br_miss / $branches" | bc)

        echo "${exe_name},${cycles},${instrs},${ipc},${cache_miss_pct},${br_miss_pct},OK" >> "$CSV_OUT"

        # 输出摘要
        log_info "  Cycles: $(printf "%'d" $cycles)"
        log_info "  IPC: ${ipc}"
        log_info "  Cache Miss: ${cache_miss_pct}%"
        log_info "  Branch Miss: ${br_miss_pct}%"
    else
        # macOS fallback: 简单计时
        local start_ns end_ns elapsed
        start_ns=$(date +%s%N 2>/dev/null || python3 -c 'import time; print(int(time.time()*1e9))')
        "$exe_path" > /dev/null 2>&1 || true
        end_ns=$(date +%s%N 2>/dev/null || python3 -c 'import time; print(int(time.time()*1e9))')
        elapsed=$(echo "scale=3; ($end_ns - $start_ns) / 1000000" | bc)
        echo "${exe_name},N/A,N/A,N/A,N/A,N/A,OK(macOS-${elapsed}ms)" >> "$CSV_OUT"
        log_info "  Elapsed: ${elapsed} ms (macOS — no perf counters)"
    fi
}

###############################################################################
# 火焰图生成 (Linux only)
###############################################################################
generate_flamegraph() {
    local exe_name="$1"
    local exe_path="${BIN_DIR}/${exe_name}"

    if [ ! -f "$exe_path" ]; then
        log_warn "Flamegraph: ${exe_name} not found, skipping"
        return
    fi

    log_perf "Recording flamegraph data for ${exe_name}..."

    local data_file="${OUT_DIR}/${exe_name}_perf.data"
    local folded_file="${OUT_DIR}/${exe_name}_folded.txt"
    local svg_file="${OUT_DIR}/${exe_name}_flamegraph.svg"

    # 检查 FlameGraph 脚本是否可用
    local FG_DIR="${FLAMEGRAPH_DIR:-/opt/FlameGraph}"
    if [ ! -d "$FG_DIR" ]; then
        log_warn "FlameGraph scripts not found at ${FG_DIR}"
        log_warn "Clone: git clone https://github.com/brendangregg/FlameGraph.git /opt/FlameGraph"
        return
    fi

    perf record -F 99 -g -o "$data_file" "$exe_path"
    perf script -i "$data_file" | "${FG_DIR}/stackcollapse-perf.pl" > "$folded_file"
    "${FG_DIR}/flamegraph.pl" "$folded_file" > "$svg_file"

    log_info "Flamegraph saved: ${svg_file}"
}

###############################################################################
# 汇总报告
###############################################################################
print_summary() {
    echo ""
    echo "================================================================================"
    echo " PERFORMANCE ANALYSIS SUMMARY"
    echo "================================================================================"
    echo "Timestamp: ${TIMESTAMP}"
    echo "Results  : ${CSV_OUT}"
    echo "Full log : ${RUN_LOG}"
    echo ""
    echo "--- Top Level Metrics ---"
    column -t -s',' "$CSV_OUT" 2>/dev/null || cat "$CSV_OUT"
    echo "================================================================================"
}

###############################################################################
# 主流程
###############################################################################
MODE="full"
TARGET_SINGLE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick)
            MODE="quick"
            EVENTS_FULL="${EVENTS_BASIC}"
            shift
            ;;
        --flamegraph)
            MODE="flamegraph"
            shift
            ;;
        --target)
            TARGET_SINGLE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--quick] [--flamegraph] [--target <name>]"
            echo "  --quick       Basic counters only (cycles, instructions, cache)"
            echo "  --flamegraph  Generate flame graph SVGs"
            echo "  --target <T>  Profile single target only"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# 主入口
detect_env

# CSV 表头
echo "target,cycles,instructions,ipc,cache_miss_pct,branch_miss_pct,status" > "$CSV_OUT"

if [ -n "$TARGET_SINGLE" ]; then
    run_perf_stat "$TARGET_SINGLE"
    if [ "$MODE" = "flamegraph" ]; then
        generate_flamegraph "$TARGET_SINGLE"
    fi
else
    for tgt in "${TARGETS[@]}"; do
        run_perf_stat "$tgt"
    done
fi

print_summary

log_info "Analysis complete. Results: ${OUT_DIR}/"
