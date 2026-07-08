###############################################################################
# Makefile  —  SIMD 高斯消元向量化实验统一构建系统
###############################################################################
# 使用方式:
#   make all              — 编译全部可执行文件
#   make standard         —  仅编译标准高斯消元部分
#   make special          —  仅编译 GF(2) Gröbner 基部分
#   make run              — 串行运行全部已编译文件
#   make run-batch        — 运行后生成 CSV 结果汇总
#   make perf-all         — 使用 perf stat 采集全部目标的硬件计数器
#   make perf-target T=<name> — 单独采集某个目标 (如 T=avx_aligned)
#   make clean            — 清除编译产物
#   make check-cpu        — 检测当前 CPU 支持的 SIMD 指令集
#
# 编译器选择:
#   CXX=g++   (默认, GCC)
#   CXX=clang++  (LLVM/Clang)
#   CXX=aarch64-linux-gnu-g++  (ARM 交叉编译)
###############################################################################

CXX      ?= g++
CXXFLAGS  = -O2 -march=native -std=c++17
LDFLAGS   =

# ---- x86 SIMD 特性标志 (由 check-cpu 目标按需启用) ----
SSE_FLAGS     = -msse4.2
AVX_FLAGS     = -mavx -mavx2
AVX512_FLAGS  = -mavx512f
FMA_FLAGS     = -mfma

# ---- ARM 交叉编译特定标志 ----
ARM_CXX      ?= aarch64-linux-gnu-g++
ARM_CXXFLAGS  = -O2 -march=armv8-a+simd -std=c++17

# ---- 可执行文件输出目录 ----
BIN_DIR     = bin
COMMON_DIR  = common
STD_DIR     = standard_gauss
SPC_DIR     = special_gauss

# ---- 创建输出目录 ----
$(shell mkdir -p $(BIN_DIR) 2>/dev/null)

###############################################################################
# 标准高斯消元目标列表
###############################################################################
STD_TARGETS = \
    $(BIN_DIR)/serial_naive        \
    $(BIN_DIR)/serial_cache_opt    \
    $(BIN_DIR)/sse_aligned         \
    $(BIN_DIR)/sse_unaligned       \
    $(BIN_DIR)/avx_aligned         \
    $(BIN_DIR)/avx_unaligned       \
    $(BIN_DIR)/avx512_aligned      \
    $(BIN_DIR)/avx512_unaligned    \
    $(BIN_DIR)/expanded_test

# ARM NEON 目标 (仅在 ARM 平台或交叉编译时构建)
NEON_TARGETS = \
    $(BIN_DIR)/neon_aligned        \
    $(BIN_DIR)/neon_unaligned

# GF(2) Gröbner 基目标
GROEBNER_TARGETS = \
    $(BIN_DIR)/gf2_eliminate_x86   \
    $(BIN_DIR)/gf2_eliminate_arm

###############################################################################
# 顶层目标
###############################################################################
.PHONY: all standard special run run-batch clean check-cpu perf-all

all: standard

standard: $(STD_TARGETS)
	@echo "[DONE] Standard Gaussian elimination targets built."

special: $(GROEBNER_TARGETS)
	@echo "[DONE] GF(2) Groebner basis targets built."

# ARM 平台额外构建 NEON
neon: $(NEON_TARGETS)
	@echo "[DONE] ARM NEON targets built."

###############################################################################
# 标准高斯消元 — 串行
###############################################################################
$(BIN_DIR)/serial_naive: $(STD_DIR)/serial_naive.cpp $(COMMON_DIR)/matrix_utils.h $(COMMON_DIR)/benchmark_utils.h
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

$(BIN_DIR)/serial_cache_opt: $(STD_DIR)/serial_cache_opt.cpp $(COMMON_DIR)/matrix_utils.h $(COMMON_DIR)/benchmark_utils.h
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

###############################################################################
# 标准高斯消元 — SSE
###############################################################################
$(BIN_DIR)/sse_aligned: $(STD_DIR)/sse_aligned.cpp $(COMMON_DIR)/matrix_utils.h $(COMMON_DIR)/benchmark_utils.h
	$(CXX) $(CXXFLAGS) $(SSE_FLAGS) $< -o $@ $(LDFLAGS)

$(BIN_DIR)/sse_unaligned: $(STD_DIR)/sse_unaligned.cpp $(COMMON_DIR)/matrix_utils.h $(COMMON_DIR)/benchmark_utils.h
	$(CXX) $(CXXFLAGS) $(SSE_FLAGS) $< -o $@ $(LDFLAGS)

###############################################################################
# 标准高斯消元 — AVX
###############################################################################
$(BIN_DIR)/avx_aligned: $(STD_DIR)/avx_aligned.cpp $(COMMON_DIR)/matrix_utils.h $(COMMON_DIR)/benchmark_utils.h
	$(CXX) $(CXXFLAGS) $(AVX_FLAGS) $< -o $@ $(LDFLAGS)

$(BIN_DIR)/avx_unaligned: $(STD_DIR)/avx_unaligned.cpp $(COMMON_DIR)/matrix_utils.h $(COMMON_DIR)/benchmark_utils.h
	$(CXX) $(CXXFLAGS) $(AVX_FLAGS) $< -o $@ $(LDFLAGS)

###############################################################################
# 标准高斯消元 — AVX-512
###############################################################################
$(BIN_DIR)/avx512_aligned: $(STD_DIR)/avx512_aligned.cpp $(COMMON_DIR)/matrix_utils.h $(COMMON_DIR)/benchmark_utils.h
	$(CXX) $(CXXFLAGS) $(AVX512_FLAGS) $< -o $@ $(LDFLAGS)

$(BIN_DIR)/avx512_unaligned: $(STD_DIR)/avx512_unaligned.cpp $(COMMON_DIR)/matrix_utils.h $(COMMON_DIR)/benchmark_utils.h
	$(CXX) $(CXXFLAGS) $(AVX512_FLAGS) $< -o $@ $(LDFLAGS)

###############################################################################
# 扩展测试
###############################################################################
$(BIN_DIR)/expanded_test: $(STD_DIR)/expanded_test.cpp $(COMMON_DIR)/matrix_utils.h $(COMMON_DIR)/benchmark_utils.h
	$(CXX) $(CXXFLAGS) $(SSE_FLAGS) $(FMA_FLAGS) $< -o $@ $(LDFLAGS)

###############################################################################
# ARM NEON (本地编译或交叉编译)
###############################################################################
$(BIN_DIR)/neon_aligned: $(STD_DIR)/neon_aligned.cpp $(COMMON_DIR)/matrix_utils.h $(COMMON_DIR)/benchmark_utils.h
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

$(BIN_DIR)/neon_unaligned: $(STD_DIR)/neon_unaligned.cpp $(COMMON_DIR)/matrix_utils.h $(COMMON_DIR)/benchmark_utils.h
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

###############################################################################
# GF(2) Gröbner 基
###############################################################################
$(BIN_DIR)/gf2_eliminate_x86: $(SPC_DIR)/gf2_eliminate_x86.cpp
	$(CXX) $(CXXFLAGS) $(SSE_FLAGS) $(AVX_FLAGS) $(AVX512_FLAGS) $< -o $@ $(LDFLAGS)

$(BIN_DIR)/gf2_eliminate_arm: $(SPC_DIR)/gf2_eliminate_arm.cpp
	$(ARM_CXX) $(ARM_CXXFLAGS) $< -o $@ $(LDFLAGS)

###############################################################################
# 运行
###############################################################################
run:
	@echo "=== Running all standard benchmarks ==="
	@for exe in $(STD_TARGETS); do \
		if [ -f "$$exe" ] || [ -f "$$exe.exe" ]; then \
			echo "--- $$exe ---"; \
			$$exe; \
		fi; \
	done

run-batch:
	@echo "dim,impl,time_ms" > results.csv
	@for exe in $(STD_TARGETS); do \
		if [ -f "$$exe" ] || [ -f "$$exe.exe" ]; then \
			$$exe 2>&1 | grep -E '^\s+[0-9]+' | \
			awk -v impl="$$(basename $$exe)" '{print $$1","impl","$$2}' >> results.csv; \
		fi; \
	done
	@echo "[DONE] Results written to results.csv"

###############################################################################
# perf 性能分析
###############################################################################
PERF_EVENTS = cycles,instructions,cache-references,cache-misses,\
              branches,branch-misses,FP_COMP_OPS_EXE:SSE_FP,\
              L1-dcache-load-misses,LLC-load-misses,\
              cpu-clock,task-clock

perf-all:
	@for exe in $(STD_TARGETS); do \
		if [ -f "$$exe" ] || [ -f "$$exe.exe" ]; then \
			echo "========================================"; \
			echo " perf stat: $$(basename $$exe)"; \
			echo "========================================"; \
			perf stat -e $(PERF_EVENTS) $$exe 2>&1; \
		fi; \
	done

perf-target:
	@if [ -z "$(T)" ]; then \
		echo "Usage: make perf-target T=<target_name>"; \
		echo "  e.g. make perf-target T=avx_aligned"; \
		exit 1; \
	fi
	perf stat -e $(PERF_EVENTS) $(BIN_DIR)/$(T)

perf-record:
	@if [ -z "$(T)" ]; then \
		echo "Usage: make perf-record T=<target_name>"; \
		exit 1; \
	fi
	perf record -e $(PERF_EVENTS) $(BIN_DIR)/$(T)
	perf report

###############################################################################
# CPU 特性检测
###############################################################################
check-cpu:
	@echo "=== CPU Info ==="
	@echo "Model name:"
	@grep -m1 "model name" /proc/cpuinfo 2>/dev/null || sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "N/A"
	@echo ""
	@echo "SIMD Flags:"
	@grep -o '\<\(sse\|sse2\|sse3\|ssse3\|sse4_1\|sse4_2\|avx\|avx2\|avx512f\|avx512cd\|avx512dq\|avx512bw\|avx512vl\|fma\|neon\|asimd\)\>' /proc/cpuinfo 2>/dev/null | sort -u || echo "N/A (non-Linux)"
	@echo ""
	@echo "Cache Info:"
	@grep -E "cache size" /proc/cpuinfo 2>/dev/null | sort -u || echo "N/A"

###############################################################################
# 清理
###############################################################################
clean:
	rm -rf $(BIN_DIR)/*.exe $(BIN_DIR)/* 2>/dev/null || true
	rm -f results.csv 2>/dev/null || true
	@echo "[CLEAN] Build artifacts removed."

###############################################################################
.PHONY: all standard special neon run run-batch clean check-cpu
.PHONY: perf-all perf-target perf-record
