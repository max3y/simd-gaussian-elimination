/**
 * ===========================================================================
 * gf2_eliminate_arm.cpp  —  GF(2) 域上 Gröbner 基高斯消元 (ARM NEON)
 * ===========================================================================
 *
 * 算法概述:
 *   在 ARM 平台上利用 NEON 128-bit SIMD 指令集加速 GF(2) 域上的
 *   Gröbner 基消元计算. 每条 NEON 向量指令处理 4 个 32-bit 位向量
 *   元素 (共 128 列), 使用 XOR 实现 GF(2) 域上的消去操作.
 *
 * NEON intrinsics:
 *   vld1q_u32(addr)          — 从对齐地址加载 4 个 uint32_t 到 Q 寄存器
 *   vst1q_u32(addr, vec)     — 将 Q 寄存器的 4 个 uint32_t 存储到对齐地址
 *   veorq_u32(a, b)          — 128-bit 逐位异或 (GF(2) 加法)
 *
 * 平台适配:
 *   鲲鹏 920 (ARMv8.2) / Apple M1-M3 / 树莓派 4-5 / 高通骁龙
 *
 * 编译:
 *   g++ -O2 -march=armv8-a+simd -std=c++17 gf2_eliminate_arm.cpp \
 *       -o gf2_eliminate_arm.exe
 *
 * 对齐编译:
 *   g++ -O2 -march=armv8-a+simd -DUSE_ALIGNMENT -std=c++17 \
 *       gf2_eliminate_arm.cpp -o gf2_eliminate_arm_aligned.exe
 * ===========================================================================
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <string>
#include <cstring>
#include <chrono>
#include <arm_neon.h>      /* ARM NEON intrinsics */

/* =========================================================================
 * 数据集配置 (可通过编译选项覆盖)
 * ========================================================================= */
#ifndef GF2_DATA_DIR
  #define GF2_DATA_DIR    "./Groebner/6_3799_2759_1953/"
#endif

#ifndef GF2_NUM_COLS
  #define GF2_NUM_COLS    3799
#endif

#ifndef GF2_NUM_PIVOTS
  #define GF2_NUM_PIVOTS  2759
#endif

#ifndef GF2_NUM_ROWS
  #define GF2_NUM_ROWS    1953
#endif

#define BITWORD_WIDTH     32
#define BENCH_REPEAT      4

typedef unsigned int bitword_t;

#define BITVEC_LEN   (GF2_NUM_COLS / BITWORD_WIDTH + 1)
#define NEON_BATCH   4     /* 128-bit / 32-bit = 4 */

/* =========================================================================
 * 全局缓冲区
 * ========================================================================= */
static bitword_t pivot_rows[GF2_NUM_COLS][BITVEC_LEN] = {{0}};
static bitword_t target_rows[GF2_NUM_ROWS][BITVEC_LEN] = {{0}};

#ifdef USE_ALIGNMENT
  static bitword_t pivot_buf[GF2_NUM_COLS]
      [(BITVEC_LEN / 16 + 1) * 16] __attribute__((aligned(64))) = {{0}};
  static bitword_t target_buf[GF2_NUM_ROWS]
      [(BITVEC_LEN / 16 + 1) * 16] __attribute__((aligned(64))) = {{0}};
#else
  static bitword_t pivot_buf[GF2_NUM_COLS][BITVEC_LEN] = {{0}};
  static bitword_t target_buf[GF2_NUM_ROWS][BITVEC_LEN] = {{0}};
#endif

/* =========================================================================
 * 基准测试包装
 * ========================================================================= */
void benchmark_gf2(void (*kernel)(bitword_t[GF2_NUM_COLS][BITVEC_LEN],
                                   bitword_t[GF2_NUM_ROWS][BITVEC_LEN]),
                   const char* tag) {
    using Clock = std::chrono::high_resolution_clock;
    auto t0 = Clock::now();
    for (int r = 0; r < BENCH_REPEAT; ++r) {
        kernel(pivot_rows, target_rows);
    }
    auto t1 = Clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << std::setw(14) << tag << ": "
              << std::fixed << std::setprecision(3) << ms << " ms\n";
}

/* =========================================================================
 * 实现 1: GF(2) 串行标量
 * ========================================================================= */
void gf2_eliminate_baseline(bitword_t pivots[GF2_NUM_COLS][BITVEC_LEN],
                            bitword_t targets[GF2_NUM_ROWS][BITVEC_LEN]) {
    std::memcpy(pivot_buf,  pivots,  sizeof(bitword_t) * GF2_NUM_COLS * BITVEC_LEN);
    std::memcpy(target_buf, targets, sizeof(bitword_t) * GF2_NUM_ROWS * BITVEC_LEN);

    for (int r = 0; r < GF2_NUM_ROWS; ++r) {
        for (int c = GF2_NUM_COLS; c >= 0; --c) {
            int widx = c / BITWORD_WIDTH;
            bitword_t mask = (bitword_t)1 << (c % BITWORD_WIDTH);

            if (target_buf[r][widx] & mask) {
                if (pivot_buf[c][widx] & mask) {
                    for (int w = BITVEC_LEN - 1; w >= 0; --w) {
                        target_buf[r][w] ^= pivot_buf[c][w];
                    }
                } else {
                    std::memcpy(pivot_buf[c], target_buf[r],
                                BITVEC_LEN * sizeof(bitword_t));
                    break;
                }
            }
        }
    }
}

/* =========================================================================
 * 实现 2: NEON 128-bit 向量化
 * ========================================================================= */
void gf2_eliminate_neon(bitword_t pivots[GF2_NUM_COLS][BITVEC_LEN],
                        bitword_t targets[GF2_NUM_ROWS][BITVEC_LEN]) {
    std::memcpy(pivot_buf,  pivots,  sizeof(bitword_t) * GF2_NUM_COLS * BITVEC_LEN);
    std::memcpy(target_buf, targets, sizeof(bitword_t) * GF2_NUM_ROWS * BITVEC_LEN);

    uint32x4_t tgt_vec, piv_vec;

    for (int r = 0; r < GF2_NUM_ROWS; ++r) {
        for (int c = GF2_NUM_COLS; c >= 0; --c) {
            int widx = c / BITWORD_WIDTH;
            bitword_t mask = (bitword_t)1 << (c % BITWORD_WIDTH);

            if (target_buf[r][widx] & mask) {
                if (pivot_buf[c][widx] & mask) {
                    /* --- NEON 批量异或: 每次 4 个 bitword_t (128-bit) --- */
                    int w;
                    for (w = 0; w < BITVEC_LEN / NEON_BATCH; ++w) {
                        tgt_vec = vld1q_u32(target_buf[r] + w * NEON_BATCH);
                        piv_vec = vld1q_u32(pivot_buf[c]  + w * NEON_BATCH);
                        vst1q_u32(target_buf[r] + w * NEON_BATCH,
                                  veorq_u32(tgt_vec, piv_vec));
                    }
                    for (int rem = w * NEON_BATCH; rem <= BITVEC_LEN - 1; ++rem) {
                        target_buf[r][rem] ^= pivot_buf[c][rem];
                    }
                } else {
                    std::memcpy(pivot_buf[c], target_buf[r],
                                BITVEC_LEN * sizeof(bitword_t));
                    break;
                }
            }
        }
    }
}

/* =========================================================================
 * 数据加载
 * ========================================================================= */
static bool load_gf2_dataset() {
    std::string ele_path = std::string(GF2_DATA_DIR) + "1.txt";
    std::ifstream ele_in(ele_path);
    if (!ele_in.is_open()) {
        std::cerr << "[FATAL] Cannot open " << ele_path << "\n";
        return false;
    }

    std::string line;
    int header, idx;
    for (int i = 0; i < GF2_NUM_PIVOTS; ++i) {
        std::getline(ele_in, line);
        std::istringstream iss(line);
        iss >> header;
        pivot_rows[header][header / BITWORD_WIDTH] |=
            (bitword_t)1 << (header % BITWORD_WIDTH);
        while (iss >> idx) {
            pivot_rows[header][idx / BITWORD_WIDTH] |=
                (bitword_t)1 << (idx % BITWORD_WIDTH);
        }
    }
    ele_in.close();

    std::string row_path = std::string(GF2_DATA_DIR) + "2.txt";
    std::ifstream row_in(row_path);
    if (!row_in.is_open()) {
        std::cerr << "[FATAL] Cannot open " << row_path << "\n";
        return false;
    }

    for (int i = 0; i < GF2_NUM_ROWS; ++i) {
        std::getline(row_in, line);
        std::istringstream iss(line);
        while (iss >> idx) {
            target_rows[i][idx / BITWORD_WIDTH] |=
                (bitword_t)1 << (idx % BITWORD_WIDTH);
        }
    }
    row_in.close();
    return true;
}

/* =========================================================================
 * 主入口
 * ========================================================================= */
int main() {
    std::cout << "================================================\n";
    std::cout << " GF(2) Groebner Basis (ARM NEON) Benchmark\n";
    std::cout << " Dataset: " << GF2_DATA_DIR << "\n";
    std::cout << " Cols=" << GF2_NUM_COLS
              << " Pivots=" << GF2_NUM_PIVOTS
              << " Rows=" << GF2_NUM_ROWS << "\n";
#ifdef USE_ALIGNMENT
    std::cout << " Mode: ALIGNED\n";
#else
    std::cout << " Mode: UNALIGNED\n";
#endif
    std::cout << "================================================\n";

    if (!load_gf2_dataset()) {
        return 1;
    }

    benchmark_gf2(gf2_eliminate_baseline, "Scalar");
    benchmark_gf2(gf2_eliminate_neon,      "NEON(128b)");

    std::cout << "\n[INFO] ARM NEON GF(2) benchmarks complete.\n";
    return 0;
}
