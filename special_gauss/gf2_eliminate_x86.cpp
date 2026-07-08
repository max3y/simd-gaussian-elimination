/**
 * ===========================================================================
 * gf2_eliminate_x86.cpp  —  GF(2) 域上 Gröbner 基高斯消元 (x86 SIMD)
 * ===========================================================================
 *
 * 算法背景:
 *   本文件实现有限域 GF(2) 上的高斯消元, 用于 Gröbner 基计算.
 *   GF(2) 域只有 {0, 1} 两个元素, 加法为 XOR(异或), 乘法为 AND.
 *
 *   与普通高斯消元的关键区别:
 *     1. 数据以位向量形式存储: 每个 unsigned int (32-bit) 打包 32 列
 *     2. 消去操作使用 XOR 替代浮点乘减: row ^= pivot
 *     3. 无需归一化 (GF(2) 中唯一的非零元就是 1)
 *     4. 主元检测通过位测试 (&) 完成
 *
 *   SIMD 对位向量的加速体现在:
 *     SSE (128-bit):  单次 XOR 操作 128 位 = 4 个 unsigned int
 *     AVX2 (256-bit): 单次 XOR 操作 256 位 = 8 个 unsigned int
 *     AVX-512:        单次 XOR 操作 512 位 = 16 个 unsigned int
 *
 * 数据文件格式:
 *   - 1.txt: 消元子数据 (每行首数字为行索引, 后跟非零列索引)
 *   - 2.txt: 被消元行数据 (每行为空格分隔的非零列索引)
 *
 * 当前配置: 数据集 6 (3799 × 2759 × 1953)
 *
 * 编译:
 *   g++ -O2 -march=native -std=c++17 gf2_eliminate_x86.cpp -o gf2_eliminate_x86.exe
 *
 * 对齐版本编译:
 *   g++ -O2 -march=native -DUSE_ALIGNMENT -std=c++17 gf2_eliminate_x86.cpp \
 *       -o gf2_eliminate_x86_aligned.exe
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
#include <immintrin.h>     /* AVX, AVX2, AVX-512 */
#include <nmmintrin.h>     /* SSE4.2 */

/* =========================================================================
 * 数据集配置宏 — 可通过 -D 编译选项覆盖
 * ========================================================================= */
#ifndef GF2_DATA_DIR
  #define GF2_DATA_DIR    "./Groebner/6_3799_2759_1953/"
#endif

#ifndef GF2_NUM_COLS
  #define GF2_NUM_COLS    3799    /* 矩阵列数 */
#endif

#ifndef GF2_NUM_PIVOTS
  #define GF2_NUM_PIVOTS  2759    /* 消元子数量 */
#endif

#ifndef GF2_NUM_ROWS
  #define GF2_NUM_ROWS    1953    /* 被消元行数量 */
#endif

#define BITWORD_WIDTH     32      /* unsigned int 位宽 */
#define BENCH_REPEAT      4       /* 基准测试重复次数 */

/* =========================================================================
 * 类型定义
 * ========================================================================= */
typedef unsigned int bitword_t;

/* 每组位向量需要的 unsigned int 个数 */
#define BITVEC_LEN  (GF2_NUM_COLS / BITWORD_WIDTH + 1)

/* 对齐批次大小 (以 bitword_t 为单位) */
#define SSE_BATCH   4    /* 128-bit / 32-bit = 4 */
#define AVX2_BATCH  8    /* 256-bit / 32-bit = 8 */
#define AVX512_BATCH 16  /* 512-bit / 32-bit = 16 */

/* =========================================================================
 * 全局缓冲区 — 源数据和可变副本
 * ========================================================================= */
static bitword_t pivot_rows[GF2_NUM_COLS][BITVEC_LEN] = {{0}};
static bitword_t target_rows[GF2_NUM_ROWS][BITVEC_LEN] = {{0}};

#ifdef USE_ALIGNMENT
  /* 对齐版本: 填充至 64 字节边界, 使用 __attribute__((aligned(64))) */
  static bitword_t pivot_buf[GF2_NUM_COLS]
      [(BITVEC_LEN / 16 + 1) * 16] __attribute__((aligned(64))) = {{0}};
  static bitword_t target_buf[GF2_NUM_ROWS]
      [(BITVEC_LEN / 16 + 1) * 16] __attribute__((aligned(64))) = {{0}};
#else
  static bitword_t pivot_buf[GF2_NUM_COLS][BITVEC_LEN] = {{0}};
  static bitword_t target_buf[GF2_NUM_ROWS][BITVEC_LEN] = {{0}};
#endif

/* =========================================================================
 * 基准测试包装器
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
 * 实现 1: GF(2) 串行标量消元 (Baseline)
 * ========================================================================= */
void gf2_eliminate_baseline(bitword_t pivots[GF2_NUM_COLS][BITVEC_LEN],
                            bitword_t targets[GF2_NUM_ROWS][BITVEC_LEN]) {
    /* 深拷贝源数据到工作缓冲区 */
    std::memcpy(pivot_buf,  pivots,  sizeof(bitword_t) * GF2_NUM_COLS * BITVEC_LEN);
    std::memcpy(target_buf, targets, sizeof(bitword_t) * GF2_NUM_ROWS * BITVEC_LEN);

    for (int r = 0; r < GF2_NUM_ROWS; ++r) {
        /* 从最高位列向最低位扫描 */
        for (int c = GF2_NUM_COLS; c >= 0; --c) {
            int word_idx = c / BITWORD_WIDTH;
            bitword_t bit_mask = (bitword_t)1 << (c % BITWORD_WIDTH);

            if (target_buf[r][word_idx] & bit_mask) {
                if (pivot_buf[c][word_idx] & bit_mask) {
                    /* 消元子存在: 异或消去 */
                    for (int w = BITVEC_LEN - 1; w >= 0; --w) {
                        target_buf[r][w] ^= pivot_buf[c][w];
                    }
                } else {
                    /* 消元子不存在: 将当前行提升为新的消元子 */
                    std::memcpy(pivot_buf[c], target_buf[r],
                                BITVEC_LEN * sizeof(bitword_t));
                    break;
                }
            }
        }
    }
}

/* =========================================================================
 * 实现 2: SSE 128-bit 向量化
 * ========================================================================= */
void gf2_eliminate_sse(bitword_t pivots[GF2_NUM_COLS][BITVEC_LEN],
                       bitword_t targets[GF2_NUM_ROWS][BITVEC_LEN]) {
    std::memcpy(pivot_buf,  pivots,  sizeof(bitword_t) * GF2_NUM_COLS * BITVEC_LEN);
    std::memcpy(target_buf, targets, sizeof(bitword_t) * GF2_NUM_ROWS * BITVEC_LEN);

    __m128i target_vec, pivot_vec;

    for (int r = 0; r < GF2_NUM_ROWS; ++r) {
        for (int c = GF2_NUM_COLS; c >= 0; --c) {
            int word_idx = c / BITWORD_WIDTH;
            bitword_t bit_mask = (bitword_t)1 << (c % BITWORD_WIDTH);

            if (target_buf[r][word_idx] & bit_mask) {
                if (pivot_buf[c][word_idx] & bit_mask) {
                    /* --- SSE 批量异或: 每次 4 个 bitword_t (128-bit) --- */
                    int w;
                    for (w = 0; w < BITVEC_LEN / SSE_BATCH; ++w) {
                        __m128i* tgt_ptr = reinterpret_cast<__m128i*>(
                            target_buf[r] + w * SSE_BATCH);
                        __m128i* piv_ptr = reinterpret_cast<__m128i*>(
                            pivot_buf[c] + w * SSE_BATCH);
                        target_vec = _mm_loadu_si128(tgt_ptr);
                        pivot_vec  = _mm_loadu_si128(piv_ptr);
                        _mm_storeu_si128(tgt_ptr,
                            _mm_xor_si128(target_vec, pivot_vec));
                    }
                    /* 尾部不足 128-bit 的部分用标量处理 */
                    for (int rem = w * SSE_BATCH; rem <= BITVEC_LEN - 1; ++rem) {
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
 * 实现 3: AVX2 256-bit 向量化
 * ========================================================================= */
void gf2_eliminate_avx2(bitword_t pivots[GF2_NUM_COLS][BITVEC_LEN],
                        bitword_t targets[GF2_NUM_ROWS][BITVEC_LEN]) {
    std::memcpy(pivot_buf,  pivots,  sizeof(bitword_t) * GF2_NUM_COLS * BITVEC_LEN);
    std::memcpy(target_buf, targets, sizeof(bitword_t) * GF2_NUM_ROWS * BITVEC_LEN);

    __m256i target_vec, pivot_vec;

    for (int r = 0; r < GF2_NUM_ROWS; ++r) {
        for (int c = GF2_NUM_COLS; c >= 0; --c) {
            int word_idx = c / BITWORD_WIDTH;
            bitword_t bit_mask = (bitword_t)1 << (c % BITWORD_WIDTH);

            if (target_buf[r][word_idx] & bit_mask) {
                if (pivot_buf[c][word_idx] & bit_mask) {
                    /* --- AVX2 批量异或: 每次 8 个 bitword_t (256-bit) --- */
                    int w;
                    for (w = 0; w < BITVEC_LEN / AVX2_BATCH; ++w) {
#ifdef USE_ALIGNMENT
                        target_vec = _mm256_load_si256(
                            (__m256i*)(target_buf[r] + w * AVX2_BATCH));
                        pivot_vec  = _mm256_load_si256(
                            (__m256i*)(pivot_buf[c] + w * AVX2_BATCH));
                        _mm256_store_si256(
                            (__m256i*)(target_buf[r] + w * AVX2_BATCH),
                            _mm256_xor_si256(target_vec, pivot_vec));
#else
                        target_vec = _mm256_loadu_si256(
                            (__m256i*)(target_buf[r] + w * AVX2_BATCH));
                        pivot_vec  = _mm256_loadu_si256(
                            (__m256i*)(pivot_buf[c] + w * AVX2_BATCH));
                        _mm256_storeu_si256(
                            (__m256i*)(target_buf[r] + w * AVX2_BATCH),
                            _mm256_xor_si256(target_vec, pivot_vec));
#endif
                    }
                    for (int rem = w * AVX2_BATCH; rem <= BITVEC_LEN - 1; ++rem) {
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
 * 实现 4: AVX-512 512-bit 向量化
 * ========================================================================= */
void gf2_eliminate_avx512(bitword_t pivots[GF2_NUM_COLS][BITVEC_LEN],
                          bitword_t targets[GF2_NUM_ROWS][BITVEC_LEN]) {
    std::memcpy(pivot_buf,  pivots,  sizeof(bitword_t) * GF2_NUM_COLS * BITVEC_LEN);
    std::memcpy(target_buf, targets, sizeof(bitword_t) * GF2_NUM_ROWS * BITVEC_LEN);

    __m512i target_vec, pivot_vec;

    for (int r = 0; r < GF2_NUM_ROWS; ++r) {
        for (int c = GF2_NUM_COLS; c >= 0; --c) {
            int word_idx = c / BITWORD_WIDTH;
            bitword_t bit_mask = (bitword_t)1 << (c % BITWORD_WIDTH);

            if (target_buf[r][word_idx] & bit_mask) {
                if (pivot_buf[c][word_idx] & bit_mask) {
                    /* --- AVX-512 批量异或: 每次 16 个 bitword_t (512-bit) --- */
                    int w;
                    for (w = 0; w < BITVEC_LEN / AVX512_BATCH; ++w) {
#ifdef USE_ALIGNMENT
                        target_vec = _mm512_load_si512(
                            (__m512i*)(target_buf[r] + w * AVX512_BATCH));
                        pivot_vec  = _mm512_load_si512(
                            (__m512i*)(pivot_buf[c] + w * AVX512_BATCH));
                        _mm512_store_si512(
                            (__m512i*)(target_buf[r] + w * AVX512_BATCH),
                            _mm512_xor_si512(target_vec, pivot_vec));
#else
                        target_vec = _mm512_loadu_si512(
                            (__m512i*)(target_buf[r] + w * AVX512_BATCH));
                        pivot_vec  = _mm512_loadu_si512(
                            (__m512i*)(pivot_buf[c] + w * AVX512_BATCH));
                        _mm512_storeu_si512(
                            (__m512i*)(target_buf[r] + w * AVX512_BATCH),
                            _mm512_xor_si512(target_vec, pivot_vec));
#endif
                    }
                    for (int rem = w * AVX512_BATCH; rem <= BITVEC_LEN - 1; ++rem) {
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
    /* 加载消元子文件 (1.txt) */
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

    /* 加载被消元行文件 (2.txt) */
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
    std::cout << " GF(2) Groebner Basis Elimination Benchmark\n";
    std::cout << " Dataset: " << GF2_DATA_DIR << "\n";
    std::cout << " Cols=" << GF2_NUM_COLS
              << " Pivots=" << GF2_NUM_PIVOTS
              << " Rows=" << GF2_NUM_ROWS << "\n";
#ifdef USE_ALIGNMENT
    std::cout << " Mode: ALIGNED (64-byte cache line)\n";
#else
    std::cout << " Mode: UNALIGNED\n";
#endif
    std::cout << "================================================\n";

    if (!load_gf2_dataset()) {
        return 1;
    }

    /* 依次运行所有实现 */
    benchmark_gf2(gf2_eliminate_baseline, "Scalar");
    benchmark_gf2(gf2_eliminate_sse,       "SSE(128b)");
    benchmark_gf2(gf2_eliminate_avx2,      "AVX2(256b)");
    benchmark_gf2(gf2_eliminate_avx512,    "AVX512(512b)");

    std::cout << "\n[INFO] All GF(2) benchmarks complete.\n";
    return 0;
}
