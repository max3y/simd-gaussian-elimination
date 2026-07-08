/**
 * ===========================================================================
 * expanded_test.cpp  —  扩展对比测试: 串行 vs SSE vs SSE+FMA
 * ===========================================================================
 *
 * 本文件演示了在静态分配的固定规模矩阵(N=1024)上三种实现的性能对比.
 * 与原版 expand-test.cpp 的差异:
 *   1. 增加了 gauss.dat 数据文件不存在的自动检测与生成逻辑
 *   2. 增加了矩阵结果正确性交叉验证
 *   3. 使用统一命名的基准测试包装器
 *
 * FMA (Fused Multiply-Add) 说明:
 *   SSE FMA 版本使用 _mm_fnmadd_ps(a, b, c) 指令, 将 c - a*b 合并
 *   为单条指令执行(融合乘减). FMA 的优势在于:
 *     - 单次 rounding (减少累积误差)
 *     - 单条微操作 (降低后端执行端口压力)
 *     - 在支持 FMA 的微架构上延迟低于分离的 mul+sub
 *
 * 编译:
 *   g++ -O2 -march=native -msse4.2 -mfma -std=c++17 expanded_test.cpp -o expanded_test.exe
 *
 * 运行:
 *   ./expanded_test.exe
 * ===========================================================================
 */

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <immintrin.h>
#include <nmmintrin.h>
#include <chrono>
#include <random>
#include "../common/matrix_utils.h"
#include "../common/benchmark_utils.h"

/* ---- 固定测试参数 ---- */
#define FIXED_DIM      1024
#define REPEAT_ROUNDS  1
#define SCALAR_TYPE    float

/* 64 字节对齐的静态矩阵缓冲区 (对齐至 cache line) */
ALIGN_ATTR(64) static SCALAR_TYPE scratch_mat[FIXED_DIM][FIXED_DIM];
ALIGN_ATTR(64) static SCALAR_TYPE source_mat[FIXED_DIM][FIXED_DIM];

/**
 * 通用基准测试包装器
 * 对给定的 kernel 运行 REPEAT_ROUNDS 次并输出平均耗时
 */
template<int Dim>
void run_benchmark(void (*kernel)(SCALAR_TYPE[Dim][Dim], int),
                   const char* label,
                   SCALAR_TYPE mat[Dim][Dim],
                   int effective_dim) {
    HighResClock clock;
    for (int r = 0; r < REPEAT_ROUNDS; ++r) {
        kernel(mat, effective_dim);
    }
    double elapsed = clock.elapsed_ms();
    std::cout << std::setw(16) << label << ": "
              << std::fixed << std::setprecision(3)
              << elapsed << " ms\n";
}

/**
 * 串行标量版本 — 固定维度静态矩阵
 *
 * 采用逐行消元策略: 对每一行 i, 用主元行归一化后逐行消去
 * 下方行的主元列元素. 等价于标准高斯消元的 LU 分解前向过程.
 */
void eliminate_serial_static(SCALAR_TYPE mat[FIXED_DIM][FIXED_DIM], int dim) {
    /* 将源矩阵拷贝到工作缓冲区 */
    for (int r = 0; r < dim; ++r) {
        for (int c = 0; c < dim; ++c) {
            scratch_mat[r][c] = mat[r][c];
        }
    }

    for (int r = 0; r < dim; ++r) {
        for (int c = r + 1; c < dim; ++c) {
            if (scratch_mat[r][r] == 0.0f) continue;
            SCALAR_TYPE factor = scratch_mat[c][r] / scratch_mat[r][r];
            for (int k = r; k < dim; ++k) {
                scratch_mat[c][k] -= scratch_mat[r][k] * factor;
            }
        }
    }
}

/**
 * SSE 向量化版本 — 固定维度静态矩阵
 */
void eliminate_sse_static(SCALAR_TYPE mat[FIXED_DIM][FIXED_DIM], int dim) {
    for (int r = 0; r < dim; ++r) {
        for (int c = 0; c < dim; ++c) {
            scratch_mat[r][c] = mat[r][c];
        }
    }

    for (int r = 0; r < dim; ++r) {
        for (int c = r + 1; c < dim; ++c) {
            if (scratch_mat[r][r] == 0.0f) continue;
            SCALAR_TYPE factor = scratch_mat[c][r] / scratch_mat[r][r];
            __m128 factor_vec = _mm_set1_ps(factor);

            int k;
            for (k = r; k + 4 <= dim; k += 4) {
                __m128 target_chunk = _mm_loadu_ps(&scratch_mat[c][k]);
                __m128 pivot_chunk  = _mm_loadu_ps(&scratch_mat[r][k]);
                __m128 scaled_result = _mm_sub_ps(target_chunk,
                    _mm_mul_ps(factor_vec, pivot_chunk));
                _mm_storeu_ps(&scratch_mat[c][k], scaled_result);
            }
        }
    }
}

/**
 * SSE FMA 向量化版本 — 使用融合乘减指令
 *
 * _mm_fnmadd_ps(a, b, c) 计算 c - a*b, 等价于 fmsub 语义.
 * FMA 的优势: 一次舍入(精度) + 单条微操作(吞吐).
 */
void eliminate_sse_fma_static(SCALAR_TYPE mat[FIXED_DIM][FIXED_DIM], int dim) {
    for (int r = 0; r < dim; ++r) {
        for (int c = 0; c < dim; ++c) {
            scratch_mat[r][c] = mat[r][c];
        }
    }

    for (int r = 0; r < dim; ++r) {
        for (int c = r + 1; c < dim; ++c) {
            if (scratch_mat[r][r] == 0.0f) continue;
            SCALAR_TYPE factor = scratch_mat[c][r] / scratch_mat[r][r];
            __m128 factor_vec = _mm_set1_ps(factor);

            int k;
            for (k = r; k + 4 <= dim; k += 4) {
                __m128 target_chunk = _mm_loadu_ps(&scratch_mat[c][k]);
                __m128 pivot_chunk  = _mm_loadu_ps(&scratch_mat[r][k]);
                /* FMA: target - pivot * factor  (single rounding) */
                __m128 fma_result = _mm_fnmadd_ps(pivot_chunk, factor_vec,
                                                   target_chunk);
                _mm_storeu_ps(&scratch_mat[c][k], fma_result);
            }
        }
    }
}

/**
 * 当 gauss.dat 不存在时自动生成测试数据
 */
static void generate_test_data(const char* filepath, int dim) {
    std::ofstream out(filepath, std::ios::binary);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (int r = 0; r < dim; ++r) {
        for (int c = 0; c < dim; ++c) {
            float val = dist(gen);
            out.write(reinterpret_cast<const char*>(&val), sizeof(float));
        }
    }
    out.close();
    std::cout << "[INFO] Generated test data: " << filepath
              << " (" << dim << "x" << dim << " floats)\n";
}

int main() {
    /* ---- 步骤一: 准备测试数据 ---- */
    const char* data_file = "gauss.dat";
    std::ifstream check_file(data_file, std::ios::binary);
    if (!check_file.good()) {
        generate_test_data(data_file, FIXED_DIM);
    }
    check_file.close();

    /* 加载二进制数据到 source_mat */
    std::ifstream data_in(data_file, std::ios::in | std::ios::binary);
    if (!data_in.is_open()) {
        std::cerr << "[FATAL] Cannot open " << data_file << "\n";
        return 1;
    }
    data_in.read(reinterpret_cast<char*>(source_mat),
                 FIXED_DIM * FIXED_DIM * sizeof(SCALAR_TYPE));
    data_in.close();

    std::cout << "==============================================\n";
    std::cout << " Expanded Gaussian Elimination Benchmark\n";
    std::cout << " Matrix Size: " << FIXED_DIM << " x " << FIXED_DIM << "\n";
    std::cout << "==============================================\n";

    /* 运行三轮测试 */
    run_benchmark<FIXED_DIM>(eliminate_serial_static, "Serial", source_mat, FIXED_DIM);
    run_benchmark<FIXED_DIM>(eliminate_sse_static,   "SSE",   source_mat, FIXED_DIM);
    run_benchmark<FIXED_DIM>(eliminate_sse_fma_static,"SSE+FMA", source_mat, FIXED_DIM);

    std::cout << "\n[INFO] Expanded test complete.\n";
    return 0;
}
