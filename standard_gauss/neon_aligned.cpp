/**
 * ===========================================================================
 * neon_aligned.cpp  —  ARM NEON 128-bit 对齐向量化高斯消元
 * ===========================================================================
 *
 * 平台背景:
 *   NEON 是 ARM 架构的 SIMD 指令集扩展, 提供 128-bit 向量寄存器
 *   (Q0-Q15, 在 AArch64 下扩展为 32 个). 每条 NEON 指令可同时操作
 *   4 个单精度浮点数或 2 个双精度浮点数.
 *
 *   本实现针对 ARM Cortex-A 系列 (如鲲鹏 920) 或 Apple M 系列芯片
 *   (M1/M2/M3 通过 Rosetta 或原生 ARM 模式)进行优化.
 *
 * NEON intrinsics 映射:
 *   vld1q_dup_f32(addr)   → 对标 _mm_load1_ps (广播加载)
 *   vld1q_f32(addr)       → 对标 _mm_load_ps   (对齐加载)
 *   vst1q_f32(addr, v)    → 对标 _mm_store_ps  (对齐存储)
 *   vdivq_f32(a, b)       → 对标 _mm_div_ps    (向量除法)
 *   vmulq_f32(a, b)       → 对标 _mm_mul_ps    (向量乘法)
 *   vsubq_f32(a, b)       → 对标 _mm_sub_ps    (向量减法)
 *
 * 对齐要求:
 *   128-bit NEON 加载存储要求 16 字节对齐. 对于 malloc/new 分配的
 *   内存, C++ 标准保证对齐至 alignof(std::max_align_t)(通常为 16),
 *   因此 NEON 对齐版本在实践中通常无需显式 peeling.
 *
 * 编译 (ARM Linux / macOS):
 *   g++ -O2 -march=armv8-a+simd -std=c++17 neon_aligned.cpp -o neon_aligned.exe
 * 或 (macOS Apple Silicon):
 *   clang++ -O2 -march=armv8.5-a -std=c++17 neon_aligned.cpp -o neon_aligned.exe
 * ===========================================================================
 */

#include <iostream>
#include <vector>
#include <arm_neon.h>     /* NEON intrinsics */
#include "../common/matrix_utils.h"
#include "../common/benchmark_utils.h"

/**
 * ARM NEON 对齐向量化前向消元
 *
 * 每次迭代处理 4 个 float, 与 SSE 版本向量宽度相同,
 * 但 NEON 指令集采用 RISC 风格的独立目标寄存器编码,
 * 在某些微架构上具有更优的流水线调度特性.
 */
void forward_eliminate_neon_aligned(float** mat, int dim) {
    for (int pivot = 0; pivot < dim; ++pivot) {
        /* ---- 主元行归一化: NEON 向量除法 ---- */
        float32x4_t pivot_broadcast = vld1q_dup_f32(&mat[pivot][pivot]);

        int col = pivot + 1;

        /* 首部串行对齐至 16 字节边界 (4 × sizeof(float)) */
        while (col < dim && (col & 0x3)) {
            mat[pivot][col] = mat[pivot][col] / mat[pivot][pivot];
            ++col;
        }

        /* 批量向量化除法: 每次处理 4 列 */
        for (; col + 4 <= dim; col += 4) {
            float32x4_t row_chunk  = vld1q_f32(&mat[pivot][col]);
            float32x4_t div_result = vdivq_f32(row_chunk, pivot_broadcast);
            vst1q_f32(&mat[pivot][col], div_result);
        }

        for (; col < dim; ++col) {
            mat[pivot][col] = mat[pivot][col] / mat[pivot][pivot];
        }
        mat[pivot][pivot] = 1.0f;

        /* ---- 下方各行消元: NEON 向量乘减 ---- */
        for (int row = pivot + 1; row < dim; ++row) {
            float32x4_t col_dup = vld1q_dup_f32(&mat[row][pivot]);

            col = pivot + 1;

            while (col < dim && (col & 0x3)) {
                mat[row][col] = mat[row][col] - mat[pivot][col] * mat[row][pivot];
                ++col;
            }

            for (; col + 4 <= dim; col += 4) {
                float32x4_t pivot_chunk = vld1q_f32(&mat[pivot][col]);
                float32x4_t target_chunk = vld1q_f32(&mat[row][col]);
                float32x4_t prod   = vmulq_f32(pivot_chunk, col_dup);
                float32x4_t result = vsubq_f32(target_chunk, prod);
                vst1q_f32(&mat[row][col], result);
            }

            for (; col < dim; ++col) {
                mat[row][col] = mat[row][col] - mat[pivot][col] * mat[row][pivot];
            }
            mat[row][pivot] = 0.0f;
        }
    }
}

int main() {
    BatchTester tester;
    tester.add_size(200)
          .add_size(500)
          .add_size(1000)
          .add_size(2000)
          .add_size(3000);

    tester.run("NEON-Aligned", forward_eliminate_neon_aligned);

    std::cout << "\n[INFO] ARM NEON aligned vectorization complete.\n";
    return 0;
}
