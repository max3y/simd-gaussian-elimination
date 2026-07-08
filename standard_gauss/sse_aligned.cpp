/**
 * ===========================================================================
 * sse_aligned.cpp  —  128-bit SSE4.2 对齐向量化高斯消元
 * ===========================================================================
 *
 * SIMD 特性:
 *   SSE (Streaming SIMD Extensions) 4.2 — Intel 在 x86-64 架构上定义
 *   的 128-bit 向量扩展. 每条 SSE 指令可同时操作 4 个单精度浮点数.
 *
 *   本实现使用以下 SSE 内联函数(intrinsics):
 *     _mm_load1_ps(addr)    — 广播加载: 将 1 个标量复制到向量 4 个 lane
 *     _mm_load_ps(addr)     — 对齐加载: 要求地址 16 字节对齐
 *     _mm_div_ps(a, b)      — 向量除法: 4 个 float 并行除
 *     _mm_mul_ps(a, b)      — 向量乘法
 *     _mm_sub_ps(a, b)      — 向量减法
 *     _mm_store_ps(addr, v) — 对齐存储: 要求地址 16 字节对齐
 *
 * 对齐策略:
 *   在进行批量 4 元素 SSE 加载/存储之前, 先用串行循环处理首部未对齐元素
 *   (peeling loop), 直至地址满足 16 字节对齐要求; 尾部剩余元素同样用串行
 *   循环收尾. 这避免了非对齐访存的额外延迟(约 1.2× ~ 1.5× 开销).
 *
 * 前提条件:
 *   矩阵维度 N 应为 4 的倍数以保证最佳效果, 但本实现兼容任意 N.
 *
 * 编译:
 *   g++ -O2 -march=native -msse4.2 -std=c++17 sse_aligned.cpp -o sse_aligned.exe
 *
 * 运行:
 *   ./sse_aligned.exe
 * ===========================================================================
 */

#include <iostream>
#include <vector>
#include <nmmintrin.h>   /* SSE4.2 intrinsics */
#include "../common/matrix_utils.h"
#include "../common/benchmark_utils.h"

/**
 * SSE 4.2 对齐向量化前向消元
 *
 * 向量化原理:
 *   内层 j 循环每次步进 4, 用一条除法/乘法/减法指令同时处理 4 列,
 *   理想情况下可获得接近 4× 的浮点吞吐提升(受限于内存带宽).
 *
 * @param mat  系数矩阵(原地修改)
 * @param dim  矩阵维度
 */
void forward_eliminate_sse_aligned(float** mat, int dim) {
    for (int pivot = 0; pivot < dim; ++pivot) {
        /* ------------------------------------------------------------
         * 阶段一: 主元行归一化 — SSE 向量除法
         * ------------------------------------------------------------
         * 将 mat[pivot][pivot] 广播至 SSE 128-bit 寄存器的全部 4 个 lane,
         * 然后对 mat[pivot][pivot+1 .. N-1] 逐块执行向量除法.
         */
        __m128 pivot_broadcast = _mm_load1_ps(&mat[pivot][pivot]);

        int col = pivot + 1;

        /* --- 首部串行对齐: 逐步推进到 16 字节对齐边界 --- */
        while (col < dim && (col & 0x3)) {
            mat[pivot][col] = mat[pivot][col] / mat[pivot][pivot];
            ++col;
        }

        /* --- 批量对齐向量化除法: 每次处理 4 列 --- */
        for (; col + 4 <= dim; col += 4) {
            __m128 row_chunk  = _mm_load_ps(&mat[pivot][col]);
            __m128 div_result = _mm_div_ps(row_chunk, pivot_broadcast);
            _mm_store_ps(&mat[pivot][col], div_result);
        }

        /* --- 尾部串行收尾 --- */
        for (; col < dim; ++col) {
            mat[pivot][col] = mat[pivot][col] / mat[pivot][pivot];
        }
        mat[pivot][pivot] = 1.0f;

        /* ------------------------------------------------------------
         * 阶段二: 下方各行消元 — SSE 向量乘减
         * ------------------------------------------------------------
         * 对 pivot 下方的每一行 row:
         *   mat[row][col] = mat[row][col] - mat[pivot][col] * mat[row][pivot]
         *
         * mat[row][pivot] 在 col 循环中是常量, 广播为向量 col_dup.
         */
        for (int row = pivot + 1; row < dim; ++row) {
            __m128 col_dup = _mm_load1_ps(&mat[row][pivot]);

            col = pivot + 1;

            /* --- 首部对齐 --- */
            while (col < dim && (col & 0x3)) {
                mat[row][col] = mat[row][col] - mat[pivot][col] * mat[row][pivot];
                ++col;
            }

            /* --- 批量向量化乘减 --- */
            for (; col + 4 <= dim; col += 4) {
                __m128 pivot_chunk = _mm_load_ps(&mat[pivot][col]);
                __m128 target_chunk = _mm_load_ps(&mat[row][col]);
                __m128 scaled_diff  = _mm_mul_ps(pivot_chunk, col_dup);
                __m128 result       = _mm_sub_ps(target_chunk, scaled_diff);
                _mm_store_ps(&mat[row][col], result);
            }

            /* --- 尾部串行 --- */
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

    tester.run("SSE-Aligned", forward_eliminate_sse_aligned);

    std::cout << "\n[INFO] SSE aligned vectorization complete.\n";
    return 0;
}
