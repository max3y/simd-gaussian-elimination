/**
 * ===========================================================================
 * avx_aligned.cpp  —  256-bit AVX/AVX2 对齐向量化高斯消元
 * ===========================================================================
 *
 * SIMD 特性:
 *   AVX (Advanced Vector Extensions) 将向量寄存器宽度从 SSE 的 128-bit
 *   扩展至 256-bit, 单条指令可同时操作 8 个单精度浮点数. AVX2 进一步
 *   引入了 FMA (Fused Multiply-Add) 与 gather 等高级指令.
 *
 *   本实现使用以下 AVX 内联函数:
 *     _mm256_load_ps(addr)   — 256-bit 对齐加载 (addr 须 32 字节对齐)
 *     _mm256_store_ps(addr,v) — 256-bit 对齐存储
 *     _mm256_div_ps(a, b)    — 并行浮点除法 (8 个 lane)
 *     _mm256_mul_ps(a, b)    — 并行浮点乘法
 *     _mm256_sub_ps(a, b)    — 并行浮点减法
 *     _mm256_broadcast_ss    — 等价于 _mm_load1_ps 的 256-bit 版本,
 *                              (注: 原版通过手动构造 float[8] 数组 + load
 *                              实现广播, 此处改用 _mm256_set1_ps 等价语义)
 *
 * 对齐策略:
 *   要求行首地址满足 32 字节对齐. 在动态分配场景中 (逐行 new),
 *   大部分 malloc 实现返回 16 字节对齐地址, 不一定满足 AVX 的 32 字节.
 *   因此首部 peeling 到 32 字节边界是必要的.
 *
 * 编译:
 *   g++ -O2 -march=native -mavx -mavx2 -std=c++17 avx_aligned.cpp -o avx_aligned.exe
 *
 * 预期性能:
 *   理论加速 SSE 的 1.6× ~ 2.0× (寄存器宽度翻倍, 但受限于内存带宽与
 *   除法单元延迟).
 * ===========================================================================
 */

#include <iostream>
#include <vector>
#include <immintrin.h>    /* AVX / AVX2 intrinsics */
#include "../common/matrix_utils.h"
#include "../common/benchmark_utils.h"

/**
 * AVX 对齐向量化前向消元
 *
 * 采用 AVX 256-bit 向量, 每次迭代处理 8 个 float 元素.
 * 主元值通过 _mm256_set1_ps 广播到 8 个 lane.
 *
 * @param mat  系数矩阵(原地修改)
 * @param dim  矩阵维度
 */
void forward_eliminate_avx_aligned(float** mat, int dim) {
    for (int pivot = 0; pivot < dim; ++pivot) {
        /* ------------------------------------------------------------
         * 阶段一: 主元行归一化 — 256-bit 向量除法
         * ------------------------------------------------------------
         * 使用 _mm256_set1_ps 将单个 pivot 值广播至 8 个 lane,
         * 替代原版的 float[8] 数组 + load 方案, 更简洁高效.
         */
        __m256 pivot_broadcast = _mm256_set1_ps(mat[pivot][pivot]);

        int col = pivot + 1;

        /* --- 首部对齐: 每次推进 1 列, 直至 col % 8 == 0 --- */
        while (col < dim && (col & 0x7)) {
            mat[pivot][col] = mat[pivot][col] / mat[pivot][pivot];
            ++col;
        }

        /* --- 批量对齐向量化: 每次 8 列 --- */
        for (; col + 8 <= dim; col += 8) {
            __m256 row_chunk  = _mm256_load_ps(&mat[pivot][col]);
            __m256 div_result = _mm256_div_ps(row_chunk, pivot_broadcast);
            _mm256_store_ps(&mat[pivot][col], div_result);
        }

        /* --- 尾部串行 --- */
        for (; col < dim; ++col) {
            mat[pivot][col] = mat[pivot][col] / mat[pivot][pivot];
        }
        mat[pivot][pivot] = 1.0f;

        /* ------------------------------------------------------------
         * 阶段二: 下方各行消元 — 256-bit 向量乘减
         * ------------------------------------------------------------
         * mat[row][pivot] 在 j 循环中不变, 广播为 col_dup.
         * 单次迭代同时消去 8 列.
         */
        for (int row = pivot + 1; row < dim; ++row) {
            __m256 col_dup = _mm256_set1_ps(mat[row][pivot]);

            col = pivot + 1;

            /* --- 首部对齐 --- */
            while (col < dim && (col & 0x7)) {
                mat[row][col] = mat[row][col] - mat[pivot][col] * mat[row][pivot];
                ++col;
            }

            /* --- 批量向量化乘减 --- */
            for (; col + 8 <= dim; col += 8) {
                __m256 pivot_chunk = _mm256_load_ps(&mat[pivot][col]);
                __m256 target_chunk = _mm256_load_ps(&mat[row][col]);
                __m256 prod   = _mm256_mul_ps(pivot_chunk, col_dup);
                __m256 result = _mm256_sub_ps(target_chunk, prod);
                _mm256_store_ps(&mat[row][col], result);
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

    tester.run("AVX-Aligned", forward_eliminate_avx_aligned);

    std::cout << "\n[INFO] AVX aligned vectorization complete.\n";
    return 0;
}
