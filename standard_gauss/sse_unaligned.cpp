/**
 * ===========================================================================
 * sse_unaligned.cpp  —  128-bit SSE4.2 非对齐向量化高斯消元
 * ===========================================================================
 *
 * 与 sse_aligned.cpp 的差异:
 *   - 使用 _mm_loadu_ps / _mm_storeu_ps (非对齐访存指令)
 *   - 无需首部对齐 peeling 循环, 代码更简洁
 *   - 非对齐访存在现代 x86 CPU (Sandy Bridge 及之后) 上的惩罚已大幅降低,
 *     但跨 cache line 边界的访问仍会产生额外周期开销
 *
 * 适用场景:
 *   无法保证矩阵各行起始地址为 16 字节对齐的情况;
 *   或 N 较小、对齐收益被 peeling 开销抵消时.
 *
 * 编译:
 *   g++ -O2 -march=native -msse4.2 -std=c++17 sse_unaligned.cpp -o sse_unaligned.exe
 * ===========================================================================
 */

#include <iostream>
#include <vector>
#include <nmmintrin.h>   /* SSE4.2 */
#include "../common/matrix_utils.h"
#include "../common/benchmark_utils.h"

/**
 * SSE 4.2 非对齐向量化前向消元
 *
 * 无首部对齐处理 — 全部使用 unaligned load/store,
 * 依靠现代 CPU 的非对齐访存硬件优化.
 */
void forward_eliminate_sse_unaligned(float** mat, int dim) {
    for (int pivot = 0; pivot < dim; ++pivot) {
        /* ---- 主元行归一化: 非对齐批量除法 ---- */
        __m128 pivot_broadcast = _mm_load1_ps(&mat[pivot][pivot]);

        int col;
        for (col = pivot + 1; col + 4 <= dim; col += 4) {
            __m128 chunk = _mm_loadu_ps(&mat[pivot][col]);
            chunk = _mm_div_ps(chunk, pivot_broadcast);
            _mm_storeu_ps(&mat[pivot][col], chunk);
        }

        /* 尾部串行 */
        for (; col < dim; ++col) {
            mat[pivot][col] = mat[pivot][col] / mat[pivot][pivot];
        }
        mat[pivot][pivot] = 1.0f;

        /* ---- 下方各行消元: 非对齐批量乘减 ---- */
        for (int row = pivot + 1; row < dim; ++row) {
            __m128 col_dup = _mm_load1_ps(&mat[row][pivot]);

            for (col = pivot + 1; col + 4 <= dim; col += 4) {
                __m128 pivot_chunk = _mm_loadu_ps(&mat[pivot][col]);
                __m128 target_chunk = _mm_loadu_ps(&mat[row][col]);
                __m128 prod   = _mm_mul_ps(pivot_chunk, col_dup);
                __m128 result = _mm_sub_ps(target_chunk, prod);
                _mm_storeu_ps(&mat[row][col], result);
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

    tester.run("SSE-Unaligned", forward_eliminate_sse_unaligned);

    std::cout << "\n[INFO] SSE unaligned vectorization complete.\n";
    return 0;
}
