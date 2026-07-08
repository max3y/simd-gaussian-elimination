/**
 * ===========================================================================
 * avx_unaligned.cpp  —  256-bit AVX 非对齐向量化高斯消元
 * ===========================================================================
 *
 * 使用 _mm256_loadu_ps / _mm256_storeu_ps 非对齐加载存储指令.
 * 无需首部 peeling, 代码路径更简单, 代价是跨 cache line 边界时
 * 可能有额外的访存延迟.
 *
 * 在 Haswell 及之后的 Intel 微架构上, 非对齐 256-bit 访存的额外
 * 代价已降至 ~1 cycle (只要不跨越 4KB 页边界). 因此对于大多数
 * 实际应用场景, unaligned 版本的简洁性优于对齐版本.
 *
 * 编译:
 *   g++ -O2 -march=native -mavx -mavx2 -std=c++17 avx_unaligned.cpp -o avx_unaligned.exe
 * ===========================================================================
 */

#include <iostream>
#include <vector>
#include <immintrin.h>
#include "../common/matrix_utils.h"
#include "../common/benchmark_utils.h"

/**
 * AVX 非对齐向量化前向消元
 * 全部使用 unaligned 加载/存储, 无首部处理循环.
 */
void forward_eliminate_avx_unaligned(float** mat, int dim) {
    for (int pivot = 0; pivot < dim; ++pivot) {
        /* ---- 主元行归一化 ---- */
        __m256 pivot_broadcast = _mm256_set1_ps(mat[pivot][pivot]);

        int col;
        for (col = pivot + 1; col + 8 <= dim; col += 8) {
            __m256 chunk = _mm256_loadu_ps(&mat[pivot][col]);
            chunk = _mm256_div_ps(chunk, pivot_broadcast);
            _mm256_storeu_ps(&mat[pivot][col], chunk);
        }

        for (; col < dim; ++col) {
            mat[pivot][col] = mat[pivot][col] / mat[pivot][pivot];
        }
        mat[pivot][pivot] = 1.0f;

        /* ---- 下方各行消元 ---- */
        for (int row = pivot + 1; row < dim; ++row) {
            __m256 col_dup = _mm256_set1_ps(mat[row][pivot]);

            for (col = pivot + 1; col + 8 <= dim; col += 8) {
                __m256 pivot_chunk = _mm256_loadu_ps(&mat[pivot][col]);
                __m256 target_chunk = _mm256_loadu_ps(&mat[row][col]);
                __m256 prod   = _mm256_mul_ps(pivot_chunk, col_dup);
                __m256 result = _mm256_sub_ps(target_chunk, prod);
                _mm256_storeu_ps(&mat[row][col], result);
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

    tester.run("AVX-Unaligned", forward_eliminate_avx_unaligned);

    std::cout << "\n[INFO] AVX unaligned vectorization complete.\n";
    return 0;
}
