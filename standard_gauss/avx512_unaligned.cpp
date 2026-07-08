/**
 * ===========================================================================
 * avx512_unaligned.cpp  —  512-bit AVX-512 非对齐向量化高斯消元
 * ===========================================================================
 *
 * 使用 _mm512_loadu_ps / _mm512_storeu_ps 进行非对齐访存.
 * 与对齐版本相比, 省略了首部 peeling 逻辑, 以轻微的访存代价
 * 换取代码简洁性与更少的条件分支.
 *
 * 编译:
 *   g++ -O2 -march=native -mavx512f -std=c++17 avx512_unaligned.cpp -o avx512_unaligned.exe
 * ===========================================================================
 */

#include <iostream>
#include <vector>
#include <immintrin.h>
#include "../common/matrix_utils.h"
#include "../common/benchmark_utils.h"

/**
 * AVX-512 非对齐向量化前向消元
 */
void forward_eliminate_avx512_unaligned(float** mat, int dim) {
    for (int pivot = 0; pivot < dim; ++pivot) {
        /* ---- 主元行归一化 ---- */
        __m512 pivot_broadcast = _mm512_set1_ps(mat[pivot][pivot]);

        int col;
        for (col = pivot + 1; col + 16 <= dim; col += 16) {
            __m512 chunk = _mm512_loadu_ps(&mat[pivot][col]);
            chunk = _mm512_div_ps(chunk, pivot_broadcast);
            _mm512_storeu_ps(&mat[pivot][col], chunk);
        }

        for (; col < dim; ++col) {
            mat[pivot][col] = mat[pivot][col] / mat[pivot][pivot];
        }
        mat[pivot][pivot] = 1.0f;

        /* ---- 下方各行消元 ---- */
        for (int row = pivot + 1; row < dim; ++row) {
            __m512 col_dup = _mm512_set1_ps(mat[row][pivot]);

            for (col = pivot + 1; col + 16 <= dim; col += 16) {
                __m512 pivot_chunk = _mm512_loadu_ps(&mat[pivot][col]);
                __m512 target_chunk = _mm512_loadu_ps(&mat[row][col]);
                __m512 prod   = _mm512_mul_ps(pivot_chunk, col_dup);
                __m512 result = _mm512_sub_ps(target_chunk, prod);
                _mm512_storeu_ps(&mat[row][col], result);
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

    tester.run("AVX512-Unaligned", forward_eliminate_avx512_unaligned);

    std::cout << "\n[INFO] AVX-512 unaligned vectorization complete.\n";
    return 0;
}
