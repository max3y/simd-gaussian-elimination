/**
 * ===========================================================================
 * avx512_aligned.cpp  —  512-bit AVX-512 对齐向量化高斯消元
 * ===========================================================================
 *
 * SIMD 特性:
 *   AVX-512 将向量宽度进一步扩展至 512-bit, 单条指令可同时操作
 *   16 个单精度浮点数. AVX-512 引入了 32 个 512-bit 向量寄存器
 *   (zmm0-zmm31), 以及掩码寄存器(k0-k7)等高级特性.
 *
 *   AVX-512 实际可用性取决于 CPU 型号:
 *     - Intel Xeon Phi (KNL/KNM), Skylake-SP/X, Ice Lake 及之后
 *     - AMD Zen 4 (通过两个 256-bit 操作拼接模拟, 非原生)
 *   在消费级 CPU 上, AVX-512 长时间未被广泛支持, 本代码运行时
 *   需先检查 /proc/cpuinfo 中 avx512f 标志位.
 *
 *   内联函数使用:
 *     _mm512_set1_ps(val)       — 广播标量到 512-bit 向量的 16 个 lane
 *     _mm512_load_ps(addr)      — 对齐加载 (64 字节对齐)
 *     _mm512_store_ps(addr, v)  — 对齐存储
 *     _mm512_div_ps(a, b)       — 向量除法 (16-way)
 *     _mm512_mul_ps(a, b)       — 向量乘法
 *     _mm512_sub_ps(a, b)       — 向量减法
 *     _mm512_fnmadd_ps(a,b,c)   — FMA: c - a*b (如支持)
 *
 * 约束:
 *   - 首部 peeling 至 64 字节对齐 (16 个 float 边界)
 *   - 编译器需开启 -mavx512f 标志
 *
 * 编译:
 *   g++ -O2 -march=native -mavx512f -std=c++17 avx512_aligned.cpp -o avx512_aligned.exe
 * ===========================================================================
 */

#include <iostream>
#include <vector>
#include <immintrin.h>    /* AVX-512 intrinsics (via immintrin.h) */
#include "../common/matrix_utils.h"
#include "../common/benchmark_utils.h"

/**
 * AVX-512 对齐向量化前向消元
 * 单次迭代处理 16 个 float — 是目前 x86 平台上最宽的向量化路径.
 */
void forward_eliminate_avx512_aligned(float** mat, int dim) {
    for (int pivot = 0; pivot < dim; ++pivot) {
        /* ---- 主元行归一化: 512-bit 批量除法 ---- */
        __m512 pivot_broadcast = _mm512_set1_ps(mat[pivot][pivot]);

        int col = pivot + 1;

        /* 首部对齐至 64 字节边界 (16 × sizeof(float)) */
        while (col < dim && (col & 0xF)) {
            mat[pivot][col] = mat[pivot][col] / mat[pivot][pivot];
            ++col;
        }

        /* 批量向量化除法: 16 元素/次 */
        for (; col + 16 <= dim; col += 16) {
            __m512 row_chunk  = _mm512_load_ps(&mat[pivot][col]);
            __m512 div_result = _mm512_div_ps(row_chunk, pivot_broadcast);
            _mm512_store_ps(&mat[pivot][col], div_result);
        }

        for (; col < dim; ++col) {
            mat[pivot][col] = mat[pivot][col] / mat[pivot][pivot];
        }
        mat[pivot][pivot] = 1.0f;

        /* ---- 下方各行消元: 512-bit 批量乘减 ---- */
        for (int row = pivot + 1; row < dim; ++row) {
            __m512 col_dup = _mm512_set1_ps(mat[row][pivot]);

            col = pivot + 1;

            while (col < dim && (col & 0xF)) {
                mat[row][col] = mat[row][col] - mat[pivot][col] * mat[row][pivot];
                ++col;
            }

            for (; col + 16 <= dim; col += 16) {
                __m512 pivot_chunk = _mm512_load_ps(&mat[pivot][col]);
                __m512 target_chunk = _mm512_load_ps(&mat[row][col]);
                __m512 prod   = _mm512_mul_ps(pivot_chunk, col_dup);
                __m512 result = _mm512_sub_ps(target_chunk, prod);
                _mm512_store_ps(&mat[row][col], result);
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

    tester.run("AVX512-Aligned", forward_eliminate_avx512_aligned);

    std::cout << "\n[INFO] AVX-512 aligned vectorization complete.\n";
    return 0;
}
