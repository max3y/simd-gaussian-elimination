/**
 * ===========================================================================
 * neon_unaligned.cpp  —  ARM NEON 128-bit 非对齐向量化高斯消元
 * ===========================================================================
 *
 * NEON 非对齐版本: 省略首部对齐 peeling, 直接使用 vld1q_f32/vst1q_f32.
 *
 * 注意: ARM 手册中 NEON 的 vld1q_f32 在 AArch64 上实际上对非对齐地址
 * 是安全的(只要不跨页), 因此 aligned vs unaligned 的区分主要是为了
 * 与 x86 版本保持对称的对比实验设计.
 *
 * 编译:
 *   g++ -O2 -march=armv8-a+simd -std=c++17 neon_unaligned.cpp -o neon_unaligned.exe
 * ===========================================================================
 */

#include <iostream>
#include <vector>
#include <arm_neon.h>
#include "../common/matrix_utils.h"
#include "../common/benchmark_utils.h"

/**
 * ARM NEON 非对齐向量化前向消元
 * 无首部对齐处理, 直接批量向量化.
 */
void forward_eliminate_neon_unaligned(float** mat, int dim) {
    for (int pivot = 0; pivot < dim; ++pivot) {
        /* ---- 主元行归一化 ---- */
        float32x4_t pivot_broadcast = vld1q_dup_f32(&mat[pivot][pivot]);

        int col;
        for (col = pivot + 1; col + 4 <= dim; col += 4) {
            float32x4_t chunk = vld1q_f32(&mat[pivot][col]);
            chunk = vdivq_f32(chunk, pivot_broadcast);
            vst1q_f32(&mat[pivot][col], chunk);
        }

        for (; col < dim; ++col) {
            mat[pivot][col] = mat[pivot][col] / mat[pivot][pivot];
        }
        mat[pivot][pivot] = 1.0f;

        /* ---- 下方各行消元 ---- */
        for (int row = pivot + 1; row < dim; ++row) {
            float32x4_t col_dup = vld1q_dup_f32(&mat[row][pivot]);

            for (col = pivot + 1; col + 4 <= dim; col += 4) {
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

    tester.run("NEON-Unaligned", forward_eliminate_neon_unaligned);

    std::cout << "\n[INFO] ARM NEON unaligned vectorization complete.\n";
    return 0;
}
