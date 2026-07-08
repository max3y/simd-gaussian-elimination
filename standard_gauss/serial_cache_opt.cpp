/**
 * ===========================================================================
 * serial_cache_opt.cpp  —  引入转置缓存的串行高斯消元
 * ===========================================================================
 *
 * 优化策略:
 *   原始串行版本中 A[k][j]*A[i][k] 的访问模式存在问题:
 *   - A[k][j] 沿行方向连续访问(友好)
 *   - A[i][k] 在 j 循环中不变, 但按列索引 k 读取 A[i][k] 需要跨行
 *
 *   本版本在消元开始前预先构建转置矩阵 trans_buf[j][i] = mat[i][j],
 *   使得后续消元阶段用 trans_buf[k][i] 替代 mat[i][k], 将跨行读取
 *   转换为沿行连续读取, 显著改善 L1/L2 cache 命中率.
 *
 *   空间开销: 额外 N×N 的静态缓冲区 (栈上分配, 上限 5000).
 *
 * 关键技术点:
 *   - 转置(transposition)是矩阵运算中最基础的 cache 友好变换之一
 *   - 在大规模矩阵上, 转置本身开销 O(N^2) 可被 O(N^3) 消元摊还
 *   - 使用静态数组而非动态分配消除 new/delete 开销
 *
 * 编译:
 *   g++ -O2 -march=native -std=c++17 serial_cache_opt.cpp -o serial_cache_opt.exe
 * ===========================================================================
 */

#include <iostream>
#include <vector>
#include "../common/matrix_utils.h"
#include "../common/benchmark_utils.h"

/* 静态转置缓冲区 — 避免动态分配开销, 同时利用栈上连续性提高 cache 效率 */
#define MAX_DIM 5000
static float transposed_cache[MAX_DIM][MAX_DIM];

/**
 * Cache 优化串行高斯消元
 *
 * 与 serial_naive 的关键差异:
 *   消元内循环中使用 transposed_cache[k][i] 替代 mat[i][k],
 *   将列方向的不连续读取转变为行方向的连续读取.
 *
 * @param mat  输入矩阵(原地消元)
 * @param dim  矩阵维度 (不可超过 MAX_DIM)
 */
void forward_eliminate_cache_opt(float** mat, int dim) {
    /* ---- 预处理: 构建转置缓存, 同时清零 mat 下三角 ---- */
    for (int r = 0; r < dim; ++r) {
        for (int c = 0; c < r; ++c) {
            transposed_cache[c][r] = mat[r][c];
            mat[r][c] = 0.0f;
        }
    }

    /* ---- 前向消元 ---- */
    for (int pivot = 0; pivot < dim; ++pivot) {
        /* 主元行归一化 */
        for (int col = pivot + 1; col < dim; ++col) {
            mat[pivot][col] = mat[pivot][col] / mat[pivot][pivot];
        }
        mat[pivot][pivot] = 1.0f;

        /* 下方各行消元: 用 transposed_cache 替代 mat[i][k] */
        for (int row = pivot + 1; row < dim; ++row) {
            for (int col = pivot + 1; col < dim; ++col) {
                mat[row][col] = mat[row][col]
                    - mat[pivot][col] * transposed_cache[pivot][row];
            }
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

    tester.run("Serial-CacheOpt", forward_eliminate_cache_opt);

    std::cout << "\n[INFO] Cache-optimized serial baseline complete.\n";
    return 0;
}
