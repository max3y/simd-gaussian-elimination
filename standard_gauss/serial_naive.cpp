/**
 * ===========================================================================
 * serial_naive.cpp  —  串行朴素高斯消元(前向消去阶段)
 * ===========================================================================
 *
 * 算法定位:
 *   本文件实现最基础的逐元素高斯消元(无分块、无向量化、无 Cache 优化),
 *   作为后续所有 SIMD 向量化版本的必要性能基线(baseline).
 *
 * 消元策略:
 *   按行顺序选取主元(pivot), 对主元行做归一化除法, 然后对下方每一行
 *   逐一执行乘减消元. 这一"逐行-逐列"双重循环结构是高斯消元的直接
 *   翻译, 时间复杂度 O(N^3).
 *
 * 性能瓶颈预分析:
 *   1. 内层循环 A[i][j] = A[i][j] - A[k][j] * A[i][k] 中, A[k][j] 的
 *      访问沿行方向连续, 但 A[i][j] 跨不同行访问——当矩阵以行优先存储
 *      且各行独立 new 分配时, 跨行访问极可能触发大量 cache miss;
 *   2. A[i][k] 在内层 j 循环中保持不变, 但编译器若无优化, 每次迭代
 *      都会重新从内存加载;
 *   3. 逐元素的除法/乘法/减法均受限于标量浮点单元延迟.
 *
 * 编译示例 (x86 GCC):
 *   g++ -O2 -march=native -std=c++17 serial_naive.cpp -o serial_naive.exe
 *
 * 运行:
 *   ./serial_naive.exe
 * ===========================================================================
 */

#include <iostream>
#include <vector>
#include "../common/matrix_utils.h"
#include "../common/benchmark_utils.h"

/**
 * 串行朴素高斯消元 — 前向消去 (无任何优化)
 *
 * 数学过程:
 *   for k = 0 .. N-1:                      // 逐行选取主元
 *       归一化: A[k][j] /= A[k][k]  (j = k+1 .. N-1)
 *       A[k][k] = 1.0
 *       消去:   A[i][j] -= A[k][j] * A[i][k]  (i = k+1 .. N-1, j = k+1 .. N-1)
 *       A[i][k] = 0.0
 *
 * @param mat  输入 N×N 系数矩阵(原地修改)
 * @param dim  矩阵维度
 */
void forward_eliminate_serial(float** mat, int dim) {
    for (int pivot = 0; pivot < dim; ++pivot) {
        /* ---- 阶段一: 主元行归一化 ---- */
        for (int col = pivot + 1; col < dim; ++col) {
            mat[pivot][col] = mat[pivot][col] / mat[pivot][pivot];
        }
        mat[pivot][pivot] = 1.0f;

        /* ---- 阶段二: 下方各行消元 ---- */
        for (int row = pivot + 1; row < dim; ++row) {
            for (int col = pivot + 1; col < dim; ++col) {
                mat[row][col] = mat[row][col] - mat[pivot][col] * mat[row][pivot];
            }
            mat[row][pivot] = 0.0f;
        }
    }
}

int main() {
    /* 多规模测试: 从小到大覆盖典型工作集 */
    BatchTester tester;
    tester.add_size(200)
          .add_size(500)
          .add_size(1000)
          .add_size(2000)
          .add_size(3000);

    tester.run("Serial-Naive", forward_eliminate_serial);

    std::cout << "\n[INFO] Serial naive baseline complete.\n";
    return 0;
}
