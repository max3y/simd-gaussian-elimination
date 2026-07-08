/**
 * ===========================================================================
 * matrix_utils.h  —  矩阵生成、验证与输出工具集
 * ===========================================================================
 * 功能说明:
 *   本模块封装了浮点矩阵实验中所需的辅助操作，包括:
 *     1. 基于固定种子的伪随机矩阵填充(确保各次运行可复现)
 *     2. 矩阵内容的人类可读输出(调试用途)
 *     3. 内存对齐判断与对齐分配/释放
 *     4. 逐元素结果校验(用于验证向量化版本与串行版本的一致性)
 *
 * 设计原则:
 *   - 采用 Mersenne Twister 19937 伪随机引擎, 种子固定为 42
 *   - 所有随机数均匀分布在 [0.0, 1.0) 区间
 *   - 对齐分配使用 posix_memalign / _aligned_malloc 按平台适配
 *   - 宏隔离平台差异, 保证跨 x86 / ARM / Mac M 系列可编译
 * ===========================================================================
 */

#ifndef MATRIX_UTILS_H
#define MATRIX_UTILS_H

#include <iostream>
#include <iomanip>
#include <random>
#include <cstdlib>
#include <cstring>
#include <cassert>

/* =========================================================================
 * 平台自适应内存对齐宏
 * =========================================================================
 * 64 字节对齐: 匹配 x86 AVX-512 的 cache line 要求, 同时兼容
 * AVX(32 字节)与 SSE(16 字节), NEON 通常仅需 16 字节对齐, 此处
 * 采用统一的最大对齐标准, 简化多平台代码管理.
 * ========================================================================= */
#if defined(_MSC_VER)
  #define ALIGN_ALLOC(ptr, size, align) \
      ptr = _aligned_malloc((size), (align))
  #define ALIGN_FREE(ptr) _aligned_free(ptr)
  #define ALIGN_ATTR(align) __declspec(align(align))
#else
  #define ALIGN_ALLOC(ptr, size, align) \
      posix_memalign((void**)&(ptr), (align), (size))
  #define ALIGN_FREE(ptr) std::free(ptr)
  #define ALIGN_ATTR(align) __attribute__((aligned(align)))
#endif

/* 默认对齐粒度: 64 字节 (cache line) */
#define CACHE_LINE_ALIGN 64

/* =========================================================================
 * 全局随机数引擎 — 固定种子确保实验可复现
 * ========================================================================= */
static std::mt19937 rng_engine(42);

/**
 * 生成 [0.0, 1.0) 区间均匀分布的单精度浮点随机数
 * 用于填充高斯消元测试矩阵, 保证每次运行生成同一组输入数据
 */
inline float random_unit_float() {
    static std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    return dist(rng_engine);
}

/**
 * 以随机浮点数填充 N×N 矩阵(二维指针数组形式)
 * @param mat   指向 float* 数组的指针(行指针数组)
 * @param dim   矩阵维度(方阵)
 */
inline void matrix_fill_random(float** mat, int dim) {
    for (int r = 0; r < dim; ++r) {
        for (int c = 0; c < dim; ++c) {
            mat[r][c] = random_unit_float();
        }
    }
}

/**
 * 以随机浮点数填充静态二维数组 (用于 expand-test 变体)
 * @param mat   静态 float[N][N] 数组引用
 * @param dim   矩阵维度
 */
template<int N>
inline void matrix_fill_random_static(float mat[N][N], int dim) {
    for (int r = 0; r < dim; ++r) {
        for (int c = 0; c < dim; ++c) {
            mat[r][c] = random_unit_float();
        }
    }
}

/**
 * 打印矩阵至标准输出(调试用途)
 * 格式: 每行空格分隔, 行尾换行
 */
inline void matrix_print(float** mat, int dim) {
    for (int r = 0; r < dim; ++r) {
        for (int c = 0; c < dim; ++c) {
            std::cout << mat[r][c] << ' ';
        }
        std::cout << std::endl;
    }
}

/**
 * 判断给定指针是否满足指定字节对齐要求
 * @param ptr     待检测内存地址
 * @param align   期望的字节对齐数(须为 2 的幂)
 * @return        true 表示满足对齐条件
 */
inline bool is_aligned(void* ptr, size_t align) {
    return (reinterpret_cast<uintptr_t>(ptr) & (align - 1)) == 0;
}

/**
 * 分配 N×N 的 float** 二维矩阵 (逐行动态分配)
 * 同时打印每行首地址的对齐诊断信息
 */
inline float** matrix_alloc_2d(int dim) {
    float** mat = new float*[dim];
    for (int r = 0; r < dim; ++r) {
        mat[r] = new float[dim];
    }
    return mat;
}

/**
 * 释放 matrix_alloc_2d 分配的二维矩阵
 */
inline void matrix_free_2d(float** mat, int dim) {
    for (int r = 0; r < dim; ++r) {
        delete[] mat[r];
    }
    delete[] mat;
}

/**
 * 验证两个矩阵是否在浮点误差范围内相等
 * @return 不一致的元素个数 (0 表示完全一致)
 */
inline int matrix_verify(float** mat_a, float** mat_b, int dim, float eps = 1e-5f) {
    int mismatch = 0;
    for (int r = 0; r < dim; ++r) {
        for (int c = 0; c < dim; ++c) {
            if (std::fabs(mat_a[r][c] - mat_b[r][c]) > eps) {
                ++mismatch;
            }
        }
    }
    return mismatch;
}

/**
 * 深拷贝矩阵: dst := src
 */
inline void matrix_copy(float** dst, float** src, int dim) {
    for (int r = 0; r < dim; ++r) {
        std::memcpy(dst[r], src[r], dim * sizeof(float));
    }
}

#endif /* MATRIX_UTILS_H */
