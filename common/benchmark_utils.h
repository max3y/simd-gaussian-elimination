/**
 * ===========================================================================
 * benchmark_utils.h  —  性能计时、批量测试与结果输出框架
 * ===========================================================================
 *
 * 设计意图:
 *   将原版散落在各 main() 中的计时与测试逻辑统一抽取封装,
 *   使各实现文件专注于算法核心, 通过一致的基准测试框架获取
 *   可比较的耗时数据.
 *
 * 核心组件:
 *   1. HighResClock    — std::chrono 高精度时钟的轻量包装
 *   2. BatchTester     — 批量多规模自动测试调度器
 *   3. report_result() — 统一格式化输出(可直接导入 Excel/Python)
 *
 * 使用范式:
 *   BatchTester tester;
 *   tester.add_size(500).add_size(1000).add_size(2000);
 *   tester.run("Serial-Baseline", forward_eliminate_serial);
 *   tester.run("AVX-Aligned",  forward_eliminate_avx_aligned);
 *   tester.summary();  // 打印加速比汇总
 * ===========================================================================
 */

#ifndef BENCHMARK_UTILS_H
#define BENCHMARK_UTILS_H

#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <string>
#include <map>
#include <cmath>
#include <algorithm>

/* =========================================================================
 * 高精度计时器 — 微秒级精度 RAII 包装
 * ========================================================================= */
class HighResClock {
public:
    using TimePoint = std::chrono::high_resolution_clock::time_point;
    using Duration  = std::chrono::duration<double, std::milli>;

    HighResClock() : start_(std::chrono::high_resolution_clock::now()) {}

    /** 重置计时起点 */
    void reset() { start_ = std::chrono::high_resolution_clock::now(); }

    /** 返回自构造/上次 reset 至今的毫秒数 */
    double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<Duration>(end - start_).count();
    }

private:
    TimePoint start_;
};

/* =========================================================================
 * 批量多规模测试器 — 自动遍历多组矩阵尺寸并记录耗时
 * ========================================================================= */
class BatchTester {
public:
    /** 注册一组待测矩阵规模 (方阵维度) */
    BatchTester& add_size(int dim) {
        sizes_.push_back(dim);
        return *this;
    }

    /**
     * 对给定实现 kernel 依次在全部已注册规模上运行并计时
     * @param label    算法名称标签(如 "AVX-Aligned")
     * @param kernel   符合 void(float**, int) 签名的消元函数
     */
    template<typename KernelFunc>
    void run(const std::string& label, KernelFunc kernel) {
        std::vector<double> times;
        std::cout << "\n[" << label << "]\n";
        std::cout << std::setw(10) << "Dim"
                  << std::setw(16) << "Time(ms)"
                  << std::setw(16) << "PerElem(ns)" << "\n";
        std::cout << std::string(42, '-') << "\n";

        for (int dim : sizes_) {
            // --- 分配矩阵 ---
            float** mat = new float*[dim];
            for (int r = 0; r < dim; ++r) {
                mat[r] = new float[dim];
            }
            matrix_fill_random(mat, dim);

            // --- 计时 ---
            HighResClock clock;
            kernel(mat, dim);
            double elapsed = clock.elapsed_ms();

            // --- 计算每元素耗时 (纳秒) ---
            double total_ops = static_cast<double>(dim) * dim * dim;
            double ns_per_elem = (elapsed * 1e6) / total_ops;

            std::cout << std::setw(10) << dim
                      << std::setw(16) << std::fixed << std::setprecision(3)
                      << elapsed
                      << std::setw(16) << std::fixed << std::setprecision(4)
                      << ns_per_elem << "\n";

            times.push_back(elapsed);
            results_[label].push_back(elapsed);

            // --- 释放 ---
            for (int r = 0; r < dim; ++r) delete[] mat[r];
            delete[] mat;
        }
    }

    /** 打印加速比汇总表: 以串行实现为基线, 对比各 SIMD 版本的加速比 */
    void summary(const std::string& baseline_label = "Serial-Naive") {
        if (results_.find(baseline_label) == results_.end()) {
            std::cerr << "[WARN] Baseline '" << baseline_label
                      << "' not found; skip summary.\n";
            return;
        }

        const auto& baseline = results_[baseline_label];
        std::cout << "\n" << std::string(72, '=') << "\n";
        std::cout << " SPEEDUP SUMMARY (Baseline: " << baseline_label << ")\n";
        std::cout << std::string(72, '=') << "\n";

        // 表头
        std::cout << std::setw(10) << "Dim";
        for (const auto& kv : results_) {
            if (kv.first == baseline_label) continue;
            std::cout << std::setw(14) << kv.first;
        }
        std::cout << "\n" << std::string(72, '-') << "\n";

        // 逐规模输出加速比
        for (size_t i = 0; i < sizes_.size(); ++i) {
            std::cout << std::setw(10) << sizes_[i];
            double base_t = baseline[i];
            for (const auto& kv : results_) {
                if (kv.first == baseline_label) continue;
                double speedup = base_t / kv.second[i];
                std::cout << std::setw(13) << std::fixed
                          << std::setprecision(2) << speedup << "x";
            }
            std::cout << "\n";
        }
        std::cout << std::string(72, '=') << "\n";
    }

    /** 获取已记录的规模列表 */
    const std::vector<int>& sizes() const { return sizes_; }

    /** 获取指定标签的计时结果 */
    const std::vector<double>& timing(const std::string& label) const {
        static std::vector<double> empty;
        auto it = results_.find(label);
        return (it != results_.end()) ? it->second : empty;
    }

private:
    std::vector<int> sizes_;
    std::map<std::string, std::vector<double>> results_;
};

#endif /* BENCHMARK_UTILS_H */
