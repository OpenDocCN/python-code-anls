# `D:\src\scipysrc\scipy\scipy\special\special\hyp2f1.h`

```
/*
 * Implementation of Gauss's hypergeometric function for complex values.
 *
 * This implementation is based on the Fortran implementation by Shanjie Zhang and
 * Jianming Jin included in specfun.f [1]_.  Computation of Gauss's hypergeometric
 * function involves handling a patchwork of special cases. By default the Zhang and Jin
 * implementation has been followed as closely as possible except for situations where
 * an improvement was obvious. We've attempted to document the reasons behind decisions
 * made by Zhang and Jin and to document the reasons for deviating from their implementation
 * when this has been done. References to the NIST Digital Library of Mathematical
 * Functions [2]_ have been added where they are appropriate. The review paper by
 * Pearson et al [3]_ is an excellent resource for best practices for numerical
 * computation of hypergeometric functions. We have followed this review paper
 * when making improvements to and correcting defects in Zhang and Jin's
 * implementation. When Pearson et al propose several competing alternatives for a
 * given case, we've used our best judgment to decide on the method to use.
 *
 * Author: Albert Steppi
 *
 * Distributed under the same license as Scipy.
 *
 * References
 * ----------
 * .. [1] S. Zhang and J.M. Jin, "Computation of Special Functions", Wiley 1996
 * .. [2] NIST Digital Library of Mathematical Functions. http://dlmf.nist.gov/,
 *        Release 1.1.1 of 2021-03-15. F. W. J. Olver, A. B. Olde Daalhuis,
 *        D. W. Lozier, B. I. Schneider, R. F. Boisvert, C. W. Clark, B. R. Miller,
 *        B. V. Saunders, H. S. Cohl, and M. A. McClain, eds.
 * .. [3] Pearson, J.W., Olver, S. & Porter, M.A.
 *        "Numerical methods for the computation of the confluent and Gauss
 *        hypergeometric functions."
 *        Numer Algor 74, 821-866 (2017). https://doi.org/10.1007/s11075-016-0173-0
 * .. [4] Raimundas Vidunas, "Degenerate Gauss Hypergeometric Functions",
 *        Kyushu Journal of Mathematics, 2007, Volume 61, Issue 1, Pages 109-135,
 * .. [5] López, J.L., Temme, N.M. New series expansions of the Gauss hypergeometric
 *        function. Adv Comput Math 39, 349-365 (2013).
 *        https://doi.org/10.1007/s10444-012-9283-y
 * """
 */

#pragma once

// 包含必要的头文件和声明

#include "config.h"     // 包含配置文件
#include "error.h"      // 包含错误处理
#include "tools.h"      // 包含工具函数

#include "binom.h"      // 二项式系数函数
#include "cephes/gamma.h"   // Cephes 库中的 Gamma 函数
#include "cephes/lanczos.h" // Cephes 库中的 Lanczos 函数
#include "cephes/poch.h"    // Cephes 库中的 Pochhammer 符号函数
#include "cephes/hyp2f1.h"  // Cephes 库中的 Gauss's hypergeometric 函数
#include "digamma.h"    // Digamma 函数定义

namespace special {
namespace detail {
    constexpr double hyp2f1_EPS = 1e-15;   // 定义 Gauss's hypergeometric 函数的精度常数
    /* 定义了 hyp2f1_MAXITER 常量，用于控制超几何函数的 Maclaurin 级数的最大迭代次数。
     * 这个值经验性地从 scipy/special/_precompute/hyp2f1_data.py 的测试案例中确定，
     * 以确保级数在合理的精度水平上收敛。原始值为 1500 或 500，现在调整为 3000，可能
     * 根据进一步分析进行调整。 */
    constexpr std::uint64_t hyp2f1_MAXITER = 3000;

    }

    SPECFUN_HOST_DEVICE inline double four_gammas(double u, double v, double w, double x) {
        double result;

        // 确保 |u| >= |v| 和 |w| >= |x|，以保证计算的一般性和准确性。
        if (std::abs(v) > std::abs(u)) {
            std::swap(u, v);
        }
        if (std::abs(x) > std::abs(w)) {
            std::swap(x, w);
        }
        /* 对于在指定范围内的参数，直接使用比率计算可能更为准确。
         * 这个范围基于 scipy/special/_precompute/hyp2f1_data.py 中相关基准的经验选择。 */
        if (std::abs(u) <= 100 && std::abs(v) <= 100 && std::abs(w) <= 100 && std::abs(x) <= 100) {
            result = cephes::Gamma(u) * cephes::Gamma(v) / (cephes::Gamma(w) * cephes::Gamma(x));
            if (std::isfinite(result) && result != 0.0) {
                return result;
            }
        }
        // 若使用 Lanczos 方法计算的结果为有限且非零，则返回该结果。
        result = four_gammas_lanczos(u, v, w, x);
        if (std::isfinite(result) && result != 0.0) {
            return result;
        }
        // 若计算结果溢出或下溢，则尝试使用对数重新计算。
        result = std::exp(cephes::lgam(v) - cephes::lgam(x) + cephes::lgam(u) - cephes::lgam(w));
        result *= cephes::gammasgn(u) * cephes::gammasgn(w) * cephes::gammasgn(v) * cephes::gammasgn(x);
        return result;
    }

    class HypergeometricSeriesGenerator {
        /* hyp2f1 的 Maclaurin 级数。
         *
         * 当 |z| < 1 时级数收敛，但在数值计算中只有当 |z| < 0.9 时才是实用的。 */
      public:
        SPECFUN_HOST_DEVICE HypergeometricSeriesGenerator(double a, double b, double c, std::complex<double> z)
            : a_(a), b_(b), c_(c), z_(z), term_(1.0), k_(0) {}

        SPECFUN_HOST_DEVICE std::complex<double> operator()() {
            std::complex<double> output = term_;
            term_ = term_ * (a_ + k_) * (b_ + k_) / ((k_ + 1) * (c_ + k_)) * z_;
            ++k_;
            return output;
        }

      private:
        double a_, b_, c_;
        std::complex<double> z_, term_;
        std::uint64_t k_;
    };
    class Hyp2f1Transform1Generator {
        /* 1 -z transformation of standard series.*/
      public:
        // 构造函数，接受参数 a, b, c 和复数 z，并初始化成员变量
        SPECFUN_HOST_DEVICE Hyp2f1Transform1Generator(double a, double b, double c, std::complex<double> z)
            : factor1_(four_gammas(c, c - a - b, c - a, c - b)),  // 计算并存储 factor1_
              factor2_(four_gammas(c, a + b - c, a, b) * std::pow(1.0 - z, c - a - b)),  // 计算并存储 factor2_
              generator1_(HypergeometricSeriesGenerator(a, b, a + b - c + 1, 1.0 - z)),  // 初始化 generator1_
              generator2_(HypergeometricSeriesGenerator(c - a, c - b, c - a - b + 1, 1.0 - z)) {}  // 初始化 generator2_

        // 操作符重载，返回两个生成器计算结果的线性组合
        SPECFUN_HOST_DEVICE std::complex<double> operator()() {
            return factor1_ * generator1_() + factor2_ * generator2_();
        }

      private:
        std::complex<double> factor1_, factor2_;
        HypergeometricSeriesGenerator generator1_, generator2_;
    };

    class Hyp2f1Transform1LimitSeriesGenerator {
        /* 1 - z transform in limit as c - a - b approaches an integer m. */
      public:
        // 构造函数，接受参数 a, b, m 和复数 z，并初始化成员变量
        SPECFUN_HOST_DEVICE Hyp2f1Transform1LimitSeriesGenerator(double a, double b, double m, std::complex<double> z)
            : d1_(special::digamma(a)),  // 初始化 d1_ 为 digamma(a)
              d2_(special::digamma(b)),  // 初始化 d2_ 为 digamma(b)
              d3_(special::digamma(1 + m)),  // 初始化 d3_ 为 digamma(1 + m)
              d4_(special::digamma(1.0)),  // 初始化 d4_ 为 digamma(1.0)
              a_(a), b_(b), m_(m), z_(z),  // 存储参数值
              log_1_z_(std::log(1.0 - z)),  // 计算并存储 log(1 - z)
              factor_(1.0 / cephes::Gamma(m + 1)),  // 计算并存储 1 / Gamma(m + 1)
              k_(0) {}  // 初始化 k_

        // 操作符重载，返回一个复数值，通过迭代计算生成器的值
        SPECFUN_HOST_DEVICE std::complex<double> operator()() {
            std::complex<double> term_ = (d1_ + d2_ - d3_ - d4_ + log_1_z_) * factor_;  // 计算 term_
            // 使用 digamma(x + 1) = digamma(x) + 1/x 更新 digamma 值
            d1_ += 1 / (a_ + k_);       // 更新 d1_ 为 digamma(a + k)
            d2_ += 1 / (b_ + k_);       // 更新 d2_ 为 digamma(b + k)
            d3_ += 1 / (1.0 + m_ + k_); // 更新 d3_ 为 digamma(1 + m + k)
            d4_ += 1 / (1.0 + k_);      // 更新 d4_ 为 digamma(1 + k)
            // 更新 factor_ 的值
            factor_ *= (a_ + k_) * (b_ + k_) / ((k_ + 1.0) * (m_ + k_ + 1)) * (1.0 - z_);
            ++k_;  // 增加 k 的值
            return term_;  // 返回 term_
        }

      private:
        double d1_, d2_, d3_, d4_, a_, b_, m_;
        std::complex<double> z_, log_1_z_, factor_;
        int k_;
    };

    class Hyp2f1Transform2Generator {
        /* 1/z transformation of standard series.*/
      public:
        // 构造函数，接受参数 a, b, c 和复数 z，并初始化成员变量
        SPECFUN_HOST_DEVICE Hyp2f1Transform2Generator(double a, double b, double c, std::complex<double> z)
            : factor1_(four_gammas(c, b - a, b, c - a) * std::pow(-z, -a)),  // 计算并存储 factor1_
              factor2_(four_gammas(c, a - b, a, c - b) * std::pow(-z, -b)),  // 计算并存储 factor2_
              generator1_(HypergeometricSeriesGenerator(a, a - c + 1, a - b + 1, 1.0 / z)),  // 初始化 generator1_
              generator2_(HypergeometricSeriesGenerator(b, b - c + 1, b - a + 1, 1.0 / z)) {}  // 初始化 generator2_

        // 操作符重载，返回两个生成器计算结果的线性组合
        SPECFUN_HOST_DEVICE std::complex<double> operator()() {
            return factor1_ * generator1_() + factor2_ * generator2_();
        }

      private:
        std::complex<double> factor1_, factor2_;
        HypergeometricSeriesGenerator generator1_, generator2_;
    };
    class Hyp2f1Transform2LimitSeriesGenerator {
        /* 1/z transform in limit as a - b approaches a non-negative integer m. (Can swap a and b to
         * handle the m a negative integer case. */
      public:
        // 构造函数，初始化类成员
        SPECFUN_HOST_DEVICE Hyp2f1Transform2LimitSeriesGenerator(double a, double b, double c, double m,
                                                                 std::complex<double> z)
            : d1_(special::digamma(1.0)),     // 计算 digamma 函数值 digamma(1.0)
              d2_(special::digamma(1 + m)),   // 计算 digamma 函数值 digamma(1 + m)
              d3_(special::digamma(a)),       // 计算 digamma 函数值 digamma(a)
              d4_(special::digamma(c - a)),   // 计算 digamma 函数值 digamma(c - a)
              a_(a), b_(b), c_(c), m_(m), z_(z),  // 初始化变量
              log_neg_z_(std::log(-z)),       // 计算 -z 的对数
              factor_(special::cephes::poch(b, m) * special::cephes::poch(1 - c + b, m) /
                      special::cephes::Gamma(m + 1)),  // 计算 Pochhammer 符号和 Gamma 函数的乘积
              k_(0) {}  // 初始化 k_ 为 0

        // 运算符重载，计算每次迭代的结果
        SPECFUN_HOST_DEVICE std::complex<double> operator()() {
            // 计算当前 term 的值
            std::complex<double> term = (d1_ + d2_ - d3_ - d4_ + log_neg_z_) * factor_;
            // 更新 digamma 函数的值
            d1_ += 1 / (1.0 + k_);         // d1 = digamma(1 + k)
            d2_ += 1 / (1.0 + m_ + k_);    // d2 = digamma(1 + m + k)
            d3_ += 1 / (a_ + k_);          // d3 = digamma(a + k)
            d4_ -= 1 / (c_ - a_ - k_ - 1); // d4 = digamma(c - a - k)
            // 更新 factor 的值
            factor_ *= (b_ + m_ + k_) * (1 - c_ + b_ + m_ + k_) / ((k_ + 1) * (m_ + k_ + 1)) / z_;
            // 更新迭代计数器 k_
            ++k_;
            return term;  // 返回当前 term 的值
        }

      private:
        double d1_, d2_, d3_, d4_, a_, b_, c_, m_;  // 双精度浮点数和复数变量声明
        std::complex<double> z_, log_neg_z_, factor_;  // 复数变量声明
        std::uint64_t k_;  // 64 位无符号整数变量声明
    };

    };

    class Hyp2f1Transform2LimitFinitePartGenerator {
        /* Initial finite sum in limit as a - b approaches a non-negative integer m. The limiting series
         * for the 1 - z transform also has an initial finite sum, but it is a standard hypergeometric
         * series. */
      public:
        // 构造函数，初始化类成员
        SPECFUN_HOST_DEVICE Hyp2f1Transform2LimitFinitePartGenerator(double b, double c, double m,
                                                                     std::complex<double> z)
            : b_(b), c_(c), m_(m), z_(z),
              term_(cephes::Gamma(m) / cephes::Gamma(c - b)),  // 计算 Gamma 函数的比值
              k_(0) {}  // 初始化 k_ 为 0

        // 运算符重载，计算每次迭代的结果
        SPECFUN_HOST_DEVICE std::complex<double> operator()() {
            // 将当前 term 的值存储到输出中
            std::complex<double> output = term_;
            // 更新 term 的值
            term_ = term_ * (b_ + k_) * (c_ - b_ - k_ - 1) / ((k_ + 1) * (m_ - k_ - 1)) / z_;
            // 更新迭代计数器 k_
            ++k_;
            return output;  // 返回当前 term 的值
        }

      private:
        double b_, c_, m_;  // 双精度浮点数和复数变量声明
        std::complex<double> z_, term_;  // 复数变量声明
        std::uint64_t k_;  // 64 位无符号整数变量声明
    };
    /* 
     * Lopez-Temme Series for Gaussian hypergeometric function [4].
     * 
     * Converges for all z with real(z) < 1, including in the regions surrounding
     * the points exp(+- i*pi/3) that are not covered by any of the standard
     * transformations.
     */
    class LopezTemmeSeriesGenerator {
    public:
        /* 
         * Constructor initializing the Lopez-Temme series generator.
         * 
         * Parameters:
         * - a: parameter 'a' of the hypergeometric function
         * - b: parameter 'b' of the hypergeometric function
         * - c: parameter 'c' of the hypergeometric function
         * - z: complex number 'z' at which the series is evaluated
         * 
         * Initializes internal state variables and precomputes initial values.
         */
        SPECFUN_HOST_DEVICE LopezTemmeSeriesGenerator(double a, double b, double c, std::complex<double> z)
            : n_(0), a_(a), b_(b), c_(c), phi_previous_(1.0), phi_(1 - 2 * b / c), z_(z), Z_(a * z / (z - 2.0)) {}

        /* 
         * Operator function to compute the next term in the series.
         * 
         * Returns:
         * - std::complex<double>: next term in the Lopez-Temme series
         * 
         * Computes the next term based on the current state and increments the series index.
         */
        SPECFUN_HOST_DEVICE std::complex<double> operator()() {
            if (n_ == 0) {
                ++n_;
                return 1.0;
            }
            if (n_ > 1) { // Update phi and Z for n>=2
                double new_phi = ((n_ - 1) * phi_previous_ - (2.0 * b_ - c_) * phi_) / (c_ + (n_ - 1));
                phi_previous_ = phi_;
                phi_ = new_phi;
                Z_ = Z_ * z_ / (z_ - 2.0) * ((a_ + (n_ - 1)) / n_);
            }
            ++n_;
            return Z_ * phi_;
        }

    private:
        std::uint64_t n_;                         // Index of the current term in the series
        double a_, b_, c_;                        // Parameters of the hypergeometric function
        double phi_previous_, phi_;               // Previous and current values of phi in the series
        std::complex<double> z_, Z_;              // Complex numbers z and Z used in computations
    };
    // 定义函数，计算 hyp2f1 变换中限制情况下的第一种变换，其中 c - a - b 接近整数 m
    SPECFUN_HOST_DEVICE std::complex<double> hyp2f1_transform1_limiting_case(double a, double b, double c, double m,
                                                                             std::complex<double> z) {
        // 初始化结果为零
        std::complex<double> result = 0.0;
        // 如果 m 大于等于零
        if (m >= 0) {
            // 如果 m 不等于零
            if (m != 0) {
                // 使用 HypergeometricSeriesGenerator 生成器，计算超几何级数
                auto series_generator = HypergeometricSeriesGenerator(a, b, 1 - m, 1.0 - z);
                // 计算固定长度的级数求和结果
                result += four_gammas(m, c, a + m, b + m) * series_eval_fixed_length(series_generator,
                                                                                     std::complex<double>{0.0, 0.0},
                                                                                     static_cast<std::uint64_t>(m));
            }
            // 计算前置因子，包括幂次、Gamma 函数和 (1 - z)^m 的计算
            std::complex<double> prefactor = std::pow(-1.0, m + 1) * special::cephes::Gamma(c) /
                                             (special::cephes::Gamma(a) * special::cephes::Gamma(b)) *
                                             std::pow(1.0 - z, m);
            // 使用 Hyp2f1Transform1LimitSeriesGenerator 生成器，计算超几何函数 2F1 的变换级数
            auto series_generator = Hyp2f1Transform1LimitSeriesGenerator(a + m, b + m, m, z);
            // 计算级数求和结果
            result += prefactor * series_eval(series_generator, std::complex<double>{0.0, 0.0}, hyp2f1_EPS,
                                              hyp2f1_MAXITER, "hyp2f1");
            // 返回最终计算结果
            return result;
        } else {
            // 如果 m 小于零
            // 计算前置因子，包括幂次、Gamma 函数和 (1 - z)^m 的计算
            result = four_gammas(-m, c, a, b) * std::pow(1.0 - z, m);
            // 使用 HypergeometricSeriesGenerator 生成器，计算超几何级数
            auto series_generator1 = HypergeometricSeriesGenerator(a + m, b + m, 1 + m, 1.0 - z);
            // 计算固定长度的级数求和结果
            result *= series_eval_fixed_length(series_generator1, std::complex<double>{0.0, 0.0},
                                               static_cast<std::uint64_t>(-m));
            // 计算前置因子，包括幂次、Gamma 函数和 (1 - z)^m 的计算
            double prefactor = std::pow(-1.0, m + 1) * special::cephes::Gamma(c) /
                               (special::cephes::Gamma(a + m) * special::cephes::Gamma(b + m));
            // 使用 Hyp2f1Transform1LimitSeriesGenerator 生成器，计算超几何函数 2F1 的变换级数
            auto series_generator2 = Hyp2f1Transform1LimitSeriesGenerator(a, b, -m, z);
            // 计算级数求和结果
            result += prefactor * series_eval(series_generator2, std::complex<double>{0.0, 0.0}, hyp2f1_EPS,
                                              hyp2f1_MAXITER, "hyp2f1");
            // 返回最终计算结果
            return result;
        }
    }
    SPECFUN_HOST_DEVICE std::complex<double> hyp2f1_transform2_limiting_case(double a, double b, double c, double m,
                                                                             std::complex<double> z) {
        // 在 a - b 接近非负整数 m 的极限情况下，进行 1 / z 变换。负整数情况可以通过交换 a 和 b 处理。
        
        // 创建 Hyp2f1Transform2LimitFinitePartGenerator 对象用于生成有限部分的级数
        auto series_generator1 = Hyp2f1Transform2LimitFinitePartGenerator(b, c, m, z);
        
        // 计算结果的第一部分，利用 Gamma 函数和幂运算
        std::complex<double> result = cephes::Gamma(c) / cephes::Gamma(a) * std::pow(-z, -b);
        
        // 使用 series_eval_fixed_length 计算有限部分的级数和
        result *= series_eval_fixed_length(series_generator1, std::complex<double>{0.0, 0.0}, static_cast<std::uint64_t>(m));
        
        // 计算前因子，包括 Gamma 函数和幂运算
        std::complex<double> prefactor = cephes::Gamma(c) / (cephes::Gamma(a) * cephes::Gamma(c - b) * std::pow(-z, a));
        
        // 计算 c - a 的值
        double n = c - a;
        
        // 如果 c - a 接近整数，则使用 Hyp2f1Transform2LimitSeriesCminusAIntGenerator 对象生成级数
        if (abs(n - std::round(n)) < hyp2f1_EPS) {
            auto series_generator2 = Hyp2f1Transform2LimitSeriesCminusAIntGenerator(a, b, c, m, n, z);
            // 使用 series_eval 计算级数和，将其加到结果中
            result += prefactor * series_eval(series_generator2, std::complex<double>{0.0, 0.0}, hyp2f1_EPS,
                                              hyp2f1_MAXITER, "hyp2f1");
            return result;
        }
        
        // 使用 Hyp2f1Transform2LimitSeriesGenerator 对象生成一般情况的级数
        auto series_generator2 = Hyp2f1Transform2LimitSeriesGenerator(a, b, c, m, z);
        
        // 使用 series_eval 计算级数和，将其加到结果中
        result += prefactor *
                  series_eval(series_generator2, std::complex<double>{0.0, 0.0}, hyp2f1_EPS, hyp2f1_MAXITER, "hyp2f1");
        
        // 返回最终计算结果
        return result;
    }
} // namespace detail



// 在 detail 命名空间外部结束
SPECFUN_HOST_DEVICE inline std::complex<double> hyp2f1(double a, double b, double c, std::complex<double> z) {
    // 定义一个特殊函数 hyp2f1，计算超几何函数 2F1(a, b; c; z) 的值，返回复数类型结果

    /* Special Cases
     * -----------------------------------------------------------------------
     * 当 a = 0 或者 b = 0 时，即使 c 是非正整数，结果固定为 1。这遵循 mpmath 的规则。 */
    if (a == 0 || b == 0) {
        return 1.0;
    }
    double z_abs = std::abs(z);
    // 当 z 为 0 时，除非 c 为 0，结果为 1
    if (z_abs == 0) {
        if (c != 0) {
            return 1.0;
        } else {
            // 返回实部为 NAN，虚部为 0，遵循 mpmath 的规则
            return std::complex<double>{std::numeric_limits<double>::quiet_NaN(), 0};
        }
    }
    bool a_neg_int = a == std::trunc(a) && a < 0;
    bool b_neg_int = b == std::trunc(b) && b < 0;
    bool c_non_pos_int = c == std::trunc(c) and c <= 0;
    /* 当 c 是非正整数，并且不满足以下条件之一：
     * (1) a 是整数且满足 c <= a < 0
     * (2) b 是整数且满足 c <= b < 0
     * 当 z = 0，a = 0，或者 b = 0 时，已经在前面处理过。
     * 我们遵循 mpmath 处理 a、b、c 中任何一个为非正整数的特殊情况。参见 [3] 处理退化情况的方式。 */
    if (c_non_pos_int && !((a_neg_int && c <= a && a < 0) || (b_neg_int && c <= b && b < 0))) {
        return std::complex<double>{std::numeric_limits<double>::infinity(), 0};
    }
    /* 当 a 或 b 是负整数时，转化为多项式计算。
     * 如果 a 和 b 都是负整数，确保在较小的 a 或 b 处终止级数。
     * 这是为了处理类似 a < c < b <= 0，a、b、c 都是非正整数的情况，
     * 在 a 处终止将导致一个 0 / 0 形式的项。 */
    std::uint64_t max_degree;
    if (a_neg_int || b_neg_int) {
        if (a_neg_int && b_neg_int) {
            max_degree = a > b ? std::abs(a) : std::abs(b);
        } else if (a_neg_int) {
            max_degree = std::abs(a);
        } else {
            max_degree = std::abs(b);
        }
        if (max_degree <= UINT64_MAX) {
            auto series_generator = detail::HypergeometricSeriesGenerator(a, b, c, z);
            return detail::series_eval_fixed_length(series_generator, std::complex<double>{0.0, 0.0}, max_degree + 1);
        } else {
            set_error("hyp2f1", SF_ERROR_NO_RESULT, NULL);
            return std::complex<double>{std::numeric_limits<double>::quiet_NaN(),
                                        std::numeric_limits<double>::quiet_NaN()};
        }
    }
    // Kummer's Theorem for z = -1; c = 1 + a - b (DLMF 15.4.26)
    // 当 z = -1 且 c = 1 + a - b 时，应用 Kummer 定理计算
    if (std::abs(z + 1.0) < detail::hyp2f1_EPS && std::abs(1 + a - b - c) < detail::hyp2f1_EPS && !c_non_pos_int) {
        return detail::four_gammas(a - b + 1, 0.5 * a + 1, a + 1, 0.5 * a - b + 1);
    }
    std::complex<double> result;


这段代码是一个计算超几何函数 \( {}_2F_1(a, b; c; z) \) 的函数实现，包含了对特殊情况的处理和数学定理的应用。
    // 判断 c - a 是否为负整数
    bool c_minus_a_neg_int = c - a == std::trunc(c - a) && c - a < 0;
    // 判断 c - b 是否为负整数
    bool c_minus_b_neg_int = c - b == std::trunc(c - b) && c - b < 0;
    /* 如果 c - a 或者 c - b 中有一个为负整数，则通过 Euler 超几何转换
     * 求解多项式。参考 DLMF 15.8.1 */
    if (c_minus_a_neg_int || c_minus_b_neg_int) {
        // 确定最大的次数
        max_degree = c_minus_b_neg_int ? std::abs(c - b) : std::abs(c - a);
        // 如果最大次数在 UINT64_MAX 范围内
        if (max_degree <= UINT64_MAX) {
            // 计算 (1 - z)^(c - a - b)
            result = std::pow(1.0 - z, c - a - b);
            // 创建超几何级数生成器
            auto series_generator = detail::HypergeometricSeriesGenerator(c - a, c - b, c, z);
            // 计算超几何级数的固定长度求和
            result *=
                detail::series_eval_fixed_length(series_generator, std::complex<double>{0.0, 0.0}, max_degree + 2);
            // 返回结果
            return result;
        } else {
            // 如果超出最大次数范围，设置错误并返回 NaN
            set_error("hyp2f1", SF_ERROR_NO_RESULT, NULL);
            return std::complex<double>{std::numeric_limits<double>::quiet_NaN(),
                                        std::numeric_limits<double>::quiet_NaN()};
        }
    }
    /* 当实部(z.real())接近1时，且 c <= a + b 时发散。
     * Todo: 实际检查溢出，而不是使用固定的容差，类似 Fortran 原始版本。 */
    if (std::abs(1 - z.real()) < detail::hyp2f1_EPS && z.imag() == 0 && c - a - b <= 0 && !c_non_pos_int) {
        // 返回正无穷大
        return std::complex<double>{std::numeric_limits<double>::infinity(), 0};
    }
    // 当 z == 1.0 且 c - a - b > 0 时，应用 Gauss 的求和定理 (DLMF 15.4.20)
    if (z == 1.0 && c - a - b > 0 && !c_non_pos_int) {
        // 返回四个 Gamma 函数的求和
        return detail::four_gammas(c, c - a - b, c - a, c - b);
    }
    /* 当 |z| < 0.9 且 z.real() >= 0 时，使用 Maclaurin 级数展开。
     * -----------------------------------------------------------------------
     * 应用 Euler 超几何转换 (DLMF 15.8.1) 来减少 a 和 b 的大小，如果可能的话。
     * 我们遵循 Zhang 和 Jin 的实现 [1]，尽管可能有更好的启发式方法来决定何时应用这种转换。
     * 现在这种方式在某些情况下会损失精度。 */
    if (z_abs < 0.9 && z.real() >= 0) {
        // 如果 c - a < a 并且 c - b < b，使用 Euler 超几何转换
        if (c - a < a && c - b < b) {
            // 计算 (1 - z)^(c - a - b)
            result = std::pow(1.0 - z, c - a - b);
            // 创建超几何级数生成器
            auto series_generator = detail::HypergeometricSeriesGenerator(c - a, c - b, c, z);
            // 计算超几何级数的固定长度求和
            result *= detail::series_eval(series_generator, std::complex<double>{0.0, 0.0}, detail::hyp2f1_EPS,
                                          detail::hyp2f1_MAXITER, "hyp2f1");
            // 返回结果
            return result;
        }
        // 创建超几何级数生成器
        auto series_generator = detail::HypergeometricSeriesGenerator(a, b, c, z);
        // 计算超几何级数
        return detail::series_eval(series_generator, std::complex<double>{0.0, 0.0}, detail::hyp2f1_EPS,
                                   detail::hyp2f1_MAXITER, "hyp2f1");
    }
    /* Points near exp(iπ/3), exp(-iπ/3) not handled by any of the standard
     * transformations. Use series of López and Temme [5]. These regions
     * were not correctly handled by Zhang and Jin's implementation.
     * -------------------------------------------------------------------------*/
    /* 处理接近 exp(iπ/3), exp(-iπ/3) 的点，这些点不被标准变换处理。
     * 使用 López 和 Temme 的级数 [5]。这些区域在 Zhang 和 Jin 的实现中处理不正确。
     * -------------------------------------------------------------------------*/
    if (0.9 <= z_abs && z_abs < 1.1 && std::abs(1.0 - z) >= 0.9 && z.real() >= 0) {
        /* This condition for applying Euler Transformation (DLMF 15.8.1)
         * was determined empirically to work better for this case than that
         * used in Zhang and Jin's implementation for |z| < 0.9,
         * real(z) >= 0. */
        /* 这个条件用于应用 Euler 变换 (DLMF 15.8.1)，经验性地确定对于这种情况比 Zhang 和 Jin 的实现中
         * 用于 |z| < 0.9, real(z) >= 0 的条件效果更好。
         */
        if ((c - a <= a && c - b < b) || (c - a < a && c - b <= b)) {
            auto series_generator = detail::LopezTemmeSeriesGenerator(c - a, c - b, c, z);
            result = std::pow(1.0 - 0.5 * z, a - c); // Lopez-Temme prefactor
            result *= detail::series_eval(series_generator, std::complex<double>{0.0, 0.0}, detail::hyp2f1_EPS,
                                          detail::hyp2f1_MAXITER, "hyp2f1");
            return std::pow(1.0 - z, c - a - b) * result; // Euler transform prefactor.
        }
        auto series_generator = detail::LopezTemmeSeriesGenerator(a, b, c, z);
        result = detail::series_eval(series_generator, std::complex<double>{0.0, 0.0}, detail::hyp2f1_EPS,
                                     detail::hyp2f1_MAXITER, "hyp2f1");
        return std::pow(1.0 - 0.5 * z, -a) * result; // Lopez-Temme prefactor.
    }
    /* z/(z - 1) transformation (DLMF 15.8.1). Avoids cancellation issues that
     * occur with Maclaurin series for real(z) < 0.
     * -------------------------------------------------------------------------*/
    /* z/(z - 1) 变换 (DLMF 15.8.1)。避免了在 real(z) < 0 时 Maclaurin 级数出现的抵消问题。
     * -------------------------------------------------------------------------*/
    if (z_abs < 1.1 && z.real() < 0) {
        if (0 < b && b < a && a < c) {
            std::swap(a, b);
        }
        auto series_generator = detail::HypergeometricSeriesGenerator(a, c - b, c, z / (z - 1.0));
        return std::pow(1.0 - z, -a) * detail::series_eval(series_generator, std::complex<double>{0.0, 0.0},
                                                           detail::hyp2f1_EPS, detail::hyp2f1_MAXITER, "hyp2f1");
    }
    /* 1 - z transformation (DLMF 15.8.4). */
    /* 1 - z 变换 (DLMF 15.8.4)。*/
    if (0.9 <= z_abs && z_abs < 1.1) {
        if (std::abs(c - a - b - std::round(c - a - b)) < detail::hyp2f1_EPS) {
            // Removable singularity when c - a - b is an integer. Need to use limiting formula.
            /* 当 c - a - b 是整数时，可移除奇点。需要使用极限公式。*/
            double m = std::round(c - a - b);
            return detail::hyp2f1_transform1_limiting_case(a, b, c, m, z);
        }
        auto series_generator = detail::Hyp2f1Transform1Generator(a, b, c, z);
        return detail::series_eval(series_generator, std::complex<double>{0.0, 0.0}, detail::hyp2f1_EPS,
                                   detail::hyp2f1_MAXITER, "hyp2f1");
    }
    /* 1/z transformation (DLMF 15.8.2). */
    /* 1/z 变换 (DLMF 15.8.2)。*/
    // 如果 a - b 的绝对值与其四舍五入值的差小于 hyp2f1_EPS
    if (std::abs(a - b - std::round(a - b)) < detail::hyp2f1_EPS) {
        // 如果 b 大于 a，则交换它们的值
        if (b > a) {
            std::swap(a, b);
        }
        // 计算四舍五入后的 a - b
        double m = std::round(a - b);
        // 调用特定情况的转换函数，返回结果
        return detail::hyp2f1_transform2_limiting_case(a, b, c, m, z);
    }
    // 创建 Hyp2f1Transform2Generator 对象，用于生成级数
    auto series_generator = detail::Hyp2f1Transform2Generator(a, b, c, z);
    // 调用级数评估函数，返回结果
    return detail::series_eval(series_generator, std::complex<double>{0.0, 0.0}, detail::hyp2f1_EPS,
                               detail::hyp2f1_MAXITER, "hyp2f1");
} // 结束命名空间 special
```