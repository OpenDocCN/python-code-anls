# `D:\src\scipysrc\scipy\scipy\special\special\digamma.h`

```
/* Translated from Cython into C++ by SciPy developers in 2024.
 * Original header comment appears below.
 */

/* An implementation of the digamma function for complex arguments.
 *
 * Author: Josh Wilson
 *
 * Distributed under the same license as Scipy.
 *
 * Sources:
 * [1] "The Digital Library of Mathematical Functions", dlmf.nist.gov
 *
 * [2] mpmath (version 0.19), http://mpmath.org
 */

#pragma once

#include "cephes/psi.h"
#include "cephes/zeta.h"
#include "config.h"
#include "error.h"
#include "trig.h"

namespace special {
namespace detail {
    // All of the following were computed with mpmath
    // Location of the positive root
    constexpr double digamma_posroot = 1.4616321449683623;
    // Value of the positive root
    constexpr double digamma_posrootval = -9.2412655217294275e-17;
    // Location of the negative root
    constexpr double digamma_negroot = -0.504083008264455409;
    // Value of the negative root
    constexpr double digamma_negrootval = 7.2897639029768949e-17;

    // Template function to compute digamma using series expansion around a root
    template <typename T>
    SPECFUN_HOST_DEVICE T digamma_zeta_series(T z, double root, double rootval) {
        T res = rootval; // Initialize result with precomputed root value
        T coeff = -1.0;   // Coefficient initialization for series expansion

        z = z - root; // Shift z by the root location
        T term;
        for (int n = 1; n < 100; n++) { // Iterate through series terms
            coeff *= -z; // Update coefficient
            term = coeff * cephes::zeta(n + 1, root); // Compute series term
            res += term; // Accumulate term into result
            // Check convergence based on relative tolerance
            if (std::abs(term) < std::numeric_limits<double>::epsilon() * std::abs(res)) {
                break; // Break loop if convergence criteria met
            }
        }
        return res; // Return computed digamma value
    }

    // Compute digamma(z + n) using forward recurrence relation
    SPECFUN_HOST_DEVICE inline std::complex<double> digamma_forward_recurrence(std::complex<double> z,
                                                                               std::complex<double> psiz, int n) {
        /* Compute digamma(z + n) using digamma(z) using the recurrence
         * relation
         *
         * digamma(z + 1) = digamma(z) + 1/z.
         *
         * See https://dlmf.nist.gov/5.5#E2 */
        std::complex<double> res = psiz; // Initialize result with digamma(z)

        for (int k = 0; k < n; k++) {
            res += 1.0 / (z + static_cast<double>(k)); // Apply recurrence relation
        }
        return res; // Return computed digamma value
    }

    // Compute digamma(z - n) using backward recurrence relation
    SPECFUN_HOST_DEVICE inline std::complex<double> digamma_backward_recurrence(std::complex<double> z,
                                                                                std::complex<double> psiz, int n) {
        /* Compute digamma(z - n) using digamma(z) and a recurrence relation. */
        std::complex<double> res = psiz; // Initialize result with digamma(z)

        for (int k = 1; k < n + 1; k++) {
            res -= 1.0 / (z - static_cast<double>(k)); // Apply recurrence relation
        }
        return res; // Return computed digamma value
    }
    /* 使用渐近级数评估对数gamma函数的复数版本（digamma）。
     * 参考文档 https://dlmf.nist.gov/5.11#E2 */

    // 奇数序列的贝尔努利数，用于渐近级数计算
    double bernoulli2k[] = {
        0.166666666666666667,  -0.0333333333333333333, 0.0238095238095238095, -0.0333333333333333333,
        0.0757575757575757576, -0.253113553113553114,  1.16666666666666667,   -7.09215686274509804,
        54.9711779448621554,   -529.124242424242424,   6192.12318840579710,   -86580.2531135531136,
        1425517.16666666667,   -27298231.0678160920,   601580873.900642368,   -15116315767.0921569};

    // 计算 z 的倒数平方
    std::complex<double> rzz = 1.0 / z / z;
    // 初始 z 的复数因子
    std::complex<double> zfac = 1.0;
    // 临时变量
    std::complex<double> term;
    // 结果变量
    std::complex<double> res;

    // 检查 z 是否为无穷大或 NaN，如果是，早期返回对数值
    if (!(std::isfinite(z.real()) && std::isfinite(z.imag()))) {
        /* 检查是否为无穷大（或 NaN），并进行早期返回。
         * 除以复数无穷大的结果依赖于实现，可能在 C++ stdlib 和 CUDA stdlib 之间有差异。 */
        return std::log(z);
    }

    // 初始结果为 log(z) - 0.5/z
    res = std::log(z) - 0.5 / z;

    // 计算渐近级数的主循环
    for (int k = 1; k < 17; k++) {
        // 更新 zfac
        zfac *= rzz;
        // 计算当前项
        term = -bernoulli2k[k - 1] * zfac / (2 * static_cast<double>(k));
        // 加到结果中
        res += term;
        // 如果当前项的绝对值小于数值限制的 epsilon 乘以结果的绝对值，则退出循环
        if (std::abs(term) < std::numeric_limits<double>::epsilon() * std::abs(res)) {
            break;
        }
    }
    // 返回最终的渐近级数计算结果
    return res;
} // namespace detail


SPECFUN_HOST_DEVICE inline double digamma(double z) {
    /* Wrap Cephes' psi to take advantage of the series expansion around
     * the smallest negative zero.
     */
    // 如果 z 接近于 detail::digamma_negroot，使用特定的级数展开计算
    if (std::abs(z - detail::digamma_negroot) < 0.3) {
        return detail::digamma_zeta_series(z, detail::digamma_negroot, detail::digamma_negrootval);
    }
    // 否则调用 cephes::psi(z) 函数计算 digamma 函数的值
    return cephes::psi(z);
}

SPECFUN_HOST_DEVICE inline float digamma(float z) { return static_cast<float>(digamma(static_cast<double>(z))); }

SPECFUN_HOST_DEVICE inline std::complex<double> digamma(std::complex<double> z) {
    /*
     * Compute the digamma function for complex arguments. The strategy
     * is:
     *
     * - Around the two zeros closest to the origin (posroot and negroot)
     * use a Taylor series with precomputed zero order coefficient.
     * - If close to the origin, use a recurrence relation to step away
     * from the origin.
     * - If close to the negative real axis, use the reflection formula
     * to move to the right halfplane.
     * - If |z| is large (> 16), use the asymptotic series.
     * - If |z| is small, use a recurrence relation to make |z| large
     * enough to use the asymptotic series.
     */
    double absz = std::abs(z);
    std::complex<double> res = 0;
    /* Use the asymptotic series for z away from the negative real axis
     * with abs(z) > smallabsz. */
    int smallabsz = 16;
    /* Use the reflection principle for z with z.real < 0 that are within
     * smallimag of the negative real axis.
     * int smallimag = 6  # unused below except in a comment */

    // 如果 z 是整数且非正数，则处理极点情况
    if (z.real() <= 0.0 && std::ceil(z.real()) == z) {
        // 处理 singularity 错误，并返回 NaN
        set_error("digamma", SF_ERROR_SINGULAR, NULL);
        return {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
    }
    // 如果 z 接近 detail::digamma_negroot，则使用特定的级数展开计算
    if (std::abs(z - detail::digamma_negroot) < 0.3) {
        // 对于第一个负根，使用特定的级数展开计算
        return detail::digamma_zeta_series(z, detail::digamma_negroot, detail::digamma_negrootval);
    }

    // 如果 z 在负实轴附近且 z.imag() 绝对值小于 smallabsz，则使用反射公式
    if (z.real() < 0 and std::abs(z.imag()) < smallabsz) {
        /* Reflection formula for digamma. See
         *
         *https://dlmf.nist.gov/5.5#E4
         */
        // 使用反射公式计算 digamma 函数的值
        res = -M_PI * cospi(z) / sinpi(z);
        z = 1.0 - z;
        absz = std::abs(z);
    }

    // 如果 abs(z) < 0.5，则使用递归关系进行一步迭代以远离极点
    if (absz < 0.5) {
        /* Use one step of the recurrence relation to step away from
         * the pole. */
        res = -1.0 / z;
        z += 1.0;
        absz = std::abs(z);
    }

    // 如果 z 接近 detail::digamma_posroot，则使用特定的级数展开计算
    if (std::abs(z - detail::digamma_posroot) < 0.5) {
        // 对于第一个正根，使用特定的级数展开计算
        res += detail::digamma_zeta_series(z, detail::digamma_posroot, detail::digamma_posrootval);
    } else if (absz > smallabsz) {
        // 如果 abs(z) > smallabsz，使用渐近级数计算 digamma 函数的值
        res += detail::digamma_asymptotic_series(z);
    } else if (z.real() >= 0.0) {
        // 否则，使用递归关系将 z 变大以便使用渐近级数
        double n = std::trunc(smallabsz - absz) + 1;
        std::complex<double> init = detail::digamma_asymptotic_series(z + n);
        res += detail::digamma_backward_recurrence(z + n, init, n);
    } else {
        // 如果 z.real() < 0，absz < smallabsz，并且 z.imag() > smallimag
        // 计算 n，它是 smallabsz - absz 的整数部分减去 1
        double n = std::trunc(smallabsz - absz) - 1;
        
        // 初始值为 digamma_asymptotic_series(z - n) 的复数
        std::complex<double> init = detail::digamma_asymptotic_series(z - n);
        
        // 使用 digamma_forward_recurrence 函数进行前向递归计算
        res += detail::digamma_forward_recurrence(z - n, init, n);
    }
    
    // 返回最终结果 res
    return res;
}

// 关闭 "special" 命名空间

SPECFUN_HOST_DEVICE // 指定函数 digamma 在特定的主机或设备上可用，可能是 GPU 或 CPU

// 计算复数 z 的 digamma 函数值并返回
inline std::complex<float> digamma(std::complex<float> z) {
    // 转换复数 z 为双精度复数，然后计算其 digamma 函数值，并将结果转换回单精度复数返回
    return static_cast<std::complex<float>>(digamma(static_cast<std::complex<double>>(z)));
}

} // 结束 "special" 命名空间


这段代码看起来是在 C++ 中定义了一个特定的函数 `digamma`，涉及到复数和命名空间的使用。
```