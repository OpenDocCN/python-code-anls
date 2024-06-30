# `D:\src\scipysrc\scipy\scipy\special\special\cephes\kolmogorov.h`

```
/*
 * Translated into C++ by SciPy developers in 2024.
 * Original header with Copyright information appears below.
 */

/*
 * File altered for inclusion in cephes module for Python:
 * Main loop commented out.... */
/*  Travis Oliphant Nov. 1998 */

/*
 * Re Kolmogorov statistics, here is Birnbaum and Tingey's (actually it was already present
 * in Smirnov's paper) formula for the
 * distribution of D+, the maximum of all positive deviations between a
 * theoretical distribution function P(x) and an empirical one Sn(x)
 * from n samples.
 *
 *     +
 *    D  =         sup     [P(x) - S (x)]
 *     n     -inf < x < inf         n
 *
 *
 *                  [n(1-d)]
 *        +            -                    v-1              n-v
 *    Pr{D   > d} =    >    C    d (d + v/n)    (1 - d - v/n)
 *        n            -   n v
 *                    v=0
 *
 * (also equals the following sum, but note the terms may be large and alternating in sign)
 * See Smirnov 1944, Dwass 1959
 *                         n
 *                         -                         v-1              n-v
 *                =  1 -   >         C    d (d + v/n)    (1 - d - v/n)
 *                         -        n v
 *                       v=[n(1-d)]+1
 *
 * [n(1-d)] is the largest integer not exceeding n(1-d).
 * nCv is the number of combinations of n things taken v at a time.
 *
 * Sources:
 * [1] Smirnov, N.V. "Approximate laws of distribution of random variables from empirical data"
 *     Usp. Mat. Nauk, 1944. http://mi.mathnet.ru/umn8798
 * [2] Birnbaum, Z. W. and Tingey, Fred H.
 *     "One-Sided Confidence Contours for Probability Distribution Functions",
 *     Ann. Math. Statist. 1951.  https://doi.org/10.1214/aoms/1177729550
 * [3] Dwass, Meyer, "The Distribution of a Generalized $\mathrm{D}^+_n$ Statistic",
 *     Ann. Math. Statist., 1959.  https://doi.org/10.1214/aoms/1177706085
 * [4] van Mulbregt, Paul, "Computing the Cumulative Distribution Function and Quantiles of the One-sided
 *     Kolmogorov-Smirnov Statistic"
 *     http://arxiv.org/abs/1802.06966
 * [5] van Mulbregt, Paul,  "Computing the Cumulative Distribution Function and Quantiles of the limit of the Two-sided
 *     Kolmogorov-Smirnov Statistic"
 *     https://arxiv.org/abs/1803.00426
 */

#pragma once

#include "../config.h"
#include "../error.h"

#include "const.h"
#include "dd_real.h"
#include "unity.h"

namespace special {
namespace cephes {

    } // namespace detail

    /**
     * @brief Compute the survival function of the Kolmogorov distribution.
     * 
     * @param x The argument at which to compute the survival function.
     * @return The survival function value at x, or NaN if x is NaN.
     */
    SPECFUN_HOST_DEVICE inline double kolmogorov(double x) {
        if (std::isnan(x)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        // Return the survival function value computed by detail::_kolmogorov
        return detail::_kolmogorov(x).sf;
    }

    /**
     * @brief Compute the cumulative distribution function of the Kolmogorov distribution.
     * 
     * @param x The argument at which to compute the cumulative distribution function.
     * @return The cumulative distribution function value at x, or NaN if x is NaN.
     */
    SPECFUN_HOST_DEVICE inline double kolmogc(double x) {
        if (std::isnan(x)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        // Return the cumulative distribution function value computed by detail::_kolmogorov
        return detail::_kolmogorov(x).cdf;
    }

} // namespace cephes
} // namespace special
    // 计算 Kolmogorov 分布的概率密度函数的负值，返回给定 x 的值
    SPECFUN_HOST_DEVICE inline double kolmogp(double x) {
        // 如果 x 是 NaN，则返回 NaN
        if (std::isnan(x)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        // 如果 x 小于等于 0，返回 -0.0
        if (x <= 0) {
            return -0.0;
        }
        // 返回 Kolmogorov 分布在给定 x 的概率密度函数的负值
        return -detail::_kolmogorov(x).pdf;
    }

    /* 
     * Kolmogorov 生存统计的函数反函数，用于双边检验。
     * 寻找满足 kolmogorov(x) = p 的 x 值。
     */
    SPECFUN_HOST_DEVICE inline double kolmogi(double p) {
        // 如果 p 是 NaN，则返回 NaN
        if (std::isnan(p)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        // 调用 detail 命名空间中的函数，返回 Kolmogorov 生存统计的反函数
        return detail::_kolmogi(p, 1 - p);
    }

    /* 
     * Kolmogorov 累积统计的函数反函数，用于双边检验。
     * 寻找满足 kolmogc(x) = p 或 kolmogorov(x) = 1-p 的 x 值。
     */
    SPECFUN_HOST_DEVICE inline double kolmogci(double p) {
        // 如果 p 是 NaN，则返回 NaN
        if (std::isnan(p)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        // 调用 detail 命名空间中的函数，返回 Kolmogorov 累积统计的反函数
        return detail::_kolmogi(1 - p, p);
    }

    // detail 命名空间的结束

    // 返回 Smirnov 分布的生存函数值
    SPECFUN_HOST_DEVICE inline double smirnov(int n, double d) {
        // 如果 d 是 NaN，则返回 NaN
        if (std::isnan(d)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        // 返回 detail 命名空间中的函数计算的 Smirnov 分布的生存函数值
        return detail::_smirnov(n, d).sf;
    }

    // 返回 Smirnov 分布的累积分布函数值
    SPECFUN_HOST_DEVICE inline double smirnovc(int n, double d) {
        // 如果 d 是 NaN，则返回 NaN
        if (std::isnan(d)) {
            return NAN;
        }
        // 返回 detail 命名空间中的函数计算的 Smirnov 分布的累积分布函数值
        return detail::_smirnov(n, d).cdf;
    }

    /* 
     * Smirnov 分布的导数函数。
     * 在 d=1/n 处有一个内部不连续点。
     */
    SPECFUN_HOST_DEVICE inline double smirnovp(int n, double d) {
        // 如果 n 不大于 0 或者 d 不在 [0, 1] 的范围内，则返回 NaN
        if (!(n > 0 && d >= 0.0 && d <= 1.0)) {
            return (std::numeric_limits<double>::quiet_NaN());
        }
        // 当 n 等于 1 时，斜率始终为 -1，即使在 d=1.0 时也是如此
        if (n == 1) {
            return -1.0;
        }
        // 当 d 为 1.0 时，返回 -0.0
        if (d == 1.0) {
            return -0.0;
        }
        // 当 d 为 0.0 时，导数是不连续的，但是从右侧趋近时极限是 -1
        if (d == 0.0) {
            return -1.0;
        }
        // 返回 detail 命名空间中的函数计算的 Smirnov 分布的导数值的负值
        return -detail::_smirnov(n, d).pdf;
    }

    // 返回 Smirnov 分布的反函数值
    SPECFUN_HOST_DEVICE inline double smirnovi(int n, double p) {
        // 如果 p 是 NaN，则返回 NaN
        if (std::isnan(p)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        // 返回 detail 命名空间中的函数计算的 Smirnov 分布的反函数值
        return detail::_smirnovi(n, p, 1 - p);
    }

    // 返回 Smirnov 分布的累积反函数值
    SPECFUN_HOST_DEVICE inline double smirnovci(int n, double p) {
        // 如果 p 是 NaN，则返回 NaN
        if (std::isnan(p)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        // 返回 detail 命名空间中的函数计算的 Smirnov 分布的累积反函数值
        return detail::_smirnovi(n, 1 - p, p);
    }
} // namespace cephes
} // namespace special
```