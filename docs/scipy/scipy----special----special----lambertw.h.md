# `D:\src\scipysrc\scipy\scipy\special\special\lambertw.h`

```
/* Translated from Cython into C++ by SciPy developers in 2023.
 * Original header with Copyright information appears below.
 */

/* Implementation of the Lambert W function [1]. Based on MPMath
 *  Implementation [2], and documentation [3].
 *
 * Copyright: Yosef Meller, 2009
 * Author email: mellerf@netvision.net.il
 *
 * Distributed under the same license as SciPy
 *
 *
 * References:
 * [1] On the Lambert W function, Adv. Comp. Math. 5 (1996) 329-359,
 *     available online: https://web.archive.org/web/20230123211413/https://cs.uwaterloo.ca/research/tr/1993/03/W.pdf
 * [2] mpmath source code,
 https://github.com/mpmath/mpmath/blob/c5939823669e1bcce151d89261b802fe0d8978b4/mpmath/functions/functions.py#L435-L461
 * [3]
 https://web.archive.org/web/20230504171447/https://mpmath.org/doc/current/functions/powers.html#lambert-w-function
 *

 * TODO: use a series expansion when extremely close to the branch point
 * at `-1/e` and make sure that the proper branch is chosen there.
 */

#pragma once

#include "config.h"
#include "error.h"
#include "evalpoly.h"

namespace special {
constexpr double EXPN1 = 0.36787944117144232159553; // exp(-1)
constexpr double OMEGA = 0.56714329040978387299997; // W(1, 0)

namespace detail {
    SPECFUN_HOST_DEVICE inline std::complex<double> lambertw_branchpt(std::complex<double> z) {
        // Series for W(z, 0) around the branch point; see 4.22 in [1].
        double coeffs[] = {-1.0 / 3.0, 1.0, -1.0};
        // Compute sqrt(2 * (e^z + 1))
        std::complex<double> p = std::sqrt(2.0 * (M_E * z + 1.0));

        return cevalpoly(coeffs, 2, p);
    }

    SPECFUN_HOST_DEVICE inline std::complex<double> lambertw_pade0(std::complex<double> z) {
        // (3, 2) Pade approximation for W(z, 0) around 0.
        double num[] = {12.85106382978723404255, 12.34042553191489361902, 1.0};
        double denom[] = {32.53191489361702127660, 14.34042553191489361702, 1.0};

        /* This only gets evaluated close to 0, so we don't need a more
         * careful algorithm that avoids overflow in the numerator for
         * large z. */
        // Compute z * Pade approximation numerator divided by Pade approximation denominator
        return z * cevalpoly(num, 2, z) / cevalpoly(denom, 2, z);
    }

    SPECFUN_HOST_DEVICE inline std::complex<double> lambertw_asy(std::complex<double> z, long k) {
        /* Compute the W function using the first two terms of the
         * asymptotic series. See 4.20 in [1].
         */
        // Compute log(z) + 2πik
        std::complex<double> w = std::log(z) + 2.0 * M_PI * k * std::complex<double>(0, 1);
        // Compute W using asymptotic approximation
        return w - std::log(w);
    }

} // namespace detail

SPECFUN_HOST_DEVICE inline std::complex<double> lambertw(std::complex<double> z, long k, double tol) {
    double absz;
    std::complex<double> w;
    std::complex<double> ew, wew, wewz, wn;

    if (std::isnan(z.real()) || std::isnan(z.imag())) {
        return z; // Return z if it contains NaN
    }
    if (z.real() == std::numeric_limits<double>::infinity()) {
        return z + 2.0 * M_PI * k * std::complex<double>(0, 1); // Return z + 2πik if real part is infinity
    }
    // 如果 z 是负无穷，则返回特定复数值，根据数学定义进行计算
    if (z.real() == -std::numeric_limits<double>::infinity()) {
        return -z + (2.0 * M_PI * k + M_PI) * std::complex<double>(0, 1);
    }
    // 如果 z 等于 0
    if (z == 0.0) {
        // 如果 k 等于 0，则返回 0
        if (k == 0) {
            return z;
        }
        // 否则，设置错误状态并返回负无穷
        set_error("lambertw", SF_ERROR_SINGULAR, NULL);
        return -std::numeric_limits<double>::infinity();
    }
    // 如果 z 等于 1 并且 k 等于 0，返回预定义的常数 OMEGA
    if (z == 1.0 && k == 0) {
        // 将此情况分开处理，因为渐近级数会发散
        return OMEGA;
    }

    // 计算 z 的绝对值
    absz = std::abs(z);
    // 根据 k 的不同值，获取 Halley 方法的初始猜测
    if (k == 0) {
        // 如果 z + EXPN1 的绝对值小于 0.3，使用分支点函数计算 Lambert W 函数
        if (std::abs(z + EXPN1) < 0.3) {
            w = detail::lambertw_branchpt(z);
        } else if (-1.0 < z.real() && z.real() < 1.5 && std::abs(z.imag()) < 1.0 &&
                   -2.5 * std::abs(z.imag()) - 0.2 < z.real()) {
            // 在经验上确定的决策边界，Pade逼近更加准确的区域
            w = detail::lambertw_pade0(z);
        } else {
            // 否则使用渐近级数计算 Lambert W 函数
            w = detail::lambertw_asy(z, k);
        }
    } else if (k == -1) {
        // 如果 absz 小于等于 EXPN1，并且 z 的虚部为 0，并且 z 的实部小于 0，使用对数函数计算 Lambert W 函数
        if (absz <= EXPN1 && z.imag() == 0.0 && z.real() < 0.0) {
            w = std::log(-z.real());
        } else {
            // 否则使用渐近级数计算 Lambert W 函数
            w = detail::lambertw_asy(z, k);
        }
    } else {
        // 对于其他情况，使用渐近级数计算 Lambert W 函数
        w = detail::lambertw_asy(z, k);
    }

    // Halley 方法迭代求解 Lambert W 函数的实部为正的情况
    if (w.real() >= 0) {
        // 重新排列公式以避免在指数函数中溢出
        for (int i = 0; i < 100; i++) {
            ew = std::exp(-w);
            wewz = w - z * ew;
            wn = w - wewz / (w + 1.0 - (w + 2.0) * wewz / (2.0 * w + 2.0));
            // 如果迭代收敛，则返回结果
            if (std::abs(wn - w) <= tol * std::abs(wn)) {
                return wn;
            }
            // 更新 w 的值进行下一次迭代
            w = wn;
        }
    } else {
        // Halley 方法迭代求解 Lambert W 函数的实部为负的情况
        for (int i = 0; i < 100; i++) {
            ew = std::exp(w);
            wew = w * ew;
            wewz = wew - z;
            wn = w - wewz / (wew + ew - (w + 2.0) * wewz / (2.0 * w + 2.0));
            // 如果迭代收敛，则返回结果
            if (std::abs(wn - w) <= tol * std::abs(wn)) {
                return wn;
            }
            // 更新 w 的值进行下一次迭代
            w = wn;
        }
    }

    // 如果迭代未收敛，设置错误状态并返回 NaN 值的复数
    set_error("lambertw", SF_ERROR_SLOW, "iteration failed to converge: %g + %gj", z.real(), z.imag());
    return {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
}

// 结束 special 命名空间

SPECFUN_HOST_DEVICE inline std::complex<float> lambertw(std::complex<float> z, long k, float tol) {
    // 调用 lambertw 函数，使用双精度版本的 z、tol 参数，返回结果转换为单精度复数
    return static_cast<std::complex<float>>(
        lambertw(static_cast<std::complex<double>>(z), k, static_cast<double>(tol)));
}

} // namespace special
```