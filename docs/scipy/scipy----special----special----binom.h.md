# `D:\src\scipysrc\scipy\scipy\special\special\binom.h`

```
// Translated from Cython into C++ by SciPy developers in 2024.
// Original authors: Pauli Virtanen, Eric Moore

// Binomial coefficient

#pragma once

#include "config.h"

#include "cephes/beta.h"
#include "cephes/gamma.h"

namespace special {

// 定义一个在特定环境中可运行的函数，计算双精度浮点数的二项式系数
SPECFUN_HOST_DEVICE inline double binom(double n, double k) {
    double kx, nx, num, den, dk, sgn;

    // 如果 n 小于 0，处理特殊情况
    if (n < 0) {
        nx = std::floor(n);
        // 如果 n 是整数且小于 0，返回 NaN
        if (n == nx) {
            return std::numeric_limits<double>::quiet_NaN();
        }
    }

    // 取 k 的整数部分
    kx = std::floor(k);
    // 如果 k 是整数并且 n 的绝对值大于 1E-8 或者 n 等于 0，使用乘法公式以减少舍入误差
    if (k == kx && (std::abs(n) > 1E-8 || n == 0)) {
        nx = std::floor(n);
        // 如果 n 是整数且 kx 大于 nx 的一半且 nx 大于 0，通过对称性减小 kx
        if (nx == n && kx > nx / 2 && nx > 0) {
            kx = nx - kx;
        }

        // 如果 kx 在合理范围内，使用乘法公式计算二项式系数
        if (kx >= 0 && kx < 20) {
            num = 1.0;
            den = 1.0;
            for (int i = 1; i < 1 + static_cast<int>(kx); i++) {
                num *= i + n - kx;
                den *= i;
                // 控制数值的大小，防止溢出
                if (std::abs(num) > 1E50) {
                    num /= den;
                    den = 1.0;
                }
            }
            return num / den;
        }
    }

    // 一般情况下的计算
    if (n >= 1E10 * k && k > 0) {
        // 避免中间结果的溢出或下溢
        return std::exp(-cephes::lbeta(1 + n - k, 1 + k) - std::log(n + 1));
    }
    if (k > 1E8 * std::abs(n)) {
        // 避免精度损失
        num = cephes::Gamma(1 + n) / std::abs(k) + cephes::Gamma(1 + n) * n / (2 * k * k); // + ...
        num /= M_PI * std::pow(std::abs(k), n);
        if (k > 0) {
            kx = std::floor(k);
            if (static_cast<int>(kx) == kx) {
                dk = k - kx;
                sgn = (static_cast<int>(kx) % 2 == 0) ? 1 : -1;
            } else {
                dk = k;
                sgn = 1;
            }
            return num * std::sin((dk - n) * M_PI) * sgn;
        }
        kx = std::floor(k);
        if (static_cast<int>(kx) == kx) {
            return 0;
        }
        return num * std::sin(k * M_PI);
    }

    // 如果以上情况均不满足，返回一般情况下的二项式系数计算结果
    return 1 / (n + 1) / cephes::beta(1 + n - k, 1 + k);
}

// 定义一个在特定环境中可运行的函数，将单精度浮点数参数转换为双精度后调用双精度版本的 binom 函数
SPECFUN_HOST_DEVICE inline float binom(float n, float k) {
    return binom(static_cast<double>(n), static_cast<double>(k));
}

} // namespace special
```