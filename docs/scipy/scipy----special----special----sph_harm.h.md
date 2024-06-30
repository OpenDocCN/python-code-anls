# `D:\src\scipysrc\scipy\scipy\special\special\sph_harm.h`

```
#pragma once

// 包含自定义的错误处理头文件
#include "error.h"
// 包含 Legendre 函数头文件
#include "legendre.h"
// 包含特殊函数头文件
#include "specfun.h"
// 包含 Kokkos 库中的多维数组支持头文件
#include "third_party/kokkos/mdspan.hpp"

// 包含 Cephes 库中的 Pochhammer 函数声明
#include "cephes/poch.h"

namespace special {

// 计算球谐函数 Y_lm(theta, phi) 的值
template <typename T>
std::complex<T> sph_harm(long m, long n, T theta, T phi) {
    // 如果 n 小于 0，设置错误并返回 NaN
    if (n < 0) {
        set_error("sph_harm", SF_ERROR_ARG, "n should not be negative");
        return NAN;
    }

    // 计算 m 的绝对值
    long m_abs = std::abs(m);
    // 如果 m 的绝对值大于 n，则返回 0
    if (m_abs > n) {
        return 0;
    }

    // 计算球谐函数的值
    std::complex<T> val = pmv(m_abs, n, std::cos(phi));
    // 如果 m 小于 0，应用相应的修正
    if (m < 0) {
        val *= std::pow(-1, m_abs) * cephes::poch(n + m_abs + 1, -2 * m_abs);
    }

    // 应用归一化系数和相位因子
    val *= std::sqrt((2 * n + 1) * cephes::poch(n + m + 1, -2 * m) / (4 * M_PI));
    val *= std::exp(std::complex(static_cast<T>(0), m * theta));

    return val;
}

// 计算所有球谐函数 Y_lm(theta, phi) 的值并存储在输出矩阵 y 中
template <typename T, typename OutMat>
void sph_harm_all(T theta, T phi, OutMat y) {
    // 确定输入矩阵 y 的维度
    long m = (y.extent(0) - 1) / 2;
    long n = y.extent(1) - 1;

    // 提取 y 矩阵的正值区域并计算 Legendre 函数值
    OutMat y_pos = std::submdspan(y, std::make_tuple(0, m + 1), std::full_extent);
    sph_legendre_all(phi, y_pos);

    // 计算所有球谐函数 Y_lm(theta, phi) 的值并应用相位修正
    for (long j = 0; j <= n; ++j) {
        for (long i = 1; i <= j; ++i) {
            y(i, j) *= std::exp(std::complex(static_cast<T>(0), i * theta));
            y(2 * m + 1 - i, j) = static_cast<T>(std::pow(-1, i)) * std::conj(y(i, j));
        }
        // 对于 j 大于 m 的部分，设置为 0
        for (long i = j + 1; i <= m; ++i) {
            y(2 * m + 1 - i, j) = 0;
        }
    }
}

} // namespace special


这些注释确保了每行代码的功能和上下文都清晰地被记录下来，使得代码的读者能够快速理解代码的作用和实现细节。
```