# `D:\src\scipysrc\scipy\scipy\special\special\sph_bessel.h`

```
/*

Implementation of spherical Bessel functions and modified spherical Bessel
functions of the first and second kinds, as well as their derivatives.

Author: Tadeusz Pudlik

Distributed under the same license as SciPy.

I attempt to correctly handle the edge cases (0 and infinity), but this is
tricky: the values of the functions often depend on the direction in which
the limit is taken. At zero, I follow the convention of numpy (1.9.2),
which treats zero differently depending on its type:

    >>> np.cos(0)/0
    inf
    >>> np.cos(0+0j)/(0+0j)
    inf + nan*j

So, real zero is assumed to be "positive zero", while complex zero has an
unspecified sign and produces nans.  Similarly, complex infinity is taken to
represent the "point at infinity", an ambiguity which for some functions
makes `nan` the correct return value.

Translated to C++ by SciPy developers in 2024.

*/

#pragma once

#include "amos.h" // Include AMOS library for additional functions
#include "error.h" // Include error handling utilities

namespace special {

template <typename T>
T sph_bessel_j(long n, T x) {
    if (std::isnan(x)) { // Check if x is NaN
        return x; // Return NaN if x is NaN
    }

    if (n < 0) { // Check if n is negative
        set_error("spherical_jn", SF_ERROR_DOMAIN, nullptr); // Set error for negative n
        return std::numeric_limits<T>::quiet_NaN(); // Return NaN for invalid n
    }

    if ((x == std::numeric_limits<T>::infinity()) || (x == -std::numeric_limits<T>::infinity())) {
        // Check if x is positive or negative infinity
        return 0; // Return 0 for infinite x
    }

    if (x == 0) { // Check if x is zero
        if (n == 0) {
            return 1; // Return 1 for sph_bessel_j(0, 0)
        }
        return 0; // Return 0 for sph_bessel_j(n, 0) where n != 0
    }

    if ((n > 0) && (n >= x)) { // Check if n > 0 and n >= x
        return std::sqrt(M_PI_2 / x) * cyl_bessel_j(n + 1 / static_cast<T>(2), x);
        // Return the spherical Bessel function using cylindrical Bessel function
    }

    T s0 = std::sin(x) / x; // Compute sin(x)/x
    if (n == 0) {
        return s0; // Return spherical Bessel function of the first kind
    }

    T s1 = (s0 - std::cos(x)) / x; // Compute (sin(x)/x - cos(x))/x
    if (n == 1) {
        return s1; // Return spherical Bessel function of the second kind
    }

    T sn;
    for (int i = 0; i < n - 1; ++i) {
        sn = (2 * i + 3) * s1 / x - s0; // Compute spherical Bessel function recursively
        s0 = s1;
        s1 = sn;
        if (std::isinf(sn)) {
            // Check if sn is infinity
            // Overflow occurred already : terminate recurrence.
            return sn; // Return infinity if overflow occurs
        }
    }

    return sn; // Return the computed spherical Bessel function
}

template <typename T>
std::complex<T> sph_bessel_j(long n, std::complex<T> z) {
    if (std::isnan(std::real(z)) || std::isnan(std::imag(z))) {
        return z; // Return z if either real or imaginary part is NaN
    }

    if (n < 0) { // Check if n is negative
        set_error("spherical_jn", SF_ERROR_DOMAIN, nullptr); // Set error for negative n
        return std::numeric_limits<T>::quiet_NaN(); // Return NaN for invalid n
    }

    if (std::real(z) == std::numeric_limits<T>::infinity() || std::real(z) == -std::numeric_limits<T>::infinity()) {
        // Check if real part of z is positive or negative infinity
        // https://dlmf.nist.gov/10.52.E3
        if (std::imag(z) == 0) {
            return 0; // Return 0 for sph_bessel_j(0, inf) or sph_bessel_j(0, -inf)
        }

        return std::complex<T>(1, 1) * std::numeric_limits<T>::infinity(); // Return infinity for complex z with infinite real part
    }

    if ((std::real(z) == 0) && (std::imag(z) == 0)) { // Check if z is zero
        if (n == 0) {
            return 1; // Return 1 for sph_bessel_j(0, 0)
        }

        return 0; // Return 0 for sph_bessel_j(n, 0) where n != 0
    }

    std::complex<T> out = std::sqrt(static_cast<T>(M_PI_2) / z) * cyl_bessel_j(n + 1 / static_cast<T>(2), z);
    if (std::imag(z) == 0) {
        return std::real(out); // Return real part if imaginary part is small (considered spurious)
    }

    return out; // Return the computed complex spherical Bessel function
}

template <typename T>
T sph_bessel_j_jac(long n, T z) {
    # 如果 n 等于 0，则根据贝塞尔函数性质返回相应的负球贝塞尔函数值
    if (n == 0) {
        return -sph_bessel_j(1, z);
    }

    # 如果 z 等于 0，则根据特定的数值表格提供的结果返回值
    if (z == static_cast<T>(0)) {
        // DLMF 10.51.2 不适用，因此使用 10.51.1 来获得精确的数值
        if (n == 1) {
            return static_cast<T>(1) / static_cast<T>(3);
        }

        # 如果 n 不等于 1，则返回 0
        return 0;
    }

    // 使用 DLMF 10.51.2 公式计算球贝塞尔函数值并返回
    return sph_bessel_j(n - 1, z) - static_cast<T>(n + 1) * sph_bessel_j(n, z) / z;
}

template <typename T>
T sph_bessel_y(long n, T x) {
    T s0, s1, sn;  // 声明变量 s0, s1, sn 用于存储递归计算的中间结果
    int idx;  // 声明变量 idx 用于迭代循环计数

    if (isnan(x)) {  // 检查 x 是否为 NaN，如果是则直接返回 x
        return x;
    }

    if (n < 0) {  // 检查 n 是否小于 0，如果是则设置错误并返回 NaN
        set_error("spherical_yn", SF_ERROR_DOMAIN, nullptr);
        return std::numeric_limits<T>::quiet_NaN();
    }

    if (x < 0) {  // 检查 x 是否小于 0，如果是则返回符号函数乘以对应正数的递归计算结果
        return std::pow(-1, n + 1) * sph_bessel_y(n, -x);
    }

    if (x == std::numeric_limits<T>::infinity() || x == -std::numeric_limits<T>::infinity()) {  // 检查 x 是否为正无穷或负无穷，如果是则返回 0
        return 0;
    }

    if (x == 0) {  // 检查 x 是否为 0，如果是则返回负无穷
        return -std::numeric_limits<T>::infinity();
    }

    s0 = -cos(x) / x;  // 计算第一个递推公式的初始值 s0
    if (n == 0) {  // 如果 n 等于 0，直接返回 s0
        return s0;
    }

    s1 = (s0 - sin(x)) / x;  // 计算第二个递推公式的初始值 s1
    if (n == 1) {  // 如果 n 等于 1，直接返回 s1
        return s1;
    }

    for (idx = 0; idx < n - 1; ++idx) {  // 循环计算递推公式直到 n-1 次
        sn = (2 * idx + 3) * s1 / x - s0;  // 计算下一个 sn 的值
        s0 = s1;  // 更新 s0 为上一个 s1
        s1 = sn;  // 更新 s1 为当前计算的 sn
        if (isinf(sn)) {  // 检查 sn 是否为无穷，如果是则终止递推
            // Overflow occurred already: terminate recurrence.
            return sn;
        }
    }

    return sn;  // 返回最终计算的 sn
}

inline float sph_bessel_y(long n, float x) { return sph_bessel_y(n, static_cast<double>(x)); }  // 将 float 类型转换为 double 类型并调用上面的 sph_bessel_y 函数

template <typename T>
std::complex<T> sph_bessel_y(long n, std::complex<T> z) {
    if (std::isnan(std::real(z)) || std::isnan(std::imag(z))) {  // 检查复数 z 的实部或虚部是否为 NaN，如果是则返回 z
        return z;
    }

    if (n < 0) {  // 检查 n 是否小于 0，如果是则设置错误并返回 NaN
        set_error("spherical_yn", SF_ERROR_DOMAIN, nullptr);
        return std::numeric_limits<T>::quiet_NaN();
    }

    if (std::real(z) == 0 && std::imag(z) == 0) {  // 检查复数 z 是否为零，如果是则返回 NaN
        // https://dlmf.nist.gov/10.52.E2
        return std::numeric_limits<T>::quiet_NaN();
    }

    if (std::real(z) == std::numeric_limits<T>::infinity() || std::real(z) == -std::numeric_limits<T>::infinity()) {  // 检查复数 z 的实部是否为正无穷或负无穷
        // https://dlmf.nist.gov/10.52.E3
        if (std::imag(z) == 0) {  // 如果虚部为 0，返回 0
            return 0;
        }

        return std::complex<T>(1, 1) * std::numeric_limits<T>::infinity();  // 否则返回复数形式的无穷
    }

    return std::sqrt(static_cast<T>(M_PI_2) / z) * cyl_bessel_y(n + 1 / static_cast<T>(2), z);  // 计算复数情况下的球贝塞尔函数
}

template <typename T>
T sph_bessel_y_jac(long n, T x) {
    if (n == 0) {  // 如果 n 等于 0，返回负的 n=1 球贝塞尔函数值
        return -sph_bessel_y(1, x);
    }

    return sph_bessel_y(n - 1, x) - static_cast<T>(n + 1) * sph_bessel_y(n, x) / x;  // 返回球贝塞尔函数的 Jacobian
}

template <typename T>
T sph_bessel_i(long n, T x) {
    if (isnan(x)) {  // 检查 x 是否为 NaN，如果是则返回 x
        return x;
    }

    if (n < 0) {  // 检查 n 是否小于 0，如果是则设置错误并返回 NaN
        set_error("spherical_in", SF_ERROR_DOMAIN, nullptr);
        return std::numeric_limits<T>::quiet_NaN();
    }

    if (x == 0) {  // 如果 x 等于 0
        // https://dlmf.nist.gov/10.52.E1
        if (n == 0) {  // 如果 n 也等于 0，返回 1
            return 1;
        }
        return 0;  // 否则返回 0
    }

    if (isinf(x)) {  // 检查 x 是否为无穷
        // https://dlmf.nist.gov/10.49.E8
        if (x == -std::numeric_limits<T>::infinity()) {  // 如果 x 为负无穷，返回 (-1)^n * 正无穷
            return std::pow(-1, n) * std::numeric_limits<T>::infinity();
        }

        return std::numeric_limits<T>::infinity();  // 否则返回正无穷
    }

    return sqrt(static_cast<T>(M_PI_2) / x) * cyl_bessel_i(n + 1 / static_cast<T>(2), x);  // 计算球贝塞尔函数 I_n(x)
}

template <typename T>
std::complex<T> sph_bessel_i(long n, std::complex<T> z) {
    if (std::isnan(std::real(z)) || std::isnan(std::imag(z))) {  // 检查复数 z 的实部或虚部是否为 NaN，如果是则返回 z
        return z;
    }

    // 如果 n 小于 0，设置错误信息并返回 NaN
    if (n < 0) {
        set_error("spherical_in", SF_ERROR_DOMAIN, nullptr);
        return std::numeric_limits<T>::quiet_NaN();
    }

    // 如果 z 的绝对值为 0
    if (std::abs(z) == 0) {
        // 返回特定情况下的值，参考链接：https://dlmf.nist.gov/10.52.E1
        if (n == 0) {
            return 1;
        }

        return 0;
    }

    // 如果 z 的实部或虚部为无穷大
    if (std::isinf(std::real(z)) || std::isinf(std::imag(z))) {
        // 参考链接：https://dlmf.nist.gov/10.52.E5
        if (std::imag(z) == 0) {
            // 如果虚部为 0
            if (std::real(z) == -std::numeric_limits<T>::infinity()) {
                return std::pow(-1, n) * std::numeric_limits<T>::infinity();
            }

            return std::numeric_limits<T>::infinity();
        }

        // 如果虚部不为 0，则返回 NaN
        return std::numeric_limits<T>::quiet_NaN();
    }

    // 默认情况，返回计算得到的值
    return std::sqrt(static_cast<T>(M_PI_2) / z) * cyl_bessel_i(n + 1 / static_cast<T>(2), z);
// 命名空间特殊函数包含了特殊数学函数的实现

template <typename T>
T sph_bessel_i_jac(long n, T z) {
    // 如果 n 等于 0，返回调用 sph_bessel_i 函数计算的第一类球贝塞尔函数的 Jacobi 形式
    if (n == 0) {
        return sph_bessel_i(1, z);
    }

    // 如果 z 等于 0
    if (z == static_cast<T>(0)) {
        // 如果 n 等于 1，返回 1/3
        if (n == 1) {
            return 1./3.;
        }
        else {
            // 否则返回 0
            return 0;
        }
    }

    // 返回第一类球贝塞尔函数的 Jacobi 形式递归计算结果
    return sph_bessel_i(n - 1, z) - static_cast<T>(n + 1) * sph_bessel_i(n, z) / z;
}

template <typename T>
T sph_bessel_k(long n, T z) {
    // 如果 z 是 NaN，返回 z
    if (isnan(z)) {
        return z;
    }

    // 如果 n 小于 0，设置错误并返回 NaN
    if (n < 0) {
        set_error("spherical_kn", SF_ERROR_DOMAIN, nullptr);
        return std::numeric_limits<T>::quiet_NaN();
    }

    // 如果 z 等于 0，返回无穷大
    if (z == 0) {
        return std::numeric_limits<T>::infinity();
    }

    // 如果 z 是无穷大
    if (isinf(z)) {
        // 根据情况返回特定值
        // https://dlmf.nist.gov/10.52.E6
        if (z == std::numeric_limits<T>::infinity()) {
            return 0;
        }

        return -std::numeric_limits<T>::infinity();
    }

    // 返回第二类球贝塞尔函数乘以开根号的结果
    return std::sqrt(M_PI_2 / z) * cyl_bessel_k(n + 1 / static_cast<T>(2), z);
}

template <typename T>
std::complex<T> sph_bessel_k(long n, std::complex<T> z) {
    // 如果 z 的实部或虚部是 NaN，返回 z
    if (std::isnan(std::real(z)) || std::isnan(std::imag(z))) {
        return z;
    }

    // 如果 n 小于 0，设置错误并返回 NaN
    if (n < 0) {
        set_error("spherical_kn", SF_ERROR_DOMAIN, nullptr);
        return std::numeric_limits<T>::quiet_NaN();
    }

    // 如果 z 的绝对值等于 0，返回 NaN
    if (std::abs(z) == 0) {
        return std::numeric_limits<T>::quiet_NaN();
    }

    // 如果 z 的实部或虚部是无穷大
    if (std::isinf(std::real(z)) || std::isinf(std::imag(z))) {
        // 根据情况返回特定值
        // https://dlmf.nist.gov/10.52.E6
        if (std::imag(z) == 0) {
            if (std::real(z) == std::numeric_limits<T>::infinity()) {
                return 0;
            }

            return -std::numeric_limits<T>::infinity();
        }

        return std::numeric_limits<T>::quiet_NaN();
    }

    // 返回第二类球贝塞尔函数乘以开根号的结果
    return std::sqrt(static_cast<T>(M_PI_2) / z) * cyl_bessel_k(n + 1 / static_cast<T>(2), z);
}

template <typename T>
T sph_bessel_k_jac(long n, T x) {
    // 如果 n 等于 0，返回负的 sph_bessel_k 函数计算的结果
    if (n == 0) {
        return -sph_bessel_k(1, x);
    }

    // 返回负的 sph_bessel_k 函数的 Jacobi 形式递归计算结果
    return -sph_bessel_k(n - 1, x) - static_cast<T>(n + 1) * sph_bessel_k(n, x) / x;
}

} // namespace special
```