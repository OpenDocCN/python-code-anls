# `D:\src\scipysrc\scipy\scipy\special\special\expint.h`

```
#pragma once

// 使用 `#pragma once` 来确保头文件只被编译一次，防止多重包含问题


#include "error.h"
#include "specfun.h"

// 包含自定义的错误处理头文件和特殊函数头文件，用于后续函数调用和错误处理


namespace special {

// 进入特殊函数的命名空间 `special`


template <typename T>
T exp1(T x) {
    // 调用 specfun 命名空间中的 e1xb 函数计算 exp1(x) 的值
    T out = specfun::e1xb(x);
    // 在计算结果上进行特定的转换和无穷处理，错误处理函数 SPECFUN_CONVINF
    SPECFUN_CONVINF("exp1", out);
    // 返回计算结果
    return out;
}

// 模板函数，计算对应类型 T 的 exp1(x)，并进行结果的转换和错误处理


template <typename T>
std::complex<T> exp1(std::complex<T> z) {
    // 调用 specfun 命名空间中的 e1z 函数计算 exp1(z) 的值
    std::complex<T> outz = specfun::e1z(z);
    // 在计算结果上进行复数类型的特定转换和无穷处理，错误处理函数 SPECFUN_ZCONVINF
    SPECFUN_ZCONVINF("exp1", outz);
    // 返回计算结果
    return outz;
}

// 模板函数，计算复数类型 std::complex<T> 的 exp1(z)，并进行结果的转换和错误处理


template <typename T>
T expi(T x) {
    // 调用 specfun 命名空间中的 eix 函数计算 expi(x) 的值
    T out = specfun::eix(x);
    // 在计算结果上进行特定的转换和无穷处理，错误处理函数 SPECFUN_CONVINF
    SPECFUN_CONVINF("expi", out);
    // 返回计算结果
    return out;
}

// 模板函数，计算对应类型 T 的 expi(x)，并进行结果的转换和错误处理


template <typename T>
std::complex<T> expi(std::complex<T> z) {
    // 调用 specfun 命名空间中的 eixz 函数计算 expi(z) 的值
    std::complex<T> outz = specfun::eixz(z);
    // 在计算结果上进行复数类型的特定转换和无穷处理，错误处理函数 SPECFUN_ZCONVINF
    SPECFUN_ZCONVINF("cexpi", outz);
    // 返回计算结果
    return outz;
}

// 模板函数，计算复数类型 std::complex<T> 的 expi(z)，并进行结果的转换和错误处理


namespace detail {

// 进入特殊函数细节的命名空间 `detail`


//
// Compute a factor of the exponential integral E1.
// This is used in scaled_exp1(x) for moderate values of x.
//
// The function uses the continued fraction expansion given in equation 5.1.22
// of Abramowitz & Stegun, "Handbook of Mathematical Functions".
// For n=1, this is
//
//    E1(x) = exp(-x)*C(x)
//
// where C(x), expressed in the notation used in A&S, is the continued fraction
//
//            1    1    1    2    2    3    3
//    C(x) = ---  ---  ---  ---  ---  ---  ---  ...
//           x +  1 +  x +  1 +  x +  1 +  x +
//
// Here, we pull a factor of 1/z out of C(x), so
//
//    E1(x) = (exp(-x)/x)*F(x)
//
// and a bit of algebra gives the continued fraction expansion of F(x) to be
//
//            1    1    1    2    2    3    3
//    F(x) = ---  ---  ---  ---  ---  ---  ---  ...
//           1 +  x +  1 +  x +  1 +  x +  1 +
//
inline double expint1_factor_cont_frac(double x) {
    // 计算指数积分 E1 的一个因子，用于适度值的 x 的 scaled_exp1(x) 函数
    // 使用 Abramowitz & Stegun 的数学函数手册中方程 5.1.22 给出的连分数展开
    // 根据 x 的大小决定截断连分数所用的项数，较大的 x 需要较少的项数
    int m = 20 + (int) (80.0 / x);
    double t0 = 0.0;
    for (int k = m; k > 0; --k) {
        t0 = k / (x + k / (1 + t0));
    }
    // 返回计算结果
    return 1 / (1 + t0);
}

// 内联函数，计算指数积分 E1 的一个因子，根据 Abramowitz & Stegun 的手册使用连分数展开


} // namespace detail

// 结束特殊函数细节的命名空间 `detail`


//
// Scaled version  of the exponential integral E_1(x).
//
// Factor E_1(x) as
//
//    E_1(x) = exp(-x)/x * F(x)
//
// This function computes F(x).
//
// F(x) has the properties:
//  * F(0) = 0
//  * F is increasing on [0, inf)
//  * lim_{x->inf} F(x) = 1.
//
inline double scaled_exp1(double x) {
    if (x < 0) {
        // 对于负数 x，返回非数字 NAN
        return NAN;
    }

    if (x == 0) {
        // 对于 x 等于零，返回零
        return 0.0;
    }

    if (x <= 1) {
        // 对于小的 x，使用简单实现足够准确
        return x * std::exp(x) * exp1(x);
    }

    if (x <= 1250) {
        // 对于适度的 x，使用连分数展开计算
        return detail::expint1_factor_cont_frac(x);
    }

    // 对于大的 x，使用渐近展开，参考 Abramowitz & Stegun 的手册中方程 5.1.51
    return 1 + (-1 + (2 + (-6 + (24 - 120 / x) / x) / x) / x) / x;
}

// 内联函数，计算指数积分 E_1(x) 的 scaled 版本，根据 x 的大小选择不同的计算方法进行计算
# 在特殊命名空间内定义的函数，用于对输入参数进行指数运算并返回结果
inline float scaled_exp1(float x) { return scaled_exp1(static_cast<double>(x)); }

# 结束特殊命名空间的声明
} // namespace special
```