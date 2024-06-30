# `D:\src\scipysrc\scipy\scipy\special\special\cephes\exp10.h`

```
/* Translated into C++ by SciPy developers in 2024.
 * Original header with Copyright information appears below.
 */

/*                                                     exp10.c
 *
 *     Base 10 exponential function
 *      (Common antilogarithm)
 *
 *
 *
 * SYNOPSIS:
 *
 * double x, y, exp10();
 *
 * y = exp10( x );
 *
 *
 *
 * DESCRIPTION:
 *
 * Returns 10 raised to the x power.
 *
 * Range reduction is accomplished by expressing the argument
 * as 10**x = 2**n 10**f, with |f| < 0.5 log10(2).
 * The Pade' form
 *
 *    1 + 2x P(x**2)/( Q(x**2) - P(x**2) )
 *
 * is used to approximate 10**f.
 *
 *
 *
 * ACCURACY:
 *
 *                      Relative error:
 * arithmetic   domain     # trials      peak         rms
 *    IEEE     -307,+307    30000       2.2e-16     5.5e-17
 *
 * ERROR MESSAGES:
 *
 *   message         condition      value returned
 * exp10 underflow    x < -MAXL10        0.0
 * exp10 overflow     x > MAXL10       INFINITY
 *
 * IEEE arithmetic: MAXL10 = 308.2547155599167.
 *
 */

/*
 * Cephes Math Library Release 2.2:  January, 1991
 * Copyright 1984, 1991 by Stephen L. Moshier
 * Direct inquiries to 30 Frost Street, Cambridge, MA 02140
 */
#pragma once

#include "../config.h"
#include "../error.h"

namespace special {
namespace cephes {

    namespace detail {

        constexpr double exp10_P[] = {
            4.09962519798587023075E-2,
            1.17452732554344059015E1,
            4.06717289936872725516E2,
            2.39423741207388267439E3,
        };

        constexpr double exp10_Q[] = {
            /* 1.00000000000000000000E0, */
            8.50936160849306532625E1,
            1.27209271178345121210E3,
            2.07960819286001865907E3,
        };

        /* static double LOG102 = 3.01029995663981195214e-1; */
        constexpr double exp10_LOG210 = 3.32192809488736234787e0;
        constexpr double exp10_LG102A = 3.01025390625000000000E-1;
        constexpr double exp10_LG102B = 4.60503898119521373889E-6;

        /* static double MAXL10 = 38.230809449325611792; */
        constexpr double exp10_MAXL10 = 308.2547155599167;

    } // namespace detail

} // namespace cephes
} // namespace special
    SPECFUN_HOST_DEVICE inline double exp10(double x) {
        double px, xx;  // 声明两个 double 类型的变量 px 和 xx
        short n;  // 声明一个 short 类型的变量 n

        if (std::isnan(x)) {  // 检查 x 是否为 NaN，如果是则返回 x
            return (x);
        }
        if (x > detail::exp10_MAXL10) {  // 如果 x 大于预定义的最大值 exp10_MAXL10
            return (std::numeric_limits<double>::infinity());  // 返回正无穷大
        }

        if (x < -detail::exp10_MAXL10) { /* Would like to use MINLOG but can't */
            set_error("exp10", SF_ERROR_UNDERFLOW, NULL);  // 设置错误类型为 exp10，错误码为 SF_ERROR_UNDERFLOW
            return (0.0);  // 返回 0.0
        }

        /* Express 10**x = 10**g 2**n
         *   = 10**g 10**( n log10(2) )
         *   = 10**( g + n log10(2) )
         */
        px = std::floor(detail::exp10_LOG210 * x + 0.5);  // 计算 10 的 x 次方的近似值的整数部分 px
        n = px;  // 将 px 转换为 short 类型的 n
        x -= px * detail::exp10_LG102A;  // 更新 x 为 x - px * detail::exp10_LG102A
        x -= px * detail::exp10_LG102B;  // 再更新 x 为 x - px * detail::exp10_LG102B

        /* rational approximation for exponential
         * of the fractional part:
         * 10**x = 1 + 2x P(x**2)/( Q(x**2) - P(x**2) )
         */
        xx = x * x;  // 计算 x 的平方
        px = x * polevl(xx, detail::exp10_P, 3);  // 使用 polevl 函数计算指定参数的多项式值
        x = px / (p1evl(xx, detail::exp10_Q, 3) - px);  // 使用 p1evl 函数计算多项式值，然后计算比值并更新 x
        x = 1.0 + std::ldexp(x, 1);  // 计算 10 的 x 次方的近似值

        /* multiply by power of 2 */
        x = std::ldexp(x, n);  // 将 x 乘以 2 的 n 次方

        return (x);  // 返回计算结果 x
    }
} // 结束特定命名空间 `special`
} // 结束整个命名空间 `cephes`
```