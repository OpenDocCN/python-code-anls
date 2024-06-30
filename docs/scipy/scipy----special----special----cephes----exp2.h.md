# `D:\src\scipysrc\scipy\scipy\special\special\cephes\exp2.h`

```
/* Translated into C++ by SciPy developers in 2024.
 * Original header with Copyright information appears below.
 */

/*                                                     exp2.c
 *
 *     Base 2 exponential function
 *
 *
 *
 * SYNOPSIS:
 *
 * double x, y, exp2();
 *
 * y = exp2( x );
 *
 *
 *
 * DESCRIPTION:
 *
 * Returns 2 raised to the x power.
 *
 * Range reduction is accomplished by separating the argument
 * into an integer k and fraction f such that
 *     x    k  f
 *    2  = 2  2.
 *
 * A Pade' form
 *
 *   1 + 2x P(x**2) / (Q(x**2) - x P(x**2) )
 *
 * approximates 2**x in the basic range [-0.5, 0.5].
 *
 *
 * ACCURACY:
 *
 *                      Relative error:
 * arithmetic   domain     # trials      peak         rms
 *    IEEE    -1022,+1024   30000       1.8e-16     5.4e-17
 *
 *
 * See exp.c for comments on error amplification.
 *
 *
 * ERROR MESSAGES:
 *
 *   message         condition      value returned
 * exp underflow    x < -MAXL2        0.0
 * exp overflow     x > MAXL2         INFINITY
 *
 * For IEEE arithmetic, MAXL2 = 1024.
 */

/*
 * Cephes Math Library Release 2.3:  March, 1995
 * Copyright 1984, 1995 by Stephen L. Moshier
 */
#pragma once

#include "../config.h"

#include "polevl.h"

namespace special {
namespace cephes {

    namespace detail {

        constexpr double exp2_P[] = {
            2.30933477057345225087E-2,
            2.02020656693165307700E1,
            1.51390680115615096133E3,
        };

        constexpr double exp2_Q[] = {
            /* 1.00000000000000000000E0, */
            2.33184211722314911771E2,
            4.36821166879210612817E3,
        };

        constexpr double exp2_MAXL2 = 1024.0;
        constexpr double exp2_MINL2 = -1024.0;

    } // namespace detail

    // 定义 exp2 函数，计算 2 的 x 次方
    SPECFUN_HOST_DEVICE inline double exp2(double x) {
        double px, xx;
        short n;

        // 检查是否为 NaN，若是则返回 x
        if (std::isnan(x)) {
            return (x);
        }
        // 如果 x 超出最大限制，返回无穷大
        if (x > detail::exp2_MAXL2) {
            return (std::numeric_limits<double>::infinity());
        }

        // 如果 x 小于最小限制，返回 0
        if (x < detail::exp2_MINL2) {
            return (0.0);
        }

        xx = x; /* save x */
        // 分离成整数部分和小数部分
        px = std::floor(x + 0.5);
        n = px;
        x = x - px;

        // 有理逼近
        xx = x * x;
        px = x * polevl(xx, detail::exp2_P, 2);
        x = px / (p1evl(xx, detail::exp2_Q, 2) - px);
        x = 1.0 + std::ldexp(x, 1);

        // 按照 2 的幂次方缩放
        x = std::ldexp(x, n);
        return (x);
    }

} // namespace cephes
} // namespace special
```