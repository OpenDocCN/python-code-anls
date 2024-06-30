# `D:\src\scipysrc\scipy\scipy\special\special\cephes\shichi.h`

```
/* Translated into C++ by SciPy developers in 2024.
 * Original header with Copyright information appears below.
 */

/*                                                     shichi.c
 *
 *     Hyperbolic sine and cosine integrals
 *
 *
 *
 * SYNOPSIS:
 *
 * double x, Chi, Shi, shichi();
 *
 * shichi( x, &Chi, &Shi );
 *
 *
 * DESCRIPTION:
 *
 * Approximates the integrals
 *
 *                            x
 *                            -
 *                           | |   cosh t - 1
 *   Chi(x) = eul + ln x +   |    -----------  dt,
 *                         | |          t
 *                          -
 *                          0
 *
 *               x
 *               -
 *              | |  sinh t
 *   Shi(x) =   |    ------  dt
 *            | |       t
 *             -
 *             0
 *
 * where eul = 0.57721566490153286061 is Euler's constant.
 * The integrals are evaluated by power series for x < 8
 * and by Chebyshev expansions for x between 8 and 88.
 * For large x, both functions approach exp(x)/2x.
 * Arguments greater than 88 in magnitude return INFINITY.
 *
 *
 * ACCURACY:
 *
 * Test interval 0 to 88.
 *                      Relative error:
 * arithmetic   function  # trials      peak         rms
 *    IEEE         Shi      30000       6.9e-16     1.6e-16
 *        Absolute error, except relative when |Chi| > 1:
 *    IEEE         Chi      30000       8.4e-16     1.4e-16
 */

/*
 * Cephes Math Library Release 2.0:  April, 1987
 * Copyright 1984, 1987 by Stephen L. Moshier
 * Direct inquiries to 30 Frost Street, Cambridge, MA 02140
 */
#pragma once

#include "../config.h"

#include "chbevl.h"
#include "const.h"

namespace special {
namespace cephes {

    } // namespace detail

    /* Sine and cosine integrals */
    SPECFUN_HOST_DEVICE inline int shichi(double x, double *si, double *ci) {
        double k, z, c, s, a, b;
        short sign;

        // Determine the sign of x and adjust if negative
        if (x < 0.0) {
            sign = -1;
            x = -x;
        } else {
            sign = 0;
        }

        // Handle the special case where x is zero
        if (x == 0.0) {
            *si = 0.0;
            *ci = -std::numeric_limits<double>::infinity();
            return (0);
        }

        // Check if x is in the range requiring Chebyshev expansion
        if (x >= 8.0) {
            goto chb;
        }

        // Check if x is in the asymptotic range
        if (x >= 88.0) {
            goto asymp;
        }

        z = x * x;

        /*     Direct power series expansion   */
        a = 1.0;
        s = 1.0;
        c = 0.0;
        k = 2.0;

        // Power series approximation for small x
        do {
            a *= z / k;
            c += a / k;
            k += 1.0;
            a /= k;
            s += a / k;
            k += 1.0;
        } while (std::abs(a / s) > detail::MACHEP);

        // Final computation based on power series
        s *= x;
        goto done;
    chb:
        /* Chebyshev series expansions */
        // 如果 x 小于 18.0，计算系数 a 和 k，然后使用 Chebyshev 多项式计算 sinh_integral(s_i) 和 cosh_integral(c_i)
        if (x < 18.0) {
            a = (576.0 / x - 52.0) / 10.0;
            k = std::exp(x) / x;
            s = k * chbevl(a, detail::shichi_S1, 22);
            c = k * chbevl(a, detail::shichi_C1, 23);
            // 跳转到 done 标签处
            goto done;
        }

        // 如果 x 不小于 18.0 且小于等于 88.0，计算系数 a 和 k，然后使用 Chebyshev 多项式计算 sinh_integral(s_i) 和 cosh_integral(c_i)
        if (x <= 88.0) {
            a = (6336.0 / x - 212.0) / 70.0;
            k = std::exp(x) / x;
            s = k * chbevl(a, detail::shichi_S2, 23);
            c = k * chbevl(a, detail::shichi_C2, 24);
            // 跳转到 done 标签处
            goto done;
        }

    asymp:
        // 如果 x 大于 1000，直接将结果设置为正无穷
        if (x > 1000) {
            *si = std::numeric_limits<double>::infinity();
            *ci = std::numeric_limits<double>::infinity();
        } else {
            /* Asymptotic expansions
             * 根据渐近展开公式计算 sinh_integral(s_i) 和 cosh_integral(c_i)
             * 参考文献：http://functions.wolfram.com/GammaBetaErf/CoshIntegral/06/02/
             *         http://functions.wolfram.com/GammaBetaErf/SinhIntegral/06/02/0001/
             */
            a = detail::hyp3f0(0.5, 1, 1, 4.0 / (x * x));
            b = detail::hyp3f0(1, 1, 1.5, 4.0 / (x * x));
            *si = std::cosh(x) / x * a + std::sinh(x) / (x * x) * b;
            *ci = std::sinh(x) / x * a + std::cosh(x) / (x * x) * b;
        }
        // 如果 sign 为真，则将结果取反
        if (sign) {
            *si = -*si;
        }
        // 返回结果
        return 0;

    done:
        // 如果 sign 为真，则将 s 取反
        if (sign) {
            s = -s;
        }
        // 将计算得到的 s_i 和 c_i 的值存入 si 和 ci 中
        *si = s;
        *ci = detail::SCIPY_EULER + std::log(x) + c;
        // 返回结果
        return (0);
} // 结束 cephes 命名空间
} // 结束 special 命名空间
```