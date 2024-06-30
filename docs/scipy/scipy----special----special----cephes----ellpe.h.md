# `D:\src\scipysrc\scipy\scipy\special\special\cephes\ellpe.h`

```
/*
 * Translated into C++ by SciPy developers in 2024.
 * Original header with Copyright information appears below.
 */

/*
 *                                                     ellpe.c
 *
 *     Complete elliptic integral of the second kind
 *
 *
 *
 * SYNOPSIS:
 *
 * double m, y, ellpe();
 *
 * y = ellpe( m );
 *
 *
 *
 * DESCRIPTION:
 *
 * Approximates the integral
 *
 *
 *            pi/2
 *             -
 *            | |                 2
 * E(m)  =    |    sqrt( 1 - m sin t ) dt
 *          | |
 *           -
 *            0
 *
 * Where m = 1 - m1, using the approximation
 *
 *      P(x)  -  x log x Q(x).
 *
 * Though there are no singularities, the argument m1 is used
 * internally rather than m for compatibility with ellpk().
 *
 * E(1) = 1; E(0) = pi/2.
 *
 *
 * ACCURACY:
 *
 *                      Relative error:
 * arithmetic   domain     # trials      peak         rms
 *    IEEE       0, 1       10000       2.1e-16     7.3e-17
 *
 *
 * ERROR MESSAGES:
 *
 *   message         condition      value returned
 * ellpe domain      x<0, x>1            0.0
 *
 */

/*
 *                                                     ellpe.c
 */

/* Elliptic integral of second kind */

/*
 * Cephes Math Library, Release 2.1:  February, 1989
 * Copyright 1984, 1987, 1989 by Stephen L. Moshier
 * Direct inquiries to 30 Frost Street, Cambridge, MA 02140
 *
 * Feb, 2002:  altered by Travis Oliphant
 * so that it is called with argument m
 * (which gets immediately converted to m1 = 1-m)
 */

#pragma once

#include "../config.h"
#include "../error.h"
#include "polevl.h"

namespace special {
namespace cephes {

    namespace detail {

        // Polynomial coefficients for the approximation of E(m)
        constexpr double ellpe_P[] = {1.53552577301013293365E-4, 2.50888492163602060990E-3, 8.68786816565889628429E-3,
                                      1.07350949056076193403E-2, 7.77395492516787092951E-3, 7.58395289413514708519E-3,
                                      1.15688436810574127319E-2, 2.18317996015557253103E-2, 5.68051945617860553470E-2,
                                      4.43147180560990850618E-1, 1.00000000000000000299E0};

        // Polynomial coefficients for the approximation of E(m)
        constexpr double ellpe_Q[] = {3.27954898576485872656E-5, 1.00962792679356715133E-3, 6.50609489976927491433E-3,
                                      1.68862163993311317300E-2, 2.61769742454493659583E-2, 3.34833904888224918614E-2,
                                      4.27180926518931511717E-2, 5.85936634471101055642E-2, 9.37499997197644278445E-2,
                                      2.49999999999888314361E-1};

    } // namespace detail

} // namespace cephes
} // namespace special
    // 定义一个内联函数 ellpe，计算椭圆积分的第二类不完全积分值
    SPECFUN_HOST_DEVICE inline double ellpe(double x) {
        // 将 x 转换为 1.0 - x，以便在计算中使用标准形式
        x = 1.0 - x;
        // 如果 x 小于等于 0.0，处理特殊情况
        if (x <= 0.0) {
            // 如果 x 等于 0.0，直接返回 1.0
            if (x == 0.0)
                return (1.0);
            // 设置错误信息并返回 NaN
            set_error("ellpe", SF_ERROR_DOMAIN, NULL);
            return (std::numeric_limits<double>::quiet_NaN());
        }
        // 如果 x 大于 1.0，利用公式 ellpe(1.0 - 1 / x) * std::sqrt(x) 递归计算
        if (x > 1.0) {
            return ellpe(1.0 - 1 / x) * std::sqrt(x);
        }
        // 对于 0 < x <= 1 的情况，计算椭圆积分的第二类不完全积分值
        return (polevl(x, detail::ellpe_P, 10) - std::log(x) * (x * polevl(x, detail::ellpe_Q, 9)));
    }
} // 结束 cephes 命名空间
} // 结束 special 命名空间
```