# `D:\src\scipysrc\scipy\scipy\special\special\cephes\gdtr.h`

```
/* Translated into C++ by SciPy developers in 2024.
 * Original header with Copyright information appears below.
 */

/*                                                     gdtr.c
 *
 *     Gamma distribution function
 *
 *
 *
 * SYNOPSIS:
 *
 * double a, b, x, y, gdtr();
 *
 * y = gdtr( a, b, x );
 *
 *
 *
 * DESCRIPTION:
 *
 * Returns the integral from zero to x of the Gamma probability
 * density function:
 *
 *
 *                x
 *        b       -
 *       a       | |   b-1  -at
 * y =  -----    |    t    e    dt
 *       -     | |
 *      | (b)   -
 *               0
 *
 *  The incomplete Gamma integral is used, according to the
 * relation
 *
 * y = igam( b, ax ).
 *
 *
 * ACCURACY:
 *
 * See igam().
 *
 * ERROR MESSAGES:
 *
 *   message         condition      value returned
 * gdtr domain         x < 0            0.0
 *
 */
/*                            gdtrc.c
 *
 *    Complemented Gamma distribution function
 *
 *
 *
 * SYNOPSIS:
 *
 * double a, b, x, y, gdtrc();
 *
 * y = gdtrc( a, b, x );
 *
 *
 *
 * DESCRIPTION:
 *
 * Returns the integral from x to infinity of the Gamma
 * probability density function:
 *
 *
 *               inf.
 *        b       -
 *       a       | |   b-1  -at
 * y =  -----    |    t    e    dt
 *       -     | |
 *      | (b)   -
 *               x
 *
 *  The incomplete Gamma integral is used, according to the
 * relation
 *
 * y = igamc( b, ax ).
 *
 *
 * ACCURACY:
 *
 * See igamc().
 *
 * ERROR MESSAGES:
 *
 *   message         condition      value returned
 * gdtrc domain         x < 0            0.0
 *
 */

/*                                                     gdtr()  */

/*
 * Cephes Math Library Release 2.3:  March,1995
 * Copyright 1984, 1987, 1995 by Stephen L. Moshier
 */
#pragma once

#include "../config.h"
#include "../error.h"

#include "igam.h"
#include "igami.h"

namespace special {
namespace cephes {

    SPECFUN_HOST_DEVICE inline double gdtr(double a, double b, double x) {

        // 检查输入参数是否合法，若 x 小于 0 则返回 NaN
        if (x < 0.0) {
            sf_error("gdtr", SF_ERROR_DOMAIN, NULL);
            return (std::numeric_limits<double>::quiet_NaN());
        }
        // 调用 igam 函数计算不完全 Gamma 函数值并返回
        return (igam(b, a * x));
    }

    SPECFUN_HOST_DEVICE inline double gdtrc(double a, double b, double x) {

        // 检查输入参数是否合法，若 x 小于 0 则返回 NaN
        if (x < 0.0) {
            set_error("gdtrc", SF_ERROR_DOMAIN, NULL);
            return (std::numeric_limits<double>::quiet_NaN());
        }
        // 调用 igamc 函数计算补充不完全 Gamma 函数值并返回
        return (igamc(b, a * x));
    }

    SPECFUN_HOST_DEVICE inline double gdtri(double a, double b, double y) {

        // 检查输入参数是否合法，若 y 不在 [0, 1] 之间或 a 小于等于 0 或 b 小于 0 则返回 NaN
        if ((y < 0.0) || (y > 1.0) || (a <= 0.0) || (b < 0.0)) {
            sf_error("gdtri", SF_ERROR_DOMAIN, NULL);
            return (std::numeric_limits<double>::quiet_NaN());
        }

        // 调用 igamci 函数计算逆不完全 Gamma 函数值并返回
        return (igamci(b, 1.0 - y) / a);
    }

} // namespace cephes
} // namespace special
```