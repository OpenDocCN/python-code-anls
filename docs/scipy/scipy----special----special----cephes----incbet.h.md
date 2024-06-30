# `D:\src\scipysrc\scipy\scipy\special\special\cephes\incbet.h`

```
/*
 * Translated into C++ by SciPy developers in 2024.
 * Original header with Copyright information appears below.
 */

/*
 * incbet.c
 *
 * Incomplete beta integral
 *
 * SYNOPSIS:
 *
 * double a, b, x, y, incbet();
 *
 * y = incbet( a, b, x );
 *
 * DESCRIPTION:
 *
 * Returns incomplete beta integral of the arguments, evaluated
 * from zero to x.  The function is defined as
 *
 *                  x
 *     -            -
 *    | (a+b)      | |  a-1     b-1
 *  -----------    |   t   (1-t)   dt.
 *   -     -     | |
 *  | (a) | (b)   -
 *                 0
 *
 * The domain of definition is 0 <= x <= 1.  In this
 * implementation a and b are restricted to positive values.
 * The integral from x to 1 may be obtained by the symmetry
 * relation
 *
 *    1 - incbet( a, b, x )  =  incbet( b, a, 1-x ).
 *
 * The integral is evaluated by a continued fraction expansion
 * or, when b*x is small, by a power series.
 *
 * ACCURACY:
 *
 * Tested at uniformly distributed random points (a,b,x) with a and b
 * in "domain" and x between 0 and 1.
 *                                        Relative error
 * arithmetic   domain     # trials      peak         rms
 *    IEEE      0,5         10000       6.9e-15     4.5e-16
 *    IEEE      0,85       250000       2.2e-13     1.7e-14
 *    IEEE      0,1000      30000       5.3e-12     6.3e-13
 *    IEEE      0,10000    250000       9.3e-11     7.1e-12
 *    IEEE      0,100000    10000       8.7e-10     4.8e-11
 * Outputs smaller than the IEEE gradual underflow threshold
 * were excluded from these statistics.
 *
 * ERROR MESSAGES:
 *   message         condition      value returned
 * incbet domain      x<0, x>1          0.0
 * incbet underflow                     0.0
 */

/*
 * Cephes Math Library, Release 2.3:  March, 1995
 * Copyright 1984, 1995 by Stephen L. Moshier
 */
#pragma once

#include "../config.h"
#include "../error.h"

#include "beta.h"
#include "const.h"

namespace special {
namespace cephes {

    // 在这里应该有实现 incbet 函数的代码，但是在提供的片段中并没有给出具体的函数实现部分。

} // namespace detail
    # 定义一个内联函数 incbet，计算不完全贝塔函数的值
    SPECFUN_HOST_DEVICE inline double incbet(double aa, double bb, double xx) {
        double a, b, t, x, xc, w, y;
        int flag;

        // 检查参数 aa 和 bb 是否为非正数，若是则报错
        if (aa <= 0.0 || bb <= 0.0)
            goto domerr;

        // 检查参数 xx 是否在有效范围内，若不在则根据具体情况返回特定值或报错
        if ((xx <= 0.0) || (xx >= 1.0)) {
            if (xx == 0.0)
                return (0.0);
            if (xx == 1.0)
                return (1.0);
        domerr:
            // 设置错误信息，指示发生域错误
            set_error("incbet", SF_ERROR_DOMAIN, NULL);
            // 返回 NaN 表示无效结果
            return (std::numeric_limits<double>::quiet_NaN());
        }

        // 初始化标志位
        flag = 0;
        // 根据 xx 和 bb 的值选择不完全贝塔函数的计算方式
        if ((bb * xx) <= 1.0 && xx <= 0.95) {
            // 使用 P-级数计算不完全贝塔函数的值
            t = detail::incbet_pseries(aa, bb, xx);
            goto done;
        }

        // 计算 1 - xx，用于后续计算
        w = 1.0 - xx;

        /* 如果 xx 大于 aa / (aa + bb)，则交换 a 和 b 的值 */
        if (xx > (aa / (aa + bb))) {
            flag = 1;
            a = bb;
            b = aa;
            xc = xx;
            x = w;
        } else {
            a = aa;
            b = bb;
            xc = w;
            x = xx;
        }

        // 根据标志位和条件选择计算方式
        if (flag == 1 && (b * x) <= 1.0 && x <= 0.95) {
            // 使用 P-级数计算不完全贝塔函数的值
            t = detail::incbet_pseries(a, b, x);
            goto done;
        }

        /* 根据 y 的符号选择更适合的收敛方法 */
        y = x * (a + b - 2.0) - (a - 1.0);
        if (y < 0.0) {
            // 使用连分数逼近法计算不完全贝塔函数的值
            w = detail::incbcf(a, b, x);
        } else {
            // 使用基于缩放的连分数逼近法计算不完全贝塔函数的值
            w = detail::incbd(a, b, x) / xc;
        }

        /* 将 w 乘以下列因子：
         * a      b   _             _     _
         * x  (1-x)   | (a+b) / ( a | (a) | (b) ) .   */

        // 计算需要的对数，以便判断是否使用对数计算
        y = a * std::log(x);
        t = b * std::log(xc);
        if ((a + b) < detail::MAXGAM && std::abs(y) < detail::MAXLOG && std::abs(t) < detail::MAXLOG) {
            // 使用对数计算不完全贝塔函数的值
            t = std::pow(xc, b);
            t *= std::pow(x, a);
            t /= a;
            t *= w;
            t *= 1.0 / beta(a, b);
            goto done;
        }
        /* 如果条件不满足，则使用对数计算 */
        y += t - lbeta(a, b);
        y += std::log(w / a);
        if (y < detail::MINLOG) {
            t = 0.0;
        } else {
            t = exp(y);
        }

    done:
        // 根据标志位决定是否对结果进行调整
        if (flag == 1) {
            if (t <= detail::MACHEP) {
                t = 1.0 - detail::MACHEP;
            } else {
                t = 1.0 - t;
            }
        }
        // 返回计算结果
        return (t);
    }
} // 结束 cephes 命名空间定义
} // 结束 special 命名空间定义
```