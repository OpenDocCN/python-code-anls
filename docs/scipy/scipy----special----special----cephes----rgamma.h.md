# `D:\src\scipysrc\scipy\scipy\special\special\cephes\rgamma.h`

```
/*
 *                                             rgamma.c
 *
 *     Reciprocal Gamma function
 *
 *
 *
 * SYNOPSIS:
 *
 * double x, y, rgamma();
 *
 * y = rgamma( x );
 *
 *
 *
 * DESCRIPTION:
 *
 * Returns one divided by the Gamma function of the argument.
 *
 * The function is approximated by a Chebyshev expansion in
 * the interval [0,1].  Range reduction is by recurrence
 * for arguments between -34.034 and +34.84425627277176174.
 * 0 is returned for positive arguments outside this
 * range.  For arguments less than -34.034 the cosecant
 * reflection formula is applied; lograrithms are employed
 * to avoid unnecessary overflow.
 *
 * The reciprocal Gamma function has no singularities,
 * but overflow and underflow may occur for large arguments.
 * These conditions return either INFINITY or 0 with
 * appropriate sign.
 *
 * ACCURACY:
 *
 *                      Relative error:
 * arithmetic   domain     # trials      peak         rms
 *    IEEE     -30,+30      30000       1.1e-15     2.0e-16
 * For arguments less than -34.034 the peak error is on the
 * order of 5e-15 (DEC), excepting overflow or underflow.
 */

/*
 * Cephes Math Library Release 2.0:  April, 1987
 * Copyright 1985, 1987 by Stephen L. Moshier
 * Direct inquiries to 30 Frost Street, Cambridge, MA 02140
 */
#pragma once

#include "../config.h"
#include "../error.h"
#include "chbevl.h"
#include "const.h"
#include "gamma.h"
#include "trig.h"

namespace special {
namespace cephes {

    namespace detail {

        /* Chebyshev coefficients for reciprocal Gamma function
         * in interval 0 to 1.  Function is 1/(x Gamma(x)) - 1
         */

        constexpr double rgamma_R[] = {
            3.13173458231230000000E-17, -6.70718606477908000000E-16, 2.20039078172259550000E-15,
            2.47691630348254132600E-13, -6.60074100411295197440E-12, 5.13850186324226978840E-11,
            1.08965386454418662084E-9,  -3.33964630686836942556E-8,  2.68975996440595483619E-7,
            2.96001177518801696639E-6,  -8.04814124978471142852E-5,  4.16609138709688864714E-4,
            5.06579864028608725080E-3,  -6.41925436109158228810E-2,  -4.98558728684003594785E-3,
            1.27546015610523951063E-1};

    } // namespace detail

} // namespace cephes
} // namespace special
    // 定义一个双精度函数 rgamma，用于计算伽玛函数的值
    SPECFUN_HOST_DEVICE double rgamma(double x) {
        double w, y, z;  // 定义双精度变量 w, y, z
        int sign;  // 定义整型变量 sign，用于记录符号

        // 如果 x 超过特定值，则返回伽玛函数的指数形式
        if (x > 34.84425627277176174) {
            return std::exp(-lgam(x));
        }
        // 如果 x 小于特定负数值
        if (x < -34.034) {
            w = -x;  // 取绝对值存入 w
            z = sinpi(w);  // 计算 sin(πw) 存入 z
            // 如果 z 等于 0，则返回 0.0
            if (z == 0.0) {
                return 0.0;
            }
            // 如果 z 小于 0，则记录 sign 为 1，z 取其绝对值
            if (z < 0.0) {
                sign = 1;
                z = -z;
            } else {
                sign = -1;  // 否则 sign 为 -1
            }

            // 计算 y，其中包括取对数、伽玛函数 lgam(w)，判断数值范围
            y = std::log(w * z) - std::log(M_PI) + lgam(w);
            // 如果 y 小于预设下限，设置错误并返回相应值
            if (y < -detail::MAXLOG) {
                set_error("rgamma", SF_ERROR_UNDERFLOW, NULL);
                return (sign * 0.0);
            }
            // 如果 y 超过预设上限，设置错误并返回相应值
            if (y > detail::MAXLOG) {
                set_error("rgamma", SF_ERROR_OVERFLOW, NULL);
                return (sign * std::numeric_limits<double>::infinity());
            }
            // 返回计算得到的伽玛函数值
            return (sign * std::exp(y));
        }
        
        // 初始化 z 为 1.0，w 为 x
        z = 1.0;
        w = x;

        // 下降递归，计算 w 大于 1.0 时的连乘
        while (w > 1.0) { /* Downward recurrence */
            w -= 1.0;
            z *= w;
        }
        // 上升递归，计算 w 小于 0.0 时的连除
        while (w < 0.0) { /* Upward recurrence */
            z /= w;
            w += 1.0;
        }
        // 如果 w 等于 0.0，返回 0.0
        if (w == 0.0) /* Nonpositive integer */
            return (0.0);
        // 如果 w 等于 1.0，返回 1/z
        if (w == 1.0) /* Other integer */
            return (1.0 / z);

        // 计算 y，利用多项式估计 chbevl，并返回结果
        y = w * (1.0 + chbevl(4.0 * w - 2.0, detail::rgamma_R, 16)) / z;
        return (y);
    }
} // 结束 cephes 命名空间定义
} // 结束 special 命名空间定义
```