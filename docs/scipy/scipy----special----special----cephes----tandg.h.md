# `D:\src\scipysrc\scipy\scipy\special\special\cephes\tandg.h`

```
/*
 * Translated into C++ by SciPy developers in 2024.
 * Original header with Copyright information appears below.
 */

/*
 * tandg.c
 *
 * Circular tangent of argument in degrees
 *
 * SYNOPSIS:
 *
 * double x, y, tandg();
 *
 * y = tandg( x );
 *
 * DESCRIPTION:
 *
 * Returns the circular tangent of the argument x in degrees.
 *
 * Range reduction is modulo pi/4.  A rational function
 *       x + x**3 P(x**2)/Q(x**2)
 * is employed in the basic interval [0, pi/4].
 *
 * ACCURACY:
 *
 *                      Relative error:
 * arithmetic   domain     # trials      peak         rms
 *    IEEE     0,10         30000      3.2e-16      8.4e-17
 *
 * ERROR MESSAGES:
 *
 *   message         condition          value returned
 * tandg total loss   x > 1.0e14 (IEEE)     0.0
 * tandg singularity  x = 180 k  +  90     INFINITY
 */

/*
 * cotdg.c
 *
 * Circular cotangent of argument in degrees
 *
 * SYNOPSIS:
 *
 * double x, y, cotdg();
 *
 * y = cotdg( x );
 *
 * DESCRIPTION:
 *
 * Returns the circular cotangent of the argument x in degrees.
 *
 * Range reduction is modulo pi/4.  A rational function
 *       x + x**3 P(x**2)/Q(x**2)
 * is employed in the basic interval [0, pi/4].
 *
 * ERROR MESSAGES:
 *
 *   message         condition          value returned
 * cotdg total loss   x > 1.0e14 (IEEE)     0.0
 * cotdg singularity  x = 180 k            INFINITY
 */

/*
 * Cephes Math Library Release 2.0:  April, 1987
 * Copyright 1984, 1987 by Stephen L. Moshier
 * Direct inquiries to 30 Frost Street, Cambridge, MA 02140
 */

#pragma once

#include "../config.h"
#include "../error.h"

namespace special {
namespace cephes {


注释：
这段代码是一个头文件的预处理指令及注释块，描述了C++版本的对角函数（tangent）和余切函数（cotangent）的转换和原始的版权信息。
    namespace detail {
        // 定义一个常量，表示 tan 和 cot 函数的极大值
        constexpr double tandg_lossth = 1.0e14;

        // 内部函数：计算 tan 或 cot 的值
        SPECFUN_HOST_DEVICE inline double tancot(double xx, int cotflg) {
            double x;
            int sign;

            /* make argument positive but save the sign */
            // 如果参数 xx 是负数，则取其绝对值并记录负号
            if (xx < 0) {
                x = -xx;
                sign = -1;
            } else {
                x = xx;
                sign = 1;
            }

            // 如果参数超出设定的极大值，返回错误并返回 0.0
            if (x > detail::tandg_lossth) {
                sf_error("tandg", SF_ERROR_NO_RESULT, NULL);
                return 0.0;
            }

            /* modulo 180 */
            // 将 x 取模 180
            x = x - 180.0 * std::floor(x / 180.0);

            // 根据 cotflg 决定是计算 tan 还是 cot
            if (cotflg) {
                // 如果需要计算 cot，根据 x 的范围进行不同的计算
                if (x <= 90.0) {
                    x = 90.0 - x;
                } else {
                    x = x - 90.0;
                    sign *= -1;
                }
            } else {
                // 如果需要计算 tan，同样根据 x 的范围进行不同的计算
                if (x > 90.0) {
                    x = 180.0 - x;
                    sign *= -1;
                }
            }

            // 处理特殊角度：0、45 和 90 度的情况
            if (x == 0.0) {
                return 0.0;
            } else if (x == 45.0) {
                return sign * 1.0;
            } else if (x == 90.0) {
                // 对于 90 度，返回正无穷并设置错误状态
                set_error((cotflg ? "cotdg" : "tandg"), SF_ERROR_SINGULAR, NULL);
                return std::numeric_limits<double>::infinity();
            }

            // 将 x 转换到 [0, 90) 区间，并返回 tan(x) 或 cot(x) 的计算结果
            return sign * std::tan(x * detail::PI180);
        }

    } // namespace detail

    // 对外接口：计算 tan(x)
    SPECFUN_HOST_DEVICE inline double tandg(double x) { return (detail::tancot(x, 0)); }

    // 对外接口：计算 cot(x)
    SPECFUN_HOST_DEVICE inline double cotdg(double x) { return (detail::tancot(x, 1)); }
} // 结束 cephes 命名空间

} // 结束 special 命名空间
```