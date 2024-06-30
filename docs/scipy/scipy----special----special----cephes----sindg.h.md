# `D:\src\scipysrc\scipy\scipy\special\special\cephes\sindg.h`

```
/* Translated into C++ by SciPy developers in 2024.
 * Original header with Copyright information appears below.
 */

/*                                                     sindg.c
 *
 *     Circular sine of angle in degrees
 *
 *
 *
 * SYNOPSIS:
 *
 * double x, y, sindg();
 *
 * y = sindg( x );
 *
 *
 *
 * DESCRIPTION:
 *
 * Range reduction is into intervals of 45 degrees.
 *
 * Two polynomial approximating functions are employed.
 * Between 0 and pi/4 the sine is approximated by
 *      x  +  x**3 P(x**2).
 * Between pi/4 and pi/2 the cosine is represented as
 *      1  -  x**2 P(x**2).
 *
 *
 *
 * ACCURACY:
 *
 *                      Relative error:
 * arithmetic   domain      # trials      peak         rms
 *    IEEE      +-1000       30000      2.3e-16      5.6e-17
 *
 * ERROR MESSAGES:
 *
 *   message           condition        value returned
 * sindg total loss   x > 1.0e14 (IEEE)     0.0
 *
 */
/*                            cosdg.c
 *
 *    Circular cosine of angle in degrees
 *
 *
 *
 * SYNOPSIS:
 *
 * double x, y, cosdg();
 *
 * y = cosdg( x );
 *
 *
 *
 * DESCRIPTION:
 *
 * Range reduction is into intervals of 45 degrees.
 *
 * Two polynomial approximating functions are employed.
 * Between 0 and pi/4 the cosine is approximated by
 *      1  -  x**2 P(x**2).
 * Between pi/4 and pi/2 the sine is represented as
 *      x  +  x**3 P(x**2).
 *
 *
 * ACCURACY:
 *
 *                      Relative error:
 * arithmetic   domain      # trials      peak         rms
 *    IEEE     +-1000        30000       2.1e-16     5.7e-17
 *  See also sin().
 *
 */

/* Cephes Math Library Release 2.0:  April, 1987
 * Copyright 1985, 1987 by Stephen L. Moshier
 * Direct inquiries to 30 Frost Street, Cambridge, MA 02140 */
#pragma once

#include "../config.h"
#include "../error.h"

#include "const.h"
#include "polevl.h"

namespace special {
namespace cephes {

    namespace detail {

        constexpr double sincof[] = {1.58962301572218447952E-10, -2.50507477628503540135E-8,
                                     2.75573136213856773549E-6,  -1.98412698295895384658E-4,
                                     8.33333333332211858862E-3,  -1.66666666666666307295E-1};

        constexpr double coscof[] = {1.13678171382044553091E-11, -2.08758833757683644217E-9, 2.75573155429816611547E-7,
                                     -2.48015872936186303776E-5, 1.38888888888806666760E-3,  -4.16666666666666348141E-2,
                                     4.99999999999999999798E-1};

        constexpr double sindg_lossth = 1.0e14;

    } // namespace detail

} // namespace cephes
} // namespace special
    SPECFUN_HOST_DEVICE inline double sindg(double x) {
        double y, z, zz;
        int j, sign;

        /* make argument positive but save the sign */
        sign = 1;
        if (x < 0) {
            x = -x;
            sign = -1;
        }

        if (x > detail::sindg_lossth) {
            set_error("sindg", SF_ERROR_NO_RESULT, NULL);
            return (0.0);
        }

        y = std::floor(x / 45.0); /* 将 x 除以 45，取整数部分 */
        
        /* strip high bits of integer part to prevent integer overflow */
        z = std::ldexp(y, -4);    /* 将 y 左移 -4 位，相当于除以 16 */
        z = std::floor(z);        /* 取 z 的整数部分，y 除以 8 的整数部分 */
        z = y - std::ldexp(z, 4); /* y 减去 z 乘以 16 的值 */

        j = z; /* 将 z 转换为整数，用于阶段角的测试 */
        
        /* map zeros to origin */
        if (j & 1) { /* 如果 j 是奇数，将 j 加 1，同时 y 加 1.0 */
            j += 1;
            y += 1.0;
        }
        j = j & 07; /* 计算余数，得到模 360 度的八分之一 */
        
        /* reflect in x axis */
        if (j > 3) { /* 如果 j 大于 3，反射在 x 轴 */
            sign = -sign;
            j -= 4;
        }

        z = x - y * 45.0;   /* x 对 45 度取模 */
        z *= detail::PI180; /* 将 z 乘以 PI/180 转换为弧度 */
        zz = z * z;

        if ((j == 1) || (j == 2)) {
            y = 1.0 - zz * polevl(zz, detail::coscof, 6); /* 如果 j 是 1 或 2，计算余弦值 */
        } else {
            y = z + z * (zz * polevl(zz, detail::sincof, 5)); /* 否则计算正弦值 */
        }

        if (sign < 0)
            y = -y;

        return (y); /* 返回最终计算结果 */
    }

    SPECFUN_HOST_DEVICE inline double cosdg(double x) {
        double y, z, zz;
        int j, sign;

        /* make argument positive */
        sign = 1;
        if (x < 0)
            x = -x;

        if (x > detail::sindg_lossth) {
            set_error("cosdg", SF_ERROR_NO_RESULT, NULL);
            return (0.0);
        }

        y = std::floor(x / 45.0); /* 将 x 除以 45，取整数部分 */
        z = std::ldexp(y, -4);    /* 将 y 左移 -4 位，相当于除以 16 */
        z = std::floor(z);        /* 取 z 的整数部分，y 除以 8 的整数部分 */
        z = y - std::ldexp(z, 4); /* y 减去 z 乘以 16 的值 */

        /* integer and fractional part modulo one octant */
        j = z;
        if (j & 1) { /* 如果 j 是奇数，将 j 加 1，同时 y 加 1.0 */
            j += 1;
            y += 1.0;
        }
        j = j & 07; /* 计算余数，得到模 360 度的八分之一 */
        
        if (j > 3) { /* 如果 j 大于 3，将 j 减去 4，并改变符号 */
            j -= 4;
            sign = -sign;
        }

        if (j > 1)
            sign = -sign;

        z = x - y * 45.0;   /* x 对 45 度取模 */
        z *= detail::PI180; /* 将 z 乘以 PI/180 转换为弧度 */

        zz = z * z;

        if ((j == 1) || (j == 2)) {
            y = z + z * (zz * polevl(zz, detail::sincof, 5)); /* 如果 j 是 1 或 2，计算正弦值 */
        } else {
            y = 1.0 - zz * polevl(zz, detail::coscof, 6); /* 否则计算余弦值 */
        }

        if (sign < 0)
            y = -y;

        return (y); /* 返回最终计算结果 */
    }

    /* Degrees, minutes, seconds to radians: */

    /* 1 arc second, in radians = 4.848136811095359935899141023579479759563533023727e-6 */

    namespace detail {
        constexpr double sindg_P64800 = 4.848136811095359935899141023579479759563533023727e-6;
    }
    # 定义一个函数 radian，计算给定角度的度分秒转换为弧度的结果
    SPECFUN_HOST_DEVICE inline double radian(double d, double m, double s) {
        # 将度、分、秒转换为弧度表示，并乘以预定义的常数 detail::sindg_P64800
        return (((d * 60.0 + m) * 60.0 + s) * detail::sindg_P64800);
    }
} // 结束 cephes 命名空间
} // 结束 special 命名空间
```