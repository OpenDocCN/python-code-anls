# `D:\src\scipysrc\scipy\scipy\special\special\cephes\sici.h`

```
/*
 * Translated into C++ by SciPy developers in 2024.
 * Original header with Copyright information appears below.
 */

/*
 *                                                     sici.c
 *
 *     Sine and cosine integrals
 *
 *
 *
 * SYNOPSIS:
 *
 * double x, Ci, Si, sici();
 *
 * sici( x, &Si, &Ci );
 *
 *
 * DESCRIPTION:
 *
 * Evaluates the integrals
 *
 *                          x
 *                          -
 *                         |  cos t - 1
 *   Ci(x) = eul + ln x +  |  --------- dt,
 *                         |      t
 *                        -
 *                         0
 *             x
 *             -
 *            |  sin t
 *   Si(x) =  |  ----- dt
 *            |    t
 *           -
 *            0
 *
 * where eul = 0.57721566490153286061 is Euler's constant.
 * The integrals are approximated by rational functions.
 * For x > 8 auxiliary functions f(x) and g(x) are employed
 * such that
 *
 * Ci(x) = f(x) sin(x) - g(x) cos(x)
 * Si(x) = pi/2 - f(x) cos(x) - g(x) sin(x)
 *
 *
 * ACCURACY:
 *    Test interval = [0,50].
 * Absolute error, except relative when > 1:
 * arithmetic   function   # trials      peak         rms
 *    IEEE        Si        30000       4.4e-16     7.3e-17
 *    IEEE        Ci        30000       6.9e-16     5.1e-17
 */

/*
 * Cephes Math Library Release 2.1:  January, 1989
 * Copyright 1984, 1987, 1989 by Stephen L. Moshier
 * Direct inquiries to 30 Frost Street, Cambridge, MA 02140
 */
#pragma once

#include "../config.h"

#include "const.h"
#include "polevl.h"

namespace special {
namespace cephes {

    // 在这里开始定义 Cephes 数学库的特殊函数实现

} // namespace cephes
} // namespace special
    // 定义一个内联函数 sici，计算给定参数 x 的正弦积分 si 和余弦积分 ci
    SPECFUN_HOST_DEVICE inline int sici(double x, double *si, double *ci) {
        double z, c, s, f, g;  // 声明双精度浮点变量 z, c, s, f, g
        short sign;  // 声明短整型变量 sign，用于记录 x 的符号

        // 检查 x 是否小于 0
        if (x < 0.0) {
            sign = -1;  // 如果是负数，记录符号为 -1
            x = -x;  // 将 x 取绝对值
        } else {
            sign = 0;  // 否则，符号为 0
        }

        // 如果 x 等于 0
        if (x == 0.0) {
            *si = 0.0;  // 正弦积分为 0
            *ci = -std::numeric_limits<double>::infinity();  // 余弦积分为负无穷
            return (0);  // 返回 0
        }

        // 如果 x 大于 1.0e9
        if (x > 1.0e9) {
            // 如果 x 是无穷大
            if (std::isinf(x)) {
                // 根据符号设置正弦积分和余弦积分
                if (sign == -1) {
                    *si = -M_PI_2;  // 负无穷
                    *ci = std::numeric_limits<double>::quiet_NaN();  // 静态的 NaN 值
                } else {
                    *si = M_PI_2;  // 正无穷
                    *ci = 0;  // 零
                }
                return 0;  // 返回 0
            }
            // 计算正弦积分和余弦积分的近似值
            *si = M_PI_2 - std::cos(x) / x;
            *ci = std::sin(x) / x;
        }

        // 如果 x 大于 4.0，跳转到渐近部分
        if (x > 4.0) {
            goto asympt;
        }

        // 计算 s 和 c
        z = x * x;
        s = x * polevl(z, detail::sici_SN, 5) / polevl(z, detail::sici_SD, 5);
        c = z * polevl(z, detail::sici_CN, 5) / polevl(z, detail::sici_CD, 5);

        // 如果符号为负，调整 s 的符号
        if (sign) {
            s = -s;
        }
        *si = s;  // 设置正弦积分的值
        *ci = detail::SCIPY_EULER + std::log(x) + c; /* real part if x < 0 */  // 设置余弦积分的值
        return (0);  // 返回 0

    asympt:

        // 在渐近区域计算 s 和 c
        s = std::sin(x);
        c = std::cos(x);
        z = 1.0 / (x * x);
        // 根据 x 的大小选择不同的系数近似值
        if (x < 8.0) {
            f = polevl(z, detail::sici_FN4, 6) / (x * p1evl(z, detail::sici_FD4, 7));
            g = z * polevl(z, detail::sici_GN4, 7) / p1evl(z, detail::sici_GD4, 7);
        } else {
            f = polevl(z, detail::sici_FN8, 8) / (x * p1evl(z, detail::sici_FD8, 8));
            g = z * polevl(z, detail::sici_GN8, 8) / p1evl(z, detail::sici_GD8, 9);
        }
        *si = M_PI_2 - f * c - g * s;  // 计算正弦积分的渐近值
        if (sign) {
            *si = -(*si);  // 如果符号为负，取相反数
        }
        *ci = f * s - g * c;  // 计算余弦积分的渐近值

        return (0);  // 返回 0
    }
} // 结束 cephes 命名空间定义
} // 结束 special 命名空间定义
```