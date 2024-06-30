# `D:\src\scipysrc\scipy\scipy\special\special\cephes\cbrt.h`

```
/*
 * Translated into C++ by SciPy developers in 2024.
 * Original header with Copyright information appears below.
 */

/*
 * cbrt.c
 * Cube root
 *
 * SYNOPSIS:
 *
 * double x, y, cbrt();
 *
 * y = cbrt( x );
 *
 * DESCRIPTION:
 *
 * Returns the cube root of the argument, which may be negative.
 *
 * Range reduction involves determining the power of 2 of
 * the argument.  A polynomial of degree 2 applied to the
 * mantissa, and multiplication by the cube root of 1, 2, or 4
 * approximates the root to within about 0.1%.  Then Newton's
 * iteration is used three times to converge to an accurate
 * result.
 *
 * ACCURACY:
 *
 * Relative error:
 * arithmetic   domain     # trials      peak         rms
 *    IEEE       0,1e308     30000      1.5e-16     5.0e-17
 */

/*
 * Cephes Math Library Release 2.2:  January, 1991
 * Copyright 1984, 1991 by Stephen L. Moshier
 * Direct inquiries to 30 Frost Street, Cambridge, MA 02140
 */

#pragma once

#include "../config.h"

namespace special {
namespace cephes {
    namespace detail {

        // 常量定义：立方根的近似值
        constexpr double CBRT2 = 1.2599210498948731647672;
        constexpr double CBRT4 = 1.5874010519681994747517;
        constexpr double CBRT2I = 0.79370052598409973737585;
        constexpr double CBRT4I = 0.62996052494743658238361;

        // 函数：计算立方根
        SPECFUN_HOST_DEVICE inline double cbrt(double x) {
            int e, rem, sign;
            double z;

            // 如果 x 不是有限值，则直接返回 x
            if (!std::isfinite(x)) {
                return x;
            }
            // 如果 x 是 0，则直接返回 0
            if (x == 0) {
                return (x);
            }
            // 判断 x 的正负性
            if (x > 0) {
                sign = 1;
            } else {
                sign = -1;
                x = -x;
            }

            z = x;

            // 提取 x 的指数部分，得到 x 的尾数部分在 [0.5, 1) 之间
            x = std::frexp(x, &e);

            // 对位于 [0.5, 1) 区间的数近似计算其立方根
            x = (((-1.3466110473359520655053e-1 * x + 5.4664601366395524503440e-1) * x - 9.5438224771509446525043e-1) *
                     x +
                 1.1399983354717293273738e0) *
                    x +
                4.0238979564544752126924e-1;

            // 将指数部分除以 3
            if (e >= 0) {
                rem = e;
                e /= 3;
                rem -= 3 * e;
                // 根据余数调整 x 的值
                if (rem == 1) {
                    x *= CBRT2;
                } else if (rem == 2) {
                    x *= CBRT4;
                }
            }
            // 如果指数部分为负数
            else {
                e = -e;
                rem = e;
                e /= 3;
                rem -= 3 * e;
                // 根据余数调整 x 的值
                if (rem == 1) {
                    x *= CBRT2I;
                } else if (rem == 2) {
                    x *= CBRT4I;
                }
                e = -e;
            }

            // 将 x 乘以 2 的幂次方
            x = std::ldexp(x, e);

            // 牛顿迭代法，用于进一步精确计算立方根
            x -= (x - (z / (x * x))) * 0.33333333333333333333;
            x -= (x - (z / (x * x))) * 0.33333333333333333333;

            // 如果原始数 x 是负数，则结果取反
            if (sign < 0)
                x = -x;
            return (x);
        }
    } // namespace detail
} // 结束特殊命名空间
} // 结束cephes命名空间
```