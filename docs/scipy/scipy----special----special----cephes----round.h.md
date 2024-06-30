# `D:\src\scipysrc\scipy\scipy\special\special\cephes\round.h`

```
/* Translated into C++ by SciPy developers in 2024.
 * Original header with Copyright information appears below.
 */

/*                                                     round.c
 *
 *     Round double to nearest or even integer valued double
 *
 *
 *
 * SYNOPSIS:
 *
 * double x, y, round();
 *
 * y = round(x);
 *
 *
 *
 * DESCRIPTION:
 *
 * Returns the nearest integer to x as a double precision
 * floating point result.  If x ends in 0.5 exactly, the
 * nearest even integer is chosen.
 *
 *
 *
 * ACCURACY:
 *
 * If x is greater than 1/(2*MACHEP), its closest machine
 * representation is already an integer, so rounding does
 * not change it.
 */

/*
 * Cephes Math Library Release 2.1:  January, 1989
 * Copyright 1984, 1987, 1989 by Stephen L. Moshier
 * Direct inquiries to 30 Frost Street, Cambridge, MA 02140
 */
#pragma once

#include "../config.h"

namespace special {
namespace cephes {

    // 定义函数 round，将浮点数 x 四舍五入为最接近或最接近的偶数值的浮点数
    double round(double x) {
        double y, r;

        // 找出不大于 x 的最大整数
        y = std::floor(x);

        // 计算 x 的小数部分
        r = x - y;

        // 如果小数部分大于 0.5，则向上取整
        if (r > 0.5) {
            goto rndup;
        }

        // 如果小数部分等于 0.5，则按偶数舍入
        if (r == 0.5) {
            // 计算偶数舍入后的结果
            r = y - 2.0 * std::floor(0.5 * y);
            // 如果偶数舍入后为 1.0，则执行向上取整操作
            if (r == 1.0) {
            rndup:
                y += 1.0;
            }
        }

        // 否则向下取整并返回结果
        return (y);
    }

} // namespace cephes
} // namespace special
```