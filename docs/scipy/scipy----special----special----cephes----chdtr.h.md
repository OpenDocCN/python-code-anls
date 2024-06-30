# `D:\src\scipysrc\scipy\scipy\special\special\cephes\chdtr.h`

```
/*
 *                                                     chdtr.c
 *
 *     Chi-square distribution
 *
 *
 *
 * SYNOPSIS:
 *
 * double df, x, y, chdtr();
 *
 * y = chdtr( df, x );
 *
 *
 *
 * DESCRIPTION:
 *
 * Returns the area under the left hand tail (from 0 to x)
 * of the Chi square probability density function with
 * v degrees of freedom.
 *
 *
 *                                  inf.
 *                                    -
 *                        1          | |  v/2-1  -t/2
 *  P( x | v )   =   -----------     |   t      e     dt
 *                    v/2  -       | |
 *                   2    | (v/2)   -
 *                                   x
 *
 * where x is the Chi-square variable.
 *
 * The incomplete Gamma integral is used, according to the
 * formula
 *
 *     y = chdtr( v, x ) = igam( v/2.0, x/2.0 ).
 *
 *
 * The arguments must both be positive.
 *
 *
 *
 * ACCURACY:
 *
 * See igam().
 *
 * ERROR MESSAGES:
 *
 *   message         condition      value returned
 * chdtr domain   x < 0 or v < 1        0.0
 */
/*                            chdtrc()
 *
 *    Complemented Chi-square distribution
 *
 *
 *
 * SYNOPSIS:
 *
 * double v, x, y, chdtrc();
 *
 * y = chdtrc( v, x );
 *
 *
 *
 * DESCRIPTION:
 *
 * Returns the area under the right hand tail (from x to
 * infinity) of the Chi square probability density function
 * with v degrees of freedom:
 *
 *
 *                                  inf.
 *                                    -
 *                        1          | |  v/2-1  -t/2
 *  P( x | v )   =   -----------     |   t      e     dt
 *                    v/2  -       | |
 *                   2    | (v/2)   -
 *                                   x
 *
 * where x is the Chi-square variable.
 *
 * The incomplete Gamma integral is used, according to the
 * formula
 *
 *    y = chdtr( v, x ) = igamc( v/2.0, x/2.0 ).
 *
 *
 * The arguments must both be positive.
 *
 *
 *
 * ACCURACY:
 *
 * See igamc().
 *
 * ERROR MESSAGES:
 *
 *   message         condition      value returned
 * chdtrc domain  x < 0 or v < 1        0.0
 */
/*                            chdtri()
 *
 *    Inverse of complemented Chi-square distribution
 *
 *
 *
 * SYNOPSIS:
 *
 * double df, x, y, chdtri();
 *
 * x = chdtri( df, y );
 *
 *
 *
 *
 * DESCRIPTION:
 *
 * Finds the Chi-square argument x such that the integral
 * from x to infinity of the Chi-square density is equal
 * to the given cumulative probability y.
 *
 * This is accomplished using the inverse Gamma integral
 * function and the relation
 *
 *    x/2 = igamci( df/2, y );
 *
 *
 *
 *
 * ACCURACY:
 *
 * See igami.c.
 *
 * ERROR MESSAGES:
 *
 *   message         condition      value returned
 * chdtri domain   y < 0 or y > 1        0.0
 *                     v < 1
 *
 */

/*                                                             chdtr() */


注释：

/*
 * 这段代码定义了一组与卡方分布相关的函数，包括计算卡方分布、卡方分布的补充函数以及补充卡方分布的逆函数。
 * 这些函数基于不完全伽马函数进行计算，用于概率密度函数的计算和逆计算。
 */
/*
 * Cephes Math Library Release 2.0:  April, 1987
 * Copyright 1984, 1987 by Stephen L. Moshier
 * Direct inquiries to 30 Frost Street, Cambridge, MA 02140
 */
#pragma once

#include "../config.h"   // 引入配置文件，可能定义了特定的编译选项和宏
#include "../error.h"    // 引入错误处理相关的头文件

#include "igam.h"         // 引入伽玛函数的头文件
#include "igami.h"        // 引入伽玛逆函数的头文件

namespace special {
namespace cephes {

    // 计算卡方分布的累积分布函数的补函数
    SPECFUN_HOST_DEVICE inline double chdtrc(double df, double x) {

        // 如果 x 小于 0，返回 1.0，根据 T. Oliphant 修改
        if (x < 0.0)
            return 1.0;
        
        // 调用 igamc 函数计算卡方分布的累积分布函数的补函数
        return (igamc(df / 2.0, x / 2.0));
    }

    // 计算卡方分布的累积分布函数
    SPECFUN_HOST_DEVICE inline double chdtr(double df, double x) {

        // 如果 x 小于 0，则设置错误信息并返回 NaN
        if ((x < 0.0)) { /* || (df < 1.0) ) */
            set_error("chdtr", SF_ERROR_DOMAIN, NULL);
            return (std::numeric_limits<double>::quiet_NaN());
        }
        
        // 调用 igam 函数计算卡方分布的累积分布函数
        return (igam(df / 2.0, x / 2.0));
    }

    // 计算卡方分布的累积分布函数的逆函数
    SPECFUN_HOST_DEVICE double chdtri(double df, double y) {
        double x;

        // 如果 y 小于 0 或大于 1，设置错误信息并返回 NaN
        if ((y < 0.0) || (y > 1.0)) { /* || (df < 1.0) ) */
            set_error("chdtri", SF_ERROR_DOMAIN, NULL);
            return (std::numeric_limits<double>::quiet_NaN());
        }

        // 调用 igamci 函数计算卡方分布的累积分布函数的逆函数
        x = igamci(0.5 * df, y);
        // 返回计算结果乘以 2
        return (2.0 * x);
    }

} // namespace cephes
} // namespace special
```