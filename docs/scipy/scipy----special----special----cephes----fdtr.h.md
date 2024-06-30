# `D:\src\scipysrc\scipy\scipy\special\special\cephes\fdtr.h`

```
/*
 * Translated into C++ by SciPy developers in 2024.
 * Original header with Copyright information appears below.
 */

/*
 * fdtr.c
 *
 *     F distribution
 *
 * SYNOPSIS:
 *
 * double df1, df2;
 * double x, y, fdtr();
 *
 * y = fdtr( df1, df2, x );
 *
 * DESCRIPTION:
 *
 * Returns the area from zero to x under the F density
 * function (also known as Snedcor's density or the
 * variance ratio density).  This is the density
 * of x = (u1/df1)/(u2/df2), where u1 and u2 are random
 * variables having Chi square distributions with df1
 * and df2 degrees of freedom, respectively.
 *
 * The incomplete beta integral is used, according to the
 * formula
 *
 *     P(x) = incbet( df1/2, df2/2, (df1*x/(df2 + df1*x) ).
 *
 * The arguments a and b are greater than zero, and x is
 * nonnegative.
 *
 * ACCURACY:
 *
 * Tested at random points (a,b,x).
 *
 *                x     a,b                     Relative error:
 * arithmetic  domain  domain     # trials      peak         rms
 *    IEEE      0,1    0,100       100000      9.8e-15     1.7e-15
 *    IEEE      1,5    0,100       100000      6.5e-15     3.5e-16
 *    IEEE      0,1    1,10000     100000      2.2e-11     3.3e-12
 *    IEEE      1,5    1,10000     100000      1.1e-11     1.7e-13
 * See also incbet.c.
 *
 * ERROR MESSAGES:
 *
 *   message         condition      value returned
 * fdtr domain     a<0, b<0, x<0         0.0
 */

/*
 * fdtrc()
 *
 *  Complemented F distribution
 *
 * SYNOPSIS:
 *
 * double df1, df2;
 * double x, y, fdtrc();
 *
 * y = fdtrc( df1, df2, x );
 *
 * DESCRIPTION:
 *
 * Returns the area from x to infinity under the F density
 * function (also known as Snedcor's density or the
 * variance ratio density).
 *
 * The incomplete beta integral is used, according to the
 * formula
 *
 *  P(x) = incbet( df2/2, df1/2, (df2/(df2 + df1*x) ).
 *
 * ACCURACY:
 *
 * Tested at random points (a,b,x) in the indicated intervals.
 *                x     a,b                     Relative error:
 * arithmetic  domain  domain     # trials      peak         rms
 *    IEEE      0,1    1,100       100000      3.7e-14     5.9e-16
 *    IEEE      1,5    1,100       100000      8.0e-15     1.6e-15
 *    IEEE      0,1    1,10000     100000      1.8e-11     3.5e-13
 *    IEEE      1,5    1,10000     100000      2.0e-11     3.0e-12
 * See also incbet.c.
 *
 * ERROR MESSAGES:
 *
 *   message         condition      value returned
 * fdtrc domain    a<0, b<0, x<0         0.0
 */
/*
 * Inverse of F distribution
 */

/*
 * SYNOPSIS:
 *
 * double df1, df2;
 * double x, p, fdtri();
 *
 * x = fdtri( df1, df2, p );
 *
 * DESCRIPTION:
 *
 * Finds the F density argument x such that the integral
 * from -infinity to x of the F density is equal to the
 * given probability p.
 *
 * This is accomplished using the inverse beta integral
 * function and the relations
 *
 *      z = incbi( df2/2, df1/2, p )
 *      x = df2 (1-z) / (df1 z).
 *
 * Note: the following relations hold for the inverse of
 * the uncomplemented F distribution:
 *
 *      z = incbi( df1/2, df2/2, p )
 *      x = df2 z / (df1 (1-z)).
 */

/*
 * ACCURACY:
 *
 * Tested at random points (a,b,p).
 *
 *              a,b                     Relative error:
 * arithmetic  domain     # trials      peak         rms
 *  For p between .001 and 1:
 *    IEEE     1,100       100000      8.3e-15     4.7e-16
 *    IEEE     1,10000     100000      2.1e-11     1.4e-13
 *  For p between 10^-6 and 10^-3:
 *    IEEE     1,100        50000      1.3e-12     8.4e-15
 *    IEEE     1,10000      50000      3.0e-12     4.8e-14
 * See also fdtrc.c.
 */

/*
 * ERROR MESSAGES:
 *
 *   message         condition      value returned
 * fdtri domain   p <= 0 or p > 1       NaN
 *                     v < 1
 */

/*
 * Cephes Math Library Release 2.3:  March, 1995
 * Copyright 1984, 1987, 1995 by Stephen L. Moshier
 */

#pragma once

#include "../config.h"
#include "../error.h"

#include "incbet.h"
#include "incbi.h"

namespace special {
namespace cephes {

    SPECFUN_HOST_DEVICE inline double fdtrc(double a, double b, double x) {
        double w;

        // 检查参数是否合法，若不合法返回 NaN
        if ((a <= 0.0) || (b <= 0.0) || (x < 0.0)) {
            set_error("fdtrc", SF_ERROR_DOMAIN, NULL);
            return std::numeric_limits<double>::quiet_NaN();
        }
        // 计算 w 值并返回结果
        w = b / (b + a * x);
        return incbet(0.5 * b, 0.5 * a, w);
    }

    SPECFUN_HOST_DEVICE inline double fdtr(double a, double b, double x) {
        double w;

        // 检查参数是否合法，若不合法返回 NaN
        if ((a <= 0.0) || (b <= 0.0) || (x < 0.0)) {
            set_error("fdtr", SF_ERROR_DOMAIN, NULL);
            return std::numeric_limits<double>::quiet_NaN();
        }
        // 计算 w 值并返回结果
        w = a * x;
        w = w / (b + w);
        return incbet(0.5 * a, 0.5 * b, w);
    }
}
}
    // 定义一个特殊函数 fdtri，用于计算 F 分布的逆累积分布函数值
    SPECFUN_HOST_DEVICE inline double fdtri(double a, double b, double y) {
        double w, x;

        // 如果输入参数 a、b、y 有任何一个小于等于 0，或者 y 不在 (0, 1] 区间内，
        // 则设置错误并返回 NaN
        if ((a <= 0.0) || (b <= 0.0) || (y <= 0.0) || (y > 1.0)) {
            set_error("fdtri", SF_ERROR_DOMAIN, NULL);
            return NAN;
        }
        // 转换 y 为补数 1-y，以便在计算中使用
        y = 1.0 - y;
        
        // 计算 x = 0.5 时的累积分布函数值 w
        w = incbet(0.5 * b, 0.5 * a, 0.5);
        
        // 如果 w 大于 y，或者 y 非常接近 0，则使用 incbi 在 (0.5 * b, 0.5 * a) 区间内解方程
        // 否则，在 (0.5 * a, 0.5 * b) 区间内解方程，避免 (b - b * w) 中的取消误差
        if (w > y || y < 0.001) {
            w = incbi(0.5 * b, 0.5 * a, y);
            x = (b - b * w) / (a * w);
        } else {
            w = incbi(0.5 * a, 0.5 * b, 1.0 - y);
            x = b * w / (a * (1.0 - w));
        }
        
        // 返回计算得到的 F 分布逆累积分布函数值 x
        return x;
    }
} // namespace cephes
} // namespace special
```