# `D:\src\scipysrc\scipy\scipy\special\special\cephes\kn.h`

```
/* Translated into C++ by SciPy developers in 2024.
 * Original header with Copyright information appears below.
 */

/*                                                     kn.c
 *
 *     Modified Bessel function, third kind, integer order
 *
 *
 *
 * SYNOPSIS:
 *
 * double x, y, kn();
 * int n;
 *
 * y = kn( n, x );
 *
 *
 *
 * DESCRIPTION:
 *
 * Returns modified Bessel function of the third kind
 * of order n of the argument.
 *
 * The range is partitioned into the two intervals [0,9.55] and
 * (9.55, infinity).  An ascending power series is used in the
 * low range, and an asymptotic expansion in the high range.
 *
 *
 *
 * ACCURACY:
 *
 *                      Relative error:
 * arithmetic   domain     # trials      peak         rms
 *    IEEE      0,30        90000       1.8e-8      3.0e-10
 *
 *  Error is high only near the crossover point x = 9.55
 * between the two expansions used.
 */

/*
 * Cephes Math Library Release 2.8:  June, 2000
 * Copyright 1984, 1987, 1988, 2000 by Stephen L. Moshier
 */

/*
 * Algorithm for Kn.
 *                        n-1
 *                    -n   -  (n-k-1)!    2   k
 * K (x)  =  0.5 (x/2)     >  -------- (-x /4)
 *  n                      -     k!
 *                        k=0
 *
 *                     inf.                                   2   k
 *        n         n   -                                   (x /4)
 *  + (-1)  0.5(x/2)    >  {p(k+1) + p(n+k+1) - 2log(x/2)} ---------
 *                      -                                  k! (n+k)!
 *                     k=0
 *
 * where  p(m) is the psi function: p(1) = -EUL and
 *
 *                       m-1
 *                        -
 *       p(m)  =  -EUL +  >  1/k
 *                        -
 *                       k=1
 *
 * For large x,
 *                                          2        2     2
 *                                       u-1     (u-1 )(u-3 )
 * K (z)  =  sqrt(pi/2z) exp(-z) { 1 + ------- + ------------ + ...}
 *  v                                        1            2
 *                                     1! (8z)     2! (8z)
 * asymptotically, where
 *
 *            2
 *     u = 4 v .
 *
 */
#pragma once

#include "../config.h"
#include "../error.h"

#include "const.h"

namespace special {
namespace cephes {

    namespace detail {

        constexpr int kn_MAXFAC = 31;  // Maximum factorial used in calculation

    }

}
}
    asymp:

        // 如果 x 超过最大对数值，设置错误并返回 0.0
        if (x > detail::MAXLOG) {
            set_error("kn", SF_ERROR_UNDERFLOW, NULL);
            return (0.0);
        }

        // 初始化变量
        k = n;                      // 将 n 赋值给 k
        pn = 4.0 * k * k;           // 计算 pn = 4.0 * k^2
        pk = 1.0;                   // 初始化 pk 为 1.0
        z0 = 8.0 * x;               // 计算 z0 = 8.0 * x
        fn = 1.0;                   // 初始化 fn 为 1.0
        t = 1.0;                    // 初始化 t 为 1.0
        s = t;                      // 初始化 s 为 t
        nkf = std::numeric_limits<double>::infinity();  // 将 nkf 初始化为正无穷大

        i = 0;                      // 初始化循环计数器 i 为 0
        do {
            z = pn - pk * pk;       // 计算 z = pn - pk^2
            t = t * z / (fn * z0);  // 更新 t = t * (pn - pk^2) / (fn * z0)
            nk1f = std::abs(t);     // 计算并取 t 的绝对值，赋值给 nk1f

            // 如果循环次数超过 n 并且 nk1f 大于 nkf，则跳出循环
            if ((i >= n) && (nk1f > nkf)) {
                goto adone;
            }

            nkf = nk1f;             // 更新 nkf 为 nk1f
            s += t;                 // 累加 t 到 s
            fn += 1.0;              // fn 自增 1.0
            pk += 2.0;              // pk 自增 2.0
            i += 1;                 // 循环计数器 i 自增 1
        } while (std::abs(t / s) > detail::MACHEP);  // 当 t / s 的绝对值大于 MACHEP 时继续循环

    adone:
        // 计算最终结果 ans，并返回
        ans = std::exp(-x) * std::sqrt(M_PI / (2.0 * x)) * s;
        return (ans);
    }
} // 结束特定的命名空间 "cephes"
} // 结束特定的命名空间 "special"
```