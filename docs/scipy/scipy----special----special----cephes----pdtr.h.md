# `D:\src\scipysrc\scipy\scipy\special\special\cephes\pdtr.h`

```
/* Translated into C++ by SciPy developers in 2024.
 * Original header with Copyright information appears below.
 */

/*                                                     pdtr.c
 *
 *     Poisson distribution
 *
 *
 *
 * SYNOPSIS:
 *
 * int k;
 * double m, y, pdtr();
 *
 * y = pdtr( k, m );
 *
 *
 *
 * DESCRIPTION:
 *
 * Returns the sum of the first k terms of the Poisson
 * distribution:
 *
 *   k         j
 *   --   -m  m
 *   >   e    --
 *   --       j!
 *  j=0
 *
 * The terms are not summed directly; instead the incomplete
 * Gamma integral is employed, according to the relation
 *
 * y = pdtr( k, m ) = igamc( k+1, m ).
 *
 * The arguments must both be nonnegative.
 *
 *
 *
 * ACCURACY:
 *
 * See igamc().
 *
 */
/*  pdtrc()
 *
 *  Complemented poisson distribution
 *
 *
 *
 * SYNOPSIS:
 *
 * int k;
 * double m, y, pdtrc();
 *
 * y = pdtrc( k, m );
 *
 *
 *
 * DESCRIPTION:
 *
 * Returns the sum of the terms k+1 to infinity of the Poisson
 * distribution:
 *
 *  inf.       j
 *   --   -m  m
 *   >   e    --
 *   --       j!
 *  j=k+1
 *
 * The terms are not summed directly; instead the incomplete
 * Gamma integral is employed, according to the formula
 *
 * y = pdtrc( k, m ) = igam( k+1, m ).
 *
 * The arguments must both be nonnegative.
 *
 *
 *
 * ACCURACY:
 *
 * See igam.c.
 *
 */
/*  pdtri()
 *
 *  Inverse Poisson distribution
 *
 *
 *
 * SYNOPSIS:
 *
 * int k;
 * double m, y, pdtr();
 *
 * m = pdtri( k, y );
 *
 *
 *
 *
 * DESCRIPTION:
 *
 * Finds the Poisson variable x such that the integral
 * from 0 to x of the Poisson density is equal to the
 * given probability y.
 *
 * This is accomplished using the inverse Gamma integral
 * function and the relation
 *
 *    m = igamci( k+1, y ).
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
 * pdtri domain    y < 0 or y >= 1       0.0
 *                     k < 0
 *
 */

/*
 * Cephes Math Library Release 2.3:  March, 1995
 * Copyright 1984, 1987, 1995 by Stephen L. Moshier
 */
#pragma once

#include "../config.h"
#include "../error.h"

#include "igam.h"
#include "igami.h"

namespace special {
namespace cephes {

    SPECFUN_HOST_DEVICE inline double pdtrc(double k, double m) {
        // Check if k or m is negative, return NaN and set error if true
        if (k < 0.0 || m < 0.0) {
            set_error("pdtrc", SF_ERROR_DOMAIN, NULL);
            return (std::numeric_limits<double>::quiet_NaN());
        }
        // Return 0 if m is 0
        if (m == 0.0) {
            return 0.0;
        }
        // Compute v as floor of k + 1 and return the complemented incomplete gamma function
        double v = std::floor(k) + 1;
        return (igam(v, m));
    }

    SPECFUN_HOST_DEVICE inline double pdtr(double k, double m) {
        // Check if k or m is negative, return NaN and set error if true
        if (k < 0 || m < 0) {
            set_error("pdtr", SF_ERROR_DOMAIN, NULL);
            return (std::numeric_limits<double>::quiet_NaN());
        }
        // Return 1 if m is 0
        if (m == 0.0) {
            return 1.0;
        }
        // Compute v as floor of k + 1 and return the incomplete gamma function
        double v = std::floor(k) + 1;
        return (igamc(v, m));
    }
    // 定义一个内联函数 pdtri，返回类型为 double，标注为在主机和设备上均可调用
    SPECFUN_HOST_DEVICE inline double pdtri(int k, double y) {
        // 声明一个变量 v
        double v;

        // 如果 k 小于 0 或者 y 小于 0.0 或者 y 大于等于 1.0，则设置错误并返回 NaN
        if ((k < 0) || (y < 0.0) || (y >= 1.0)) {
            set_error("pdtri", SF_ERROR_DOMAIN, NULL);
            return (std::numeric_limits<double>::quiet_NaN());
        }

        // 将 v 设为 k + 1
        v = k + 1;
        
        // 调用 igamci 函数，计算 k+1 和 y 的 incomplete gamma 函数的互补函数值，将结果赋给 v
        v = igamci(v, y);
        
        // 返回计算结果 v
        return (v);
    }
} // end of namespace cephes
} // end of namespace special
```