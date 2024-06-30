# `D:\src\scipysrc\scipy\scipy\special\special\cephes\hyperg.h`

```
/*
 * Translated into C++ by SciPy developers in 2024.
 * Original header with Copyright information appears below.
 */

/*
 * hyperg.c
 *
 * Confluent hypergeometric function
 *
 * SYNOPSIS:
 *
 * double a, b, x, y, hyperg();
 *
 * y = hyperg( a, b, x );
 *
 * DESCRIPTION:
 *
 * Computes the confluent hypergeometric function
 *
 *                          1           2
 *                       a x    a(a+1) x
 *   F ( a,b;x )  =  1 + ---- + --------- + ...
 *  1 1                  b 1!   b(b+1) 2!
 *
 * Many higher transcendental functions are special cases of
 * this power series.
 *
 * As is evident from the formula, b must not be a negative
 * integer or zero unless a is an integer with 0 >= a > b.
 *
 * The routine attempts both a direct summation of the series
 * and an asymptotic expansion.  In each case error due to
 * roundoff, cancellation, and nonconvergence is estimated.
 * The result with smaller estimated error is returned.
 *
 * ACCURACY:
 *
 * Tested at random points (a, b, x), all three variables
 * ranging from 0 to 30.
 *                      Relative error:
 * arithmetic   domain     # trials      peak         rms
 *    IEEE      0,30        30000       1.8e-14     1.1e-15
 *
 * Larger errors can be observed when b is near a negative
 * integer or zero.  Certain combinations of arguments yield
 * serious cancellation error in the power series summation
 * and also are not in the region of near convergence of the
 * asymptotic series.  An error message is printed if the
 * self-estimated relative error is greater than 1.0e-12.
 */

/*
 * Cephes Math Library Release 2.8:  June, 2000
 * Copyright 1984, 1987, 1988, 2000 by Stephen L. Moshier
 */

#pragma once

#include "../config.h"
#include "../error.h"

#include "const.h"
#include "gamma.h"

namespace special {
namespace cephes {

    } // namespace detail

    // Function to compute the confluent hypergeometric function F(a, b; x)
    SPECFUN_HOST_DEVICE inline double hyperg(double a, double b, double x) {
        double asum, psum, acanc, pcanc, temp;

        // Check if a Kummer transformation will help improve computation
        temp = b - a;
        if (std::abs(temp) < 0.001 * std::abs(a))
            return (exp(x) * hyperg(temp, b, -x));

        // Try both power series and asymptotic series, choosing the one with smaller error estimate
        if (std::abs(x) < 10 + std::abs(a) + std::abs(b)) {
            // Compute power series and asymptotic series
            psum = detail::hy1f1p(a, b, x, &pcanc);
            if (pcanc < 1.0e-15)
                goto done;
            asum = detail::hy1f1a(a, b, x, &acanc);
        } else {
            // Compute asymptotic series and power series
            psum = detail::hy1f1a(a, b, x, &pcanc);
            if (pcanc < 1.0e-15)
                goto done;
            asum = detail::hy1f1p(a, b, x, &acanc);
        }

        // Choose the result with less estimated error
        if (acanc < pcanc) {
            pcanc = acanc;
            psum = asum;
        }

    done:
        // Print an error message if the estimated error is too large
        if (pcanc > 1.0e-12)
            set_error("hyperg", SF_ERROR_LOSS, NULL);

        return (psum);
    }



    # 这是一个代码块的结束标记，与之前的代码块（函数、循环、条件语句等）相匹配
} // 结束 cephes 命名空间
} // 结束 special 命名空间
```