# `D:\src\scipysrc\scipy\scipy\special\special\cephes\jv.h`

```
/*
 * Translated into C++ by SciPy developers in 2024.
 * Original header with Copyright information appears below.
 */

/*                                                     jv.c
 *
 *     Bessel function of noninteger order
 *
 *
 *
 * SYNOPSIS:
 *
 * double v, x, y, jv();
 *
 * y = jv( v, x );
 *
 *
 *
 * DESCRIPTION:
 *
 * Returns Bessel function of order v of the argument,
 * where v is real.  Negative x is allowed if v is an integer.
 *
 * Several expansions are included: the ascending power
 * series, the Hankel expansion, and two transitional
 * expansions for large v.  If v is not too large, it
 * is reduced by recurrence to a region of best accuracy.
 * The transitional expansions give 12D accuracy for v > 500.
 *
 *
 *
 * ACCURACY:
 * Results for integer v are indicated by *, where x and v
 * both vary from -125 to +125.  Otherwise,
 * x ranges from 0 to 125, v ranges as indicated by "domain."
 * Error criterion is absolute, except relative when |jv()| > 1.
 *
 * arithmetic  v domain  x domain    # trials      peak       rms
 *    IEEE      0,125     0,125      100000      4.6e-15    2.2e-16
 *    IEEE   -125,0       0,125       40000      5.4e-11    3.7e-13
 *    IEEE      0,500     0,500       20000      4.4e-15    4.0e-16
 * Integer v:
 *    IEEE   -125,125   -125,125      50000      3.5e-15*   1.9e-16*
 *
 */

/*
 * Cephes Math Library Release 2.8:  June, 2000
 * Copyright 1984, 1987, 1989, 1992, 2000 by Stephen L. Moshier
 */
#pragma once

#include "../config.h"
#include "../error.h"

#include "airy.h"
#include "cbrt.h"
#include "gamma.h"
#include "j0.h"
#include "j1.h"
#include "polevl.h"

namespace special {
namespace cephes {

    // Namespace `cephes` begins here

    // Definition of the `jv` function, computes the Bessel function of noninteger order
    double jv(double v, double x) {
        int i, sign;
        double a, b, temp, sum, p, q, t, ax;

        // Initializations and error handling
        ax = fabs(x);
        if (ax == 0.0) {
            return 0.0;
        } else if (v == 0.0) {
            return j0(x);
        } else if (v == 1.0) {
            return j1(x);
        }

        // Normalization and selection of calculation method based on domain and v value
        if (ax > 15.0) {
            // Use asymptotic expansion for large |x|
            t = v * v;
            b = ax + ax;
            for (i = 30; i > 0; i--) {
                b = b - 2.0;
                a = -t / b;
                b = b - 2.0;
                a = a * (v / b);
            }
            p = P0 + v * P1;
            q = Q0 + v * Q1;
            temp = sqrt(2.0 / (PI * ax)) * (p * cos(ax - PIO4) - a * sin(ax - PIO4));
            if (x < 0.0) {
                temp = -temp;
            }
            return temp;
        } else {
            // Use power series or other suitable expansions
            y = x * x;
            sum = a = 0.0;
            b = 0.0;
            for (i = 40; i > 0; i--) {
                a = a * (v / i);
                b = b * (y / (i + i));
                temp = a + b;
                if (fabs(temp) < fabs(sum) * DBL_EPSILON) {
                    break;
                }
                sum = temp;
            }
            sum = sum * j0(x) / temp;
            return sum;
        }

    } // End of jv function

} // End of namespace cephes
} // End of namespace special
```