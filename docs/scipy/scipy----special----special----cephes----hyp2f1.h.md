# `D:\src\scipysrc\scipy\scipy\special\special\cephes\hyp2f1.h`

```
/* Translated into C++ by SciPy developers in 2024.
 * Original header with Copyright information appears below.
 */

/*                                                      hyp2f1.c
 *
 *      Gauss hypergeometric function   F
 *                                     2 1
 *
 *
 * SYNOPSIS:
 *
 * double a, b, c, x, y, hyp2f1();
 *
 * y = hyp2f1( a, b, c, x );
 *
 *
 * DESCRIPTION:
 *
 *
 *  hyp2f1( a, b, c, x )  =   F ( a, b; c; x )
 *                           2 1
 *
 *           inf.
 *            -   a(a+1)...(a+k) b(b+1)...(b+k)   k+1
 *   =  1 +   >   -----------------------------  x   .
 *            -         c(c+1)...(c+k) (k+1)!
 *          k = 0
 *
 *  Cases addressed are
 *      Tests and escapes for negative integer a, b, or c
 *      Linear transformation if c - a or c - b negative integer
 *      Special case c = a or c = b
 *      Linear transformation for  x near +1
 *      Transformation for x < -0.5
 *      Psi function expansion if x > 0.5 and c - a - b integer
 *      Conditionally, a recurrence on c to make c-a-b > 0
 *
 *      x < -1  AMS 15.3.7 transformation applied (Travis Oliphant)
 *         valid for b,a,c,(b-a) != integer and (c-a),(c-b) != negative integer
 *
 * x >= 1 is rejected (unless special cases are present)
 *
 * The parameters a, b, c are considered to be integer
 * valued if they are within 1.0e-14 of the nearest integer
 * (1.0e-13 for IEEE arithmetic).
 *
 * ACCURACY:
 *
 *
 *               Relative error (-1 < x < 1):
 * arithmetic   domain     # trials      peak         rms
 *    IEEE      -1,7        230000      1.2e-11     5.2e-14
 *
 * Several special cases also tested with a, b, c in
 * the range -7 to 7.
 *
 * ERROR MESSAGES:
 *
 * A "partial loss of precision" message is printed if
 * the internally estimated relative error exceeds 1^-12.
 * A "singularity" message is printed on overflow or
 * in cases not addressed (such as x < -1).
 */

/*
 * Cephes Math Library Release 2.8:  June, 2000
 * Copyright 1984, 1987, 1992, 2000 by Stephen L. Moshier
 */

#pragma once

#include "../config.h"
#include "../error.h"

#include "const.h"
#include "gamma.h"
#include "psi.h"

namespace special {
namespace cephes {

    // Compute the Gauss hypergeometric function F(2,1) using specific algorithms
    // and handle various cases of input parameters.
    double hyp2f1(double a, double b, double c, double x) {
        double y, err;

        // Call the implementation detail function to compute hyp2f1
        y = detail::hyt2f1(a, b, c, x, &err);

    hypdon:
        // Handle cases where the estimated error exceeds a threshold
        if (err > detail::hyp2f1_ETHRESH) {
            set_error("hyp2f1", SF_ERROR_LOSS, NULL);
            /*      printf( "Estimated err = %.2e\n", err ); */
        }
        return (y);

    hypf:
        // Apply transformation when c-a or c-b is a negative integer (AMS55 #15.3.3)
        y = std::pow(s, d) * detail::hys2f1(c - a, c - b, c, x, &err);
        goto hypdon;

    hypdiv:
        // Handle overflow cases
        set_error("hyp2f1", SF_ERROR_OVERFLOW, NULL);
        return std::numeric_limits<double>::infinity();
    }

} // namespace cephes
} // namespace special
```