# `D:\src\scipysrc\scipy\scipy\special\special\cephes\airy.h`

```
/*
 * Translated into C++ by SciPy developers in 2024.
 * Original header with Copyright information appears below.
 */

/*
 *                                                     airy.c
 *
 *     Airy function
 *
 *
 *
 * SYNOPSIS:
 *
 * double x, ai, aip, bi, bip;
 * int airy();
 *
 * airy( x, _&ai, _&aip, _&bi, _&bip );
 *
 *
 *
 * DESCRIPTION:
 *
 * Solution of the differential equation
 *
 *     y"(x) = xy.
 *
 * The function returns the two independent solutions Ai, Bi
 * and their first derivatives Ai'(x), Bi'(x).
 *
 * Evaluation is by power series summation for small x,
 * by rational minimax approximations for large x.
 *
 *
 *
 * ACCURACY:
 * Error criterion is absolute when function <= 1, relative
 * when function > 1, except * denotes relative error criterion.
 * For large negative x, the absolute error increases as x^1.5.
 * For large positive x, the relative error increases as x^1.5.
 *
 * Arithmetic  domain   function  # trials      peak         rms
 * IEEE        -10, 0     Ai        10000       1.6e-15     2.7e-16
 * IEEE          0, 10    Ai        10000       2.3e-14*    1.8e-15*
 * IEEE        -10, 0     Ai'       10000       4.6e-15     7.6e-16
 * IEEE          0, 10    Ai'       10000       1.8e-14*    1.5e-15*
 * IEEE        -10, 10    Bi        30000       4.2e-15     5.3e-16
 * IEEE        -10, 10    Bi'       30000       4.9e-15     7.3e-16
 *
 */

/*
 * Cephes Math Library Release 2.8:  June, 2000
 * Copyright 1984, 1987, 1989, 2000 by Stephen L. Moshier
 */

#pragma once

#include "../config.h"
#include "const.h"
#include "polevl.h"

namespace special {
namespace cephes {

    } // namespace detail

    }

    // Inline function definition for the airy function
    inline int airy(float xf, float *aif, float *aipf, float *bif, float *bipf) {
        // Declare variables to store the results
        double ai;
        double aip;
        double bi;
        double bip;
        
        // Call the airy function from the cephes namespace
        int res = cephes::airy(xf, &ai, &aip, &bi, &bip);

        // Assign results to the provided pointers
        *aif = ai;
        *aipf = aip;
        *bif = bi;
        *bipf = bip;

        // Return the result of the airy function call
        return res;
    }

} // namespace cephes
} // namespace special
```