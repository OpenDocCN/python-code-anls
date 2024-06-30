# `D:\src\scipysrc\scipy\scipy\special\special\cephes\owens_t.h`

```
/*
 * Translated into C++ by SciPy developers in 2024.
 * Original header with Copyright information appears below.
 */

/*
 * Copyright Benjamin Sobotta 2012
 *
 * Use, modification and distribution are subject to the
 * Boost Software License, Version 1.0. (See accompanying file
 * LICENSE_1_0.txt or copy at https://www.boost.org/LICENSE_1_0.txt)
 */

/*
 * Reference:
 * Mike Patefield, David Tandy
 * FAST AND ACCURATE CALCULATION OF OWEN'S T-FUNCTION
 * Journal of Statistical Software, 5 (5), 1-25
 */

#pragma once

#include "../config.h"

#include "ndtr.h"
#include "unity.h"

namespace special {
namespace cephes {

    // Function to compute Owen's T-function
    SPECFUN_HOST_DEVICE inline double owens_t(double h, double a) {
        double result, fabs_a, fabs_ah, normh, normah;

        // Handle NaN inputs
        if (std::isnan(h) || std::isnan(a)) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        // Ensure h is non-negative due to symmetry property
        h = std::abs(h);

        /*
         * Use equation (2) in the paper to remap the arguments such that
         * h >= 0 and 0 <= a <= 1 for the call of the actual computation
         * routine.
         */
        fabs_a = std::abs(a);
        fabs_ah = fabs_a * h;

        // Special cases handling
        if (fabs_a == std::numeric_limits<double>::infinity()) {
            // Case when |a| is infinity, see page 13 in the referenced paper
            result = 0.5 * detail::owens_t_norm2(h);
        } else if (h == std::numeric_limits<double>::infinity()) {
            // Case when h is infinity
            result = 0;
        } else if (fabs_a <= 1) {
            // General case when |a| <= 1
            result = detail::owens_t_dispatch(h, fabs_a, fabs_ah);
        } else {
            // General case when |a| > 1
            if (fabs_ah <= 0.67) {
                normh = detail::owens_t_norm1(h);
                normah = detail::owens_t_norm1(fabs_ah);
                result = 0.25 - normh * normah - detail::owens_t_dispatch(fabs_ah, (1 / fabs_a), h);
            } else {
                normh = detail::owens_t_norm2(h);
                normah = detail::owens_t_norm2(fabs_ah);
                result = (normh + normah) / 2 - normh * normah - detail::owens_t_dispatch(fabs_ah, (1 / fabs_a), h);
            }
        }

        // Adjust result if a is negative due to symmetry property
        if (a < 0) {
            // exploit that T(h,-a) == -T(h,a)
            return -result;
        }

        // Return the computed result of Owen's T-function
        return result;
    }

} // namespace cephes
} // namespace special
```