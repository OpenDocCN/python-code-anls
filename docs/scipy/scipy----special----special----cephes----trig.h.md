# `D:\src\scipysrc\scipy\scipy\special\special\cephes\trig.h`

```
/*
 * Translated into C++ by SciPy developers in 2024.
 *
 * Original author: Josh Wilson, 2020.
 */

/*
 * Implement sin(pi * x) and cos(pi * x) for real x. Since the periods
 * of these functions are integral (and thus representable in double
 * precision), it's possible to compute them with greater accuracy
 * than sin(x) and cos(x).
 */
#pragma once

#include "../config.h"

namespace special {
namespace cephes {

    /* Compute sin(pi * x). */
    template <typename T>
    SPECFUN_HOST_DEVICE T sinpi(T x) {
        // Initialize sign
        T s = 1.0;

        // Adjust x and sign if x is negative
        if (x < 0.0) {
            x = -x;
            s = -1.0;
        }

        // Compute the fractional part of x mod 2
        T r = std::fmod(x, 2.0);

        // Determine the appropriate quadrant and compute sin(pi * r)
        if (r < 0.5) {
            return s * std::sin(M_PI * r);
        } else if (r > 1.5) {
            return s * std::sin(M_PI * (r - 2.0));
        } else {
            return -s * std::sin(M_PI * (r - 1.0));
        }
    }

    /* Compute cos(pi * x) */
    template <typename T>
    SPECFUN_HOST_DEVICE T cospi(T x) {
        // Adjust x if negative
        if (x < 0.0) {
            x = -x;
        }

        // Compute the fractional part of x mod 2
        T r = std::fmod(x, 2.0);

        // Handle special cases for cos(pi * x)
        if (r == 0.5) {
            // Avoid returning -0.0
            return 0.0;
        }
        if (r < 1.0) {
            return -std::sin(M_PI * (r - 0.5));
        } else {
            return std::sin(M_PI * (r - 1.5));
        }
    }
} // namespace cephes
} // namespace special
```