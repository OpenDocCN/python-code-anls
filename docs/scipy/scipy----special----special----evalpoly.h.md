# `D:\src\scipysrc\scipy\scipy\special\special\evalpoly.h`

```
/* Translated from Cython into C++ by SciPy developers in 2024.
 *
 * Original author: Josh Wilson, 2016.
 */

/* Evaluate polynomials.
 *
 * All of the coefficients are stored in reverse order, i.e. if the
 * polynomial is
 *
 *     u_n x^n + u_{n - 1} x^{n - 1} + ... + u_0,
 *
 * then coeffs[0] = u_n, coeffs[1] = u_{n - 1}, ..., coeffs[n] = u_0.
 *
 * References
 * ----------
 * [1] Knuth, "The Art of Computer Programming, Volume II"
 */

#pragma once

#include "config.h"

namespace special {

/* Evaluate a polynomial with real coefficients at a complex point.
 *
 * Uses equation (3) in section 4.6.4 of [1]. Note that it is more
 * efficient than Horner's method.
 */
SPECFUN_HOST_DEVICE inline std::complex<double> cevalpoly(const double *coeffs, int degree, std::complex<double> z) {
    // Initialize coefficients a and b from the input array
    double a = coeffs[0];
    double b = coeffs[1];
    // Compute 2 times the real part of z
    double r = 2 * z.real();
    // Compute the squared norm of z
    double s = std::norm(z);
    double tmp;

    // Loop through the coefficients to evaluate the polynomial
    for (int j = 2; j < degree + 1; j++) {
        // Store current b in tmp for temporary storage
        tmp = b;
        // Update b using the given formula
        b = std::fma(-s, a, coeffs[j]);
        // Update a using the given formula
        a = std::fma(r, a, tmp);
    }

    // Return the evaluated polynomial value at complex point z
    return z * a + b;
}

} // namespace special
```