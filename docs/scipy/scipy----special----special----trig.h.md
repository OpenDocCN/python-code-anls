# `D:\src\scipysrc\scipy\scipy\special\special\trig.h`

```
/* Translated from Cython into C++ by SciPy developers in 2023.
 *
 * Original author: Josh Wilson, 2016.
 */

/* Implement sin(pi*z) and cos(pi*z) for complex z. Since the periods
 * of these functions are integral (and thus better representable in
 * floating point), it's possible to compute them with greater accuracy
 * than sin(z), cos(z).
 */

#pragma once

#include "cephes/trig.h"
#include "config.h"
#include "evalpoly.h"

namespace special {

template <typename T>
SPECFUN_HOST_DEVICE T sinpi(T x) {
    // Use Cephes library's sinpi function to compute sin(pi*x)
    return cephes::sinpi(x);
}

template <typename T>
SPECFUN_HOST_DEVICE std::complex<T> sinpi(std::complex<T> z) {
    T x = z.real();
    T piy = M_PI * z.imag();
    T abspiy = std::abs(piy);
    T sinpix = cephes::sinpi(x);
    T cospix = cephes::cospi(x);

    if (abspiy < 700) {
        // Use hyperbolic functions to compute sin(pi*z) for smaller |pi*y|
        return {sinpix * std::cosh(piy), cospix * std::sinh(piy)};
    }

    /* Have to be careful--sinh/cosh could overflow while cos/sin are small.
     * At this large of values
     *
     * cosh(y) ~ exp(y)/2
     * sinh(y) ~ sgn(y)*exp(y)/2
     *
     * so we can compute exp(y/2), scale by the right factor of sin/cos
     * and then multiply by exp(y/2) to avoid overflow. */
    T exphpiy = std::exp(abspiy / 2);
    T coshfac;
    T sinhfac;
    if (exphpiy == std::numeric_limits<T>::infinity()) {
        if (sinpix == 0.0) {
            // Preserve the sign of zero.
            coshfac = std::copysign(0.0, sinpix);
        } else {
            coshfac = std::copysign(std::numeric_limits<T>::infinity(), sinpix);
        }
        if (cospix == 0.0) {
            // Preserve the sign of zero.
            sinhfac = std::copysign(0.0, cospix);
        } else {
            sinhfac = std::copysign(std::numeric_limits<T>::infinity(), cospix);
        }
        // Return infinity values adjusted by copysign depending on sinpi/cospi results
        return {coshfac, sinhfac};
    }

    // Calculate hyperbolic factors scaled by exp(y/2) for large |pi*y|
    coshfac = 0.5 * sinpix * exphpiy;
    sinhfac = 0.5 * cospix * exphpiy;
    return {coshfac * exphpiy, sinhfac * exphpiy};
}

template <typename T>
SPECFUN_HOST_DEVICE T cospi(T x) {
    // Use Cephes library's cospi function to compute cos(pi*x)
    return cephes::cospi(x);
}

template <typename T>
SPECFUN_HOST_DEVICE std::complex<T> cospi(std::complex<T> z) {
    T x = z.real();
    T piy = M_PI * z.imag();
    T abspiy = std::abs(piy);
    T sinpix = cephes::sinpi(x);
    T cospix = cephes::cospi(x);

    if (abspiy < 700) {
        // Use hyperbolic functions to compute cos(pi*z) for smaller |pi*y|
        return {cospix * std::cosh(piy), -sinpix * std::sinh(piy)};
    }

    // Similar overflow handling as in sinpi(std::complex<T>)
    T exphpiy = std::exp(abspiy / 2);
    T coshfac;
    T sinhfac;
    // 如果 exphpiy 是无穷大
    if (exphpiy == std::numeric_limits<T>::infinity()) {
        // 如果 sinpix 等于 0.0
        if (sinpix == 0.0) {
            // 保留零的符号。
            coshfac = std::copysign(0.0, cospix);
        } else {
            // 设置 coshfac 为 cospix 的符号乘以正无穷大。
            coshfac = std::copysign(std::numeric_limits<T>::infinity(), cospix);
        }
        // 如果 cospix 等于 0.0
        if (cospix == 0.0) {
            // 保留零的符号。
            sinhfac = std::copysign(0.0, sinpix);
        } else {
            // 设置 sinhfac 为 sinpix 的符号乘以正无穷大。
            sinhfac = std::copysign(std::numeric_limits<T>::infinity(), sinpix);
        }
        // 返回由 coshfac 和 sinhfac 组成的元组。
        return {coshfac, sinhfac};
    }

    // 计算普通情况下的 coshfac 和 sinhfac
    coshfac = 0.5 * cospix * exphpiy;
    sinhfac = 0.5 * sinpix * exphpiy;
    // 返回经过调整后的 coshfac 和 sinhfac。
    return {coshfac * exphpiy, sinhfac * exphpiy};
}

} // namespace special
```