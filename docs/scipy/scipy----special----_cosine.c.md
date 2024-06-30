# `D:\src\scipysrc\scipy\scipy\special\_cosine.c`

```
/*
 *  The functions
 *
 *    cosine_cdf
 *    cosine_invcdf
 *
 *  defined here are the kernels for the ufuncs
 *
 *    _cosine_cdf
 *    _cosine_invcdf
 *
 *  defined in scipy.special._ufuncs.
 *
 *  The ufuncs are used by the class scipy.stats.cosine_gen.
 */

#include "special_wrappers.h"
#include <math.h>

// M_PI64 is the 64 bit floating point representation of π, e.g.
//   >>> math.pi.hex()
//   '0x1.921fb54442d18p+1'
// It is used in the function cosine_cdf_pade_approx_at_neg_pi,
// which depends on this value being the 64 bit representation.
// Do not replace this with M_PI from math.h or NPY_PI from the
// numpy header files.
#define M_PI64 0x1.921fb54442d18p+1

//
// p and q (below) are the coefficients in the numerator and denominator
// polynomials (resp.) of the Pade approximation of
//     f(x) = (pi + x + sin(x))/(2*pi)
// at x=-pi.  The coefficients are ordered from lowest degree to highest.
// These values are used in the function cosine_cdf_pade_approx_at_neg_pi(x).
//
// These coefficients can be derived by using mpmath as follows:
//
//    import mpmath
//
//    def f(x):
//        return (mpmath.pi + x + mpmath.sin(x)) / (2*mpmath.pi)
//
//    # Note: 40 digits might be overkill; a few more digits than the default
//    # might be sufficient.
//    mpmath.mp.dps = 40
//    ts = mpmath.taylor(f, -mpmath.pi, 20)
//    p, q = mpmath.pade(ts, 9, 10)
//
// (A python script with that code is in special/_precompute/cosine_cdf.py.)
//
// The following are the values after converting to 64 bit floating point:
// p = [0.0,
//      0.0,
//      0.0,
//      0.026525823848649224,
//      0.0,
//      -0.0007883197097740538,
//      0.0,
//      1.0235408442872927e-05,
//      0.0,
//      -3.8360369451359084e-08]
// q = [1.0,
//      0.0,
//      0.020281047093125535,
//      0.0,
//      0.00020944197182753272,
//      0.0,
//      1.4162345851873058e-06,
//      0.0,
//      6.498171564823105e-09,
//      0.0,
//      1.6955280904096042e-11]
//


//
// Compute the CDF of the standard cosine distribution for x close to but
// not less than -π.  A Pade approximant is used to avoid the loss of
// precision that occurs in the formula 1/2 + (x + sin(x))/(2*pi) when
// x is near -π.
//
static
double cosine_cdf_pade_approx_at_neg_pi(double x)
{
    double h, h2, h3;
    double numer, denom;
    double numer_coeffs[] = {-3.8360369451359084e-08,
                             1.0235408442872927e-05,
                             -0.0007883197097740538,
                             0.026525823848649224};
    double denom_coeffs[] = {1.6955280904096042e-11,
                             6.498171564823105e-09,
                             1.4162345851873058e-06,
                             0.00020944197182753272,
                             0.020281047093125535,
                             1.0};

    // M_PI64 is not exactly π.  In fact, float64(π - M_PI64) is
    // 1.2246467991473532e-16.  h is supposed to be x + π, so to compute
    // h, we should add M_PI64 to x.

    h = x + M_PI64;
    h2 = h * h;
    h3 = h * h2;

    // Evaluate the numerator and denominator polynomials using Horner's method

    numer = numer_coeffs[3];
    numer = numer * h2 + numer_coeffs[2];
    numer = numer * h2 + numer_coeffs[1];
    numer = numer * h2 + numer_coeffs[0];

    denom = denom_coeffs[5];
    denom = denom * h2 + denom_coeffs[4];
    denom = denom * h2 + denom_coeffs[3];
    denom = denom * h2 + denom_coeffs[2];
    denom = denom * h2 + denom_coeffs[1];
    denom = denom * h2 + denom_coeffs[0];

    // Compute and return the Pade approximant
    return numer / denom;
}
    // 将 x 增加一个极小的常数 M_PI64，并存储在 h 中，以提高计算精度
    h = (x + M_PI64) + 1.2246467991473532e-16;
    // 计算 h 的平方
    h2 = h*h;
    // 计算 h 的立方
    h3 = h2*h;
    // 使用 cephes_polevl_wrap 函数计算数值系数的多项式函数，得到分子
    numer = h3 * cephes_polevl_wrap(h2, numer_coeffs,
                      sizeof(numer_coeffs)/sizeof(numer_coeffs[0]) - 1);
    // 使用 cephes_polevl_wrap 函数计算分母的多项式函数
    denom = cephes_polevl_wrap(h2, denom_coeffs,
                   sizeof(denom_coeffs)/sizeof(denom_coeffs[0]) - 1);
    // 返回计算结果的比值
    return numer / denom;
// 定义函数，计算余弦分布的累积分布函数（CDF）
double cosine_cdf(double x)
{
    // 如果 x 大于等于 π，返回 1
    if (x >= M_PI64) {
        return 1;
    }
    // 如果 x 小于 -π，返回 0
    if (x < -M_PI64) {
        return 0;
    }
    // 如果 x 在 -π 到 -1.6 之间，调用 Pade 近似函数进行计算
    if (x < -1.6) {
        return cosine_cdf_pade_approx_at_neg_pi(x);
    }
    // 否则，使用公式计算余弦分布的累积分布函数值
    return 0.5 + (x + sin(x))/(2*M_PI64);
}

// 以下注释描述了余弦分布的累积分布函数的反函数推导过程和其近似计算的系数定义

// 余弦分布的累积分布函数为 p = (pi + x + sin(x)) / (2*pi)，我们需要其反函数。
// 将公式化简为 pi*(2*p - 1) = x + sin(x)，定义 f(x) = x + sin(x)，则其反函数 g(x) 满足 x = g(pi*(2*p - 1))。

// 下面的函数 _p2 和 _q2 使用 Pade 近似逆函数的系数，这些系数是余弦分布累积分布函数的反函数在 p=0.5 处的 Pade 近似值。
// 这些系数通过以下步骤导出：
// 1. 在 x = 0 处找到 x + sin(x) 的泰勒多项式系数，使用了22阶泰勒多项式。
// 2. 将这些系数“反转”，以找到反函数的泰勒多项式。反转方法见 https://en.wikipedia.org/wiki/Bell_polynomials#Reversion_of_series。
// 3. 将反函数的泰勒多项式转换为 (11, 10) 阶 Pade 近似。系数以增序排列，并转换为64位浮点数。

static
double _p2(double t)
{
    // _p2 函数的系数数组
    double coeffs[] = {-6.8448463845552725e-09,
                       3.4900934227012284e-06,
                       -0.00030539712907115167,
                       0.009350454384541677,
                       -0.11602142940208726,
                       0.5};
    double v;

    // 使用 cephes_polevl_wrap 函数计算 Pade 近似值
    v = cephes_polevl_wrap(t, coeffs, sizeof(coeffs) / sizeof(coeffs[0]) - 1);
    return v;
}

static
double _q2(double t)
{
    // _q2 函数的系数数组
    double coeffs[] = {-5.579679571562129e-08,
                       1.3728570152788793e-05,
                       -0.0008916919927321117,
                       0.022927496105281435,
                       -0.25287619213750784,
                       1.0};
    double v;

    // 使用 cephes_polevl_wrap 函数计算 Pade 近似值
    v = cephes_polevl_wrap(t, coeffs, sizeof(coeffs) / sizeof(coeffs[0]) - 1);
    return v;
}
// Part of the asymptotic expansion of the inverse function at p=0.
//
// See, for example, the wikipedia article "Kepler's equation"
// (https://en.wikipedia.org/wiki/Kepler%27s_equation).  In particular, see the
// series expansion for the inverse Kepler equation when the eccentricity e is 1.
//
// Static function to approximate a polynomial based on the given series of coefficients
static
double _poly_approx(double s)
{
    double s2; // Square of input parameter s
    double p; // Result of polynomial approximation
    double coeffs[] = {1.1911667949082915e-08,
                       1.683039183039183e-07,
                       43.0/17248000,
                       1.0/25200,
                       1.0/1400,
                       1.0/60,
                       1.0}; // Coefficients for the polynomial approximation

    //
    // p(s) = s + (1/60) * s**3 + (1/1400) * s**5 + (1/25200) * s**7 +
    //        (43/17248000) * s**9 + (1213/7207200000) * s**11 +
    //        (151439/12713500800000) * s**13 + ...
    //
    // Here we include terms up to s**13.
    //
    s2 = s*s; // Compute s squared
    // Calculate the polynomial approximation using the given coefficients
    p = s * cephes_polevl_wrap(s2, coeffs, sizeof(coeffs)/sizeof(coeffs[0]) - 1);
    return p; // Return the resulting polynomial approximation
}

//
// Cosine distribution inverse CDF (aka percent point function).
//
// Function to compute the inverse cumulative distribution function (CDF) of the
// cosine distribution based on a given probability value p.
double cosine_invcdf(double p)
{
    double x; // Resulting angle value
    int sgn = 1; // Sign of the resulting angle, initialized to positive

    // Check if p is out of valid range [0, 1], return NaN if true
    if ((p < 0) || (p > 1)) {
        return NAN;
    }
    // Special case for p approaching 0, return negative pi
    if (p <= 1e-48) {
        return -M_PI64;
    }
    // Special case for p equals 1, return positive pi
    if (p == 1) {
        return M_PI64;
    }

    // Adjust p and sign for better calculation accuracy
    if (p > 0.5) {
        p = 1.0 - p;
        sgn = -1;
    }

    // If p is small enough, use polynomial approximation to compute x
    if (p < 0.0925) {
        x = _poly_approx(cbrt(12 * M_PI64 * p)) - M_PI64;
    } else {
        double y, y2;
        y = M_PI64 * (2 * p - 1);
        y2 = y * y;
        // Compute x using a rational approximation (_p2(y2) / _q2(y2))
        x = y * _p2(y2) / _q2(y2);
    }

    // For p values between 0.0018 and 0.42, refine the estimate using Halley's method
    if ((0.0018 < p) && (p < 0.42)) {
        // Apply one iteration of Halley's method to improve x
        //    f(x)   = pi + x + sin(x) - y,
        //    f'(x)  = 1 + cos(x),
        //    f''(x) = -sin(x)
        // where y = 2*pi*p.
        double f0, f1, f2;
        f0 = M_PI64 + x + sin(x) - 2 * M_PI64 * p;
        f1 = 1 + cos(x);
        f2 = -sin(x);
        x = x - 2 * f0 * f1 / (2 * f1 * f1 - f0 * f2);
    }

    return sgn * x; // Return the final result with appropriate sign
}
```