# `D:\src\scipysrc\scipy\scipy\special\special\loggamma.h`

```
/* Translated from Cython into C++ by SciPy developers in 2024.
 * Original header comment appears below.
 */

/* An implementation of the principal branch of the logarithm of
 * Gamma. Also contains implementations of Gamma and 1/Gamma which are
 * easily computed from log-Gamma.
 *
 * Author: Josh Wilson
 *
 * Distributed under the same license as Scipy.
 *
 * References
 * ----------
 * [1] Hare, "Computing the Principal Branch of log-Gamma",
 *     Journal of Algorithms, 1997.
 *
 * [2] Julia,
 *     https://github.com/JuliaLang/julia/blob/master/base/special/gamma.jl
 */

#pragma once

#include "cephes/gamma.h"
#include "cephes/rgamma.h"
#include "config.h"
#include "error.h"
#include "evalpoly.h"
#include "trig.h"
#include "zlog1.h"

namespace special {

namespace detail {
    constexpr double loggamma_SMALLX = 7;
    constexpr double loggamma_SMALLY = 7;
    constexpr double loggamma_HLOG2PI = 0.918938533204672742;      // log(2*pi)/2
    constexpr double loggamma_LOGPI = 1.1447298858494001741434262; // log(pi)
    constexpr double loggamma_TAYLOR_RADIUS = 0.2;

    SPECFUN_HOST_DEVICE std::complex<double> loggamma_stirling(std::complex<double> z) {
        /* Stirling series for log-Gamma
         *
         * The coefficients are B[2*n]/(2*n*(2*n - 1)) where B[2*n] is the
         * (2*n)th Bernoulli number. See (1.1) in [1].
         */
        double coeffs[] = {-2.955065359477124183E-2,  6.4102564102564102564E-3, -1.9175269175269175269E-3,
                           8.4175084175084175084E-4,  -5.952380952380952381E-4, 7.9365079365079365079E-4,
                           -2.7777777777777777778E-3, 8.3333333333333333333E-2};
        std::complex<double> rz = 1.0 / z;
        std::complex<double> rzz = rz / z;

        return (z - 0.5) * std::log(z) - z + loggamma_HLOG2PI + rz * cevalpoly(coeffs, 7, rzz);
    }

    SPECFUN_HOST_DEVICE std::complex<double> loggamma_recurrence(std::complex<double> z) {
        /* Backward recurrence relation.
         *
         * See Proposition 2.2 in [1] and the Julia implementation [2].
         *
         */
        int signflips = 0;              // 记录符号变化次数
        int sb = 0;                     // 上一个符号位
        std::complex<double> shiftprod = z;  // 乘积的累积

        z += 1.0;
        int nsb;
        while (z.real() <= loggamma_SMALLX) {
            shiftprod *= z;            // 累积乘积
            nsb = std::signbit(shiftprod.imag());  // 获取虚部的符号位
            signflips += nsb != 0 && sb == 0 ? 1 : 0;  // 计算符号变化次数
            sb = nsb;                   // 更新上一个符号位
            z += 1.0;                   // 自增 z
        }
        return loggamma_stirling(z) - std::log(shiftprod) - signflips * 2 * M_PI * std::complex<double>(0, 1);
    }
    # 定义函数 loggamma_taylor，计算对数 Gamma 函数的 Taylor 级数近似
    SPECFUN_HOST_DEVICE std::complex<double> loggamma_taylor(std::complex<double> z) {
        /* Taylor series for log-Gamma around z = 1.
         *
         * It is
         *
         * loggamma(z + 1) = -gamma*z + zeta(2)*z**2/2 - zeta(3)*z**3/3 ...
         *
         * where gamma is the Euler-Mascheroni constant.
         */

        // 定义 Taylor 级数的系数
        double coeffs[] = {
            -4.3478266053040259361E-2, 4.5454556293204669442E-2, -4.7619070330142227991E-2, 5.000004769810169364E-2,
            -5.2631679379616660734E-2, 5.5555767627403611102E-2, -5.8823978658684582339E-2, 6.2500955141213040742E-2,
            -6.6668705882420468033E-2, 7.1432946295361336059E-2, -7.6932516411352191473E-2, 8.3353840546109004025E-2,
            -9.0954017145829042233E-2, 1.0009945751278180853E-1, -1.1133426586956469049E-1, 1.2550966952474304242E-1,
            -1.4404989676884611812E-1, 1.6955717699740818995E-1, -2.0738555102867398527E-1, 2.7058080842778454788E-1,
            -4.0068563438653142847E-1, 8.2246703342411321824E-1, -5.7721566490153286061E-1};

        // 将 z 减去 1，以进行 Taylor 级数的计算
        z -= 1.0;
        // 返回 z 乘以给定系数的多项式值
        return z * cevalpoly(coeffs, 22, z);
    }
} // namespace detail

// 计算双精度浮点数 x 的对数 Gamma 函数，返回 NaN 如果 x < 0
SPECFUN_HOST_DEVICE inline double loggamma(double x) {
    if (x < 0.0) {
        // 如果输入值小于 0，则返回双精度浮点数的 NaN
        return std::numeric_limits<double>::quiet_NaN();
    }
    // 调用 cephes 库中的 lgam 函数计算对数 Gamma 函数的值
    return cephes::lgam(x);
}

// 计算单精度浮点数 x 的对数 Gamma 函数，调用双精度版本的 loggamma
SPECFUN_HOST_DEVICE inline float loggamma(float x) { return loggamma(static_cast<double>(x)); }

// 计算复数 z 的对数 Gamma 函数，返回主支路的值
SPECFUN_HOST_DEVICE inline std::complex<double> loggamma(std::complex<double> z) {
    // 计算对数 Gamma 函数的主支路

    // 如果实部或虚部为 NaN，则返回复数 NaN
    if (std::isnan(z.real()) || std::isnan(z.imag())) {
        return {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
    }
    // 如果实部小于等于 0 并且 z 是整数，则设定错误状态并返回复数 NaN
    if (z.real() <= 0 and z == std::floor(z.real())) {
        set_error("loggamma", SF_ERROR_SINGULAR, NULL);
        return {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
    }
    // 如果实部大于 loggamma_SMALLX 或者虚部绝对值大于 loggamma_SMALLY，则调用 detail 命名空间中的 loggamma_stirling 函数
    if (z.real() > detail::loggamma_SMALLX || std::abs(z.imag()) > detail::loggamma_SMALLY) {
        return detail::loggamma_stirling(z);
    }
    // 如果 z 和 1 的差的绝对值小于 loggamma_TAYLOR_RADIUS，则调用 detail 命名空间中的 loggamma_taylor 函数
    if (std::abs(z - 1.0) < detail::loggamma_TAYLOR_RADIUS) {
        return detail::loggamma_taylor(z);
    }
    // 如果 z 和 2 的差的绝对值小于 loggamma_TAYLOR_RADIUS，则根据递推关系和围绕 1 的 Taylor 级数，返回计算结果
    if (std::abs(z - 2.0) < detail::loggamma_TAYLOR_RADIUS) {
        return detail::zlog1(z - 1.0) + detail::loggamma_taylor(z - 1.0);
    }
    // 如果实部小于 0.1，则使用反射公式计算对数 Gamma 函数的值
    if (z.real() < 0.1) {
        // 根据 Proposition 3.1 中的建议，计算反射公式的结果
        double tmp = std::copysign(2 * M_PI, z.imag()) * std::floor(0.5 * z.real() + 0.25);
        return std::complex<double>(detail::loggamma_LOGPI, tmp) - std::log(sinpi(z)) - loggamma(1.0 - z);
    }
    // 如果虚部非负，则调用 detail 命名空间中的 loggamma_recurrence 函数
    if (std::signbit(z.imag()) == 0) {
        return detail::loggamma_recurrence(z);
    }
    // 对虚部取共轭，然后调用 detail 命名空间中的 loggamma_recurrence 函数
    return std::conj(detail::loggamma_recurrence(std::conj(z)));
}

// 计算复数 z 的单精度浮点数版本的对数 Gamma 函数，转换为双精度复数后调用双精度版本的 loggamma
SPECFUN_HOST_DEVICE inline std::complex<float> loggamma(std::complex<float> z) {
    return static_cast<std::complex<float>>(loggamma(static_cast<std::complex<double>>(z)));
}

// 计算双精度浮点数 z 的 Gamma 函数的倒数，返回 exp(-loggamma(z))
SPECFUN_HOST_DEVICE inline double rgamma(double z) { return cephes::rgamma(z); }

// 计算单精度浮点数 z 的 Gamma 函数的倒数，调用双精度版本的 rgamma
SPECFUN_HOST_DEVICE inline float rgamma(float z) { return rgamma(static_cast<double>(z)); }

// 计算复数 z 的 Gamma 函数的倒数，返回 1/Gamma(z)，利用 loggamma 函数计算
SPECFUN_HOST_DEVICE inline std::complex<double> rgamma(std::complex<double> z) {
    // 使用 loggamma 计算 1/Gamma(z)
    if (z.real() <= 0 && z == std::floor(z.real())) {
        // 当 z 为 0、-1、-2 等时，返回 0
        return 0.0;
    }
    // 返回 exp(-loggamma(z))
    return std::exp(-loggamma(z));
}

// 计算单精度浮点数 z 的复数版本的 Gamma 函数的倒数，转换为双精度复数后调用双精度版本的 rgamma
SPECFUN_HOST_DEVICE inline std::complex<float> rgamma(std::complex<float> z) {
    return static_cast<std::complex<float>>(rgamma(static_cast<std::complex<double>>(z)));
}

} // namespace special
```