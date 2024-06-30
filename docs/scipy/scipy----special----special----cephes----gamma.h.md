# `D:\src\scipysrc\scipy\scipy\special\special\cephes\gamma.h`

```
/*
 * Translated into C++ by SciPy developers in 2024.
 * Original header with Copyright information appears below.
 */
/*
 *     Gamma function
 *
 * SYNOPSIS:
 *
 * double x, y, Gamma();
 *
 * y = Gamma( x );
 *
 * DESCRIPTION:
 *
 * Returns Gamma function of the argument.  The result is
 * correctly signed.
 *
 * Arguments |x| <= 34 are reduced by recurrence and the function
 * approximated by a rational function of degree 6/7 in the
 * interval (2,3).  Large arguments are handled by Stirling's
 * formula. Large negative arguments are made positive using
 * a reflection formula.
 *
 * ACCURACY:
 *
 *                      Relative error:
 * arithmetic   domain     # trials      peak         rms
 *    IEEE    -170,-33      20000       2.3e-15     3.3e-16
 *    IEEE     -33,  33     20000       9.4e-16     2.2e-16
 *    IEEE      33, 171.6   20000       2.3e-15     3.2e-16
 *
 * Error for arguments outside the test range will be larger
 * owing to error amplification by the exponential function.
 */
/*
 *                                                     lgam()
 *
 *     Natural logarithm of Gamma function
 *
 * SYNOPSIS:
 *
 * double x, y, lgam();
 *
 * y = lgam( x );
 *
 * DESCRIPTION:
 *
 * Returns the base e (2.718...) logarithm of the absolute
 * value of the Gamma function of the argument.
 *
 * For arguments greater than 13, the logarithm of the Gamma
 * function is approximated by the logarithmic version of
 * Stirling's formula using a polynomial approximation of
 * degree 4. Arguments between -33 and +33 are reduced by
 * recurrence to the interval [2,3] of a rational approximation.
 * The cosecant reflection formula is employed for arguments
 * less than -33.
 *
 * Arguments greater than MAXLGM return INFINITY and an error
 * message.  MAXLGM = 2.556348e305 for IEEE arithmetic.
 *
 * ACCURACY:
 *
 * arithmetic      domain        # trials     peak         rms
 *    IEEE    0, 3                 28000     5.4e-16     1.1e-16
 *    IEEE    2.718, 2.556e305     40000     3.5e-16     8.3e-17
 *
 * The error criterion was relative when the function magnitude
 * was greater than one but absolute when it was less than one.
 *
 * The following test used the relative error criterion, though
 * at certain points the relative error could be much higher than
 * indicated.
 *    IEEE    -200, -4             10000     4.8e-16     1.3e-16
 */
/*
 * Cephes Math Library Release 2.2:  July, 1992
 * Copyright 1984, 1987, 1989, 1992 by Stephen L. Moshier
 * Direct inquiries to 30 Frost Street, Cambridge, MA 02140
 */
#pragma once

#include "../config.h"
#include "../error.h"
#include "const.h"
#include "polevl.h"
#include "trig.h"

namespace special {
namespace cephes {
    namespace detail {
        // Gamma函数的系数数组，用于计算P
        constexpr double gamma_P[] = {1.60119522476751861407E-4, 1.19135147006586384913E-3, 1.04213797561761569935E-2,
                                      4.76367800457137231464E-2, 2.07448227648435975150E-1, 4.94214826801497100753E-1,
                                      9.99999999999999996796E-1};

        // Gamma函数的系数数组，用于计算Q
        constexpr double gamma_Q[] = {-2.31581873324120129819E-5, 5.39605580493303397842E-4, -4.45641913851797240494E-3,
                                      1.18139785222060435552E-2,  3.58236398605498653373E-2, -2.34591795718243348568E-1,
                                      7.14304917030273074085E-2,  1.00000000000000000320E0};

        /* Stirling公式用于Gamma函数 */
        // Stirling公式的系数数组，适用于33 <= x <= 172
        constexpr double gamma_STIR[5] = {
            7.87311395793093628397E-4, -2.29549961613378126380E-4, -2.68132617805781232825E-3,
            3.47222221605458667310E-3, 8.33333333333482257126E-2,
        };

        // Stirling公式中的最大值
        constexpr double MAXSTIR = 143.01608;

        /* 使用Stirling公式计算的Gamma函数。
         * 对于x >= MAXGAM，返回无穷大。
         * 多项式STIR适用于33 <= x <= 172。
         */
        SPECFUN_HOST_DEVICE inline double stirf(double x) {
            double y, w, v;

            if (x >= MAXGAM) {
                return (std::numeric_limits<double>::infinity());
            }
            w = 1.0 / x;
            // 计算多项式P的值
            w = 1.0 + w * special::cephes::polevl(w, gamma_STIR, 4);
            y = std::exp(x);
            if (x > MAXSTIR) { /* 避免pow()中的溢出 */
                v = std::pow(x, 0.5 * x - 0.25);
                y = v * (v / y);
            } else {
                y = std::pow(x, x - 0.5) / y;
            }
            // 计算Gamma函数的值
            y = SQRTPI * y * w;
            return (y);
        }
    } // namespace detail
    // 定义 Gamma 函数，计算 Gamma(x) 的值
    SPECFUN_HOST_DEVICE inline double Gamma(double x) {
        // 定义变量 p, q, z 以及整数 i
        double p, q, z;
        int i;
        // 默认 Gamma 函数符号设为正
        int sgngam = 1;
    
        // 检查 x 是否为有限值，如果不是则直接返回 x
        if (!std::isfinite(x)) {
            return x;
        }
        // 计算 q 为 x 的绝对值
        q = std::abs(x);
    
        // 如果 q 大于 33.0
        if (q > 33.0) {
            // 如果 x 小于 0.0
            if (x < 0.0) {
                // p 为 q 的下取整
                p = std::floor(q);
                // 如果 p 等于 q，则跳转到错误处理标签 gamnan
                if (p == q) {
                gamnan:
                    // 设置错误信息为 "Gamma"，错误类型为 SF_ERROR_OVERFLOW，无附加数据
                    set_error("Gamma", SF_ERROR_OVERFLOW, NULL);
                    // 返回正无穷大
                    return (std::numeric_limits<double>::infinity());
                }
                // i 为 p 的整数部分
                i = p;
                // 如果 i 是偶数，设定 Gamma 函数符号为负
                if ((i & 1) == 0) {
                    sgngam = -1;
                }
                // 计算 z 为 q 减去 p
                z = q - p;
                // 如果 z 大于 0.5，则增加 p 为 p + 1.0，并重新计算 z
                if (z > 0.5) {
                    p += 1.0;
                    z = q - p;
                }
                // 计算 z 为 q 乘以 sinpi(z)
                z = q * sinpi(z);
                // 如果 z 等于 0.0，则返回 Gamma 函数符号乘以正无穷大
                if (z == 0.0) {
                    return (sgngam * std::numeric_limits<double>::infinity());
                }
                // 计算 z 为 z 的绝对值乘以 M_PI 除以 stirf(q)
                z = std::abs(z);
                z = M_PI / (z * detail::stirf(q));
            } else {
                // 如果 x 大于等于 0.0，计算 z 为 stirf(x)
                z = detail::stirf(x);
            }
            // 返回 Gamma 函数符号乘以 z
            return (sgngam * z);
        }
    
        // 如果 q 不大于 33.0，则进入以下循环和条件判断
    
        // 初始化 z 为 1.0
        z = 1.0;
        // 当 x 大于等于 3.0 时，循环递减 x，并将 z 乘以 x
        while (x >= 3.0) {
            x -= 1.0;
            z *= x;
        }
    
        // 当 x 小于 0.0 时的处理
        while (x < 0.0) {
            // 如果 x 大于 -1.E-9，则跳转到 small 标签
            if (x > -1.E-9) {
                goto small;
            }
            // z 除以 x，并 x 增加 1.0
            z /= x;
            x += 1.0;
        }
    
        // 当 x 小于 2.0 但不小于 0.0 时的处理
        while (x < 2.0) {
            // 如果 x 小于 1.e-9，则跳转到 small 标签
            if (x < 1.e-9) {
                goto small;
            }
            // z 除以 x，并 x 增加 1.0
            z /= x;
            x += 1.0;
        }
    
        // 如果 x 等于 2.0，则直接返回 z
        if (x == 2.0) {
            return (z);
        }
    
        // 如果 x 大于 2.0，则减去 2.0 并计算 p, q 的值
        x -= 2.0;
        p = polevl(x, detail::gamma_P, 6);
        q = polevl(x, detail::gamma_Q, 7);
        // 返回 z 乘以 p 除以 q
        return (z * p / q);
    
    small:
        // 如果 x 等于 0.0，则跳转到 gamnan 标签
        if (x == 0.0) {
            goto gamnan;
        } else
            // 返回 z 除以 ((1.0 + 0.5772156649015329 * x) * x)
            return (z / ((1.0 + 0.5772156649015329 * x) * x));
    }
    namespace detail {
        /* A[]: Stirling's formula expansion of log Gamma
         * B[], C[]: log Gamma function between 2 and 3
         */
        // Stirling公式对log Gamma的展开系数A
        constexpr double gamma_A[] = {8.11614167470508450300E-4, -5.95061904284301438324E-4, 7.93650340457716943945E-4,
                                      -2.77777777730099687205E-3, 8.33333333333331927722E-2};

        // log Gamma函数在2到3之间的系数B
        constexpr double gamma_B[] = {-1.37825152569120859100E3, -3.88016315134637840924E4, -3.31612992738871184744E5,
                                      -1.16237097492762307383E6, -1.72173700820839662146E6, -8.53555664245765465627E5};

        // log Gamma函数在2到3之间的系数C
        constexpr double gamma_C[] = {
            /* 1.00000000000000000000E0, */
            -3.51815701436523470549E2, -1.70642106651881159223E4, -2.20528590553854454839E5,
            -1.13933444367982507207E6, -2.53252307177582951285E6, -2.01889141433532773231E6};

        /* log( sqrt( 2*pi ) ) */
        // log( sqrt( 2*pi ) ) 的值
        constexpr double LS2PI = 0.91893853320467274178;

        // Gamma函数的最大可接受参数值
        constexpr double MAXLGM = 2.556348e305;

        /* Disable optimizations for this function on 32 bit systems when compiling with GCC.
         * We've found that enabling optimizations can result in degraded precision
         * for this asymptotic approximation in that case. */
        // 在使用GCC编译时，禁用32位系统上对此函数的优化。
        // 我们发现在这种情况下启用优化可能会导致渐近逼近的精度下降。
#if defined(__GNUC__) && defined(__i386__)
#pragma GCC push_options
#pragma GCC optimize("00")
#endif

// 定义一个函数 lgam_large_x，用于计算对于大于 1000 的 x 值的对数 Gamma 函数的值
SPECFUN_HOST_DEVICE inline double lgam_large_x(double x) {
    // 根据 Stirling 公式计算 q 值
    double q = (x - 0.5) * std::log(x) - x + LS2PI;
    
    // 如果 x 大于 1.0e8，直接返回 q 值
    if (x > 1.0e8) {
        return q;
    }
    
    // 计算 p 值
    double p = 1.0 / (x * x);
    p = ((7.9365079365079365079365e-4 * p - 2.7777777777777777777778e-3) * p + 0.0833333333333333333333) / x;
    
    // 返回 q + p
    return q + p;
}

#if defined(__GNUC__) && defined(__i386__)
#pragma GCC pop_options
#endif

// 定义一个函数 lgam_sgn，计算带符号的对数 Gamma 函数值，并设置符号
SPECFUN_HOST_DEVICE inline double lgam_sgn(double x, int *sign) {
    double p, q, u, w, z;
    int i;
    
    // 默认符号为正
    *sign = 1;
    
    // 如果 x 不是有限值，直接返回 x
    if (!std::isfinite(x)) {
        return x;
    }
    
    // 当 x 小于 -34.0 时的处理逻辑
    if (x < -34.0) {
        q = -x;
        w = lgam_sgn(q, sign); // 递归调用 lgam_sgn 计算 w
        p = std::floor(q);
        
        // 如果 p 等于 q，说明 q 是整数，发生奇异性
        if (p == q) {
        lgsing:
            // 设置错误信息，并返回正无穷
            set_error("lgam", SF_ERROR_SINGULAR, NULL);
            return (std::numeric_limits<double>::infinity());
        }
        
        i = p;
        if ((i & 1) == 0) {
            *sign = -1; // 如果 i 是偶数，符号为负
        } else {
            *sign = 1; // 如果 i 是奇数，符号为正
        }
        
        z = q - p;
        if (z > 0.5) {
            p += 1.0;
            z = p - q;
        }
        
        z = q * sinpi(z);
        if (z == 0.0) {
            goto lgsing; // 如果 z 是 0，跳转到 lgsing 标签处处理
        }
        
        // 计算 z 的对数值
        z = LOGPI - std::log(z) - w;
        return (z);
    }
    
    // 当 x 小于 13.0 且不是小于 -34.0 的情况下的处理逻辑
    if (x < 13.0) {
        z = 1.0;
        p = 0.0;
        u = x;
        
        // 循环计算 z 和 p
        while (u >= 3.0) {
            p -= 1.0;
            u = x + p;
            z *= u;
        }
        
        while (u < 2.0) {
            if (u == 0.0) {
                goto lgsing; // 如果 u 是 0，跳转到 lgsing 标签处处理
            }
            z /= u;
            p += 1.0;
            u = x + p;
        }
        
        // 如果 z 小于 0，设置符号为负
        if (z < 0.0) {
            *sign = -1;
            z = -z;
        } else {
            *sign = 1; // 否则设置符号为正
        }
        
        // 如果 u 等于 2.0，返回 z 的对数值
        if (u == 2.0) {
            return (std::log(z));
        }
        
        // 计算 p，并返回 z 的对数值加上 p
        p -= 2.0;
        x = x + p;
        p = x * polevl(x, gamma_B, 5) / p1evl(x, gamma_C, 6);
        return (std::log(z) + p);
    }
    
    // 当 x 大于 MAXLGM 时，返回对应的符号乘以正无穷
    if (x > MAXLGM) {
        return (*sign * std::numeric_limits<double>::infinity());
    }
    
    // 当 x 大于等于 1000.0 且小于 13.0 时，调用 lgam_large_x 函数计算结果
    if (x >= 1000.0) {
        return lgam_large_x(x);
    }
    
    // 默认情况下，计算 q 值和 p 值，并返回 q + p
    q = (x - 0.5) * std::log(x) - x + LS2PI;
    p = 1.0 / (x * x);
    return q + polevl(p, gamma_A, 4) / x;
}
    /* Logarithm of Gamma function */
    // 计算 Gamma 函数的对数值
    SPECFUN_HOST_DEVICE inline double lgam(double x) {
        // 调用具体实现函数 lgam_sgn，获取 Gamma 函数对数值及其符号
        int sign;
        return detail::lgam_sgn(x, &sign);
    }

    /* Sign of the Gamma function */
    // 返回 Gamma 函数的符号
    SPECFUN_HOST_DEVICE inline double gammasgn(double x) {
        double fx;

        // 如果 x 是 NaN，则返回 x 自身
        if (std::isnan(x)) {
            return x;
        }
        // 如果 x 大于 0，则 Gamma 函数的符号为正
        if (x > 0) {
            return 1.0;
        } else {
            // 否则，计算 x 的整数部分 fx
            fx = std::floor(x);
            // 如果 x 是整数，则 Gamma 函数的符号为 0
            if (x - fx == 0.0) {
                return 0.0;
            // 如果 x 的整数部分 fx 是奇数，则 Gamma 函数的符号为 -1
            } else if (static_cast<int>(fx) % 2) {
                return -1.0;
            // 否则， Gamma 函数的符号为 1
            } else {
                return 1.0;
            }
        }
    }
} // namespace cephes
} // namespace special
```