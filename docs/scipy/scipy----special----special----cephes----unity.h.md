# `D:\src\scipysrc\scipy\scipy\special\special\cephes\unity.h`

```
/* Translated into C++ by SciPy developers in 2024. */

/*                                                     unity.c
 *
 * Relative error approximations for function arguments near
 * unity.
 *
 *    log1p(x) = log(1+x)
 *    expm1(x) = exp(x) - 1
 *    cosm1(x) = cos(x) - 1
 *    lgam1p(x) = lgam(1+x)
 *
 */

/* Scipy changes:
 * - 06-10-2016: added lgam1p
 */
#pragma once

#include "../config.h"

#include "const.h"
#include "gamma.h"
#include "polevl.h"
#include "zeta.h"

namespace special {
namespace cephes {

    namespace detail {

        /* log1p(x) = log(1 + x)  */

        /* Coefficients for log(1+x) = x - x**2/2 + x**3 P(x)/Q(x)
         * 1/sqrt(2) <= x < sqrt(2)
         * Theoretical peak relative error = 2.32e-20
         */

        constexpr double unity_LP[] = {
            4.5270000862445199635215E-5, 4.9854102823193375972212E-1, 6.5787325942061044846969E0,
            2.9911919328553073277375E1,  6.0949667980987787057556E1,  5.7112963590585538103336E1,
            2.0039553499201281259648E1,
        };

        constexpr double unity_LQ[] = {
            /* 1.0000000000000000000000E0, */
            1.5062909083469192043167E1, 8.3047565967967209469434E1, 2.2176239823732856465394E2,
            3.0909872225312059774938E2, 2.1642788614495947685003E2, 6.0118660497603843919306E1,
        };

    } // namespace detail

    /* Definition of the function log1p(x), computes log(1 + x) */
    SPECFUN_HOST_DEVICE inline double log1p(double x) {
        double z;

        z = 1.0 + x;
        if ((z < M_SQRT1_2) || (z > M_SQRT2))
            return (std::log(z));  // Return natural logarithm if z is out of specified range
        z = x * x;
        // Approximate log(1 + x) using a polynomial approximation for better precision
        z = -0.5 * z + x * (z * polevl(x, detail::unity_LP, 6) / p1evl(x, detail::unity_LQ, 6));
        return (x + z);  // Return the computed log(1 + x)
    }

    /* Definition of the function log(1 + x) - x, computes log(1 + x) - x */
    SPECFUN_HOST_DEVICE inline double log1pmx(double x) {
        if (std::abs(x) < 0.5) {
            uint64_t n;
            double xfac = x;
            double term;
            double res = 0;

            // Taylor series expansion for log(1 + x) - x for small x
            for (n = 2; n < detail::MAXITER; n++) {
                xfac *= -x;
                term = xfac / n;
                res += term;
                if (std::abs(term) < detail::MACHEP * std::abs(res)) {
                    break;
                }
            }
            return res;  // Return the computed value of log(1 + x) - x
        } else {
            return log1p(x) - x;  // Use log1p(x) function for larger values of x
        }
    }

    /* expm1(x) = exp(x) - 1  */

    /*  e^x =  1 + 2x P(x^2)/( Q(x^2) - P(x^2) )
     * -0.5 <= x <= 0.5
     */

    namespace detail {

        constexpr double unity_EP[3] = {
            1.2617719307481059087798E-4,
            3.0299440770744196129956E-2,
            9.9999999999999999991025E-1,
        };

        constexpr double unity_EQ[4] = {
            3.0019850513866445504159E-6,
            2.5244834034968410419224E-3,
            2.2726554820815502876593E-1,
            2.0000000000000000000897E0,
        };

    } // namespace detail

} // namespace cephes
} // namespace special
    SPECFUN_HOST_DEVICE inline double expm1(double x) {
        double r, xx;

        // 如果 x 不是有限数
        if (!std::isfinite(x)) {
            // 如果 x 是 NaN，直接返回 x
            if (std::isnan(x)) {
                return x;
            } 
            // 如果 x 大于 0，返回 x
            else if (x > 0) {
                return x;
            } 
            // 否则返回 -1.0
            else {
                return -1.0;
            }
        }
        // 如果 x 小于 -0.5 或者大于 0.5，直接返回 exp(x) - 1.0
        if ((x < -0.5) || (x > 0.5))
            return (std::exp(x) - 1.0);
        
        // 计算 xx = x * x
        xx = x * x;
        // r = x * polevl(xx, detail::unity_EP, 2)
        r = x * polevl(xx, detail::unity_EP, 2);
        // r = r / (polevl(xx, detail::unity_EQ, 3) - r)
        r = r / (polevl(xx, detail::unity_EQ, 3) - r);
        // 返回 r + r
        return (r + r);
    }

    /* cosm1(x) = cos(x) - 1  */

    namespace detail {
        // 定义常量数组 unity_coscof
        constexpr double unity_coscof[7] = {
            4.7377507964246204691685E-14, -1.1470284843425359765671E-11, 2.0876754287081521758361E-9,
            -2.7557319214999787979814E-7, 2.4801587301570552304991E-5,   -1.3888888888888872993737E-3,
            4.1666666666666666609054E-2,
        };

    }

    SPECFUN_HOST_DEVICE inline double cosm1(double x) {
        double xx;

        // 如果 x 小于 -pi/4 或者大于 pi/4，返回 cos(x) - 1.0
        if ((x < -M_PI_4) || (x > M_PI_4))
            return (std::cos(x) - 1.0);
        
        // 计算 xx = x * x
        xx = x * x;
        // 计算 xx = -0.5 * xx + xx * xx * polevl(xx, detail::unity_coscof, 6)
        xx = -0.5 * xx + xx * xx * polevl(xx, detail::unity_coscof, 6);
        // 返回 xx
        return xx;
    }

    namespace detail {
        /* Compute lgam(x + 1) around x = 0 using its Taylor series. */
        // 计算 lgam(x + 1) 的 Taylor 级数，当 x = 0 时
        SPECFUN_HOST_DEVICE inline double lgam1p_taylor(double x) {
            int n;
            double xfac, coeff, res;

            // 如果 x 等于 0，直接返回 0
            if (x == 0) {
                return 0;
            }
            // 初始化 res = -SCIPY_EULER * x
            res = -SCIPY_EULER * x;
            xfac = -x;
            // 循环计算 Taylor 级数的每一项
            for (n = 2; n < 42; n++) {
                xfac *= -x;
                // coeff = zeta(n, 1) * xfac / n
                coeff = special::cephes::zeta(n, 1) * xfac / n;
                // res += coeff
                res += coeff;
                // 如果当前项的绝对值小于 MACHEP * res 的绝对值，跳出循环
                if (std::abs(coeff) < detail::MACHEP * std::abs(res)) {
                    break;
                }
            }

            // 返回计算结果 res
            return res;
        }
    } // namespace detail

    /* Compute lgam(x + 1). */
    // 计算 lgam(x + 1)
    SPECFUN_HOST_DEVICE inline double lgam1p(double x) {
        // 如果 x 的绝对值小于等于 0.5，使用 Taylor 级数计算 lgam(x + 1)
        if (std::abs(x) <= 0.5) {
            return detail::lgam1p_taylor(x);
        } 
        // 如果 x - 1 的绝对值小于 0.5，返回 std::log(x) + detail::lgam1p_taylor(x - 1)
        else if (std::abs(x - 1) < 0.5) {
            return std::log(x) + detail::lgam1p_taylor(x - 1);
        } 
        // 否则调用 lgam(x + 1) 函数
        else {
            return lgam(x + 1);
        }
    }
} // 结束 cephes 命名空间
} // 结束 special 命名空间
```