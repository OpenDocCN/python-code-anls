# `D:\src\scipysrc\scipy\scipy\special\special\cephes\igam.h`

```
/*
 * Translated into C++ by SciPy developers in 2024.
 * Original header with Copyright information appears below.
 */

/*
 * igam.c
 *
 * Incomplete Gamma integral
 *
 * SYNOPSIS:
 *
 * double a, x, y, igam();
 *
 * y = igam( a, x );
 *
 * DESCRIPTION:
 *
 * The function is defined by
 *
 *                           x
 *                            -
 *                   1       | |  -t  a-1
 *  igam(a,x)  =   -----     |   e   t   dt.
 *                  -      | |
 *                 | (a)    -
 *                           0
 *
 * In this implementation both arguments must be positive.
 * The integral is evaluated by either a power series or
 * continued fraction expansion, depending on the relative
 * values of a and x.
 *
 * ACCURACY:
 *
 *                      Relative error:
 * arithmetic   domain     # trials      peak         rms
 *    IEEE      0,30       200000       3.6e-14     2.9e-15
 *    IEEE      0,100      300000       9.9e-14     1.5e-14
 */

/*
 * igamc()
 *
 * Complemented incomplete Gamma integral
 *
 * SYNOPSIS:
 *
 * double a, x, y, igamc();
 *
 * y = igamc( a, x );
 *
 * DESCRIPTION:
 *
 * The function is defined by
 *
 *  igamc(a,x)   =   1 - igam(a,x)
 *
 *                            inf.
 *                              -
 *                     1       | |  -t  a-1
 *               =   -----     |   e   t   dt.
 *                    -      | |
 *                   | (a)    -
 *                             x
 *
 * In this implementation both arguments must be positive.
 * The integral is evaluated by either a power series or
 * continued fraction expansion, depending on the relative
 * values of a and x.
 *
 * ACCURACY:
 *
 * Tested at random a, x.
 *                a         x                      Relative error:
 * arithmetic   domain   domain     # trials      peak         rms
 *    IEEE     0.5,100   0,100      200000       1.9e-14     1.7e-15
 *    IEEE     0.01,0.5  0,100      200000       1.4e-13     1.6e-15
 */

/*
 * Cephes Math Library Release 2.0:  April, 1987
 * Copyright 1985, 1987 by Stephen L. Moshier
 * Direct inquiries to 30 Frost Street, Cambridge, MA 02140
 */

/*
 * Sources
 * [1] "The Digital Library of Mathematical Functions", dlmf.nist.gov
 * [2] Maddock et. al., "Incomplete Gamma Functions",
 *     https://www.boost.org/doc/libs/1_61_0/libs/math/doc/html/math_toolkit/sf_gamma/igamma.html
 */

/*
 * Scipy changes:
 * - 05-01-2016: added asymptotic expansion for igam to improve the
 *   a ~ x regime.
 * - 06-19-2016: additional series expansion added for igamc to
 *   improve accuracy at small arguments.
 * - 06-24-2016: better choice of domain for the asymptotic series;
 *   improvements in accuracy for the asymptotic series when a and x
 *   are very close.
 */

#pragma once

#include "../config.h"
#include "../error.h"

#include "const.h"
#include "gamma.h"
#include "igam_asymp_coeff.h"
#include "lanczos.h"


这段代码是一个 C++ 的头文件，声明了关于不完全 Gamma 函数和其补充函数的说明和实现细节，同时包括了版权信息和引用的来源。
// 包含特定的头文件 "ndtr.h" 和 "unity.h"
#include "ndtr.h"
#include "unity.h"

// 声明一个命名空间 special，内部包含命名空间 cephes
namespace special {
namespace cephes {

    // 空的命名空间 detail

    } // namespace detail

    // 内联函数 igamc 的声明
    SPECFUN_HOST_DEVICE inline double igamc(double a, double x);

    // 定义内联函数 igam，计算不完全 Gamma 函数
    SPECFUN_HOST_DEVICE inline double igam(double a, double x) {
        // 变量声明：计算参数 x 与 a 的差的绝对值除以 a
        double absxma_a;

        // 条件：如果 x 或 a 小于 0
        if (x < 0 || a < 0) {
            // 设置错误信息并返回 NaN
            set_error("gammainc", SF_ERROR_DOMAIN, NULL);
            return std::numeric_limits<double>::quiet_NaN();
        } else if (a == 0) { // 条件：如果 a 等于 0
            if (x > 0) {
                return 1; // 返回 1
            } else {
                return std::numeric_limits<double>::quiet_NaN(); // 返回 NaN
            }
        } else if (x == 0) { // 条件：如果 x 等于 0
            /* Zero integration limit */
            return 0; // 返回 0，表示积分下限为零
        } else if (std::isinf(a)) { // 条件：如果 a 是无穷大
            if (std::isinf(x)) {
                return std::numeric_limits<double>::quiet_NaN(); // 返回 NaN
            }
            return 0; // 返回 0
        } else if (std::isinf(x)) { // 条件：如果 x 是无穷大
            return 1; // 返回 1
        }

        // 计算参数 x 与 a 的差的绝对值除以 a
        absxma_a = std::abs(x - a) / a;

        // 条件：在渐近区域，即当 a ~ x 时，参考 [2]
        if ((a > detail::igam_SMALL) && (a < detail::igam_LARGE) && (absxma_a < detail::igam_SMALLRATIO)) {
            return detail::asymptotic_series(a, x, detail::IGAM); // 调用渐近级数计算函数并返回结果
        } else if ((a > detail::igam_LARGE) && (absxma_a < detail::igam_LARGERATIO / std::sqrt(a))) {
            return detail::asymptotic_series(a, x, detail::IGAM); // 调用渐近级数计算函数并返回结果
        }

        // 条件：如果 x 大于 1 且大于 a
        if ((x > 1.0) && (x > a)) {
            return (1.0 - igamc(a, x)); // 返回 1 - igamc(a, x)
        }

        // 调用级数计算函数并返回结果
        return detail::igam_series(a, x);
    }

} // namespace special
} // namespace cephes
    // 计算不完全伽玛函数的互补函数 igamc(a, x)
    // 参数 a：伽玛函数的参数
    // 参数 x：自变量
    SPECFUN_HOST_DEVICE double igamc(double a, double x) {
        double absxma_a;  // 计算 |x - a| / a

        // 处理异常情况：a 或 x 小于 0，返回 NaN
        if (x < 0 || a < 0) {
            set_error("gammaincc", SF_ERROR_DOMAIN, NULL);
            return std::numeric_limits<double>::quiet_NaN();
        } else if (a == 0) {  // 当 a 等于 0 时的特殊情况处理
            if (x > 0) {
                return 0;
            } else {
                return std::numeric_limits<double>::quiet_NaN();
            }
        } else if (x == 0) {  // 当 x 等于 0 时的特殊情况处理
            return 1;
        } else if (std::isinf(a)) {  // 当 a 为无穷大时的特殊情况处理
            if (std::isinf(x)) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            return 1;
        } else if (std::isinf(x)) {  // 当 x 为无穷大时的特殊情况处理
            return 0;
        }

        /* 渐近区域处理，其中 a ~ x；参见 [2]。 */
        absxma_a = std::abs(x - a) / a;
        // 对于满足一定条件的情况，采用渐近级数计算
        if ((a > detail::igam_SMALL) && (a < detail::igam_LARGE) && (absxma_a < detail::igam_SMALLRATIO)) {
            return detail::asymptotic_series(a, x, detail::IGAMC);
        } else if ((a > detail::igam_LARGE) && (absxma_a < detail::igam_LARGERATIO / std::sqrt(a))) {
            return detail::asymptotic_series(a, x, detail::IGAMC);
        }

        /* 其他情况处理；参见 [2]。 */
        if (x > 1.1) {
            if (x < a) {
                return 1.0 - detail::igam_series(a, x);
            } else {
                return detail::igamc_continued_fraction(a, x);
            }
        } else if (x <= 0.5) {
            if (-0.4 / std::log(x) < a) {
                return 1.0 - detail::igam_series(a, x);
            } else {
                return detail::igamc_series(a, x);
            }
        } else {
            if (x * 1.1 < a) {
                return 1.0 - detail::igam_series(a, x);
            } else {
                return detail::igamc_series(a, x);
            }
        }
    }
} // 结束 cephes 命名空间
} // 结束 special 命名空间
```