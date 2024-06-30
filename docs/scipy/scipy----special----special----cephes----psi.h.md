# `D:\src\scipysrc\scipy\scipy\special\special\cephes\psi.h`

```
/*
 * Translated into C++ by SciPy developers in 2024.
 * Original header with Copyright information appears below.
 */

/*
 *                                                     psi.c
 *
 *     Psi (digamma) function
 *
 *
 * SYNOPSIS:
 *
 * double x, y, psi();
 *
 * y = psi( x );
 *
 *
 * DESCRIPTION:
 *
 *              d      -
 *   psi(x)  =  -- ln | (x)
 *              dx
 *
 * is the logarithmic derivative of the gamma function.
 * For integer x,
 *                   n-1
 *                    -
 * psi(n) = -EUL  +   >  1/k.
 *                    -
 *                   k=1
 *
 * This formula is used for 0 < n <= 10.  If x is negative, it
 * is transformed to a positive argument by the reflection
 * formula  psi(1-x) = psi(x) + pi cot(pi x).
 * For general positive x, the argument is made greater than 10
 * using the recurrence  psi(x+1) = psi(x) + 1/x.
 * Then the following asymptotic expansion is applied:
 *
 *                           inf.   B
 *                            -      2k
 * psi(x) = log(x) - 1/2x -   >   -------
 *                            -        2k
 *                           k=1   2k x
 *
 * where the B2k are Bernoulli numbers.
 *
 * ACCURACY:
 *    Relative error (except absolute when |psi| < 1):
 * arithmetic   domain     # trials      peak         rms
 *    IEEE      0,30        30000       1.3e-15     1.4e-16
 *    IEEE      -30,0       40000       1.5e-15     2.2e-16
 *
 * ERROR MESSAGES:
 *     message         condition      value returned
 * psi singularity    x integer <=0      INFINITY
 */

/*
 * Cephes Math Library Release 2.8:  June, 2000
 * Copyright 1984, 1987, 1992, 2000 by Stephen L. Moshier
 */

/*
 * Code for the rational approximation on [1, 2] is:
 *
 * (C) Copyright John Maddock 2006.
 * Use, modification and distribution are subject to the
 * Boost Software License, Version 1.0. (See accompanying file
 * LICENSE_1_0.txt or copy at https://www.boost.org/LICENSE_1_0.txt)
 */

#pragma once

#include "../config.h"
#include "../error.h"
#include "const.h"
#include "polevl.h"

namespace special {
namespace cephes {


This set of comments provides explanations for each section of the given C++ code file, covering its purpose, mathematical definitions, accuracy details, copyright information, and relevant licenses.
    namespace detail {
        constexpr double psi_A[] = {8.33333333333333333333E-2,  -2.10927960927960927961E-2, 7.57575757575757575758E-3,
                                    -4.16666666666666666667E-3, 3.96825396825396825397E-3,  -8.33333333333333333333E-3,
                                    8.33333333333333333333E-2};

        constexpr float psi_Y = 0.99558162689208984f;

        constexpr double psi_root1 = 1569415565.0 / 1073741824.0;
        constexpr double psi_root2 = (381566830.0 / 1073741824.0) / 1073741824.0;
        constexpr double psi_root3 = 0.9016312093258695918615325266959189453125e-19;

        constexpr double psi_P[] = {-0.0020713321167745952, -0.045251321448739056, -0.28919126444774784,
                                    -0.65031853770896507,   -0.32555031186804491,  0.25479851061131551};
        constexpr double psi_Q[] = {-0.55789841321675513e-6,
                                    0.0021284987017821144,
                                    0.054151797245674225,
                                    0.43593529692665969,
                                    1.4606242909763515,
                                    2.0767117023730469,
                                    1.0};

        // 计算在区间 [1, 2] 上的对数斯特林数的第一种形式的逼近
        SPECFUN_HOST_DEVICE double digamma_imp_1_2(double x) {
            /*
             * Rational approximation on [1, 2] taken from Boost.
             *
             * Now for the approximation, we use the form:
             *
             * digamma(x) = (x - root) * (Y + R(x-1))
             *
             * Where root is the location of the positive root of digamma,
             * Y is a constant, and R is optimised for low absolute error
             * compared to Y.
             *
             * Maximum Deviation Found:               1.466e-18
             * At double precision, max error found:  2.452e-17
             */
            double r, g;

            g = x - psi_root1;  // 计算 x - root1
            g -= psi_root2;     // 进一步减去 root2
            g -= psi_root3;     // 最后减去 root3
            r = special::cephes::polevl(x - 1.0, psi_P, 5) / special::cephes::polevl(x - 1.0, psi_Q, 6);  // 使用 P 和 Q 进行有理逼近

            return g * psi_Y + g * r;  // 返回逼近结果
        }

        // 针对大于等于 1e17 的情况进行的对数斯特林数第一种形式的渐近逼近
        SPECFUN_HOST_DEVICE double psi_asy(double x) {
            double y, z;

            if (x < 1.0e17) {
                z = 1.0 / (x * x);  // 计算 z = 1 / (x * x)
                y = z * special::cephes::polevl(z, psi_A, 6);  // 使用 A 进行多项式逼近
            } else {
                y = 0.0;  // 对于大于等于 1e17 的 x，直接返回 0
            }

            return std::log(x) - (0.5 / x) - y;  // 返回渐近逼近的结果
        }
    } // namespace detail
    // 计算 psi 函数的值，psi 是 polygamma 函数的一种特殊情况
    SPECFUN_HOST_DEVICE double psi(double x) {
        // 初始化变量 y、q、r
        double y = 0.0;
        double q, r;
        int i, n;

        // 检查 x 是否为 NaN
        if (std::isnan(x)) {
            return x;  // 如果是 NaN，直接返回 x
        } else if (x == std::numeric_limits<double>::infinity()) {
            return x;  // 如果是正无穷大，直接返回 x
        } else if (x == -std::numeric_limits<double>::infinity()) {
            return std::numeric_limits<double>::quiet_NaN();  // 如果是负无穷大，返回 NaN
        } else if (x == 0) {
            // 如果 x 等于 0，设置错误信息并返回符号与 -x 相反的无穷大
            set_error("psi", SF_ERROR_SINGULAR, NULL);
            return std::copysign(std::numeric_limits<double>::infinity(), -x);
        } else if (x < 0.0) {
            // 如果 x 小于 0，进行参数化处理以便计算 tan(pi * x)
            r = std::modf(x, &q);  // 分解 x 为整数部分 q 和小数部分 r
            if (r == 0.0) {
                // 如果 r 为 0，表示 x 是整数，设置错误信息并返回 NaN
                set_error("psi", SF_ERROR_SINGULAR, NULL);
                return std::numeric_limits<double>::quiet_NaN();
            }
            // 计算 y，调整 x 为 1.0 - x
            y = -M_PI / std::tan(M_PI * r);
            x = 1.0 - x;
        }

        // 检查 x 是否为正整数且不大于 10
        if ((x <= 10.0) && (x == std::floor(x))) {
            n = static_cast<int>(x);
            // 计算 psi 的精确值直到 n-1
            for (i = 1; i < n; i++) {
                y += 1.0 / i;
            }
            // 减去欧拉常数
            y -= detail::SCIPY_EULER;
            return y;
        }

        // 使用递推关系将 x 移动到 [1, 2] 区间
        if (x < 1.0) {
            y -= 1.0 / x;
            x += 1.0;
        } else if (x < 10.0) {
            // 如果 x 在 (1, 10] 区间内，使用循环将 x 移动到 [1, 2] 区间
            while (x > 2.0) {
                x -= 1.0;
                y += 1.0 / x;
            }
        }
        // 如果 x 在 [1, 2] 区间内，计算 psi 的值
        if ((1.0 <= x) && (x <= 2.0)) {
            y += detail::digamma_imp_1_2(x);  // 调用特定区间 [1, 2] 的 digamma 函数
            return y;
        }

        // 对于较大的 x，使用渐近级数近似计算 psi 的值
        y += detail::psi_asy(x);  // 调用渐近级数计算 psi 的值
        return y;  // 返回计算结果
    }
} // 结束 cephes 命名空间的定义
} // 结束 special 命名空间的定义
```