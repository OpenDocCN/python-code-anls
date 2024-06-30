# `D:\src\scipysrc\scipy\scipy\special\special\cephes\zeta.h`

```
/*
 * Translated into C++ by SciPy developers in 2024.
 * Original header with Copyright information appears below.
 */

/*
 * zeta.c
 *
 * Riemann zeta function of two arguments
 *
 * SYNOPSIS:
 *
 * double x, q, y, zeta();
 *
 * y = zeta( x, q );
 *
 * DESCRIPTION:
 *
 * zeta(x,q) = sum_{k=0}^inf (k+q)^(-x)
 *           = 1 + 2^(-x) + 3^(-x) + ...
 *
 * where x > 1 and q is not a negative integer or zero.
 * The Euler-Maclaurin summation formula is used to obtain
 * the expansion:
 *
 * zeta(x,q) = sum_{k=1}^n (k+q)^(-x)
 *           + 1 / (x-1) * (n+q)^(-x)
 *           + 1/2 * (n+q)^(-x) / x
 *           - sum_{j=1}^inf B_{2j} * x(x+1)...(x+2j-1) / (2j)! * (n+q)^(-x)
 *
 * ACCURACY:
 *
 * REFERENCE:
 *
 * Gradshteyn, I. S., and I. M. Ryzhik, Tables of Integrals,
 * Series, and Products, p. 1073; Academic Press, 1980.
 */

/*
 * Cephes Math Library Release 2.0:  April, 1987
 * Copyright 1984, 1987 by Stephen L. Moshier
 * Direct inquiries to 30 Frost Street, Cambridge, MA 02140
 */
#pragma once

#include "../config.h"
#include "../error.h"
#include "const.h"

namespace special {
namespace cephes {

    namespace detail {
        /* Expansion coefficients
         * for Euler-Maclaurin summation formula
         * (2k)! / B2k
         * where B2k are Bernoulli numbers
         */
        constexpr double zeta_A[] = {
            12.0,
            -720.0,
            30240.0,
            -1209600.0,
            47900160.0,
            -1.8924375803183791606e9, /*1.307674368e12/691 */
            7.47242496e10,
            -2.950130727918164224e12,  /*1.067062284288e16/3617 */
            1.1646782814350067249e14,  /*5.109094217170944e18/43867 */
            -4.5979787224074726105e15, /*8.028576626982912e20/174611 */
            1.8152105401943546773e17,  /*1.5511210043330985984e23/854513 */
            -7.1661652561756670113e18  /*1.6938241367317436694528e27/236364091 */
        };

        /* 30 Nov 86 -- error in third coefficient fixed */
    } // namespace detail

} // namespace cephes
} // namespace special
    // 定义一个双精度函数 zeta，接受两个双精度参数 x 和 q
    SPECFUN_HOST_DEVICE double inline zeta(double x, double q) {
        int i;  // 声明整型变量 i，用于循环计数
        double a, b, k, s, t, w;  // 声明双精度变量 a, b, k, s, t, w，用于计算和存储中间结果

        // 如果 x 等于 1.0，跳转到返回无穷大的标签
        if (x == 1.0)
            goto retinf;

        // 如果 x 小于 1.0，说明输入不在定义域内，跳转到定义域错误的标签
        if (x < 1.0) {
        domerr:
            set_error("zeta", SF_ERROR_DOMAIN, NULL);  // 设置错误消息，指示定义域错误
            return (std::numeric_limits<double>::quiet_NaN());  // 返回 NaN 表示未定义的结果
        }

        // 如果 q 小于等于 0.0
        if (q <= 0.0) {
            // 如果 q 是整数，设置奇点错误消息
            if (q == floor(q)) {
                set_error("zeta", SF_ERROR_SINGULAR, NULL);
            retinf:
                return (std::numeric_limits<double>::infinity());  // 返回正无穷表示结果为无穷
            }
            // 如果 x 不是整数，则跳转到定义域错误的标签，因为 q^-x 没有定义
            if (x != std::floor(x))
                goto domerr;
        }

        // 如果 q 大于 1e8，使用渐近展开计算 Riemann zeta 函数
        if (q > 1e8) {
            return (1 / (x - 1) + 1 / (2 * q)) * std::pow(q, 1 - x);
        }

        // 使用 Euler-Maclaurin 求和公式计算 Riemann zeta 函数

        // 计算初值 s，即 q^-x
        s = std::pow(q, -x);
        a = q;
        i = 0;
        b = 0.0;
        // 循环直到满足条件
        while ((i < 9) || (a <= 9.0)) {
            i += 1;
            a += 1.0;
            b = std::pow(a, -x);
            s += b;
            // 如果 b/s 的绝对值小于机器精度，跳转到完成标签
            if (std::abs(b / s) < detail::MACHEP)
                goto done;
        }

        // 继续计算余项 w
        w = a;
        s += b * w / (x - 1.0);
        s -= 0.5 * b;
        a = 1.0;
        k = 0.0;
        // 进一步计算 Riemann zeta 函数的近似值
        for (i = 0; i < 12; i++) {
            a *= x + k;
            b /= w;
            t = a * b / detail::zeta_A[i];
            s = s + t;
            t = std::abs(t / s);
            // 如果 t 小于机器精度，跳转到完成标签
            if (t < detail::MACHEP)
                goto done;
            k += 1.0;
            a *= x + k;
            b /= w;
            k += 1.0;
        }
    done:
        return (s);  // 返回计算得到的 Riemann zeta 函数值
    }
} // 结束 cephes 命名空间
} // 结束 special 命名空间
```