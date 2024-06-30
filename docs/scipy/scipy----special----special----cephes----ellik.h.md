# `D:\src\scipysrc\scipy\scipy\special\special\cephes\ellik.h`

```
/* Translated into C++ by SciPy developers in 2024.
 * Original header with Copyright information appears below.
 */

/*                                                     ellik.c
 *
 *     Incomplete elliptic integral of the first kind
 *
 *
 *
 * SYNOPSIS:
 *
 * double phi, m, y, ellik();
 *
 * y = ellik( phi, m );
 *
 *
 *
 * DESCRIPTION:
 *
 * Approximates the integral
 *
 *
 *
 *                phi
 *                 -
 *                | |
 *                |           dt
 * F(phi | m) =   |    ------------------
 *                |                   2
 *              | |    sqrt( 1 - m sin t )
 *               -
 *                0
 *
 * of amplitude phi and modulus m, using the arithmetic -
 * geometric mean algorithm.
 *
 *
 *
 *
 * ACCURACY:
 *
 * Tested at random points with m in [0, 1] and phi as indicated.
 *
 *                      Relative error:
 * arithmetic   domain     # trials      peak         rms
 *    IEEE     -10,10       200000      7.4e-16     1.0e-16
 *
 *
 */

/*
 * Cephes Math Library Release 2.0:  April, 1987
 * Copyright 1984, 1987 by Stephen L. Moshier
 * Direct inquiries to 30 Frost Street, Cambridge, MA 02140
 */
/* Copyright 2014, Eric W. Moore */

/*     Incomplete elliptic integral of first kind      */

// 包含一次椭圆积分的头文件声明
#pragma once

// 包含配置和错误处理的头文件
#include "../config.h"
#include "../error.h"

// 包含常数定义和完整椭圆积分的头文件
#include "const.h"
#include "ellpk.h"

// 定义特殊函数的命名空间
namespace special {
namespace cephes {

} // namespace detail
    // 定义一个内联函数 ellik，计算椭圆积分 K(phi, m)
    SPECFUN_HOST_DEVICE inline double ellik(double phi, double m) {
        // 定义变量 a, b, c, e, temp, t, K, denom, npio2
        double a, b, c, e, temp, t, K, denom, npio2;
        // 定义整型变量 d, mod, sign

        // 如果 phi 或 m 是 NaN，则返回 NaN
        if (std::isnan(phi) || std::isnan(m))
            return std::numeric_limits<double>::quiet_NaN();
        // 如果 m 大于 1.0，则返回 NaN
        if (m > 1.0)
            return std::numeric_limits<double>::quiet_NaN();
        // 如果 phi 或 m 是无穷大，处理特殊情况
        if (std::isinf(phi) || std::isinf(m)) {
            // 如果 m 是无穷大且 phi 是有限数，则返回 0.0
            if (std::isinf(m) && std::isfinite(phi))
                return 0.0;
            // 如果 phi 是无穷大且 m 是有限数，则返回 phi
            else if (std::isinf(phi) && std::isfinite(m))
                return phi;
            // 其他情况返回 NaN
            else
                return std::numeric_limits<double>::quiet_NaN();
        }
        // 如果 m 等于 0.0，则直接返回 phi
        if (m == 0.0)
            return (phi);

        // 计算 a = 1.0 - m
        a = 1.0 - m;
        // 如果 a 等于 0.0，处理特殊情况
        if (a == 0.0) {
            // 如果 phi 的绝对值大于等于 π/2，则设置错误并返回无穷大
            if (std::abs(phi) >= (double) M_PI_2) {
                set_error("ellik", SF_ERROR_SINGULAR, NULL);
                return (std::numeric_limits<double>::infinity());
            }
            // 否则根据 DLMF 19.6.8, 4.23.42 返回结果
            return std::asinh(std::tan(phi));
        }

        // 计算 npio2 = floor(phi / M_PI_2)
        npio2 = floor(phi / M_PI_2);
        // 如果 npio2 的绝对值模 2 是奇数，则 npio2 加 1
        if (std::fmod(std::abs(npio2), 2.0) == 1.0)
            npio2 += 1;
        
        // 如果 npio2 不等于 0.0，则计算 K = ellpk(a) 并调整 phi
        if (npio2 != 0.0) {
            K = ellpk(a);
            phi = phi - npio2 * M_PI_2;
        } else
            K = 0.0;

        // 如果 phi 小于 0.0，调整 phi 和 sign
        if (phi < 0.0) {
            phi = -phi;
            sign = -1;
        } else
            sign = 0;

        // 如果 a 大于 1.0，调用 detail::ellik_neg_m(phi, m) 处理
        if (a > 1.0) {
            temp = detail::ellik_neg_m(phi, m);
            goto done; // 跳转至完成标记 done
        }

        // 计算 b = sqrt(a), t = tan(phi)
        b = std::sqrt(a);
        t = std::tan(phi);

        // 如果 tan(phi) 的绝对值大于 10.0，进行特殊处理
        if (std::abs(t) > 10.0) {
            // 变换幅度 e
            e = 1.0 / (b * t);
            // 避免多次递归
            if (std::abs(e) < 10.0) {
                e = std::atan(e);
                // 如果 npio2 等于 0，则重新计算 K
                if (npio2 == 0)
                    K = ellpk(a);
                // 计算结果并跳转至完成标记
                temp = K - ellik(e, m);
                goto done;
            }
        }

        // 初始化循环变量 a, c, d, mod
        a = 1.0;
        c = std::sqrt(m);
        d = 1;
        mod = 0;

        // 循环直到满足精度条件
        while (std::abs(c / a) > detail::MACHEP) {
            temp = b / a;
            phi = phi + atan(t * temp) + mod * M_PI;
            denom = 1.0 - temp * t * t;
            // 如果分母的绝对值大于 10 * detail::MACHEP，更新 t 和 mod
            if (std::abs(denom) > 10 * detail::MACHEP) {
                t = t * (1.0 + temp) / denom;
                mod = (phi + M_PI_2) / M_PI;
            } else {
                t = std::tan(phi);
                mod = static_cast<int>(std::floor((phi - std::atan(t)) / M_PI));
            }
            // 更新 a, b, c, d
            c = (a - b) / 2.0;
            temp = std::sqrt(a * b);
            a = (a + b) / 2.0;
            b = temp;
            d += d;
        }

        // 计算最终结果
        temp = (std::atan(t) + mod * M_PI) / (d * a);

    done:
        // 如果 sign 小于 0，取相反数
        if (sign < 0)
            temp = -temp;
        // 加上 npio2 * K
        temp += npio2 * K;
        // 返回结果
        return (temp);
    }
} // 结束 cephes 命名空间
} // 结束 special 命名空间
```