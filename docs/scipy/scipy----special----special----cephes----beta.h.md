# `D:\src\scipysrc\scipy\scipy\special\special\cephes\beta.h`

```
/* Translated into C++ by SciPy developers in 2024.
 * Original header with Copyright information appears below.
 */

/*                                                     beta.c
 *
 *     Beta function
 *
 *
 *
 * SYNOPSIS:
 *
 * double a, b, y, beta();
 *
 * y = beta( a, b );
 *
 *
 *
 * DESCRIPTION:
 *
 *                   -     -
 *                  | (a) | (b)
 * beta( a, b )  =  -----------.
 *                     -
 *                    | (a+b)
 *
 * For large arguments the logarithm of the function is
 * evaluated using lgam(), then exponentiated.
 *
 *
 *
 * ACCURACY:
 *
 *                      Relative error:
 * arithmetic   domain     # trials      peak         rms
 *    IEEE       0,30       30000       8.1e-14     1.1e-14
 *
 * ERROR MESSAGES:
 *
 *   message         condition          value returned
 * beta overflow    log(beta) > MAXLOG       0.0
 *                  a or b <0 integer        0.0
 *
 */

/*
 * Cephes Math Library Release 2.0:  April, 1987
 * Copyright 1984, 1987 by Stephen L. Moshier
 * Direct inquiries to 30 Frost Street, Cambridge, MA 02140
 */
#pragma once

#include "../config.h"
#include "const.h"
#include "gamma.h"

namespace special {
namespace cephes {

    SPECFUN_HOST_DEVICE double beta(double, double);
    SPECFUN_HOST_DEVICE double lbeta(double, double);

    namespace detail {
        constexpr double beta_ASYMP_FACTOR = 1e6;

        /*
         * Asymptotic expansion for  ln(|B(a, b)|) for a > ASYMP_FACTOR*max(|b|, 1).
         */
        SPECFUN_HOST_DEVICE inline double lbeta_asymp(double a, double b, int *sgn) {
            double r = lgam_sgn(b, sgn);  // Calculate sign and logarithm of gamma function of b
            r -= b * std::log(a);  // Subtract b times natural logarithm of a

            r += b * (1 - b) / (2 * a);  // Add b times (1 - b) divided by (2 * a)
            r += b * (1 - b) * (1 - 2 * b) / (12 * a * a);  // Add b times (1 - b) times (1 - 2 * b) divided by (12 * a squared)
            r += -b * b * (1 - b) * (1 - b) / (12 * a * a * a);  // Subtract b squared times (1 - b) times (1 - b) divided by (12 * a cubed)

            return r;  // Return the result
        }

        /*
         * Special case for a negative integer argument
         */

        SPECFUN_HOST_DEVICE inline double beta_negint(int a, double b) {
            int sgn;
            if (b == static_cast<int>(b) && 1 - a - b > 0) {  // Check if b is an integer and 1 - a - b is positive
                sgn = (static_cast<int>(b) % 2 == 0) ? 1 : -1;  // Determine sign based on whether b is even or odd
                return sgn * special::cephes::beta(1 - a - b, b);  // Return sign times beta(1 - a - b, b)
            } else {
                set_error("lbeta", SF_ERROR_OVERFLOW, NULL);  // Set error for overflow condition
                return std::numeric_limits<double>::infinity();  // Return infinity as error value
            }
        }

        SPECFUN_HOST_DEVICE inline double lbeta_negint(int a, double b) {
            double r;
            if (b == static_cast<int>(b) && 1 - a - b > 0) {  // Check if b is an integer and 1 - a - b is positive
                r = special::cephes::lbeta(1 - a - b, b);  // Compute lbeta(1 - a - b, b)
                return r;  // Return the result
            } else {
                set_error("lbeta", SF_ERROR_OVERFLOW, NULL);  // Set error for overflow condition
                return std::numeric_limits<double>::infinity();  // Return infinity as error value
            }
        }
    } // namespace detail
} // namespace cephes
} // namespace special
    // 定义一个内联函数 beta，计算 beta 函数的值
    SPECFUN_HOST_DEVICE inline double beta(double a, double b) {
        double y;
        int sign = 1;

        // 如果 a 小于等于 0.0
        if (a <= 0.0) {
            // 如果 a 是整数
            if (a == std::floor(a)) {
                // 如果 a 是整数且 a 是 int 类型
                if (a == static_cast<int>(a)) {
                    // 调用 beta_negint 函数计算特定整数参数的 beta 函数值
                    return detail::beta_negint(static_cast<int>(a), b);
                } else {
                    // 否则跳转到溢出处理
                    goto overflow;
                }
            }
        }

        // 如果 b 小于等于 0.0
        if (b <= 0.0) {
            // 如果 b 是整数
            if (b == std::floor(b)) {
                // 如果 b 是整数且 b 是 int 类型
                if (b == static_cast<int>(b)) {
                    // 调用 beta_negint 函数计算特定整数参数的 beta 函数值
                    return detail::beta_negint(static_cast<int>(b), a);
                } else {
                    // 否则跳转到溢出处理
                    goto overflow;
                }
            }
        }

        // 如果 |a| 小于 |b|，交换 a 和 b 的值
        if (std::abs(a) < std::abs(b)) {
            y = a;
            a = b;
            b = y;
        }

        // 如果 a 的绝对值大于 detail::beta_ASYMP_FACTOR 乘以 b 的绝对值，并且 a 大于 detail::beta_ASYMP_FACTOR
        if (std::abs(a) > detail::beta_ASYMP_FACTOR * std::abs(b) && a > detail::beta_ASYMP_FACTOR) {
            /* 避免 lgam(a + b) - lgam(a) 中的精度损失 */
            // 调用 lbeta_asymp 函数计算 a 和 b 的 beta 函数的渐近值
            y = detail::lbeta_asymp(a, b, &sign);
            // 返回 sign 乘以 e 的 y 次方
            return sign * std::exp(y);
        }

        // 计算 a + b
        y = a + b;
        // 如果 y 的绝对值大于 detail::MAXGAM，或者 a、b 的绝对值大于 detail::MAXGAM
        if (std::abs(y) > detail::MAXGAM || std::abs(a) > detail::MAXGAM || std::abs(b) > detail::MAXGAM) {
            int sgngam;
            // 计算 lgam_sgn(y) 并记录其符号
            y = detail::lgam_sgn(y, &sgngam);
            // 保持符号的一致性
            sign *= sgngam; /* 保持符号 */
            // 计算 lgam_sgn(b) - lgam_sgn(y) 并记录其符号
            y = detail::lgam_sgn(b, &sgngam) - y;
            // 保持符号的一致性
            sign *= sgngam;
            // 计算 lgam_sgn(a) + y 并记录其符号
            y = detail::lgam_sgn(a, &sgngam) + y;
            // 保持符号的一致性
            sign *= sgngam;
            // 如果 y 大于 detail::MAXLOG，跳转到溢出处理
            if (y > detail::MAXLOG) {
                goto overflow;
            }
            // 返回 sign 乘以 e 的 y 次方
            return (sign * std::exp(y));
        }

        // 计算 Gamma(y)，Gamma(a)，Gamma(b)
        y = Gamma(y);
        a = Gamma(a);
        b = Gamma(b);
        // 如果 y 等于 0.0，跳转到溢出处理
        if (y == 0.0)
            goto overflow;

        // 如果 |a| - |y| 的绝对值大于 |b| - |y| 的绝对值
        if (std::abs(std::abs(a) - std::abs(y)) > std::abs(std::abs(b) - std::abs(y))) {
            // 计算 b/y，并乘以 a
            y = b / y;
            y *= a;
        } else {
            // 计算 a/y，并乘以 b
            y = a / y;
            y *= b;
        }

        // 返回 y
        return (y);

    // 溢出处理
    overflow:
        // 设置错误信息
        set_error("beta", SF_ERROR_OVERFLOW, NULL);
        // 返回 sign 乘以 double 类型的无穷大
        return (sign * std::numeric_limits<double>::infinity());
    }
    // 定义一个内联函数 lbeta，计算对数 Beta 函数的值，a 和 b 是函数的参数
    SPECFUN_HOST_DEVICE inline double lbeta(double a, double b) {
        double y;  // 声明变量 y，用于存储计算结果
        int sign;   // 声明变量 sign，用于存储符号信息，初始化为 1

        sign = 1;   // 将 sign 设置为 1

        // 如果 a 小于等于 0
        if (a <= 0.0) {
            // 如果 a 是整数
            if (a == std::floor(a)) {
                // 如果 a 是整数类型的值
                if (a == static_cast<int>(a)) {
                    // 调用 detail 命名空间下的 lbeta_negint 函数，计算结果并返回
                    return detail::lbeta_negint(static_cast<int>(a), b);
                } else {
                    // 否则跳转到 over 标签处
                    goto over;
                }
            }
        }

        // 如果 b 小于等于 0
        if (b <= 0.0) {
            // 如果 b 是整数
            if (b == std::floor(b)) {
                // 如果 b 是整数类型的值
                if (b == static_cast<int>(b)) {
                    // 调用 detail 命名空间下的 lbeta_negint 函数，计算结果并返回
                    return detail::lbeta_negint(static_cast<int>(b), a);
                } else {
                    // 否则跳转到 over 标签处
                    goto over;
                }
            }
        }

        // 如果 |a| < |b|，交换 a 和 b 的值
        if (std::abs(a) < std::abs(b)) {
            y = a;
            a = b;
            b = y;
        }

        // 如果 |a| > beta_ASYMP_FACTOR * |b| 并且 a > beta_ASYMP_FACTOR
        if (std::abs(a) > detail::beta_ASYMP_FACTOR * std::abs(b) && a > detail::beta_ASYMP_FACTOR) {
            // 使用 detail 命名空间下的 lbeta_asymp 函数计算 y 和 sign 的值
            y = detail::lbeta_asymp(a, b, &sign);
            // 返回计算结果 y
            return y;
        }

        // 计算 a + b，并检查是否超出预定义的最大值 MAXGAM
        y = a + b;
        if (std::abs(y) > detail::MAXGAM || std::abs(a) > detail::MAXGAM || std::abs(b) > detail::MAXGAM) {
            int sgngam;
            // 使用 detail 命名空间下的 lgam_sgn 函数计算 y 的符号信息，并更新 sign
            y = detail::lgam_sgn(y, &sgngam);
            sign *= sgngam; // 更新符号信息
            // 使用 detail 命名空间下的 lgam_sgn 函数计算 b 的符号信息，并计算 y 的值
            y = detail::lgam_sgn(b, &sgngam) - y;
            sign *= sgngam; // 更新符号信息
            // 使用 detail 命名空间下的 lgam_sgn 函数计算 a 的符号信息，并计算 y 的值
            y = detail::lgam_sgn(a, &sgngam) + y;
            sign *= sgngam; // 更新符号信息
            // 返回计算结果 y
            return (y);
        }

        // 计算 Gamma 函数的值
        y = Gamma(y);
        a = Gamma(a);
        b = Gamma(b);
        // 如果 y 的值为 0
        if (y == 0.0) {
        over:
            // 设置错误信息为 "lbeta"，错误类型为 SF_ERROR_OVERFLOW
            set_error("lbeta", SF_ERROR_OVERFLOW, NULL);
            // 返回符号乘以正无穷大的结果
            return (sign * std::numeric_limits<double>::infinity());
        }

        // 如果 |a| - |y| 的绝对值大于 |b| - |y| 的绝对值
        if (std::abs(std::abs(a) - std::abs(y)) > std::abs(std::abs(b) - std::abs(y))) {
            // 计算 y = b / y * a
            y = b / y;
            y *= a;
        } else {
            // 计算 y = a / y * b
            y = a / y;
            y *= b;
        }

        // 如果 y 的值小于 0，将 y 取相反数
        if (y < 0) {
            y = -y;
        }

        // 返回 y 的自然对数值
        return (std::log(y));
    }
} // 结束 cephes 命名空间定义
} // 结束 special 命名空间定义
```