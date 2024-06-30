# `D:\src\scipysrc\scipy\scipy\special\special\cephes\polevl.h`

```
/* Translated into C++ by SciPy developers in 2024.
 * Original header with Copyright information appears below.
 */

/*                                                     polevl.c
 *                                                     p1evl.c
 *
 *     Evaluate polynomial
 *
 *
 *
 * SYNOPSIS:
 *
 * int N;
 * double x, y, coef[N+1], polevl[];
 *
 * y = polevl( x, coef, N );
 *
 *
 *
 * DESCRIPTION:
 *
 * Evaluates polynomial of degree N:
 *
 *                     2          N
 * y  =  C  + C x + C x  +...+ C x
 *        0    1     2          N
 *
 * Coefficients are stored in reverse order:
 *
 * coef[0] = C  , ..., coef[N] = C  .
 *            N                   0
 *
 * The function p1evl() assumes that c_N = 1.0 so that coefficent
 * is omitted from the array.  Its calling arguments are
 * otherwise the same as polevl().
 *
 *
 * SPEED:
 *
 * In the interest of speed, there are no checks for out
 * of bounds arithmetic.  This routine is used by most of
 * the functions in the library.  Depending on available
 * equipment features, the user may wish to rewrite the
 * program in microcode or assembly language.
 *
 */

/*
 * Cephes Math Library Release 2.1:  December, 1988
 * Copyright 1984, 1987, 1988 by Stephen L. Moshier
 * Direct inquiries to 30 Frost Street, Cambridge, MA 02140
 */

/* Sources:
 * [1] Holin et. al., "Polynomial and Rational Function Evaluation",
 *     https://www.boost.org/doc/libs/1_61_0/libs/math/doc/html/math_toolkit/roots/rational.html
 */

/* Scipy changes:
 * - 06-23-2016: add code for evaluating rational functions
 */

#pragma once

#include "../config.h"

namespace special {
namespace cephes {
    SPECFUN_HOST_DEVICE inline double polevl(double x, const double coef[], int N) {
        double ans;
        int i;
        const double *p;

        p = coef;  // Point p to the start of coef array
        ans = *p++;  // Initialize ans with coef[0] and move p to next coefficient

        i = N;  // Initialize i to N

        // Loop to evaluate polynomial using Horner's method
        do {
            ans = ans * x + *p++;  // Multiply current ans by x and add next coef element
        } while (--i);  // Decrement i and continue until all coefficients are processed

        return (ans);  // Return evaluated polynomial value
    }

    /*                                                     p1evl() */
    /*                                          N
     * Evaluate polynomial when coefficient of x  is 1.0.
     * That is, C_{N} is assumed to be 1, and that coefficient
     * is not included in the input array coef.
     * coef must have length N and contain the polynomial coefficients
     * stored as
     *     coef[0] = C_{N-1}
     *     coef[1] = C_{N-2}
     *          ...
     *     coef[N-2] = C_1
     *     coef[N-1] = C_0
     * Otherwise same as polevl.
     */

    SPECFUN_HOST_DEVICE inline double p1evl(double x, const double coef[], int N) {
        double ans;
        const double *p;
        int i;

        p = coef;  // Point p to the start of coef array
        ans = x + *p++;  // Initialize ans with x + coef[0] and move p to next coefficient

        i = N - 1;  // Initialize i to N-1

        // Loop to evaluate polynomial using Horner's method
        do
            ans = ans * x + *p++;  // Multiply current ans by x and add next coef element
        while (--i);  // Decrement i and continue until all coefficients are processed

        return (ans);  // Return evaluated polynomial value
    }

    /* Evaluate a rational function. See [1]. */

    /* The function ratevl is only used once in cephes/lanczos.h. */
}
}
    // 定义一个内联函数，用于计算分子和分母多项式的值，以计算有理分式的值
    SPECFUN_HOST_DEVICE inline double ratevl(double x, const double num[], int M, const double denom[], int N) {
        // 声明变量
        int i, dir;
        double y, num_ans, denom_ans;
        // 计算 x 的绝对值
        double absx = std::abs(x);
        // 指向常量指针 p
        const double *p;

        // 根据 absx 的大小选择计算方向和起始指针
        if (absx > 1) {
            // 当 absx 大于 1 时，以 1/x 作为变量计算多项式
            dir = -1;
            p = num + M;
            y = 1 / x;
        } else {
            // 当 absx 小于等于 1 时，以 x 作为变量计算多项式
            dir = 1;
            p = num;
            y = x;
        }

        // 计算分子的多项式值
        num_ans = *p;  // 初始化为 p 指向的值
        p += dir;  // 移动 p 指针
        for (i = 1; i <= M; i++) {
            num_ans = num_ans * y + *p;  // 计算多项式值
            p += dir;  // 移动 p 指针
        }

        // 根据 absx 的大小选择分母的起始指针
        if (absx > 1) {
            p = denom + N;
        } else {
            p = denom;
        }

        // 计算分母的多项式值
        denom_ans = *p;  // 初始化为 p 指向的值
        p += dir;  // 移动 p 指针
        for (i = 1; i <= N; i++) {
            denom_ans = denom_ans * y + *p;  // 计算多项式值
            p += dir;  // 移动 p 指针
        }

        // 根据 absx 的大小选择返回值的计算方式
        if (absx > 1) {
            i = M - N;
            // 返回 x 的幂次方乘以分子的值除以分母的值
            return std::pow(x, i) * num_ans / denom_ans;
        } else {
            // 返回分子的值除以分母的值
            return num_ans / denom_ans;
        }
    }
} // 结束 cephes 命名空间
} // 结束 special 命名空间
```