# `D:\src\scipysrc\scipy\scipy\special\special\cephes\chbevl.h`

```
/*
 * chbevl.c
 *
 * Evaluate Chebyshev series
 *
 * SYNOPSIS:
 *
 * int N;
 * double x, y, coef[N], chebevl();
 *
 * y = chbevl( x, coef, N );
 *
 * DESCRIPTION:
 *
 * Evaluates the series
 *
 *        N-1
 *         - '
 *  y  =   >   coef[i] T (x/2)
 *         -            i
 *        i=0
 *
 * of Chebyshev polynomials Ti at argument x/2.
 *
 * Coefficients are stored in reverse order, i.e. the zero
 * order term is last in the array.  Note N is the number of
 * coefficients, not the order.
 *
 * If coefficients are for the interval a to b, x must
 * have been transformed to x -> 2(2x - b - a)/(b-a) before
 * entering the routine.  This maps x from (a, b) to (-1, 1),
 * over which the Chebyshev polynomials are defined.
 *
 * If the coefficients are for the inverted interval, in
 * which (a, b) is mapped to (1/b, 1/a), the transformation
 * required is x -> 2(2ab/x - b - a)/(b-a).  If b is infinity,
 * this becomes x -> 4a/x - 1.
 *
 * SPEED:
 *
 * Taking advantage of the recurrence properties of the
 * Chebyshev polynomials, the routine requires one more
 * addition per loop than evaluating a nested polynomial of
 * the same degree.
 */

/* chbevl.c */

/*
 * Cephes Math Library Release 2.0:  April, 1987
 * Copyright 1985, 1987 by Stephen L. Moshier
 * Direct inquiries to 30 Frost Street, Cambridge, MA 02140
 */

// 声明该文件只包含一次，避免重复定义
#pragma once

// 包含配置文件，这里假设位于上级目录下的 config.h
#include "../config.h"

// 声明特殊数学函数库的命名空间 special::cephes
namespace special {
namespace cephes {

    // 函数定义：计算 Chebyshev 级数的值
    // 参数 x: 计算的参数
    // 参数 array[]: 存储 Chebyshev 系数的数组
    // 参数 n: 系数的数量
    SPECFUN_HOST_DEVICE double chbevl(double x, const double array[], int n) {
        // 定义局部变量
        double b0, b1, b2;
        const double *p;
        int i;

        // 指针指向数组的起始位置
        p = array;
        // 初始化 b0 为数组第一个元素，b1 为 0
        b0 = *p++;
        b1 = 0.0;
        // 设置循环计数器 i 为 n-1
        i = n - 1;

        // 开始循环计算 Chebyshev 级数
        do {
            // 保存当前 b1 到 b2
            b2 = b1;
            // 更新 b1 和 b0
            b1 = b0;
            b0 = x * b1 - b2 + *p++;
        } while (--i); // 循环 n-1 次

        // 返回计算结果
        return (0.5 * (b0 - b2));
    }

} // namespace cephes
} // namespace special
```