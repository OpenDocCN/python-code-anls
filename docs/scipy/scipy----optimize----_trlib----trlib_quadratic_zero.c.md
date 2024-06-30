# `D:\src\scipysrc\scipy\scipy\optimize\_trlib\trlib_quadratic_zero.c`

```
/* MIT License
 *
 * Copyright (c) 2016--2017 Felix Lenders
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 */

#include "trlib_private.h"
#include "trlib.h"

// 计算二次方程的根
trlib_int_t trlib_quadratic_zero(trlib_flt_t c_abs, trlib_flt_t c_lin, trlib_flt_t tol,
        trlib_int_t verbose, trlib_int_t unicode, char *prefix, FILE *fout,
        trlib_flt_t *t1, trlib_flt_t *t2) {
    trlib_int_t n  = 0;   // 根的数量
    trlib_flt_t q = 0.0;   // 判别式
    trlib_flt_t dq = 0.0;  // 判别式导数
    trlib_flt_t lin_sq = c_lin * c_lin;  // 线性系数的平方
    *t1 = 0.0;    // 第一个根
    *t2 = 0.0;    // 第二个根

    if (fabs(c_abs) > tol * lin_sq) {
        // 非退化二次方程的情况
        // 计算判别式
        q = lin_sq - 4.0 * c_abs;
        if (fabs(q) <= (TRLIB_EPS * c_lin) * (TRLIB_EPS * c_lin)) {
            // 两个不同的零点，但判别式很小 --> 数值上的双重零点
            // 初始化在标准公式给出的根上，判别式为零，让牛顿迭代法做剩余工作
            n = 2;
            *t1 = -0.5 * c_lin; *t2 = *t1;
        }
        else if (q < 0.0) {
            // 判别式为负，复数根
            n = 2;
            *t1 = 0.0; *t2 = 0.0;
            return n;
        }
        else {
            // 判别式足够大，有两个不同的实根
            n = 2;
            // 从加号开始以避免取消误差
            *t1 = -0.5 * (c_lin + copysign(sqrt(q), c_lin));
            *t2 = c_abs / (*t1);
            // 确保 t1 是较小的根
            if (*t2 < *t1) { q = *t2; *t2 = *t1; *t1 = q; }
        }
    }
    else {
        // 退化情况，有两个重复根
        n = 2;
        if (c_lin < 0.0) { *t1 = 0.0; *t2 = -c_lin; }
        else { *t1 = -c_lin; *t2 = 0.0; }
    }

    // 牛顿校正
    q = (*t1 + c_lin) * (*t1) + c_abs; dq = 2.0 * (*t1) + c_lin;
    if (dq != 0.0) { *t1 = *t1 - q / dq; }
    q = (*t2 + c_lin) * (*t2) + c_abs; dq = 2.0 * (*t2) + c_lin;
    if (dq != 0.0) { *t2 = *t2 - q / dq; }

    // 返回根的数量
    return n;
}
```