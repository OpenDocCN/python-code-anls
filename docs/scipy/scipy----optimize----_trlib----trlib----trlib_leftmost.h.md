# `D:\src\scipysrc\scipy\scipy\optimize\_trlib\trlib\trlib_leftmost.h`

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

#ifndef TRLIB_LEFTMOST_H
#define TRLIB_LEFTMOST_H

// 定义三个常量
#define TRLIB_LMR_CONV          (0)    // 收敛条件
#define TRLIB_LMR_ITMAX         (-1)   // 最大迭代次数
#define TRLIB_LMR_NEWTON_BREAK  (3)    // 牛顿法中断条件

// 声明函数 trlib_leftmost，计算左最特征值
trlib_int_t trlib_leftmost(
        trlib_int_t nirblk, trlib_int_t *irblk, trlib_flt_t *diag, trlib_flt_t *offdiag,
        trlib_int_t warm, trlib_flt_t leftmost_minor, trlib_int_t itmax, trlib_flt_t tol_abs,
        trlib_int_t verbose, trlib_int_t unicode, char *prefix, FILE *fout,
        trlib_int_t *timing, trlib_int_t *ileftmost, trlib_flt_t *leftmost);

// 声明函数 trlib_leftmost_irreducible，计算不可约矩阵的左最特征值
trlib_int_t trlib_leftmost_irreducible(
        trlib_int_t n, trlib_flt_t *diag, trlib_flt_t *offdiag,
        trlib_int_t warm, trlib_flt_t leftmost_minor, trlib_int_t itmax, trlib_flt_t tol_abs,
        trlib_int_t verbose, trlib_int_t unicode, char *prefix, FILE *fout,
        trlib_int_t *timing, trlib_flt_t *leftmost, trlib_int_t *iter_pr);

/** size that has to be allocated for :c:data:`timing` in :c:func:`trlib_leftmost_irreducible` and :c:func:`trlib_leftmost`
 * 计算在 trlib_leftmost_irreducible 和 trlib_leftmost 函数中需要为 timing 数组分配的大小
 */
trlib_int_t trlib_leftmost_timing_size(void);

#endif
```