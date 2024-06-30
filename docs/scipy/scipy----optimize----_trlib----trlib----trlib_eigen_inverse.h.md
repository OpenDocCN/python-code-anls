# `D:\src\scipysrc\scipy\scipy\optimize\_trlib\trlib\trlib_eigen_inverse.h`

```
/*
 * MIT License
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

#ifndef TRLIB_EIGEN_INVERSE_H
#define TRLIB_EIGEN_INVERSE_H

// 定义返回值：收敛
#define TRLIB_EIR_CONV          (0)
// 定义返回值：达到最大迭代次数
#define TRLIB_EIR_ITMAX         (-1)
// 定义返回值：失败因子
#define TRLIB_EIR_FAIL_FACTOR   (-2)
// 定义返回值：线性求解失败
#define TRLIB_EIR_FAIL_LINSOLVE (-3)

// 定义起始向量的数量
#define TRLIB_EIR_N_STARTVEC    (5)

// 函数声明：计算特征值逆矩阵
trlib_int_t trlib_eigen_inverse(
        trlib_int_t n, trlib_flt_t *diag, trlib_flt_t *offdiag,
        trlib_flt_t lam_init, trlib_int_t itmax, trlib_flt_t tol_abs,
        trlib_flt_t *ones, trlib_flt_t *diag_fac, trlib_flt_t *offdiag_fac,
        trlib_flt_t *eig, trlib_int_t verbose, trlib_int_t unicode, char *prefix, FILE *fout,
        trlib_int_t *timing, trlib_flt_t *lam_pert, trlib_flt_t *pert, trlib_int_t *iter_inv);

/** 计算在 trlib_eigen_inverse 函数中需要为 timing 参数分配的大小 */
trlib_int_t trlib_eigen_timing_size(void);

#endif
```