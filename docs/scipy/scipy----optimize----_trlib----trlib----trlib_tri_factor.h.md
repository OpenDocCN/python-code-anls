# `D:\src\scipysrc\scipy\scipy\optimize\_trlib\trlib\trlib_tri_factor.h`

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

#ifndef TRLIB_TRI_FACTOR_H
#define TRLIB_TRI_FACTOR_H

/*
 * 定义了各种转换和失败条件的常量
 */
#define TRLIB_TTR_CONV_BOUND    (0)          // 边界条件转换
#define TRLIB_TTR_CONV_INTERIOR (1)          // 内部条件转换
#define TRLIB_TTR_HARD          (2)          // 硬条件
#define TRLIB_TTR_NEWTON_BREAK  (3)          // 牛顿法中断条件
#define TRLIB_TTR_HARD_INIT_LAM (4)          // 初始化硬条件
#define TRLIB_TTR_ITMAX         (-1)         // 最大迭代次数达到
#define TRLIB_TTR_FAIL_FACTOR   (-2)         // 因子分解失败
#define TRLIB_TTR_FAIL_LINSOLVE (-3)         // 线性求解失败
#define TRLIB_TTR_FAIL_EIG      (-4)         // 特征值求解失败
#define TRLIB_TTR_FAIL_LM       (-5)         // LM 方法失败
#define TRLIB_TTR_FAIL_HARD     (-10)        // 硬条件失败

/*
 * 定义了三角因子分解函数 trlib_tri_factor_min 的参数及其类型
 */
trlib_int_t trlib_tri_factor_min(
    trlib_int_t nirblk, trlib_int_t *irblk, trlib_flt_t *diag, trlib_flt_t *offdiag,
    trlib_flt_t *neglin, trlib_flt_t radius, 
    trlib_int_t itmax, trlib_flt_t tol_rel, trlib_flt_t tol_newton_tiny,
    trlib_int_t pos_def, trlib_int_t equality,
    trlib_int_t *warm0, trlib_flt_t *lam0, trlib_int_t *warm, trlib_flt_t *lam,
    trlib_int_t *warm_leftmost, trlib_int_t *ileftmost, trlib_flt_t *leftmost,
    trlib_int_t *warm_fac0, trlib_flt_t *diag_fac0, trlib_flt_t *offdiag_fac0,
    trlib_int_t *warm_fac, trlib_flt_t *diag_fac, trlib_flt_t *offdiag_fac,
    trlib_flt_t *sol0, trlib_flt_t *sol, trlib_flt_t *ones, trlib_flt_t *fwork,
    trlib_int_t refine,
    trlib_int_t verbose, trlib_int_t unicode, char *prefix, FILE *fout,
    trlib_int_t *timing, trlib_flt_t *obj, trlib_int_t *iter_newton, trlib_int_t *sub_fail);

/*
 * 定义了正则化 Umin 函数 trlib_tri_factor_regularized_umin 的参数及其类型
 */
trlib_int_t trlib_tri_factor_regularized_umin(
    trlib_int_t n, trlib_flt_t *diag, trlib_flt_t *offdiag,
    trlib_flt_t *neglin, trlib_flt_t lam,
    trlib_flt_t *sol,
    trlib_flt_t *ones, trlib_flt_t *fwork,
    trlib_int_t refine,
    trlib_int_t verbose, trlib_int_t unicode, char *prefix, FILE *fout,
    trlib_int_t *timing, trlib_flt_t *norm_sol, trlib_int_t *sub_fail);

/*
 * 定义了获取正则化因子函数 trlib_tri_factor_get_regularization 的参数及其类型
 */
trlib_int_t trlib_tri_factor_get_regularization(
    trlib_int_t n, trlib_flt_t *diag, trlib_flt_t *offdiag,
    ...
    trlib_flt_t *neglin,  // 指向 trlib_flt_t 类型的指针 neglin
    trlib_flt_t *lam,     // 指向 trlib_flt_t 类型的指针 lam
    trlib_flt_t sigma,    // trlib_flt_t 类型的变量 sigma
    trlib_flt_t sigma_l,  // trlib_flt_t 类型的变量 sigma_l
    trlib_flt_t sigma_u,  // trlib_flt_t 类型的变量 sigma_u
    trlib_flt_t *sol,     // 指向 trlib_flt_t 类型的指针 sol
    trlib_flt_t *ones,    // 指向 trlib_flt_t 类型的指针 ones
    trlib_flt_t *fwork,   // 指向 trlib_flt_t 类型的指针 fwork
    trlib_int_t refine,   // trlib_int_t 类型的变量 refine
    trlib_int_t verbose,  // trlib_int_t 类型的变量 verbose
    trlib_int_t unicode,  // trlib_int_t 类型的变量 unicode
    char *prefix,         // 指向 char 类型的指针 prefix
    FILE *fout,           // 指向 FILE 结构的指针 fout
    trlib_int_t *timing,  // 指向 trlib_int_t 类型的指针 timing
    trlib_flt_t *norm_sol, // 指向 trlib_flt_t 类型的指针 norm_sol
    trlib_int_t *sub_fail  // 指向 trlib_int_t 类型的指针 sub_fail
);
/**
 *  Compute diagonal regularization to make tridiagonal matrix positive definite
 *
 *  :param n: dimension, ensure :math:`n > 0`
 *  :type n: trlib_int_t, input
 *  :param diag: pointer to array holding diagonal of :math:`T`, length :c:data:`n`
 *  :type diag: trlib_flt_t, input
 *  :param offdiag: pointer to array holding offdiagonal of :math:`T`, length :c:data:`n-1`
 *  :type offdiag: trlib_flt_t, input
 *  :param tol_away: tolerance that diagonal entries in factorization should be away from zero, relative to previous entry. Good default :math:`10^{-12}`.
 *  :type tol_away: trlib_flt_t, input
 *  :param security_step: factor greater ``1.0`` that defines a margin to get away from zero in the step taken. Good default ``2.0``.
 *  :type security_step: trlib_flt_t, input
 *  :param regdiag: pointer to array holding regularization term, length :c:data:`n`
 *  :type regdiag: trlib_flt_t, input/output
 *
 *  :returns: ``0``
 *  :rtype: trlib_int_t
 */
trlib_int_t trlib_tri_factor_regularize_posdef(
    trlib_int_t n, trlib_flt_t *diag, trlib_flt_t *offdiag,
    trlib_flt_t tol_away, trlib_flt_t security_step, trlib_flt_t *regdiag);

/**
 *  Gives information on memory that has to be allocated for :c:func:`trlib_tri_factor_min`
 *  
 *  :param n: dimension, ensure :math:`n > 0`
 *  :type n: trlib_int_t, input
 *  :param fwork_size: size of floating point workspace fwork that has to be allocated for :c:func:`trlib_tri_factor_min`
 *  :type fwork_size: trlib_flt_t, output
 *  
 *  :returns: ``0``
 *  :rtype: trlib_int_t
 */
trlib_int_t trlib_tri_factor_memory_size(trlib_int_t n);

/**
 *  Size that has to be allocated for :c:data:`timing` in :c:func:`trlib_tri_factor_min`
 */
trlib_int_t trlib_tri_timing_size(void);
```