# `D:\src\scipysrc\scipy\scipy\optimize\_trlib\trlib\trlib_krylov.h`

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

#ifndef TRLIB_KRYLOV_H
#define TRLIB_KRYLOV_H

// 定义清除策略的常量
#define TRLIB_CLR_CONV_BOUND    (0)       // 边界收敛
#define TRLIB_CLR_CONV_INTERIOR (1)       // 内点收敛
#define TRLIB_CLR_APPROX_HARD   (2)       // 硬近似
#define TRLIB_CLR_NEWTON_BREAK  (3)       // 牛顿法中断
#define TRLIB_CLR_HARD_INIT_LAM (4)       // 硬初始化 λ
#define TRLIB_CLR_FAIL_HARD     (5)       // 失败：硬条件
#define TRLIB_CLR_UNBDBEL       (6)       // 无界
#define TRLIB_CLR_UNLIKE_CONV   (7)       // 不同的收敛
#define TRLIB_CLR_CONTINUE      (10)      // 继续
#define TRLIB_CLR_ITMAX         (-1)      // 迭代最大次数
#define TRLIB_CLR_FAIL_FACTOR   (-3)      // 失败：因子
#define TRLIB_CLR_FAIL_LINSOLVE (-4)      // 失败：线性求解
#define TRLIB_CLR_FAIL_NUMERIC  (-5)      // 失败：数值
#define TRLIB_CLR_FAIL_TTR      (-7)      // 失败：TTR
#define TRLIB_CLR_PCINDEF       (-8)      // 不定 PC
#define TRLIB_CLR_UNEXPECT_INT  (-9)      // 意外的整数

// 定义线性求解器类型的常量
#define TRLIB_CLT_CG            (1)       // 共轭梯度法
#define TRLIB_CLT_L             (2)       // L 方法

// 定义线性代数操作的常量
#define TRLIB_CLA_TRIVIAL       (0)       // 平凡操作
#define TRLIB_CLA_INIT          (1)       // 初始化
#define TRLIB_CLA_RETRANSF      (2)       // 重新转换
#define TRLIB_CLA_UPDATE_STATIO (3)       // 更新静态操作
#define TRLIB_CLA_UPDATE_GRAD   (4)       // 更新梯度
#define TRLIB_CLA_UPDATE_DIR    (5)       // 更新方向
#define TRLIB_CLA_NEW_KRYLOV    (6)       // 新的 Krylov 空间
#define TRLIB_CLA_CONV_HARD     (7)       // 硬条件的收敛
#define TRLIB_CLA_OBJVAL        (8)       // 目标值

// 定义线性求解策略的常量
#define TRLIB_CLS_INIT          (1)       // 初始化
#define TRLIB_CLS_HOTSTART      (2)       // 热启动
#define TRLIB_CLS_HOTSTART_G    (3)       // 热启动 G
#define TRLIB_CLS_HOTSTART_P    (4)       // 热启动 P
#define TRLIB_CLS_HOTSTART_R    (5)       // 热启动 R
#define TRLIB_CLS_HOTSTART_T    (6)       // 热启动 T
#define TRLIB_CLS_VEC_INIT      (7)       // 向量初始化
#define TRLIB_CLS_CG_NEW_ITER   (8)       // 共轭梯度法新迭代
#define TRLIB_CLS_CG_UPDATE_S   (9)       // 共轭梯度法更新 S
#define TRLIB_CLS_CG_UPDATE_GV  (10)      // 共轭梯度法更新 GV
#define TRLIB_CLS_CG_UPDATE_P   (11)      // 共轭梯度法更新 P
#define TRLIB_CLS_LANCZOS_SWT   (12)      // Lanczos 开关
#define TRLIB_CLS_L_UPDATE_P    (13)      // L 方法更新 P
#define TRLIB_CLS_L_CMP_CONV    (14)      // L 方法比较收敛
#define TRLIB_CLS_L_CMP_CONV_RT (15)      // L 方法比较收敛率
#define TRLIB_CLS_L_CHK_CONV    (16)      // L 方法检查收敛
#define TRLIB_CLS_L_NEW_ITER    (17)      // L 方法新迭代
#define TRLIB_CLS_CG_IF_IRBLK_P (18)      // 共轭梯度法若 IRBLK P
#define TRLIB_CLS_CG_IF_IRBLK_C (19)      // 共轭梯度法若 IRBLK C
#define TRLIB_CLS_CG_IF_IRBLK_N (20)      // 共轭梯度法若 IRBLK N

// 定义线性代数条件的常量
#define TRLIB_CLC_NO_EXP_INV    (0)       // 没有预期逆
#define TRLIB_CLC_EXP_INV_LOC   (1)       // 本地预期逆
/** Define a constant TRLIB_CLC_EXP_INV_GLO with value 2 */
#define TRLIB_CLC_EXP_INV_GLO   (2)

/** Define constants for different control types in trlib_krylov_min */
#define TRLIB_CLT_CG_INT        (0)
#define TRLIB_CLT_CG_BOUND      (1)
#define TRLIB_CLT_LANCZOS       (2)
#define TRLIB_CLT_HOTSTART      (3)

/** 
 *  Perform Krylov iterative minimization using various parameters and workspace arrays.
 *
 *  :param init: initial value
 *  :param radius: trust region radius
 *  :param equality: equality constraints flag
 *  :param itmax: maximum number of iterations
 *  :param itmax_lanczos: maximum number of iterations for Lanczos method
 *  :param tol_rel_i: relative tolerance for initial values
 *  :param tol_abs_i: absolute tolerance for initial values
 *  :param tol_rel_b: relative tolerance for boundary values
 *  :param tol_abs_b: absolute tolerance for boundary values
 *  :param zero: zero threshold
 *  :param obj_lo: lower bound for objective function
 *  :param ctl_invariant: control type for invariant part
 *  :param convexify: convexification parameter
 *  :param earlyterm: early termination flag
 *  :param g_dot_g: dot product of gradient with itself
 *  :param v_dot_g: dot product of step direction with gradient
 *  :param p_dot_Hp: dot product of conjugate direction with Hessian applied to it
 *  :param iwork: integer workspace array
 *  :param fwork: floating point workspace array
 *  :param refine: refinement parameter
 *  :param verbose: verbosity level
 *  :param unicode: Unicode support flag
 *  :param prefix: string prefix
 *  :param fout: output file stream
 *  :param timing: array to store timing information
 *  :param action: action control array
 *  :param iter: iteration count array
 *  :param ityp: type information array
 *  :param flt1: auxiliary floating point value 1
 *  :param flt2: auxiliary floating point value 2
 *  :param flt3: auxiliary floating point value 3
 *  
 *  :returns: result code
 *  :rtype: trlib_int_t
 */
trlib_int_t trlib_krylov_min(
    trlib_int_t init, trlib_flt_t radius, trlib_int_t equality, trlib_int_t itmax, trlib_int_t itmax_lanczos,
    trlib_flt_t tol_rel_i, trlib_flt_t tol_abs_i,
    trlib_flt_t tol_rel_b, trlib_flt_t tol_abs_b, trlib_flt_t zero, trlib_flt_t obj_lo,
    trlib_int_t ctl_invariant, trlib_int_t convexify, trlib_int_t earlyterm,
    trlib_flt_t g_dot_g, trlib_flt_t v_dot_g, trlib_flt_t p_dot_Hp,
    trlib_int_t *iwork, trlib_flt_t *fwork, trlib_int_t refine,
    trlib_int_t verbose, trlib_int_t unicode, char *prefix, FILE *fout, trlib_int_t *timing,
    trlib_int_t *action, trlib_int_t *iter, trlib_int_t *ityp,
    trlib_flt_t *flt1, trlib_flt_t *flt2, trlib_flt_t *flt3
);

/** 
 *  Prepares floating point workspace for :c:func::`trlib_krylov_min`.
 *
 *  :param itmax: maximum number of iterations
 *  :type itmax: trlib_int_t, input
 *  :param fwork: floating point workspace to be initialized
 *  :type fwork: trlib_flt_t, input/output
 *  
 *  :returns: ``0``
 *  :rtype: trlib_int_t
 */
trlib_int_t trlib_krylov_prepare_memory(trlib_int_t itmax, trlib_flt_t *fwork);

/** 
 *  Determines the memory sizes required for integer and floating point workspaces
 *  for :c:func::`trlib_krylov_min`.
 *
 *  :param itmax: maximum number of iterations
 *  :type itmax: trlib_int_t, input
 *  :param iwork_size: size of integer workspace iwork
 *  :type iwork_size: trlib_int_t, output
 *  :param fwork_size: size of floating point workspace fwork
 *  :type fwork_size: trlib_int_t, output
 *  :param h_pointer: start index of vector h for reverse communication in action TRLIB_CLA_RETRANSF
 *  :type h_pointer: trlib_int_t, output
 *  
 *  :returns: ``0``
 *  :rtype: trlib_int_t
 */
trlib_int_t trlib_krylov_memory_size(trlib_int_t itmax, trlib_int_t *iwork_size, trlib_int_t *fwork_size, trlib_int_t *h_pointer);

/** 
 *  Determines the size of memory required for the timing array in :c:func::`trlib_krylov_min`.
 *
 *  :returns: ``0``
 *  :rtype: trlib_int_t
 */
trlib_int_t trlib_krylov_timing_size(void);

/** 
 *  Provides a pointer to the negative gradient of the tridiagonal problem.
 *
 *  :param itmax: maximum number of iterations
 *  :type itmax: trlib_int_t, input
 *  :param gt_pointer: pointer to negative gradient of tridiagonal subproblem
 *  :type gt_pointer: trlib_int_t, output
 *
 *  :returns: ``0``
 *  :rtype: trlib_int_t
 */
trlib_int_t trlib_krylov_gt(trlib_int_t itmax, trlib_int_t *gt_pointer);
```