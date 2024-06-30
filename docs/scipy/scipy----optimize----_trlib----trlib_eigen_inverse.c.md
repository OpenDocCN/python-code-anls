# `D:\src\scipysrc\scipy\scipy\optimize\_trlib\trlib_eigen_inverse.c`

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

#include "trlib_private.h"
#include "trlib.h"

// Function definition for trlib_eigen_inverse
trlib_int_t trlib_eigen_inverse(
        trlib_int_t n, trlib_flt_t *diag, trlib_flt_t *offdiag, 
        trlib_flt_t lam_init, trlib_int_t itmax, trlib_flt_t tol_abs,
        trlib_flt_t *ones, trlib_flt_t *diag_fac, trlib_flt_t *offdiag_fac,
        trlib_flt_t *eig, trlib_int_t verbose, trlib_int_t unicode, char *prefix, FILE *fout,
        trlib_int_t *timing, trlib_flt_t *lam_pert, trlib_flt_t *pert, trlib_int_t *iter_inv) {
    
    // Local variables
    #if TRLIB_MEASURE_TIME
        struct timespec verystart, start, end;
        TRLIB_TIC(verystart)
    #endif
    
    trlib_int_t info_fac = 0;                            // status variable for factorization
    trlib_flt_t invnorm = 0.0;                           // 1/norm of eig before normalization
    trlib_flt_t minuslam = - lam_init;                   // negative of current estimation of eigenvalue
    trlib_int_t inc = 1;                                 // increment value for loops
    trlib_int_t nm = n-1;                                // n-1, used as a loop boundary
    
    trlib_int_t seeds[TRLIB_EIR_N_STARTVEC];             // array to store seeds
    trlib_flt_t residuals[TRLIB_EIR_N_STARTVEC];         // array to store residuals
    
    trlib_int_t jj = 0;                                  // loop index
    trlib_int_t kk = 0;                                  // loop index
    trlib_int_t seedpivot = 0;                           // variable for seed pivot
    
    *iter_inv = 0;                                       // initialize iteration counter
    *pert = 0.0;                                         // initialize perturbation factor
    
    info_fac = 0;                                        // initialize status variable
    invnorm = 0.0;                                       // initialize inverse norm
    minuslam = - lam_init;                               // initialize negative lambda
    
    // obtain factorization of T - lam*I, perturb until possible
    // iter_inv is misused in this loop as flag if we can find a suitable lambda to start with
    *iter_inv = TRLIB_EIR_FAIL_FACTOR;                   // set iter_inv to indicate initial failure
    
    while (*pert <= 1.0/TRLIB_EPS) {
        // 当 *pert 小于等于 1.0/TRLIB_EPS 时循环执行以下操作

        // set diag_fac to diag - lam
        TRLIB_DCOPY(&n, diag, &inc, diag_fac, &inc) // 将 diag 复制到 diag_fac
        TRLIB_DAXPY(&n, &minuslam, ones, &inc, diag_fac, &inc) // diag_fac = diag_fac - lam
        TRLIB_DCOPY(&nm, offdiag, &inc, offdiag_fac, &inc) // 将 offdiag 复制到 offdiag_fac
        TRLIB_DPTTRF(&n, diag_fac, offdiag_fac, &info_fac); // 计算因式分解
        if (info_fac == 0) { *iter_inv = 0; break; }
        // 如果因式分解成功（info_fac == 0），则重置迭代次数并退出循环

        if (*pert == 0.0) { 
            // 如果 *pert 等于 0.0
            *pert = TRLIB_EPS_POW_4 * fmax(1.0, -lam_init);
        }
        else { 
            // 否则
            *pert = 10.0*(*pert);
        }
        minuslam = *pert - lam_init;
        // 更新 minuslam 的值为 *pert - lam_init
    }
    *lam_pert = -minuslam;
    // 更新 lam_pert 的值为 -minuslam

    if ( *iter_inv == TRLIB_EIR_FAIL_FACTOR ) { TRLIB_PRINTLN_2("Failure on factorizing in inverse correction!") TRLIB_RETURN(TRLIB_EIR_FAIL_FACTOR) }
    // 如果 *iter_inv 等于 TRLIB_EIR_FAIL_FACTOR，则打印错误信息并返回错误代码 TRLIB_EIR_FAIL_FACTOR

    // try with TRLIB_EIR_N_STARTVEC different start vectors and hope that it converges for one
    // 尝试使用 TRLIB_EIR_N_STARTVEC 不同的起始向量，并希望至少有一个收敛
    seeds[0] = time(NULL);
    for(jj = 1; jj < TRLIB_EIR_N_STARTVEC; ++jj ) { seeds[jj] = rand(); }
    // 设置随机数种子数组 seeds

    for(jj = 0; jj < TRLIB_EIR_N_STARTVEC; ++jj ) {
        *iter_inv = 0;
        srand((unsigned) seeds[jj]);
        // 根据不同的种子 seeds[jj] 初始化随机数生成器

        for(kk = 0; kk < n; ++kk ) { eig[kk] = ((trlib_flt_t)rand()/(trlib_flt_t)RAND_MAX); }
        // 使用随机数填充 eig 数组

        TRLIB_DNRM2(invnorm, &n, eig, &inc) invnorm = 1.0/invnorm;
        TRLIB_DSCAL(&n, &invnorm, eig, &inc) // 对 eig 进行归一化操作
        // 归一化 eig 向量

        // perform inverse iteration
        // 执行逆迭代
        while (1) {
            *iter_inv += 1;
            // 增加迭代次数计数器

            if ( *iter_inv > itmax ) { break; }
            // 如果迭代次数超过最大允许值 itmax，则退出循环

            // solve (T - lam*I)*eig_new = eig_old
            // 解方程 (T - lam*I)*eig_new = eig_old
            TRLIB_DPTTRS(&n, &inc, diag_fac, offdiag_fac, eig, &n, &info_fac)
            if( info_fac != 0 ) { TRLIB_PRINTLN_2("Failure on solving inverse correction!") TRLIB_RETURN(TRLIB_EIR_FAIL_LINSOLVE) }
            // 如果求解失败（info_fac != 0），则打印错误信息并返回错误代码 TRLIB_EIR_FAIL_LINSOLVE

            // normalize eig
            // 归一化 eig 向量
            TRLIB_DNRM2(invnorm, &n, eig, &inc) invnorm = 1.0/invnorm;
            TRLIB_DSCAL(&n, &invnorm, eig, &inc)

            residuals[jj] = fabs(invnorm - *pert);
            // 计算残差 residuals[jj]

            // check for convergence
            // 检查是否收敛
            if (residuals[jj] <= tol_abs ) { TRLIB_RETURN(TRLIB_EIR_CONV) }
            // 如果残差小于等于给定的容差 tol_abs，则返回收敛代码 TRLIB_EIR_CONV
        }
    }

    // no convergence with any of the starting values.
    // 未在任何起始值上收敛。
    // take the seed with least residual and redo computation
    // 选择残差最小的种子，并重新计算
    for(jj = 0; jj < TRLIB_EIR_N_STARTVEC; ++jj) { if (residuals[jj] < residuals[seedpivot]) { seedpivot = jj; } }

    *iter_inv = 0;
    srand((unsigned) seeds[seedpivot]);
    // 根据选定的种子 seedpivot 初始化随机数生成器

    for(kk = 0; kk < n; ++kk ) { eig[kk] = ((trlib_flt_t)rand()/(trlib_flt_t)RAND_MAX); }
    // 使用随机数填充 eig 数组

    TRLIB_DNRM2(invnorm, &n, eig, &inc) invnorm = 1.0/invnorm;
    TRLIB_DSCAL(&n, &invnorm, eig, &inc) // 对 eig 进行归一化操作
    // 归一化 eig 向量
    // perform inverse iteration
    // 执行逆迭代
    while (1) {
        *iter_inv += 1;  // 递增指针指向的值，用于记录迭代次数

        if ( *iter_inv > itmax ) { break; }  // 如果迭代次数超过最大限制，则跳出循环

        // 解方程 (T - lam*I)*eig_new = eig_old
        TRLIB_DPTTRS(&n, &inc, diag_fac, offdiag_fac, eig, &n, &info_fac)
        if( info_fac != 0 ) { TRLIB_PRINTLN_2("Failure on solving inverse correction!") TRLIB_RETURN(TRLIB_EIR_FAIL_LINSOLVE) }
        // 如果解方程失败，则打印错误信息并返回错误代码

        // 对 eig 进行归一化
        TRLIB_DNRM2(invnorm, &n, eig, &inc) invnorm = 1.0/invnorm;
        TRLIB_DSCAL(&n, &invnorm, eig, &inc)
        // 计算 eig 的范数并归一化，确保 eig 的长度为 1

        residuals[seedpivot] = fabs(invnorm - *pert);
        // 计算残差，即当前的 invnorm 与 *pert 之间的绝对差值

        // 检查是否收敛
        if (residuals[seedpivot] <= tol_abs ) { TRLIB_RETURN(TRLIB_EIR_CONV) }
        // 如果残差小于等于给定的容差 tol_abs，则返回收敛状态
    }
    
    TRLIB_RETURN(TRLIB_EIR_ITMAX)
    // 如果迭代达到最大次数 itmax，则返回达到最大迭代次数的状态码
}

`
}
# 结束当前代码块

trlib_int_t trlib_eigen_timing_size() {
    # 如果宏 TRLIB_MEASURE_TIME 被定义
#if TRLIB_MEASURE_TIME
        # 返回 1 加上常量 TRLIB_SIZE_TIMING_LINALG 的值，作为函数返回值
        return 1 + TRLIB_SIZE_TIMING_LINALG;
#endif
    # 如果宏 TRLIB_MEASURE_TIME 没有定义，返回 0
    return 0;
}
```