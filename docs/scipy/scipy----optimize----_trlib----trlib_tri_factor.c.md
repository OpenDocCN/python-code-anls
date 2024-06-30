# `D:\src\scipysrc\scipy\scipy\optimize\_trlib\trlib_tri_factor.c`

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

/*
 * Function: trlib_tri_factor_min
 *
 * Description:
 * Performs minimum degree nested dissection ordering and computes a factorization of a sparse matrix.
 *
 * Parameters:
 * nirblk - number of blocks in the matrix
 * irblk - array indicating the block boundaries
 * diag - array holding the diagonal elements of the matrix
 * offdiag - array holding the off-diagonal elements of the matrix
 * neglin - array holding additional linear terms
 * radius - optimization parameter
 * itmax - maximum number of iterations
 * tol_rel - relative tolerance
 * tol_newton_tiny - tolerance for Newton iterations
 * pos_def - indicator for positive definiteness
 * equality - indicator for equality constraints
 * warm0 - initial warm start information (block 0)
 * lam0 - initial lambda values (block 0)
 * warm - warm start information (current block)
 * lam - current lambda values (current block)
 * warm_leftmost - warm start information for leftmost block
 * ileftmost - index of the leftmost block
 * leftmost - information related to the leftmost block
 * warm_fac0 - initial warm factorization information (block 0)
 * diag_fac0 - diagonal factorization elements (block 0)
 * offdiag_fac0 - off-diagonal factorization elements (block 0)
 * warm_fac - warm factorization information (current block)
 * diag_fac - diagonal factorization elements (current block)
 * offdiag_fac - off-diagonal factorization elements (current block)
 * sol0 - initial solution vector (block 0)
 * sol - current solution vector (current block)
 * ones - array of ones used in calculations
 * fwork - workspace for calculations
 * refine - indicator for refinement
 * verbose - verbosity level
 * unicode - indicator for Unicode support
 * prefix - prefix for output messages
 * fout - output file stream
 * timing - array holding timing information
 * obj - objective function value
 * iter_newton - number of Newton iterations performed
 * sub_fail - indicator for failure of subroutine
 *
 * Returns:
 * Integer indicating success or failure of the function.
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
    trlib_int_t *timing, trlib_flt_t *obj, trlib_int_t *iter_newton, trlib_int_t *sub_fail) {
    // use notation of Gould paper
    // h = h(lam) denotes solution of (T+lam I) * h = -lin

    trlib_int_t *leftmost_timing = NULL;
    trlib_int_t *eigen_timing = NULL;
    
    // local variables
    #if TRLIB_MEASURE_TIME
        struct timespec verystart, start, end;
        TRLIB_TIC(verystart)
        leftmost_timing = timing + 1 + TRLIB_SIZE_TIMING_LINALG;
        eigen_timing = timing + 1 + TRLIB_SIZE_TIMING_LINALG + trlib_leftmost_timing_size();
    #endif
    
    /* this is based on Theorem 5.8 in Gould paper,
     * the data for the first block has a 0 suffix,
     * the data for the \ell block has a l suffix */
    
    trlib_int_t n0 = irblk[1];         // dimension of first block
    trlib_int_t nl;                    // dimension of block corresponding to leftmost
    trlib_int_t nm0 = irblk[1]-1;      // length of offdiagonal of first block
    trlib_int_t info_fac = 0;                                // 存储因子化信息的变量
    trlib_int_t ret = 0;                                     // 返回代码的变量
    trlib_flt_t lam_pert = 0.0;                           // 左侧最小特征值扰动，作为 lam 的起始值
    trlib_flt_t norm_sol0 = 0.0;                          // h_0(lam) 的范数
    trlib_int_t jj = 0;                                      // 本地迭代计数器
    trlib_flt_t dlam     = 0.0;                           // 牛顿迭代的增量
    trlib_int_t inc = 1;                                     // 向量存储中的增量
    trlib_flt_t *w = fwork;                               // 辅助向量，用于牛顿迭代
    trlib_flt_t *diag_lam = fwork+(irblk[nirblk]);        // 存储 diag + lam 的向量，如果我们自己实现迭代修正，可以保存它
    trlib_flt_t *work = fwork+2*(irblk[nirblk]);          // 迭代修正的工作空间
    trlib_flt_t ferr = 0.0;                               // 来自迭代修正的前向误差界限
    trlib_flt_t berr = 0.0;                               // 来自迭代修正的后向误差界限
    trlib_flt_t pert_low, pert_up;                        // lambda 扰动的下界和上界
    trlib_flt_t dot = 0.0, dot2 = 0.0;                    // 保存点积
    trlib_flt_t invD_norm_w_sq = 0.0;                     // || w ||_{D^-1}^2
    *iter_newton = 0;                                // 牛顿迭代计数器

    // FIXME: 确保各种不同的热启动正常工作

    // 初始化:
    *sub_fail = 0;                                         // 将 sub_fail 设置为 0，作为安全防护

    // 将 sol 设置为 0 作为安全防护
    memset(sol, 0, irblk[nirblk]*sizeof(trlib_flt_t));

    // 首先确保 lam0, h_0 的准确性
    TRLIB_PRINTLN_1("Solving trust region problem, radius %e; starting on first irreducible block", radius)
    // 如果 nirblk 大于 1 并且 *warm0 为真，则说明通过热启动提供了解
    if (nirblk > 1 && *warm0) {
        TRLIB_DNRM2(norm_sol0, &n0, sol0, &inc)
        if (unicode) { TRLIB_PRINTLN_1("Solution provided via warmstart, \u03bb\u2080 = %e, \u2016h\u2080\u2016 = %e", *lam0, norm_sol0) }
        else { TRLIB_PRINTLN_1("Solution provided via warmstart, lam0 = %e, ||h0|| = %e", *lam0, norm_sol0) }
        // 如果 ||h0|| - radius < 0.0，则违反了约束条件，切换到冷启动
        if (norm_sol0-radius < 0.0) {
            if (unicode) { TRLIB_PRINTLN_1("  violates \u2016h\u2080\u2016 - radius \u2265 0, but is %e, switch to coldstart", norm_sol0-radius) }
            else { TRLIB_PRINTLN_1("  violates ||h0|| - radius >= 0, but is %e, switch to coldstart", norm_sol0-radius) }
            *warm0 = 0;                                      // 将 *warm0 设置为 0，切换到冷启动
        }
    }
    *warm0 = 1;                                              // 将 *warm0 设置为 1，表示热启动有效

    // 测试是否足够精确地解决了第一个不可约块的信任区问题，
    // 否则构建与最左边特征向量的线性组合，以解决信任区约束
    // 检查条件：如果 lam0 与 norm_sol0 的差的绝对值大于等于 TRLIB_EPS_POW_5 乘以 radius
    if ( fabs(radius - norm_sol0) >= TRLIB_EPS_POW_5*radius ) {
        // 进入条件：lam0 为零且非等式情况下，返回 TRLIB_TTR_CONV_INTERIOR
        if(*lam0 == 0.0 && !equality) { ret = TRLIB_TTR_CONV_INTERIOR; }
        else {
            // 如果使用 Unicode 输出字符串 " Found λ₀ with tr residual %e! Bail out with h₀ + α eig"
            if (unicode) { TRLIB_PRINTLN_1(" Found \u03bb\u2080 with tr residual %e! Bail out with h\u2080 + \u03b1 eig", radius - norm_sol0) }
            // 否则输出字符串 " Found lam0 with tr residual %e! Bail out with h0 + alpha eig"
            else { TRLIB_PRINTLN_1(" Found lam0 with tr residual %e! Bail out with h0 + alpha eig", radius - norm_sol0) }
            // 调用求逆特征向量函数，更新 *sub_fail 的值
            *sub_fail = trlib_eigen_inverse(n0, diag, offdiag,
                    *leftmost, 10, TRLIB_EPS_POW_5, ones,
                    diag_fac, offdiag_fac, sol,
                    verbose-2, unicode, " EI", fout, eigen_timing, &ferr, &berr, &jj); // 可安全地覆盖 ferr、berr、jj 的结果。只有 eigenvector 是感兴趣的
            // 如果求逆特征向量失败且不是 -1，输出字符串 "Failure in eigenvector computation: %ld"，并返回 TRLIB_TTR_FAIL_EIG
            if (*sub_fail != 0 && *sub_fail != -1) { TRLIB_PRINTLN_2("Failure in eigenvector computation: %ld", *sub_fail) TRLIB_RETURN(TRLIB_TTR_FAIL_EIG) }
            // 如果求逆特征向量达到 itmax，输出字符串 "In eigenvector computation itmax reached, continue with approximate eigenvector"
            if (*sub_fail == -1) { TRLIB_PRINTLN_2("In eigenvector computation itmax reached, continue with approximate eigenvector") }
            // 计算解作为 h0 和特征向量的线性组合
            // ||h0 + t*eig||^2 = ||h_0||^2 + t * <h0, eig> + t^2 = radius^2
            TRLIB_DDOT(dot, &n0, sol0, &inc, sol, &inc); // dot = <h0, eig>
            // 解二次方程 norm_sol0^2 - radius^2 = 2.0 * dot * t + t^2，求 t 的值
            trlib_quadratic_zero( norm_sol0*norm_sol0 - radius*radius, 2.0*dot, TRLIB_EPS_POW_75, verbose - 3, unicode, prefix, fout, &ferr, &berr);
            // 选择对应较小目标的解
            // 作为 t 的二次函数，没有偏移
            // q(t) = 1/2 * leftmost * t^2 + (leftmost * <eig, h0> + <eig, lin>) * t
            TRLIB_DDOT(dot2, &n0, sol, &inc, neglin, &inc); // dot2 = - <eig, lin>
            // 如果 .5*(*leftmost)*ferr*ferr + ((*leftmost)*dot - dot2)*ferr 小于等于 .5*(*leftmost)*berr*berr + ((*leftmost)*dot - dot2)*berr
            if( .5*(*leftmost)*ferr*ferr + ((*leftmost)*dot - dot2)*ferr <= .5*(*leftmost)*berr*berr + ((*leftmost)*dot - dot2)*berr) {
                // sol0 = sol0 + ferr * sol
                TRLIB_DAXPY(&n0, &ferr, sol, &inc, sol0, &inc)
            }
            else {
                // sol0 = sol0 + berr * sol
                TRLIB_DAXPY(&n0, &berr, sol, &inc, sol0, &inc)
            }
            // 设置返回值为 TRLIB_TTR_HARD_INIT_LAM
            ret = TRLIB_TTR_HARD_INIT_LAM;
        }
    }


    /* 处于已知准确 lam0、h_0 存在的情况下，到达第一个不可约块
     * 调用定理 5.8：
     * (i)  如果 lam0 >= -leftmost，则 lam0, h_0 对解决问题有效
     * (ii) 如果 lam0 < -leftmost，则需要构造一个到 lam = -leftmost 的解 */

    // 快速退出：只有一个不可约块
    if (nirblk == 1) {
        // 将 lam 设置为 lam0，设置 *warm 为 1
        *lam = *lam0; *warm = 1;
        // sol <-- sol0
        TRLIB_DCOPY(&n0, sol0, &inc, sol, &inc) // sol <== sol0
        // 计算目标函数值。首先将 2*gradient 存储在 w 中，然后计算 obj = .5*(sol, w)
        TRLIB_DCOPY(&n0, neglin, &inc, w, &inc) ferr = -2.0; TRLIB_DSCAL(&n0, &ferr, w, &inc) ferr = 1.0; // w <-- -2 neglin
        TRLIB_DLAGTM("N", &n0, &inc, &ferr, offdiag, diag, offdiag, sol, &n0, &ferr, w, &n0) // w <-- T*sol + w
        TRLIB_DDOT(dot, &n0, sol, &inc, w, &inc) *obj = 0.5*dot; // obj = .5*(sol, w)
        // 返回 ret
        TRLIB_RETURN(ret)
    }
    // 现在我们有准确的 lam，调用定理 5.8
    // 检查是否 lam <= leftmost --> 在这种情况下，第一个块的信息描述了所有内容
    if (unicode) { TRLIB_PRINTLN_1("\n检查 \u03bb\u2080 是否提供全局解，获取不可约块的最左特征值") }
    else { TRLIB_PRINTLN_1("\n检查 lam0 是否提供全局解，获取不可约块的最左特征值") }
    // 如果 *warm_leftmost 为假，则计算最左特征值
    if(!*warm_leftmost) {
        *sub_fail = trlib_leftmost(nirblk, irblk, diag, offdiag, 0, leftmost[nirblk-1], 1000, TRLIB_EPS_POW_75, verbose-2, unicode, " LM ", fout, leftmost_timing, ileftmost, leftmost);
        *warm_leftmost = 1;
    }
    // 输出最左特征值的值及其所在块的信息
    TRLIB_PRINTLN_1("    leftmost = %e (block %ld)", leftmost[*ileftmost], *ileftmost)
    // 如果 lam0 大于等于 -leftmost[*ileftmost]，则根据情况输出相应信息
    if(*lam0 >= -leftmost[*ileftmost]) {
        if (unicode) { TRLIB_PRINTLN_1("  \u03bb\u2080 \u2265 -leftmost \u21d2 \u03bb = \u03bb\u2080, 退出：h\u2080(\u03bb\u2080)") }
        else { TRLIB_PRINTLN_1("  lam0 >= -leftmost => lam = lam0, 退出：h0(lam0)") }
        // 将 lam 和相关状态设为 lam0 和 1
        *lam = *lam0; *warm = 1;
        // 将 sol0 复制到 sol
        TRLIB_DCOPY(&n0, sol0, &inc, sol, &inc) // sol <== sol0
        // 计算目标函数。首先将 2*gradient 存储在 w 中，然后计算 obj = .5*(sol, w)
        TRLIB_DCOPY(&n0, neglin, &inc, w, &inc) ferr = -2.0; TRLIB_DSCAL(&n0, &ferr, w, &inc) ferr = 1.0; // w <-- -2 neglin
        TRLIB_DLAGTM("N", &n0, &inc, &ferr, offdiag, diag, offdiag, sol, &n0, &ferr, w, &n0) // w <-- T*sol + w
        TRLIB_DDOT(dot, &n0, sol, &inc, w, &inc) *obj = 0.5*dot; // obj = .5*(sol, w)
        // 返回结果
        TRLIB_RETURN(ret)
    }
}

trlib_int_t trlib_tri_factor_regularized_umin(
    trlib_int_t n, trlib_flt_t *diag, trlib_flt_t *offdiag,
    trlib_flt_t *neglin, trlib_flt_t lam,
    trlib_flt_t *sol,
    trlib_flt_t *ones, trlib_flt_t *fwork,
    trlib_int_t refine,
    trlib_int_t verbose, trlib_int_t unicode, char *prefix, FILE *fout,
    trlib_int_t *timing, trlib_flt_t *norm_sol, trlib_int_t *sub_fail) {

    // local variables
    #if TRLIB_MEASURE_TIME
        struct timespec verystart, start, end;
        TRLIB_TIC(verystart)
    #endif

    trlib_flt_t *diag_lam = fwork;        // vector that holds diag + lam, could be saved if we would implement iterative refinement ourselves
    trlib_flt_t *diag_fac = fwork+n;      // vector that holds diagonal of factor of diag + lam
    trlib_flt_t *offdiag_fac = fwork+2*n; // vector that holds offdiagonal of factor of diag + lam
    trlib_flt_t *work = fwork+3*n;        // workspace for iterative refinement
    trlib_flt_t ferr = 0.0;               // forward  error bound from iterative refinement
    trlib_flt_t berr = 0.0;               // backward error bound from iterative refinement
    trlib_int_t inc = 1;                  // vector increment
    trlib_int_t info_fac = 0;             // LAPACK return code
    trlib_int_t nm = n-1;

    // factorize T + lam0 I
    TRLIB_DCOPY(&n, diag, &inc, diag_lam, &inc) // diag_lam <-- diag
    TRLIB_DAXPY(&n, &lam, ones, &inc, diag_lam, &inc) // diag_lam <-- lam0 + diag_lam
    TRLIB_DCOPY(&n, diag_lam, &inc, diag_fac, &inc) // diag_fac <-- diag_lam
    TRLIB_DCOPY(&nm, offdiag, &inc, offdiag_fac, &inc) // offdiag_fac <-- offdiag
    TRLIB_DPTTRF(&n, diag_fac, offdiag_fac, &info_fac) // compute factorization
    if (info_fac != 0) { TRLIB_RETURN(TRLIB_TTR_FAIL_FACTOR); } // factorization failed, switch to coldastart

    TRLIB_DCOPY(&n, neglin, &inc, sol, &inc) // sol <-- neglin
    TRLIB_DPTTRS(&n, &inc, diag_fac, offdiag_fac, sol, &n, &info_fac) // sol <-- (T+lam I)^-1 sol
    if (info_fac != 0) { TRLIB_PRINTLN_2("Failure on backsolve for h") TRLIB_RETURN(TRLIB_TTR_FAIL_LINSOLVE) }
    if (refine) { TRLIB_DPTRFS(&n, &inc, diag_lam, offdiag, diag_fac, offdiag_fac, neglin, &n, sol, &n, &ferr, &berr, work, &info_fac) }
    if (info_fac != 0) { TRLIB_PRINTLN_2("Failure on iterative refinement for h") TRLIB_RETURN(TRLIB_TTR_FAIL_LINSOLVE) }

    TRLIB_DNRM2(*norm_sol, &n, sol, &inc)
    TRLIB_RETURN(TRLIB_TTR_CONV_INTERIOR);
}

trlib_int_t trlib_tri_factor_get_regularization(
    trlib_int_t n, trlib_flt_t *diag, trlib_flt_t *offdiag,
    trlib_flt_t *neglin, trlib_flt_t *lam,
    trlib_flt_t sigma, trlib_flt_t sigma_l, trlib_flt_t sigma_u,
    trlib_flt_t *sol,
    trlib_flt_t *ones, trlib_flt_t *fwork,
    trlib_int_t refine,
    trlib_int_t verbose, trlib_int_t unicode, char *prefix, FILE *fout,
    trlib_int_t *timing, trlib_flt_t *norm_sol, trlib_int_t *sub_fail) {

    // local variables


注释：
    //if TRLIB_MEASURE_TIME
        struct timespec verystart, start, end;
        TRLIB_TIC(verystart)
    //endif

    // diag_lam：存储 diag + lam 的向量，如果我们实现迭代细化，则可以保存
    trlib_flt_t *diag_lam = fwork;        // vector that holds diag + lam, could be saved if we would implement iterative refinement ourselves

    // diag_fac：存储 diag + lam 的因子的对角线向量
    trlib_flt_t *diag_fac = fwork+n;      // vector that holds diagonal of factor of diag + lam

    // offdiag_fac：存储 diag + lam 的因子的非对角线向量
    trlib_flt_t *offdiag_fac = fwork+2*n; // vector that holds offdiagonal of factor of diag + lam

    // work：用于迭代细化的工作空间
    trlib_flt_t *work = fwork+3*n;        // workspace for iterative refinement

    // aux：辅助向量，ds/n 的结果
    trlib_flt_t *aux  = fwork+5*n;        // auxiliary vector ds/n

    // ferr：通过迭代细化获得的前向误差界限
    trlib_flt_t ferr = 0.0;               // forward  error bound from iterative refinement

    // berr：通过迭代细化获得的后向误差界限
    trlib_flt_t berr = 0.0;               // backward error bound from iterative refinement

    // inc：向量的增量
    trlib_int_t inc = 1;                  // vector increment

    // info_fac：LAPACK 返回码
    trlib_int_t info_fac;                 // LAPACK return code

    // nm：n-1，用作局部循环变量
    trlib_int_t nm = n-1;

    // lambda_l：lambda 的下界
    trlib_flt_t lambda_l = 0.0;           // lower bound on lambda

    // lambda_u：lambda 的上界
    trlib_flt_t lambda_u = 1e20;          // upper bound on lambda

    // jj：局部循环变量
    trlib_int_t jj = 0;                   // local loop variable

    // dlam：lambda 的步长
    trlib_flt_t dlam = 0.0;               // step in lambda

    // dn：范数的导数
    trlib_flt_t dn = 0.0;                 // derivative of norm

    // 获取适合的 lambda 以确保存在因子化
    info_fac = 1;
    while(info_fac != 0 && jj < 500) {
        // 对 T + lam0 I 进行因子化
        TRLIB_DCOPY(&n, diag, &inc, diag_lam, &inc) // diag_lam <-- diag
        TRLIB_DAXPY(&n, lam, ones, &inc, diag_lam, &inc) // diag_lam <-- lam0 + diag_lam
        TRLIB_DCOPY(&n, diag_lam, &inc, diag_fac, &inc) // diag_fac <-- diag_lam
        TRLIB_DCOPY(&nm, offdiag, &inc, offdiag_fac, &inc) // offdiag_fac <-- offdiag
        TRLIB_DPTTRF(&n, diag_fac, offdiag_fac, &info_fac) // compute factorization
        if(info_fac == 0) { break; }
        if(*lam > lambda_u) { break; }
        lambda_l = *lam;
        //if (*lam == 0.0) { *lam = 1.0; }
        *lam = 2.0 * (*lam); jj++;
    }
    if (info_fac != 0) { TRLIB_RETURN(TRLIB_TTR_FAIL_FACTOR); } // factorization failed
    TRLIB_PRINTLN_1("Initial Regularization Factor that allows Cholesky: %e", *lam);

    // sol：解向量初始化为 neglin
    TRLIB_DCOPY(&n, neglin, &inc, sol, &inc) // sol <-- neglin

    // 解线性方程 (T+lam I)^-1 sol
    TRLIB_DPTTRS(&n, &inc, diag_fac, offdiag_fac, sol, &n, &info_fac) // sol <-- (T+lam I)^-1 sol
    if (info_fac != 0) { TRLIB_PRINTLN_2("Failure on backsolve for h") TRLIB_RETURN(TRLIB_TTR_FAIL_LINSOLVE) }

    // 如果需要，进行迭代细化
    if (refine) { TRLIB_DPTRFS(&n, &inc, diag_lam, offdiag, diag_fac, offdiag_fac, neglin, &n, sol, &n, &ferr, &berr, work, &info_fac) }
    if (info_fac != 0) { TRLIB_PRINTLN_2("Failure on iterative refinement for h") TRLIB_RETURN(TRLIB_TTR_FAIL_LINSOLVE) }

    // 计算解向量的范数
    TRLIB_DNRM2(*norm_sol, &n, sol, &inc)

    jj = 0;
    // 打印调试信息
    TRLIB_PRINTLN_2("%ld\t Reg %e\t Reg/Norm %e\t lb %e ub %e", jj, *lam, *lam/(*norm_sol), sigma_l, sigma_u);

    // 检查是否可接受
    // 检查 lam 是否在 [norm_sol * sigma_l, norm_sol * sigma_u] 范围内
    if( *norm_sol * sigma_l <= *lam && *lam <= *norm_sol * sigma_u ) {
        // 打印带有正则化因子 lam 和 lam/norm_sol 的信息
        TRLIB_PRINTLN_1("Exit with Regularization Factor %e and Reg/Norm %e", *lam, *lam/(*norm_sol))
        // 返回内部收敛结果代码 TRLIB_TTR_CONV_INTERIOR
        TRLIB_RETURN(TRLIB_TTR_CONV_INTERIOR);
    }
    // 如果 lam 不在指定范围内，代码块结束
}
}

trlib_int_t trlib_tri_factor_regularize_posdef(
    trlib_int_t n, trlib_flt_t *diag, trlib_flt_t *offdiag,
    trlib_flt_t tol_away, trlib_flt_t security_step, trlib_flt_t *regdiag) {
    /* 
    修改对角线元素以便进行因式分解
    Cholesky因式分解的对角线递推关系为：
    diag_fac[0] = diag[0]
    diag_fac[i+1] = diag[i+1] - offdiag[i]*offdiag[i] / diag_fac[i]
    我们需要确保 diag_fac > 0 
    */
    
    trlib_flt_t diag_fac = 0.0;  // 初始化 diag_fac
    trlib_int_t pivot = 0;  // 初始化 pivot
    
    regdiag[0] = 0.0;  // 初值化 regdiag[0]
    if (diag[0] <= tol_away) { regdiag[0] = security_step*tol_away; }  // 如果 diag[0] 小于等于 tol_away，则设置 regdiag[0] 的值
    diag_fac = diag[0] + regdiag[0];  // 更新 diag_fac 的值

    for(pivot = 0; pivot < n-1; ++pivot) {
        regdiag[pivot+1] = 0.0;  // 初值化 regdiag[pivot+1]
        if ( diag[pivot+1] - offdiag[pivot]*offdiag[pivot]/diag_fac <= tol_away * diag_fac ) {
            // 如果 diag[pivot+1] - offdiag[pivot]*offdiag[pivot]/diag_fac 小于等于 tol_away * diag_fac，则设置 regdiag[pivot+1] 的值
            regdiag[pivot+1] = security_step * fabs(offdiag[pivot]*offdiag[pivot]/diag_fac - diag[pivot+1]);
        }
        diag_fac = diag[pivot+1] + regdiag[pivot+1] - offdiag[pivot]*offdiag[pivot]/diag_fac;  // 更新 diag_fac 的值
    }

    return 0;  // 返回值
}


trlib_int_t trlib_tri_timing_size() {
#if TRLIB_MEASURE_TIME
    return 1+TRLIB_SIZE_TIMING_LINALG+trlib_leftmost_timing_size()+trlib_eigen_timing_size();
#endif
    return 0;  // 返回值
}

trlib_int_t trlib_tri_factor_memory_size(trlib_int_t n) {
    return 6*n;  // 返回值为 6*n
}
```