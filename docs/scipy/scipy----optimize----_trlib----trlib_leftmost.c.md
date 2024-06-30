# `D:\src\scipysrc\scipy\scipy\optimize\_trlib\trlib_leftmost.c`

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

/* Function: trlib_leftmost
 * -------------------------
 * Determines the leftmost eigenvalue of a set of irreducible blocks of a matrix.
 *
 * Parameters:
 * - nirblk: Number of irreducible blocks.
 * - irblk: Array of size nirblk+1 defining the boundaries of the irreducible blocks.
 * - diag: Array containing diagonal elements of the matrix blocks.
 * - offdiag: Array containing off-diagonal elements of the matrix blocks.
 * - warm: Flag indicating warm start (1) or cold start (0).
 * - leftmost_minor: Threshold for convergence of the leftmost eigenvalue.
 * - itmax: Maximum number of iterations allowed.
 * - tol_abs: Absolute tolerance for convergence.
 * - verbose: Verbosity level for output.
 * - unicode: Flag indicating whether to use Unicode characters in output.
 * - prefix: Prefix string for output messages.
 * - fout: File pointer for output.
 * - timing: Array for storing timing information.
 * - ileftmost: Pointer to store the index of the leftmost eigenvalue block.
 * - leftmost: Array to store the leftmost eigenvalue of each block.
 *
 * Returns:
 * - 0 if successful, nonzero otherwise.
 */
trlib_int_t trlib_leftmost(
        trlib_int_t nirblk, trlib_int_t *irblk, trlib_flt_t *diag, trlib_flt_t *offdiag,
        trlib_int_t warm, trlib_flt_t leftmost_minor, trlib_int_t itmax, trlib_flt_t tol_abs,
        trlib_int_t verbose, trlib_int_t unicode, char *prefix, FILE *fout,
        trlib_int_t *timing, trlib_int_t *ileftmost, trlib_flt_t *leftmost) {
    trlib_int_t ret = 0, curit = 0;
    if(! warm) {
        trlib_int_t curret = 0;
        trlib_int_t ii = 0;
        ret = 0;
        for(ii = 0; ii < nirblk; ++ii) {
            // Compute the leftmost eigenvalue for each irreducible block
            curret = trlib_leftmost_irreducible(irblk[ii+1]-irblk[ii], diag+irblk[ii], offdiag+irblk[ii], 0, 0.0, itmax,
                tol_abs, verbose, unicode, prefix, fout, timing, leftmost+ii, &curit);
            if (curret == 0) { ret = curret; }
        }
        *ileftmost = 0;
        // Find the index of the block with the smallest leftmost eigenvalue
        for(ii = 1; ii < nirblk; ++ii) {
            if (leftmost[ii] < leftmost[*ileftmost]) { *ileftmost = ii; }
        }
    }
    else {
        // Compute the leftmost eigenvalue for the last irreducible block
        ret = trlib_leftmost_irreducible(irblk[nirblk] - irblk[nirblk-1], diag+irblk[nirblk-1], offdiag+irblk[nirblk-1],
                1, leftmost_minor, itmax, tol_abs, verbose, unicode, prefix, fout, timing, leftmost+nirblk-1, &curit);
        // Update the index of the block with the smallest leftmost eigenvalue if needed
        if (leftmost[nirblk-1] < leftmost[*ileftmost]) { *ileftmost = nirblk-1; }
    }
    return ret;
}

/* Function: trlib_leftmost_irreducible
 * -------------------------------------
 * Computes the leftmost eigenvalue of an irreducible matrix block.
 *
 * Parameters:
 * - n: Size of the matrix block.
 * - diag: Array containing diagonal elements of the matrix block.
 * - offdiag: Array containing off-diagonal elements of the matrix block.
 * - warm: Flag indicating warm start (1) or cold start (0).
 * - leftmost_minor: Threshold for convergence of the leftmost eigenvalue.
 * - itmax: Maximum number of iterations allowed.
 * - tol_abs: Absolute tolerance for convergence.
 * - verbose: Verbosity level for output.
 * - unicode: Flag indicating whether to use Unicode characters in output.
 * - prefix: Prefix string for output messages.
 * - fout: File pointer for output.
 * - timing: Array for storing timing information.
 * - leftmost: Pointer to store the leftmost eigenvalue.
 * - iter_pr: Pointer to store the number of iterations performed.
 *
 * Returns:
 * - 0 if successful, nonzero otherwise.
 */
trlib_int_t trlib_leftmost_irreducible(
        trlib_int_t n, trlib_flt_t *diag, trlib_flt_t *offdiag,
        trlib_int_t warm, trlib_flt_t leftmost_minor, trlib_int_t itmax, trlib_flt_t tol_abs,
        trlib_int_t verbose, trlib_int_t unicode, char *prefix, FILE *fout,
        trlib_int_t *timing, trlib_flt_t *leftmost, trlib_int_t *iter_pr) {
    // Local variables
    #if TRLIB_MEASURE_TIME
        struct timespec verystart, start, end;
        TRLIB_TIC(verystart)
    #endif
    // Function implementation goes here
}
    trlib_int_t jj = 0;                     // local counter variable
    trlib_flt_t low = 0.0;                  // lower bracket variable: low <= leftmost       for desired value
    trlib_flt_t up = 0.0;                   // upper bracket variable:        leftmost <= up for desired value
    trlib_flt_t leftmost_attempt = 0.0;     // trial step for leftmost eigenvalue
    trlib_flt_t dleftmost = 0.0;            // increment
    trlib_flt_t prlp = 0.0;                 // value of Parlett-Reid-Last-Pivot function
    trlib_flt_t obyprlp = 0.0;              // quotient used in Cholesky computation
    trlib_flt_t dprlp = 0.0;                // derivative of Parlett-Reid-Last-Pivot function wrt to leftmost
    trlib_flt_t ddprlp = 0.0;               // second derivative of Parlett-Reid-Last-Pivot function wrt to leftmost
    trlib_int_t n_neg_piv = 0;              // number of negative pivots in factorization
    trlib_flt_t quad_abs = 0.0;             // absolute  coefficient in quadratic model
    trlib_flt_t quad_lin = 0.0;             // linear    coefficient in quadratic model
    trlib_flt_t quad_qua = 0.0;             // quadratic coefficient in quadratic model
    trlib_flt_t zerodum = 0.0;              // dummy return variables from quadratic equation
    trlib_flt_t oabs0 = 0.0, oabs1 = 0.0;   // temporaries in Gershgorin limit computation

    trlib_int_t continue_outer_loop = 0;    // local spaghetti code control variable
    trlib_int_t model_type = 0;
    trlib_int_t ii = 0;
    *leftmost = 0.0;                        // estimation of desired leftmost eigenvalue
    *iter_pr = 0;                           // iteration counter

    // trivial case: one-dimensional. return diagonal value
    if (n == 1) { *leftmost = diag[0]; TRLIB_RETURN(TRLIB_LMR_CONV) }

    /* set bracket interval derived from Gershgorin circles
       Gershgorin:
        eigenvalues are contained in the union of balls centered at
        diag_i with radius sum of absolute values in column i, except diagonal element
       this estimation is rough and could be improved by circle component analysis
              determine if worth doing */

    // Compute initial bracket [low, up] using Gershgorin circles
    oabs0 = fabs(offdiag[0]); oabs1 = fabs(offdiag[n-2]);
    low = fmin( diag[0] - oabs0, diag[n-1] - oabs1 );
    up  = fmax( diag[0] + oabs0, diag[n-1] + oabs1 );
    for(ii = 1; ii < n-1; ++ii ) {
        oabs1 = fabs(offdiag[ii]);
        low = fmin( low, diag[ii] - oabs0 - oabs1 );
        up  = fmax( up,  diag[ii] + oabs0 + oabs1 );
        oabs0 = oabs1;
    }

    /* set leftmost to sensible initialization
       on warmstart, provided leftmost is eigenvalue of principal (n-1) * (n-1) submatrix
          by eigenvalue interlacing theorem desired value <= provided leftmost
       on coldstart, start close lower bound as hopefully this is a good estimation */
    // 如果 warm 为真，则执行以下操作
    if ( warm ) {
        // 将 up 更新为 up 和 leftmost_minor 中较小的值，确保 leftmost 是 Parlett-Reid 值的上界，稍微做些扰动
        up = fmin(up, leftmost_minor);
        // 更新 leftmost，使其比 leftmost_minor 小一点，具体扰动量为 .1*(up-low)
        *leftmost = leftmost_minor - .1*(up-low); //*leftmost = leftmost_minor - TRLIB_EPS_POW_4;
    }
    // 如果 warm 为假，则执行以下操作
    else {
        // 将 leftmost_minor 设置为 0.0，确保 leftmost 从下界 low 开始，并做一个较小的扰动
        leftmost_minor = 0.0;
        // 初始化 leftmost，使其在 lower bound 和 upper bound 之间，扰动量为 .1*(up-low)
        *leftmost = low + .1*(up-low);
    };
    // 执行 Parlett-Reid 迭代，注意这里假设 n > 1
    // 将最大迭代次数 itmax 扩展为 itmax*n
    itmax = itmax*n;
}

trlib_int_t trlib_leftmost_timing_size() {
#if TRLIB_MEASURE_TIME
    return 1;  // 如果定义了 TRLIB_MEASURE_TIME 宏，则返回 1
#endif
    return 0;  // 如果未定义 TRLIB_MEASURE_TIME 宏，则返回 0
}
```