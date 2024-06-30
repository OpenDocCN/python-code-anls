# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\dgsequ.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/
/*! @file dgsequ.c
 * \brief Computes row and column scalings
 *
 * <pre>
 * -- SuperLU routine (version 2.0) --
 * Univ. of California Berkeley, Xerox Palo Alto Research Center,
 * and Lawrence Berkeley National Lab.
 * November 15, 1997
 *
 * Modified from LAPACK routine DGEEQU
 * </pre>
 */
/*
 * File name:    dgsequ.c
 * History:     Modified from LAPACK routine DGEEQU
 */
#include <math.h>
#include "slu_ddefs.h"


注释：
/*! \brief
 *
 * <pre>
 * Purpose   
 *   =======   
 *
 *   DGSEQU computes row and column scalings intended to equilibrate an   
 *   M-by-N sparse matrix A and reduce its condition number. R returns the row
 *   scale factors and C the column scale factors, chosen to try to make   
 *   the largest element in each row and column of the matrix B with   
 *   elements B(i,j)=R(i)*A(i,j)*C(j) have absolute value 1.   
 *
 *   R(i) and C(j) are restricted to be between SMLNUM = smallest safe   
 *   number and BIGNUM = largest safe number.  Use of these scaling   
 *   factors is not guaranteed to reduce the condition number of A but   
 *   works well in practice.   
 *
 *   See supermatrix.h for the definition of 'SuperMatrix' structure.
 *
 *   Arguments   
 *   =========   
 *
 *   A       (input) SuperMatrix*
 *           The matrix of dimension (A->nrow, A->ncol) whose equilibration
 *           factors are to be computed. The type of A can be:
 *           Stype = SLU_NC; Dtype = SLU_D; Mtype = SLU_GE.
 *        
 *   R       (output) double*, size A->nrow
 *           If INFO = 0 or INFO > M, R contains the row scale factors   
 *           for A.
 *        
 *   C       (output) double*, size A->ncol
 *           If INFO = 0,  C contains the column scale factors for A.
 *        
 *   ROWCND  (output) double*
 *           If INFO = 0 or INFO > M, ROWCND contains the ratio of the   
 *           smallest R(i) to the largest R(i).  If ROWCND >= 0.1 and   
 *           AMAX is neither too large nor too small, it is not worth   
 *           scaling by R.
 *        
 *   COLCND  (output) double*
 *           If INFO = 0, COLCND contains the ratio of the smallest   
 *           C(i) to the largest C(i).  If COLCND >= 0.1, it is not   
 *           worth scaling by C.
 *        
 *   AMAX    (output) double*
 *           Absolute value of largest matrix element.  If AMAX is very   
 *           close to overflow or very close to underflow, the matrix   
 *           should be scaled.
 *        
 *   INFO    (output) int*
 *           = 0:  successful exit   
 *           < 0:  if INFO = -i, the i-th argument had an illegal value   
 *           > 0:  if INFO = i,  and i is   
 *                 <= A->nrow:  the i-th row of A is exactly zero   
 *                 >  A->ncol:  the (i-M)-th column of A is exactly zero   
 *
 *   ===================================================================== 
 * </pre>
 */
void
dgsequ(SuperMatrix *A, double *r, double *c, double *rowcnd,
       double *colcnd, double *amax, int *info)
{

    /* Local variables */
    NCformat *Astore;  // 非压缩列存储格式的超级矩阵 A 的存储结构
    double   *Aval;    // 矩阵 A 的非零元素数组
    int_t i, j;        // 循环变量
    int   irow;        // 行号
    double rcmin, rcmax;  // 行缩放因子的最小值和最大值
    double bignum, smlnum;  // 安全数的最大值和最小值
    extern double dmach(char *);  // 外部函数声明，返回浮点数参数的机器精度

    /* Test the input parameters. */
    *info = 0;  // 初始化 info 为 0
    if ( A->nrow < 0 || A->ncol < 0 ||
         A->Stype != SLU_NC || A->Dtype != SLU_D || A->Mtype != SLU_GE )
        *info = -1;  // 如果输入参数有误，将 info 设为 -1
    if (*info != 0) {
        int ii = -(*info);  // ii 是 info 的绝对值
    input_error("dgsequ", &ii);
    return;
    }

    /* Quick return if possible */
    if ( A->nrow == 0 || A->ncol == 0 ) {
    *rowcnd = 1.;
    *colcnd = 1.;
    *amax = 0.;
    return;
    }


    Astore = A->Store;
    Aval = Astore->nzval;


    /* Get machine constants. */
    smlnum = dmach("S");  /* slamch_("S"); */
    bignum = 1. / smlnum;


    /* Compute row scale factors. */
    for (i = 0; i < A->nrow; ++i) r[i] = 0.;


    /* Find the maximum element in each row. */
    for (j = 0; j < A->ncol; ++j)
    for (i = Astore->colptr[j]; i < Astore->colptr[j+1]; ++i) {
        irow = Astore->rowind[i];
        r[irow] = SUPERLU_MAX( r[irow], fabs(Aval[i]) );
    }


    /* Find the maximum and minimum scale factors. */
    rcmin = bignum;
    rcmax = 0.;
    for (i = 0; i < A->nrow; ++i) {
    rcmax = SUPERLU_MAX(rcmax, r[i]);
    rcmin = SUPERLU_MIN(rcmin, r[i]);
    }
    *amax = rcmax;


    if (rcmin == 0.) {
    /* Find the first zero scale factor and return an error code. */
    for (i = 0; i < A->nrow; ++i)
        if (r[i] == 0.) {
        *info = i + 1;
        return;
        }
    } else {
    /* Invert the scale factors. */
    for (i = 0; i < A->nrow; ++i)
        r[i] = 1. / SUPERLU_MIN( SUPERLU_MAX( r[i], smlnum ), bignum );
    /* Compute ROWCND = min(R(I)) / max(R(I)) */
    *rowcnd = SUPERLU_MAX( rcmin, smlnum ) / SUPERLU_MIN( rcmax, bignum );
    }


    /* Compute column scale factors */
    for (j = 0; j < A->ncol; ++j) c[j] = 0.;


    /* Find the maximum element in each column, assuming the row
       scalings computed above. */
    for (j = 0; j < A->ncol; ++j)
    for (i = Astore->colptr[j]; i < Astore->colptr[j+1]; ++i) {
        irow = Astore->rowind[i];
        c[j] = SUPERLU_MAX( c[j], fabs(Aval[i]) * r[irow] );
    }


    /* Find the maximum and minimum scale factors. */
    rcmin = bignum;
    rcmax = 0.;
    for (j = 0; j < A->ncol; ++j) {
    rcmax = SUPERLU_MAX(rcmax, c[j]);
    rcmin = SUPERLU_MIN(rcmin, c[j]);
    }


    if (rcmin == 0.) {
    /* Find the first zero scale factor and return an error code. */
    for (j = 0; j < A->ncol; ++j)
        if ( c[j] == 0. ) {
        *info = A->nrow + j + 1;
        return;
        }
    } else {
    /* Invert the scale factors. */
    for (j = 0; j < A->ncol; ++j)
        c[j] = 1. / SUPERLU_MIN( SUPERLU_MAX( c[j], smlnum ), bignum);
    /* Compute COLCND = min(C(J)) / max(C(J)) */
    *colcnd = SUPERLU_MAX( rcmin, smlnum ) / SUPERLU_MIN( rcmax, bignum );
    }


    return;
} /* dgsequ */


注释：


// 结束了名为 dgsequ 的函数或代码块的定义。这种注释通常用于标记代码块的结束，提高代码的可读性和维护性。
```