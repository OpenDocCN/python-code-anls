# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\cgsequ.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/
/*! @file cgsequ.c
 * \brief Computes row and column scalings
 *
 * <pre>
 * -- SuperLU routine (version 2.0) --
 * Univ. of California Berkeley, Xerox Palo Alto Research Center,
 * and Lawrence Berkeley National Lab.
 * November 15, 1997
 *
 * Modified from LAPACK routine CGEEQU
 * </pre>
 */
/*
 * File name:    cgsequ.c
 * History:     Modified from LAPACK routine CGEEQU
 */

#include <math.h>  // 包含数学函数库头文件
#include "slu_cdefs.h"  // 包含自定义的 SLU C 语言定义文件
/*! \brief
 *
 * <pre>
 * Purpose   
 *   =======   
 *
 *   CGSEQU computes row and column scalings intended to equilibrate an   
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
 *           Stype = SLU_NC; Dtype = SLU_C; Mtype = SLU_GE.
 *        
 *   R       (output) float*, size A->nrow
 *           If INFO = 0 or INFO > M, R contains the row scale factors   
 *           for A.
 *        
 *   C       (output) float*, size A->ncol
 *           If INFO = 0,  C contains the column scale factors for A.
 *        
 *   ROWCND  (output) float*
 *           If INFO = 0 or INFO > M, ROWCND contains the ratio of the   
 *           smallest R(i) to the largest R(i).  If ROWCND >= 0.1 and   
 *           AMAX is neither too large nor too small, it is not worth   
 *           scaling by R.
 *        
 *   COLCND  (output) float*
 *           If INFO = 0, COLCND contains the ratio of the smallest   
 *           C(i) to the largest C(i).  If COLCND >= 0.1, it is not   
 *           worth scaling by C.
 *        
 *   AMAX    (output) float*
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
cgsequ(SuperMatrix *A, float *r, float *c, float *rowcnd,
       float *colcnd, float *amax, int *info)
{
    /* Local variables */
    NCformat *Astore;
    singlecomplex *Aval;
    int_t i, j;
    int irow;
    float rcmin, rcmax;
    float bignum, smlnum;
    extern float smach(char *);
    
    /* Test the input parameters. */
    *info = 0;
    // 检查输入参数是否合法
    if ( A->nrow < 0 || A->ncol < 0 ||
         A->Stype != SLU_NC || A->Dtype != SLU_C || A->Mtype != SLU_GE )
        *info = -1;
    
    if (*info != 0) {
        int ii = -(*info);
        ```
        返回错误码
        ```
        return;
    }
    ```
    设置安全数值边界
    ```
    smlnum = smach("Safe minimum");
    bignum = 1.0 / smlnum;
    ```
    初始化行和列的缩放因子和条件数
    ```
    rcmin = bignum;
    rcmax = 0.0;
    ```
    遍历每一行
    ```
    for (i = 0; i < A->nrow; ++i) {
        ```
        计算行中的绝对值最大值
        ```
        r[i] = 0.0;
        for (j = A->rowptr[i]; j < A->rowptr[i+1]; ++j) {
            r[i] += c_abs1(&A->nzval[j]);
        }
        ```
        更新最大和最小的行缩放因子
        ```
        if (r[i] > rcmax) rcmax = r[i];
        if (r[i] < rcmin) rcmin = r[i];
    }
    ```
    如果行缩放因子范围适当，计算行条件数
    ```
    *rowcnd = rcmin / rcmax;
    ```
    如果行条件数过小，无需缩放
    ```
    if (*rowcnd >= 0.1) return;
    ```
    遍历每一列
    ```
    for (j = 0; j < A->ncol; ++j) {
        ```
        计算列中的绝对值最大值
        ```
        c[j] = 0.0;
        for (i = A->colbeg[j]; i < A->colend[j]; ++i) {
            c[j] += c_abs1(&A->nzval[i]);
        }
        ```
        更新最大和最小的列缩放因子
        ```
        if (c[j] > rcmax) rcmax = c[j];
        if (c[j] < rcmin) rcmin = c[j];
    }
    ```
    计算列条件数
    ```
    *colcnd = rcmin / rcmax;
    ```
    如果列条件数过小，无需缩放
    ```
    if (*colcnd >= 0.1) return;
    ```
    计算矩阵的最大绝对值元素
    ```
    *amax = 0.0;
    for (i = 0; i < A->nrow; ++i) {
        for (j = A->rowptr[i]; j < A->rowptr[i+1]; ++j) {
            if (c_abs1(&A->nzval[j]) > *amax) {
                *amax = c_abs1(&A->nzval[j]);
            }
        }
    }
    ```
    返回成功退出状态
    ```
    *info = 0;
}
    input_error("cgsequ", &ii);
    return;
    }



    // 调用函数处理输入错误
    input_error("cgsequ", &ii);
    // 函数执行完毕，直接返回
    return;
    }



    /* Quick return if possible */
    if ( A->nrow == 0 || A->ncol == 0 ) {
    *rowcnd = 1.;
    *colcnd = 1.;
    *amax = 0.;
    return;
    }



    // 如果矩阵 A 的行数或列数为 0，则快速返回
    if ( A->nrow == 0 || A->ncol == 0 ) {
    // 将 rowcnd、colcnd 设置为 1，amax 设置为 0
    *rowcnd = 1.;
    *colcnd = 1.;
    *amax = 0.;
    // 函数执行完毕，直接返回
    return;
    }



    Astore = A->Store;
    Aval = Astore->nzval;



    // 获取矩阵 A 的存储结构和非零元素数组
    Astore = A->Store;
    Aval = Astore->nzval;



    /* Get machine constants. */
    smlnum = smach("S");  /* slamch_("S"); */
    bignum = 1. / smlnum;



    // 获取机器常数
    smlnum = smach("S");  // 使用 smach 函数获取小常数 smlnum
    bignum = 1. / smlnum; // 计算大常数 bignum



    /* Compute row scale factors. */
    for (i = 0; i < A->nrow; ++i) r[i] = 0.;



    // 计算行比例因子
    for (i = 0; i < A->nrow; ++i) r[i] = 0.;



    /* Find the maximum element in each row. */
    for (j = 0; j < A->ncol; ++j)
    for (i = Astore->colptr[j]; i < Astore->colptr[j+1]; ++i) {
        irow = Astore->rowind[i];
        r[irow] = SUPERLU_MAX( r[irow], c_abs1(&Aval[i]) );
    }



    // 找到每行中的最大元素
    for (j = 0; j < A->ncol; ++j)
    for (i = Astore->colptr[j]; i < Astore->colptr[j+1]; ++i) {
        irow = Astore->rowind[i];
        // 更新行比例因子 r[irow] 为当前行元素的绝对值与当前 r[irow] 的最大值
        r[irow] = SUPERLU_MAX( r[irow], c_abs1(&Aval[i]) );
    }



    /* Find the maximum and minimum scale factors. */
    rcmin = bignum;
    rcmax = 0.;
    for (i = 0; i < A->nrow; ++i) {
    rcmax = SUPERLU_MAX(rcmax, r[i]);
    rcmin = SUPERLU_MIN(rcmin, r[i]);
    }
    *amax = rcmax;



    // 找到最大和最小的比例因子
    rcmin = bignum;
    rcmax = 0.;
    for (i = 0; i < A->nrow; ++i) {
    // 更新 rcmax 和 rcmin
    rcmax = SUPERLU_MAX(rcmax, r[i]);
    rcmin = SUPERLU_MIN(rcmin, r[i]);
    }
    // 将最大的行比例因子赋给 amax
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



    // 如果最小的行比例因子为 0
    if (rcmin == 0.) {
    /* 找到第一个为零的行比例因子，并返回一个错误代码。 */
    for (i = 0; i < A->nrow; ++i)
        if (r[i] == 0.) {
        *info = i + 1;
        return;
        }
    } else {
    /* 反转比例因子。 */
    for (i = 0; i < A->nrow; ++i)
        r[i] = 1. / SUPERLU_MIN( SUPERLU_MAX( r[i], smlnum ), bignum );
    /* 计算 ROWCND = min(R(I)) / max(R(I)) */
    *rowcnd = SUPERLU_MAX( rcmin, smlnum ) / SUPERLU_MIN( rcmax, bignum );
    }



    /* Compute column scale factors */
    for (j = 0; j < A->ncol; ++j) c[j] = 0.;



    // 计算列比例因子
    for (j = 0; j < A->ncol; ++j) c[j] = 0.;



    /* Find the maximum element in each column, assuming the row
       scalings computed above. */
    for (j = 0; j < A->ncol; ++j)
    for (i = Astore->colptr[j]; i < Astore->colptr[j+1]; ++i) {
        irow = Astore->rowind[i];
        c[j] = SUPERLU_MAX( c[j], c_abs1(&Aval[i]) * r[irow] );
    }



    // 找到每列中的最大元素，假设上面计算的行比例因子已经应用
    for (j = 0; j < A->ncol; ++j)
    for (i = Astore->colptr[j]; i < Astore->colptr[j+1]; ++i) {
        irow = Astore->rowind[i];
        // 更新列比例因子 c[j] 为当前列元素的绝对值乘以 r[irow] 的最大值
        c[j] = SUPERLU_MAX( c[j], c_abs1(&Aval[i]) * r[irow] );
    }



    /* Find the maximum and minimum scale factors. */
    rcmin = bignum;
    rcmax = 0.;
    for (j = 0; j < A->ncol; ++j) {
    rcmax = SUPERLU_MAX(rcmax, c[j]);
    rcmin = SUPERLU_MIN(rcmin, c[j]);
    }



    // 找到最大和最小的比例因子
    rcmin = bignum;
    rcmax = 0.;
    for (j = 0; j < A->ncol; ++j) {
    // 更新 rcmax 和 rcmin
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



    // 如果最小的列比例因子为 0
    if (rcmin == 0.) {
    /* 找到第一个为零的列比例因子，并返回一个错误代码。 */
    for (j = 0; j < A
} /* cgsequ */


注释：


// 这行代码看起来像是一段注释或是代码的结束标记，不过它没有对应的起始标记。
// 在C风格的语言中，"}"通常用于表示代码块的结束，但这行单独出现在代码中没有实际功能。
// 如果这是一种特定的编码风格或者约定，请确认其用途是否与代码结构相关。
// 如果它是错误的，可能会导致编译或运行时出现问题。
```