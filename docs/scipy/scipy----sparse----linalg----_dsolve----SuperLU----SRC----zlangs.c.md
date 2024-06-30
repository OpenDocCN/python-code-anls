# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\zlangs.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file zlangs.c
 * \brief Returns the value of the one norm
 *
 * <pre>
 * -- SuperLU routine (version 2.0) --
 * Univ. of California Berkeley, Xerox Palo Alto Research Center,
 * and Lawrence Berkeley National Lab.
 * November 15, 1997
 *
 * Modified from lapack routine ZLANGE 
 * </pre>
 */
/*
 * File name:    zlangs.c
 * History:     Modified from lapack routine ZLANGE
 */
#include <math.h>
#include "slu_zdefs.h"

/*! \brief
 *
 * <pre>
 * Purpose   
 *   =======   
 *
 *   ZLANGS returns the value of the one norm, or the Frobenius norm, or 
 *   the infinity norm, or the element of largest absolute value of a 
 *   real matrix A.   
 *
 *   Description   
 *   ===========   
 *
 *   ZLANGE returns the value   
 *
 *      ZLANGE = ( max(abs(A(i,j))), NORM = 'M' or 'm'   
 *               (   
 *               ( norm1(A),         NORM = '1', 'O' or 'o'   
 *               (   
 *               ( normI(A),         NORM = 'I' or 'i'   
 *               (   
 *               ( normF(A),         NORM = 'F', 'f', 'E' or 'e'   
 *
 *   where  norm1  denotes the  one norm of a matrix (maximum column sum), 
 *   normI  denotes the  infinity norm  of a matrix  (maximum row sum) and 
 *   normF  denotes the  Frobenius norm of a matrix (square root of sum of 
 *   squares).  Note that  max(abs(A(i,j)))  is not a  matrix norm.   
 *
 *   Arguments   
 *   =========   
 *
 *   NORM    (input) CHARACTER*1   
 *           Specifies the value to be returned in ZLANGE as described above.   
 *   A       (input) SuperMatrix*
 *           The M by N sparse matrix A. 
 *
 *  =====================================================================
 * </pre>
 */

double zlangs(char *norm, SuperMatrix *A)
{
    /* Local variables */
    NCformat *Astore;
    doublecomplex   *Aval;
    int      i, j, irow;
    double   value = 0., sum;
    double   *rwork;

    // 获取 A 的存储格式
    Astore = A->Store;
    // 获取 A 的非零元素数组
    Aval   = Astore->nzval;
    
    // 如果 A 的行数或列数为零，则直接返回 0
    if ( SUPERLU_MIN(A->nrow, A->ncol) == 0) {
    } else if (strncmp(norm, "M", 1)==0) {
        // 计算 max(abs(A(i,j)))
        for (j = 0; j < A->ncol; ++j)
            for (i = Astore->colptr[j]; i < Astore->colptr[j+1]; i++)
                value = SUPERLU_MAX( value, z_abs( &Aval[i]) );
    
    } else if (strncmp(norm, "O", 1)==0 || *(unsigned char *)norm == '1') {
        // 计算 norm1(A)
        for (j = 0; j < A->ncol; ++j) {
            sum = 0.;
            for (i = Astore->colptr[j]; i < Astore->colptr[j+1]; i++) 
                sum += z_abs( &Aval[i] );
            value = SUPERLU_MAX(value,sum);
        }
    
    } else if (strncmp(norm, "I", 1)==0) {
        // 计算 normI(A)
    # 分配空间以存储大小为 A->nrow 的双精度数组，并检查分配是否成功
    if ( !(rwork = (double *) SUPERLU_MALLOC(A->nrow * sizeof(double))) )
        ABORT("SUPERLU_MALLOC fails for rwork.");

    # 将 rwork 数组中所有元素初始化为 0
    for (i = 0; i < A->nrow; ++i) rwork[i] = 0.;

    # 计算矩阵 A 的列压缩存储中每列的 1-范数，并存储到 rwork 数组中对应的行索引位置
    for (j = 0; j < A->ncol; ++j)
        for (i = Astore->colptr[j]; i < Astore->colptr[j+1]; i++) {
            irow = Astore->rowind[i];
            rwork[irow] += z_abs( &Aval[i] );
        }

    # 找出 rwork 数组中的最大值，即矩阵 A 的无穷范数 norm(A)
    for (i = 0; i < A->nrow; ++i)
        value = SUPERLU_MAX(value, rwork[i]);

    # 释放 rwork 数组占用的内存空间
    SUPERLU_FREE (rwork);

    # 如果指定的范数是 "F" 或 "E"，抛出未实现异常
    } else if (strncmp(norm, "F", 1)==0 || strncmp(norm, "E", 1)==0) {
        ABORT("Not implemented.");
    # 否则，指定的范数非法，抛出异常
    } else
        ABORT("Illegal norm specified.");

    # 返回计算得到的矩阵范数的值
    return (value);
} /* zlangs */


注释：


# 结束对 "zlangs" 的代码块或函数定义，闭合大括号表示代码块的结束
```