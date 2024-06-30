# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\clangs.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file clangs.c
 * \brief Returns the value of the one norm
 *
 * <pre>
 * -- SuperLU routine (version 2.0) --
 * Univ. of California Berkeley, Xerox Palo Alto Research Center,
 * and Lawrence Berkeley National Lab.
 * November 15, 1997
 *
 * Modified from lapack routine CLANGE 
 * </pre>
 */
/*
 * File name:    clangs.c
 * History:     Modified from lapack routine CLANGE
 */
#include <math.h>
#include "slu_cdefs.h"

/*! \brief
 *
 * <pre>
 * Purpose   
 *   =======   
 *
 *   CLANGS returns the value of the one norm, or the Frobenius norm, or 
 *   the infinity norm, or the element of largest absolute value of a 
 *   real matrix A.   
 *
 *   Description   
 *   ===========   
 *
 *   CLANGE returns the value   
 *
 *      CLANGE = ( max(abs(A(i,j))), NORM = 'M' or 'm'   
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
 *           Specifies the value to be returned in CLANGE as described above.   
 *   A       (input) SuperMatrix*
 *           The M by N sparse matrix A. 
 *
 *  =====================================================================
 * </pre>
 */

float clangs(char *norm, SuperMatrix *A)
{
    /* Local variables */
    NCformat *Astore;
    singlecomplex *Aval;
    int i, j, irow;
    float value = 0., sum;
    float *rwork;

    // 获取稀疏矩阵A的存储格式
    Astore = A->Store;
    // 获取矩阵A的非零元素数组
    Aval = Astore->nzval;
    
    // 如果矩阵A的行数或列数为0，则直接返回0
    if (SUPERLU_MIN(A->nrow, A->ncol) == 0) {
    } else if (strncmp(norm, "M", 1) == 0) {
        // 计算矩阵A中元素的最大绝对值
        for (j = 0; j < A->ncol; ++j)
            for (i = Astore->colptr[j]; i < Astore->colptr[j + 1]; i++)
                value = SUPERLU_MAX(value, c_abs(&Aval[i]));
    
    } else if (strncmp(norm, "O", 1) == 0 || *(unsigned char *)norm == '1') {
        // 计算矩阵A的1范数（列和的最大值）
        for (j = 0; j < A->ncol; ++j) {
            sum = 0.;
            for (i = Astore->colptr[j]; i < Astore->colptr[j + 1]; i++) 
                sum += c_abs(&Aval[i]);
            value = SUPERLU_MAX(value, sum);
        }
    
    } else if (strncmp(norm, "I", 1) == 0) {
        // 计算矩阵A的无穷范数（行和的最大值）
    # 如果条件为真，分配 rwork 数组用于存储每行的元素绝对值之和
    if (!(rwork = (float *) SUPERLU_MALLOC(A->nrow * sizeof(float))))
        ABORT("SUPERLU_MALLOC fails for rwork.");
    
    # 初始化 rwork 数组的所有元素为 0
    for (i = 0; i < A->nrow; ++i)
        rwork[i] = 0.;
    
    # 计算矩阵 A 的列的绝对值之和，存储在 rwork 中的对应行
    for (j = 0; j < A->ncol; ++j)
        for (i = Astore->colptr[j]; i < Astore->colptr[j+1]; i++) {
            irow = Astore->rowind[i];
            rwork[irow] += c_abs(&Aval[i]);
        }
    
    # 在 rwork 数组中找到最大值，作为矩阵的范数值
    for (i = 0; i < A->nrow; ++i)
        value = SUPERLU_MAX(value, rwork[i]);
    
    # 释放 rwork 数组占用的内存
    SUPERLU_FREE(rwork);
    
    # 如果 norm 参数指定为 "F" 或 "E"，报错，暂未实现此功能
    } else if (strncmp(norm, "F", 1) == 0 || strncmp(norm, "E", 1) == 0) {
        ABORT("Not implemented.");
    
    # 如果 norm 参数非法，报错
    } else
        ABORT("Illegal norm specified.");

    # 返回计算得到的矩阵范数值
    return (value);
} /* clangs */
```