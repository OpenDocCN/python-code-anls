# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\csp_blas3.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file csp_blas3.c
 * \brief Sparse BLAS3, using some dense BLAS3 operations
 *
 * <pre>
 * -- SuperLU routine (version 2.0) --
 * Univ. of California Berkeley, Xerox Palo Alto Research Center,
 * and Lawrence Berkeley National Lab.
 * November 15, 1997
 * </pre>
 */
/*
 * File name:        sp_blas3.c
 * Purpose:        Sparse BLAS3, using some dense BLAS3 operations.
 */

#include "slu_cdefs.h"

// 定义函数 sp_cgemm，用于执行稀疏矩阵乘法运算，结合部分稠密矩阵乘法操作
int
sp_cgemm(char *transa, char *transb, int m, int n, int k, 
         singlecomplex alpha, SuperMatrix *A, singlecomplex *b, int ldb, 
         singlecomplex beta, singlecomplex *c, int ldc)
{
    // 设置增量变量
    int    incx = 1, incy = 1;
    int    j;

    // 循环遍历列数 n
    for (j = 0; j < n; ++j) {
        // 调用 sp_cgemv 函数执行稀疏矩阵向量乘法运算
        sp_cgemv(transa, alpha, A, &b[ldb*j], incx, beta, &c[ldc*j], incy);
    }
    // 返回操作成功
    return 0;    
}
```