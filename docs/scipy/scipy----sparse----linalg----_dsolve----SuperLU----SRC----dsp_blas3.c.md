# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\dsp_blas3.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file dsp_blas3.c
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

#include "slu_ddefs.h"

// 定义函数 sp_dgemm，执行稀疏 BLAS3 运算，利用部分密集 BLAS3 操作
int
sp_dgemm(char *transa, char *transb, int m, int n, int k, 
         double alpha, SuperMatrix *A, double *b, int ldb, 
         double beta, double *c, int ldc)
{
    // 设置向量的增量为 1
    int    incx = 1, incy = 1;
    // 循环处理每一列
    int    j;

    for (j = 0; j < n; ++j) {
        // 调用 sp_dgemv 函数，执行矩阵向量乘法
        sp_dgemv(transa, alpha, A, &b[ldb*j], incx, beta, &c[ldc*j], incy);
    }
    // 返回执行成功
    return 0;    
}
```