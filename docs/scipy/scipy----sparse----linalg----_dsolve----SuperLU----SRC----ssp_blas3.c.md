# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\ssp_blas3.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file ssp_blas3.c
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

#include "slu_sdefs.h"

// 函数定义，实现稀疏矩阵乘法，利用一些密集矩阵乘法的操作
int
sp_sgemm(char *transa, char *transb, int m, int n, int k, 
         float alpha, SuperMatrix *A, float *b, int ldb, 
         float beta, float *c, int ldc)
{
    // 定义增量值
    int    incx = 1, incy = 1;
    int    j;

    // 循环遍历列向量
    for (j = 0; j < n; ++j) {
        // 调用稀疏矩阵向量乘法函数 sp_sgemv
        sp_sgemv(transa, alpha, A, &b[ldb*j], incx, beta, &c[ldc*j], incy);
    }
    // 返回操作成功的标志
    return 0;    
}
```