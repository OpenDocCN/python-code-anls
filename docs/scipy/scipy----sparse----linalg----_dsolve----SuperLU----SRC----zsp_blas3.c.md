# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\zsp_blas3.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file zsp_blas3.c
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

#include "slu_zdefs.h"

// 定义函数 sp_zgemm，实现稀疏矩阵乘法，利用了部分稠密矩阵乘法的操作
int
sp_zgemm(char *transa, char *transb, int m, int n, int k, 
         doublecomplex alpha, SuperMatrix *A, doublecomplex *b, int ldb, 
         doublecomplex beta, doublecomplex *c, int ldc)
{
    // 设置增量为 1，用于矩阵乘法中向量的增量步长
    int    incx = 1, incy = 1;
    // 循环遍历列向量
    for (int j = 0; j < n; ++j) {
        // 调用 sp_zgemv 函数，执行稀疏矩阵-向量乘法，将结果存入矩阵 c 的列向量中
        sp_zgemv(transa, alpha, A, &b[ldb*j], incx, beta, &c[ldc*j], incy);
    }
    // 返回操作成功
    return 0;    
}
```