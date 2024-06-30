# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\cmyblas2.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file cmyblas2.c
 * \brief Level 2 Blas operations
 * 
 * <pre>
 * -- SuperLU routine (version 2.0) --
 * Univ. of California Berkeley, Xerox Palo Alto Research Center,
 * and Lawrence Berkeley National Lab.
 * November 15, 1997
 * </pre>
 * <pre>
 * Purpose:
 *     Level 2 BLAS operations: solves and matvec, written in C.
 * Note:
 *     This is only used when the system lacks an efficient BLAS library.
 * </pre>
 */
/*
 * File name:        cmyblas2.c
 */
#include "slu_scomplex.h"

/*! \brief Solves a dense UNIT lower triangular system
 * 
 * The unit lower 
 * triangular matrix is stored in a 2D array M(1:nrow,1:ncol). 
 * The solution will be returned in the rhs vector.
 */
void clsolve ( int ldm, int ncol, singlecomplex *M, singlecomplex *rhs )
{
    int k;
    singlecomplex x0, x1, x2, x3, temp;
    singlecomplex *M0;
    singlecomplex *Mki0, *Mki1, *Mki2, *Mki3;
    register int firstcol = 0;

    M0 = &M[0];

    // 循环处理每四列，直到列数少于四列
    while ( firstcol < ncol - 3 ) { /* Do 4 columns */
        // 指向当前 M 的起始位置
        Mki0 = M0 + 1;
        Mki1 = Mki0 + ldm + 1;
        Mki2 = Mki1 + ldm + 1;
        Mki3 = Mki2 + ldm + 1;

        // 处理第一列
        x0 = rhs[firstcol];
        cc_mult(&temp, &x0, Mki0); Mki0++;
        c_sub(&x1, &rhs[firstcol+1], &temp);
        cc_mult(&temp, &x0, Mki0); Mki0++;
        c_sub(&x2, &rhs[firstcol+2], &temp);
        cc_mult(&temp, &x1, Mki1); Mki1++;
        c_sub(&x2, &x2, &temp);
        cc_mult(&temp, &x0, Mki0); Mki0++;
        c_sub(&x3, &rhs[firstcol+3], &temp);
        cc_mult(&temp, &x1, Mki1); Mki1++;
        c_sub(&x3, &x3, &temp);
        cc_mult(&temp, &x2, Mki2); Mki2++;
        c_sub(&x3, &x3, &temp);

        // 更新 rhs 向量
        rhs[++firstcol] = x1;
        rhs[++firstcol] = x2;
        rhs[++firstcol] = x3;
        ++firstcol;

        // 处理剩余的列
        for (k = firstcol; k < ncol; k++) {
            cc_mult(&temp, &x0, Mki0); Mki0++;
            c_sub(&rhs[k], &rhs[k], &temp);
            cc_mult(&temp, &x1, Mki1); Mki1++;
            c_sub(&rhs[k], &rhs[k], &temp);
            cc_mult(&temp, &x2, Mki2); Mki2++;
            c_sub(&rhs[k], &rhs[k], &temp);
            cc_mult(&temp, &x3, Mki3); Mki3++;
            c_sub(&rhs[k], &rhs[k], &temp);
        }

        // 移动到下一组四列的起始位置
        M0 += 4 * ldm + 4;
    }

    // 处理剩余的列，如果列数不足四列但至少有两列
    if ( firstcol < ncol - 1 ) { /* Do 2 columns */
        Mki0 = M0 + 1;
        Mki1 = Mki0 + ldm + 1;

        // 处理第一列
        x0 = rhs[firstcol];
        cc_mult(&temp, &x0, Mki0); Mki0++;
        c_sub(&x1, &rhs[firstcol+1], &temp);

        // 更新 rhs 向量
        rhs[++firstcol] = x1;
        ++firstcol;

        // 处理剩余的列
        for (k = firstcol; k < ncol; k++) {
            cc_mult(&temp, &x0, Mki0); Mki0++;
            c_sub(&rhs[k], &rhs[k], &temp);
            cc_mult(&temp, &x1, Mki1); Mki1++;
            c_sub(&rhs[k], &rhs[k], &temp);
        } 
    }
}
/*! \brief 解决稠密上三角系统的线性方程组。
 *
 * 上三角矩阵存储在二维数组 M(1:ldm,1:ncol) 中。解将存储在 rhs 向量中。
 */
void cusolve (int ldm, int ncol, singlecomplex *M, singlecomplex *rhs)
{
    singlecomplex xj, temp;   /* 定义临时变量 */
    int jcol, j, irow;        /* 定义循环变量 */

    jcol = ncol - 1;          /* 初始化列索引 */

    for (j = 0; j < ncol; j++) {  /* 外层循环遍历每列 */

    c_div(&xj, &rhs[jcol], &M[jcol + jcol*ldm]); /* 计算 xj = rhs[jcol] / M(jcol, jcol) */
    rhs[jcol] = xj;            /* 将计算结果存入 rhs[jcol] */
    
    for (irow = 0; irow < jcol; irow++) {  /* 内层循环遍历当前列之前的每行 */
        cc_mult(&temp, &xj, &M[irow+jcol*ldm]); /* 计算 temp = xj * M(irow, jcol) */
        c_sub(&rhs[irow], &rhs[irow], &temp);  /* 更新 rhs[irow] = rhs[irow] - temp */
    }

    jcol--;    /* 更新列索引，准备处理下一列 */

    }
}


/*! \brief 执行稠密矩阵与向量的乘法：Mxvec = Mxvec + M * vec。
 *
 * 输入矩阵为 M(1:nrow,1:ncol)，乘积存储在 Mxvec[] 中。
 */
void cmatvec (int ldm, int nrow, int ncol, singlecomplex *M, singlecomplex *vec, singlecomplex *Mxvec)
{
    singlecomplex vi0, vi1, vi2, vi3;   /* 定义向量元素和临时变量 */
    singlecomplex *M0, temp;            /* 定义矩阵指针和临时变量 */
    singlecomplex *Mki0, *Mki1, *Mki2, *Mki3;  /* 定义矩阵块指针 */
    register int firstcol = 0;          /* 定义寄存器变量 */

    M0 = &M[0];                         /* 初始化矩阵 M 的指针 */

    while ( firstcol < ncol - 3 ) {    /* 处理每四列 */
    Mki0 = M0;
    Mki1 = Mki0 + ldm;
    Mki2 = Mki1 + ldm;
    Mki3 = Mki2 + ldm;

    vi0 = vec[firstcol++];   /* 依次获取向量元素 */
    vi1 = vec[firstcol++];
    vi2 = vec[firstcol++];
    vi3 = vec[firstcol++];    
    
    for (int k = 0; k < nrow; k++) {   /* 遍历每行 */
        cc_mult(&temp, &vi0, Mki0); Mki0++;
        c_add(&Mxvec[k], &Mxvec[k], &temp);
        cc_mult(&temp, &vi1, Mki1); Mki1++;
        c_add(&Mxvec[k], &Mxvec[k], &temp);
        cc_mult(&temp, &vi2, Mki2); Mki2++;
        c_add(&Mxvec[k], &Mxvec[k], &temp);
        cc_mult(&temp, &vi3, Mki3); Mki3++;
        c_add(&Mxvec[k], &Mxvec[k], &temp);
    }

    M0 += 4 * ldm;   /* 更新矩阵块指针 */
    }

    while ( firstcol < ncol ) {        /* 处理剩余的一列 */
     Mki0 = M0;
    vi0 = vec[firstcol++];
    for (int k = 0; k < nrow; k++) {
        cc_mult(&temp, &vi0, Mki0); Mki0++;
        c_add(&Mxvec[k], &Mxvec[k], &temp);
    }
    M0 += ldm;   /* 更新矩阵块指针 */
    }
    
}
```