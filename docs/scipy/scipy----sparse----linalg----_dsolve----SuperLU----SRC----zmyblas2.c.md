# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\zmyblas2.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file zmyblas2.c
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
 * File name:        zmyblas2.c
 */
#include "slu_dcomplex.h"

/*! \brief Solves a dense UNIT lower triangular system
 * 
 * The unit lower 
 * triangular matrix is stored in a 2D array M(1:nrow,1:ncol). 
 * The solution will be returned in the rhs vector.
 */
void zlsolve ( int ldm, int ncol, doublecomplex *M, doublecomplex *rhs )
{
    int k;
    doublecomplex x0, x1, x2, x3, temp;
    doublecomplex *M0;
    doublecomplex *Mki0, *Mki1, *Mki2, *Mki3;
    register int firstcol = 0;

    // 设置指针 M0 指向 M 的起始位置
    M0 = &M[0];

    // 循环处理每 4 列的情况，直到剩余列数少于 4 列
    while ( firstcol < ncol - 3 ) { /* Do 4 columns */
          // 设置指针 Mki0 - Mki3 指向当前处理列的对角线元素
          Mki0 = M0 + 1;
          Mki1 = Mki0 + ldm + 1;
          Mki2 = Mki1 + ldm + 1;
          Mki3 = Mki2 + ldm + 1;

          // 初始化 x0，并计算与当前列的乘积
          x0 = rhs[firstcol];
          zz_mult(&temp, &x0, Mki0); Mki0++;
          z_sub(&x1, &rhs[firstcol+1], &temp);

          // 类似地，计算 x2 和 x3
          zz_mult(&temp, &x0, Mki0); Mki0++;
          z_sub(&x2, &rhs[firstcol+2], &temp);
          zz_mult(&temp, &x1, Mki1); Mki1++;
          z_sub(&x2, &x2, &temp);
          zz_mult(&temp, &x0, Mki0); Mki0++;
          z_sub(&x3, &rhs[firstcol+3], &temp);
          zz_mult(&temp, &x1, Mki1); Mki1++;
          z_sub(&x3, &x3, &temp);
          zz_mult(&temp, &x2, Mki2); Mki2++;
          z_sub(&x3, &x3, &temp);

          // 将计算得到的 x1, x2, x3 存入 rhs 向量中
          rhs[++firstcol] = x1;
          rhs[++firstcol] = x2;
          rhs[++firstcol] = x3;
          ++firstcol;

          // 对于剩余的列，继续进行类似的操作
          for (k = firstcol; k < ncol; k++) {
                zz_mult(&temp, &x0, Mki0); Mki0++;
                z_sub(&rhs[k], &rhs[k], &temp);
                zz_mult(&temp, &x1, Mki1); Mki1++;
                z_sub(&rhs[k], &rhs[k], &temp);
                zz_mult(&temp, &x2, Mki2); Mki2++;
                z_sub(&rhs[k], &rhs[k], &temp);
                zz_mult(&temp, &x3, Mki3); Mki3++;
                z_sub(&rhs[k], &rhs[k], &temp);
          }

          // 移动 M0 指针到下一组 4 列的起始位置
          M0 += 4 * ldm + 4;
    }

    // 处理剩余不足 4 列的情况
    if ( firstcol < ncol - 1 ) { /* Do 2 columns */
        Mki0 = M0 + 1;
        Mki1 = Mki0 + ldm + 1;

        // 计算 x0 和 x1
        x0 = rhs[firstcol];
        zz_mult(&temp, &x0, Mki0); Mki0++;
        z_sub(&x1, &rhs[firstcol+1], &temp);

        // 将计算得到的 x1 存入 rhs 向量中
        rhs[++firstcol] = x1;
        ++firstcol;

        // 对于剩余的列，继续进行类似的操作
        for (k = firstcol; k < ncol; k++) {
            zz_mult(&temp, &x0, Mki0); Mki0++;
            z_sub(&rhs[k], &rhs[k], &temp);
            zz_mult(&temp, &x1, Mki1); Mki1++;
            z_sub(&rhs[k], &rhs[k], &temp);
        } 
    }
}
/*! \brief Solves a dense upper triangular system.
 *
 * The upper triangular matrix is
 * stored in a 2-dim array M(1:ldm,1:ncol). The solution will be returned
 * in the rhs vector.
 */
void zusolve (int ldm, int ncol, doublecomplex *M, doublecomplex *rhs)
{
    doublecomplex xj, temp;
    int jcol, j, irow;

    jcol = ncol - 1;  /* Initialize jcol to the last column index */

    for (j = 0; j < ncol; j++) {  /* Iterate over each column from right to left */

        z_div(&xj, &rhs[jcol], &M[jcol + jcol*ldm]); /* Compute xj = rhs[jcol] / M(jcol, jcol) */
        rhs[jcol] = xj;  /* Store the computed solution back into rhs[jcol] */

        for (irow = 0; irow < jcol; irow++) {
            zz_mult(&temp, &xj, &M[irow+jcol*ldm]); /* Compute temp = xj * M(irow, jcol) */
            z_sub(&rhs[irow], &rhs[irow], &temp);  /* Update rhs[irow] -= temp */
        }

        jcol--;  /* Move to the next column to the left */
    }
}

/*! \brief Performs a dense matrix-vector multiply: Mxvec = Mxvec + M * vec.
 *
 * The input matrix is M(1:nrow,1:ncol); The product is returned in Mxvec[].
 */
void zmatvec (int ldm, int nrow, int ncol, doublecomplex *M, doublecomplex *vec, doublecomplex *Mxvec)
{
    doublecomplex vi0, vi1, vi2, vi3;
    doublecomplex *M0, temp;
    doublecomplex *Mki0, *Mki1, *Mki2, *Mki3;
    register int firstcol = 0;
    int k;

    M0 = &M[0];  /* Start with the first column of matrix M */

    while ( firstcol < ncol - 3 ) {    /* Process 4 columns at a time */
        Mki0 = M0;
        Mki1 = Mki0 + ldm;
        Mki2 = Mki1 + ldm;
        Mki3 = Mki2 + ldm;

        vi0 = vec[firstcol++];  /* Load vector elements into vi0, vi1, vi2, vi3 */
        vi1 = vec[firstcol++];
        vi2 = vec[firstcol++];
        vi3 = vec[firstcol++];    

        for (k = 0; k < nrow; k++) {
            zz_mult(&temp, &vi0, Mki0); Mki0++;
            z_add(&Mxvec[k], &Mxvec[k], &temp);
            zz_mult(&temp, &vi1, Mki1); Mki1++;
            z_add(&Mxvec[k], &Mxvec[k], &temp);
            zz_mult(&temp, &vi2, Mki2); Mki2++;
            z_add(&Mxvec[k], &Mxvec[k], &temp);
            zz_mult(&temp, &vi3, Mki3); Mki3++;
            z_add(&Mxvec[k], &Mxvec[k], &temp);
        }

        M0 += 4 * ldm;  /* Move to the next set of 4 columns */
    }

    while ( firstcol < ncol ) {        /* Process the remaining columns one by one */
        Mki0 = M0;
        vi0 = vec[firstcol++];

        for (k = 0; k < nrow; k++) {
            zz_mult(&temp, &vi0, Mki0); Mki0++;
            z_add(&Mxvec[k], &Mxvec[k], &temp);
        }

        M0 += ldm;  /* Move to the next column */
    }
}
```