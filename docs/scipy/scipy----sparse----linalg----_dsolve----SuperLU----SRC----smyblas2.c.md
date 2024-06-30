# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\smyblas2.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file smyblas2.c
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
 * File name:        smyblas2.c
 */

/*! \brief Solves a dense UNIT lower triangular system
 *
 *  The unit lower 
 * triangular matrix is stored in a 2D array M(1:nrow,1:ncol). 
 * The solution will be returned in the rhs vector.
 */
void slsolve ( int ldm, int ncol, float *M, float *rhs )
{
    int k;
    float x0, x1, x2, x3, x4, x5, x6, x7;
    float *M0;
    register float *Mki0, *Mki1, *Mki2, *Mki3, *Mki4, *Mki5, *Mki6, *Mki7;
    register int firstcol = 0;

    M0 = &M[0];

    // 处理每一组 8 列数据
    while ( firstcol < ncol - 7 ) { /* Do 8 columns */
      // 指针 M0 指向当前需要处理的下三角矩阵的起始位置
      Mki0 = M0 + 1;
      Mki1 = Mki0 + ldm + 1;
      Mki2 = Mki1 + ldm + 1;
      Mki3 = Mki2 + ldm + 1;
      Mki4 = Mki3 + ldm + 1;
      Mki5 = Mki4 + ldm + 1;
      Mki6 = Mki5 + ldm + 1;
      Mki7 = Mki6 + ldm + 1;

      // 利用部分前置计算优化，计算 rhs 向量中的解
      x0 = rhs[firstcol];
      x1 = rhs[firstcol+1] - x0 * *Mki0++;
      x2 = rhs[firstcol+2] - x0 * *Mki0++ - x1 * *Mki1++;
      x3 = rhs[firstcol+3] - x0 * *Mki0++ - x1 * *Mki1++ - x2 * *Mki2++;
      x4 = rhs[firstcol+4] - x0 * *Mki0++ - x1 * *Mki1++ - x2 * *Mki2++
                       - x3 * *Mki3++;
      x5 = rhs[firstcol+5] - x0 * *Mki0++ - x1 * *Mki1++ - x2 * *Mki2++
                       - x3 * *Mki3++ - x4 * *Mki4++;
      x6 = rhs[firstcol+6] - x0 * *Mki0++ - x1 * *Mki1++ - x2 * *Mki2++
                       - x3 * *Mki3++ - x4 * *Mki4++ - x5 * *Mki5++;
      x7 = rhs[firstcol+7] - x0 * *Mki0++ - x1 * *Mki1++ - x2 * *Mki2++
                       - x3 * *Mki3++ - x4 * *Mki4++ - x5 * *Mki5++
               - x6 * *Mki6++;

      // 更新 rhs 向量中的值
      rhs[++firstcol] = x1;
      rhs[++firstcol] = x2;
      rhs[++firstcol] = x3;
      rhs[++firstcol] = x4;
      rhs[++firstcol] = x5;
      rhs[++firstcol] = x6;
      rhs[++firstcol] = x7;
      ++firstcol;
    
      // 处理剩余的列
      for (k = firstcol; k < ncol; k++)
        rhs[k] = rhs[k] - x0 * *Mki0++ - x1 * *Mki1++
                    - x2 * *Mki2++ - x3 * *Mki3++
                        - x4 * *Mki4++ - x5 * *Mki5++
            - x6 * *Mki6++ - x7 * *Mki7++;
 
      // 移动到下一组 8 列数据的起始位置
      M0 += 8 * ldm + 8;
    }
}
    while ( firstcol < ncol - 3 ) { /* Do 4 columns */
      // 对4列进行操作，循环条件是 firstcol 小于 ncol - 3
      Mki0 = M0 + 1;
      Mki1 = Mki0 + ldm + 1;
      Mki2 = Mki1 + ldm + 1;
      Mki3 = Mki2 + ldm + 1;

      x0 = rhs[firstcol];
      x1 = rhs[firstcol+1] - x0 * *Mki0++;
      x2 = rhs[firstcol+2] - x0 * *Mki0++ - x1 * *Mki1++;
      x3 = rhs[firstcol+3] - x0 * *Mki0++ - x1 * *Mki1++ - x2 * *Mki2++;

      rhs[++firstcol] = x1;
      rhs[++firstcol] = x2;
      rhs[++firstcol] = x3;
      ++firstcol;
    
      for (k = firstcol; k < ncol; k++)
    // 更新右手边向量 rhs 中的值
    rhs[k] = rhs[k] - x0 * *Mki0++ - x1 * *Mki1++
                    - x2 * *Mki2++ - x3 * *Mki3++;
 
      M0 += 4 * ldm + 4;
    }

    if ( firstcol < ncol - 1 ) { /* Do 2 columns */
      // 对2列进行操作，条件是 firstcol 小于 ncol - 1
      Mki0 = M0 + 1;
      Mki1 = Mki0 + ldm + 1;

      x0 = rhs[firstcol];
      x1 = rhs[firstcol+1] - x0 * *Mki0++;

      rhs[++firstcol] = x1;
      ++firstcol;
    
      for (k = firstcol; k < ncol; k++)
    // 更新右手边向量 rhs 中的值
    rhs[k] = rhs[k] - x0 * *Mki0++ - x1 * *Mki1++;
 
    }
/*! \brief Solves a dense upper triangular system
 * 
 * The upper triangular matrix is
 * stored in a 2-dim array M(1:ldm,1:ncol). The solution will be returned
 * in the rhs vector.
 */
void
susolve (int ldm, int ncol, float *M, float *rhs)
{
    float xj;           /* 存储临时计算结果 */
    int jcol, j, irow;  /* jcol: 当前列数，j: 循环变量，irow: 行数 */

    jcol = ncol - 1;    /* 初始化 jcol 为最后一列的索引 */

    for (j = 0; j < ncol; j++) {    /* 遍历每一列 */

    xj = rhs[jcol] / M[jcol + jcol*ldm];         /* M(jcol, jcol) */
    rhs[jcol] = xj;    /* 更新 rhs 向量中的值 */
    
    for (irow = 0; irow < jcol; irow++)
        rhs[irow] -= xj * M[irow + jcol*ldm];    /* M(irow, jcol) 的更新 */

    jcol--;    /* 列数减一，处理下一列 */
    }
}

/*! \brief Performs a dense matrix-vector multiply: Mxvec = Mxvec + M * vec.
 * 
 * The input matrix is M(1:nrow,1:ncol); The product is returned in Mxvec[].
 */
void smatvec (int ldm, int nrow, int ncol, float *M, float *vec, float *Mxvec)
{
    float vi0, vi1, vi2, vi3, vi4, vi5, vi6, vi7;    /* 存储向量元素 */
    float *M0;    /* 指向矩阵 M 的起始地址 */
    register float *Mki0, *Mki1, *Mki2, *Mki3, *Mki4, *Mki5, *Mki6, *Mki7;    /* 寄存器变量，指向当前列的元素 */
    register int firstcol = 0;    /* 当前处理的第一列索引 */
    int k;    /* 循环变量 */

    M0 = &M[0];    /* 初始化 M0 指向 M 的第一个元素 */

    while ( firstcol < ncol - 7 ) {    /* 每次处理 8 列 */

    Mki0 = M0;
    Mki1 = Mki0 + ldm;
    Mki2 = Mki1 + ldm;
    Mki3 = Mki2 + ldm;
    Mki4 = Mki3 + ldm;
    Mki5 = Mki4 + ldm;
    Mki6 = Mki5 + ldm;
    Mki7 = Mki6 + ldm;

    vi0 = vec[firstcol++];    /* 从向量 vec 中读取元素 */
    vi1 = vec[firstcol++];
    vi2 = vec[firstcol++];
    vi3 = vec[firstcol++];    
    vi4 = vec[firstcol++];
    vi5 = vec[firstcol++];
    vi6 = vec[firstcol++];
    vi7 = vec[firstcol++];    

    for (k = 0; k < nrow; k++) 
        Mxvec[k] += vi0 * *Mki0++ + vi1 * *Mki1++
              + vi2 * *Mki2++ + vi3 * *Mki3++ 
              + vi4 * *Mki4++ + vi5 * *Mki5++
              + vi6 * *Mki6++ + vi7 * *Mki7++;    /* 执行矩阵向量乘法 */

    M0 += 8 * ldm;    /* 移动 M0 到下一组列的起始位置 */
    }

    while ( firstcol < ncol - 3 ) {    /* 每次处理 4 列 */

    Mki0 = M0;
    Mki1 = Mki0 + ldm;
    Mki2 = Mki1 + ldm;
    Mki3 = Mki2 + ldm;

    vi0 = vec[firstcol++];
    vi1 = vec[firstcol++];
    vi2 = vec[firstcol++];
    vi3 = vec[firstcol++];    
    for (k = 0; k < nrow; k++) 
        Mxvec[k] += vi0 * *Mki0++ + vi1 * *Mki1++
              + vi2 * *Mki2++ + vi3 * *Mki3++ ;    /* 执行矩阵向量乘法 */

    M0 += 4 * ldm;    /* 移动 M0 到下一组列的起始位置 */
    }

    while ( firstcol < ncol ) {        /* 处理剩余的 1 列 */

    Mki0 = M0;
    vi0 = vec[firstcol++];
    for (k = 0; k < nrow; k++)
        Mxvec[k] += vi0 * *Mki0++;    /* 执行矩阵向量乘法 */

    M0 += ldm;    /* 移动 M0 到下一列的起始位置 */
    }
}
```