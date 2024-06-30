# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\dmyblas2.c`

```
/*! @file dmyblas2.c
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
 * File name:        dmyblas2.c
 */

/*! \brief Solves a dense UNIT lower triangular system
 *
 *  The unit lower 
 * triangular matrix is stored in a 2D array M(1:nrow,1:ncol). 
 * The solution will be returned in the rhs vector.
 */
void dlsolve ( int ldm, int ncol, double *M, double *rhs )
{
    int k;
    double x0, x1, x2, x3, x4, x5, x6, x7;  // 声明变量用于存储中间计算结果
    double *M0;  // 定义指针变量 M0，用于迭代访问 M 数组
    register double *Mki0, *Mki1, *Mki2, *Mki3, *Mki4, *Mki5, *Mki6, *Mki7;  // 声明寄存器变量用于加速访问
    register int firstcol = 0;  // 初始化第一列的索引为 0

    M0 = &M[0];  // 将 M 的地址赋给 M0

    while ( firstcol < ncol - 7 ) { /* Do 8 columns */
      // 设置指针变量 Mki0 到 Mki7，指向当前要处理的列的元素
      Mki0 = M0 + 1;
      Mki1 = Mki0 + ldm + 1;
      Mki2 = Mki1 + ldm + 1;
      Mki3 = Mki2 + ldm + 1;
      Mki4 = Mki3 + ldm + 1;
      Mki5 = Mki4 + ldm + 1;
      Mki6 = Mki5 + ldm + 1;
      Mki7 = Mki6 + ldm + 1;

      // 计算解向量 rhs 中的元素，使用部分前向替换法（forward substitution）
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

      // 更新 rhs 向量中的元素
      rhs[++firstcol] = x1;
      rhs[++firstcol] = x2;
      rhs[++firstcol] = x3;
      rhs[++firstcol] = x4;
      rhs[++firstcol] = x5;
      rhs[++firstcol] = x6;
      rhs[++firstcol] = x7;
      ++firstcol;
    
      // 继续用部分前向替换法更新 rhs 中的剩余元素
      for (k = firstcol; k < ncol; k++)
    rhs[k] = rhs[k] - x0 * *Mki0++ - x1 * *Mki1++
                    - x2 * *Mki2++ - x3 * *Mki3++
                        - x4 * *Mki4++ - x5 * *Mki5++
            - x6 * *Mki6++ - x7 * *Mki7++;
 
      // 移动 M0 指针到下一组 8 列的起始位置
      M0 += 8 * ldm + 8;
    }
}
    while ( firstcol < ncol - 3 ) { /* Do 4 columns */
      Mki0 = M0 + 1;  // 计算第一个矩阵 M 的偏移量
      Mki1 = Mki0 + ldm + 1;  // 计算第二个矩阵 M 的偏移量
      Mki2 = Mki1 + ldm + 1;  // 计算第三个矩阵 M 的偏移量
      Mki3 = Mki2 + ldm + 1;  // 计算第四个矩阵 M 的偏移量

      x0 = rhs[firstcol];  // 读取当前列的 rhs 值
      x1 = rhs[firstcol+1] - x0 * *Mki0++;  // 计算第一列的更新值
      x2 = rhs[firstcol+2] - x0 * *Mki0++ - x1 * *Mki1++;  // 计算第二列的更新值
      x3 = rhs[firstcol+3] - x0 * *Mki0++ - x1 * *Mki1++ - x2 * *Mki2++;  // 计算第三列的更新值

      rhs[++firstcol] = x1;  // 更新第一列的 rhs 值
      rhs[++firstcol] = x2;  // 更新第二列的 rhs 值
      rhs[++firstcol] = x3;  // 更新第三列的 rhs 值
      ++firstcol;  // 移动到下一列
    
      for (k = firstcol; k < ncol; k++)
        rhs[k] = rhs[k] - x0 * *Mki0++ - x1 * *Mki1++
                        - x2 * *Mki2++ - x3 * *Mki3++;  // 更新剩余列的 rhs 值
 
      M0 += 4 * ldm + 4;  // 更新矩阵 M 的起始位置
    }

    if ( firstcol < ncol - 1 ) { /* Do 2 columns */
      Mki0 = M0 + 1;  // 计算第一个矩阵 M 的偏移量
      Mki1 = Mki0 + ldm + 1;  // 计算第二个矩阵 M 的偏移量

      x0 = rhs[firstcol];  // 读取当前列的 rhs 值
      x1 = rhs[firstcol+1] - x0 * *Mki0++;  // 计算第一列的更新值

      rhs[++firstcol] = x1;  // 更新第一列的 rhs 值
      ++firstcol;  // 移动到下一列
    
      for (k = firstcol; k < ncol; k++)
        rhs[k] = rhs[k] - x0 * *Mki0++ - x1 * *Mki1++;  // 更新剩余列的 rhs 值
    }
/*! \brief Solves a dense upper triangular system
 * 
 * The upper triangular matrix is
 * stored in a 2-dim array M(1:ldm,1:ncol). The solution will be returned
 * in the rhs vector.
 */
void dusolve (int ldm, int ncol, double *M, double *rhs)
{
    double xj;      /* 存储临时解 */
    int jcol, j, irow;   /* 定义循环索引变量 */

    jcol = ncol - 1;    /* 初始化列索引为最后一列 */

    for (j = 0; j < ncol; j++) {    /* 外层循环，对每一列进行处理 */

        xj = rhs[jcol] / M[jcol + jcol*ldm];         /* M(jcol, jcol) 的求解 */
        rhs[jcol] = xj;    /* 将解存入 rhs 向量 */
        
        for (irow = 0; irow < jcol; irow++)
            rhs[irow] -= xj * M[irow + jcol*ldm];    /* M(irow, jcol) 的更新 */

        jcol--;    /* 处理下一列 */
    }
}


/*! \brief Performs a dense matrix-vector multiply: Mxvec = Mxvec + M * vec.
 * 
 * The input matrix is M(1:nrow,1:ncol); The product is returned in Mxvec[].
 */
void dmatvec (int ldm, int nrow, int ncol, double *M, double *vec, double *Mxvec)
{
    double vi0, vi1, vi2, vi3, vi4, vi5, vi6, vi7;   /* 存储向量元素 */
    double *M0;    /* 指向矩阵 M 的起始位置 */
    register double *Mki0, *Mki1, *Mki2, *Mki3, *Mki4, *Mki5, *Mki6, *Mki7;   /* 寄存器变量用于加速访问 */
    register int firstcol = 0;   /* 待处理的第一列索引 */
    int k;    /* 循环索引 */

    M0 = &M[0];    /* M 的起始位置 */

    while ( firstcol < ncol - 7 ) {    /* 处理 8 列 */

        Mki0 = M0;
        Mki1 = Mki0 + ldm;
        Mki2 = Mki1 + ldm;
        Mki3 = Mki2 + ldm;
        Mki4 = Mki3 + ldm;
        Mki5 = Mki4 + ldm;
        Mki6 = Mki5 + ldm;
        Mki7 = Mki6 + ldm;

        vi0 = vec[firstcol++];
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
                    + vi6 * *Mki6++ + vi7 * *Mki7++;

        M0 += 8 * ldm;    /* 更新 M 的指针位置 */
    }

    while ( firstcol < ncol - 3 ) {    /* 处理 4 列 */

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
                    + vi2 * *Mki2++ + vi3 * *Mki3++;

        M0 += 4 * ldm;    /* 更新 M 的指针位置 */
    }

    while ( firstcol < ncol ) {        /* 处理剩余的 1 列 */

        Mki0 = M0;
        vi0 = vec[firstcol++];
        
        for (k = 0; k < nrow; k++)
            Mxvec[k] += vi0 * *Mki0++;

        M0 += ldm;    /* 更新 M 的指针位置 */
    }
}
```