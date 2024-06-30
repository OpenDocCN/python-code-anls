# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\dcolumn_bmod.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file dcolumn_bmod.c
 *  \brief performs numeric block updates
 *
 * <pre>
 * -- SuperLU routine (version 3.0) --
 * Univ. of California Berkeley, Xerox Palo Alto Research Center,
 * and Lawrence Berkeley National Lab.
 * October 15, 2003
 *
 * Copyright (c) 1994 by Xerox Corporation.  All rights reserved.
 *
 * THIS MATERIAL IS PROVIDED AS IS, WITH ABSOLUTELY NO WARRANTY
 * EXPRESSED OR IMPLIED.  ANY USE IS AT YOUR OWN RISK.
 * 
 *  Permission is hereby granted to use or copy this program for any
 *  purpose, provided the above notices are retained on all copies.
 *  Permission to modify the code and to distribute modified code is
 *  granted, provided the above notices are retained, and a notice that
 *  the code was modified is included with the above copyright notice.
 * </pre>
*/

#include <stdio.h>
#include <stdlib.h>
#include "slu_ddefs.h"

/*! \brief 
 *
 * <pre>
 * Purpose:
 * ========
 * Performs numeric block updates (sup-col) in topological order.
 * It features: col-col, 2cols-col, 3cols-col, and sup-col updates.
 * Special processing on the supernodal portion of L\\U[*,j]
 * Return value:   0 - successful return
 *               > 0 - number of bytes allocated when run out of space
 * </pre>
 */
int
dcolumn_bmod (
         const int  jcol,      /* in */                                // 当前处理的列索引
         const int  nseg,      /* in */                                 // 与列相关的分段数
         double     *dense,      /* in */                                // 密集矩阵的数据
         double     *tempv,      /* working array */                     // 工作数组
         int        *segrep,  /* in */                                   // 分段的代表元素数组
         int        *repfnz,  /* in */                                   // 非零元的代表元素数组
         int        fpanelc,  /* in -- first column in the current panel */ // 当前面板的第一列索引
         GlobalLU_t *Glu,     /* modified */                             // 全局LU因子结构
         SuperLUStat_t *stat  /* output */                               // 统计信息输出
         )
{

#ifdef _CRAY
    _fcd ftcs1 = _cptofcd("L", strlen("L")),                             // 定义_Cray特定的_fcd类型变量
         ftcs2 = _cptofcd("N", strlen("N")),
         ftcs3 = _cptofcd("U", strlen("U"));
#endif
    int         incx = 1, incy = 1;                                      // 向量x和y的增量
    double      alpha, beta;                                             // 乘法的标量参数

    /* krep = representative of current k-th supernode
     * fsupc = first supernodal column
     * nsupc = no of columns in supernode
     * nsupr = no of rows in supernode (used as leading dimension)
     * luptr = location of supernodal LU-block in storage
     * kfnz = first nonz in the k-th supernodal segment
     * no_zeros = no of leading zeros in a supernodal U-segment
     */
    double      ukj, ukj1, ukj2;                                         // 当前处理的LU因子的元素
    int_t        luptr, luptr1, luptr2;                                   // LU分解块的位置
    int          fsupc, nsupc, nsupr, segsze;                             // 超节点的相关属性
    int          nrow;      /* No of rows in the matrix of matrix-vector */ // 矩阵向量乘法中的行数
    int          jcolp1, jsupno, k, ksub, krep, krep_ind, ksupno;          // 迭代索引和超节点编号
    int_t        lptr, kfnz, isub, irow, i;                               // 迭代索引和LU数据结构的索引
    int_t        no_zeros, new_next, ufirst, nextlu;                       // LU因子的特殊处理属性
    # 第一个列在小LU更新中的位置
    int          fst_col; /* First column within small LU update */
    # 当前面板的第一个列与当前snode的第一个列之间的距离
    int          d_fsupc; /* Distance between the first column of the current
                 panel and the first column of the current snode. */
    # 指向超节点和超节点编号的指针
    int          *xsup, *supno;
    # L 分解中的非零元素索引和超节点列索引
    int_t        *lsub, *xlsub;
    # L 分解中的非零元素值和超节点列值
    double       *lusup;
    # 超节点列值的行索引
    int_t        *xlusup;
    # LU 分解中最大的非零元素数
    int_t        nzlumax;
    # 临时向量，用于计算
    double       *tempv1;
    # 零常量
    double      zero = 0.0;
    # 一常量
    double      one = 1.0;
    # 负一常量
    double      none = -1.0;
    # 内存错误标志
    int_t        mem_error;
    # 统计操作次数的指针
    flops_t      *ops = stat->ops;

    # 获得全局LU数据结构中的相关指针和参数
    xsup    = Glu->xsup;
    supno   = Glu->supno;
    lsub    = Glu->lsub;
    xlsub   = Glu->xlsub;
    lusup   = (double *) Glu->lusup;
    xlusup  = Glu->xlusup;
    nzlumax = Glu->nzlumax;
    jcolp1 = jcol + 1;
    jsupno = supno[jcol];
    
    /* 
     * 对于U[*,j]中每个非零超节点段按拓扑顺序处理 
     */
    k = nseg - 1;
    for (ksub = 0; ksub < nseg; ksub++) {

    krep = segrep[k];
    k--;
    ksupno = supno[krep];
    if ( jsupno != ksupno ) { /* 如果当前处理的超节点不在矩形区域外 */

        fsupc = xsup[ksupno];  /* 当前超节点的首列索引 */
        fst_col = SUPERLU_MAX ( fsupc, fpanelc );  /* 首列索引和当前面板列索引的较大者 */

          /* 当前超节点到当前面板列的距离；
           如果 fsupc > fpanelc，则 d_fsupc=0。 */
          d_fsupc = fst_col - fsupc;

        luptr = xlusup[fst_col] + d_fsupc;  /* LU 分解后的非零元素索引 */
        lptr = xlsub[fsupc] + d_fsupc;  /* LU 分解前的非零元素索引 */

        kfnz = repfnz[krep];
        kfnz = SUPERLU_MAX ( kfnz, fpanelc );  /* krep 列的第一个非零元素所在的列索引 */

        segsze = krep - kfnz + 1;  /* 超节点包含的列数 */
        nsupc = krep - fst_col + 1;  /* 当前超节点的列数 */
        nsupr = xlsub[fsupc+1] - xlsub[fsupc];    /* 导致行号 */
        nrow = nsupr - d_fsupc - nsupc;  /* 导致行数 */
        krep_ind = lptr + nsupc - 1;  /* 密集矩阵中的索引 */

        ops[TRSV] += segsze * (segsze - 1);  /* TRSV 操作次数增加 */
        ops[GEMV] += 2 * nrow * segsze;  /* GEMV 操作次数增加 */


        /* 
         * 情况 1: 更新大小为 1 的 U 段 -- 列 - 列 更新 
         */
        if ( segsze == 1 ) {
          ukj = dense[lsub[krep_ind]];  /* 密集矩阵中的元素 */
        luptr += nsupr*(nsupc-1) + nsupc;  /* LU 分解后的索引位置 */

        for (i = lptr + nsupc; i < xlsub[fsupc+1]; ++i) {
            irow = lsub[i];
            dense[irow] -=  ukj*lusup[luptr];
            luptr++;
        }

        } else if ( segsze <= 3 ) {
        ukj = dense[lsub[krep_ind]];
        luptr += nsupr*(nsupc-1) + nsupc-1;
        ukj1 = dense[lsub[krep_ind - 1]];
        luptr1 = luptr - nsupr;

        if ( segsze == 2 ) { /* 情况 2: 2 列 - 列 更新 */
            ukj -= ukj1 * lusup[luptr1];
            dense[lsub[krep_ind]] = ukj;
            for (i = lptr + nsupc; i < xlsub[fsupc+1]; ++i) {
                irow = lsub[i];
                luptr++;
                luptr1++;
                dense[irow] -= ( ukj*lusup[luptr]
                    + ukj1*lusup[luptr1] );
            }
        } else { /* 情况 3: 3 列 - 列 更新 */
            ukj2 = dense[lsub[krep_ind - 2]];
            luptr2 = luptr1 - nsupr;
            ukj1 -= ukj2 * lusup[luptr2-1];
            ukj = ukj - ukj1*lusup[luptr1] - ukj2*lusup[luptr2];
            dense[lsub[krep_ind]] = ukj;
            dense[lsub[krep_ind-1]] = ukj1;
            for (i = lptr + nsupc; i < xlsub[fsupc+1]; ++i) {
                irow = lsub[i];
                luptr++;
                luptr1++;
                luptr2++;
                dense[irow] -= ( ukj*lusup[luptr]
                 + ukj1*lusup[luptr1] + ukj2*lusup[luptr2] );
            }
        }



        } else {
          /*
         * 情况: 超 - 列 更新
         * 执行三角解和块更新，
         * 然后将超 - 列更新的结果散布到密集矩阵中
         */

        no_zeros = kfnz - fst_col;

            /* 将密集矩阵中 U[*,j] 段复制到 tempv 中 */
            isub = lptr + no_zeros;
            for (i = 0; i < segsze; i++) {
              irow = lsub[isub];
            tempv[i] = dense[irow];
            ++isub; 
            }

            /* 密集矩阵的三角解 -- 开始有效的三角形区域 */
        luptr += nsupr * no_zeros + no_zeros;
#ifdef USE_VENDOR_BLAS
#ifdef _CRAY
        STRSV( ftcs1, ftcs2, ftcs3, &segsze, &lusup[luptr], 
               &nsupr, tempv, &incx );
#else        
        dtrsv_( "L", "N", "U", &segsze, &lusup[luptr], 
               &nsupr, tempv, &incx );
#endif        
         luptr += segsze;  /* 移动到下一个密集矩阵-向量操作的起始位置 */
        tempv1 = &tempv[segsze];
                alpha = one;
                beta = zero;
#ifdef _CRAY
        SGEMV( ftcs2, &nrow, &segsze, &alpha, &lusup[luptr], 
               &nsupr, tempv, &incx, &beta, tempv1, &incy );
#else
        dgemv_( "N", &nrow, &segsze, &alpha, &lusup[luptr], 
               &nsupr, tempv, &incx, &beta, tempv1, &incy );
#endif
#else
        dlsolve ( nsupr, segsze, &lusup[luptr], tempv );

         luptr += segsze;  /* 移动到下一个密集矩阵-向量操作的起始位置 */
        tempv1 = &tempv[segsze];
        dmatvec (nsupr, nrow , segsze, &lusup[luptr], tempv, tempv1);
#endif
        
        
                /* 将 tempv[] 散布到 SPA dense[] 作为临时存储 */
                isub = lptr + no_zeros;
                for (i = 0; i < segsze; i++) {
                    irow = lsub[isub];
                    dense[irow] = tempv[i];
                    tempv[i] = zero;
                    ++isub;
                }

        /* 将 tempv1[] 散布到 SPA dense[] */
        for (i = 0; i < nrow; i++) {
            irow = lsub[isub];
            dense[irow] -= tempv1[i];
            tempv1[i] = zero;
            ++isub;
        }
        }
        
    } /* if jsupno ... */

    } /* for each segment... */

    /*
     *    处理 L\U[*,j] 的超节点部分
     */
    nextlu = xlusup[jcol];
    fsupc = xsup[jsupno];

    /* 将 SPA dense 复制到 L\U[*,j] */
    new_next = nextlu + xlsub[fsupc+1] - xlsub[fsupc];
    while ( new_next > nzlumax ) {
    mem_error = dLUMemXpand(jcol, nextlu, LUSUP, &nzlumax, Glu);
    if (mem_error) return (mem_error);
    lusup = (double *) Glu->lusup;
    lsub = Glu->lsub;
    }

    for (isub = xlsub[fsupc]; isub < xlsub[fsupc+1]; isub++) {
      irow = lsub[isub];
    lusup[nextlu] = dense[irow];
        dense[irow] = zero;
    ++nextlu;
    }

    xlusup[jcolp1] = nextlu;    /* 完成 L\U[*,jcol] */

    /* 用于更新面板内部（同时在当前超节点内部），应从面板的第一列或超节点的第一列开始。
     * 有两种情况：
     *    1) fsupc < fpanelc，则 fst_col := fpanelc
     *    2) fsupc >= fpanelc，则 fst_col := fsupc
     */
    fst_col = SUPERLU_MAX ( fsupc, fpanelc );

    if ( fst_col < jcol ) {

      /* 当前超节点和当前面板之间的距离。
       * 如果 fsupc >= fpanelc，则 d_fsupc=0。
       */
      d_fsupc = fst_col - fsupc;

    lptr = xlsub[fsupc] + d_fsupc;
    luptr = xlusup[fst_col] + d_fsupc;
    nsupr = xlsub[fsupc+1] - xlsub[fsupc];    /* 主导维度 */
    nsupc = jcol - fst_col;    /* 不包括 jcol */
    nrow = nsupr - d_fsupc - nsupc;
    # 计算 ufirst 的值，xlusup 是存储 snode L\U(jsupno) 开始位置的数组，jcol 是列索引
    ufirst = xlusup[jcol] + d_fsupc;

    # 更新 ops 数组中 TRSV 操作的计数器，nsupc 为当前超节点的列数
    ops[TRSV] += nsupc * (nsupc - 1);

    # 更新 ops 数组中 GEMV 操作的计数器，nrow 为行数，nsupc 为当前超节点的列数
    ops[GEMV] += 2 * nrow * nsupc;
#ifdef USE_VENDOR_BLAS
#ifdef _CRAY
    // 使用 CRAY 平台特定的 STRSV 函数进行向量的三角解算
    STRSV( ftcs1, ftcs2, ftcs3, &nsupc, &lusup[luptr], 
           &nsupr, &lusup[ufirst], &incx );
#else
    // 使用通用的 dtrsv_ 函数进行向量的三角解算
    dtrsv_( "L", "N", "U", &nsupc, &lusup[luptr], 
           &nsupr, &lusup[ufirst], &incx );
#endif
    
    alpha = none; beta = one; /* y := beta*y + alpha*A*x */
    
#ifdef _CRAY
    // 使用 CRAY 平台特定的 SGEMV 函数进行矩阵-向量乘法
    SGEMV( ftcs2, &nrow, &nsupc, &alpha, &lusup[luptr+nsupc], &nsupr,
           &lusup[ufirst], &incx, &beta, &lusup[ufirst+nsupc], &incy );
#else
    // 使用通用的 dgemv_ 函数进行矩阵-向量乘法
    dgemv_( "N", &nrow, &nsupc, &alpha, &lusup[luptr+nsupc], &nsupr,
           &lusup[ufirst], &incx, &beta, &lusup[ufirst+nsupc], &incy );
#endif
#else
    // 当未定义 USE_VENDOR_BLAS 时，使用自定义的线性方程组求解函数 dlsolve
    dlsolve ( nsupr, nsupc, &lusup[luptr], &lusup[ufirst] );

    // 当未定义 USE_VENDOR_BLAS 时，使用自定义的矩阵向量乘法函数 dmatvec
    dmatvec ( nsupr, nrow, nsupc, &lusup[luptr+nsupc],
        &lusup[ufirst], tempv );
    
        /* Copy updates from tempv[*] into lusup[*] */
    // 将 tempv 中的更新拷贝到 lusup 中对应位置
    isub = ufirst + nsupc;
    for (i = 0; i < nrow; i++) {
        lusup[isub] -= tempv[i];
        tempv[i] = 0.0;
        ++isub;
    }

#endif
    
    
    } /* if fst_col < jcol ... */ 

    // 函数执行完毕，返回 0 表示正常结束
    return 0;
}
```