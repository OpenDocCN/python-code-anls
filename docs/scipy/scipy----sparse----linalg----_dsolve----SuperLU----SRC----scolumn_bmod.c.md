# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\scolumn_bmod.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file scolumn_bmod.c
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
#include "slu_sdefs.h"


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
scolumn_bmod (
         const int  jcol,      /* in */                                // 列jcol的索引
         const int  nseg,      /* in */                                 // 分段的数量
         float     *dense,      /* in */                                // 密集矩阵的指针
         float     *tempv,      /* working array */                     // 工作数组
         int        *segrep,  /* in */                                  // 分段的起始索引
         int        *repfnz,  /* in */                                  // 非零项的索引
         int        fpanelc,  /* in -- first column in the current panel */ // 当前面板中的第一列
         GlobalLU_t *Glu,     /* modified */                             // 修改后的全局LU结构
         SuperLUStat_t *stat  /* output */                               // 输出统计信息
         )
{

#ifdef _CRAY
    _fcd ftcs1 = _cptofcd("L", strlen("L")),                            // 字符串到_fcd转换
         ftcs2 = _cptofcd("N", strlen("N")),                            // 字符串到_fcd转换
         ftcs3 = _cptofcd("U", strlen("U"));                            // 字符串到_fcd转换
#endif
    int         incx = 1, incy = 1;                                     // x和y的增量
    float      alpha, beta;                                             // alpha和beta的值
    
    /* krep = representative of current k-th supernode
     * fsupc = first supernodal column
     * nsupc = no of columns in supernode
     * nsupr = no of rows in supernode (used as leading dimension)
     * luptr = location of supernodal LU-block in storage
     * kfnz = first nonz in the k-th supernodal segment
     * no_zeros = no of leading zeros in a supernodal U-segment
     */
    float      ukj, ukj1, ukj2;                                         // U的部分元素
    int_t        luptr, luptr1, luptr2;                                  // LU块的位置
    int          fsupc, nsupc, nsupr, segsze;                            // 超节点的相关信息
    int          nrow;      /* No of rows in the matrix of matrix-vector */ // 矩阵-向量乘法中的行数
    int          jcolp1, jsupno, k, ksub, krep, krep_ind, ksupno;        // 各种索引和计数器
    int_t        lptr, kfnz, isub, irow, i;
    int_t        no_zeros, new_next, ufirst, nextlu;                     // 各种索引和计数器
    # 第一个列在小LU更新中的起始列
    int          fst_col; /* First column within small LU update */
    # 当前面板的第一列与当前snode的第一列之间的距离
    int          d_fsupc; /* Distance between the first column of the current
                 panel and the first column of the current snode. */
    # xsup和supno指针
    int          *xsup, *supno;
    # Glu中的lsub和xlsub数组
    int_t        *lsub, *xlsub;
    # Glu中的lusup和xlusup数组，类型转换为float指针
    float       *lusup;
    int_t        *xlusup;
    # LU因子中最大非零元素数目
    int_t        nzlumax;
    # 临时向量tempv1
    float       *tempv1;
    # 浮点数常量定义：零、一、负一
    float      zero = 0.0;
    float      one = 1.0;
    float      none = -1.0;
    # 内存错误标志
    int_t        mem_error;
    # 操作统计数据指针
    flops_t      *ops = stat->ops;

    # 从Glu结构中提取相关指针和数据
    xsup    = Glu->xsup;
    supno   = Glu->supno;
    lsub    = Glu->lsub;
    xlsub   = Glu->xlsub;
    lusup   = (float *) Glu->lusup;
    xlusup  = Glu->xlusup;
    nzlumax = Glu->nzlumax;
    jcolp1 = jcol + 1;
    jsupno = supno[jcol];
    
    /* 
     * 对于U[*,j]中每个非零超节点段的拓扑排序
     */
    # 初始化循环索引
    k = nseg - 1;
    # 对于每个超节点段
    for (ksub = 0; ksub < nseg; ksub++) {

    # 获取当前超节点段的代表节点
    krep = segrep[k];
    # 递减k值
    k--;
    # 获取当前代表节点的超节点编号
    ksupno = supno[krep];
    if ( jsupno != ksupno ) { /* 如果 jsupno 不等于 ksupno，则当前在超节点矩形区域之外 */

        fsupc = xsup[ksupno];
        fst_col = SUPERLU_MAX ( fsupc, fpanelc );

          /* 当前超节点到当前面板的距离；
           如果 fsupc > fpanelc，则 d_fsupc=0。 */
          d_fsupc = fst_col - fsupc; 

        luptr = xlusup[fst_col] + d_fsupc;
        lptr = xlsub[fsupc] + d_fsupc;

        kfnz = repfnz[krep];
        kfnz = SUPERLU_MAX ( kfnz, fpanelc );

        segsze = krep - kfnz + 1;
        nsupc = krep - fst_col + 1;
        nsupr = xlsub[fsupc+1] - xlsub[fsupc];    /* 主维度 */
        nrow = nsupr - d_fsupc - nsupc;
        krep_ind = lptr + nsupc - 1;

        ops[TRSV] += segsze * (segsze - 1);  // 更新 TRSV 运算次数统计
        ops[GEMV] += 2 * nrow * segsze;      // 更新 GEMV 运算次数统计


        /* 
         * 情况 1: 更新大小为 1 的 U-段 -- 列-列更新 
         */
        if ( segsze == 1 ) {
          ukj = dense[lsub[krep_ind]];
        luptr += nsupr*(nsupc-1) + nsupc;

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

        if ( segsze == 2 ) { /* 情况 2: 2 列-列更新 */
            ukj -= ukj1 * lusup[luptr1];
            dense[lsub[krep_ind]] = ukj;
            for (i = lptr + nsupc; i < xlsub[fsupc+1]; ++i) {
                irow = lsub[i];
                luptr++;
                luptr1++;
                dense[irow] -= ( ukj*lusup[luptr]
                    + ukj1*lusup[luptr1] );
            }
        } else { /* 情况 3: 3 列-列更新 */
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
         * 情况: 超节点-列更新
         * 执行三角求解和块更新，
         * 然后将超节点-列更新的结果散布到 dense 中
         */

        no_zeros = kfnz - fst_col;

            /* 从 dense 中复制 U[*,j] 段到 tempv 中 */
            isub = lptr + no_zeros;
            for (i = 0; i < segsze; i++) {
              irow = lsub[isub];
            tempv[i] = dense[irow];
            ++isub; 
            }

            /* 密集矩阵的三角求解 -- 开始有效三角区域 */
        luptr += nsupr * no_zeros + no_zeros; 
        
#ifdef USE_VENDOR_BLAS
#ifdef _CRAY
        STRSV( ftcs1, ftcs2, ftcs3, &segsze, &lusup[luptr], 
               &nsupr, tempv, &incx );
#else        
        strsv_( "L", "N", "U", &segsze, &lusup[luptr], 
               &nsupr, tempv, &incx );
#endif        
         luptr += segsze;  /* 密集矩阵向量乘法 */
        tempv1 = &tempv[segsze];
                alpha = one;
                beta = zero;
#ifdef _CRAY
        SGEMV( ftcs2, &nrow, &segsze, &alpha, &lusup[luptr], 
               &nsupr, tempv, &incx, &beta, tempv1, &incy );
#else
        sgemv_( "N", &nrow, &segsze, &alpha, &lusup[luptr], 
               &nsupr, tempv, &incx, &beta, tempv1, &incy );
#endif
#else
        slsolve ( nsupr, segsze, &lusup[luptr], tempv );

         luptr += segsze;  /* 密集矩阵向量乘法 */
        tempv1 = &tempv[segsze];
        smatvec (nsupr, nrow , segsze, &lusup[luptr], tempv, tempv1);
#endif
        
        
                /* 将 tempv[] 散列到 SPA dense[] 作为临时存储 */
                isub = lptr + no_zeros;
                for (i = 0; i < segsze; i++) {
                    irow = lsub[isub];
                    dense[irow] = tempv[i];
                    tempv[i] = zero;
                    ++isub;
                }

        /* 将 tempv1[] 散列到 SPA dense[] */
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
    mem_error = sLUMemXpand(jcol, nextlu, LUSUP, &nzlumax, Glu);
    if (mem_error) return (mem_error);
    lusup = (float *) Glu->lusup;
    lsub = Glu->lsub;
    }

    for (isub = xlsub[fsupc]; isub < xlsub[fsupc+1]; isub++) {
      irow = lsub[isub];
    lusup[nextlu] = dense[irow];
        dense[irow] = zero;
    ++nextlu;
    }

    xlusup[jcolp1] = nextlu;    /* 结束 L\U[*,jcol] */

    /* 对于面板内的更多更新（也在当前超节点内），应从面板的第一列或超节点的第一列开始。有两种情况：
     *    1) fsupc < fpanelc，那么 fst_col := fpanelc
     *    2) fsupc >= fpanelc，那么 fst_col := fsupc
     */
    fst_col = SUPERLU_MAX ( fsupc, fpanelc );

    if ( fst_col < jcol ) {

      /* 当前超节点和当前面板之间的距离。
       如果 fsupc >= fpanelc，则 d_fsupc=0。 */
      d_fsupc = fst_col - fsupc;

    lptr = xlsub[fsupc] + d_fsupc;
    luptr = xlusup[fst_col] + d_fsupc;
    nsupr = xlsub[fsupc+1] - xlsub[fsupc];    /* 主维度 */
    nsupc = jcol - fst_col;    /* 不包括 jcol */
    nrow = nsupr - d_fsupc - nsupc;
    # 计算并指向 snode L\U(jsupno) 中 jcol 起始位置
    ufirst = xlusup[jcol] + d_fsupc;    

    # 更新 TRSV 操作计数，增加 nsupc * (nsupc - 1)
    ops[TRSV] += nsupc * (nsupc - 1);

    # 更新 GEMV 操作计数，增加 2 * nrow * nsupc
    ops[GEMV] += 2 * nrow * nsupc;
#ifdef USE_VENDOR_BLAS
#ifdef _CRAY
    STRSV( ftcs1, ftcs2, ftcs3, &nsupc, &lusup[luptr], 
           &nsupr, &lusup[ufirst], &incx );
#else
    strsv_( "L", "N", "U", &nsupc, &lusup[luptr], 
           &nsupr, &lusup[ufirst], &incx );
#endif
    
    alpha = none; beta = one; /* y := beta*y + alpha*A*x */
    
#ifdef _CRAY
    SGEMV( ftcs2, &nrow, &nsupc, &alpha, &lusup[luptr+nsupc], &nsupr,
           &lusup[ufirst], &incx, &beta, &lusup[ufirst+nsupc], &incy );
#else
    sgemv_( "N", &nrow, &nsupc, &alpha, &lusup[luptr+nsupc], &nsupr,
           &lusup[ufirst], &incx, &beta, &lusup[ufirst+nsupc], &incy );
#endif
#else
    slsolve ( nsupr, nsupc, &lusup[luptr], &lusup[ufirst] );

    smatvec ( nsupr, nrow, nsupc, &lusup[luptr+nsupc],
        &lusup[ufirst], tempv );
    
    /* Copy updates from tempv[*] into lusup[*] */
    isub = ufirst + nsupc;
    for (i = 0; i < nrow; i++) {
        lusup[isub] -= tempv[i];
        tempv[i] = 0.0;
        ++isub;
    }
#endif
```