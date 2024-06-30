# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\ccolumn_bmod.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file ccolumn_bmod.c
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
#include "slu_cdefs.h"


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
ccolumn_bmod (
         const int  jcol,      /* in */                                // 输入参数：当前列
         const int  nseg,      /* in */                                // 输入参数：分段数目
         singlecomplex     *dense,      /* in */                        // 输入参数：密集矩阵
         singlecomplex     *tempv,      /* working array */            // 工作数组：临时向量
         int        *segrep,  /* in */                                // 输入参数：分段的代表
         int        *repfnz,  /* in */                                // 输入参数：分段的第一个非零元索引
         int        fpanelc,  /* in -- first column in the current panel */  // 输入参数：当前面板中的第一列
         GlobalLU_t *Glu,     /* modified */                           // 修改参数：全局LU结构体
         SuperLUStat_t *stat  /* output */                             // 输出参数：统计信息
         )
{

#ifdef _CRAY
    _fcd ftcs1 = _cptofcd("L", strlen("L")),                           // CRAY 平台下的字符转换
         ftcs2 = _cptofcd("N", strlen("N")),
         ftcs3 = _cptofcd("U", strlen("U"));
#endif
    int         incx = 1, incy = 1;                                    // 向量的步长
    singlecomplex      alpha, beta;                                    // 数值常量
    
    /* krep = representative of current k-th supernode
     * fsupc = first supernodal column
     * nsupc = no of columns in supernode
     * nsupr = no of rows in supernode (used as leading dimension)
     * luptr = location of supernodal LU-block in storage
     * kfnz = first nonz in the k-th supernodal segment
     * no_zeros = no of leading zeros in a supernodal U-segment
     */
    singlecomplex      ukj, ukj1, ukj2;                                // 临时变量
    int_t        luptr, luptr1, luptr2;                                // LU 分解相关指针
    int          fsupc, nsupc, nsupr, segsze;                          // 超节点相关参数
    int          nrow;      /* No of rows in the matrix of matrix-vector */  // 矩阵向量乘法的行数
    int          jcolp1, jsupno, k, ksub, krep, krep_ind, ksupno;       // 循环索引和超节点索引
    int_t        lptr, kfnz, isub, irow, i;                            // 循环变量和子列索引
    # 定义整型变量：no_zeros, new_next, ufirst, nextlu
    # fst_col：当前小LU更新中的第一列
    # d_fsupc：当前面板的第一列与当前snodes的第一列之间的距离
    int_t        no_zeros, new_next, ufirst, nextlu;
    int          fst_col; /* First column within small LU update */
    int          d_fsupc; /* Distance between the first column of the current
                 panel and the first column of the current snode. */
    int          *xsup, *supno;
    int_t        *lsub, *xlsub;
    singlecomplex       *lusup;
    int_t        *xlusup;
    int_t        nzlumax;
    singlecomplex       *tempv1;
    
    # 定义复数常量：zero为复数0, one为复数1, none为复数-1
    singlecomplex      zero = {0.0, 0.0};
    singlecomplex      one = {1.0, 0.0};
    singlecomplex      none = {-1.0, 0.0};
    singlecomplex     comp_temp, comp_temp1;
    int_t        mem_error;
    
    # 从状态统计结构体中获取操作数指针
    flops_t      *ops = stat->ops;

    # 将全局LU结构体中的数组赋值给局部变量
    xsup    = Glu->xsup;
    supno   = Glu->supno;
    lsub    = Glu->lsub;
    xlsub   = Glu->xlsub;
    lusup   = (singlecomplex *) Glu->lusup;
    xlusup  = Glu->xlusup;
    nzlumax = Glu->nzlumax;
    
    # 初始化jcolp1为jcol的下一个列索引，初始化jsupno为supno数组中索引为jcol的值
    jcolp1 = jcol + 1;
    jsupno = supno[jcol];
    
    /* 
     * 对于U[*,j]中每个非零超节点段，按拓扑顺序处理
     */
    k = nseg - 1;
    for (ksub = 0; ksub < nseg; ksub++) {
    
        # 从segrep数组中获取当前超节点段的代表行
        krep = segrep[k];
        k--;
        
        # 获取ksupno为当前代表行krep在supno数组中的值
        ksupno = supno[krep];
#ifdef USE_VENDOR_BLAS
#ifdef _CRAY
        CTRSV( ftcs1, ftcs2, ftcs3, &segsze, &lusup[luptr], 
               &nsupr, tempv, &incx );
#else        
        // 使用BLAS库中的CTRSV函数求解线性方程组，解释见下方注释
        ctrsv_( "L", "N", "U", &segsze, &lusup[luptr], 
               &nsupr, tempv, &incx );
#endif        
         luptr += segsze;  /* Dense matrix-vector */
        tempv1 = &tempv[segsze];
                alpha = one;
                beta = zero;
#ifdef _CRAY
        CGEMV( ftcs2, &nrow, &segsze, &alpha, &lusup[luptr], 
               &nsupr, tempv, &incx, &beta, tempv1, &incy );
#else
        // 使用BLAS库中的CGEMV函数进行矩阵向量乘法，解释见下方注释
        cgemv_( "N", &nrow, &segsze, &alpha, &lusup[luptr], 
               &nsupr, tempv, &incx, &beta, tempv1, &incy );
#endif
#else
        // 当未定义USE_VENDOR_BLAS时，使用自定义函数解决线性方程组和进行矩阵向量乘法
        clsolve ( nsupr, segsze, &lusup[luptr], tempv );

         luptr += segsze;  /* Dense matrix-vector */
        tempv1 = &tempv[segsze];
        cmatvec (nsupr, nrow , segsze, &lusup[luptr], tempv, tempv1);
#endif
        
        
                /* Scatter tempv[] into SPA dense[] as a temporary storage */
                // 将tempv[]的值散布到SPA dense[]数组中作为临时存储
                isub = lptr + no_zeros;
                for (i = 0; i < segsze; i++) {
                    irow = lsub[isub];
                    dense[irow] = tempv[i];
                    tempv[i] = zero;
                    ++isub;
                }

        /* Scatter tempv1[] into SPA dense[] */
        // 将tempv1[]的值散布到SPA dense[]数组中
        for (i = 0; i < nrow; i++) {
            irow = lsub[isub];
            c_sub(&dense[irow], &dense[irow], &tempv1[i]);
            tempv1[i] = zero;
            ++isub;
        }
        }
        
    } /* if jsupno ... */

    } /* for each segment... */

    /*
     *    Process the supernodal portion of L\U[*,j]
     */
    nextlu = xlusup[jcol];
    fsupc = xsup[jsupno];

    /* Copy the SPA dense into L\U[*,j] */
    // 将SPA dense数组复制到L\U[*,j]中
    new_next = nextlu + xlsub[fsupc+1] - xlsub[fsupc];
    while ( new_next > nzlumax ) {
    // 如果空间不足，则扩展内存
    mem_error = cLUMemXpand(jcol, nextlu, LUSUP, &nzlumax, Glu);
    if (mem_error) return (mem_error);
    lusup = (singlecomplex *) Glu->lusup;
    lsub = Glu->lsub;
    }

    for (isub = xlsub[fsupc]; isub < xlsub[fsupc+1]; isub++) {
      irow = lsub[isub];
    lusup[nextlu] = dense[irow];
        dense[irow] = zero;
    ++nextlu;
    }

    xlusup[jcolp1] = nextlu;    /* Close L\U[*,jcol] */

    /* For more updates within the panel (also within the current supernode), 
     * should start from the first column of the panel, or the first column 
     * of the supernode, whichever is bigger. There are 2 cases:
     *    1) fsupc < fpanelc, then fst_col := fpanelc
     *    2) fsupc >= fpanelc, then fst_col := fsupc
     */
    // 确定更新范围的起始列号
    fst_col = SUPERLU_MAX ( fsupc, fpanelc );

    if ( fst_col < jcol ) {

      /* Distance between the current supernode and the current panel.
       d_fsupc=0 if fsupc >= fpanelc. */
      // 当前超节点和当前面板之间的距离
      d_fsupc = fst_col - fsupc;

    lptr = xlsub[fsupc] + d_fsupc;
    luptr = xlusup[fst_col] + d_fsupc;
    nsupr = xlsub[fsupc+1] - xlsub[fsupc];    /* Leading dimension */
    nsupc = jcol - fst_col;    /* Excluding jcol */
    # 计算矩阵的行数，即非零元素总数减去第一列之前的非零元素数和列数的差
    nrow = nsupr - d_fsupc - nsupc;

    /* 指向snode L\U(jsupno)中jcol的起始位置 */
    ufirst = xlusup[jcol] + d_fsupc;    

    # 更新TRSV操作的计数器，计算量与nsupc相关
    ops[TRSV] += 4 * nsupc * (nsupc - 1);
    # 更新GEMV操作的计数器，计算量与nrow和nsupc相关
    ops[GEMV] += 8 * nrow * nsupc;
#ifdef USE_VENDOR_BLAS
#ifdef _CRAY
    CTRSV( ftcs1, ftcs2, ftcs3, &nsupc, &lusup[luptr], 
           &nsupr, &lusup[ufirst], &incx );
#else
    ctrsv_( "L", "N", "U", &nsupc, &lusup[luptr], 
           &nsupr, &lusup[ufirst], &incx );
#endif
    
    alpha = none; beta = one; /* y := beta*y + alpha*A*x */
    
#ifdef _CRAY
    CGEMV( ftcs2, &nrow, &nsupc, &alpha, &lusup[luptr+nsupc], &nsupr,
           &lusup[ufirst], &incx, &beta, &lusup[ufirst+nsupc], &incy );
#else
    cgemv_( "N", &nrow, &nsupc, &alpha, &lusup[luptr+nsupc], &nsupr,
           &lusup[ufirst], &incx, &beta, &lusup[ufirst+nsupc], &incy );
#endif
#else
    clsolve ( nsupr, nsupc, &lusup[luptr], &lusup[ufirst] );
    
    cmatvec ( nsupr, nrow, nsupc, &lusup[luptr+nsupc],
        &lusup[ufirst], tempv );
    
    /* Copy updates from tempv[*] into lusup[*] */
    isub = ufirst + nsupc;
    for (i = 0; i < nrow; i++) {
        c_sub(&lusup[isub], &lusup[isub], &tempv[i]);
        tempv[i] = zero;
        ++isub;
    }
#endif
```