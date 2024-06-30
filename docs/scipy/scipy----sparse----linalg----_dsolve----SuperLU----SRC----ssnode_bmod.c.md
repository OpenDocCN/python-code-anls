# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\ssnode_bmod.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file ssnode_bmod.c
 * \brief Performs numeric block updates within the relaxed snode.
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
 * Permission is hereby granted to use or copy this program for any
 * purpose, provided the above notices are retained on all copies.
 * Permission to modify the code and to distribute modified code is
 * granted, provided the above notices are retained, and a notice that
 * the code was modified is included with the above copyright notice.
 * </pre>
 */


#include "slu_sdefs.h"


/*! \brief Performs numeric block updates within the relaxed snode. 
 */
int
ssnode_bmod (
        const int  jcol,      /* in */                             // 列索引，指定要更新的列
        const int  jsupno,    /* in */                             // 超节点编号，指定所属的超节点
        const int  fsupc,     /* in */                             // 第一个非零行的全局列索引
        float     *dense,    /* in */                             // 密集矩阵的列数据
        float     *tempv,    /* working array */                  // 工作数组，用于中间计算
        GlobalLU_t *Glu,      /* modified */                       // 全局 LU 因子数据结构
        SuperLUStat_t *stat   /* output */                         // 超节点统计信息
        )
{
#ifdef USE_VENDOR_BLAS
#ifdef _CRAY
    _fcd ftcs1 = _cptofcd("L", strlen("L")),                      // BLAS 函数参数：指定操作的三角部分为下三角
     ftcs2 = _cptofcd("N", strlen("N")),                          // BLAS 函数参数：指定不进行转置
     ftcs3 = _cptofcd("U", strlen("U"));                          // BLAS 函数参数：指定操作的三角部分为上三角
#endif
    int            incx = 1, incy = 1;                            // BLAS 函数参数：向量增量
    float         alpha = -1.0, beta = 1.0;                       // BLAS 函数参数：矩阵乘法的标量因子
#endif

    int     nsupc, nsupr, nrow;
    int_t   isub, irow;
    int_t   ufirst, nextlu;
    int_t   *lsub, *xlsub;
    float *lusup;
    int_t   *xlusup, luptr;
    flops_t *ops = stat->ops;                                     // 操作计数结构体中的操作计数指针

    lsub    = Glu->lsub;                                          // LU 因子中的非零元素行索引数组
    xlsub   = Glu->xlsub;                                         // LU 因子中每一列起始位置索引数组
    lusup   = (float *) Glu->lusup;                                // LU 因子中的非零元素值数组
    xlusup  = Glu->xlusup;                                         // LU 因子中每一列起始位置索引数组

    nextlu = xlusup[jcol];                                         // 下一列的起始位置索引

    /*
     *    Process the supernodal portion of L\U[*,j]
     */
    for (isub = xlsub[fsupc]; isub < xlsub[fsupc+1]; isub++) {      // 遍历超节点中的每一行
      irow = lsub[isub];                                            // 获取当前行索引
    lusup[nextlu] = dense[irow];                                    // 更新 LU 因子中的值
    dense[irow] = 0;                                                // 清空输入矩阵中的值
    ++nextlu;                                                       // 更新下一个位置的索引
    }

    xlusup[jcol + 1] = nextlu;                                      // 更新下一列的起始位置索引

    if ( fsupc < jcol ) {

    luptr = xlusup[fsupc];                                          // LU 因子中第一个非零元素的索引
    nsupr = xlsub[fsupc+1] - xlsub[fsupc];                           // 超节点行数
    nsupc = jcol - fsupc;                                           // 超节点列数（不包括 jcol）
    ufirst = xlusup[jcol];                                          // 当前列在 LU 因子中的起始位置索引
    nrow = nsupr - nsupc;                                           // 非零行数

    ops[TRSV] += nsupc * (nsupc - 1);                                // 更新操作统计信息：TRSV 操作次数
    ops[GEMV] += 2 * nrow * nsupc;                                   // 更新操作统计信息：GEMV 操作次数

#ifdef USE_VENDOR_BLAS
#ifdef _CRAY
    STRSV( ftcs1, ftcs2, ftcs3, &nsupc, &lusup[luptr], &nsupr, 
          &lusup[ufirst], &incx );                                  // 使用 BLAS 函数进行向量三角求解操作
    // 调用 SGEMV 函数，执行稠密矩阵向量乘法操作
    SGEMV(ftcs2,            // 稠密矩阵乘向量的操作类型或标志
          &nrow,            // 矩阵行数
          &nsupc,           // 超节点列数
          &alpha,           // 乘法操作中的 alpha 系数
          &lusup[luptr+nsupc],  // LU 因子中存储的超节点数据，偏移为 luptr+nsupc
          &nsupr,           // LU 因子的行数
          &lusup[ufirst],   // LU 因子的第一列数据，偏移为 ufirst
          &incx,            // X 向量的增量
          &beta,            // 乘法操作中的 beta 系数
          &lusup[ufirst+nsupc],  // 结果向量的存储位置，偏移为 ufirst+nsupc
          &incy);           // Y 向量的增量
#else
#if SCIPY_FIX
       if (nsupr < nsupc) {
           /* 提前失败以避免将无效参数传递给 TRSV。 */
           ABORT("failed to factorize matrix");
       }
#endif
    // 调用 BLAS 库中的 strsv 函数进行三角矩阵求解
    strsv_( "L", "N", "U", &nsupc, &lusup[luptr], &nsupr, 
          &lusup[ufirst], &incx );
    // 调用 BLAS 库中的 sgemv 函数进行矩阵向量乘法
    sgemv_( "N", &nrow, &nsupc, &alpha, &lusup[luptr+nsupc], &nsupr, 
        &lusup[ufirst], &incx, &beta, &lusup[ufirst+nsupc], &incy );
#endif
#else
    // 调用 slsolve 函数求解线性方程组
    slsolve ( nsupr, nsupc, &lusup[luptr], &lusup[ufirst] );
    // 调用 smatvec 函数进行矩阵向量乘法
    smatvec ( nsupr, nrow, nsupc, &lusup[luptr+nsupc], 
            &lusup[ufirst], &tempv[0] );

    int_t i, iptr; 
        /* Scatter tempv[*] into lusup[*] */
    // 将 tempv 数组的值散布到 lusup 数组中
    iptr = ufirst + nsupc;
    for (i = 0; i < nrow; i++) {
        lusup[iptr++] -= tempv[i];
        tempv[i] = 0.0;
    }
#endif

    }

    // 函数返回值为 0，表示执行成功
    return 0;
}
```