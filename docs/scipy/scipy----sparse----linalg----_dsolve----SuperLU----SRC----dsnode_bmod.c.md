# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\dsnode_bmod.c`

```
/*! @file dsnode_bmod.c
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


#include "slu_ddefs.h"


/*! \brief Performs numeric block updates within the relaxed snode. 
 */
int
dsnode_bmod (
        const int  jcol,      /* in */    // 列指标 jcol
        const int  jsupno,    /* in */    // 超节点编号 jsupno
        const int  fsupc,     /* in */    // 第一个超节点列的编号 fsupc
        double     *dense,    /* in */    // 密集矩阵的列 dense
        double     *tempv,    /* working array */  // 工作数组 tempv
        GlobalLU_t *Glu,      /* modified */  // 全局 LU 分解结构 Glu
        SuperLUStat_t *stat   /* output */     // 统计信息结构 stat
        )
{
#ifdef USE_VENDOR_BLAS
#ifdef _CRAY
    _fcd ftcs1 = _cptofcd("L", strlen("L")),    // FORTRAN 字符串描述符
     ftcs2 = _cptofcd("N", strlen("N")),        // FORTRAN 字符串描述符
     ftcs3 = _cptofcd("U", strlen("U"));        // FORTRAN 字符串描述符
#endif
    int            incx = 1, incy = 1;          // BLAS 函数中的增量值
    double         alpha = -1.0, beta = 1.0;    // BLAS 函数中的常数
#endif

    int     nsupc, nsupr, nrow;                  // 整数变量：超节点列数、超节点行数、行数
    int_t   isub, irow;                          // 整数变量：子列索引、行索引
    int_t   ufirst, nextlu;                      // 整数变量：超节点起始位置、下一个位置
    int_t   *lsub, *xlsub;                       // 整数数组：非零元素的行索引、行索引指针
    double *lusup;                               // 双精度数组：超节点 L 上的非零元素
    int_t   *xlusup, luptr;                      // 整数数组：超节点 L 上的非零元素指针、超节点起始指针
    flops_t *ops = stat->ops;                    // 浮点运算统计：操作数

    lsub    = Glu->lsub;                         // 获取全局 LU 结构中的 lsub 数组
    xlsub   = Glu->xlsub;                        // 获取全局 LU 结构中的 xlsub 数组
    lusup   = (double *) Glu->lusup;             // 获取全局 LU 结构中的 lusup 数组
    xlusup  = Glu->xlusup;                       // 获取全局 LU 结构中的 xlusup 数组

    nextlu = xlusup[jcol];                       // 下一个超节点位置初始化为 jcol 列的起始位置
    
    /*
     *    Process the supernodal portion of L\U[*,j]
     */
    for (isub = xlsub[fsupc]; isub < xlsub[fsupc+1]; isub++) {
      irow = lsub[isub];                         // 获取当前子列对应的行索引
    lusup[nextlu] = dense[irow];                 // 将 dense 中的值复制到 L 上
    dense[irow] = 0;                             // 将 dense 中的值置零
    ++nextlu;                                     // 移动到下一个位置
    }

    xlusup[jcol + 1] = nextlu;                    // 初始化 xlusup 数组，为下一列

    if ( fsupc < jcol ) {

    luptr = xlusup[fsupc];                       // 获取超节点 L 的起始位置
    nsupr = xlsub[fsupc+1] - xlsub[fsupc];        // 计算超节点的行数
    nsupc = jcol - fsupc;                        // 计算超节点的列数（不包括 jcol）
    ufirst = xlusup[jcol];                       // 获取超节点 U 的起始位置
    nrow = nsupr - nsupc;                        // 计算需要处理的行数

    ops[TRSV] += nsupc * (nsupc - 1);             // 统计向量三角求解操作次数
    ops[GEMV] += 2 * nrow * nsupc;                // 统计一般矩阵-向量乘法操作次数

#ifdef USE_VENDOR_BLAS
#ifdef _CRAY
    STRSV( ftcs1, ftcs2, ftcs3, &nsupc, &lusup[luptr], &nsupr, 
          &lusup[ufirst], &incx );               // 使用 BLAS 的 TRSV 操作进行向量三角求解
    SGEMV( ftcs2, &nrow, &nsupc, &alpha, &lusup[luptr+nsupc], &nsupr, 
          &lusup[ufirst], &incx, &beta, &lusup[ufirst+nsupc], &incy );



    // 调用 SGEMV 函数进行矩阵向量乘法操作
    // ftcs2: SGEMV 函数的参数，可能是某种配置或标志
    // &nrow: 矩阵的行数，作为 SGEMV 函数的参数
    // &nsupc: 列数或矩阵的非零元素个数，作为 SGEMV 函数的参数
    // &alpha: 乘法的标量系数，作为 SGEMV 函数的参数
    // &lusup[luptr+nsupc]: 指向矩阵数据的指针，作为 SGEMV 函数的参数
    // &nsupr: 矩阵的行数，作为 SGEMV 函数的参数
    // &lusup[ufirst]: 指向矩阵数据的指针，作为 SGEMV 函数的参数
    // &incx: 向量 x 的增量，作为 SGEMV 函数的参数
    // &beta: 乘法的标量系数，作为 SGEMV 函数的参数
    // &lusup[ufirst+nsupc]: 指向矩阵数据的指针，作为 SGEMV 函数的参数
    // &incy: 向量 y 的增量，作为 SGEMV 函数的参数
#else
#if SCIPY_FIX
       // 如果 nsupr 小于 nsupc，则发生矩阵分解失败，提前终止程序
       if (nsupr < nsupc) {
           ABORT("failed to factorize matrix");
       }
#endif
    // 使用 BLAS 函数 dtrsv 对下三角矩阵进行向前/向后替换操作
    dtrsv_( "L", "N", "U", &nsupc, &lusup[luptr], &nsupr, 
          &lusup[ufirst], &incx );
    // 使用 BLAS 函数 dgemv 对矩阵进行一般矩阵-向量乘法操作
    dgemv_( "N", &nrow, &nsupc, &alpha, &lusup[luptr+nsupc], &nsupr, 
        &lusup[ufirst], &incx, &beta, &lusup[ufirst+nsupc], &incy );
#endif
#else
    // 调用自定义函数 dlsolve 进行下三角矩阵的求解操作
    dlsolve ( nsupr, nsupc, &lusup[luptr], &lusup[ufirst] );
    // 调用自定义函数 dmatvec 进行矩阵与向量乘法操作
    dmatvec ( nsupr, nrow, nsupc, &lusup[luptr+nsupc], 
            &lusup[ufirst], &tempv[0] );

    int_t i, iptr; 
        /* Scatter tempv[*] into lusup[*] */
    // 将 tempv[*] 散布到 lusup[*] 中
    iptr = ufirst + nsupc;
    for (i = 0; i < nrow; i++) {
        lusup[iptr++] -= tempv[i];
        tempv[i] = 0.0;
    }
#endif

    }

    // 函数执行成功，返回 0
    return 0;
}
```