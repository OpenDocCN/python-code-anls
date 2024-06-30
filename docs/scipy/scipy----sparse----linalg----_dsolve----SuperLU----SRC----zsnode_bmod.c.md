# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\zsnode_bmod.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file zsnode_bmod.c
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


#include "slu_zdefs.h"


/*! \brief Performs numeric block updates within the relaxed snode. 
 */
int
zsnode_bmod (
        const int  jcol,      /* in */                                     // 列号
        const int  jsupno,    /* in */                                     // 超节点号
        const int  fsupc,     /* in */                                     // 起始列号
        doublecomplex     *dense,    /* in */                              // 密集矩阵
        doublecomplex     *tempv,    /* working array */                   // 工作数组
        GlobalLU_t *Glu,      /* modified */                               // 全局 LU 结构体
        SuperLUStat_t *stat   /* output */                                 // 统计信息
        )
{
#ifdef USE_VENDOR_BLAS
#ifdef _CRAY
    _fcd ftcs1 = _cptofcd("L", strlen("L")),                               // Fortran 字符串描述符
     ftcs2 = _cptofcd("N", strlen("N")),                                    // Fortran 字符串描述符
     ftcs3 = _cptofcd("U", strlen("U"));                                    // Fortran 字符串描述符
#endif
    int            incx = 1, incy = 1;                                      // 向量增量
    doublecomplex         alpha = {-1.0, 0.0},  beta = {1.0, 0.0};           // BLAS 中使用的标量
#endif

    doublecomplex   comp_zero = {0.0, 0.0};                                  // 复数零
    int     nsupc, nsupr, nrow;                                              // 超节点列数、行数
    int_t   isub, irow;                                                      // 列表索引、行索引
    int_t   ufirst, nextlu;                                                  // 起始索引、下一个 LU 位置
    int_t   *lsub, *xlsub;                                                   // 列表、列索引数组
    doublecomplex *lusup;                                                    // LU 超节点
    int_t   *xlusup, luptr;                                                  // 超节点索引、LU 指针
    flops_t *ops = stat->ops;                                                // FLOPS 统计

    lsub    = Glu->lsub;                                                     // 获取 LU 结构体的列列表
    xlsub   = Glu->xlsub;                                                    // 获取 LU 结构体的列索引数组
    lusup   = (doublecomplex *) Glu->lusup;                                  // 获取 LU 结构体的 LU 超节点
    xlusup  = Glu->xlusup;                                                   // 获取 LU 结构体的 LU 超节点索引

    nextlu = xlusup[jcol];                                                   // 下一个 LU 位置初始化为当前列的起始位置
    
    /*
     *    Process the supernodal portion of L\U[*,j]
     */
    for (isub = xlsub[fsupc]; isub < xlsub[fsupc+1]; isub++) {               // 遍历当前列的非零元素
      irow = lsub[isub];                                                     // 获取行索引
      lusup[nextlu] = dense[irow];                                           // 更新 LU 超节点元素
        dense[irow] = comp_zero;                                             // 将原始矩阵中对应位置置为零
    ++nextlu;                                                                 // 更新下一个 LU 位置
    }

    xlusup[jcol + 1] = nextlu;    /* Initialize xlusup for next column */    // 更新下一列的起始 LU 位置
    
    if ( fsupc < jcol ) {

    luptr = xlusup[fsupc];                                                   // 获取 LU 指针起始位置
    nsupr = xlsub[fsupc+1] - xlsub[fsupc];                                    // 当前超节点行数
    nsupc = jcol - fsupc;    /* Excluding jcol */                             // 当前超节点列数（不包括 jcol）
    ufirst = xlusup[jcol];    /* Points to the beginning of column
                   jcol in supernode L\U(jsupno). */                          // 指向超节点 L\U(jsupno) 中列 jcol 的起始位置
    nrow = nsupr - nsupc;                                                     // 非超节点行数

    ops[TRSV] += 4 * nsupc * (nsupc - 1);                                     // 更新 TRSV 操作数统计
    ops[GEMV] += 8 * nrow * nsupc;                                            // 更新 GEMV 操作数统计

#ifdef USE_VENDOR_BLAS
#ifdef _CRAY
    # 调用名为 CTRSV 的函数，进行特定线性代数计算，更新 ftcs1, ftcs2, ftcs3
    # nsupc 为输入参数，表示列数
    # lusup[luptr] 表示 LU 分解中 L 和 U 的组合，根据 luptr 指针索引
    # nsupr 表示行数，作为输入参数
    CTRSV( ftcs1, ftcs2, ftcs3, &nsupc, &lusup[luptr], &nsupr, 
          &lusup[ufirst], &incx );

    # 调用名为 CGEMV 的函数，进行特定矩阵-向量乘法
    # ftcs2 为输入参数，表示运算类型
    # nrow 表示矩阵行数，作为输入参数
    # nsupc 表示列数，作为输入参数
    # alpha 表示乘法运算中的系数，beta 表示加法运算中的系数
    # lusup[luptr+nsupc] 表示矩阵中特定位置的数据，根据 luptr 指针索引
    # lusup[ufirst] 表示矩阵中另一个位置的数据，根据 ufirst 索引
    # incx 表示向量 x 的步长
    # lusup[ufirst+nsupc] 表示输出向量的存储位置，根据 ufirst 索引
    # incy 表示向量 y 的步长
    CGEMV( ftcs2, &nrow, &nsupc, &alpha, &lusup[luptr+nsupc], &nsupr, 
        &lusup[ufirst], &incx, &beta, &lusup[ufirst+nsupc], &incy );
#else
#if SCIPY_FIX
       if (nsupr < nsupc) {
           /* 如果非零元素行数少于列数，直接失败，避免向 TRSV 传递无效参数 */
           ABORT("failed to factorize matrix");
       }
#endif
    // 解方程 ztrsv 函数调用
    ztrsv_( "L", "N", "U", &nsupc, &lusup[luptr], &nsupr, 
          &lusup[ufirst], &incx );
    // 矩阵-向量乘法 zgemv 函数调用
    zgemv_( "N", &nrow, &nsupc, &alpha, &lusup[luptr+nsupc], &nsupr, 
        &lusup[ufirst], &incx, &beta, &lusup[ufirst+nsupc], &incy );
#endif
#else
    // 低三角矩阵求解 zlsolve 函数调用
    zlsolve ( nsupr, nsupc, &lusup[luptr], &lusup[ufirst] );
    // 矩阵-向量乘法 zmatvec 函数调用
    zmatvec ( nsupr, nrow, nsupc, &lusup[luptr+nsupc], 
            &lusup[ufirst], &tempv[0] );

    int_t i, iptr; 
        /* 将 tempv[*] 散射到 lusup[*] */
    iptr = ufirst + nsupc;
    for (i = 0; i < nrow; i++) {
        z_sub(&lusup[iptr], &lusup[iptr], &tempv[i]);
            ++iptr;
        tempv[i] = comp_zero;
    }
#endif

    }

    // 函数执行成功，返回 0
    return 0;
}
```