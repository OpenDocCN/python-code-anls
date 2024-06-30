# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\zcolumn_bmod.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file zcolumn_bmod.c
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
#include "slu_zdefs.h"


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
zcolumn_bmod (
         const int  jcol,      /* in */                              // 输入参数，列索引
         const int  nseg,      /* in */                              // 输入参数，段数
         doublecomplex     *dense,      /* in */                     // 输入参数，密集矩阵
         doublecomplex     *tempv,      /* working array */          // 工作数组
         int        *segrep,  /* in */                              // 输入参数，段的代表点
         int        *repfnz,  /* in */                              // 输入参数，非零值的代表点
         int        fpanelc,  /* in -- first column in the current panel */  // 输入参数，当前面板中的第一列
         GlobalLU_t *Glu,     /* modified */                         // 修改参数，全局 LU 数据结构
         SuperLUStat_t *stat  /* output */                           // 输出参数，统计信息
         )
{

#ifdef _CRAY
    _fcd ftcs1 = _cptofcd("L", strlen("L")),                         // 定义 _fcd 类型，将 "L" 转换为 _fcd 类型
         ftcs2 = _cptofcd("N", strlen("N")),                         // 定义 _fcd 类型，将 "N" 转换为 _fcd 类型
         ftcs3 = _cptofcd("U", strlen("U"));                         // 定义 _fcd 类型，将 "U" 转换为 _fcd 类型
#endif
    int         incx = 1, incy = 1;                                  // 定义整型变量 incx 和 incy，初始化为 1
    doublecomplex      alpha, beta;                                  // 定义复数变量 alpha 和 beta
    
    /* krep = representative of current k-th supernode
     * fsupc = first supernodal column
     * nsupc = no of columns in supernode
     * nsupr = no of rows in supernode (used as leading dimension)
     * luptr = location of supernodal LU-block in storage
     * kfnz = first nonz in the k-th supernodal segment
     * no_zeros = no of leading zeros in a supernodal U-segment
     */
    doublecomplex      ukj, ukj1, ukj2;                              // 定义复数变量 ukj, ukj1, ukj2
    int_t        luptr, luptr1, luptr2;                              // 定义整型长整型变量 luptr, luptr1, luptr2
    int          fsupc, nsupc, nsupr, segsze;                        // 定义整型变量 fsupc, nsupc, nsupr, segsze
    int          nrow;      /* No of rows in the matrix of matrix-vector */  // 矩阵-向量乘法中的行数
    int          jcolp1, jsupno, k, ksub, krep, krep_ind, ksupno;     // 定义整型变量 jcolp1, jsupno, k, ksub, krep, krep_ind, ksupno
    int_t        lptr, kfnz, isub, irow, i;                          // 定义长整型整型变量 lptr, kfnz, isub, irow, i
    int_t        no_zeros, new_next, ufirst, nextlu;
    // 定义整型变量：no_zeros, new_next, ufirst, nextlu

    int          fst_col; /* First column within small LU update */
    // 定义整型变量：fst_col，表示小LU更新中的第一列

    int          d_fsupc; /* Distance between the first column of the current
                 panel and the first column of the current snode. */
    // 定义整型变量：d_fsupc，表示当前面板的第一列与当前超节点第一列之间的距离

    int          *xsup, *supno;
    // 定义整型指针：xsup 和 supno

    int_t        *lsub, *xlsub;
    // 定义整型指针：lsub 和 xlsub

    doublecomplex       *lusup;
    // 定义复数类型指针：lusup

    int_t        *xlusup;
    // 定义整型指针：xlusup

    int_t        nzlumax;
    // 定义整型变量：nzlumax，表示LU分解中非零元的最大数量

    doublecomplex       *tempv1;
    // 定义复数类型指针：tempv1

    doublecomplex      zero = {0.0, 0.0};
    // 初始化复数类型变量 zero，值为 (0.0, 0.0)

    doublecomplex      one = {1.0, 0.0};
    // 初始化复数类型变量 one，值为 (1.0, 0.0)

    doublecomplex      none = {-1.0, 0.0};
    // 初始化复数类型变量 none，值为 (-1.0, 0.0)

    doublecomplex     comp_temp, comp_temp1;
    // 定义复数类型变量：comp_temp 和 comp_temp1

    int_t        mem_error;
    // 定义整型变量：mem_error，表示内存错误状态

    flops_t      *ops = stat->ops;
    // 定义 flops_t 类型指针 ops，指向 stat 结构体中的 ops 成员

    xsup    = Glu->xsup;
    // 将 Glu 结构体中的 xsup 成员赋给 xsup 指针

    supno   = Glu->supno;
    // 将 Glu 结构体中的 supno 成员赋给 supno 指针

    lsub    = Glu->lsub;
    // 将 Glu 结构体中的 lsub 成员赋给 lsub 指针

    xlsub   = Glu->xlsub;
    // 将 Glu 结构体中的 xlsub 成员赋给 xlsub 指针

    lusup   = (doublecomplex *) Glu->lusup;
    // 将 Glu 结构体中的 lusup 成员转换为 doublecomplex 类型指针，并赋给 lusup 指针

    xlusup  = Glu->xlusup;
    // 将 Glu 结构体中的 xlusup 成员赋给 xlusup 指针

    nzlumax = Glu->nzlumax;
    // 将 Glu 结构体中的 nzlumax 成员赋给 nzlumax 变量

    jcolp1 = jcol + 1;
    // 计算 jcol 的下一个值，赋给 jcolp1 变量

    jsupno = supno[jcol];
    // 获取 supno 数组中索引为 jcol 的值，赋给 jsupno 变量

    /* 
     * For each nonz supernode segment of U[*,j] in topological order 
     */
    // 对于 U[*,j] 的每个非零超节点段，按拓扑顺序处理

    k = nseg - 1;
    // 初始化 k 变量为 nseg - 1

    for (ksub = 0; ksub < nseg; ksub++) {
    // 循环遍历每个超节点段

    krep = segrep[k];
    // 获取 segrep 数组中索引为 k 的值，赋给 krep 变量
    k--;
    // k 自减 1

    ksupno = supno[krep];
    // 获取 supno 数组中索引为 krep 的值，赋给 ksupno 变量
#ifdef USE_VENDOR_BLAS
#ifdef _CRAY
        CTRSV( ftcs1, ftcs2, ftcs3, &segsze, &lusup[luptr], 
               &nsupr, tempv, &incx );
#else        
        ztrsv_( "L", "N", "U", &segsze, &lusup[luptr], 
               &nsupr, tempv, &incx );
#endif        
         luptr += segsze;  /* 密集矩阵-向量运算 */

        tempv1 = &tempv[segsze];
                alpha = one;
                beta = zero;
#ifdef _CRAY
        CGEMV( ftcs2, &nrow, &segsze, &alpha, &lusup[luptr], 
               &nsupr, tempv, &incx, &beta, tempv1, &incy );
#else
        zgemv_( "N", &nrow, &segsze, &alpha, &lusup[luptr], 
               &nsupr, tempv, &incx, &beta, tempv1, &incy );
#endif
#else
        zlsolve ( nsupr, segsze, &lusup[luptr], tempv );

         luptr += segsze;  /* 密集矩阵-向量运算 */
        tempv1 = &tempv[segsze];
        zmatvec (nsupr, nrow , segsze, &lusup[luptr], tempv, tempv1);
#endif
        
                /* 将 tempv[] 散布到 SPA dense[] 中作为临时存储 */
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
            z_sub(&dense[irow], &dense[irow], &tempv1[i]);
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
    mem_error = zLUMemXpand(jcol, nextlu, LUSUP, &nzlumax, Glu);
    if (mem_error) return (mem_error);
    lusup = (doublecomplex *) Glu->lusup;
    lsub = Glu->lsub;
    }

    for (isub = xlsub[fsupc]; isub < xlsub[fsupc+1]; isub++) {
      irow = lsub[isub];
    lusup[nextlu] = dense[irow];
        dense[irow] = zero;
    ++nextlu;
    }

    xlusup[jcolp1] = nextlu;    /* 完成 L\U[*,jcol] */

    /* 对于面板内的更多更新（也包括当前超节点内的更新），应从面板的第一列或超节点的第一列开始，取两者中较大者。有两种情况：
     *    1) fsupc < fpanelc，则 fst_col := fpanelc
     *    2) fsupc >= fpanelc，则 fst_col := fsupc
     */
    fst_col = SUPERLU_MAX ( fsupc, fpanelc );

    if ( fst_col < jcol ) {

      /* 当前超节点与当前面板之间的距离。
       若 fsupc >= fpanelc，则 d_fsupc=0。 */
      d_fsupc = fst_col - fsupc;

    lptr = xlsub[fsupc] + d_fsupc;
    luptr = xlusup[fst_col] + d_fsupc;
    nsupr = xlsub[fsupc+1] - xlsub[fsupc];    /* 主导维度 */
    nsupc = jcol - fst_col;    /* 不包括 jcol */
    # 计算 nrow 的值，即非零元素行数
    nrow = nsupr - d_fsupc - nsupc;

    # 计算 jcol 在 snode L\U(jsupno) 中的起始位置
    ufirst = xlusup[jcol] + d_fsupc;    

    # 更新 TRSV 操作数统计：进行三角解法时的操作次数估计
    ops[TRSV] += 4 * nsupc * (nsupc - 1);
    
    # 更新 GEMV 操作数统计：进行矩阵-向量乘法时的操作次数估计
    ops[GEMV] += 8 * nrow * nsupc;
#ifdef USE_VENDOR_BLAS
#ifdef _CRAY
    // 调用 Cray 特定的双精度复数向量解算函数 CTRSV
    CTRSV( ftcs1, ftcs2, ftcs3, &nsupc, &lusup[luptr], 
           &nsupr, &lusup[ufirst], &incx );
#else
    // 调用通用的双精度复数向量解算函数 ztrsv
    ztrsv_( "L", "N", "U", &nsupc, &lusup[luptr], 
           &nsupr, &lusup[ufirst], &incx );
#endif
    
    // 设置 alpha 为 none，beta 为 one，用于下一步的矩阵-向量乘法操作 y := beta*y + alpha*A*x
    alpha = none; beta = one;

#ifdef _CRAY
    // 调用 Cray 特定的双精度复数矩阵-向量乘法函数 CGEMV
    CGEMV( ftcs2, &nrow, &nsupc, &alpha, &lusup[luptr+nsupc], &nsupr,
           &lusup[ufirst], &incx, &beta, &lusup[ufirst+nsupc], &incy );
#else
    // 调用通用的双精度复数矩阵-向量乘法函数 zgemv
    zgemv_( "N", &nrow, &nsupc, &alpha, &lusup[luptr+nsupc], &nsupr,
           &lusup[ufirst], &incx, &beta, &lusup[ufirst+nsupc], &incy );
#endif
#else
    // 如果未定义 USE_VENDOR_BLAS，执行自定义的复杂型解算函数 zlsolve
    zlsolve ( nsupr, nsupc, &lusup[luptr], &lusup[ufirst] );

    // 执行矩阵-向量乘法函数 zmatvec
    zmatvec ( nsupr, nrow, nsupc, &lusup[luptr+nsupc],
        &lusup[ufirst], tempv );
    
    /* 将更新从 tempv[*] 复制到 lusup[*] */
    isub = ufirst + nsupc;
    for (i = 0; i < nrow; i++) {
        // 调用复数减法函数 z_sub，将 tempv[i] 的值减去 lusup[isub] 的值，更新 lusup[isub]
        z_sub(&lusup[isub], &lusup[isub], &tempv[i]);
        // 将 tempv[i] 置为零
        tempv[i] = zero;
        // isub 加一，指向下一个元素
        ++isub;
    }

#endif

// 函数结束，返回0
} /* if fst_col < jcol ... */ 

// 返回主函数
return 0;
}
```