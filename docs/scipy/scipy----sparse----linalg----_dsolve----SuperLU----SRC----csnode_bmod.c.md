# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\csnode_bmod.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file csnode_bmod.c
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


#include "slu_cdefs.h"


/*! \brief Performs numeric block updates within the relaxed snode. 
 */
int
csnode_bmod (
        const int  jcol,      /* in */
        const int  jsupno,    /* in */
        const int  fsupc,     /* in */
        singlecomplex     *dense,    /* in */
        singlecomplex     *tempv,    /* working array */
        GlobalLU_t *Glu,      /* modified */
        SuperLUStat_t *stat   /* output */
        )
{
#ifdef USE_VENDOR_BLAS
#ifdef _CRAY
    _fcd ftcs1 = _cptofcd("L", strlen("L")),
     ftcs2 = _cptofcd("N", strlen("N")),
     ftcs3 = _cptofcd("U", strlen("U"));
#endif
    int            incx = 1, incy = 1;
    singlecomplex         alpha = {-1.0, 0.0},  beta = {1.0, 0.0};
#endif

    singlecomplex   comp_zero = {0.0, 0.0};
    int     nsupc, nsupr, nrow;
    int_t   isub, irow;
    int_t   ufirst, nextlu;
    int_t   *lsub, *xlsub;
    singlecomplex *lusup;
    int_t   *xlusup, luptr;
    flops_t *ops = stat->ops;

    lsub    = Glu->lsub;    /* Column indices of the supernodes */
    xlsub   = Glu->xlsub;   /* Starting position of each supernode in lsub */
    lusup   = (singlecomplex *) Glu->lusup;  /* Supernodal values of L and U */
    xlusup  = Glu->xlusup;   /* Starting position of each supernode in lusup */

    nextlu = xlusup[jcol];   /* Next position in lusup to be updated */

    /*
     *    Process the supernodal portion of L\U[*,j]
     */
    for (isub = xlsub[fsupc]; isub < xlsub[fsupc+1]; isub++) {
        irow = lsub[isub];    /* Row index within the current supernode */
        lusup[nextlu] = dense[irow];  /* Update L\U with dense matrix values */
        dense[irow] = comp_zero;      /* Clear dense matrix after update */
        ++nextlu;
    }

    xlusup[jcol + 1] = nextlu;    /* Initialize xlusup for next column */
    
    if ( fsupc < jcol ) {

        luptr = xlusup[fsupc];   /* Starting position of L\U[*,jcol] in lusup */
        nsupr = xlsub[fsupc+1] - xlsub[fsupc];   /* Number of rows in the current supernode */
        nsupc = jcol - fsupc;    /* Number of columns in the current supernode, excluding jcol */
        ufirst = xlusup[jcol];   /* Starting position of column jcol in lusup */
        nrow = nsupr - nsupc;    /* Number of rows excluding the current supernode */

        ops[TRSV] += 4 * nsupc * (nsupc - 1);   /* Counting flops for TRSV operation */
        ops[GEMV] += 8 * nrow * nsupc;          /* Counting flops for GEMV operation */
    }

#ifdef USE_VENDOR_BLAS
#ifdef _CRAY
    /* Vendor specific BLAS implementation for optimized operations */
    # 调用名为 CTRSV 的函数，执行特定的线性方程求解操作
    CTRSV( ftcs1, ftcs2, ftcs3, &nsupc, &lusup[luptr], &nsupr, 
          &lusup[ufirst], &incx );
    
    # 调用名为 CGEMV 的函数，执行特定的矩阵向量乘法操作
    CGEMV( ftcs2, &nrow, &nsupc, &alpha, &lusup[luptr+nsupc], &nsupr, 
        &lusup[ufirst], &incx, &beta, &lusup[ufirst+nsupc], &incy );
#else
#if SCIPY_FIX
       // 如果列数小于行数，直接失败，避免将无效的参数传递给 TRSV 函数
       if (nsupr < nsupc) {
           ABORT("failed to factorize matrix");
       }
#endif
    // 解线性方程 L * x = b，其中 L 是单位下三角矩阵，N 表示普通（非转置）模式，U 表示 A 是上三角矩阵
    ctrsv_( "L", "N", "U", &nsupc, &lusup[luptr], &nsupr, 
          &lusup[ufirst], &incx );
    // 计算矩阵向量乘积 y = alpha*A*x + beta*y，其中 A 是 nsupr x nsupc 的矩阵，alpha 和 beta 是标量
    cgemv_( "N", &nrow, &nsupc, &alpha, &lusup[luptr+nsupc], &nsupr, 
        &lusup[ufirst], &incx, &beta, &lusup[ufirst+nsupc], &incy );
#endif
#else
    // 解线性方程 L * x = b，其中 L 是单位下三角矩阵
    clsolve ( nsupr, nsupc, &lusup[luptr], &lusup[ufirst] );
    // 计算矩阵向量乘积 y = A*x，其中 A 是 nsupr x nrow 的矩阵
    cmatvec ( nsupr, nrow, nsupc, &lusup[luptr+nsupc], 
            &lusup[ufirst], &tempv[0] );

    int_t i, iptr; 
        // 将 tempv[*] 散布到 lusup[*] 中
    iptr = ufirst + nsupc;
    for (i = 0; i < nrow; i++) {
        // lusup[iptr] = lusup[iptr] - tempv[i]
        c_sub(&lusup[iptr], &lusup[iptr], &tempv[i]);
            ++iptr;
        // tempv[i] 置为零
        tempv[i] = comp_zero;
    }
#endif

    }

    // 返回成功
    return 0;
}
```