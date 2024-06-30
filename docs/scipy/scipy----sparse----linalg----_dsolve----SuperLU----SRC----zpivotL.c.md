# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\zpivotL.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file zpivotL.c
 * \brief Performs numerical pivoting
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


#include <math.h>
#include <stdlib.h>
#include "slu_zdefs.h"

#undef DEBUG

/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *   Performs the numerical pivoting on the current column of L,
 *   and the CDIV operation.
 *
 *   Pivot policy:
 *   (1) Compute thresh = u * max_(i>=j) abs(A_ij);
 *   (2) IF user specifies pivot row k and abs(A_kj) >= thresh THEN
 *           pivot row = k;
 *       ELSE IF abs(A_jj) >= thresh THEN
 *           pivot row = j;
 *       ELSE
 *           pivot row = m;
 * 
 *   Note: If you absolutely want to use a given pivot order, then set u=0.0.
 *
 *   Return value: 0      success;
 *                 i > 0  U(i,i) is exactly zero.
 * </pre>
 */

int
zpivotL(
        const int  jcol,     /* in */
        const double u,      /* in - diagonal pivoting threshold */
        int        *usepr,   /* re-use the pivot sequence given by perm_r/iperm_r */
        int        *perm_r,  /* may be modified */
        int        *iperm_r, /* in - inverse of perm_r */
        int        *iperm_c, /* in - used to find diagonal of Pc*A*Pc' */
        int        *pivrow,  /* out */
        GlobalLU_t *Glu,     /* modified - global LU data structures */
    SuperLUStat_t *stat  /* output */
       )
{
    // 定义复数值为1.0 + 0.0i
    doublecomplex one = {1.0, 0.0};
    int          fsupc;        /* first column in the supernode */
    int          nsupc;        /* no of columns in the supernode */
    int          nsupr;     /* no of rows in the supernode */
    int_t        lptr;        /* points to the starting subscript of the supernode */
    int          pivptr, old_pivptr, diag, diagind;
    double       pivmax, rtemp, thresh;
    doublecomplex       temp;
    doublecomplex       *lu_sup_ptr; 
    doublecomplex       *lu_col_ptr;
    int_t        *lsub_ptr;
    int_t        isub, icol, k, itemp;
    int_t        *lsub, *xlsub;
    doublecomplex       *lusup;


注释部分已添加。
    # 指针变量，指向整型数据类型，用于存储 xlusup 数组的地址
    int_t        *xlusup;
    
    # 指针变量，指向 flops_t 结构体中的 ops 成员，用于操作统计数据
    flops_t      *ops = stat->ops;

    /* Initialize pointers */
    
    # lsub 指向 Glu 结构体中的 lsub 数组，存储非零元素的行索引
    lsub       = Glu->lsub;
    
    # xlsub 指向 Glu 结构体中的 xlsub 数组，存储每个超节点起始位置的索引
    xlsub      = Glu->xlsub;
    
    # lusup 转换为 doublecomplex 指针，指向 Glu 结构体中的 lusup 数组，存储 LU 分解的因子
    lusup      = (doublecomplex *) Glu->lusup;
    
    # xlusup 指向 Glu 结构体中的 xlusup 数组，存储每个超节点起始位置的索引
    xlusup     = Glu->xlusup;
    
    # fsupc 存储列 jcol 所属的超节点的起始列号
    fsupc      = (Glu->xsup)[(Glu->supno)[jcol]];
    
    # nsupc 表示 jcol 列除去 jcol 本身后的超节点内列数，应大于等于 0
    nsupc      = jcol - fsupc;            /* excluding jcol; nsupc >= 0 */
    
    # lptr 是超节点 fsupc 在 xlsub 数组中的起始位置，存储列索引
    lptr       = xlsub[fsupc];
    
    # nsupr 是超节点 fsupc 内部非零元素的行数
    nsupr      = xlsub[fsupc+1] - lptr;
    
    # lu_sup_ptr 指向当前超节点的起始位置，存储 LU 分解因子 lusup 数组的指针
    lu_sup_ptr = &lusup[xlusup[fsupc]];    /* start of the current supernode */
    
    # lu_col_ptr 指向列 jcol 在超节点内的起始位置，存储 LU 分解因子 lusup 数组的指针
    lu_col_ptr = &lusup[xlusup[jcol]];    /* start of jcol in the supernode */
    
    # lsub_ptr 指向超节点 fsupc 的行索引起始位置
    lsub_ptr   = &lsub[lptr];    /* start of row indices of the supernode */
#ifdef DEBUG
if ( jcol == MIN_COL ) {
    printf("Before cdiv: col %d\n", jcol);
    for (k = nsupc; k < nsupr; k++) 
    printf("  lu[%d] %f\n", lsub_ptr[k], lu_col_ptr[k]);
}
#endif
    
/* 确定用于部分选点的最大绝对数值；
   同时搜索用户指定的选点和对角元素 */
if ( *usepr ) *pivrow = iperm_r[jcol];
diagind = iperm_c[jcol];
pivmax = 0.0;
pivptr = nsupc;
diag = EMPTY;
old_pivptr = nsupc;
for (isub = nsupc; isub < nsupr; ++isub) {
    rtemp = z_abs1 (&lu_col_ptr[isub]);
    if ( rtemp > pivmax ) {
        pivmax = rtemp;
        pivptr = isub;
    }
    if ( *usepr && lsub_ptr[isub] == *pivrow ) old_pivptr = isub;
    if ( lsub_ptr[isub] == diagind ) diag = isub;
}

/* 检测是否奇异 */
if ( pivmax == 0.0 ) {
#if 1
#if SCIPY_FIX
    if (pivptr < nsupr) {
        *pivrow = lsub_ptr[pivptr];
    }
    else {
        *pivrow = diagind;
    }
#else
    *pivrow = lsub_ptr[pivptr];
#endif
    perm_r[*pivrow] = jcol;
#else
    perm_r[diagind] = jcol;
#endif
    *usepr = 0;
    return (jcol+1);
}

thresh = u * pivmax;
    
/* 根据策略选择适当的选点元素 */
if ( *usepr ) {
    rtemp = z_abs1 (&lu_col_ptr[old_pivptr]);
    if ( rtemp != 0.0 && rtemp >= thresh )
        pivptr = old_pivptr;
    else
        *usepr = 0;
}
if ( *usepr == 0 ) {
    /* 使用对角选点？ */
    if ( diag >= 0 ) { /* 存在对角元素 */
        rtemp = z_abs1 (&lu_col_ptr[diag]);
        if ( rtemp != 0.0 && rtemp >= thresh ) pivptr = diag;
    }
    *pivrow = lsub_ptr[pivptr];
}

/* 记录选点行 */
perm_r[*pivrow] = jcol;

/* 交换行下标 */
if ( pivptr != nsupc ) {
    itemp = lsub_ptr[pivptr];
    lsub_ptr[pivptr] = lsub_ptr[nsupc];
    lsub_ptr[nsupc] = itemp;

    /* 也交换数值，确保 L 与 A 索引方式一致 */
    for (icol = 0; icol <= nsupc; icol++) {
        itemp = pivptr + icol * nsupr;
        temp = lu_sup_ptr[itemp];
        lu_sup_ptr[itemp] = lu_sup_ptr[nsupc + icol*nsupr];
        lu_sup_ptr[nsupc + icol*nsupr] = temp;
    }
}

/* cdiv 操作 */
ops[FACT] += 10 * (nsupr - nsupc);

z_div(&temp, &one, &lu_col_ptr[nsupc]);
for (k = nsupc+1; k < nsupr; k++) 
    zz_mult(&lu_col_ptr[k], &lu_col_ptr[k], &temp);

return 0;
}
```