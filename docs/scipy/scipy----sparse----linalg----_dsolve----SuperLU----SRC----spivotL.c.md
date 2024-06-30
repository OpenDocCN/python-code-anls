# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\spivotL.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file spivotL.c
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
#include "slu_sdefs.h"

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
spivotL(
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

    int          fsupc;        /* first column in the supernode */
    int          nsupc;        /* no of columns in the supernode */
    int          nsupr;     /* no of rows in the supernode */
    int_t        lptr;        /* points to the starting subscript of the supernode */
    int          pivptr, old_pivptr, diag, diagind;
    float       pivmax, rtemp, thresh;
    float       temp;
    float       *lu_sup_ptr; 
    float       *lu_col_ptr;
    int_t        *lsub_ptr;
    int_t        isub, icol, k, itemp;
    int_t        *lsub, *xlsub;
    float       *lusup;
    int_t        *xlusup;
    flops_t      *ops = stat->ops;

    // 计算超节点的第一列和列数
    fsupc = Glu->xsup[jcol];
    nsupc = Glu->xsup[jcol+1] - fsupc;

    // 初始化 pivmax 为 0
    pivmax = 0.0;

    // 设置 lsub 和 xlsub 指针
    lsub = Glu->lsub;
    xlsub = Glu->xlsub;

    // 设置 lusup 和 xlusup 指针
    lusup = Glu->lusup;
    xlusup = Glu->xlusup;

    // 计算 thresh 阈值
    thresh = u * fabs(lusup[xlusup[fsupc]]);

    // 遍历超节点中的列
    for (k = 0; k < nsupc; ++k) {
        // 当前列的指针
        icol = fsupc + k;
        // 当前列的超节点的行数
        nsupr = xlsub[icol + 1] - xlsub[icol];

        // 当前列对应的 L 超节点中的第一个元素的指针
        lu_sup_ptr = &lusup[xlusup[icol]];

        // 遍历当前列的所有行
        for (isub = 0; isub < nsupr; ++isub) {
            // 当前行的行号
            itemp = lsub[xlsub[icol] + isub];
            // 如果行号大于列号，并且绝对值大于当前的 pivmax，则更新 pivmax 和 pivptr
            if (itemp > icol && fabs(lu_sup_ptr[isub]) > pivmax) {
                pivmax = fabs(lu_sup_ptr[isub]);
                pivptr = itemp;
            }
        }
    }

    // 如果 pivmax 大于阈值 thresh，则使用找到的主元行 pivptr
    if (pivmax >= thresh) {
        *pivrow = pivptr;
    } else {
        // 否则使用当前列的主元行 jcol
        *pivrow = jcol;
    }

    // 统计操作数
    ops[FACT] += 2 * nsupr;

    // 返回成功
    return 0;
}
    /* 初始化指针 */
    lsub       = Glu->lsub;                    /* 指向 L 行索引数组 */
    xlsub      = Glu->xlsub;                   /* 指向 L 列偏移数组 */
    lusup      = (float *) Glu->lusup;          /* 指向 U 非零元素数组 */
    xlusup     = Glu->xlusup;                  /* 指向 U 列偏移数组 */
    fsupc      = (Glu->xsup)[(Glu->supno)[jcol]];  /* 第 jcol 列的首个超节点的起始列索引 */
    nsupc      = jcol - fsupc;                  /* 不包括 jcol 自身的超节点列数，nsupc >= 0 */
    lptr       = xlsub[fsupc];                  /* 超节点 fsupc 的第一个非零元素在 lsub 中的起始位置 */
    nsupr      = xlsub[fsupc+1] - lptr;         /* 超节点 fsupc 的行数 */
    lu_sup_ptr = &lusup[xlusup[fsupc]];         /* 当前超节点的 LU 分解中 U 部分的起始位置 */
    lu_col_ptr = &lusup[xlusup[jcol]];          /* 当前列 jcol 在其所在超节点中 U 部分的起始位置 */
    lsub_ptr   = &lsub[lptr];                  /* 当前超节点的行索引数组的起始位置 */
#ifdef DEBUG
if ( jcol == MIN_COL ) {
    printf("Before cdiv: col %d\n", jcol);
    for (k = nsupc; k < nsupr; k++) 
    printf("  lu[%d] %f\n", lsub_ptr[k], lu_col_ptr[k]);
}
#endif
    
    /* 确定部分选点的最大绝对数值；
       同时搜索用户指定的选点和对角元素。 */
    if ( *usepr ) *pivrow = iperm_r[jcol];
    diagind = iperm_c[jcol];  // 获取列 jcol 的对角元素在列置换数组中的索引
    pivmax = 0.0;  // 部分选点的最大绝对数值初始化为0
    pivptr = nsupc;  // 选点指针初始化为 nsupc，即从第一个非对角元素开始
    diag = EMPTY;  // 对角元素的索引初始化为 EMPTY，表示未找到对应的对角元素
    old_pivptr = nsupc;  // 旧选点指针初始化为 nsupc
    
    // 遍历列 jcol 中的非对角元素
    for (isub = nsupc; isub < nsupr; ++isub) {
        rtemp = fabs (lu_col_ptr[isub]);  // 获取列 jcol 中当前元素的绝对值
        if ( rtemp > pivmax ) {  // 如果当前元素的绝对值大于部分选点的最大绝对数值
            pivmax = rtemp;  // 更新部分选点的最大绝对数值
            pivptr = isub;  // 更新选点指针为当前元素的索引
        }
        if ( *usepr && lsub_ptr[isub] == *pivrow ) old_pivptr = isub;  // 如果使用部分选点且当前元素是用户指定的选点，更新旧选点指针
        if ( lsub_ptr[isub] == diagind ) diag = isub;  // 如果当前元素是对角元素，更新对角元素的索引
    }

    /* 测试是否奇异 */
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
    *usepr = 0;  // 禁用部分选点
    return (jcol+1);  // 返回列 jcol+1
    }

    thresh = u * pivmax;  // 计算阈值，用于选择适当的选点元素

    /* 根据策略选择适当的选点元素 */
    if ( *usepr ) {
        rtemp = fabs (lu_col_ptr[old_pivptr]);  // 获取旧选点元素的绝对值
        if ( rtemp != 0.0 && rtemp >= thresh )  // 如果旧选点元素不为零且大于等于阈值
            pivptr = old_pivptr;  // 更新选点指针为旧选点指针
        else
            *usepr = 0;  // 否则禁用部分选点
    }
    if ( *usepr == 0 ) {
    /* 使用对角选点？ */
    if ( diag >= 0 ) { /* 对角元素存在 */
        rtemp = fabs (lu_col_ptr[diag]);  // 获取对角元素的绝对值
        if ( rtemp != 0.0 && rtemp >= thresh ) pivptr = diag;  // 如果对角元素不为零且大于等于阈值，则更新选点指针为对角元素的索引
        }
    *pivrow = lsub_ptr[pivptr];  // 更新选点行
    }
    
    /* 记录选点行 */
    perm_r[*pivrow] = jcol;  // 在行置换数组中记录选点行对应的列 jcol
    
    /* 交换行下标 */
    if ( pivptr != nsupc ) {
    itemp = lsub_ptr[pivptr];
    lsub_ptr[pivptr] = lsub_ptr[nsupc];
    lsub_ptr[nsupc] = itemp;

    /* 同时交换数值，确保 L 与 A 的索引方式一致 */
    for (icol = 0; icol <= nsupc; icol++) {
        itemp = pivptr + icol * nsupr;
        temp = lu_sup_ptr[itemp];
        lu_sup_ptr[itemp] = lu_sup_ptr[nsupc + icol*nsupr];
        lu_sup_ptr[nsupc + icol*nsupr] = temp;
    }
    } /* if */

    /* cdiv 操作 */
    ops[FACT] += nsupr - nsupc;  // 更新操作数
    
    temp = 1.0 / lu_col_ptr[nsupc];  // 计算倒数
    for (k = nsupc+1; k < nsupr; k++) 
    lu_col_ptr[k] *= temp;  // 对列进行缩放

    return 0;  // 返回0，表示成功执行
}
```