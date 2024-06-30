# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\dpivotL.c`

```
/*! @file dpivotL.c
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
#include "slu_ddefs.h"

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
dpivotL(
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
    int          nsupr;        /* no of rows in the supernode */
    int_t        lptr;         /* points to the starting subscript of the supernode */
    int          pivptr, old_pivptr, diag, diagind;
    double       pivmax, rtemp, thresh;
    double       temp;
    double       *lu_sup_ptr; 
    double       *lu_col_ptr;
    int_t        *lsub_ptr;
    int_t        isub, icol, k, itemp;
    int_t        *lsub, *xlsub;
    double       *lusup;
    int_t        *xlusup;
    flops_t      *ops = stat->ops;

    // 计算超节点的第一列和列数
    fsupc = Glu->xusub[jcol];
    nsupc = Glu->xusub[jcol + 1] - fsupc;
    // 计算超节点的行数
    nsupr = Glu->lusup_end[jcol] - Glu->lusup_beg[jcol];
    // 超节点的起始指针
    lptr = Glu->lusup_beg[jcol];
    // 初始化 pivmax 为 0.0
    pivmax = 0.0;
    // 初始化 pivptr 和 old_pivptr 为 -1
    pivptr = old_pivptr = -1;
    // 得到对角线元素的列索引
    diag = iperm_c[jcol];
    // 得到对角线元素的全局索引
    diagind = perm_r[diag];

    // 计算 u 与第一列元素绝对值最大值的乘积，作为阈值 thresh
    thresh = u * fabs(Glu->lusup[lptr]);

    // 遍历超节点的列
    for (isub = 0; isub < nsupc; ++isub) {
        icol = fsupc + isub;  // 超节点中当前列的全局列索引
        // 获取当前列的起始位置
        lptr = Glu->lusup_beg[icol];
        // 获取当前列的列指针
        lu_col_ptr = Glu->lusup + lptr;
        // 获取当前列的行索引指针
        lsub_ptr = Glu->lsub + Glu->xlsub[icol];
        // 获取当前列的行数
        nsupr = Glu->lusup_end[icol] - lptr;

        // 如果当前列是对角线列
        if (icol == diagind) {
            // 遍历当前列的所有行
            for (k = 0; k < nsupr; ++k) {
                // 获取当前行的全局行索引
                itemp = iperm_r[lsub_ptr[k]];
                // 如果行索引大于等于对角线行索引，并且绝对值大于阈值 thresh
                if (itemp >= diag && fabs(*lu_col_ptr) >= thresh) {
                    // 更新 pivot row 为当前行
                    *pivrow = itemp;
                    return 0;
                }
                lu_col_ptr++; // 移动到下一行元素
            }
        } else { // 如果当前列不是对角线列
            // 遍历当前列的所有行
            for (k = 0; k < nsupr; ++k) {
                // 获取当前行的全局行索引
                itemp = iperm_r[lsub_ptr[k]];
                // 如果绝对值大于阈值 thresh
                if (fabs(*lu_col_ptr) >= thresh) {
                    // 更新 pivot row 为当前行
                    *pivrow = itemp;
                    return 0;
                }
                lu_col_ptr++; // 移动到下一行元素
            }
        }
    }

    // 如果没有找到合适的 pivot row，返回默认值 m
    *pivrow = Glu->numrow;
    return 0;
}
    /* Initialize pointers */
    /* 初始化指针变量 */

    lsub       = Glu->lsub;
    /* 将全局 LU 因子结构中的 lsub 指针赋值给局部变量 lsub */

    xlsub      = Glu->xlsub;
    /* 将全局 LU 因子结构中的 xlsub 指针赋值给局部变量 xlsub */

    lusup      = (double *) Glu->lusup;
    /* 将全局 LU 因子结构中的 lusup 指针转换为 double 类型，并赋值给局部变量 lusup */

    xlusup     = Glu->xlusup;
    /* 将全局 LU 因子结构中的 xlusup 指针赋值给局部变量 xlusup */

    fsupc      = (Glu->xsup)[(Glu->supno)[jcol]];
    /* 根据列 jcol 对应的超节点编号，获取其首列的起始位置索引，并赋值给 fsupc */

    nsupc      = jcol - fsupc;            /* excluding jcol; nsupc >= 0 */
    /* 计算当前列 jcol 在超节点中的列偏移量，并赋值给 nsupc，保证 nsupc >= 0 */

    lptr       = xlsub[fsupc];
    /* 根据超节点首列 fsupc 获取 lsub 中该超节点起始行索引的位置，并赋值给 lptr */

    nsupr      = xlsub[fsupc+1] - lptr;
    /* 计算超节点的行数，并赋值给 nsupr */

    lu_sup_ptr = &lusup[xlusup[fsupc]];    /* start of the current supernode */
    /* 根据超节点首列 fsupc 获取 lusup 中该超节点在全局存储中的起始位置，并赋值给 lu_sup_ptr */

    lu_col_ptr = &lusup[xlusup[jcol]];    /* start of jcol in the supernode */
    /* 根据列 jcol 获取 lusup 中该列在超节点中的起始位置，并赋值给 lu_col_ptr */

    lsub_ptr   = &lsub[lptr];    /* start of row indices of the supernode */
    /* 根据 lptr 获取 lsub 中超节点行索引的起始位置，并赋值给 lsub_ptr */
#ifdef DEBUG
// 如果处于调试模式，输出当前列的信息
if ( jcol == MIN_COL ) {
    printf("Before cdiv: col %d\n", jcol);
    for (k = nsupc; k < nsupr; k++) 
        printf("  lu[%d] %f\n", lsub_ptr[k], lu_col_ptr[k]);
}
#endif
    
/* 确定用于部分选点的最大绝对数值；
   同时搜索用户指定的选点和对角元素 */
if ( *usepr ) *pivrow = iperm_r[jcol]; // 如果使用用户指定的选点，将其保存到*pivrow中
diagind = iperm_c[jcol]; // 获取列jcol对应的列置换后的对角元素索引
pivmax = 0.0; // 初始化选点的最大绝对数值
pivptr = nsupc; // 初始化选点的索引为当前列的起始位置
diag = EMPTY; // 初始化对角元素索引为EMPTY（空）
old_pivptr = nsupc; // 旧的选点索引也初始化为当前列的起始位置
for (isub = nsupc; isub < nsupr; ++isub) {
    rtemp = fabs (lu_col_ptr[isub]); // 计算当前元素的绝对值
    if ( rtemp > pivmax ) { // 如果当前元素绝对值大于当前选点的最大绝对数值
        pivmax = rtemp; // 更新选点的最大绝对数值
        pivptr = isub; // 更新选点的索引
    }
    if ( *usepr && lsub_ptr[isub] == *pivrow ) old_pivptr = isub; // 如果使用用户指定的选点，并且当前行索引与*pivrow相等，则更新旧的选点索引
    if ( lsub_ptr[isub] == diagind ) diag = isub; // 如果当前行索引与对角元素索引相等，则更新对角元素索引
}

/* 检测奇异性 */
if ( pivmax == 0.0 ) {
#if 0
    // 没有有效的选点
    // jcol 表示 U 的秩
    // 报告秩，让 dgstrf 处理选点
#if 1
#if SCIPY_FIX
    // 如果在 SCIPY 修复模式下
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
#endif
    *usepr = 0; // 将 usepr 标记为不使用用户指定的选点
    return (jcol+1); // 返回当前列的秩
}

thresh = u * pivmax; // 计算阈值作为选点的界限

/* 根据我们的策略选择适当的选点元素 */
if ( *usepr ) {
    rtemp = fabs (lu_col_ptr[old_pivptr]); // 计算旧选点元素的绝对值
    if ( rtemp != 0.0 && rtemp >= thresh )
        pivptr = old_pivptr; // 如果旧选点元素绝对值大于等于阈值，则更新选点索引为旧选点索引
    else
        *usepr = 0; // 否则不使用用户指定的选点
}
if ( *usepr == 0 ) {
/* 使用对角选点？ */
if ( diag >= 0 ) { /* 对角元素存在 */
    rtemp = fabs (lu_col_ptr[diag]); // 计算对角元素的绝对值
    if ( rtemp != 0.0 && rtemp >= thresh ) pivptr = diag; // 如果对角元素绝对值大于等于阈值，则更新选点索引为对角元素索引
    }
*pivrow = lsub_ptr[pivptr]; // 将选点行索引保存到*pivrow中
}

/* 记录选点行 */
perm_r[*pivrow] = jcol; // 将当前列 jcol 的秩保存到选点行索引对应的位置

/* 交换行索引 */
if ( pivptr != nsupc ) {
itemp = lsub_ptr[pivptr]; // 临时保存选点行索引
lsub_ptr[pivptr] = lsub_ptr[nsupc]; // 将选点行索引更新为当前列的起始行索引
lsub_ptr[nsupc] = itemp; // 将当前列的起始行索引更新为选点行索引

/* 交换数值，使得 L 的索引与 A 相同 */
for (icol = 0; icol <= nsupc; icol++) {
    itemp = pivptr + icol * nsupr; // 计算选点位置的索引
    temp = lu_sup_ptr[itemp]; // 临时保存选点位置的数值
    lu_sup_ptr[itemp] = lu_sup_ptr[nsupc + icol*nsupr]; // 将选点位置的数值更新为当前列的起始位置的数值
    lu_sup_ptr[nsupc + icol*nsupr] = temp; // 将当前列的起始位置的数值更新为选点位置的数值
}
} /* if */

/* cdiv 操作 */
ops[FACT] += nsupr - nsupc; // 更新操作数

temp = 1.0 / lu_col_ptr[nsupc]; // 计算当前列的倒数
for (k = nsupc+1; k < nsupr; k++) 
lu_col_ptr[k] *= temp; // 将当前列后续元素乘以倒数值，实现除法操作

return 0; // 返回成功
}
```