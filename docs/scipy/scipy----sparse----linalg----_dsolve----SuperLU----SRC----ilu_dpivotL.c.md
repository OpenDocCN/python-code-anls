# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\ilu_dpivotL.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file ilu_dpivotL.c
 * \brief Performs numerical pivoting
 *
 * <pre>
 * -- SuperLU routine (version 4.0) --
 * Lawrence Berkeley National Laboratory
 * June 30, 2009
 * </pre>
 */

#include <math.h>
#include <stdlib.h>
#include "slu_ddefs.h"

#ifndef SGN
#define SGN(x) ((x)>=0?1:-1)
#endif

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
 *         pivot row = k;
 *     ELSE IF abs(A_jj) >= thresh THEN
 *         pivot row = j;
 *     ELSE
 *         pivot row = m;
 *
 *   Note: If you absolutely want to use a given pivot order, then set u=0.0.
 *
 *   Return value: 0      success;
 *           i > 0  U(i,i) is exactly zero.
 * </pre>
 */

int
ilu_dpivotL(
    const int  jcol,     /* in */
    const double u,      /* in - diagonal pivoting threshold */
    int       *usepr,   /* re-use the pivot sequence given by
                  * perm_r/iperm_r */
    int       *perm_r,  /* may be modified */
    int       diagind,  /* diagonal of Pc*A*Pc' */
    int       *swap,    /* in/out record the row permutation */
    int       *iswap,   /* in/out inverse of swap, it is the same as
                perm_r after the factorization */
    int       *marker,  /* in */
    int       *pivrow,  /* in/out, as an input if *usepr!=0 */
    double       fill_tol, /* in - fill tolerance of current column
                  * used for a singular column */
    milu_t       milu,     /* in */
    double       drop_sum, /* in - computed in ilu_dcopy_to_ucol()
                                (MILU only) */
    GlobalLU_t *Glu,     /* modified - global LU data structures */
    SuperLUStat_t *stat  /* output */
       )
{

    int         n;     /* number of columns */
    int         fsupc;  /* first column in the supernode */
    int         nsupc;  /* no of columns in the supernode */
    int         nsupr;  /* no of rows in the supernode */
    int_t     lptr;     /* points to the starting subscript of the supernode */
    register int     pivptr;
    int         old_pivptr, diag, ptr0;
    register double  pivmax, rtemp;
    double     thresh;
    double     temp;
    double     *lu_sup_ptr;
    double     *lu_col_ptr;
    int_t     *lsub_ptr;
    register int     isub, icol, k, itemp;
    int_t     *lsub, *xlsub;
    double     *lusup;
    int_t     *xlusup;
    flops_t     *ops = stat->ops;
    int         info;

    /* Initialize pointers */
    n           = Glu->n;         // 获取全局 LU 数据结构中的列数
    lsub       = Glu->lsub;       // 获取全局 LU 数据结构中的列下标数组
    xlsub      = Glu->xlsub;
    // 获取全局 LU 结构中 xlsub 数组的引用

    lusup      = (double *) Glu->lusup;
    // 将 Glu 结构中 lusup 数组转换为 double 指针类型

    xlusup     = Glu->xlusup;
    // 获取全局 LU 结构中 xlusup 数组的引用

    fsupc      = (Glu->xsup)[(Glu->supno)[jcol]];
    // 根据列索引 jcol 计算出 fsupc，表示 jcol 所属的超节点的首列索引

    nsupc      = jcol - fsupc;        /* excluding jcol; nsupc >= 0 */
    // 计算 nsupc，表示 jcol 所属的超节点中排除 jcol 本身后的列数，应大于等于 0

    lptr       = xlsub[fsupc];
    // 获取超节点 fsupc 的 xlsub 数组中的起始位置索引

    nsupr      = xlsub[fsupc+1] - lptr;
    // 计算超节点 fsupc 的行数，即 xlsub 数组中下一个超节点起始位置索引与当前超节点起始位置索引之差

    lu_sup_ptr = &lusup[xlusup[fsupc]]; /* start of the current supernode */
    // 获取当前超节点 fsupc 在 lusup 数组中的起始位置指针

    lu_col_ptr = &lusup[xlusup[jcol]];    /* start of jcol in the supernode */
    // 获取列 jcol 在 lusup 数组中的起始位置指针，即 jcol 所属超节点的起始位置

    lsub_ptr   = &lsub[lptr];    /* start of row indices of the supernode */
    // 获取超节点 fsupc 对应的行索引数组 lsub 中的起始位置指针

    /* Determine the largest abs numerical value for partial pivoting;
       Also search for user-specified pivot, and diagonal element. */
    // 确定部分主元选取中的最大绝对数值；
    // 同时搜索用户指定的主元和对角元素。
    pivmax = -1.0;
    // 初始化部分主元选取的最大绝对数值为 -1.0
    pivptr = nsupc;
    // 初始化主元位置指针为 nsupc
    diag = EMPTY;
    // 初始化对角元素位置为 EMPTY
    old_pivptr = nsupc;
    // 初始化旧主元位置为 nsupc
    ptr0 = EMPTY;
    // 初始化 ptr0 为 EMPTY
    for (isub = nsupc; isub < nsupr; ++isub) {
        // 遍历超节点 fsupc 中的行索引数组 lsub 的元素

        if (marker[lsub_ptr[isub]] > jcol)
            continue; /* do not overlap with a later relaxed supernode */
        // 如果当前行索引指向的列在后面的松弛超节点中，则跳过当前行索引

        switch (milu) {
            case SMILU_1:
                rtemp = fabs(lu_col_ptr[isub] + drop_sum);
                break;
            case SMILU_2:
            case SMILU_3:
                /* In this case, drop_sum contains the sum of the abs. value */
                rtemp = fabs(lu_col_ptr[isub]);
                break;
            case SILU:
            default:
                rtemp = fabs(lu_col_ptr[isub]);
                break;
        }
        // 根据 milu 的不同情况计算 rtemp，即当前元素的绝对值

        if (rtemp > pivmax) { pivmax = rtemp; pivptr = isub; }
        // 更新部分主元选取的最大绝对数值和其位置指针

        if (*usepr && lsub_ptr[isub] == *pivrow) old_pivptr = isub;
        // 如果使用用户指定的主元，并且当前行索引与指定主元行索引相同，则更新旧主元位置指针

        if (lsub_ptr[isub] == diagind) diag = isub;
        // 如果当前行索引与对角元素索引相同，则更新对角元素位置指针

        if (ptr0 == EMPTY) ptr0 = isub;
        // 如果 ptr0 为初始值 EMPTY，则更新为当前行索引
    }

    if (milu == SMILU_2 || milu == SMILU_3) pivmax += drop_sum;
    // 若 milu 为 SMILU_2 或 SMILU_3，则在部分主元选取的最大绝对数值上加上 drop_sum

    /* Test for singularity */
    // 检测是否出现奇异性
    if (pivmax < 0.0) {
#if SCIPY_FIX
ABORT("[0]: matrix is singular");
#else
fprintf(stderr, "[0]: jcol=%d, SINGULAR!!!\n", jcol);
fflush(stderr);
exit(1);
#endif

如果定义了 SCIPY_FIX 宏，则执行 ABORT 函数并输出错误信息，表示矩阵是奇异的。否则，输出错误信息到标准错误流，指示 jcol 值和矩阵奇异性，并强制刷新 stderr，然后退出程序。


}
if ( pivmax == 0.0 ) {

如果 pivmax 的值为 0.0，则执行以下操作：


if (diag != EMPTY)
*pivrow = lsub_ptr[pivptr = diag];
else if (ptr0 != EMPTY)
*pivrow = lsub_ptr[pivptr = ptr0];
else {
/* look for the first row which does not
belong to any later supernodes */
for (icol = jcol; icol < n; icol++)
if (marker[swap[icol]] <= jcol) break;
if (icol >= n) {
#if SCIPY_FIX
ABORT("[1]: matrix is singular");
#else
fprintf(stderr, "[1]: jcol=%d, SINGULAR!!!\n", jcol);
fflush(stderr);
exit(1);
#endif
}

*pivrow = swap[icol];

/* pick up the pivot row */
for (isub = nsupc; isub < nsupr; ++isub)
if ( lsub_ptr[isub] == *pivrow ) { pivptr = isub; break; }
}
pivmax = fill_tol;
lu_col_ptr[pivptr] = pivmax;
*usepr = 0;
#ifdef DEBUG
printf("[0] ZERO PIVOT: FILL (%d, %d).\n", *pivrow, jcol);
fflush(stdout);
#endif
info =jcol + 1;
} /* if (*pivrow == 0.0) */
else {
thresh = u * pivmax;

/* Choose appropriate pivotal element by our policy. */
if ( *usepr ) {
switch (milu) {
case SMILU_1:
rtemp = fabs(lu_col_ptr[old_pivptr] + drop_sum);
break;
case SMILU_2:
case SMILU_3:
rtemp = fabs(lu_col_ptr[old_pivptr]) + drop_sum;
break;
case SILU:
default:
rtemp = fabs(lu_col_ptr[old_pivptr]);
break;
}
if ( rtemp != 0.0 && rtemp >= thresh ) pivptr = old_pivptr;
else *usepr = 0;
}
if ( *usepr == 0 ) {
/* Use diagonal pivot? */
if ( diag >= 0 ) { /* diagonal exists */
switch (milu) {
case SMILU_1:
rtemp = fabs(lu_col_ptr[diag] + drop_sum);
break;
case SMILU_2:
case SMILU_3:
rtemp = fabs(lu_col_ptr[diag]) + drop_sum;
break;
case SILU:
default:
rtemp = fabs(lu_col_ptr[diag]);
break;
}
if ( rtemp != 0.0 && rtemp >= thresh ) pivptr = diag;
}
*pivrow = lsub_ptr[pivptr];
}
info = 0;

/* Reset the diagonal */
switch (milu) {
case SMILU_1:
lu_col_ptr[pivptr] += drop_sum;
break;
case SMILU_2:
case SMILU_3:
lu_col_ptr[pivptr] += SGN(lu_col_ptr[pivptr]) * drop_sum;
break;
case SILU:
default:
break;
}

} /* else */

/* Record pivot row */
perm_r[*pivrow] = jcol;
if (jcol < n - 1) {
register int t1, t2, t;
t1 = iswap[*pivrow]; t2 = jcol;
if (t1 != t2) {
t = swap[t1]; swap[t1] = swap[t2]; swap[t2] = t;
t1 = swap[t1]; t2 = t;
t = iswap[t1]; iswap[t1] = iswap[t2]; iswap[t2] = t;
}

根据条件和策略选择适当的主元素，并更新 LU 分解的相关数据结构。在计算中记录并更新置换数组以及进行必要的对角线重设。
    } /* if (jcol < n - 1) */
    /* 如果 jcol 小于 n - 1，则执行下面的代码块 */

    /* Interchange row subscripts */
    /* 交换行索引 */

    if ( pivptr != nsupc ) {
    /* 如果 pivptr 不等于 nsupc，则执行下面的代码块 */

    itemp = lsub_ptr[pivptr];
    /* 交换 lsub_ptr[pivptr] 和 lsub_ptr[nsupc] 的值 */
    lsub_ptr[pivptr] = lsub_ptr[nsupc];
    lsub_ptr[nsupc] = itemp;

    /* Interchange numerical values as well, for the whole snode, such 
     * that L is indexed the same way as A.
     */
    /* 也交换数值，确保整个 snode 的 L 索引与 A 相同 */

    for (icol = 0; icol <= nsupc; icol++) {
        /* 遍历列 icol */

        itemp = pivptr + icol * nsupr;
        /* 计算 lu_sup_ptr 中的索引位置 */
        temp = lu_sup_ptr[itemp];
        lu_sup_ptr[itemp] = lu_sup_ptr[nsupc + icol*nsupr];
        lu_sup_ptr[nsupc + icol*nsupr] = temp;
        /* 交换 lu_sup_ptr 中的数值 */
    }

    } /* if */
    /* 结束条件判断块 */

    /* cdiv operation */
    /* cdiv 操作 */

    ops[FACT] += nsupr - nsupc;
    /* 更新 ops[FACT] 的值 */

    temp = 1.0 / lu_col_ptr[nsupc];
    /* 计算 lu_col_ptr[nsupc] 的倒数 */

    for (k = nsupc+1; k < nsupr; k++) lu_col_ptr[k] *= temp;
    /* 对 lu_col_ptr[nsupc+1] 到 lu_col_ptr[nsupr-1] 的元素进行缩放操作 */

    return info;
    /* 返回变量 info */
}


注释：


# 这是一个代码块的结束标记，对应于前面的一个代码块的开始标记{
```