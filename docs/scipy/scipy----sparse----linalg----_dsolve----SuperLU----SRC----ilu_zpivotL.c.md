# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\ilu_zpivotL.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file ilu_zpivotL.c
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
#include "slu_zdefs.h"

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
ilu_zpivotL(
    const int  jcol,     /* in */
    const double u,      /* in - diagonal pivoting threshold */
    int       *usepr,    /* re-use the pivot sequence given by
                           * perm_r/iperm_r */
    int       *perm_r,   /* may be modified */
    int       diagind,   /* diagonal of Pc*A*Pc' */
    int       *swap,     /* in/out record the row permutation */
    int       *iswap,    /* in/out inverse of swap, it is the same as
                           * perm_r after the factorization */
    int       *marker,   /* in */
    int       *pivrow,   /* in/out, as an input if *usepr!=0 */
    double    fill_tol,  /* in - fill tolerance of current column
                           * used for a singular column */
    milu_t    milu,      /* in */
    doublecomplex drop_sum, /* in - computed in ilu_zcopy_to_ucol()
                              * (MILU only) */
    GlobalLU_t *Glu,     /* modified - global LU data structures */
    SuperLUStat_t *stat  /* output */
)
{

    int         n;          /* number of columns */
    int         fsupc;      /* first column in the supernode */
    int         nsupc;      /* no of columns in the supernode */
    int         nsupr;      /* no of rows in the supernode */
    int_t       lptr;       /* points to the starting subscript of the supernode */
    register int pivptr;
    int         old_pivptr, diag, ptr0;
    register double pivmax, rtemp;
    double      thresh;
    doublecomplex temp;
    doublecomplex *lu_sup_ptr;
    doublecomplex *lu_col_ptr;
    int_t       *lsub_ptr;
    register int isub, icol, k, itemp;
    int_t       *lsub, *xlsub;
    doublecomplex *lusup;
    int_t       *xlusup;
    flops_t     *ops = stat->ops;
    int         info;
    doublecomplex one = {1.0, 0.0};

    /* Function starts here */
    /* Initialize pointers */
    /* 设置指针初始值 */
    n = Glu->n;
    /* 从全局 LU 结构体中获取矩阵的维度 */

    lsub = Glu->lsub;
    /* 获取非零元素的行索引 */

    xlsub = Glu->xlsub;
    /* 获取每一列在 lsub 数组中的起始和终止索引 */

    lusup = (doublecomplex *) Glu->lusup;
    /* 获取 LU 分解后的超节点数据 */

    xlusup = Glu->xlusup;
    /* 获取每个超节点在 lusup 数组中的起始位置 */

    fsupc = (Glu->xsup)[(Glu->supno)[jcol]];
    /* 获取列 jcol 所在的超节点的第一列 */

    nsupc = jcol - fsupc; /* excluding jcol; nsupc >= 0 */
    /* 计算 jcol 所在的超节点中不包括 jcol 在内的列数，确保非负 */

    lptr = xlsub[fsupc];
    /* 获取超节点 fsupc 在 lsub 数组中的起始位置 */

    nsupr = xlsub[fsupc+1] - lptr;
    /* 获取超节点 fsupc 中包含的行数 */

    lu_sup_ptr = &lusup[xlusup[fsupc]];
    /* 获取当前超节点 fsupc 在 lusup 数组中的起始位置 */

    lu_col_ptr = &lusup[xlusup[jcol]];
    /* 获取列 jcol 在 lusup 数组中的起始位置 */

    lsub_ptr = &lsub[lptr];
    /* 获取超节点 fsupc 中非零元素的行索引数组的起始位置 */

    /* Determine the largest abs numerical value for partial pivoting;
       Also search for user-specified pivot, and diagonal element. */
    /* 确定用于部分选主的最大绝对数值；
       同时搜索用户指定的主元和对角线元素。 */
    pivmax = -1.0;
    /* 部分选主的最大绝对数值初始化 */

    pivptr = nsupc;
    /* 部分选主的主元指针初始化为 nsupc */

    diag = EMPTY;
    /* 对角线元素初始化为 EMPTY */

    old_pivptr = nsupc;
    /* 旧的主元指针初始化为 nsupc */

    ptr0 = EMPTY;
    /* ptr0 初始化为 EMPTY */

    for (isub = nsupc; isub < nsupr; ++isub) {
        /* 遍历超节点 fsupc 中的每一行 */

        if (marker[lsub_ptr[isub]] > jcol)
            continue; /* do not overlap with a later relaxed supernode */
        /* 如果当前行的标记大于 jcol，则跳过该行，避免与后续松弛的超节点重叠 */

        switch (milu) {
            /* 根据选择的 ILU 类型进行不同处理 */

            case SMILU_1:
                z_add(&temp, &lu_col_ptr[isub], &drop_sum);
                /* 对于 SMILU_1，计算 temp = lu_col_ptr[isub] + drop_sum */
                rtemp = z_abs1(&temp);
                /* 计算 temp 的 1-范数 */
                break;

            case SMILU_2:
            case SMILU_3:
                /* 对于 SMILU_2 或 SMILU_3 */
                /* In this case, drop_sum contains the sum of the abs. value */
                /* 在这种情况下，drop_sum 包含绝对值的总和 */
                rtemp = z_abs1(&lu_col_ptr[isub]);
                /* 计算 lu_col_ptr[isub] 的 1-范数 */
                break;

            case SILU:
            default:
                rtemp = z_abs1(&lu_col_ptr[isub]);
                /* 默认情况下，计算 lu_col_ptr[isub] 的 1-范数 */
                break;
        }

        if (rtemp > pivmax) { pivmax = rtemp; pivptr = isub; }
        /* 更新部分选主的最大绝对数值和主元指针 */

        if (*usepr && lsub_ptr[isub] == *pivrow) old_pivptr = isub;
        /* 如果启用用户主元行，且当前行索引等于指定的主元行索引，则更新旧的主元指针 */

        if (lsub_ptr[isub] == diagind) diag = isub;
        /* 如果当前行索引等于指定的对角线元素索引，则更新对角线元素 */

        if (ptr0 == EMPTY) ptr0 = isub;
        /* 如果 ptr0 仍然是 EMPTY，则将其更新为当前行索引 */
    }

    if (milu == SMILU_2 || milu == SMILU_3) pivmax += drop_sum.r;
    /* 对于 SMILU_2 或 SMILU_3 类型，增加 drop_sum.r 到部分选主的最大绝对数值 */

    /* Test for singularity */
    /* 测试奇异性 */
    if (pivmax < 0.0) {
        /* 如果部分选主的最大绝对数值小于 0 */
#if SCIPY_FIX
ABORT("[0]: matrix is singular");
#else
fprintf(stderr, "[0]: jcol=%d, SINGULAR!!!\n", jcol);
fflush(stderr);
exit(1);
#endif

如果定义了 SCIPY_FIX 宏，则调用 ABORT 函数并传入指定的错误信息字符串，表示矩阵是奇异的。否则，使用 fprintf 将错误信息打印到标准错误输出，然后退出程序。


}
if ( pivmax == 0.0 ) {

如果 pivmax 等于 0.0，则执行以下操作：


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

如果 diag 不为 EMPTY，则将 *pivrow 设置为 lsub_ptr[diag]，并更新 pivptr。如果 diag 为空，但 ptr0 不为空，则将 *pivrow 设置为 lsub_ptr[ptr0]，并更新 pivptr。否则，查找第一个不属于后续超级节点的行，更新 *pivrow 和 pivptr。如果找不到这样的行，根据 SCIPY_FIX 宏的定义，使用 ABORT 函数或 fprintf 输出错误信息并退出程序。


pivmax = fill_tol;
lu_col_ptr[pivptr].r = pivmax;
lu_col_ptr[pivptr].i = 0.0;
*usepr = 0;
#ifdef DEBUG
printf("[0] ZERO PIVOT: FILL (%d, %d).\n", *pivrow, jcol);
fflush(stdout);
#endif
info =jcol + 1;
} /* if (*pivrow == 0.0) */
else {
thresh = u * pivmax;

设置 pivmax 为 fill_tol，然后更新 lu_col_ptr[pivptr] 的实部为 pivmax，虚部为 0.0，同时将 *usepr 设置为 0。如果定义了 DEBUG 宏，则输出零主元的填充信息。最后，将 info 设置为 jcol + 1。


/* Choose appropriate pivotal element by our policy. */
if ( *usepr ) {
switch (milu) {
case SMILU_1:
z_add(&temp, &lu_col_ptr[old_pivptr], &drop_sum);
rtemp = z_abs1(&temp);
break;
case SMILU_2:
case SMILU_3:
rtemp = z_abs1(&lu_col_ptr[old_pivptr]) + drop_sum.r;
break;
case SILU:
default:
rtemp = z_abs1(&lu_col_ptr[old_pivptr]);
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
z_add(&temp, &lu_col_ptr[diag], &drop_sum);
rtemp = z_abs1(&temp);
break;
case SMILU_2:
case SMILU_3:
rtemp = z_abs1(&lu_col_ptr[diag]) + drop_sum.r;
break;
case SILU:
default:
rtemp = z_abs1(&lu_col_ptr[diag]);
break;
}
if ( rtemp != 0.0 && rtemp >= thresh ) pivptr = diag;
}
*pivrow = lsub_ptr[pivptr];
}
info = 0;

根据策略选择适当的主元。如果 *usepr 为真，则根据不同的 milu 类型计算临时值，并根据阈值 thresh 决定是否使用旧的主元位置 old_pivptr。否则，如果 *usepr 为假，则考虑使用 diag 位置的对角主元（如果存在），同样根据 milu 类型和阈值 thresh 进行判断，并更新 *pivrow。最后将 info 设置为 0。


/* Reset the diagonal */
switch (milu) {
case SMILU_1:
z_add(&lu_col_ptr[pivptr], &lu_col_ptr[pivptr], &drop_sum);
break;
case SMILU_2:
case SMILU_3:
temp = z_sgn(&lu_col_ptr[pivptr]);
zz_mult(&temp, &temp, &drop_sum);
z_add(&lu_col_ptr[pivptr], &lu_col_ptr[pivptr], &drop_sum);
break;
case SILU:
default:
break;
}

根据 milu 类型重置对角元素 lu_col_ptr[pivptr]。


} /* else */

/* Record pivot row */

结束主元选择的条件判断，并记录选定的主元行。
    # 将当前列的列索引赋值给 perm_r 数组中对应的行索引位置
    perm_r[*pivrow] = jcol;
    
    # 如果当前列索引 jcol 小于 n - 1
    if (jcol < n - 1) {
        # 定义寄存器变量 t1, t2, t 并进行赋值
        register int t1, t2, t;
        t1 = iswap[*pivrow]; t2 = jcol;
        
        # 如果 t1 不等于 t2
        if (t1 != t2) {
            t = swap[t1]; swap[t1] = swap[t2]; swap[t2] = t;
            t1 = swap[t1]; t2 = t;
            t = iswap[t1]; iswap[t1] = iswap[t2]; iswap[t2] = t;
        }
    } /* if (jcol < n - 1) */

    /* 交换行索引 */
    if (pivptr != nsupc) {
        itemp = lsub_ptr[pivptr];
        lsub_ptr[pivptr] = lsub_ptr[nsupc];
        lsub_ptr[nsupc] = itemp;

        /* 交换数值，确保整个超节点（snode）的 L 与 A 以相同的方式索引 */
        for (icol = 0; icol <= nsupc; icol++) {
            itemp = pivptr + icol * nsupr;
            temp = lu_sup_ptr[itemp];
            lu_sup_ptr[itemp] = lu_sup_ptr[nsupc + icol*nsupr];
            lu_sup_ptr[nsupc + icol*nsupr] = temp;
        }
    } /* if */

    /* 执行 cdiv 操作 */
    ops[FACT] += 10 * (nsupr - nsupc);
    z_div(&temp, &one, &lu_col_ptr[nsupc]);
    for (k = nsupc+1; k < nsupr; k++) 
        zz_mult(&lu_col_ptr[k], &lu_col_ptr[k], &temp);

    return info;
}


注释：

# 这是一个代码块的结束标志，表示前面的代码块或函数定义已经结束
```