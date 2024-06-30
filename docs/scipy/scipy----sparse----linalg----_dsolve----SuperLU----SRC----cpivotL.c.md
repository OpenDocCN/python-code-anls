# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\cpivotL.c`

```
    /*! \file
    版权声明和许可信息
    Copyright (c) 2003, The Regents of the University of California, through
    Lawrence Berkeley National Laboratory (subject to receipt of any required 
    approvals from U.S. Dept. of Energy) 

    All rights reserved. 

    The source code is distributed under BSD license, see the file License.txt
    at the top-level directory.
    */

    /*! @file cpivotL.c
     * \brief 执行数值轴点
     *
     * <pre>
     * -- SuperLU routine (version 3.0) --
     * Univ. of California Berkeley, Xerox Palo Alto Research Center,
     * and Lawrence Berkeley National Lab.
     * October 15, 2003
     *
     * 版权所有 (c) 1994 Xerox Corporation。保留所有权利。
     *
     * 本材料按原样提供，不作任何明示或暗示的担保。使用者承担所有风险。
     * 
     * 授权使用或复制此程序用于任何目的，只要以上声明保留在所有副本上。
     * 允许修改代码并分发修改后的代码，前提是保留以上声明，并在以上版权声明中包含代码已修改的通知。
     * </pre>
     */


    #include <math.h>
    #include <stdlib.h>
    #include "slu_cdefs.h"

    #undef DEBUG

    /*! \brief
     *
     * <pre>
     * Purpose
     * =======
     *   对 L 的当前列执行数值轴点和 CDIV 操作。
     *
     *   轴点策略：
     *   (1) 计算 thresh = u * max_(i>=j) abs(A_ij);
     *   (2) IF 用户指定轴点行 k 并且 abs(A_kj) >= thresh THEN
     *           轴点行 = k;
     *       ELSE IF abs(A_jj) >= thresh THEN
     *           轴点行 = j;
     *       ELSE
     *           轴点行 = m;
     * 
     *   注意：如果您绝对要使用给定的轴点顺序，则设置 u=0.0。
     *
     *   返回值: 0      成功;
     *           i > 0  U(i,i) 正好为零。
     * </pre>
     */

    int
    cpivotL(
            const int  jcol,     /* in */
            const double u,      /* in - 对角轴点阈值 */
            int        *usepr,   /* 重用由 perm_r/iperm_r 给出的轴点顺序 */
            int        *perm_r,  /* 可能会被修改 */
            int        *iperm_r, /* in - perm_r 的逆 */
            int        *iperm_c, /* in - 用于找到 Pc*A*Pc' 的对角线 */
            int        *pivrow,  /* out */
            GlobalLU_t *Glu,     /* 修改 - 全局 LU 数据结构 */
            SuperLUStat_t *stat  /* 输出 */
           )
    {

        singlecomplex one = {1.0, 0.0};  // 单精度复数 1
        int          fsupc;              // 超节点中的第一列
        int          nsupc;              // 超节点中的列数
        int          nsupr;              // 超节点中的行数
        int_t        lptr;               // 指向超节点起始下标
        int          pivptr, old_pivptr, diag, diagind;
        float        pivmax, rtemp, thresh;
        singlecomplex temp;
        singlecomplex *lu_sup_ptr;       // 指向超节点 LU 数据
        singlecomplex *lu_col_ptr;
        int_t        *lsub_ptr;          // 超节点的行索引指针
        int_t        isub, icol, k, itemp;
        int_t        *lsub, *xlsub;
        singlecomplex *lusup;
    /* 指向整型数组的指针，用于存储 xlusup */
    int_t        *xlusup;
    /* 指向 flops_t 结构体中的 ops 成员的指针 */
    flops_t      *ops = stat->ops;

    /* 初始化指针 */
    /* 指向 Glu 结构体中 lsub 数组的指针 */
    lsub       = Glu->lsub;
    /* 指向 Glu 结构体中 xlsub 数组的指针 */
    xlsub      = Glu->xlsub;
    /* 将 Glu 结构体中的 lusup 转换为 singlecomplex 类型的指针 */
    lusup      = (singlecomplex *) Glu->lusup;
    /* 指向 Glu 结构体中 xlusup 数组的指针 */
    xlusup     = Glu->xlusup;
    /* fsupc 是 Glu 结构体中 xsup 数组的索引，代表列 jcol 所在的超节点的起始列 */
    fsupc      = (Glu->xsup)[(Glu->supno)[jcol]];
    /* 计算超节点中不包括列 jcol 的列数 */
    nsupc      = jcol - fsupc;            /* excluding jcol; nsupc >= 0 */
    /* 超节点 fsupc 的第一个非零元素在 lsub 数组中的起始位置 */
    lptr       = xlsub[fsupc];
    /* 超节点 fsupc 在 lsub 数组中的行索引数量 */
    nsupr      = xlsub[fsupc+1] - lptr;
    /* 指向当前超节点的 LU 分解数据的指针 */
    lu_sup_ptr = &lusup[xlusup[fsupc]];    /* start of the current supernode */
    /* 指向超节点中列 jcol 的 LU 分解数据的指针 */
    lu_col_ptr = &lusup[xlusup[jcol]];    /* start of jcol in the supernode */
    /* 指向超节点中行索引的起始位置 */
    lsub_ptr   = &lsub[lptr];    /* start of row indices of the supernode */
#ifdef DEBUG
if ( jcol == MIN_COL ) {
    printf("Before cdiv: col %d\n", jcol);
    for (k = nsupc; k < nsupr; k++) 
    printf("  lu[%d] %f\n", lsub_ptr[k], lu_col_ptr[k]);
}
#endif
    
/* Determine the largest abs numerical value for partial pivoting;
   Also search for user-specified pivot, and diagonal element. */
if ( *usepr ) *pivrow = iperm_r[jcol];
diagind = iperm_c[jcol];
pivmax = 0.0;
pivptr = nsupc;
diag = EMPTY;
old_pivptr = nsupc;
for (isub = nsupc; isub < nsupr; ++isub) {
    rtemp = c_abs1 (&lu_col_ptr[isub]);
    if ( rtemp > pivmax ) {
        pivmax = rtemp;
        pivptr = isub;
    }
    if ( *usepr && lsub_ptr[isub] == *pivrow ) old_pivptr = isub;
    if ( lsub_ptr[isub] == diagind ) diag = isub;
}

/* Test for singularity */
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

/* Choose appropriate pivotal element by our policy. */
if ( *usepr ) {
    rtemp = c_abs1 (&lu_col_ptr[old_pivptr]);
    if ( rtemp != 0.0 && rtemp >= thresh )
        pivptr = old_pivptr;
    else
        *usepr = 0;
}
if ( *usepr == 0 ) {
    /* Use diagonal pivot? */
    if ( diag >= 0 ) { /* diagonal exists */
        rtemp = c_abs1 (&lu_col_ptr[diag]);
        if ( rtemp != 0.0 && rtemp >= thresh ) pivptr = diag;
    }
    *pivrow = lsub_ptr[pivptr];
}

/* Record pivot row */
perm_r[*pivrow] = jcol;

/* Interchange row subscripts */
if ( pivptr != nsupc ) {
    itemp = lsub_ptr[pivptr];
    lsub_ptr[pivptr] = lsub_ptr[nsupc];
    lsub_ptr[nsupc] = itemp;

    /* Interchange numerical values as well, for the whole snode, such 
     * that L is indexed the same way as A.
     */
    for (icol = 0; icol <= nsupc; icol++) {
        itemp = pivptr + icol * nsupr;
        temp = lu_sup_ptr[itemp];
        lu_sup_ptr[itemp] = lu_sup_ptr[nsupc + icol*nsupr];
        lu_sup_ptr[nsupc + icol*nsupr] = temp;
    }
} /* if */

/* cdiv operation */
ops[FACT] += 10 * (nsupr - nsupc);

c_div(&temp, &one, &lu_col_ptr[nsupc]);
for (k = nsupc+1; k < nsupr; k++) 
    cc_mult(&lu_col_ptr[k], &lu_col_ptr[k], &temp);

return 0;


注释：

#ifdef DEBUG
if ( jcol == MIN_COL ) {
    printf("Before cdiv: col %d\n", jcol);
    for (k = nsupc; k < nsupr; k++) 
        printf("  lu[%d] %f\n", lsub_ptr[k], lu_col_ptr[k]);
}
#endif

/* 确定用于部分主元选取的最大绝对值数值；
   同时搜索用户指定的主元和对角元素。 */
if ( *usepr ) *pivrow = iperm_r[jcol];
diagind = iperm_c[jcol];
pivmax = 0.0;
pivptr = nsupc;
diag = EMPTY;
old_pivptr = nsupc;
for (isub = nsupc; isub < nsupr; ++isub) {
    rtemp = c_abs1 (&lu_col_ptr[isub]);
    if ( rtemp > pivmax ) {
        pivmax = rtemp;
        pivptr = isub;
    }
    if ( *usepr && lsub_ptr[isub] == *pivrow ) old_pivptr = isub;
    if ( lsub_ptr[isub] == diagind ) diag = isub;
}

/* 检测奇异性 */
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

/* 根据策略选择适当的主元素 */
if ( *usepr ) {
    rtemp = c_abs1 (&lu_col_ptr[old_pivptr]);
    if ( rtemp != 0.0 && rtemp >= thresh )
        pivptr = old_pivptr;
    else
        *usepr = 0;
}
if ( *usepr == 0 ) {
    /* 使用对角线主元？ */
    if ( diag >= 0 ) { /* 对角元存在 */
        rtemp = c_abs1 (&lu_col_ptr[diag]);
        if ( rtemp != 0.0 && rtemp >= thresh ) pivptr = diag;
    }
    *pivrow = lsub_ptr[pivptr];
}

/* 记录主元行 */
perm_r[*pivrow] = jcol;

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
ops[FACT] += 10 * (nsupr - nsupc);

c_div(&temp, &one, &lu_col_ptr[nsupc]);
for (k = nsupc+1; k < nsupr; k++) 
    cc_mult(&lu_col_ptr[k], &lu_col_ptr[k], &temp);

return 0;
```