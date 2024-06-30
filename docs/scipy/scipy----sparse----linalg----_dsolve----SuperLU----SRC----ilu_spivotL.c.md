# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\ilu_spivotL.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file ilu_spivotL.c
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
#include "slu_sdefs.h"

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
ilu_spivotL(
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
    float       drop_sum, /* in - computed in ilu_scopy_to_ucol()
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
    register float  pivmax, rtemp;
    float     thresh;
    float     temp;
    float     *lu_sup_ptr;
    float     *lu_col_ptr;
    int_t     *lsub_ptr;
    register int     isub, icol, k, itemp;
    int_t     *lsub, *xlsub;
    float     *lusup;
    int_t     *xlusup;
    flops_t     *ops = stat->ops;
    int         info;

    /* Initialize pointers */
    n           = Glu->n;
    lsub       = Glu->lsub;



    xlsub       = Glu->xlsub;
    lusup       = Glu->lusup;
    xlusup      = Glu->xlusup;

    /* Begin pivoting */
    fsupc       = Glu->sup_to_col[jcol];
    nsupc       = Glu->xsup[fsupc+1] - Glu->xsup[fsupc];
    nsupr       = Glu->supno[jcol];
    lptr        = Glu->xsup[fsupc];

    pivptr      = xlsub[fsupc];
    old_pivptr  = pivptr;
    diag        = jcol;

    /* Initialize the maximum absolute value */
    pivmax = 0.0;

    /* Search the supernode for the maximum absolute value */
    for (isub = 0; isub < nsupr; ++isub) {
        icol = lsub[lptr + isub];
        if (icol < jcol) {
            rtemp = fabs(lusup[pivptr]);
            if (rtemp > pivmax) {
                pivmax = rtemp;
                pivptr = lptr + isub;
            }
        } else {
            break;
        }
    }

    /* Compute the numerical threshold for pivoting */
    thresh = u * pivmax;

    /* Determine the pivot row */
    if (*usepr != 0) {
        k = *pivrow;
        if (fabs(lusup[lptr + k - fsupc]) >= thresh) {
            pivptr = lptr + k - fsupc;
        } else if (fabs(lusup[pivptr]) < thresh) {
            pivptr = nsupr + fsupc;
        }
    } else {
        if (fabs(lusup[pivptr]) < thresh) {
            pivptr = nsupr + fsupc;
        }
    }

    /* Update the pivot row */
    *pivrow = pivptr - lptr + fsupc;

    /* Perform the column modification and CDIV operation */
    if (pivptr != old_pivptr) {
        ptr0 = xlusup[fsupc];
        temp = lusup[pivptr];
        lusup[pivptr] = lusup[old_pivptr];
        lusup[old_pivptr] = temp;

        for (isub = 0; isub < nsupr; ++isub) {
            icol = lsub[lptr + isub];
            if (icol < jcol) {
                lusup[ptr0] /= lusup[old_pivptr];
                itemp = xlsub[icol + 1] - xlsub[icol];
                lusup[ptr0 + 1:itemp] -= lusup[old_pivptr + 1:itemp] * lusup[ptr0];
                ptr0 += itemp;
            } else {
                break;
            }
        }

        ops[FACT] += 2 * (ptr0 - xlusup[fsupc]);
    }

    return 0;
}
    xlsub      = Glu->xlsub;
    // 从全局 LU 分解对象 Glu 中获取 xlsub 数组的引用，存储每列首元素的偏移量
    lusup      = (float *) Glu->lusup;
    // 从全局 LU 分解对象 Glu 中获取 lusup 数组的引用，并将其类型转换为 float 指针
    xlusup     = Glu->xlusup;
    // 从全局 LU 分解对象 Glu 中获取 xlusup 数组的引用，存储每列超节点首元素的偏移量
    fsupc      = (Glu->xsup)[(Glu->supno)[jcol]];
    // 从全局 LU 分解对象 Glu 中获取 xsup 数组的引用，并根据 jcol 所在超节点编号找到其首元素列
    nsupc      = jcol - fsupc;        /* excluding jcol; nsupc >= 0 */
    // 计算当前列 jcol 与其所属超节点首列 fsupc 之间的偏移量，得到非结点列数 nsupc（不包括 jcol，nsupc >= 0）
    lptr       = xlsub[fsupc];
    // 计算超节点 fsupc 的起始位置，即 xlsub 中第 fsupc 列的起始位置
    nsupr      = xlsub[fsupc+1] - lptr;
    // 计算超节点 fsupc 中的非零元素行数，即 xlsub 中 fsupc+1 列的起始位置减去 fsupc 列的起始位置
    lu_sup_ptr = &lusup[xlusup[fsupc]]; /* start of the current supernode */
    // 获取当前超节点 fsupc 的 LU 分解数据起始位置，即 lusup 中 xlusup[fsupc] 处的指针
    lu_col_ptr = &lusup[xlusup[jcol]];    /* start of jcol in the supernode */
    // 获取当前列 jcol 在其所在超节点中的 LU 分解数据起始位置，即 lusup 中 xlusup[jcol] 处的指针
    lsub_ptr   = &lsub[lptr];    /* start of row indices of the supernode */
    // 获取当前超节点 fsupc 中行索引数组 lsub 的起始位置，即 lsub 中第 lptr 个元素的指针

    /* Determine the largest abs numerical value for partial pivoting;
       Also search for user-specified pivot, and diagonal element. */
    // 确定用于部分选主元的最大绝对数值；
    // 同时搜索用户指定的主元和对角线元素。
    pivmax = -1.0;
    // 初始化部分选主元的最大绝对数值为 -1.0
    pivptr = nsupc;
    // 初始化部分选主元的索引为 nsupc
    diag = EMPTY;
    // 初始化对角线元素索引为 EMPTY
    old_pivptr = nsupc;
    // 初始化旧的部分选主元索引为 nsupc
    ptr0 = EMPTY;
    // 初始化 ptr0 为 EMPTY
    for (isub = nsupc; isub < nsupr; ++isub) {
        // 遍历超节点 fsupc 中的非零元素行索引数组
        if (marker[lsub_ptr[isub]] > jcol)
            continue; /* do not overlap with a later relaxed supernode */
        // 如果当前行索引已被标记且大于 jcol，则跳过，避免与后续放松的超节点重叠

        switch (milu) {
            case SMILU_1:
            // 如果选用 SMILU_1 预处理方法
                rtemp = fabs(lu_col_ptr[isub] + drop_sum);
                // 计算当前 LU 分解元素绝对值加上 drop_sum 的绝对值
                break;
            case SMILU_2:
            case SMILU_3:
                // 如果选用 SMILU_2 或 SMILU_3 预处理方法
                    /* In this case, drop_sum contains the sum of the abs. value */
                // 在这种情况下，drop_sum 包含绝对值的总和
                rtemp = fabs(lu_col_ptr[isub]);
                // 计算当前 LU 分解元素的绝对值
                break;
            case SILU:
            default:
                // 如果选用 SILU 或默认预处理方法
                rtemp = fabs(lu_col_ptr[isub]);
                // 计算当前 LU 分解元素的绝对值
                break;
        }
        if (rtemp > pivmax) { pivmax = rtemp; pivptr = isub; }
        // 更新部分选主元的最大绝对数值及其索引
        if (*usepr && lsub_ptr[isub] == *pivrow) old_pivptr = isub;
        // 如果使用用户指定主元，并且当前行索引等于指定主元行索引，则更新旧的部分选主元索引
        if (lsub_ptr[isub] == diagind) diag = isub;
        // 如果当前行索引等于对角线元素索引，则更新对角线元素索引
        if (ptr0 == EMPTY) ptr0 = isub;
        // 如果 ptr0 为 EMPTY，则更新为当前行索引
    }

    if (milu == SMILU_2 || milu == SMILU_3) pivmax += drop_sum;
    // 如果选用 SMILU_2 或 SMILU_3 预处理方法，则将部分选主元的最大绝对数值加上 drop_sum

    /* Test for singularity */
    // 检测是否为奇异矩阵
    if (pivmax < 0.0) {
    // 如果部分选主元的最大绝对数值小于 0
#if SCIPY_FIX
    // 如果定义了 SCIPY_FIX 宏，则输出错误信息并终止程序
    ABORT("[0]: matrix is singular");
#else
    // 否则，输出错误信息到标准错误流并终止程序
    fprintf(stderr, "[0]: jcol=%d, SINGULAR!!!\n", jcol);
    // 刷新标准错误流
    fflush(stderr);
    // 退出程序
    exit(1);
#endif
    }

    // 如果最大主元值为 0
    if ( pivmax == 0.0 ) {
        // 如果有定义 diag
        if (diag != EMPTY)
            // 将主元行指针指向对应的行起始索引处
            *pivrow = lsub_ptr[pivptr = diag];
        // 否则，如果有定义 ptr0
        else if (ptr0 != EMPTY)
            // 将主元行指针指向对应的行起始索引处
            *pivrow = lsub_ptr[pivptr = ptr0];
        else {
            /* look for the first row which does not
               belong to any later supernodes */
            // 寻找第一个不属于后续超节点的行
            for (icol = jcol; icol < n; icol++)
                // 如果标记数组中的值小于或等于当前列号 jcol，则退出循环
                if (marker[swap[icol]] <= jcol) break;
            // 如果 icol 大于等于 n，说明未找到符合条件的行
            if (icol >= n) {
#if SCIPY_FIX
                // 如果定义了 SCIPY_FIX 宏，则输出错误信息并终止程序
                ABORT("[1]: matrix is singular");
#else
                // 否则，输出错误信息到标准错误流并终止程序
                fprintf(stderr, "[1]: jcol=%d, SINGULAR!!!\n", jcol);
                // 刷新标准错误流
                fflush(stderr);
                // 退出程序
                exit(1);
#endif
            }

            // 将主元行指针指向 swap 数组中的特定位置
            *pivrow = swap[icol];

            /* pick up the pivot row */
            // 选取主元行
            for (isub = nsupc; isub < nsupr; ++isub)
                // 如果在非零下标数组中找到了对应的行，更新主元行指针
                if ( lsub_ptr[isub] == *pivrow ) { pivptr = isub; break; }
        }
        // 将主元最大值设为 fill_tol
        pivmax = fill_tol;
        // 设置不使用预选主元
        *usepr = 0;
#ifdef DEBUG
        // 输出调试信息，指示零主元情况下的填充
        printf("[0] ZERO PIVOT: FILL (%d, %d).\n", *pivrow, jcol);
        // 刷新标准输出流
        fflush(stdout);
#endif
        // 更新 info 信息，返回 jcol+1
        info = jcol + 1;
    } /* if (*pivrow == 0.0) */
    else {
        // 计算阈值
        thresh = u * pivmax;

        /* Choose appropriate pivotal element by our policy. */
        // 根据策略选择适当的主元元素
        if ( *usepr ) {
            switch (milu) {
                case SMILU_1:
                    // 计算绝对值并考虑丢弃项
                    rtemp = fabs(lu_col_ptr[old_pivptr] + drop_sum);
                    break;
                case SMILU_2:
                case SMILU_3:
                    // 计算绝对值并考虑丢弃项
                    rtemp = fabs(lu_col_ptr[old_pivptr]) + drop_sum;
                    break;
                case SILU:
                default:
                    // 计算绝对值
                    rtemp = fabs(lu_col_ptr[old_pivptr]);
                    break;
            }
            // 如果 rtemp 非零且大于等于阈值，则选择原来的预选主元
            if ( rtemp != 0.0 && rtemp >= thresh ) pivptr = old_pivptr;
            else *usepr = 0;
        }
        // 如果不使用预选主元
        if ( *usepr == 0 ) {
            /* Use diagonal pivot? */
            // 使用对角主元？
            if ( diag >= 0 ) { /* diagonal exists */
                switch (milu) {
                    case SMILU_1:
                        // 计算绝对值并考虑丢弃项
                        rtemp = fabs(lu_col_ptr[diag] + drop_sum);
                        break;
                    case SMILU_2:
                    case SMILU_3:
                        // 计算绝对值并考虑丢弃项
                        rtemp = fabs(lu_col_ptr[diag]) + drop_sum;
                        break;
                    case SILU:
                    default:
                        // 计算绝对值
                        rtemp = fabs(lu_col_ptr[diag]);
                        break;
                }
                // 如果 rtemp 非零且大于等于阈值，则选择对角主元
                if ( rtemp != 0.0 && rtemp >= thresh ) pivptr = diag;
            }
            // 更新主元行指针
            *pivrow = lsub_ptr[pivptr];
        }
        // 重置 info 为 0
        info = 0;

        /* Reset the diagonal */
        // 重置对角元素
        switch (milu) {
            case SMILU_1:
                // 增加丢弃项到主元元素
                lu_col_ptr[pivptr] += drop_sum;
                break;
            case SMILU_2:
            case SMILU_3:
                // 根据主元元素符号增加丢弃项
                lu_col_ptr[pivptr] += SGN(lu_col_ptr[pivptr]) * drop_sum;
                break;
            case SILU:
            default:
                // 默认情况下不做任何操作
                break;
        }

    } /* else */

    /* Record pivot row */
    // 记录主元行
    perm_r[*pivrow] = jcol;
    // 如果 jcol 小于 n-1
    if (jcol < n - 1) {
        // 交换 swap 数组中的两个位置
        register int t1, t2, t;
        t1 = iswap[*pivrow]; t2 = jcol;
        // 如果 t1 不等于 t2，则进行交换操作
        if (t1 != t2) {
            t = swap[t1]; swap[t1] = swap[t2]; swap[t2] = t;
            t1 = swap[t1]; t2 = t;
            t = iswap[t1]; iswap[t1] = iswap[t2]; iswap[t2] = t;
        }

    }


注释：

        // 交换完毕
    }


这样，我们已经为给定的代码添加了详细的注释，解释了每一行代码的作用和含义。
    } /* if (jcol < n - 1) */
    
    /* 如果 jcol 小于 n - 1，则执行以下代码块 */

    /* Interchange row subscripts */
    /* 交换行标 */
    if ( pivptr != nsupc ) {
        itemp = lsub_ptr[pivptr];
        lsub_ptr[pivptr] = lsub_ptr[nsupc];
        lsub_ptr[nsupc] = itemp;

        /* Interchange numerical values as well, for the whole snode, such 
         * that L is indexed the same way as A.
         */
        /* 同样交换数值，以确保整个 snode 的 L 和 A 索引相同 */
        for (icol = 0; icol <= nsupc; icol++) {
            itemp = pivptr + icol * nsupr;
            temp = lu_sup_ptr[itemp];
            lu_sup_ptr[itemp] = lu_sup_ptr[nsupc + icol*nsupr];
            lu_sup_ptr[nsupc + icol*nsupr] = temp;
        }
    } /* if */

    /* cdiv operation */
    /* cdiv 操作 */
    ops[FACT] += nsupr - nsupc;
    temp = 1.0 / lu_col_ptr[nsupc];
    for (k = nsupc+1; k < nsupr; k++) lu_col_ptr[k] *= temp;

    return info;
}



# 这行代码表示一个代码块的结束，对应于之前的一个代码块的开始
```