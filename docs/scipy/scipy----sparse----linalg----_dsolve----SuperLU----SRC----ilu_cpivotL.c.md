# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\ilu_cpivotL.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file ilu_cpivotL.c
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
#include "slu_cdefs.h"

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
ilu_cpivotL(
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
    singlecomplex       drop_sum, /* in - computed in ilu_ccopy_to_ucol()
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
    singlecomplex     temp;
    singlecomplex     *lu_sup_ptr;
    singlecomplex     *lu_col_ptr;
    int_t     *lsub_ptr;
    register int     isub, icol, k, itemp;
    int_t     *lsub, *xlsub;
    singlecomplex     *lusup;
    int_t     *xlusup;
    flops_t     *ops = stat->ops;
    int         info;
    singlecomplex one = {1.0, 0.0};

    // 主程序开始
    /* Initialize pointers */
    n           = Glu->n;             /* 将全局变量 Glu 中的 n 赋给本地变量 n */
    lsub       = Glu->lsub;           /* 将 Glu 中的 lsub 数组赋给本地变量 lsub */
    xlsub      = Glu->xlsub;          /* 将 Glu 中的 xlsub 数组赋给本地变量 xlsub */
    lusup      = (singlecomplex *) Glu->lusup; /* 将 Glu 中的 lusup 数组转换为单精度复数类型并赋给本地变量 lusup */
    xlusup     = Glu->xlusup;         /* 将 Glu 中的 xlusup 数组赋给本地变量 xlusup */
    fsupc      = (Glu->xsup)[(Glu->supno)[jcol]];  /* 计算 jcol 对应的超节点首列的索引 */
    nsupc      = jcol - fsupc;        /* 计算当前列 jcol 所属的超节点编号，不包括 jcol 自身；nsupc >= 0 */
    lptr       = xlsub[fsupc];        /* 当前超节点首列 fsupc 在 lsub 数组中的起始位置 */
    nsupr      = xlsub[fsupc+1] - lptr; /* 当前超节点包含的行数 */
    lu_sup_ptr = &lusup[xlusup[fsupc]]; /* 当前超节点的起始位置在 lusup 数组中 */
    lu_col_ptr = &lusup[xlusup[jcol]]; /* 列 jcol 在当前超节点中的起始位置 */
    lsub_ptr   = &lsub[lptr];         /* 当前超节点中行索引的起始位置 */

    /* Determine the largest abs numerical value for partial pivoting;
       Also search for user-specified pivot, and diagonal element. */
    pivmax = -1.0;                    /* 初始化部分主元选取的最大绝对值 */
    pivptr = nsupc;                   /* 部分主元选取的当前位置初始化为 nsupc */
    diag = EMPTY;                     /* 对角元索引初始化为 EMPTY */
    old_pivptr = nsupc;               /* 上一次选取的主元位置初始化为 nsupc */
    ptr0 = EMPTY;                     /* 第一个非空主元位置初始化为 EMPTY */
    for (isub = nsupc; isub < nsupr; ++isub) {
        if (marker[lsub_ptr[isub]] > jcol)
            continue; /* do not overlap with a later relaxed supernode */
        /* 如果当前行索引对应的 marker 值大于 jcol，说明与后面的放松超节点有重叠，跳过当前行 */

        switch (milu) {
            case SMILU_1:
                c_add(&temp, &lu_col_ptr[isub], &drop_sum);  /* 计算 temp = lu_col_ptr[isub] + drop_sum */
                rtemp = c_abs1(&temp);                      /* 计算 temp 的绝对值 */
                break;
            case SMILU_2:
            case SMILU_3:
                /* In this case, drop_sum contains the sum of the abs. value */
                rtemp = c_abs1(&lu_col_ptr[isub]);           /* 直接计算 lu_col_ptr[isub] 的绝对值 */
                break;
            case SILU:
            default:
                rtemp = c_abs1(&lu_col_ptr[isub]);           /* 默认情况下计算 lu_col_ptr[isub] 的绝对值 */
                break;
        }
        if (rtemp > pivmax) { pivmax = rtemp; pivptr = isub; }  /* 更新部分主元选取的最大绝对值及其位置 */
        if (*usepr && lsub_ptr[isub] == *pivrow) old_pivptr = isub;  /* 根据用户指定的主元行索引更新上一次选取的主元位置 */
        if (lsub_ptr[isub] == diagind) diag = isub;         /* 更新对角元的索引位置 */
        if (ptr0 == EMPTY) ptr0 = isub;                     /* 记录第一个非空主元位置 */
    }

    if (milu == SMILU_2 || milu == SMILU_3) pivmax += drop_sum.r; /* 如果是 SMILU_2 或 SMILU_3 类型，加上 drop_sum.r */

    /* Test for singularity */
    if (pivmax < 0.0) {
#if SCIPY_FIX
    // 如果定义了 SCIPY_FIX 宏，则输出错误信息并中止程序
    ABORT("[0]: matrix is singular");
#else
    // 否则，输出错误信息并将错误信息刷新到标准错误流
    fprintf(stderr, "[0]: jcol=%d, SINGULAR!!!\n", jcol);
    fflush(stderr);
    // 然后退出程序，返回状态码 1
    exit(1);
#endif
    } // 结束 if (*pivrow == 0.0)

    // 如果 pivmax 等于 0.0
    if ( pivmax == 0.0 ) {
        // 如果定义了 diag 宏且不为空
        if (diag != EMPTY)
            // 将 *pivrow 设置为 lsub_ptr[pivptr = diag]
            *pivrow = lsub_ptr[pivptr = diag];
        else if (ptr0 != EMPTY)
            // 否则，将 *pivrow 设置为 lsub_ptr[pivptr = ptr0]
            *pivrow = lsub_ptr[pivptr = ptr0];
        else {
            /* 寻找第一个不属于任何后续超节点的行 */
            for (icol = jcol; icol < n; icol++)
                // 如果 marker[swap[icol]] <= jcol，就跳出循环
                if (marker[swap[icol]] <= jcol) break;
            // 如果 icol 大于或等于 n
            if (icol >= n) {
#if SCIPY_FIX
                // 如果定义了 SCIPY_FIX 宏，则输出错误信息并中止程序
                ABORT("[1]: matrix is singular");
#else
                // 否则，输出错误信息并将错误信息刷新到标准错误流
                fprintf(stderr, "[1]: jcol=%d, SINGULAR!!!\n", jcol);
                fflush(stderr);
                // 然后退出程序，返回状态码 1
                exit(1);
#endif
            }
            // 将 *pivrow 设置为 swap[icol]
            *pivrow = swap[icol];

            /* 挑选主元行 */
            for (isub = nsupc; isub < nsupr; ++isub)
                // 如果 lsub_ptr[isub] 等于 *pivrow，则将 pivptr 设置为 isub 并跳出循环
                if ( lsub_ptr[isub] == *pivrow ) { pivptr = isub; break; }
        }
        // 将 pivmax 设置为 fill_tol
        pivmax = fill_tol;
        // 将 lu_col_ptr[pivptr] 的实部设为 pivmax，虚部设为 0.0
        lu_col_ptr[pivptr].r = pivmax;
        lu_col_ptr[pivptr].i = 0.0;
        // 将 *usepr 设置为 0
        *usepr = 0;
#ifdef DEBUG
        // 如果定义了 DEBUG 宏，则输出调试信息到标准输出流
        printf("[0] ZERO PIVOT: FILL (%d, %d).\n", *pivrow, jcol);
        fflush(stdout);
#endif
        // 将 info 设置为 jcol + 1
        info = jcol + 1;
    } /* 结束 if (*pivrow == 0.0) */
    else {
        // 将 thresh 设置为 u * pivmax

        /* 根据我们的策略选择适当的主元素 */
        if ( *usepr ) {
            // 如果 *usepr 非零，则根据 milu 的值选择适当的主元
            switch (milu) {
                case SMILU_1:
                    // 对 temp、lu_col_ptr[old_pivptr] 和 drop_sum 执行 c_add 操作
                    c_add(&temp, &lu_col_ptr[old_pivptr], &drop_sum);
                    // 计算 temp 的绝对值之和
                    rtemp = c_abs1(&temp);
                    break;
                case SMILU_2:
                case SMILU_3:
                    // 计算 lu_col_ptr[old_pivptr] 的绝对值与 drop_sum.r 的和
                    rtemp = c_abs1(&lu_col_ptr[old_pivptr]) + drop_sum.r;
                    break;
                case SILU:
                default:
                    // 计算 lu_col_ptr[old_pivptr] 的绝对值
                    rtemp = c_abs1(&lu_col_ptr[old_pivptr]);
                    break;
            }
            // 如果 rtemp 非零且大于等于 thresh，则 pivptr 设为 old_pivptr，否则将 *usepr 设置为 0
            if ( rtemp != 0.0 && rtemp >= thresh ) pivptr = old_pivptr;
            else *usepr = 0;
        }
        // 如果 *usepr 为 0
        if ( *usepr == 0 ) {
            /* 使用对角线主元？ */
            if ( diag >= 0 ) { /* 对角线存在 */
                switch (milu) {
                    case SMILU_1:
                        // 对 temp、lu_col_ptr[diag] 和 drop_sum 执行 c_add 操作
                        c_add(&temp, &lu_col_ptr[diag], &drop_sum);
                        // 计算 temp 的绝对值之和
                        rtemp = c_abs1(&temp);
                        break;
                    case SMILU_2:
                    case SMILU_3:
                        // 计算 lu_col_ptr[diag] 的绝对值与 drop_sum.r 的和
                        rtemp = c_abs1(&lu_col_ptr[diag]) + drop_sum.r;
                        break;
                    case SILU:
                    default:
                        // 计算 lu_col_ptr[diag] 的绝对值
                        rtemp = c_abs1(&lu_col_ptr[diag]);
                        break;
                }
                // 如果 rtemp 非零且大于等于 thresh，则 pivptr 设为 diag
                if ( rtemp != 0.0 && rtemp >= thresh ) pivptr = diag;
            }
            // 将 *pivrow 设置为 lsub_ptr[pivptr]
            *pivrow = lsub_ptr[pivptr];
        }
        // 将 info 设为 0

        /* 重置对角线 */
        switch (milu) {
            case SMILU_1:
                // 对 lu_col_ptr[pivptr] 和 drop_sum 执行 c_add 操作
                c_add(&lu_col_ptr[pivptr], &lu_col_ptr[pivptr], &drop_sum);
                break;
            case SMILU_2:
            case SMILU_3:
                // 对 lu_col_ptr[pivptr] 执行 c_sgn 操作，并将结果与 drop_sum 相乘，然后与 lu_col_ptr[pivptr] 相加
                temp = c_sgn(&lu_col_ptr[pivptr]);
                cc_mult(&temp, &temp, &drop_sum);
                c_add(&lu_col_ptr[pivptr], &lu_col_ptr[pivptr], &drop_sum);
                break;
            case SILU:
            default:
                break;
        }

    } /* else */

    /* 记录主元行 */
    # 将 jcol 的值赋给 perm_r[*pivrow]，更新排列数组
    perm_r[*pivrow] = jcol;

    # 如果 jcol 小于 n - 1
    if (jcol < n - 1) {
        register int t1, t2, t;
        # 将 iswap[*pivrow] 和 jcol 的值保存到 t1 和 t2
        t1 = iswap[*pivrow]; t2 = jcol;
        # 如果 t1 不等于 t2，则交换 swap 数组中的元素
        if (t1 != t2) {
            t = swap[t1]; swap[t1] = swap[t2]; swap[t2] = t;
            t1 = swap[t1]; t2 = t;
            t = iswap[t1]; iswap[t1] = iswap[t2]; iswap[t2] = t;
        }
    } /* if (jcol < n - 1) */

    /* 交换行下标 */
    if ( pivptr != nsupc ) {
        # 交换 lsub_ptr[pivptr] 和 lsub_ptr[nsupc] 的值
        itemp = lsub_ptr[pivptr];
        lsub_ptr[pivptr] = lsub_ptr[nsupc];
        lsub_ptr[nsupc] = itemp;

        /* 对整个 snode 的数值也进行交换，确保 L 和 A 的索引方式相同 */
        for (icol = 0; icol <= nsupc; icol++) {
            itemp = pivptr + icol * nsupr;
            temp = lu_sup_ptr[itemp];
            lu_sup_ptr[itemp] = lu_sup_ptr[nsupc + icol*nsupr];
            lu_sup_ptr[nsupc + icol*nsupr] = temp;
        }
    } /* if */

    /* 执行 cdiv 操作 */
    # 更新操作数的计数（FACT 表示因子分解），增加 10 * (nsupr - nsupc)
    ops[FACT] += 10 * (nsupr - nsupc);
    # 调用 c_div 函数，计算 lu_col_ptr[nsupc] 的倒数，并将结果存入 temp
    c_div(&temp, &one, &lu_col_ptr[nsupc]);
    # 对 lu_col_ptr[nsupc+1] 到 lu_col_ptr[nsupr-1] 的元素执行 cc_mult 运算
    for (k = nsupc+1; k < nsupr; k++) 
        cc_mult(&lu_col_ptr[k], &lu_col_ptr[k], &temp);

    # 返回 info 变量的值作为函数结果
    return info;
}



# 这行代码是一个单独的右花括号 '}'，通常用于结束一个代码块，例如函数、类或控制流语句的结束。
```