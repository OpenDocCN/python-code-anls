# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\ilu_zcopy_to_ucol.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file ilu_zcopy_to_ucol.c
 * \brief Copy a computed column of U to the compressed data structure
 * and drop some small entries
 *
 * <pre>
 * -- SuperLU routine (version 4.1) --
 * Lawrence Berkeley National Laboratory
 * November, 2010
 * </pre>
 */

#include "slu_zdefs.h"

#ifdef DEBUG
int num_drop_U;
#endif

extern void zcopy_(int *, doublecomplex [], int *, doublecomplex [], int *);

#if 0
static doublecomplex *A;  /* used in _compare_ only */
static int _compare_(const void *a, const void *b)
{
    register int *x = (int *)a, *y = (int *)b;
    register double xx = z_abs1(&A[*x]), yy = z_abs1(&A[*y]);
    if (xx > yy) return -1;
    else if (xx < yy) return 1;
    else return 0;
}
#endif

/**
 * @brief Copy a computed column of U to the compressed data structure and drop some small entries
 *
 * This function copies a computed column of U to the compressed data structure and applies 
 * dropping rules to remove small entries based on specified criteria.
 *
 * @param jcol      Column index of U to be copied
 * @param nseg      Number of segments in U
 * @param segrep    Array of segment representatives
 * @param repfnz    Array of first nonzeros in each segment
 * @param perm_r    Row permutation vector
 * @param dense     Dense matrix representation of U (modified to reset to zero)
 * @param drop_rule Dropping rule identifier
 * @param milu      Indicates whether to use modified ILU or not
 * @param drop_tol   Dropping tolerance
 * @param quota     Maximum nonzero entries allowed
 * @param sum       Output parameter to store the sum of dropped entries
 * @param nnzUj     Number of nonzeros in column j of U (input/output)
 * @param Glu       GlobalLU_t structure containing global LU factors (modified)
 * @param work      Working space array
 *
 * @return 0 if successful, -1 if memory allocation fails
 */
int
ilu_zcopy_to_ucol(
          int     jcol,       
          int     nseg,       
          int     *segrep,    
          int     *repfnz,    
          int     *perm_r,    
          doublecomplex     *dense,   
          int       drop_rule,
          milu_t     milu,       
          double     drop_tol, 
          int     quota,    
          doublecomplex     *sum,       
          int     *nnzUj,   
          GlobalLU_t *Glu,       
          double     *work       
          )
{
/*
 * Gather from SPA dense[*] to global ucol[*].
 */
    int       ksub, krep, ksupno, kfnz, segsze;
    int       i, k; 
    int       fsupc, isub, irow;
    int       jsupno;
    int_t     new_next, nextu, mem_error;
    int       *xsup, *supno;
    int_t     *lsub, *xlsub;
    doublecomplex    *ucol;
    int_t     *usub, *xusub;
    int_t     nzumax;
    int       m; /* number of entries in the nonzero U-segments */
    register double d_max = 0.0, d_min = 1.0 / dmach("Safe minimum");
    register double tmp;
    doublecomplex zero = {0.0, 0.0};
    int i_1 = 1;

    xsup    = Glu->xsup;
    supno   = Glu->supno;
    lsub    = Glu->lsub;
    xlsub   = Glu->xlsub;
    ucol    = (doublecomplex *) Glu->ucol;
    usub    = Glu->usub;
    xusub   = Glu->xusub;
    nzumax  = Glu->nzumax;

    *sum = zero;
    if (drop_rule == NODROP) {
    drop_tol = -1.0, quota = Glu->n;
    }

    jsupno = supno[jcol];
    nextu  = xusub[jcol];
    k = nseg - 1;
    for (ksub = 0; ksub < nseg; ksub++) {
    krep = segrep[k--];
    ksupno = supno[krep];
    if ( ksupno != jsupno ) { /* 如果 ksupno 不等于 jsupno，则进入 ucol[] */
        kfnz = repfnz[krep];
        if ( kfnz != EMPTY ) {    /* 如果 kfnz 不是空值，则表示非零 U 段 */

        fsupc = xsup[ksupno];
        isub = xlsub[fsupc] + kfnz - fsupc;  /* 计算在 lsub 中的索引位置 */
        segsze = krep - kfnz + 1;   /* 计算段的大小 */

        new_next = nextu + segsze;  /* 计算新的 nextu 的位置 */
        while ( new_next > nzumax ) {  /* 如果超过了 nzumax，则扩展内存 */
            if ((mem_error = zLUMemXpand(jcol, nextu, UCOL, &nzumax,
                Glu)) != 0)
            return (mem_error);
            ucol = Glu->ucol;   /* 更新 ucol 的指针 */
            if ((mem_error = zLUMemXpand(jcol, nextu, USUB, &nzumax,
                Glu)) != 0)
            return (mem_error);
            usub = Glu->usub;   /* 更新 usub 的指针 */
            lsub = Glu->lsub;   /* 更新 lsub 的指针 */
        }

        for (i = 0; i < segsze; i++) {
            irow = lsub[isub++];   /* 获取 lsub 中的行索引 */
                 tmp = z_abs1(&dense[irow]);   /* 计算 dense 数组中元素的绝对值 */

            /* 第一个丢弃规则 */
            if (quota > 0 && tmp >= drop_tol) {
            if (tmp > d_max) d_max = tmp;   /* 更新 d_max */
            if (tmp < d_min) d_min = tmp;   /* 更新 d_min */
            usub[nextu] = perm_r[irow];   /* 更新 usub */
            ucol[nextu] = dense[irow];   /* 更新 ucol */
            nextu++;   /* 更新 nextu 的位置 */
            } else {
            switch (milu) {
                case SMILU_1:
                case SMILU_2:
                                z_add(sum, sum, &dense[irow]);   /* 对 sum 进行加法操作 */
                break;
                case SMILU_3:
                /* *sum += fabs(dense[irow]);*/   /* 计算 dense[irow] 的绝对值并添加到 sum */
                sum->r += tmp;   /* 将 tmp 加到 sum 的实部上 */
                break;
                case SILU:
                default:
                break;
            }
#ifdef DEBUG
            num_drop_U++;
#endif
            /* 如果定义了 DEBUG 宏，则增加 num_drop_U 计数 */

            }
            /* 结束当前的 if 语句块 */

            dense[irow] = zero;
        }
        /* 将 dense[irow] 设置为 zero */

        }
        /* 结束当前的 if 语句块 */

    }
    /* 结束当前的 for 循环 */

    } /* for each segment... */
    /* 结束当前的注释段落，说明前面的 for 循环是针对每个段的 */

    xusub[jcol + 1] = nextu;      /* Close U[*,jcol] */
    /* 设置 xusub[jcol + 1] 的值为 nextu，表示关闭 U 矩阵的第 jcol 列 */
    m = xusub[jcol + 1] - xusub[jcol];
    /* 计算当前列的非零元素个数 */

    /* second dropping rule */
    /* 第二个丢弃规则 */

    if (drop_rule & DROP_SECONDARY && m > quota) {
    /* 如果设置了 DROP_SECONDARY 并且当前列的非零元素个数 m 超过了 quota */

    register double tol = d_max;
    /* 声明并初始化 tol 为 d_max */

    register int m0 = xusub[jcol] + m - 1;
    /* 计算 m0 的值 */

    if (quota > 0) {
        /* 如果 quota 大于 0 */

        if (drop_rule & DROP_INTERP) {
        /* 如果设置了 DROP_INTERP */

        d_max = 1.0 / d_max; d_min = 1.0 / d_min;
        /* 更新 d_max 和 d_min 的值 */

        tol = 1.0 / (d_max + (d_min - d_max) * quota / m);
        /* 更新 tol 的值 */
        } else {
                i_1 = xusub[jcol];
                /* 设置 i_1 的初始值 */

                for (i = 0; i < m; ++i, ++i_1) work[i] = z_abs1(&ucol[i_1]);
        /* 遍历并计算 work 数组的值 */

        tol = dqselect(m, work, quota);
        /* 使用 dqselect 计算 tol 的值 */
#if 0
        A = &ucol[xusub[jcol]];
        /* 设置 A 指针的值 */

        for (i = 0; i < m; i++) work[i] = i;
        /* 初始化 work 数组的值 */

        qsort(work, m, sizeof(int), _compare_);
        /* 对 work 数组进行排序 */

        tol = fabs(usub[xusub[jcol] + work[quota]]);
        /* 更新 tol 的值 */
#endif
        }
    }
    /* 结束当前的 if 语句块 */

    for (i = xusub[jcol]; i <= m0; ) {
        /* 循环直到 i 大于 m0 */

        if (z_abs1(&ucol[i]) <= tol) {
        /* 如果 ucol[i] 的绝对值小于等于 tol */

        switch (milu) {
            /* 根据 milu 的值进行不同的处理 */

            case SMILU_1:
            case SMILU_2:
            z_add(sum, sum, &ucol[i]);
            /* 调用 z_add 函数更新 sum 的值 */
            break;

            case SMILU_3:
            sum->r += tmp;
            /* 更新 sum->r 的值 */
            break;

            case SILU:
            default:
            break;
        }
        /* 结束 switch 语句块 */

        ucol[i] = ucol[m0];
        /* 更新 ucol[i] 的值 */
        usub[i] = usub[m0];
        /* 更新 usub[i] 的值 */
        m0--;
        /* 减少 m0 的值 */
        m--;
#ifdef DEBUG
        num_drop_U++;
        /* 如果定义了 DEBUG 宏，则增加 num_drop_U 计数 */
#endif
        xusub[jcol + 1]--;
        /* 减少 xusub[jcol + 1] 的值 */
        continue;
        /* 继续下一次循环 */
        }
        i++;
    }
    /* 结束当前的 for 循环 */

    }

    if (milu == SMILU_2) {
        sum->r = z_abs1(sum); sum->i = 0.0;
    }
    /* 如果 milu 等于 SMILU_2，则更新 sum 的实部 */

    if (milu == SMILU_3) sum->i = 0.0;
    /* 如果 milu 等于 SMILU_3，则更新 sum 的虚部 */

    *nnzUj += m;
    /* 更新 nnzUj 指向的值 */

    return 0;
}
/* 函数结束 */
```