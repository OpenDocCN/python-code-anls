# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\ilu_dcopy_to_ucol.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file ilu_dcopy_to_ucol.c
 * \brief Copy a computed column of U to the compressed data structure
 * and drop some small entries
 *
 * <pre>
 * -- SuperLU routine (version 4.1) --
 * Lawrence Berkeley National Laboratory
 * November, 2010
 * </pre>
 */

#include "slu_ddefs.h"

#ifdef DEBUG
int num_drop_U;
#endif

extern void dcopy_(int *, double [], int *, double [], int *);

#if 0
static double *A;  /* used in _compare_ only */
static int _compare_(const void *a, const void *b)
{
    register int *x = (int *)a, *y = (int *)b;
    register double xx = fabs(A[*x]), yy = fabs(A[*y]);
    if (xx > yy) return -1;
    else if (xx < yy) return 1;
    else return 0;
}
#endif

/*! \brief
 * Copy a computed column of U to the compressed data structure and drop some small entries.
 *
 * This function implements a routine in SuperLU (version 4.1) that copies a computed column
 * of the matrix U to a compressed data structure while potentially dropping entries based on
 * a specified drop rule.
 *
 * \param jcol      Column index of U to be copied (input).
 * \param nseg      Number of segments in the column (input).
 * \param segrep    Segment representation array (input).
 * \param repfnz    Representation of first non-zero in each segment (input).
 * \param perm_r    Permutation vector for rows (input).
 * \param dense     Dense vector to be modified/reset (modified - reset to zero on return).
 * \param drop_rule Drop rule indicator (input).
 * \param milu      Indicator for modified ILU preconditioning (input).
 * \param drop_tol  Drop tolerance value (input).
 * \param quota     Maximum nonzero entries allowed (input).
 * \param sum       Sum of dropped entries (output).
 * \param nnzUj     Number of nonzero entries in the U column (input-output).
 * \param Glu       Global LU data structure (modified).
 * \param work      Working space with minimum size n (input).
 *
 * \return          Integer indicating success (0) or failure (non-zero).
 */
int
ilu_dcopy_to_ucol(
          int     jcol,       /* in */
          int     nseg,       /* in */
          int     *segrep,  /* in */
          int     *repfnz,  /* in */
          int     *perm_r,  /* in */
          double     *dense,   /* modified - reset to zero on return */
          int       drop_rule,/* in */
          milu_t     milu,       /* in */
          double     drop_tol, /* in */
          int     quota,    /* maximum nonzero entries allowed */
          double     *sum,       /* out - the sum of dropped entries */
          int     *nnzUj,   /* in - out */
          GlobalLU_t *Glu,       /* modified */
          double     *work       /* working space with minimum size n,
                    * used by the second dropping rule */
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
    double    *ucol;
    int_t     *usub, *xusub;
    int_t     nzumax;
    int       m; /* number of entries in the nonzero U-segments */
    register double d_max = 0.0, d_min = 1.0 / dmach("Safe minimum");
    register double tmp;
    double zero = 0.0;
    int i_1 = 1;

    xsup    = Glu->xsup;
    supno   = Glu->supno;
    lsub    = Glu->lsub;
    xlsub   = Glu->xlsub;
    ucol    = (double *) Glu->ucol;
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
    if ( ksupno != jsupno ) { /* 如果 ksupno 不等于 jsupno，则进入 ucol[] 段 */
        kfnz = repfnz[krep];
        if ( kfnz != EMPTY ) {    /* 如果 kfnz 不等于 EMPTY，则表示存在非零 U 段 */

        fsupc = xsup[ksupno];  /* 获取列的起始位置 */
        isub = xlsub[fsupc] + kfnz - fsupc;  /* 计算子列的起始位置 */
        segsze = krep - kfnz + 1;  /* 计算段的大小 */

        new_next = nextu + segsze;  /* 计算新的下一个位置 */
        while ( new_next > nzumax ) {  /* 如果超过预分配的内存上限 */
            if ((mem_error = dLUMemXpand(jcol, nextu, UCOL, &nzumax,
                Glu)) != 0)  /* 扩展 UCOL 的内存 */
            return (mem_error);
            ucol = Glu->ucol;  /* 更新 ucol 指针 */
            if ((mem_error = dLUMemXpand(jcol, nextu, USUB, &nzumax,
                Glu)) != 0)  /* 扩展 USUB 的内存 */
            return (mem_error);
            usub = Glu->usub;  /* 更新 usub 指针 */
            lsub = Glu->lsub;  /* 更新 lsub 指针 */
        }

        for (i = 0; i < segsze; i++) {  /* 遍历段中的每个元素 */
            irow = lsub[isub++];  /* 获取行索引 */
            tmp = fabs(dense[irow]);  /* 计算绝对值 */

            /* 第一种丢弃规则 */
            if (quota > 0 && tmp >= drop_tol) {  /* 如果配额大于 0 并且 tmp 大于或等于丢弃容忍度 */
            if (tmp > d_max) d_max = tmp;  /* 更新 d_max */
            if (tmp < d_min) d_min = tmp;  /* 更新 d_min */
            usub[nextu] = perm_r[irow];  /* 存储列排列的行索引 */
            ucol[nextu] = dense[irow];  /* 存储列的值 */
            nextu++;  /* 更新下一个位置 */
            } else {
            switch (milu) {  /* 根据 milu 的值进行不同的处理 */
                case SMILU_1:
                case SMILU_2:
                *sum += dense[irow];  /* 累加 dense[irow] 到 *sum */
                break;
                case SMILU_3:
                /* *sum += fabs(dense[irow]);*/  /* 第三种情况下的处理 */
                *sum += tmp;  /* 累加 tmp 到 *sum */
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
            /* 结束 if 块 */

            dense[irow] = zero;
        }

        }

    }

    } /* for each segment... */

    xusub[jcol + 1] = nextu;      /* Close U[*,jcol] */
    /* 设置 U 的列 jcol 的结束位置 */

    m = xusub[jcol + 1] - xusub[jcol];
    /* 计算 U 的列 jcol 的非零元素数目 */

    /* second dropping rule */
    /* 第二个丢弃规则 */
    if (drop_rule & DROP_SECONDARY && m > quota) {
    /* 如果满足第二丢弃规则且列 jcol 的非零元素数目大于 quota */
    register double tol = d_max;
    /* 声明并初始化 tol 为 d_max */
    register int m0 = xusub[jcol] + m - 1;
    /* 声明并计算 m0 作为列 jcol 中非零元素的最后一个索引 */

    if (quota > 0) {
        /* 如果 quota 大于 0 */
        if (drop_rule & DROP_INTERP) {
        /* 如果定义了 DROP_INTERP */
        d_max = 1.0 / d_max; d_min = 1.0 / d_min;
        /* 更新 d_max 和 d_min 的倒数 */
        tol = 1.0 / (d_max + (d_min - d_max) * quota / m);
        /* 更新 tol 为新的丢弃阈值 */
        } else {
        /* 否则 */
        dcopy_(&m, &ucol[xusub[jcol]], &i_1, work, &i_1);
        /* 复制列 jcol 中的非零元素到 work 数组 */
        tol = dqselect(m, work, quota);
#if 0
        A = &ucol[xusub[jcol]];
        for (i = 0; i < m; i++) work[i] = i;
        qsort(work, m, sizeof(int), _compare_);
        tol = fabs(usub[xusub[jcol] + work[quota]]);
#endif
        }
    }
    for (i = xusub[jcol]; i <= m0; ) {
        /* 遍历列 jcol 中的非零元素 */
        if (fabs(ucol[i]) <= tol) {
        /* 如果元素的绝对值小于等于 tol */
        switch (milu) {
            /* 根据 milu 的值执行不同的操作 */
            case SMILU_1:
            case SMILU_2:
            *sum += ucol[i];
            /* 更新 sum 变量 */
            break;
            case SMILU_3:
            *sum += fabs(ucol[i]);
            /* 更新 sum 变量为绝对值 */
            break;
            case SILU:
            default:
            break;
        }
        ucol[i] = ucol[m0];
        /* 将当前元素替换为列 jcol 中最后一个非零元素 */
        usub[i] = usub[m0];
        /* 更新对应的行索引 */
        m0--;
        /* 更新 m0 为下一个位置 */
        m--;
#ifdef DEBUG
        num_drop_U++;
        /* 如果定义了 DEBUG 宏，则增加 num_drop_U 计数 */
#endif
        xusub[jcol + 1]--;
        /* 减少列 jcol 的非零元素数目 */
        continue;
        /* 继续循环 */
        }
        i++;
    }
    }

    if (milu == SMILU_2) *sum = fabs(*sum);
    /* 如果 milu 是 SMILU_2，则更新 sum 为其绝对值 */

    *nnzUj += m;
    /* 更新 nnzUj 变量 */

    return 0;
}
/* 结束函数 */
```