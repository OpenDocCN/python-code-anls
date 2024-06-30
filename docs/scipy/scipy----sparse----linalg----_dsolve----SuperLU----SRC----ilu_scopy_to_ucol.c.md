# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\ilu_scopy_to_ucol.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file ilu_scopy_to_ucol.c
 * \brief Copy a computed column of U to the compressed data structure
 * and drop some small entries
 *
 * <pre>
 * -- SuperLU routine (version 4.1) --
 * Lawrence Berkeley National Laboratory
 * November, 2010
 * </pre>
 */

#include "slu_sdefs.h"

#ifdef DEBUG
int num_drop_U;
#endif

extern void scopy_(int *, float [], int *, float [], int *);

#if 0
static float *A;  /* used in _compare_ only */
static int _compare_(const void *a, const void *b)
{
    register int *x = (int *)a, *y = (int *)b;
    register double xx = fabs(A[*x]), yy = fabs(A[*y]);
    if (xx > yy) return -1;
    else if (xx < yy) return 1;
    else return 0;
}
#endif

/**
 * @brief Copy a computed column of U to the compressed data structure and drop some small entries
 *
 * This function copies a computed column of U (a sparse matrix) to a compressed data structure
 * and potentially drops small entries based on specified criteria.
 *
 * @param jcol      Column index of U to copy
 * @param nseg      Number of segments in U
 * @param segrep    Segment representatives
 * @param repfnz    First non-zero in each segment
 * @param perm_r    Permutation vector for rows
 * @param dense     Dense vector (modified to zero on return)
 * @param drop_rule Drop rule identifier
 * @param milu      Modified ILU preconditioner parameters
 * @param drop_tol  Drop tolerance
 * @param quota     Maximum nonzero entries allowed
 * @param sum       Output: sum of dropped entries
 * @param nnzUj     Input/Output: number of nonzero entries in column j of U
 * @param Glu       Global LU decomposition structure (modified)
 * @param work      Working space
 *
 * @return integer indicating success or failure
 */
int
ilu_scopy_to_ucol(
          int     jcol,       /* in */
          int     nseg,       /* in */
          int     *segrep,    /* in */
          int     *repfnz,    /* in */
          int     *perm_r,    /* in */
          float   *dense,     /* modified - reset to zero on return */
          int     drop_rule,  /* in */
          milu_t  milu,       /* in */
          double  drop_tol,   /* in */
          int     quota,      /* in */
          float   *sum,       /* out */
          int     *nnzUj,     /* in - out */
          GlobalLU_t *Glu,    /* modified */
          float   *work       /* working space with minimum size n */
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
    float    *ucol;
    int_t     *usub, *xusub;
    int_t     nzumax;
    int       m; /* number of entries in the nonzero U-segments */
    register float d_max = 0.0, d_min = 1.0 / smach("Safe minimum");
    register double tmp;
    float zero = 0.0;
    int i_1 = 1;

    xsup    = Glu->xsup;
    supno   = Glu->supno;
    lsub    = Glu->lsub;
    xlsub   = Glu->xlsub;
    ucol    = (float *) Glu->ucol;
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
    if ( ksupno != jsupno ) { /* 如果 ksupno 不等于 jsupno，则进入 ucol[] 中 */
        kfnz = repfnz[krep];
        if ( kfnz != EMPTY ) {    /* 如果 kfnz 不等于 EMPTY，则表示存在非零 U-段 */

            fsupc = xsup[ksupno];
            isub = xlsub[fsupc] + kfnz - fsupc;
            segsze = krep - kfnz + 1;

            new_next = nextu + segsze;
            while ( new_next > nzumax ) {
                /* 如果 new_next 超出 nzumax，则扩展内存 */
                if ((mem_error = sLUMemXpand(jcol, nextu, UCOL, &nzumax,
                    Glu)) != 0)
                    return (mem_error);
                ucol = Glu->ucol;
                if ((mem_error = sLUMemXpand(jcol, nextu, USUB, &nzumax,
                    Glu)) != 0)
                    return (mem_error);
                usub = Glu->usub;
                lsub = Glu->lsub;
            }

            for (i = 0; i < segsze; i++) {
                irow = lsub[isub++];
                tmp = fabs(dense[irow]);

                /* 第一个放弃规则 */
                if (quota > 0 && tmp >= drop_tol) {
                    if (tmp > d_max) d_max = tmp;
                    if (tmp < d_min) d_min = tmp;
                    usub[nextu] = perm_r[irow];
                    ucol[nextu] = dense[irow];
                    nextu++;
                } else {
                    switch (milu) {
                        case SMILU_1:
                        case SMILU_2:
                            /* *sum += dense[irow]; */
                            *sum += dense[irow];
                            break;
                        case SMILU_3:
                            /* *sum += fabs(dense[irow]); */
                            *sum += tmp;
                            break;
                        case SILU:
                        default:
                            break;
                    }
                }
            }
        }
    }
#ifdef DEBUG
            num_drop_U++;
#endif
            // 如果定义了 DEBUG 宏，则增加 num_drop_U 计数
            }
            // 结束当前的块语句

            // 将 dense[irow] 设置为 zero
            dense[irow] = zero;
        }

        }

    }

    } /* for each segment... */

    // 关闭 U[*,jcol]
    xusub[jcol + 1] = nextu;
    // 计算 U[*,jcol] 的长度
    m = xusub[jcol + 1] - xusub[jcol];

    /* second dropping rule */
    // 如果满足第二个丢弃规则，并且长度 m 大于 quota
    if (drop_rule & DROP_SECONDARY && m > quota) {
    register double tol = d_max;
    register int m0 = xusub[jcol] + m - 1;

    // 如果 quota 大于 0
    if (quota > 0) {
        // 如果设置了 DROP_INTERP 标志
        if (drop_rule & DROP_INTERP) {
        // 调整 d_max 和 d_min 的值
        d_max = 1.0 / d_max; d_min = 1.0 / d_min;
        // 计算容差 tol
        tol = 1.0 / (d_max + (d_min - d_max) * quota / m);
        } else {
        // 复制 ucol[xusub[jcol]] 到 work
        scopy_(&m, &ucol[xusub[jcol]], &i_1, work, &i_1);
        // 使用 sqselect 函数选择值并计算容差 tol
        tol = sqselect(m, work, quota);
#if 0
        // A = &ucol[xusub[jcol]];
        // 初始化 work 数组
        // 使用 _compare_ 函数排序 work 数组
        // 计算 tol 为 usub[xusub[jcol] + work[quota]] 的绝对值
#endif
        }
    }
    // 遍历范围为 xusub[jcol] 到 m0
    for (i = xusub[jcol]; i <= m0; ) {
        // 如果 ucol[i] 的绝对值小于等于 tol
        if (fabs(ucol[i]) <= tol) {
        // 根据 milu 的值进行处理
        switch (milu) {
            case SMILU_1:
            case SMILU_2:
            // 将 ucol[i] 的值加到 *sum 上
            *sum += ucol[i];
            break;
            case SMILU_3:
            // 将 ucol[i] 的绝对值加到 *sum 上
            *sum += fabs(ucol[i]);
            break;
            case SILU:
            default:
            // 默认情况不进行任何操作
            break;
        }
        // 移动数据以减少 m0 和 m 的大小
        ucol[i] = ucol[m0];
        usub[i] = usub[m0];
        m0--;
        m--;
#ifdef DEBUG
        // 如果定义了 DEBUG 宏，则增加 num_drop_U 计数
        num_drop_U++;
#endif
        // 减少 xusub[jcol + 1] 的值
        xusub[jcol + 1]--;
        continue;
        }
        // 增加 i 的值
        i++;
    }
    }

    // 如果 milu 的值是 SMILU_2，则取 *sum 的绝对值
    if (milu == SMILU_2) *sum = fabs(*sum);

    // 增加 *nnzUj 的值
    *nnzUj += m;

    // 返回 0 表示执行成功
    return 0;
}
```