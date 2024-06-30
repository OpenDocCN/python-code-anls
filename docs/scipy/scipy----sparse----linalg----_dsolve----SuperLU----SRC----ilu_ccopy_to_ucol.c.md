# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\ilu_ccopy_to_ucol.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file ilu_ccopy_to_ucol.c
 * \brief Copy a computed column of U to the compressed data structure
 * and drop some small entries
 *
 * <pre>
 * -- SuperLU routine (version 4.1) --
 * Lawrence Berkeley National Laboratory
 * November, 2010
 * </pre>
 */

#include "slu_cdefs.h"

#ifdef DEBUG
int num_drop_U;
#endif

extern void ccopy_(int *, singlecomplex [], int *, singlecomplex [], int *);

#if 0
static singlecomplex *A;  /* used in _compare_ only */
static int _compare_(const void *a, const void *b)
{
    register int *x = (int *)a, *y = (int *)b;
    register float xx = c_abs1(&A[*x]), yy = c_abs1(&A[*y]);
    if (xx > yy) return -1;
    else if (xx < yy) return 1;
    else return 0;
}
#endif

/*! 
 * \brief Copy a computed column of U to the compressed data structure and drop some small entries.
 *
 * This function copies a computed column of U to the compressed data structure (ucol) and applies 
 * dropping rules to remove small entries. The dropped entries are accumulated in 'sum'.
 *
 * \param jcol      Column index of U to be copied.
 * \param nseg      Number of segments in the column.
 * \param segrep    Array of segment representatives.
 * \param repfnz    Array of row indices where each segment starts.
 * \param perm_r    Row permutation vector.
 * \param dense     Dense matrix representing U.
 * \param drop_rule Dropping rule identifier.
 * \param milu      MILU structure.
 * \param drop_tol  Dropping tolerance.
 * \param quota     Maximum nonzero entries allowed.
 * \param sum       Output parameter, sum of dropped entries.
 * \param nnzUj     Input and output parameter, number of nonzero entries in column jcol.
 * \param Glu       Global LU data structure.
 * \param work      Working space with minimum size n.
 *
 * \return          Integer status indicating success or failure.
 */
int
ilu_ccopy_to_ucol(
          int     jcol,       /* in */
          int     nseg,       /* in */
          int     *segrep,    /* in */
          int     *repfnz,    /* in */
          int     *perm_r,    /* in */
          singlecomplex     *dense,   /* modified - reset to zero on return */
          int       drop_rule,/* in */
          milu_t     milu,       /* in */
          double     drop_tol, /* in */
          int     quota,    /* maximum nonzero entries allowed */
          singlecomplex     *sum,       /* out - the sum of dropped entries */
          int     *nnzUj,   /* in - out */
          GlobalLU_t *Glu,       /* modified */
          float     *work       /* working space with minimum size n,
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
    singlecomplex    *ucol;
    int_t     *usub, *xusub;
    int_t     nzumax;
    int       m; /* number of entries in the nonzero U-segments */
    register float d_max = 0.0, d_min = 1.0 / smach("Safe minimum");
    register double tmp;
    singlecomplex zero = {0.0, 0.0};
    int i_1 = 1;

    xsup    = Glu->xsup;        /* Pointer to supernode starting indices */
    supno   = Glu->supno;       /* Pointer to supernode numbers */
    lsub    = Glu->lsub;        /* Pointer to L subscripts */
    xlsub   = Glu->xlsub;       /* Pointer to column pointers in L subscript array */
    ucol    = (singlecomplex *) Glu->ucol; /* Pointer to the beginning of ucol */
    usub    = Glu->usub;        /* Pointer to U subscripts */
    xusub   = Glu->xusub;       /* Pointer to column pointers in U subscript array */
    nzumax  = Glu->nzumax;      /* Maximum size of U-subscript array */

    *sum = zero;                /* Initialize sum of dropped entries to zero */
    if (drop_rule == NODROP) {  /* If no dropping is specified */
        drop_tol = -1.0, quota = Glu->n;  /* Set drop tolerance to -1 and quota to total number of columns */
    }

    jsupno = supno[jcol];       /* Supernode number of column jcol */
    nextu  = xusub[jcol];       /* Starting index into usub[] for column jcol */
    k = nseg - 1;               /* Initialize k to the last segment index */
    for (ksub = 0; ksub < nseg; ksub++) {
        krep = segrep[k--];     /* Segment representative */
        ksupno = supno[krep];   /* Supernode number associated with segment */

        /*! 
         * Gather data from dense[*] to global ucol[*].
         */
        segsze = repfnz[krep+1] - repfnz[krep];  /* Segment size */
        fsupc = xsup[ksupno];   /* First supernode column */
        for (kfnz = repfnz[krep]; kfnz < repfnz[krep+1]; kfnz++) {
            irow = perm_r[kfnz];    /* Permuted row index */
            isub = xlsub[fsupc]++;  /* Position in Lsub */
            lsub[isub] = irow;      /* Store row index in Lsub */
            ccopy_(&segsze, &dense[irow], &i_1, &ucol[nextu], &i_1);  /* Copy dense[*] to ucol[*] */
            nextu += segsze;        /* Increment to next position in ucol */
        }
    }

    return 0;   /* Return success */
}
    // 如果 ksupno 不等于 jsupno，则需要将数据放入 ucol[] 中
    if ( ksupno != jsupno ) { /* Should go into ucol[] */
        // 获取 repfnz[krep] 的值赋给 kfnz
        kfnz = repfnz[krep];
        // 如果 kfnz 不等于 EMPTY，则说明存在非零的 U 段

        // 获取 fsupc，即 xsup[ksupno] 的值
        fsupc = xsup[ksupno];
        // 计算 isub 的值，即 xlsub[fsupc] + kfnz - fsupc
        isub = xlsub[fsupc] + kfnz - fsupc;
        // 计算 segsze 的值，即 krep - kfnz + 1
        segsze = krep - kfnz + 1;

        // 计算 new_next 的值，即 nextu + segsze
        new_next = nextu + segsze;
        // 如果 new_next 超过 nzumax，则进行内存扩展
        while ( new_next > nzumax ) {
            // 如果 cLUMemXpand 操作出错，则返回 mem_error
            if ((mem_error = cLUMemXpand(jcol, nextu, UCOL, &nzumax,
                Glu)) != 0)
            return (mem_error);
            // 获取扩展后的 ucol 指针
            ucol = Glu->ucol;
            // 如果 cLUMemXpand 操作出错，则返回 mem_error
            if ((mem_error = cLUMemXpand(jcol, nextu, USUB, &nzumax,
                Glu)) != 0)
            return (mem_error);
            // 获取扩展后的 usub 和 lsub 指针
            usub = Glu->usub;
            lsub = Glu->lsub;
        }

        // 遍历 segsze 次
        for (i = 0; i < segsze; i++) {
            // 获取 irow，即 lsub[isub++] 的值
            irow = lsub[isub++];
            // 计算 tmp，即 dense[irow] 的绝对值
            tmp = c_abs1(&dense[irow]);

            // 第一个丢弃规则
            if (quota > 0 && tmp >= drop_tol) {
                // 如果 tmp 大于 d_max，则更新 d_max
                if (tmp > d_max) d_max = tmp;
                // 如果 tmp 小于 d_min，则更新 d_min
                if (tmp < d_min) d_min = tmp;
                // 将 perm_r[irow] 赋给 usub[nextu]
                usub[nextu] = perm_r[irow];
                // 将 dense[irow] 赋给 ucol[nextu]
                ucol[nextu] = dense[irow];
                // nextu 自增
                nextu++;
            } else {
                // 根据 milu 的值执行相应操作
                switch (milu) {
                    case SMILU_1:
                    case SMILU_2:
                        // 将 dense[irow] 加到 sum 上
                        c_add(sum, sum, &dense[irow]);
                        break;
                    case SMILU_3:
                        // 将 tmp 加到 sum->r 上
                        sum->r += tmp;
                        break;
                    case SILU:
                    default:
                        break;
                }
            }
        }
#ifdef DEBUG
            num_drop_U++;
#endif
            /* 如果定义了 DEBUG 宏，增加 num_drop_U 的计数 */

            }
            dense[irow] = zero;
        }

        }

    }

    } /* for each segment... */

    xusub[jcol + 1] = nextu;      /* 关闭 U[*,jcol] 的边界 */
    m = xusub[jcol + 1] - xusub[jcol];

    /* 第二个丢弃规则 */
    if (drop_rule & DROP_SECONDARY && m > quota) {
    register double tol = d_max;
    register int m0 = xusub[jcol] + m - 1;

    if (quota > 0) {
        if (drop_rule & DROP_INTERP) {
        d_max = 1.0 / d_max; d_min = 1.0 / d_min;
        tol = 1.0 / (d_max + (d_min - d_max) * quota / m);
        } else {
                i_1 = xusub[jcol];
                for (i = 0; i < m; ++i, ++i_1) work[i] = c_abs1(&ucol[i_1]);
        tol = sqselect(m, work, quota);
#if 0
        A = &ucol[xusub[jcol]];
        for (i = 0; i < m; i++) work[i] = i;
        qsort(work, m, sizeof(int), _compare_);
        tol = fabs(usub[xusub[jcol] + work[quota]]);
#endif
        }
    }
    /* 根据第二个丢弃规则丢弃元素 */
    for (i = xusub[jcol]; i <= m0; ) {
        if (c_abs1(&ucol[i]) <= tol) {
        switch (milu) {
            case SMILU_1:
            case SMILU_2:
            c_add(sum, sum, &ucol[i]);
            break;
            case SMILU_3:
            sum->r += tmp;
            break;
            case SILU:
            default:
            break;
        }
        ucol[i] = ucol[m0];
        usub[i] = usub[m0];
        m0--;
        m--;
#ifdef DEBUG
        num_drop_U++;
#endif
        xusub[jcol + 1]--;
        continue;
        }
        i++;
    }
    }

    if (milu == SMILU_2) {
        sum->r = c_abs1(sum); sum->i = 0.0;
    }
    if (milu == SMILU_3) sum->i = 0.0;

    *nnzUj += m;

    return 0;
}
/* 结束函数 */
```