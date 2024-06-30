# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\ilu_ddrop_row.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file ilu_ddrop_row.c
 * \brief Drop small rows from L
 *
 * <pre>
 * -- SuperLU routine (version 4.1) --
 * Lawrence Berkeley National Laboratory.
 * June 30, 2009
 * </pre>
 */

#include <math.h>
#include <stdlib.h>
#include "slu_ddefs.h"

extern void dswap_(int *, double [], int *, double [], int *);
extern void daxpy_(int *, double *, double [], int *, double [], int *);
extern void dcopy_(int *, double [], int *, double [], int *);
extern double dasum_(int *, double *, int *);
extern double dnrm2_(int *, double *, int *);
extern double dnrm2_(int *, double [], int *);
extern int idamax_(int *, double [], int *);

#if 0
static double *A;  /* used in _compare_ only */
static int _compare_(const void *a, const void *b)
{
    register int *x = (int *)a, *y = (int *)b;
    if (A[*x] - A[*y] > 0.0) return -1;
    else if (A[*x] - A[*y] < 0.0) return 1;
    else return 0;
}
#endif

/*! \brief
 * <pre>
 * Purpose
 * =======
 *    ilu_ddrop_row() - Drop some small rows from the previous 
 *    supernode (L-part only).
 * </pre>
 */
int ilu_ddrop_row(
    superlu_options_t *options, /* options */
    int    first,        /* index of the first column in the supernode */
    int    last,        /* index of the last column in the supernode */
    double drop_tol,    /* dropping parameter */
    int    quota,        /* maximum nonzero entries allowed */
    int    *nnzLj,        /* in/out number of nonzeros in L(:, 1:last) */
    double *fill_tol,   /* in/out - on exit, fill_tol=-num_zero_pivots,
                 * does not change if options->ILU_MILU != SMILU1 */
    GlobalLU_t *Glu,    /* modified */
    double dwork[],   /* working space
                         * the length of dwork[] should be no less than
                 * the number of rows in the supernode */
    double dwork2[], /* working space with the same size as dwork[],
                 * used only by the second dropping rule */
    int    lastc        /* if lastc == 0, there is nothing after the
                 * working supernode [first:last];
                 * if lastc == 1, there is one more column after
                 * the working supernode. */ )
{
    register int i, j, k, m1;
    register int nzlc; /* number of nonzeros in column last+1 */
    int_t xlusup_first, xlsub_first;
    int m, n; /* m x n is the size of the supernode */
    int r = 0; /* number of dropped rows */
    register double *temp;
    register double *lusup = (double *) Glu->lusup;
    int_t *lsub = Glu->lsub;
    int_t *xlsub = Glu->xlsub;
    int_t *xlusup = Glu->xlusup;
    register double d_max = 0.0, d_min = 1.0;
    int    drop_rule = options->ILU_DropRule;

    // 计算超节点的行列数
    m = last - first + 1;
    n = xlsub[first+1] - xlsub[first];

    // 循环处理每一行
    for (i = 0; i < m; ++i) {
        // 当前行在 L 中的起始位置
        xlusup_first = xlusup[first] + i * n;
        // 当前行的行指标
        xlsub_first = xlsub[first] + i;

        // 检查是否存在下一列
        if (lastc == 0) {
            nzlc = 0;
        } else {
            // 下一列的非零元素数目
            nzlc = xlsub[first+1] - xlsub_first;
        }

        // 计算第 i 行的最大和最小值
        d_max = -1.0;
        d_min = 1.0;
        for (j = 0; j < nzlc; ++j) {
            double tmp = fabs(lusup[xlusup_first + j]);
            if (tmp > d_max) d_max = tmp;
            if (tmp < d_min) d_min = tmp;
        }

        // 根据 ILU drop rule 判断是否要丢弃当前行
        if (drop_rule == DROP_BASIC) {
            if (d_min < drop_tol * d_max && quota > 0) {
                // 将当前行标记为要丢弃的行
                lsub[xlsub_first] = EMPTY;
                r++; // 增加丢弃行数计数器
                quota--; // 减少剩余非零元素配额
            }
        } else if (drop_rule == DROP_SECONDARY) {
            double d_eps = drop_tol * d_max;
            // 判断是否符合第二规则要求
            if (d_min < d_eps && d_eps > 0.0 && quota > 0) {
                // 执行第二规则下的丢弃操作
                d_eps *= fill_tol[i];
                temp = &dwork[xlsub_first];
                for (k = 0; k < nzlc; ++k) {
                    *temp += lusup[xlusup_first + k] * d_eps;
                    temp++;
                }
                lsub[xlsub_first] = EMPTY;
                r++; // 增加丢弃行数计数器
                quota--; // 减少剩余非零元素配额
            }
        }
    }

    // 返回实际丢弃的行数
    return r;
}
    milu_t milu = options->ILU_MILU;
    norm_t nrm = options->ILU_Norm;
    double zero = 0.0;
    double one = 1.0;
    double none = -1.0;
    int i_1 = 1;
    int inc_diag; /* inc_diag = m + 1 */
    int nzp = 0;  /* number of zero pivots */
    double alpha = pow((double)(Glu->n), -1.0 / options->ILU_MILU_Dim);

    xlusup_first = xlusup[first];
    xlsub_first = xlsub[first];
    m = xlusup[first + 1] - xlusup_first;
    n = last - first + 1;
    m1 = m - 1;
    inc_diag = m + 1;
    nzlc = lastc ? (xlusup[last + 2] - xlusup[last + 1]) : 0;
    temp = dwork - n;

    /* Quick return if nothing to do. */
    if (m == 0 || m == n || drop_rule == NODROP)
    {
    *nnzLj += m * n;
    return 0;
    }

    /* basic dropping: ILU(tau) */
    for (i = n; i <= m1; )
    {
    /* the average abs value of ith row */
    switch (nrm)
    {
        case ONE_NORM:
        // 计算第i行的元素的一范数平均值
        temp[i] = dasum_(&n, &lusup[xlusup_first + i], &m) / (double)n;
        break;
        case TWO_NORM:
        // 计算第i行的元素的二范数
        temp[i] = dnrm2_(&n, &lusup[xlusup_first + i], &m)
            / sqrt((double)n);
        break;
        case INF_NORM:
        default:
        // 找到第i行绝对值最大的元素
        k = idamax_(&n, &lusup[xlusup_first + i], &m) - 1;
        temp[i] = fabs(lusup[xlusup_first + i + m * k]);
        break;
    }

    /* drop small entries due to drop_tol */
    if (drop_rule & DROP_BASIC && temp[i] < drop_tol)
    {
        r++;
        /* drop the current row and move the last undropped row here */
        if (r > 1) /* add to last row */
        {
        /* accumulate the sum (for MILU) */
        switch (milu)
        {
            case SMILU_1:
            case SMILU_2:
            // 更新最后一行与当前行的和
            daxpy_(&n, &one, &lusup[xlusup_first + i], &m,
                &lusup[xlusup_first + m - 1], &m);
            break;
            case SMILU_3:
            // 对每列进行累加（用于SMILU_3）
            for (j = 0; j < n; j++)
                lusup[xlusup_first + (m - 1) + j * m] +=
                    fabs(lusup[xlusup_first + i + j * m]);
            break;
            case SILU:
            default:
            break;
        }
        // 将最后一行复制到当前行
        dcopy_(&n, &lusup[xlusup_first + m1], &m,
                       &lusup[xlusup_first + i], &m);
        } /* if (r > 1) */
        else /* move to last row */
        {
        // 将最后一行与当前行交换
        dswap_(&n, &lusup[xlusup_first + m1], &m,
            &lusup[xlusup_first + i], &m);
        if (milu == SMILU_3)
            // 对每列取绝对值（用于SMILU_3）
            for (j = 0; j < n; j++) {
            lusup[xlusup_first + m1 + j * m] =
                fabs(lusup[xlusup_first + m1 + j * m]);
                    }
        }
        // 更新lsub数组
        lsub[xlsub_first + i] = lsub[xlsub_first + m1];
        m1--;
        continue;
    } /* if dropping */
    else
    {
        // 更新d_max和d_min
        if (temp[i] > d_max) d_max = temp[i];
        if (temp[i] < d_min) d_min = temp[i];
    }
    i++;
    } /* for */

    /* Secondary dropping: drop more rows according to the quota. */
    quota = ceil((double)quota / (double)n);
    if (drop_rule & DROP_SECONDARY && m - r > quota)
    {
    // 更新tol为d_max
    register double tol = d_max;
    /* 计算第二个丢弃容差 */
    if (quota > n)
    {
        /* 如果使用插值法丢弃 */
        if (drop_rule & DROP_INTERP)
        {
            d_max = 1.0 / d_max; d_min = 1.0 / d_min;
            tol = 1.0 / (d_max + (d_min - d_max) * quota / (m - n - r));
        }
        else /* 使用快速选择 */
        {
            int len = m1 - n + 1;
            /* 复制数组 dwork 到 dwork2 */
            dcopy_(&len, dwork, &i_1, dwork2, &i_1);
            /* 使用快速选择算法计算丢弃容差 */
            tol = dqselect(len, dwork2, quota - n);
        # 注释掉的代码段，不会被编译或执行
#if 0
        register int *itemp = iwork - n;
        A = temp;
        for (i = n; i <= m1; i++) itemp[i] = i;
        qsort(iwork, m1 - n + 1, sizeof(int), _compare_);
        tol = temp[itemp[quota]];
#endif
        }
    }

    # 遍历从 n 到 m1 的每一行
    for (i = n; i <= m1; )
    {
        if (temp[i] <= tol)
        {
        register int j;
        r++;
        /* 删除当前行并将最后一行移到这里 */
        if (r > 1) /* 添加到最后一行 */
        {
            /* 累加和（用于 MILU） */
            switch (milu)
            {
            case SMILU_1:
            case SMILU_2:
                daxpy_(&n, &one, &lusup[xlusup_first + i], &m,
                    &lusup[xlusup_first + m - 1], &m);
                break;
            case SMILU_3:
                for (j = 0; j < n; j++)
                    lusup[xlusup_first + (m - 1) + j * m] +=
                        fabs(lusup[xlusup_first + i + j * m]);
                break;
            case SILU:
            default:
                break;
            }
            dcopy_(&n, &lusup[xlusup_first + m1], &m,
                &lusup[xlusup_first + i], &m);
        } /* if (r > 1) */
        else /* 移动到最后一行 */
        {
            dswap_(&n, &lusup[xlusup_first + m1], &m,
                &lusup[xlusup_first + i], &m);
            if (milu == SMILU_3)
                for (j = 0; j < n; j++) {
                    lusup[xlusup_first + m1 + j * m] =
                        fabs(lusup[xlusup_first + m1 + j * m]);
                }
        }
        lsub[xlsub_first + i] = lsub[xlsub_first + m1];
        m1--;
        temp[i] = temp[m1];

        continue;
        }
        i++;

    } /* for */

    } /* if secondary dropping */

    # 将 temp 数组从 n 到 m 全部置为 0.0
    for (i = n; i < m; i++) temp[i] = 0.0;

    # 如果没有行被丢弃，则直接返回
    if (r == 0)
    {
    *nnzLj += m * n;
    return 0;
    }

    /* 将丢弃的条目添加到对角线上 */
    if (milu != SILU)
    {
    register int j;
    double t;
    double omega;
    for (j = 0; j < n; j++)
    {
        t = lusup[xlusup_first + (m - 1) + j * m];
            if (t == zero) continue;
        if (t > zero)
        omega = SUPERLU_MIN(2.0 * (1.0 - alpha) / t, 1.0);
        else
        omega = SUPERLU_MAX(2.0 * (1.0 - alpha) / t, -1.0);
        t *= omega;

         switch (milu)
        {
        case SMILU_1:
            if (t != none) {
            lusup[xlusup_first + j * inc_diag] *= (one + t);
                    }
            else
            {
            lusup[xlusup_first + j * inc_diag] *= *fill_tol;
#ifdef DEBUG
            printf("[1] ZERO PIVOT: FILL col %d.\n", first + j);
            fflush(stdout);
#endif
            nzp++;
            }
            break;
        case SMILU_2:
            lusup[xlusup_first + j * inc_diag] *= (1.0 + fabs(t));
            break;
        case SMILU_3:
            lusup[xlusup_first + j * inc_diag] *= (one + t);
            break;
        case SILU:
        default:
            break;
        }
    }
    if (nzp > 0) *fill_tol = -nzp;
    }
    /* 移除内存中被丢弃的条目并修正指针。 */
    m1 = m - r;  // 计算新的行数，减去被丢弃的行数
    for (j = 1; j < n; j++)
    {
        register int tmp1, tmp2;
        tmp1 = xlusup_first + j * m1;  // 计算新的列偏移
        tmp2 = xlusup_first + j * m;   // 计算旧的列偏移
        for (i = 0; i < m1; i++)
            lusup[i + tmp1] = lusup[i + tmp2];  // 重新调整 L 上三角矩阵的数据
    }
    for (i = 0; i < nzlc; i++)
        lusup[xlusup_first + i + n * m1] = lusup[xlusup_first + i + n * m];  // 更新 L 上三角矩阵中额外的列数据

    for (i = 0; i < nzlc; i++)
        lsub[xlsub[last + 1] - r + i] = lsub[xlsub[last + 1] + i];  // 更新 L 列表中剩余子列表的指针

    for (i = first + 1; i <= last + 1; i++)
    {
        xlusup[i] -= r * (i - first);  // 调整 L 上三角矩阵列指针
        xlsub[i] -= r;  // 调整 L 列表中剩余子列表的指针
    }

    if (lastc)
    {
        xlusup[last + 2] -= r * n;  // 如果有剩余的列数据，调整 L 上三角矩阵列指针
        xlsub[last + 2] -= r;  // 如果有剩余的列数据，调整 L 列表中剩余子列表的指针
    }

    *nnzLj += (m - r) * n;  // 更新 L 上三角矩阵非零元素的数量
    return r;  // 返回更新后的行数
}


注释：


# 这是一个单独的右花括号 '}'，用于闭合一个代码块或者数据结构。
```