# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\ilu_zdrop_row.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file ilu_zdrop_row.c
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
#include "slu_zdefs.h"

extern void zswap_(int *, doublecomplex [], int *, doublecomplex [], int *);
extern void zaxpy_(int *, doublecomplex *, doublecomplex [], int *, doublecomplex [], int *);
extern void zcopy_(int *, doublecomplex [], int *, doublecomplex [], int *);
extern void dcopy_(int *, double [], int *, double [], int *);
extern double dzasum_(int *, doublecomplex *, int *);
extern double dznrm2_(int *, doublecomplex *, int *);
extern double dnrm2_(int *, double [], int *);
extern int izamax_(int *, doublecomplex [], int *);

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
 *    ilu_zdrop_row() - Drop some small rows from the previous 
 *    supernode (L-part only).
 * </pre>
 */
int ilu_zdrop_row(
    superlu_options_t *options, /* options - SuperLU配置选项 */
    int    first,        /* index of the first column in the supernode - 超节点中第一列的索引 */
    int    last,        /* index of the last column in the supernode - 超节点中最后一列的索引 */
    double drop_tol,    /* dropping parameter - 丢弃阈值参数 */
    int    quota,        /* maximum nonzero entries allowed - 允许的最大非零条目数 */
    int    *nnzLj,        /* in/out number of nonzeros in L(:, 1:last) - L(:, 1:last)中非零元素的数目 */
    double *fill_tol,   /* in/out - on exit, fill_tol=-num_zero_pivots,
                 * does not change if options->ILU_MILU != SMILU1 - 填充阈值，在退出时为-num_zero_pivots，
                 * 如果 options->ILU_MILU != SMILU1，则不变 */
    GlobalLU_t *Glu,    /* modified - 修改后的GlobalLU_t结构 */
    double dwork[],   /* working space
                         * the length of dwork[] should be no less than
                 * the number of rows in the supernode - 工作空间，dwork[]的长度应不少于超节点中的行数 */
    double dwork2[], /* working space with the same size as dwork[],
                 * used only by the second dropping rule - 与dwork[]大小相同的工作空间，
                 * 仅由第二个丢弃规则使用 */
    int    lastc        /* if lastc == 0, there is nothing after the
                 * working supernode [first:last];
                 * if lastc == 1, there is one more column after
                 * the working supernode. */ )
{
    register int i, j, k, m1;
    register int nzlc; /* number of nonzeros in column last+1 - 列last+1中的非零元素数 */
    int_t xlusup_first, xlsub_first;
    int m, n; /* m x n is the size of the supernode - 超节点的大小为 m x n */
    int r = 0; /* number of dropped rows - 被丢弃的行数 */
    register double *temp;
    register doublecomplex *lusup = (doublecomplex *) Glu->lusup;
    int_t *lsub = Glu->lsub;
    // 指针 xlsub 指向 Glu 结构体的 xlsub 数组
    int_t *xlsub = Glu->xlsub;
    // 指针 xlusup 指向 Glu 结构体的 xlusup 数组
    int_t *xlusup = Glu->xlusup;
    // 寄存器变量 d_max 和 d_min 初始化为 0.0 和 1.0
    register double d_max = 0.0, d_min = 1.0;
    // 从选项中获取 ILU_DropRule，即删除规则
    int    drop_rule = options->ILU_DropRule;
    // 从选项中获取 ILU_MILU，即 MILU 类型
    milu_t milu = options->ILU_MILU;
    // 从选项中获取 ILU_Norm，即规范类型
    norm_t nrm = options->ILU_Norm;
    // 复数常量 one 初始化为 {1.0, 0.0}
    doublecomplex one = {1.0, 0.0};
    // 复数常量 none 初始化为 {-1.0, 0.0}
    doublecomplex none = {-1.0, 0.0};
    // 整型常量 i_1 初始化为 1
    int i_1 = 1;
    // inc_diag 表示主对角线上元素的增量，初始化为 m + 1
    int inc_diag; /* inc_diag = m + 1 */
    // nzp 表示零主元的数量，初始化为 0
    int nzp = 0;  /* number of zero pivots */
    // alpha 初始化为 (Glu->n) 的 -1.0 / options->ILU_MILU_Dim 次幂
    double alpha = pow((double)(Glu->n), -1.0 / options->ILU_MILU_Dim);

    // 获取 xlusup 数组中的第 first 个元素
    xlusup_first = xlusup[first];
    // 获取 xlsub 数组中的第 first 个元素
    xlsub_first = xlsub[first];
    // m 表示当前行中元素的数量，计算方式为 xlusup[first+1] - xlusup_first
    m = xlusup[first + 1] - xlusup_first;
    // n 表示从 first 到 last 之间的列数
    n = last - first + 1;
    // m1 表示 m 减去 1
    m1 = m - 1;
    // inc_diag 表示主对角线上元素的增量，初始化为 m + 1
    inc_diag = m + 1;
    // 如果 lastc 为真，则 nzlc 表示 xlusup[last+2] 到 xlusup[last+1] 之间的非零元素数，否则为 0
    nzlc = lastc ? (xlusup[last + 2] - xlusup[last + 1]) : 0;
    // temp 指向 dwork 减去 n 处
    temp = dwork - n;

    /* 如果没有工作要做，则快速返回 */
    if (m == 0 || m == n || drop_rule == NODROP)
    {
        // 更新 nnzLj 指向的值，增加 m * n
        *nnzLj += m * n;
        // 返回 0 表示成功完成
        return 0;
    }

    /* 基本的删除策略: ILU(tau) */
    for (i = n; i <= m1; )
    {
        /* 第 i 行的平均绝对值 */
        switch (nrm)
        {
            // 采用一范数
            case ONE_NORM:
                temp[i] = dzasum_(&n, &lusup[xlusup_first + i], &m) / (double)n;
                break;
            // 采用二范数
            case TWO_NORM:
                temp[i] = dznrm2_(&n, &lusup[xlusup_first + i], &m)
                          / sqrt((double)n);
                break;
            // 采用无穷范数，或默认情况
            case INF_NORM:
            default:
                // 找出第 i 行中的最大元素的索引 k
                k = izamax_(&n, &lusup[xlusup_first + i], &m) - 1;
                // 计算第 i 行中第 k 列的复数绝对值
                temp[i] = z_abs1(&lusup[xlusup_first + i + m * k]);
                break;
        }

        /* 根据 drop_tol 删除小的条目 */
        if (drop_rule & DROP_BASIC && temp[i] < drop_tol)
        {
            r++;
            /* 删除当前行并将最后一个未删除的行移到此处 */
            if (r > 1) /* 添加到最后一行 */
            {
                /* 累积和（用于 MILU） */
                switch (milu)
                {
                    // 第一种 SMILU 类型
                    case SMILU_1:
                    case SMILU_2:
                        // 将第 i 行的内容加到最后一行
                        zaxpy_(&n, &one, &lusup[xlusup_first + i], &m,
                               &lusup[xlusup_first + m - 1], &m);
                        break;
                    // 第三种 SMILU 类型
                    case SMILU_3:
                        // 对每一列进行累加
                        for (j = 0; j < n; j++)
                            lusup[xlusup_first + (m - 1) + j * m].r +=
                                z_abs1(&lusup[xlusup_first + i + j * m]);
                        break;
                    // SILU 类型或默认情况
                    case SILU:
                    default:
                        break;
                }
                // 将最后一行的内容复制到第 i 行
                zcopy_(&n, &lusup[xlusup_first + m1], &m,
                       &lusup[xlusup_first + i], &m);
            } /* if (r > 1) */
            else /* 移动到最后一行 */
            {
                // 交换最后一行和第 i 行的内容
                zswap_(&n, &lusup[xlusup_first + m1], &m,
                       &lusup[xlusup_first + i], &m);
                // 对于 SMILU_3，将每一列的复数绝对值置于最后一行
                if (milu == SMILU_3)
                    for (j = 0; j < n; j++) {
                        lusup[xlusup_first + m1 + j * m].r =
                            z_abs1(&lusup[xlusup_first + m1 + j * m]);
                        lusup[xlusup_first + m1 + j * m].i = 0.0;
                    }
            }
            // 更新 lsub 数组中第 i 个元素的值，将其设为最后一行对应的值
            lsub[xlsub_first + i] = lsub[xlsub_first + m1];
            // m1 减少，表示删除了一行
            m1--;
            // 继续处理下一行
            continue;
        } /* if dropping */
        else
        {
            // 如果 temp[i] 大于 d_max，则更新 d_max
            if (temp[i] > d_max) d_max = temp[i];
            // 如果 temp[i] 小于 d_min，则更新 d_min
            if (temp[i] < d_min) d_min = temp[i];
        }
        // 处理下一行
        i++;
    } /* for */
    /* Secondary dropping: drop more rows according to the quota. */
    // 计算次要删除：根据配额删除更多的行
    quota = ceil((double)quota / (double)n);  // 将配额向上取整，以适应行数
    if (drop_rule & DROP_SECONDARY && m - r > quota)
    {
    register double tol = d_max;

    /* Calculate the second dropping tolerance */
    // 计算第二次删除的容差
    if (quota > n)
    {
        if (drop_rule & DROP_INTERP) /* by interpolation */
        {
        d_max = 1.0 / d_max; d_min = 1.0 / d_min;
        tol = 1.0 / (d_max + (d_min - d_max) * quota / (m - n - r));
        }
        else /* by quick select */
        {
        int len = m1 - n + 1;
        dcopy_(&len, dwork, &i_1, dwork2, &i_1);
        tol = dqselect(len, dwork2, quota - n);
        }
    }
#if 0
        register int *itemp = iwork - n;
        // 定义指向iwork数组起始位置前n个位置的指针
        A = temp;
        // 将A指向temp数组
        for (i = n; i <= m1; i++) itemp[i] = i;
        // 初始化itemp数组，使其包含从n到m1的整数序列
        qsort(iwork, m1 - n + 1, sizeof(int), _compare_);
        // 对iwork数组中的元素进行快速排序，排序依据为_compare_函数
        tol = temp[itemp[quota]];
        // 设置tol为temp数组中第quota个元素对应的值
#endif
        }
    }

    for (i = n; i <= m1; )
    {
        if (temp[i] <= tol)
        {
        register int j;
        r++;
        /* drop the current row and move the last undropped row here */
        if (r > 1) /* add to last row */
        {
            /* accumulate the sum (for MILU) */
            switch (milu)
            {
            case SMILU_1:
            case SMILU_2:
                zaxpy_(&n, &one, &lusup[xlusup_first + i], &m,
                    &lusup[xlusup_first + m - 1], &m);
                // 执行向量加法操作，用于MILU的累加操作
                break;
            case SMILU_3:
                for (j = 0; j < n; j++)
                lusup[xlusup_first + (m - 1) + j * m].r +=
                     z_abs1(&lusup[xlusup_first + i + j * m]);
                // 对MILU类型为SMILU_3的情况，执行复数绝对值的累加操作
                break;
            case SILU:
            default:
                break;
            }
            zcopy_(&n, &lusup[xlusup_first + m1], &m,
                &lusup[xlusup_first + i], &m);
            // 执行向量复制操作，用于将m1行的数据复制到当前行i处
        } /* if (r > 1) */
        else /* move to last row */
        {
            zswap_(&n, &lusup[xlusup_first + m1], &m,
                &lusup[xlusup_first + i], &m);
            // 执行向量交换操作，用于将m1行的数据与当前行i处数据交换
            if (milu == SMILU_3)
            for (j = 0; j < n; j++) {
                lusup[xlusup_first + m1 + j * m].r =
                    z_abs1(&lusup[xlusup_first + m1 + j * m]);
                lusup[xlusup_first + m1 + j * m].i = 0.0;
                        }
            // 对于MILU类型为SMILU_3的情况，执行复数绝对值的处理
        }
        lsub[xlsub_first + i] = lsub[xlsub_first + m1];
        // 将lsub数组中第xlsub_first+i个元素设置为lsub数组中第xlsub_first+m1个元素的值
        m1--;
        // 将m1减1
        temp[i] = temp[m1];
        // 将temp数组中第i个元素设置为temp数组中第m1个元素的值

        continue;
        }
        i++;

    } /* for */

    } /* if secondary dropping */

    for (i = n; i < m; i++) temp[i] = 0.0;
    // 将temp数组中从n到m-1的元素设置为0.0

    if (r == 0)
    {
    *nnzLj += m * n;
    // 将nnzLj指向的值增加m乘以n
    return 0;
    // 返回0
    }

    /* add dropped entries to the diagnal */
    if (milu != SILU)
    {
    register int j;
    doublecomplex t;
    double omega;
    for (j = 0; j < n; j++)
    {
        t = lusup[xlusup_first + (m - 1) + j * m];
            if (t.r == 0.0 && t.i == 0.0) continue;
            omega = SUPERLU_MIN(2.0 * (1.0 - alpha) / z_abs1(&t), 1.0);
        zd_mult(&t, &t, omega);

         switch (milu)
        {
        case SMILU_1:
            if ( !(z_eq(&t, &none)) ) {
                        z_add(&t, &t, &one);
                        zz_mult(&lusup[xlusup_first + j * inc_diag],
                              &lusup[xlusup_first + j * inc_diag],
                                          &t);
                    }
            else
            {
                        zd_mult(
                                &lusup[xlusup_first + j * inc_diag],
                    &lusup[xlusup_first + j * inc_diag],
                                *fill_tol);
#ifdef DEBUG
            printf("[1] ZERO PIVOT: FILL col %d.\n", first + j);
            fflush(stdout);
                        }
            // 对于SMILU_1类型，根据特定条件执行乘法操作或零填充操作
#endif
#endif
            // 增加非零元素计数器
            nzp++;
            }
            // 结束 switch 语句块
            break;
        case SMILU_2:
                    // 计算并更新对角线元素
                    zd_mult(&lusup[xlusup_first + j * inc_diag],
                                          &lusup[xlusup_first + j * inc_diag],
                                          1.0 + z_abs1(&t));
            // 结束 case SMILU_2
            break;
        case SMILU_3:
                    // 更新 t 并计算矩阵元素
                    z_add(&t, &t, &one);
                    zz_mult(&lusup[xlusup_first + j * inc_diag],
                                  &lusup[xlusup_first + j * inc_diag],
                                      &t);
            // 结束 case SMILU_3
            break;
        case SILU:
        default:
            // 默认情况和 SILU 情况不执行任何操作
            break;
        }
    }
    // 如果非零元素计数大于零，则更新填充因子容差值
    if (nzp > 0) *fill_tol = -nzp;
    }

    /* Remove dropped entries from the memory and fix the pointers. */
    // 更新 m1 为 m - r
    m1 = m - r;
    // 遍历列索引
    for (j = 1; j < n; j++)
    {
    register int tmp1, tmp2;
    // 计算临时变量 tmp1 和 tmp2
    tmp1 = xlusup_first + j * m1;
    tmp2 = xlusup_first + j * m;
    // 移动和更新 L 上的非零元素
    for (i = 0; i < m1; i++)
        lusup[i + tmp1] = lusup[i + tmp2];
    }
    // 移动并更新 lusup 中剩余的非零元素
    for (i = 0; i < nzlc; i++)
    lusup[xlusup_first + i + n * m1] = lusup[xlusup_first + i + n * m];
    // 移动并更新 lsub 中的非零元素
    for (i = 0; i < nzlc; i++)
    lsub[xlsub[last + 1] - r + i] = lsub[xlsub[last + 1] + i];
    // 更新 xlusup 和 xlsub 的指针
    for (i = first + 1; i <= last + 1; i++)
    {
    xlusup[i] -= r * (i - first);
    xlsub[i] -= r;
    }
    // 如果 lastc 为真，则更新 xlusup 和 xlsub 的末尾指针
    if (lastc)
    {
    xlusup[last + 2] -= r * n;
    xlsub[last + 2] -= r;
    }

    // 更新 nnzLj 的值
    *nnzLj += (m - r) * n;
    // 返回结果 r
    return r;
}
```