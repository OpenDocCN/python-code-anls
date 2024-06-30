# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\ilu_cdrop_row.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file ilu_cdrop_row.c
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
#include "slu_cdefs.h"

extern void cswap_(int *, singlecomplex [], int *, singlecomplex [], int *);
extern void caxpy_(int *, singlecomplex *, singlecomplex [], int *, singlecomplex [], int *);
extern void ccopy_(int *, singlecomplex [], int *, singlecomplex [], int *);
extern void scopy_(int *, float [], int *, float [], int *);
extern float scasum_(int *, singlecomplex *, int *);
extern float scnrm2_(int *, singlecomplex *, int *);
extern double dnrm2_(int *, double [], int *);
extern int icamax_(int *, singlecomplex [], int *);

#if 0
static float *A;  /* used in _compare_ only */
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
 *    ilu_cdrop_row() - Drop some small rows from the previous 
 *    supernode (L-part only).
 * </pre>
 */
int ilu_cdrop_row(
    superlu_options_t *options, /* options: 超级LU选项结构体指针 */
    int    first,        /* first: 超节点中第一列的索引 */
    int    last,        /* last: 超节点中最后一列的索引 */
    double drop_tol,    /* drop_tol: 下降参数 */
    int    quota,        /* quota: 允许的最大非零条目数 */
    int    *nnzLj,        /* nnzLj: L(:, 1:last)中非零条目的数量（输入/输出） */
    double *fill_tol,   /* fill_tol: 出口时为-num_zero_pivots，如果options->ILU_MILU != SMILU1则不更改 */
    GlobalLU_t *Glu,    /* Glu: 全局LU分解结构体指针（被修改） */
    float swork[],   /* swork: 工作空间，长度不小于超节点中的行数 */
    float swork2[], /* swork2: 与swork[]大小相同的工作空间，仅由第二个下降规则使用 */
    int    lastc        /* lastc: 如果lastc == 0，表示工作超节点[first:last]后没有内容；
                           如果lastc == 1，则工作超节点后还有一列。 */ )
{
    register int i, j, k, m1;
    register int nzlc; /* nzlc: 列last+1中的非零条目数 */
    int_t xlusup_first, xlsub_first;
    int m, n; /* m x n是超节点的大小 */
    int r = 0; /* 被删除的行数 */
    register float *temp;
    register singlecomplex *lusup = (singlecomplex *) Glu->lusup;
    int_t *lsub = Glu->lsub;
    int_t *xlsub = Glu->xlsub;
    // 获取指向 Glu 结构体中 xlusup 数组的指针
    int_t *xlusup = Glu->xlusup;
    // 初始化浮点数变量 d_max 和 d_min
    register float d_max = 0.0, d_min = 1.0;
    // 获取 ILU_DropRule 参数值
    int drop_rule = options->ILU_DropRule;
    // 获取 ILU_MILU 参数值
    milu_t milu = options->ILU_MILU;
    // 获取 ILU_Norm 参数值
    norm_t nrm = options->ILU_Norm;
    // 初始化单精度复数变量 one 和 none
    singlecomplex one = {1.0, 0.0};
    singlecomplex none = {-1.0, 0.0};
    // 设置整数变量 i_1 的值为 1
    int i_1 = 1;
    // 初始化 inc_diag 变量，inc_diag = m + 1
    int inc_diag;
    // 初始化 nzp 变量，表示零主元的数量
    int nzp = 0;
    // 计算 alpha 参数值
    float alpha = pow((double)(Glu->n), -1.0 / options->ILU_MILU_Dim);

    // 获取 xlusup 数组中的第 first 个元素并赋值给 xlusup_first
    xlusup_first = xlusup[first];
    // 获取 xlsub 数组中的第 first 个元素并赋值给 xlsub_first
    xlsub_first = xlsub[first];
    // 计算 m 的值，即 xlusup[first+1] 到 xlusup_first 的距离
    m = xlusup[first + 1] - xlusup_first;
    // 计算 n 的值，即 last - first + 1
    n = last - first + 1;
    // 计算 m1 的值，即 m - 1
    m1 = m - 1;
    // 初始化 inc_diag 变量，inc_diag = m + 1
    inc_diag = m + 1;
    // 计算 nzlc 的值，如果 lastc 为真则计算最后一列的非零元素数目
    nzlc = lastc ? (xlusup[last + 2] - xlusup[last + 1]) : 0;
    // 计算 temp 的值，指向 swork 的前 n 个元素
    temp = swork - n;

    /* 如果没有需要执行的操作，直接返回 */
    if (m == 0 || m == n || drop_rule == NODROP)
    {
        // 更新 nnzLj 的值，增加 m * n
        *nnzLj += m * n;
        // 返回 0 表示没有错误
        return 0;
    }

    /* 基本的 dropping 策略: ILU(tau) */
    for (i = n; i <= m1; )
    {
        /* 计算第 i 行的平均绝对值 */
        switch (nrm)
        {
            case ONE_NORM:
                // 调用 scasum_ 函数计算绝对值和，并除以 n 得到平均值
                temp[i] = scasum_(&n, &lusup[xlusup_first + i], &m) / (double)n;
                break;
            case TWO_NORM:
                // 调用 scnrm2_ 函数计算范数并除以 sqrt(n) 得到平均值
                temp[i] = scnrm2_(&n, &lusup[xlusup_first + i], &m)
                           / sqrt((double)n);
                break;
            case INF_NORM:
            default:
                // 调用 icamax_ 函数找到最大元素的索引 k，并计算绝对值
                k = icamax_(&n, &lusup[xlusup_first + i], &m) - 1;
                temp[i] = c_abs1(&lusup[xlusup_first + i + m * k]);
                break;
        }

        /* 根据 drop_tol 丢弃小元素 */
        if (drop_rule & DROP_BASIC && temp[i] < drop_tol)
        {
            r++;
            /* 丢弃当前行并将最后一个未丢弃的行移动到当前位置 */
            if (r > 1) /* 添加到最后一行 */
            {
                /* 根据 MILU 类型累加和 */
                switch (milu)
                {
                    case SMILU_1:
                    case SMILU_2:
                        // 调用 caxpy_ 函数将向量加到最后一行
                        caxpy_(&n, &one, &lusup[xlusup_first + i], &m,
                               &lusup[xlusup_first + m - 1], &m);
                        break;
                    case SMILU_3:
                        // 将第 i 行的绝对值累加到最后一行的相应元素上
                        for (j = 0; j < n; j++)
                            lusup[xlusup_first + (m - 1) + j * m].r +=
                                c_abs1(&lusup[xlusup_first + i + j * m]);
                        break;
                    case SILU:
                    default:
                        break;
                }
                // 将最后一行的内容复制到第 i 行
                ccopy_(&n, &lusup[xlusup_first + m1], &m,
                       &lusup[xlusup_first + i], &m);
            } /* if (r > 1) */
            else /* 移动到最后一行 */
            {
                // 交换最后一行和第 i 行的内容
                cswap_(&n, &lusup[xlusup_first + m1], &m,
                       &lusup[xlusup_first + i], &m);
                // 如果 MILU 类型为 SMILU_3，则将最后一行的绝对值赋给虚部为 0
                if (milu == SMILU_3)
                    for (j = 0; j < n; j++) {
                        lusup[xlusup_first + m1 + j * m].r =
                            c_abs1(&lusup[xlusup_first + m1 + j * m]);
                        lusup[xlusup_first + m1 + j * m].i = 0.0;
                    }
            }
            // 更新 lsub 数组的值
            lsub[xlsub_first + i] = lsub[xlsub_first + m1];
            // 减少 m1 的值，表示剩余未处理的行数减少
            m1--;
            // 继续处理下一行
            continue;
        } /* if dropping */
        else
        {
            // 更新 d_max 和 d_min 的值
            if (temp[i] > d_max) d_max = temp[i];
            if (temp[i] < d_min) d_min = temp[i];
        }
        // 处理下一行
        i++;
    } /* for */
    /* Secondary dropping: drop more rows according to the quota. */
    // 计算二次删除的配额，将总配额按照行数平均分配
    quota = ceil((double)quota / (double)n);
    // 如果满足二次删除条件且剩余行数超过配额
    if (drop_rule & DROP_SECONDARY && m - r > quota)
    {
        register double tol = d_max;

        /* Calculate the second dropping tolerance */
        // 计算第二次删除的容忍度
        if (quota > n)
        {
            // 如果配额大于行数，根据删除规则选择计算方法
            if (drop_rule & DROP_INTERP) /* by interpolation */
            {
                // 使用插值法计算
                d_max = 1.0 / d_max; d_min = 1.0 / d_min;
                tol = 1.0 / (d_max + (d_min - d_max) * quota / (m - n - r));
            }
            else /* by quick select */
            {
                // 使用快速选择法计算
                int len = m1 - n + 1;
                scopy_(&len, swork, &i_1, swork2, &i_1);
                tol = sqselect(len, swork2, quota - n);
                // 使用快速选择法得出的容忍度
            }
        }
    }
    # 如果定义了宏 `if 0`，则执行以下代码块
    register int *itemp = iwork - n;  # 定义整型指针 `itemp`，指向 `iwork - n` 的地址偏移
    A = temp;  # 将 `temp` 赋值给 `A`
    for (i = n; i <= m1; i++) itemp[i] = i;  # 初始化 `itemp` 数组，从 `n` 到 `m1`，值为 `i`
    qsort(iwork, m1 - n + 1, sizeof(int), _compare_);  # 对 `iwork` 数组进行快速排序，排序大小为 `m1 - n + 1`，每个元素大小为 `sizeof(int)`，使用 `_compare_` 函数进行比较
    tol = temp[itemp[quota]];  # 从 `temp` 数组中取出 `itemp[quota]` 索引位置的值，赋给 `tol`
#endif
    }
}

for (i = n; i <= m1; )
{
    if (temp[i] <= tol)
    {
        register int j;  # 定义整型变量 `j`
        r++;  # `r` 自增
        /* drop the current row and move the last undropped row here */
        if (r > 1) /* add to last row */
        {
            /* accumulate the sum (for MILU) */
            switch (milu)
            {
            case SMILU_1:
            case SMILU_2:
                caxpy_(&n, &one, &lusup[xlusup_first + i], &m,
                    &lusup[xlusup_first + m - 1], &m);  # 调用 `caxpy_` 函数，对 `lusup` 数组进行线性变换操作
                break;
            case SMILU_3:
                for (j = 0; j < n; j++)
                    lusup[xlusup_first + (m - 1) + j * m].r +=
                         c_abs1(&lusup[xlusup_first + i + j * m]);  # 计算 `lusup` 数组中的一些数值加和，用于 MILU
                break;
            case SILU:
            default:
                break;
            }
            ccopy_(&n, &lusup[xlusup_first + m1], &m,
                &lusup[xlusup_first + i], &m);  # 复制 `lusup` 数组中的一部分数据
        } /* if (r > 1) */
        else /* move to last row */
        {
            cswap_(&n, &lusup[xlusup_first + m1], &m,
                &lusup[xlusup_first + i], &m);  # 交换 `lusup` 数组中的两部分数据
            if (milu == SMILU_3)
                for (j = 0; j < n; j++) {
                    lusup[xlusup_first + m1 + j * m].r =
                        c_abs1(&lusup[xlusup_first + m1 + j * m]);  # 对 `lusup` 数组中的一部分数据求绝对值
                    lusup[xlusup_first + m1 + j * m].i = 0.0;  # 将 `lusup` 数组中的一部分数据虚部设为 `0.0`
                }
        }
        lsub[xlsub_first + i] = lsub[xlsub_first + m1];  # 复制 `lsub` 数组中的一部分数据
        m1--;  # `m1` 自减
        temp[i] = temp[m1];  # 将 `temp` 数组中的一部分数据赋给 `temp[i]`

        continue;  # 跳过当前循环，执行下一次循环
    }
    i++;  # `i` 自增

} /* for */

} /* if secondary dropping */

for (i = n; i < m; i++) temp[i] = 0.0;  # 将 `temp` 数组中的一部分数据设为 `0.0`

if (r == 0)
{
*nnzLj += m * n;  # 将 `m * n` 加到 `*nnzLj` 变量上
return 0;  # 函数返回 `0`
}

/* add dropped entries to the diagnal */
if (milu != SILU)
{
register int j;  # 定义整型变量 `j`
singlecomplex t;  # 定义复数结构体 `t`
float omega;  # 定义浮点数 `omega`
for (j = 0; j < n; j++)
{
    t = lusup[xlusup_first + (m - 1) + j * m];  # 从 `lusup` 数组中取出一个复数，赋给 `t`
        if (t.r == 0.0 && t.i == 0.0) continue;  # 如果 `t` 的实部和虚部都为 `0.0`，则跳过当前循环
        omega = SUPERLU_MIN(2.0 * (1.0 - alpha) / c_abs1(&t), 1.0);  # 计算 `omega` 的值，取两者中的较小值
    cs_mult(&t, &t, omega);  # 对复数 `t` 进行数乘操作

     switch (milu)
    {
    case SMILU_1:
        if ( !(c_eq(&t, &none)) ) {
                    c_add(&t, &t, &one);  # 复数 `t` 加 `1.0`
                    cc_mult(&lusup[xlusup_first + j * inc_diag],
                          &lusup[xlusup_first + j * inc_diag],
                                      &t);  # 对 `lusup` 数组中的一部分数据进行复数乘法操作
                }
        else
        {
                    cs_mult(
                            &lusup[xlusup_first + j * inc_diag],
                &lusup[xlusup_first + j * inc_diag],
                            *fill_tol);  # 对 `lusup` 数组中的一部分数据进行数乘操作
#ifdef DEBUG
        printf("[1] ZERO PIVOT: FILL col %d.\n", first + j);  # 输出调试信息，表示零主元，填充列 `first + j`
        fflush(stdout);  # 刷新标准输出流
#endif
    }
}
#endif
            nzp++;
            // 增加非零元素计数器

            }
            // 结束当前的 switch 语句块

            break;
        case SMILU_2:
                    cs_mult(&lusup[xlusup_first + j * inc_diag],
                                          &lusup[xlusup_first + j * inc_diag],
                                          1.0 + c_abs1(&t));
            // 对 SMILU_2 情况下的 L 上的超节点进行乘法操作

            break;
        case SMILU_3:
                    c_add(&t, &t, &one);
                    cc_mult(&lusup[xlusup_first + j * inc_diag],
                                  &lusup[xlusup_first + j * inc_diag],
                                      &t);
            // 对 SMILU_3 情况下的 L 上的超节点进行复数加法和乘法操作

            break;
        case SILU:
        default:
            // SILU 或默认情况下，不执行任何操作
            break;
        }
    }
    // 遍历完所有的列 j

    if (nzp > 0) *fill_tol = -nzp;
    // 如果非零元素计数器大于零，则更新填充因子的负值

    }

    /* Remove dropped entries from the memory and fix the pointers. */
    // 从内存中删除被丢弃的条目并修复指针

    m1 = m - r;
    for (j = 1; j < n; j++)
    {
    register int tmp1, tmp2;
    tmp1 = xlusup_first + j * m1;
    tmp2 = xlusup_first + j * m;
    for (i = 0; i < m1; i++)
        lusup[i + tmp1] = lusup[i + tmp2];
    // 更新 L 上三角矩阵的非零元素，将被丢弃的条目替换成保留的条目
    }
    for (i = 0; i < nzlc; i++)
    lusup[xlusup_first + i + n * m1] = lusup[xlusup_first + i + n * m];
    // 更新 L 上的超节点，移除被丢弃的条目并修正指针
    for (i = 0; i < nzlc; i++)
    lsub[xlsub[last + 1] - r + i] = lsub[xlsub[last + 1] + i];
    // 更新 L 的行索引，移除被丢弃的条目

    for (i = first + 1; i <= last + 1; i++)
    {
    xlusup[i] -= r * (i - first);
    xlsub[i] -= r;
    // 更新 L 上的超节点的指针，修正由于删除条目而变化的指针位置
    }

    if (lastc)
    {
    xlusup[last + 2] -= r * n;
    xlsub[last + 2] -= r;
    // 如果存在最后一列，更新其在 L 上的超节点的指针
    }

    *nnzLj += (m - r) * n;
    // 更新 L 的非零元素计数器

    return r;
    // 返回当前操作删除的条目数量
}
```