# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\ilu_sdrop_row.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file ilu_sdrop_row.c
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
#include "slu_sdefs.h"

/*! \brief
 * <pre>
 * Purpose
 * =======
 *    ilu_sdrop_row() - Drop some small rows from the previous 
 *    supernode (L-part only).
 * </pre>
 */
int ilu_sdrop_row(
    superlu_options_t *options, /* options */  // 参数：SuperLU选项结构体指针，控制算法行为的选项
    int    first,        /* index of the first column in the supernode */  // 参数：第一个列索引，指示超节点中的第一个列
    int    last,        /* index of the last column in the supernode */  // 参数：最后一个列索引，指示超节点中的最后一个列
    double drop_tol,    /* dropping parameter */  // 参数：丢弃参数，用于确定哪些行将被丢弃
    int    quota,        /* maximum nonzero entries allowed */  // 参数：允许的最大非零条目数
    int    *nnzLj,        /* in/out number of nonzeros in L(:, 1:last) */  // 参数：L矩阵中从第一列到last列的非零元素数量（输入输出参数）
    double *fill_tol,   /* in/out - on exit, fill_tol=-num_zero_pivots,
                 * does not change if options->ILU_MILU != SMILU1 */  // 参数：填充容差，表示填充元素的数量变化（输入输出参数）
    GlobalLU_t *Glu,    /* modified */  // 参数：全局LU分解结构体指针，存储LU分解相关信息
    float swork[],   /* working space
                         * the length of swork[] should be no less than
                 * the number of rows in the supernode */  // 参数：工作空间数组，长度应不少于超节点中的行数
    float swork2[], /* working space with the same size as swork[],
                 * used only by the second dropping rule */  // 参数：与swork大小相同的工作空间，仅由第二个丢弃规则使用
    int    lastc        /* if lastc == 0, there is nothing after the
                 * working supernode [first:last];
                 * if lastc == 1, there is one more column after
                 * the working supernode. */ )  // 参数：标志位，指示工作超节点之后是否还有列
{
    register int i, j, k, m1;
    register int nzlc; /* number of nonzeros in column last+1 */  // 变量：列last+1中的非零元素数量
    int_t xlusup_first, xlsub_first;
    int m, n; /* m x n is the size of the supernode */  // 变量：超节点的大小为 m x n
    int r = 0; /* number of dropped rows */  // 变量：被丢弃的行数
    register float *temp;
    register float *lusup = (float *) Glu->lusup;  // 变量：LU分解中L部分的数据指针
    int_t *lsub = Glu->lsub;  // 变量：LU分解中L部分的行索引指针
    int_t *xlsub = Glu->xlsub;  // 变量：LU分解中L部分的列索引起始位置指针
    int_t *xlusup = Glu->xlusup;  // 变量：LU分解中L部分的行索引起始位置指针
    register float d_max = 0.0, d_min = 1.0;  // 变量：最大和最小值初始化
    int    drop_rule = options->ILU_DropRule;  // 变量：丢弃规则，从选项中获取
    milu_t milu = options->ILU_MILU;  // 变量：MILU类型，从选项中获取
    norm_t nrm = options->ILU_Norm;  // 使用 ILU 算法的规范类型
    float zero = 0.0;  // 浮点数零值
    float one = 1.0;  // 浮点数一值
    float none = -1.0;  // 浮点数负一值
    int i_1 = 1;  // 整数1
    int inc_diag; /* inc_diag = m + 1 */  // 对角线增量，等于 m + 1
    int nzp = 0;  /* number of zero pivots */  // 零主元的数量
    float alpha = pow((double)(Glu->n), -1.0 / options->ILU_MILU_Dim);  // 指定公式中的 alpha 值

    xlusup_first = xlusup[first];  // 第一个 xlusup 元素
    xlsub_first = xlsub[first];  // 第一个 xlsub 元素
    m = xlusup[first + 1] - xlusup_first;  // m 的计算
    n = last - first + 1;  // n 的计算
    m1 = m - 1;  // m-1
    inc_diag = m + 1;  // inc_diag 的赋值为 m + 1
    nzlc = lastc ? (xlusup[last + 2] - xlusup[last + 1]) : 0;  // nzlc 的计算，如果 lastc 为真则为 xlusup[last + 2] - xlusup[last + 1]，否则为 0
    temp = swork - n;  // temp 的计算

    /* Quick return if nothing to do. */
    if (m == 0 || m == n || drop_rule == NODROP)
    {
        *nnzLj += m * n;  // 更新 nnzLj
        return 0;  // 快速返回
    }

    /* basic dropping: ILU(tau) */
    for (i = n; i <= m1; )  // 开始基本的 ILU(tau) dropping
    {
        /* the average abs value of ith row */
        switch (nrm)  // 根据规范类型选择不同的操作
        {
            case ONE_NORM:
                temp[i] = sasum_(&n, &lusup[xlusup_first + i], &m) / (double)n;  // 第 i 行的平均绝对值，使用 sasum 函数
                break;
            case TWO_NORM:
                temp[i] = snrm2_(&n, &lusup[xlusup_first + i], &m)
                    / sqrt((double)n);  // 第 i 行的二范数，使用 snrm2 函数
                break;
            case INF_NORM:
            default:
                k = isamax_(&n, &lusup[xlusup_first + i], &m) - 1;  // 第 i 行的无穷范数，使用 isamax 函数
                temp[i] = fabs(lusup[xlusup_first + i + m * k]);  // 绝对值操作
                break;
        }

        /* drop small entries due to drop_tol */
        if (drop_rule & DROP_BASIC && temp[i] < drop_tol)  // 如果符合基本 dropping 条件且小于 drop_tol
        {
            r++;  // 增加 r 计数

            /* drop the current row and move the last undropped row here */
            if (r > 1) /* add to last row */  // 如果 r 大于1，加到最后一行
            {
                /* accumulate the sum (for MILU) */
                switch (milu)  // 根据 MILU 类型执行不同操作
                {
                    case SMILU_1:
                    case SMILU_2:
                        saxpy_(&n, &one, &lusup[xlusup_first + i], &m,
                            &lusup[xlusup_first + m - 1], &m);  // 使用 saxpy 函数
                        break;
                    case SMILU_3:
                        for (j = 0; j < n; j++)
                            lusup[xlusup_first + (m - 1) + j * m] +=
                                fabs(lusup[xlusup_first + i + j * m]);  // SMILU_3 类型的操作
                        break;
                    case SILU:
                    default:
                        break;
                }
                scopy_(&n, &lusup[xlusup_first + m1], &m,
                       &lusup[xlusup_first + i], &m);  // 使用 scopy 函数
            } /* if (r > 1) */
            else /* move to last row */  // 否则，移动到最后一行
            {
                sswap_(&n, &lusup[xlusup_first + m1], &m,
                    &lusup[xlusup_first + i], &m);  // 使用 sswap 函数
                if (milu == SMILU_3)
                    for (j = 0; j < n; j++) {
                    lusup[xlusup_first + m1 + j * m] =
                        fabs(lusup[xlusup_first + m1 + j * m]);  // SMILU_3 类型的操作
                    }
            }
            lsub[xlsub_first + i] = lsub[xlsub_first + m1];  // 更新 lsub 数组
            m1--;  // m1 自减
            continue;  // 继续下一次循环
        } /* if dropping */
        else
        {
            if (temp[i] > d_max) d_max = temp[i];  // 更新 d_max
            if (temp[i] < d_min) d_min = temp[i];  // 更新 d_min
        }
        i++;  // i 自增
    } /* for */

    /* Secondary dropping: drop more rows according to the quota. */
    quota = ceil((double)quota / (double)n);  // 计算配额
    if (drop_rule & DROP_SECONDARY && m - r > quota)  // 如果符合次要 dropping 条件且 m - r 大于 quota
    {
        register double tol = d_max;  // 设置 tol 值

        /* Calculate the second dropping tolerance */
        if (quota > n)
    {
        // 如果按插值法删除规则
        if (drop_rule & DROP_INTERP) /* by interpolation */
        {
            // 计算最大值和最小值的倒数
            d_max = 1.0 / d_max;
            d_min = 1.0 / d_min;
            // 计算公差，用于确定容忍度
            tol = 1.0 / (d_max + (d_min - d_max) * quota / (m - n - r));
        }
        else /* by quick select */
        {
            // 计算待处理区间长度
            int len = m1 - n + 1;
            // 复制待排序数组
            scopy_(&len, swork, &i_1, swork2, &i_1);
            // 使用快速选择算法计算容忍度
            tol = sqselect(len, swork2, quota - n);
#if 0
        register int *itemp = iwork - n;
        A = temp;
        for (i = n; i <= m1; i++) itemp[i] = i;
        qsort(iwork, m1 - n + 1, sizeof(int), _compare_);
        tol = temp[itemp[quota]];
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
                saxpy_(&n, &one, &lusup[xlusup_first + i], &m,
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
            scopy_(&n, &lusup[xlusup_first + m1], &m,
                &lusup[xlusup_first + i], &m);
        } /* if (r > 1) */
        else /* move to last row */
        {
            sswap_(&n, &lusup[xlusup_first + m1], &m,
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

    for (i = n; i < m; i++) temp[i] = 0.0;

    if (r == 0)
    {
    *nnzLj += m * n;
    return 0;
    }

    /* add dropped entries to the diagnal */
    if (milu != SILU)
    {
    register int j;
    float t;
    float omega;
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


注释：

#if 0
        register int *itemp = iwork - n;
        A = temp;
        for (i = n; i <= m1; i++) itemp[i] = i;
        qsort(iwork, m1 - n + 1, sizeof(int), _compare_);
        tol = temp[itemp[quota]];
#endif
        }
    }

    for (i = n; i <= m1; )
    /* 移除内存中已删除的条目并修正指针。*/
    m1 = m - r;
    /* 循环处理每列，从第二列开始到第n列 */
    for (j = 1; j < n; j++)
    {
        register int tmp1, tmp2;
        /* 计算第j列在压缩存储数组中的起始位置，分别对应于移除部分和未移除部分 */
        tmp1 = xlusup_first + j * m1;
        tmp2 = xlusup_first + j * m;
        /* 将第j列中被移除部分之后的数据向前移动，以填补空缺 */
        for (i = 0; i < m1; i++)
            lusup[i + tmp1] = lusup[i + tmp2];
    }
    
    /* 更新剩余非零元素的列指针数组，将被移除部分之后的列指针前移 */
    for (i = 0; i < nzlc; i++)
        lusup[xlusup_first + i + n * m1] = lusup[xlusup_first + i + n * m];
    
    /* 更新行下标数组，将被移除部分之后的行下标前移 */
    for (i = 0; i < nzlc; i++)
        lsub[xlsub[last + 1] - r + i] = lsub[xlsub[last + 1] + i];
    
    /* 调整列指针数组中影响的范围，使其适应已移除的条目 */
    for (i = first + 1; i <= last + 1; i++)
    {
        xlusup[i] -= r * (i - first);
        xlsub[i] -= r;
    }
    
    /* 如果存在最后一列的处理 */
    if (lastc)
    {
        xlusup[last + 2] -= r * n;
        xlsub[last + 2] -= r;
    }
    
    /* 更新非零元素总数 */
    *nnzLj += (m - r) * n;
    
    /* 返回已移除的条目数量 */
    return r;
}



# 这行代码关闭了一个代码块。在大多数编程语言中，这种语法用于结束一个函数、循环或条件语句的定义。
```