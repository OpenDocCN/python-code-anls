# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\zdiagonal.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file zdiagonal.c
 * \brief Auxiliary routines to work with diagonal elements
 *
 * <pre>
 * -- SuperLU routine (version 4.0) --
 * Lawrence Berkeley National Laboratory
 * June 30, 2009
 * </pre>
 */

#include "slu_zdefs.h"

int zfill_diag(int n, NCformat *Astore)
/* fill explicit zeros on the diagonal entries, so that the matrix is not
   structurally singular. */
{
    // 指向稀疏矩阵非零元素的复数数组
    doublecomplex *nzval = (doublecomplex *)Astore->nzval;
    // 行索引数组
    int_t *rowind = Astore->rowind;
    // 列指针数组
    int_t *colptr = Astore->colptr;
    // 矩阵非零元素的总数
    int_t nnz = colptr[n];
    // 需要填充的非零元素个数
    int fill = 0;
    // 新的非零元素数组和行索引数组
    doublecomplex *nzval_new;
    int_t *rowind_new;
    // 零元素
    doublecomplex zero = {0.0, 0.0};
    // 循环变量
    int_t i, j, diag;

    // 遍历每一行
    for (i = 0; i < n; i++)
    {
        // 初始化对角线标志
        diag = -1;
        // 在当前行的非零元素中查找对角线元素的位置
        for (j = colptr[i]; j < colptr[i + 1]; j++)
            if (rowind[j] == i) diag = j;
        // 如果对角线元素不存在，则需要填充
        if (diag < 0) fill++;
    }

    // 如果需要填充
    if (fill)
    {
        // 分配新的非零元素数组和行索引数组
        nzval_new = doublecomplexMalloc(nnz + fill);
        rowind_new = intMalloc(nnz + fill);
        fill = 0;
        // 复制原始数据并填充缺失的对角线元素
        for (i = 0; i < n; i++)
        {
            diag = -1;
            for (j = colptr[i] - fill; j < colptr[i + 1]; j++)
            {
                // 复制行索引和非零元素
                if ((rowind_new[j + fill] = rowind[j]) == i) diag = j;
                nzval_new[j + fill] = nzval[j];
            }
            // 如果对角线元素不存在，则添加零元素
            if (diag < 0)
            {
                rowind_new[colptr[i + 1] + fill] = i;
                nzval_new[colptr[i + 1] + fill] = zero;
                fill++;
            }
            // 更新列指针数组
            colptr[i + 1] += fill;
        }
        // 更新稀疏矩阵结构体中的非零元素数组和行索引数组
        Astore->nzval = nzval_new;
        Astore->rowind = rowind_new;
        // 释放原始的非零元素数组和行索引数组
        SUPERLU_FREE(nzval);
        SUPERLU_FREE(rowind);
    }

    // 更新稀疏矩阵结构体中的非零元素总数
    Astore->nnz += fill;
    // 返回需要填充的对角线元素个数
    return fill;
}

int zdominate(int n, NCformat *Astore)
/* make the matrix diagonally dominant */
{
    // 指向稀疏矩阵非零元素的复数数组
    doublecomplex *nzval = (doublecomplex *)Astore->nzval;
    // 行索引数组
    int_t *rowind = Astore->rowind;
    // 列指针数组
    int_t *colptr = Astore->colptr;
    // 矩阵非零元素的总数
    int_t nnz = colptr[n];
    // 需要填充的非零元素个数
    int fill = 0;
    // 新的非零元素数组和行索引数组
    doublecomplex *nzval_new;
    int_t *rowind_new;
    // 循环变量
    int_t i, j, diag;
    // 对角线元素和非对角线元素之和
    double s;

    // 遍历每一行
    for (i = 0; i < n; i++)
    {
        // 初始化对角线标志
        diag = -1;
        // 在当前行的非零元素中查找对角线元素的位置
        for (j = colptr[i]; j < colptr[i + 1]; j++)
            if (rowind[j] == i) diag = j;
        // 如果对角线元素不存在，则需要填充
        if (diag < 0) fill++;
    }

    // 如果需要填充
    if (fill)
    {
        // 分配新的非零元素数组和行索引数组
        nzval_new = doublecomplexMalloc(nnz + fill);
        rowind_new = intMalloc(nnz + fill);
        fill = 0;
        // 复制原始数据并填充缺失的对角线元素
        for (i = 0; i < n; i++)
        {
            diag = -1;
            for (j = colptr[i] - fill; j < colptr[i + 1]; j++)
            {
                // 复制行索引和非零元素
                if ((rowind_new[j + fill] = rowind[j]) == i) diag = j;
                nzval_new[j + fill] = nzval[j];
            }
            // 如果对角线元素不存在，则添加零元素
            if (diag < 0)
            {
                rowind_new[colptr[i + 1] + fill] = i;
                nzval_new[colptr[i + 1] + fill] = zero;
                fill++;
            }
            // 更新列指针数组
            colptr[i + 1] += fill;
        }
        // 更新稀疏矩阵结构体中的非零元素数组和行索引数组
        Astore->nzval = nzval_new;
        Astore->rowind = rowind_new;
        // 释放原始的非零元素数组和行索引数组
        SUPERLU_FREE(nzval);
        SUPERLU_FREE(rowind);
    }

    // 返回需要填充的对角线元素个数
    return fill;
}
    {
        # 设置一个非常小的初始值s和一个负的对角线指示符diag
        s = 1e-6;
        diag = -1;
        # 遍历列i在列指针数组中的范围，考虑填充因子fill
        for (j = colptr[i] - fill; j < colptr[i + 1]; j++)
        {
            # 如果当前行索引等于i，则更新新行索引数组，同时更新对角线指示符diag的位置
            if ((rowind_new[j + fill] = rowind[j]) == i) diag = j;
            # 更新新非零值数组中的值为当前非零值数组的值
            nzval_new[j + fill] = nzval[j];
            # 计算s值，累加新非零值的绝对值
            s += z_abs1(&nzval_new[j + fill]);
        }
        # 如果找到了对角线元素
        if (diag >= 0) {
            # 更新新非零值数组中对角线元素的实部为s乘以3.0
            nzval_new[diag+fill].r = s * 3.0;
            # 更新新非零值数组中对角线元素的虚部为0.0
            nzval_new[diag+fill].i = 0.0;
        } else {
            # 如果未找到对角线元素，则更新新行索引数组和新非零值数组的下一个位置
            rowind_new[colptr[i + 1] + fill] = i;
            nzval_new[colptr[i + 1] + fill].r = s * 3.0;
            nzval_new[colptr[i + 1] + fill].i = 0.0;
            # 增加填充因子的计数
            fill++;
        }
        # 更新列指针数组中下一列的值，增加填充因子的累计值
        colptr[i + 1] += fill;
    }
    # 更新Astore结构体中的非零值数组和行索引数组
    Astore->nzval = nzval_new;
    Astore->rowind = rowind_new;
    # 释放原始非零值和行索引数组的内存空间
    SUPERLU_FREE(nzval);
    SUPERLU_FREE(rowind);
    }
    else
    {
    # 对于未分配填充因子的情况，对每一列进行操作
    for (i = 0; i < n; i++)
    {
        # 设置一个非常小的初始值s和一个负的对角线指示符diag
        s = 1e-6;
        diag = -1;
        # 遍历列i在列指针数组中的范围
        for (j = colptr[i]; j < colptr[i + 1]; j++)
        {
            # 如果当前行索引等于i，则更新对角线指示符diag的位置
            if (rowind[j] == i) diag = j;
            # 计算s值，累加当前非零值数组中的绝对值
            s += z_abs1(&nzval[j]);
        }
        # 更新原始非零值数组中对角线元素的实部为s乘以3.0
        nzval[diag].r = s * 3.0;
        # 更新原始非零值数组中对角线元素的虚部为0.0
        nzval[diag].i = 0.0;
    }
    }
    # 更新Astore结构体中的非零元素数量
    Astore->nnz += fill;
    # 返回填充因子的累计值
    return fill;
}


注释：


# 这是一个代码块的结束标记，匹配之前的代码块起始标记 {
```