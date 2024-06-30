# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\cdiagonal.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file cdiagonal.c
 * \brief Auxiliary routines to work with diagonal elements
 *
 * <pre>
 * -- SuperLU routine (version 4.0) --
 * Lawrence Berkeley National Laboratory
 * June 30, 2009
 * </pre>
 */

#include "slu_cdefs.h"

/**
 * \brief Fill explicit zeros on the diagonal entries, so that the matrix is not
 * structurally singular.
 *
 * \param n     Dimension of the matrix.
 * \param Astore Pointer to the matrix in compressed column format (NCformat).
 * \return      Number of zeros filled on the diagonal.
 */
int cfill_diag(int n, NCformat *Astore)
{
    singlecomplex *nzval = (singlecomplex *)Astore->nzval;
    int_t *rowind = Astore->rowind;
    int_t *colptr = Astore->colptr;
    int_t nnz = colptr[n];
    int fill = 0;
    singlecomplex *nzval_new;
    singlecomplex zero = {0.0, 0.0};
    int_t *rowind_new;
    int i, j, diag;

    for (i = 0; i < n; i++)
    {
        diag = -1;
        for (j = colptr[i]; j < colptr[i + 1]; j++)
            if (rowind[j] == i) diag = j;
        if (diag < 0) fill++;
    }

    if (fill)
    {
        nzval_new = complexMalloc(nnz + fill);
        rowind_new = intMalloc(nnz + fill);
        fill = 0;
        for (i = 0; i < n; i++)
        {
            diag = -1;
            for (j = colptr[i] - fill; j < colptr[i + 1]; j++)
            {
                if ((rowind_new[j + fill] = rowind[j]) == i) diag = j;
                nzval_new[j + fill] = nzval[j];
            }
            if (diag < 0)
            {
                rowind_new[colptr[i + 1] + fill] = i;
                nzval_new[colptr[i + 1] + fill] = zero;
                fill++;
            }
            colptr[i + 1] += fill;
        }
        Astore->nzval = nzval_new;
        Astore->rowind = rowind_new;
        SUPERLU_FREE(nzval);
        SUPERLU_FREE(rowind);
    }

    Astore->nnz += fill;
    return fill;
}

/**
 * \brief Make the matrix diagonally dominant.
 *
 * \param n     Dimension of the matrix.
 * \param Astore Pointer to the matrix in compressed column format (NCformat).
 * \return      Number of modifications made to achieve diagonal dominance.
 */
int cdominate(int n, NCformat *Astore)
{
    singlecomplex *nzval = (singlecomplex *)Astore->nzval;
    int_t *rowind = Astore->rowind;
    int_t *colptr = Astore->colptr;
    int_t nnz = colptr[n];
    int fill = 0;
    singlecomplex *nzval_new;
    int_t *rowind_new;
    int_t i, j, diag;
    double s;

    for (i = 0; i < n; i++)
    {
        diag = -1;
        for (j = colptr[i]; j < colptr[i + 1]; j++)
            if (rowind[j] == i) diag = j;
        if (diag < 0) fill++;
    }

    if (fill)
    {
        nzval_new = complexMalloc(nnz + fill);
        rowind_new = intMalloc(nnz + fill);
        fill = 0;
        for (i = 0; i < n; i++)
        {
    {
        // 初始化一个非零元素填充计数器
        s = 1e-6;
        // 对角线指示器，初始为-1
        diag = -1;
        // 遍历第 i 列的非零元素
        for (j = colptr[i] - fill; j < colptr[i + 1]; j++)
        {
            // 将原始行索引复制到新的行索引数组中，并检查是否为对角线元素
            if ((rowind_new[j + fill] = rowind[j]) == i) diag = j;
            // 复制非零值到新的非零值数组中
            nzval_new[j + fill] = nzval[j];
            // 计算非零元素的绝对值的累加和
            s += c_abs1(&nzval_new[j + fill]);
        }
        // 如果存在对角线元素
        if (diag >= 0) {
            // 更新对角线元素的实部为累加和乘以3
            nzval_new[diag+fill].r = s * 3.0;
            // 更新对角线元素的虚部为0
            nzval_new[diag+fill].i = 0.0;
        } else {
            // 在新行索引数组的末尾添加当前列的行索引
            rowind_new[colptr[i + 1] + fill] = i;
            // 设置新的非零值数组末尾元素的实部为累加和乘以3
            nzval_new[colptr[i + 1] + fill].r = s * 3.0;
            // 设置新的非零值数组末尾元素的虚部为0
            nzval_new[colptr[i + 1] + fill].i = 0.0;
            // 增加非零元素填充计数器
            fill++;
        }
        // 更新列指针数组中第 i+1 列的结束位置
        colptr[i + 1] += fill;
    }
    
    // 更新稀疏矩阵 A 的存储结构中的非零值和行索引数组为新的数组
    Astore->nzval = nzval_new;
    Astore->rowind = rowind_new;
    // 释放原始的非零值和行索引数组的内存空间
    SUPERLU_FREE(nzval);
    SUPERLU_FREE(rowind);
    }
    else
    {
    // 处理未填充的情况，即直接操作原始的非零值和行索引数组
    for (i = 0; i < n; i++)
    {
        // 初始化一个非零元素填充计数器
        s = 1e-6;
        // 对角线指示器，初始为-1
        diag = -1;
        // 遍历第 i 列的非零元素
        for (j = colptr[i]; j < colptr[i + 1]; j++)
        {
            // 如果当前非零元素的行索引与列索引相等，则标记为对角线元素
            if (rowind[j] == i) diag = j;
            // 计算非零元素的绝对值的累加和
            s += c_abs1(&nzval[j]);
        }
        // 更新对角线元素的实部为累加和乘以3
        nzval[diag].r = s * 3.0;
        // 更新对角线元素的虚部为0
        nzval[diag].i = 0.0;
    }
    
    }
    
    // 更新稀疏矩阵 A 的非零元素数量
    Astore->nnz += fill;
    // 返回填充的非零元素数量
    return fill;
}



# 这行代码是一个单独的右大括号 '}'，通常用于结束代码块或数据结构的定义。
```