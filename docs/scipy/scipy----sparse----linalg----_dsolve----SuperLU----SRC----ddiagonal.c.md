# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\ddiagonal.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file ddiagonal.c
 * \brief Auxiliary routines to work with diagonal elements
 *
 * <pre>
 * -- SuperLU routine (version 4.0) --
 * Lawrence Berkeley National Laboratory
 * June 30, 2009
 * </pre>
 */

#include "slu_ddefs.h"

int dfill_diag(int n, NCformat *Astore)
/* fill explicit zeros on the diagonal entries, so that the matrix is not
   structurally singular. */
{
    double *nzval = (double *)Astore->nzval;  // 获取稀疏矩阵的非零值数组
    int_t *rowind = Astore->rowind;            // 获取稀疏矩阵的行索引数组
    int_t *colptr = Astore->colptr;            // 获取稀疏矩阵的列指针数组
    int_t nnz = colptr[n];                     // 获取矩阵中非零元素的数量
    int fill = 0;                              // 计数需要填充的对角线零元素数量
    double *nzval_new;                         // 新的非零值数组
    double zero = 0.0;                         // 表示零的常量
    int_t *rowind_new;                         // 新的行索引数组
    int i, j, diag;                            // 循环变量和对角线元素位置

    // 遍历每一行，检查对角线元素是否存在
    for (i = 0; i < n; i++)
    {
        diag = -1;  // 初始化对角线元素位置为不存在
        // 在当前行的列指针范围内查找对角线元素
        for (j = colptr[i]; j < colptr[i + 1]; j++)
            if (rowind[j] == i) diag = j;  // 如果找到对角线元素，记录其位置
        if (diag < 0) fill++;  // 如果未找到对角线元素，增加填充计数
    }

    // 如果存在需要填充的对角线元素
    if (fill)
    {
        nzval_new = doubleMalloc(nnz + fill);  // 分配新的非零值数组内存空间
        rowind_new = intMalloc(nnz + fill);    // 分配新的行索引数组内存空间
        fill = 0;  // 重置填充计数为零

        // 再次遍历每一行，填充缺失的对角线元素
        for (i = 0; i < n; i++)
        {
            diag = -1;  // 初始化对角线元素位置为不存在
            // 复制当前行的非零元素到新的数组中
            for (j = colptr[i] - fill; j < colptr[i + 1]; j++)
            {
                if ((rowind_new[j + fill] = rowind[j]) == i) diag = j;  // 复制行索引并查找对角线元素位置
                nzval_new[j + fill] = nzval[j];  // 复制非零值
            }
            // 如果未找到对角线元素，添加零元素到新的数组中
            if (diag < 0)
            {
                rowind_new[colptr[i + 1] + fill] = i;  // 添加缺失的行索引
                nzval_new[colptr[i + 1] + fill] = zero;  // 添加零元素
                fill++;  // 填充计数增加
            }
            colptr[i + 1] += fill;  // 更新列指针数组
        }

        Astore->nzval = nzval_new;  // 更新稀疏矩阵结构中的非零值数组
        Astore->rowind = rowind_new;  // 更新稀疏矩阵结构中的行索引数组
        SUPERLU_FREE(nzval);  // 释放旧的非零值数组内存
        SUPERLU_FREE(rowind);  // 释放旧的行索引数组内存
    }

    Astore->nnz += fill;  // 更新稀疏矩阵结构中的非零元素数量
    return fill;  // 返回填充的对角线零元素数量
}

int ddominate(int n, NCformat *Astore)
/* make the matrix diagonally dominant */
{
    double *nzval = (double *)Astore->nzval;  // 获取稀疏矩阵的非零值数组
    int_t *rowind = Astore->rowind;            // 获取稀疏矩阵的行索引数组
    int_t *colptr = Astore->colptr;            // 获取稀疏矩阵的列指针数组
    int_t nnz = colptr[n];                     // 获取矩阵中非零元素的数量
    int fill = 0;                              // 计数需要填充的对角线零元素数量
    double *nzval_new;                         // 新的非零值数组
    int_t *rowind_new;                         // 新的行索引数组
    int_t i, j, diag;                          // 循环变量和对角线元素位置
    double s;                                  

    // 遍历每一行，检查对角线元素是否存在
    for (i = 0; i < n; i++)
    {
        diag = -1;  // 初始化对角线元素位置为不存在
        // 在当前行的列指针范围内查找对角线元素
        for (j = colptr[i]; j < colptr[i + 1]; j++)
            if (rowind[j] == i) diag = j;  // 如果找到对角线元素，记录其位置
        if (diag < 0) fill++;  // 如果未找到对角线元素，增加填充计数
    }

    // 如果存在需要填充的对角线元素
    if (fill)
    {
        nzval_new = doubleMalloc(nnz + fill);  // 分配新的非零值数组内存空间
        rowind_new = intMalloc(nnz + fill);    // 分配新的行索引数组内存空间
        fill = 0;  // 重置填充计数为零

        // 再次遍历每一行，使矩阵成为对角占优形式
        for (i = 0; i < n; i++)
        {
            s = 1e-6;  // 初始化元素和为一个小的正数
            diag = -1;  // 初始化对角线元素位置为不存在
            // 复制当前行的非零元素到新的数组中，并计算绝对值和
            for (j = colptr[i] - fill; j < colptr[i + 1]; j++)
            {
                if ((rowind_new[j + fill] = rowind[j]) == i) diag = j;  // 复制行索引并查找对角线元素位置
                s += fabs(nzval_new[j + fill] = nzval[j]);  // 计算绝对值和并复制非零值
            }
            // 如果找到对角线元素，更新其值为绝对值和的三倍
            if (diag >= 0) {
                nzval_new[diag+fill] = s * 3.0;
            } else {
                rowind_new[colptr[i + 1] + fill] = i;  // 添加缺失的行索引
                nzval_new[colptr[i + 1] + fill] = s * 3.0;  // 添加对角线元素为绝对值和的三倍
                fill++;  // 填充计数增加
            }
            colptr[i + 1] += fill;  // 更新列指针数组
        }

        Astore->nzval = nzval_new;  // 更新稀疏矩阵结构中的非零值数组
        Astore->rowind = rowind_new;  // 更新稀疏
    # 更新 Astore 结构体中的行索引数组为新的 rowind_new
    Astore->rowind = rowind_new;
    
    # 释放存储在 nzval 指针指向位置的内存空间
    SUPERLU_FREE(nzval);
    
    # 释放存储在 rowind 指针指向位置的内存空间
    SUPERLU_FREE(rowind);
    }
    else
    {
    # 遍历每一列
    for (i = 0; i < n; i++)
    {
        # 初始化 s 为一个小的正数
        s = 1e-6;
        
        # 初始化对角线索引 diag 为 -1
        diag = -1;
        
        # 遍历第 i 列中的每个非零元素
        for (j = colptr[i]; j < colptr[i + 1]; j++)
        {
            # 如果行索引等于列索引 i，则更新 diag 为当前列索引 j
            if (rowind[j] == i) diag = j;
            
            # 将当前非零元素的绝对值加到 s 上
            s += fabs(nzval[j]);
        }
        
        # 将对角线元素的值设置为 s 乘以 3.0
        nzval[diag] = s * 3.0;
    }
    }
    
    # 更新 Astore 结构体中的非零元素个数 nnz
    Astore->nnz += fill;
    
    # 返回填充量 fill
    return fill;
}



# 这是一个单独的右大括号，用于结束一个代码块或数据结构的定义
```