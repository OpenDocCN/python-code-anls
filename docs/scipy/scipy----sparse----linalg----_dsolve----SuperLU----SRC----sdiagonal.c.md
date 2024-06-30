# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\sdiagonal.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file sdiagonal.c
 * \brief Auxiliary routines to work with diagonal elements
 *
 * <pre>
 * -- SuperLU routine (version 4.0) --
 * Lawrence Berkeley National Laboratory
 * June 30, 2009
 * </pre>
 */

#include "slu_sdefs.h"

/**
 * @brief Fill explicit zeros on the diagonal entries, so that the matrix is not structurally singular.
 *
 * @param n Number of rows/columns in the matrix
 * @param Astore Compressed sparse row (CSR) format of the matrix
 * @return Number of diagonal entries filled with zeros
 */
int sfill_diag(int n, NCformat *Astore)
{
    float *nzval = (float *)Astore->nzval;  // Pointer to non-zero values in CSR format
    int_t *rowind = Astore->rowind;          // Row indices of the non-zero elements
    int_t *colptr = Astore->colptr;          // Column pointers for the start of each column
    int_t nnz = colptr[n];                   // Total number of non-zeros in the matrix
    int fill = 0;                            // Counter for diagonal entries filled with zeros
    float *nzval_new;                        // New array for non-zero values after filling
    float zero = 0.0;                        // Value of zero
    int_t *rowind_new;                       // New array for row indices after filling
    int i, j, diag;                          // Loop variables and diagonal index

    // Loop through each row to check and fill missing diagonal entries
    for (i = 0; i < n; i++)
    {
        diag = -1;  // Initialize diagonal index as non-existent
        // Search for the diagonal entry in the current row
        for (j = colptr[i]; j < colptr[i + 1]; j++)
        {
            if (rowind[j] == i)  // Found diagonal entry
                diag = j;
        }
        if (diag < 0)  // Diagonal entry not found
            fill++;   // Increment fill count
    }

    if (fill > 0)
    {
        // Allocate memory for the new arrays with additional space for zero-filled diagonals
        nzval_new = floatMalloc(nnz + fill);
        rowind_new = intMalloc(nnz + fill);
        fill = 0;  // Reset fill count
        // Loop through each row again to fill zeros for missing diagonal entries
        for (i = 0; i < n; i++)
        {
            diag = -1;  // Initialize diagonal index as non-existent
            // Copy existing non-zero values and row indices to new arrays
            for (j = colptr[i] - fill; j < colptr[i + 1]; j++)
            {
                rowind_new[j + fill] = rowind[j];
                nzval_new[j + fill] = nzval[j];
                if (rowind[j] == i)  // Found diagonal entry
                    diag = j;
            }
            if (diag < 0)
            {
                // Insert new diagonal entry with zero value
                rowind_new[colptr[i + 1] + fill] = i;
                nzval_new[colptr[i + 1] + fill] = zero;
                fill++;  // Increment fill count
            }
            colptr[i + 1] += fill;  // Update column pointer
        }
        // Update Astore with the new arrays
        Astore->nzval = nzval_new;
        Astore->rowind = rowind_new;
        // Free the old arrays
        SUPERLU_FREE(nzval);
        SUPERLU_FREE(rowind);
    }

    // Update the total number of non-zeros in Astore after filling
    Astore->nnz += fill;

    // Return the number of diagonal entries filled with zeros
    return fill;
}

/**
 * @brief Make the matrix diagonally dominant.
 *
 * @param n Number of rows/columns in the matrix
 * @param Astore Compressed sparse row (CSR) format of the matrix
 * @return 0 if successful
 */
int sdominate(int n, NCformat *Astore)
{
    float *nzval = (float *)Astore->nzval;  // Pointer to non-zero values in CSR format
    int_t *rowind = Astore->rowind;          // Row indices of the non-zero elements
    int_t *colptr = Astore->colptr;          // Column pointers for the start of each column
    int_t nnz = colptr[n];                   // Total number of non-zeros in the matrix
    int fill = 0;                            // Counter for diagonal entries filled with zeros
    float *nzval_new;                        // New array for non-zero values after modification
    int_t *rowind_new;                       // New array for row indices after modification
    int_t i, j, diag;                        // Loop variables and diagonal index
    double s;                                // Sum of absolute values of non-zeros in a row

    // Loop through each row to check and make it diagonally dominant
    for (i = 0; i < n; i++)
    {
        diag = -1;  // Initialize diagonal index as non-existent
        // Search for the diagonal entry in the current row
        for (j = colptr[i]; j < colptr[i + 1]; j++)
        {
            if (rowind[j] == i)  // Found diagonal entry
                diag = j;
        }
        if (diag < 0)  // Diagonal entry not found
            fill++;   // Increment fill count
    }

    if (fill > 0)
    {
        // Allocate memory for the new arrays with additional space for modified diagonal entries
        nzval_new = floatMalloc(nnz + fill);
        rowind_new = intMalloc(nnz + fill);
        fill = 0;  // Reset fill count
        // Loop through each row again to modify diagonal entries to make the matrix diagonally dominant
        for (i = 0; i < n; i++)
        {
            s = 1e-6;  // Initialize s as a small positive value
            diag = -1;  // Initialize diagonal index as non-existent
            // Copy existing non-zero values and row indices to new arrays
            for (j = colptr[i] - fill; j < colptr[i + 1]; j++)
            {
                rowind_new[j + fill] = rowind[j];
                nzval_new[j + fill] = nzval[j];
                s += fabs(nzval[j]);  // Accumulate absolute values of non-zero entries in the row
                if (rowind[j] == i)  // Found diagonal entry
                    diag = j;
            }
            if (diag >= 0)
            {
                // Modify existing diagonal entry to ensure diagonal dominance
                nzval_new[diag + fill] = s * 3.0;
            }
            else
            {
                // Insert new diagonal entry with modified value to ensure diagonal dominance
                rowind_new[colptr[i + 1] + fill] = i;
                nzval_new[colptr[i + 1] + fill] = s * 3.0;
                fill++;  // Increment fill count
            }
            colptr[i + 1] += fill;  // Update column pointer
        }
        // Update Astore with the new arrays
        Astore->nzval = nzval_new;
        Astore->rowind = rowind_new;
    }

    // Return 0 to indicate success
    return 0;
}
    # 释放 nzval 数组的内存
    SUPERLU_FREE(nzval);
    # 释放 rowind 数组的内存
    SUPERLU_FREE(rowind);
    }
    else
    {
    # 遍历每一列
    for (i = 0; i < n; i++)
    {
        # 初始化 s 为一个小的数值
        s = 1e-6;
        # 初始化对角线元素位置为 -1
        diag = -1;
        # 遍历第 i 列中的所有非零元素
        for (j = colptr[i]; j < colptr[i + 1]; j++)
        {
        # 如果当前非零元素所在行与列相等，则将 diag 设为当前 j 的值
        if (rowind[j] == i) diag = j;
        # 计算第 i 列所有非零元素的绝对值之和，并加到 s 上
        s += fabs(nzval[j]);
        }
        # 将第 i 列的对角线元素设为 s 的三倍
        nzval[diag] = s * 3.0;
    }
    }
    # 更新 Astore 结构体中的非零元素个数
    Astore->nnz += fill;
    # 返回 fill 变量的值作为函数的结果
    return fill;
}


注释：


# 这是一个单独的右大括号 '}'，用于结束一个代码块或数据结构的定义。
# 在编程中，大括号通常用于定义函数、类、循环或条件语句的范围。
# 在这里，该右大括号可能是用于结束一个函数或条件语句的块，具体取决于代码的上下文。
```