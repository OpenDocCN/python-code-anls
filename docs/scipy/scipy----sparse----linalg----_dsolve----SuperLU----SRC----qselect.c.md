# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\qselect.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file qselect.c
 * \brief Quickselect: returns the k-th (zero-based) largest value in A[].
 *
 * <pre>
 * -- SuperLU routine (version 4.1) --
 * Lawrence Berkeley National Laboratory.
 * November, 2010
 * </pre>
 */

#include "slu_ddefs.h"

// 定义一个函数，用于从双精度浮点数数组中找到第k大的元素值
double dqselect(int n, double A[], int k)
{
    // 声明变量 i, j, p 和 val，用于索引和存储值
    register int i, j, p;
    register double val;

    // 确保 k 的值在有效范围内
    k = SUPERLU_MAX(k, 0); // 将 k 限制在不小于 0 的范围内
    k = SUPERLU_MIN(k, n - 1); // 将 k 限制在不超过 n-1 的范围内

    // 开始进行快速选择算法
    while (n > 1)
    {
        // 初始化 i 和 j，以及选择的基准位置 p 和其值 val
        i = 0; j = n-1;
        p = j; val = A[p];

        // 开始分区过程，将数组 A 按照基准 val 进行分割
        while (i < j)
        {
            // 从左向右找到第一个大于等于基准的元素位置
            for (; A[i] >= val && i < p; i++);
            if (A[i] < val) { A[p] = A[i]; p = i; } // 将找到的元素移到基准位置

            // 从右向左找到第一个小于等于基准的元素位置
            for (; A[j] <= val && j > p; j--);
            if (A[j] > val) { A[p] = A[j]; p = j; } // 将找到的元素移到基准位置
        }

        // 将基准值放置到最终的位置上
        A[p] = val;

        // 如果基准位置 p 正好是要找的第 k 大的位置，则返回该值
        if (p == k) return val;
        else if (p > k) n = p; // 如果基准位置大于 k，则在左侧继续查找
        else
        {
            p++;
            n -= p; A += p; k -= p; // 否则，在右侧继续查找
        }
    }

    return A[0]; // 如果数组长度为1，直接返回第一个元素
}

// 定义一个函数，用于从单精度浮点数数组中找到第k大的元素值
float sqselect(int n, float A[], int k)
{
    // 声明变量 i, j, p 和 val，用于索引和存储值
    register int i, j, p;
    register float val;

    // 确保 k 的值在有效范围内
    k = SUPERLU_MAX(k, 0); // 将 k 限制在不小于 0 的范围内
    k = SUPERLU_MIN(k, n - 1); // 将 k 限制在不超过 n-1 的范围内

    // 开始进行快速选择算法
    while (n > 1)
    {
        // 初始化 i 和 j，以及选择的基准位置 p 和其值 val
        i = 0; j = n-1;
        p = j; val = A[p];

        // 开始分区过程，将数组 A 按照基准 val 进行分割
        while (i < j)
        {
            // 从左向右找到第一个大于等于基准的元素位置
            for (; A[i] >= val && i < p; i++);
            if (A[i] < val) { A[p] = A[i]; p = i; } // 将找到的元素移到基准位置

            // 从右向左找到第一个小于等于基准的元素位置
            for (; A[j] <= val && j > p; j--);
            if (A[j] > val) { A[p] = A[j]; p = j; } // 将找到的元素移到基准位置
        }

        // 将基准值放置到最终的位置上
        A[p] = val;

        // 如果基准位置 p 正好是要找的第 k 大的位置，则返回该值
        if (p == k) return val;
        else if (p > k) n = p; // 如果基准位置大于 k，则在左侧继续查找
        else
        {
            p++;
            n -= p; A += p; k -= p; // 否则，在右侧继续查找
        }
    }

    return A[0]; // 如果数组长度为1，直接返回第一个元素
}
```