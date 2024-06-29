# `.\numpy\numpy\_core\src\common\npy_partition.h`

```py
#ifndef NUMPY_CORE_SRC_COMMON_PARTITION_H_
#define NUMPY_CORE_SRC_COMMON_PARTITION_H_

#include "npy_sort.h"

/* Python include is for future object sorts */
#include <Python.h>

#include <numpy/ndarraytypes.h>
#include <numpy/npy_common.h>

// 定义最大的递归分区栈深度为50
#define NPY_MAX_PIVOT_STACK 50

// 声明一个用于对数组进行分区的函数指针类型
typedef int (PyArray_PartitionFunc)(void *, npy_intp, npy_intp,
                                    npy_intp *, npy_intp *, npy_intp,
                                    void *);

// 声明一个用于对数组进行参数分区的函数指针类型
typedef int (PyArray_ArgPartitionFunc)(void *, npy_intp *, npy_intp, npy_intp,
                                       npy_intp *, npy_intp *, npy_intp,
                                       void *);

// 如果是 C++ 环境，则采用 extern "C" 包裹代码
#ifdef __cplusplus
extern "C" {
#endif

// 根据数据类型和选择类型获取对应的分区函数指针
NPY_NO_EXPORT PyArray_PartitionFunc *
get_partition_func(int type, NPY_SELECTKIND which);

// 根据数据类型和选择类型获取对应的参数分区函数指针
NPY_NO_EXPORT PyArray_ArgPartitionFunc *
get_argpartition_func(int type, NPY_SELECTKIND which);

// 如果是 C++ 环境，结束 extern "C" 区块
#ifdef __cplusplus
}
#endif

#endif
```