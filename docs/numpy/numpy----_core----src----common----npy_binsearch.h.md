# `.\numpy\numpy\_core\src\common\npy_binsearch.h`

```
#ifndef __NPY_BINSEARCH_H__
#define __NPY_BINSEARCH_H__

#include "npy_sort.h"
#include <numpy/npy_common.h>
#include <numpy/ndarraytypes.h>

// 如果没有定义__NPY_BINSEARCH_H__，则定义__NPY_BINSEARCH_H__，避免重复包含

#ifdef __cplusplus
extern "C" {
#endif

// 声明 PyArray_BinSearchFunc 类型的函数指针，该函数用于二分查找
typedef void (PyArray_BinSearchFunc)(const char*, const char*, char*,
                                     npy_intp, npy_intp,
                                     npy_intp, npy_intp, npy_intp,
                                     PyArrayObject*);

// 声明 PyArray_ArgBinSearchFunc 类型的函数指针，该函数用于带有参数的二分查找
typedef int (PyArray_ArgBinSearchFunc)(const char*, const char*,
                                       const char*, char*,
                                       npy_intp, npy_intp, npy_intp,
                                       npy_intp, npy_intp, npy_intp,
                                       PyArrayObject*);

// 获取适合指定数据类型和查找方向的二分查找函数指针
NPY_NO_EXPORT PyArray_BinSearchFunc* get_binsearch_func(PyArray_Descr *dtype, NPY_SEARCHSIDE side);

// 获取适合指定数据类型和查找方向的带参数的二分查找函数指针
NPY_NO_EXPORT PyArray_ArgBinSearchFunc* get_argbinsearch_func(PyArray_Descr *dtype, NPY_SEARCHSIDE side);

#ifdef __cplusplus
}
#endif

#endif
```