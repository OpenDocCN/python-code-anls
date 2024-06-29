# `D:\src\scipysrc\numpy\numpy\_core\src\npysort\highway_qsort.hpp`

```
#ifndef NUMPY_SRC_COMMON_NPYSORT_HWY_SIMD_QSORT_HPP
#define NUMPY_SRC_COMMON_NPYSORT_HWY_SIMD_QSORT_HPP

// 如果未定义 NUMPY_SRC_COMMON_NPYSORT_HWY_SIMD_QSORT_HPP 宏，则包含该头文件，以防止多次包含


#include "common.hpp"

// 包含 common.hpp 头文件，引入通用的常量、函数和数据结构定义


namespace np { namespace highway { namespace qsort_simd {

// 命名空间声明：np -> highway -> qsort_simd，用于组织 qsort_simd 相关的函数和数据结构


#ifndef NPY_DISABLE_OPTIMIZATION
    #include "highway_qsort.dispatch.h"
#endif

// 如果未定义 NPY_DISABLE_OPTIMIZATION 宏，则包含 highway_qsort.dispatch.h 头文件，该文件可能包含了针对 QSort 和 QSelect 函数的优化版本的声明和定义


NPY_CPU_DISPATCH_DECLARE(template <typename T> void QSort, (T *arr, npy_intp size))
NPY_CPU_DISPATCH_DECLARE(template <typename T> void QSelect, (T* arr, npy_intp num, npy_intp kth))

// 使用 NPY_CPU_DISPATCH_DECLARE 宏声明 QSort 和 QSelect 函数模板，这些函数用于对数组进行排序和选择操作


#ifndef NPY_DISABLE_OPTIMIZATION
    #include "highway_qsort_16bit.dispatch.h"
#endif

// 如果未定义 NPY_DISABLE_OPTIMIZATION 宏，则包含 highway_qsort_16bit.dispatch.h 头文件，该文件可能包含了针对 QSort 和 QSelect 函数在 16 位数据类型上的优化版本的声明和定义


NPY_CPU_DISPATCH_DECLARE(template <typename T> void QSort, (T *arr, npy_intp size))
NPY_CPU_DISPATCH_DECLARE(template <typename T> void QSelect, (T* arr, npy_intp num, npy_intp kth))

// 使用 NPY_CPU_DISPATCH_DECLARE 宏声明另一组 QSort 和 QSelect 函数模板，针对 16 位数据类型


} } } // np::highway::qsort_simd

// 命名空间封闭：结束 np -> highway -> qsort_simd 命名空间的声明


#endif // NUMPY_SRC_COMMON_NPYSORT_HWY_SIMD_QSORT_HPP

// 结束 NUMPY_SRC_COMMON_NPYSORT_HWY_SIMD_QSORT_HPP 头文件的条件编译指令
```