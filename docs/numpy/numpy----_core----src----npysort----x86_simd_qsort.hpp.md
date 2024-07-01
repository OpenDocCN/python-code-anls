# `.\numpy\numpy\_core\src\npysort\x86_simd_qsort.hpp`

```py
#ifndef NUMPY_SRC_COMMON_NPYSORT_X86_SIMD_QSORT_HPP
#define NUMPY_SRC_COMMON_NPYSORT_X86_SIMD_QSORT_HPP

// 如果未定义 NUMPY_SRC_COMMON_NPYSORT_X86_SIMD_QSORT_HPP 宏，则开始此头文件的条件编译


#include "common.hpp"

// 包含 common.hpp 头文件，用于引入常用的公共功能和定义


namespace np { namespace qsort_simd {

// 命名空间 np::qsort_simd 的开始


#ifndef NPY_DISABLE_OPTIMIZATION
    #include "x86_simd_qsort.dispatch.h"
#endif

// 如果未禁用优化，则包含 x86_simd_qsort.dispatch.h 头文件，该文件用于优化排序算法的分发


NPY_CPU_DISPATCH_DECLARE(template <typename T> void QSort, (T *arr, npy_intp size))

// 声明模板函数 QSort，用于对类型 T 的数组 arr 进行快速排序，size 为数组大小


NPY_CPU_DISPATCH_DECLARE(template <typename T> void QSelect, (T* arr, npy_intp num, npy_intp kth))

// 声明模板函数 QSelect，用于在类型 T 的数组 arr 中选择第 kth 小的元素，num 是数组中的元素数目


#ifndef NPY_DISABLE_OPTIMIZATION
    #include "x86_simd_argsort.dispatch.h"
#endif

// 如果未禁用优化，则包含 x86_simd_argsort.dispatch.h 头文件，用于优化参数排序算法的分发


NPY_CPU_DISPATCH_DECLARE(template <typename T> void ArgQSort, (T *arr, npy_intp* arg, npy_intp size))

// 声明模板函数 ArgQSort，用于对类型 T 的数组 arr 和对应的参数数组 arg 进行排序，size 是数组大小


NPY_CPU_DISPATCH_DECLARE(template <typename T> void ArgQSelect, (T *arr, npy_intp* arg, npy_intp kth, npy_intp size))

// 声明模板函数 ArgQSelect，用于在类型 T 的数组 arr 中基于参数数组 arg 选择第 kth 小的元素，size 是数组大小


#ifndef NPY_DISABLE_OPTIMIZATION
    #include "x86_simd_qsort_16bit.dispatch.h"
#endif

// 如果未禁用优化，则包含 x86_simd_qsort_16bit.dispatch.h 头文件，用于优化16位整数排序算法的分发


NPY_CPU_DISPATCH_DECLARE(template <typename T> void QSort, (T *arr, npy_intp size))

// 再次声明模板函数 QSort，用于对类型 T 的数组 arr 进行快速排序，size 为数组大小


NPY_CPU_DISPATCH_DECLARE(template <typename T> void QSelect, (T* arr, npy_intp num, npy_intp kth))

// 再次声明模板函数 QSelect，用于在类型 T 的数组 arr 中选择第 kth 小的元素，num 是数组中的元素数目


} } // np::qsort_simd

// 命名空间 np::qsort_simd 的结束


#endif // NUMPY_SRC_COMMON_NPYSORT_X86_SIMD_QSORT_HPP

// 结束条件编译指令，关闭 NUMPY_SRC_COMMON_NPYSORT_X86_SIMD_QSORT_HPP 宏定义的作用范围
```