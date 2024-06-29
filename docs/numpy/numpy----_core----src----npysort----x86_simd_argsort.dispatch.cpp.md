# `.\numpy\numpy\_core\src\npysort\x86_simd_argsort.dispatch.cpp`

```
#ifndef __CYGWIN__
#ifndef __CYGWIN__ 检查是否未定义宏 __CYGWIN__
#include "x86_simd_qsort.hpp"
包含头文件 "x86_simd_qsort.hpp"

#include "x86-simd-sort/src/x86simdsort-static-incl.h"
包含头文件 "x86-simd-sort/src/x86simdsort-static-incl.h"

#define DISPATCH_ARG_METHODS(TYPE) \
宏定义 DISPATCH_ARG_METHODS(TYPE)，该宏用于生成模板函数

template<> void NPY_CPU_DISPATCH_CURFX(ArgQSelect)(TYPE* arr, npy_intp* arg, npy_intp num, npy_intp kth) \
模板特化，定义类型为 TYPE 的函数 NPY_CPU_DISPATCH_CURFX(ArgQSelect)，参数为 arr 数组，arg 数组，num 元素数目，kth 第 k 小元素位置
{ \
    调用 x86simdsortStatic::argselect 函数，将 arr 转换为 size_t* 类型的 arg 数组，选择第 kth 小的元素
    x86simdsortStatic::argselect(arr, reinterpret_cast<size_t*>(arg), kth, num, true); \
} \

template<> void NPY_CPU_DISPATCH_CURFX(ArgQSort)(TYPE* arr, npy_intp *arg, npy_intp size) \
模板特化，定义类型为 TYPE 的函数 NPY_CPU_DISPATCH_CURFX(ArgQSort)，参数为 arr 数组，arg 数组，size 元素数目
{ \
    调用 x86simdsortStatic::argsort 函数，将 arr 转换为 size_t* 类型的 arg 数组，按升序排序
    x86simdsortStatic::argsort(arr, reinterpret_cast<size_t*>(arg), size, true); \
} \

namespace np { namespace qsort_simd {
声明命名空间 np::qsort_simd

    DISPATCH_ARG_METHODS(uint32_t)
    通过 DISPATCH_ARG_METHODS 宏生成 uint32_t 类型的函数实现

    DISPATCH_ARG_METHODS(int32_t)
    通过 DISPATCH_ARG_METHODS 宏生成 int32_t 类型的函数实现

    DISPATCH_ARG_METHODS(float)
    通过 DISPATCH_ARG_METHODS 宏生成 float 类型的函数实现

    DISPATCH_ARG_METHODS(uint64_t)
    通过 DISPATCH_ARG_METHODS 宏生成 uint64_t 类型的函数实现

    DISPATCH_ARG_METHODS(int64_t)
    通过 DISPATCH_ARG_METHODS 宏生成 int64_t 类型的函数实现

    DISPATCH_ARG_METHODS(double)
    通过 DISPATCH_ARG_METHODS 宏生成 double 类型的函数实现

}} // namespace np::simd

#endif // __CYGWIN__
#endif // __CYGWIN__
```