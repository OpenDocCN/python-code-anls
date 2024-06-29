# `.\numpy\numpy\_core\src\npysort\x86_simd_qsort_16bit.dispatch.cpp`

```py
/*
 * 包含 x86 SIMD 快速排序头文件
 */
#include "x86_simd_qsort.hpp"

/*
 * 如果不是在 Cygwin 环境下编译，则包含 x86 SIMD 排序的静态头文件
 */
#ifndef __CYGWIN__
#include "x86-simd-sort/src/x86simdsort-static-incl.h"

/*
 * 对于 MSVC 编译器，由于未设置 __AVX512VBMI2__ 宏，需要手动包含此文件
 */
#ifdef _MSC_VER
#include "x86-simd-sort/src/avx512-16bit-qsort.hpp"
#endif

namespace np { namespace qsort_simd {

/*
 * QSelect 分发函数：
 */

/*
 * 特化模板，用于处理 Half 类型数组的 QSelect 操作
 */
template<> void NPY_CPU_DISPATCH_CURFX(QSelect)(Half *arr, npy_intp num, npy_intp kth)
{
    /*
     * 如果支持 AVX512_SPR 指令集，则调用 x86simdsortStatic 的 qselect 函数处理 _Float16 数组
     */
#if defined(NPY_HAVE_AVX512_SPR)
    x86simdsortStatic::qselect(reinterpret_cast<_Float16*>(arr), kth, num, true);
#else
    /*
     * 否则，调用 avx512_qselect_fp16 函数处理 uint16_t 数组
     */
    avx512_qselect_fp16(reinterpret_cast<uint16_t*>(arr), kth, num, true, false);
#endif
}

/*
 * 特化模板，用于处理 uint16_t 类型数组的 QSelect 操作
 */
template<> void NPY_CPU_DISPATCH_CURFX(QSelect)(uint16_t *arr, npy_intp num, npy_intp kth)
{
    /*
     * 调用 x86simdsortStatic 的 qselect 函数处理 uint16_t 数组
     */
    x86simdsortStatic::qselect(arr, kth, num);
}

/*
 * 特化模板，用于处理 int16_t 类型数组的 QSelect 操作
 */
template<> void NPY_CPU_DISPATCH_CURFX(QSelect)(int16_t *arr, npy_intp num, npy_intp kth)
{
    /*
     * 调用 x86simdsortStatic 的 qselect 函数处理 int16_t 数组
     */
    x86simdsortStatic::qselect(arr, kth, num);
}

/*
 * QSort 分发函数：
 */

/*
 * 特化模板，用于处理 Half 类型数组的 QSort 操作
 */
template<> void NPY_CPU_DISPATCH_CURFX(QSort)(Half *arr, npy_intp size)
{
    /*
     * 如果支持 AVX512_SPR 指令集，则调用 x86simdsortStatic 的 qsort 函数处理 _Float16 数组
     */
#if defined(NPY_HAVE_AVX512_SPR)
    x86simdsortStatic::qsort(reinterpret_cast<_Float16*>(arr), size, true);
#else
    /*
     * 否则，调用 avx512_qsort_fp16 函数处理 uint16_t 数组
     */
    avx512_qsort_fp16(reinterpret_cast<uint16_t*>(arr), size, true, false);
#endif
}

/*
 * 特化模板，用于处理 uint16_t 类型数组的 QSort 操作
 */
template<> void NPY_CPU_DISPATCH_CURFX(QSort)(uint16_t *arr, npy_intp size)
{
    /*
     * 调用 x86simdsortStatic 的 qsort 函数处理 uint16_t 数组
     */
    x86simdsortStatic::qsort(arr, size);
}

/*
 * 特化模板，用于处理 int16_t 类型数组的 QSort 操作
 */
template<> void NPY_CPU_DISPATCH_CURFX(QSort)(int16_t *arr, npy_intp size)
{
    /*
     * 调用 x86simdsortStatic 的 qsort 函数处理 int16_t 数组
     */
    x86simdsortStatic::qsort(arr, size);
}

}} // namespace np::qsort_simd

#endif // __CYGWIN__
```