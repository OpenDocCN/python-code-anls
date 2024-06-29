# `.\numpy\numpy\_core\src\npysort\heapsort.cpp`

```
/*
 * The purpose of this module is to add faster sort functions
 * that are type-specific.  This is done by altering the
 * function table for the builtin descriptors.
 *
 * These sorting functions are copied almost directly from numarray
 * with a few modifications (complex comparisons compare the imaginary
 * part if the real parts are equal, for example), and the names
 * are changed.
 *
 * The original sorting code is due to Charles R. Harris who wrote
 * it for numarray.
 */

/*
 * Quick sort is usually the fastest, but the worst case scenario can
 * be slower than the merge and heap sorts.  The merge sort requires
 * extra memory and so for large arrays may not be useful.
 *
 * The merge sort is *stable*, meaning that equal components
 * are unmoved from their entry versions, so it can be used to
 * implement lexicographic sorting on multiple keys.
 *
 * The heap sort is included for completeness.
 */

#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include "npy_sort.h"              /* 包含排序相关的头文件 */
#include "npysort_common.h"        /* 包含排序通用的头文件 */
#include "numpy_tag.h"             /* 包含 numpy 标签相关的头文件 */

#include "npysort_heapsort.h"      /* 包含堆排序的头文件 */

#include <cstdlib>                 /* 包含标准库函数 */

#define NOT_USED NPY_UNUSED(unused) /* 定义一个宏，表示未使用的变量 */
#define PYA_QS_STACK 100            /* 定义快速排序的堆栈大小 */
#define SMALL_QUICKSORT 15          /* 定义快速排序的阈值 */
#define SMALL_MERGESORT 20          /* 定义归并排序的阈值 */
#define SMALL_STRING 16             /* 定义字符串的阈值 */


/*
 *****************************************************************************
 **                             GENERIC SORT                                **
 *****************************************************************************
 */

NPY_NO_EXPORT int
npy_heapsort(void *start, npy_intp num, void *varr)
{
    PyArrayObject *arr = (PyArrayObject *)varr;   /* 将输入转换为 PyArrayObject 类型 */
    npy_intp elsize = PyArray_ITEMSIZE(arr);      /* 获取数组元素的大小 */
    PyArray_CompareFunc *cmp = PyDataType_GetArrFuncs(PyArray_DESCR(arr))->compare;  /* 获取比较函数 */

    if (elsize == 0) {
        return 0;  /* 如果元素大小为0，无需排序 */
    }

    char *tmp = (char *)malloc(elsize);  /* 分配临时空间 */
    char *a = (char *)start - elsize;     /* 起始位置 */

    npy_intp i, j, l;

    if (tmp == NULL) {
        return -NPY_ENOMEM;  /* 如果内存分配失败，返回内存不足错误 */
    }

    for (l = num >> 1; l > 0; --l) {
        GENERIC_COPY(tmp, a + l * elsize, elsize);  /* 复制元素到临时空间 */
        for (i = l, j = l << 1; j <= num;) {
            if (j < num &&
                cmp(a + j * elsize, a + (j + 1) * elsize, arr) < 0) {
                ++j;  /* 如果右边元素比较大，移动到下一个 */
            }
            if (cmp(tmp, a + j * elsize, arr) < 0) {
                GENERIC_COPY(a + i * elsize, a + j * elsize, elsize);  /* 复制元素 */
                i = j;  /* 更新位置 */
                j += j; /* 增加步长 */
            }
            else {
                break;  /* 跳出循环 */
            }
        }
        GENERIC_COPY(a + i * elsize, tmp, elsize);  /* 复制元素 */
    }
    // 循环执行直到 num 变为 1
    for (; num > 1;) {
        // 将 a[num * elsize] 处的元素复制到 tmp 中
        GENERIC_COPY(tmp, a + num * elsize, elsize);
        // 将 a[elsize] 处的元素复制到 a[num * elsize] 处
        GENERIC_COPY(a + num * elsize, a + elsize, elsize);
        // num 减少 1
        num -= 1;
        // 开始堆排序的调整过程
        for (i = 1, j = 2; j <= num;) {
            // 比较 a[j * elsize] 和 a[(j + 1) * elsize]，选择较大的
            if (j < num &&
                cmp(a + j * elsize, a + (j + 1) * elsize, arr) < 0) {
                ++j;
            }
            // 如果 tmp 小于 a[j * elsize]，则进行元素交换
            if (cmp(tmp, a + j * elsize, arr) < 0) {
                GENERIC_COPY(a + i * elsize, a + j * elsize, elsize);
                i = j;
                j += j;  // j 的增加方式有误，应该为 j *= 2
            }
            else {
                break;
            }
        }
        // 将 tmp 复制到 a[i * elsize] 处
        GENERIC_COPY(a + i * elsize, tmp, elsize);
    }

    // 释放 tmp 所占用的内存
    free(tmp);
    // 返回 0 表示排序完成
    return 0;
}

NPY_NO_EXPORT int
npy_aheapsort(void *vv, npy_intp *tosort, npy_intp n, void *varr)
{
    // 将输入指针 vv 转换为 char 类型，这样可以按字节访问数据
    char *v = (char *)vv;
    // 将输入指针 varr 转换为 PyArrayObject 类型，以便访问 NumPy 数组的信息和方法
    PyArrayObject *arr = (PyArrayObject *)varr;
    // 计算数组元素的大小（以字节为单位）
    npy_intp elsize = PyArray_ITEMSIZE(arr);
    // 获取 NumPy 数组的比较函数指针，用于比较数组元素
    PyArray_CompareFunc *cmp = PyDataType_GetArrFuncs(PyArray_DESCR(arr))->compare;
    // 定义指向排序后索引数组的指针 a，进行堆排序需要偏移一位
    npy_intp *a, i, j, l, tmp;

    /* The array needs to be offset by one for heapsort indexing */
    // 对排序索引数组 tosort 进行偏移，使其能正确用于堆排序
    a = tosort - 1;

    // 建立最大堆
    for (l = n >> 1; l > 0; --l) {
        tmp = a[l];
        for (i = l, j = l << 1; j <= n;) {
            // 比较子节点和父节点的值，并根据需要交换它们，以维持堆的性质
            if (j < n &&
                cmp(v + a[j] * elsize, v + a[j + 1] * elsize, arr) < 0) {
                ++j;
            }
            if (cmp(v + tmp * elsize, v + a[j] * elsize, arr) < 0) {
                a[i] = a[j];
                i = j;
                j += j;
            }
            else {
                break;
            }
        }
        a[i] = tmp;
    }

    // 堆排序
    for (; n > 1;) {
        tmp = a[n];
        a[n] = a[1];
        n -= 1;
        for (i = 1, j = 2; j <= n;) {
            // 比较子节点和父节点的值，并根据需要交换它们，以维持堆的性质
            if (j < n &&
                cmp(v + a[j] * elsize, v + a[j + 1] * elsize, arr) < 0) {
                ++j;
            }
            if (cmp(v + tmp * elsize, v + a[j] * elsize, arr) < 0) {
                a[i] = a[j];
                i = j;
                j += j;
            }
            else {
                break;
            }
        }
        a[i] = tmp;
    }

    // 堆排序完成，返回 0 表示成功
    return 0;
}

/***************************************
 * C > C++ dispatch
 ***************************************/

// 布尔类型的堆排序特化模板实例化及函数定义
template NPY_NO_EXPORT int
heapsort_<npy::bool_tag, npy_bool>(npy_bool *, npy_intp);
NPY_NO_EXPORT int
heapsort_bool(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    // 调用布尔类型的堆排序函数并返回结果
    return heapsort_<npy::bool_tag>((npy_bool *)start, n);
}

// 字节类型的堆排序特化模板实例化及函数定义
template NPY_NO_EXPORT int
heapsort_<npy::byte_tag, npy_byte>(npy_byte *, npy_intp);
NPY_NO_EXPORT int
heapsort_byte(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    // 调用字节类型的堆排序函数并返回结果
    return heapsort_<npy::byte_tag>((npy_byte *)start, n);
}

// 无符号字节类型的堆排序特化模板实例化及函数定义
template NPY_NO_EXPORT int
heapsort_<npy::ubyte_tag, npy_ubyte>(npy_ubyte *, npy_intp);
NPY_NO_EXPORT int
heapsort_ubyte(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    // 调用无符号字节类型的堆排序函数并返回结果
    return heapsort_<npy::ubyte_tag>((npy_ubyte *)start, n);
}

// 短整型类型的堆排序特化模板实例化及函数定义
template NPY_NO_EXPORT int
heapsort_<npy::short_tag, npy_short>(npy_short *, npy_intp);
NPY_NO_EXPORT int
heapsort_short(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    // 调用短整型类型的堆排序函数并返回结果
    return heapsort_<npy::short_tag>((npy_short *)start, n);
}

// 无符号短整型类型的堆排序特化模板实例化及函数定义
template NPY_NO_EXPORT int
heapsort_<npy::ushort_tag, npy_ushort>(npy_ushort *, npy_intp);
NPY_NO_EXPORT int
heapsort_ushort(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    // 调用无符号短整型类型的堆排序函数并返回结果
    return heapsort_<npy::ushort_tag>((npy_ushort *)start, n);
}

// 整型类型的堆排序特化模板实例化及函数定义
template NPY_NO_EXPORT int
heapsort_<npy::int_tag, npy_int>(npy_int *, npy_intp);
NPY_NO_EXPORT int
heapsort_int(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    // 调用整型类型的堆排序函数并返回结果
    return heapsort_<npy::int_tag>((npy_int *)start, n);
}

// 模板函数的声明部分截至此处
template NPY_NO_EXPORT int
heapsort_<npy::uint_tag, npy_uint>(npy_uint *, npy_intp);
# 声明一个模板函数 heapsort_，用于对无符号整数数组进行堆排序

NPY_NO_EXPORT int
heapsort_uint(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    # 调用具体化的 heapsort_ 函数来对无符号整数数组进行堆排序，并返回排序结果
    return heapsort_<npy::uint_tag>((npy_uint *)start, n);
}

template NPY_NO_EXPORT int
heapsort_<npy::long_tag, npy_long>(npy_long *, npy_intp);
# 声明一个模板函数 heapsort_，用于对长整型数组进行堆排序

NPY_NO_EXPORT int
heapsort_long(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    # 调用具体化的 heapsort_ 函数来对长整型数组进行堆排序，并返回排序结果
    return heapsort_<npy::long_tag>((npy_long *)start, n);
}

template NPY_NO_EXPORT int
heapsort_<npy::ulong_tag, npy_ulong>(npy_ulong *, npy_intp);
# 声明一个模板函数 heapsort_，用于对无符号长整型数组进行堆排序

NPY_NO_EXPORT int
heapsort_ulong(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    # 调用具体化的 heapsort_ 函数来对无符号长整型数组进行堆排序，并返回排序结果
    return heapsort_<npy::ulong_tag>((npy_ulong *)start, n);
}

template NPY_NO_EXPORT int
heapsort_<npy::longlong_tag, npy_longlong>(npy_longlong *, npy_intp);
# 声明一个模板函数 heapsort_，用于对长长整型数组进行堆排序

NPY_NO_EXPORT int
heapsort_longlong(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    # 调用具体化的 heapsort_ 函数来对长长整型数组进行堆排序，并返回排序结果
    return heapsort_<npy::longlong_tag>((npy_longlong *)start, n);
}

template NPY_NO_EXPORT int
heapsort_<npy::ulonglong_tag, npy_ulonglong>(npy_ulonglong *, npy_intp);
# 声明一个模板函数 heapsort_，用于对无符号长长整型数组进行堆排序

NPY_NO_EXPORT int
heapsort_ulonglong(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    # 调用具体化的 heapsort_ 函数来对无符号长长整型数组进行堆排序，并返回排序结果
    return heapsort_<npy::ulonglong_tag>((npy_ulonglong *)start, n);
}

template NPY_NO_EXPORT int
heapsort_<npy::half_tag, npy_half>(npy_half *, npy_intp);
# 声明一个模板函数 heapsort_，用于对半精度浮点数数组进行堆排序

NPY_NO_EXPORT int
heapsort_half(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    # 调用具体化的 heapsort_ 函数来对半精度浮点数数组进行堆排序，并返回排序结果
    return heapsort_<npy::half_tag>((npy_half *)start, n);
}

template NPY_NO_EXPORT int
heapsort_<npy::float_tag, npy_float>(npy_float *, npy_intp);
# 声明一个模板函数 heapsort_，用于对单精度浮点数数组进行堆排序

NPY_NO_EXPORT int
heapsort_float(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    # 调用具体化的 heapsort_ 函数来对单精度浮点数数组进行堆排序，并返回排序结果
    return heapsort_<npy::float_tag>((npy_float *)start, n);
}

template NPY_NO_EXPORT int
heapsort_<npy::double_tag, npy_double>(npy_double *, npy_intp);
# 声明一个模板函数 heapsort_，用于对双精度浮点数数组进行堆排序

NPY_NO_EXPORT int
heapsort_double(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    # 调用具体化的 heapsort_ 函数来对双精度浮点数数组进行堆排序，并返回排序结果
    return heapsort_<npy::double_tag>((npy_double *)start, n);
}

template NPY_NO_EXPORT int
heapsort_<npy::longdouble_tag, npy_longdouble>(npy_longdouble *, npy_intp);
# 声明一个模板函数 heapsort_，用于对长双精度浮点数数组进行堆排序

NPY_NO_EXPORT int
heapsort_longdouble(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    # 调用具体化的 heapsort_ 函数来对长双精度浮点数数组进行堆排序，并返回排序结果
    return heapsort_<npy::longdouble_tag>((npy_longdouble *)start, n);
}

template NPY_NO_EXPORT int
heapsort_<npy::cfloat_tag, npy_cfloat>(npy_cfloat *, npy_intp);
# 声明一个模板函数 heapsort_，用于对复数浮点数数组进行堆排序

NPY_NO_EXPORT int
heapsort_cfloat(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    # 调用具体化的 heapsort_ 函数来对复数浮点数数组进行堆排序，并返回排序结果
    return heapsort_<npy::cfloat_tag>((npy_cfloat *)start, n);
}

template NPY_NO_EXPORT int
heapsort_<npy::cdouble_tag, npy_cdouble>(npy_cdouble *, npy_intp);
# 声明一个模板函数 heapsort_，用于对复数双精度浮点数数组进行堆排序

NPY_NO_EXPORT int
heapsort_cdouble(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    # 调用具体化的 heapsort_ 函数来对复数双精度浮点数数组进行堆排序，并返回排序结果
    return heapsort_<npy::cdouble_tag>((npy_cdouble *)start, n);
}

template NPY_NO_EXPORT int
heapsort_<npy::clongdouble_tag, npy_clongdouble>(npy_clongdouble *, npy_intp);
# 声明一个模板函数 heapsort_，用于对复数长双精度浮点数数组进行堆排序

NPY_NO_EXPORT int
heapsort_clongdouble(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    # 调用具体化的 heapsort_ 函数来对复数长双精度浮点数数组进行堆排序，并返回排序结果
    return heapsort_<npy::clongdouble_tag>((npy_clongdouble *)start, n);
}
//cpp
// 声明一个模板函数，用于对日期时间类型进行堆排序，函数参数为日期时间数组和数组长度
template NPY_NO_EXPORT int heapsort_<npy::datetime_tag, npy_datetime>(npy_datetime *, npy_intp);

// 实现对日期时间类型进行堆排序的函数，函数参数为起始地址、数组长度和一个未使用的参数（varr）
NPY_NO_EXPORT int heapsort_datetime(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    // 调用模板函数 heapsort_<npy::datetime_tag> 进行堆排序，返回排序后的结果
    return heapsort_<npy::datetime_tag>((npy_datetime *)start, n);
}

// 类似地，声明一个模板函数，用于对时间差类型进行堆排序
template NPY_NO_EXPORT int heapsort_<npy::timedelta_tag, npy_timedelta>(npy_timedelta *, npy_intp);

// 实现对时间差类型进行堆排序的函数，参数和用法与上一个函数相似
NPY_NO_EXPORT int heapsort_timedelta(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    return heapsort_<npy::timedelta_tag>((npy_timedelta *)start, n);
}

// 声明一个模板函数，用于对布尔类型进行堆排序
template NPY_NO_EXPORT int aheapsort_<npy::bool_tag, npy_bool>(npy_bool *vv, npy_intp *tosort, npy_intp n);

// 实现对布尔类型进行堆排序的函数，参数和用法与上一个函数相似
NPY_NO_EXPORT int aheapsort_bool(void *vv, npy_intp *tosort, npy_intp n, void *NPY_UNUSED(varr))
{
    return aheapsort_<npy::bool_tag>((npy_bool *)vv, tosort, n);
}

// 类似地，声明模板函数和实现函数，用于对字节类型进行堆排序
template NPY_NO_EXPORT int aheapsort_<npy::byte_tag, npy_byte>(npy_byte *vv, npy_intp *tosort, npy_intp n);
NPY_NO_EXPORT int aheapsort_byte(void *vv, npy_intp *tosort, npy_intp n, void *NPY_UNUSED(varr))
{
    return aheapsort_<npy::byte_tag>((npy_byte *)vv, tosort, n);
}

// 对无符号字节类型进行堆排序
template NPY_NO_EXPORT int aheapsort_<npy::ubyte_tag, npy_ubyte>(npy_ubyte *vv, npy_intp *tosort, npy_intp n);
NPY_NO_EXPORT int aheapsort_ubyte(void *vv, npy_intp *tosort, npy_intp n, void *NPY_UNUSED(varr))
{
    return aheapsort_<npy::ubyte_tag>((npy_ubyte *)vv, tosort, n);
}

// 对短整型进行堆排序
template NPY_NO_EXPORT int aheapsort_<npy::short_tag, npy_short>(npy_short *vv, npy_intp *tosort, npy_intp n);
NPY_NO_EXPORT int aheapsort_short(void *vv, npy_intp *tosort, npy_intp n, void *NPY_UNUSED(varr))
{
    return aheapsort_<npy::short_tag>((npy_short *)vv, tosort, n);
}

// 对无符号短整型进行堆排序
template NPY_NO_EXPORT int aheapsort_<npy::ushort_tag, npy_ushort>(npy_ushort *vv, npy_intp *tosort, npy_intp n);
NPY_NO_EXPORT int aheapsort_ushort(void *vv, npy_intp *tosort, npy_intp n, void *NPY_UNUSED(varr))
{
    return aheapsort_<npy::ushort_tag>((npy_ushort *)vv, tosort, n);
}

// 对整型进行堆排序
template NPY_NO_EXPORT int aheapsort_<npy::int_tag, npy_int>(npy_int *vv, npy_intp *tosort, npy_intp n);
NPY_NO_EXPORT int aheapsort_int(void *vv, npy_intp *tosort, npy_intp n, void *NPY_UNUSED(varr))
{
    return aheapsort_<npy::int_tag>((npy_int *)vv, tosort, n);
}

// 对无符号整型进行堆排序
template NPY_NO_EXPORT int aheapsort_<npy::uint_tag, npy_uint>(npy_uint *vv, npy_intp *tosort, npy_intp n);
NPY_NO_EXPORT int aheapsort_uint(void *vv, npy_intp *tosort, npy_intp n, void *NPY_UNUSED(varr))
{
    return aheapsort_<npy::uint_tag>((npy_uint *)vv, tosort, n);
}

// 对长整型进行堆排序
template NPY_NO_EXPORT int aheapsort_<npy::long_tag, npy_long>(npy_long *vv, npy_intp *tosort, npy_intp n);
NPY_NO_EXPORT int aheapsort_long(void *vv, npy_intp *tosort, npy_intp n, void *NPY_UNUSED(varr))
{
    return aheapsort_<npy::long_tag>((npy_long *)vv, tosort, n);
}
# 实例化模板，对无符号长整型数组进行堆排序
template NPY_NO_EXPORT int
aheapsort_<npy::ulong_tag, npy_ulong>(npy_ulong *vv, npy_intp *tosort,
                                      npy_intp n);
# 声明对无符号长整型数组进行堆排序的函数
NPY_NO_EXPORT int
aheapsort_ulong(void *vv, npy_intp *tosort, npy_intp n, void *NPY_UNUSED(varr))
{
    return aheapsort_<npy::ulong_tag>((npy_ulong *)vv, tosort, n);
}

# 实例化模板，对长长整型数组进行堆排序
template NPY_NO_EXPORT int
aheapsort_<npy::longlong_tag, npy_longlong>(npy_longlong *vv, npy_intp *tosort,
                                            npy_intp n);
# 声明对长长整型数组进行堆排序的函数
NPY_NO_EXPORT int
aheapsort_longlong(void *vv, npy_intp *tosort, npy_intp n,
                   void *NPY_UNUSED(varr))
{
    return aheapsort_<npy::longlong_tag>((npy_longlong *)vv, tosort, n);
}

# 实例化模板，对无符号长长整型数组进行堆排序
template NPY_NO_EXPORT int
aheapsort_<npy::ulonglong_tag, npy_ulonglong>(npy_ulonglong *vv,
                                              npy_intp *tosort, npy_intp n);
# 声明对无符号长长整型数组进行堆排序的函数
NPY_NO_EXPORT int
aheapsort_ulonglong(void *vv, npy_intp *tosort, npy_intp n,
                    void *NPY_UNUSED(varr))
{
    return aheapsort_<npy::ulonglong_tag>((npy_ulonglong *)vv, tosort, n);
}

# 实例化模板，对半精度浮点数组进行堆排序
template NPY_NO_EXPORT int
aheapsort_<npy::half_tag, npy_half>(npy_half *vv, npy_intp *tosort,
                                    npy_intp n);
# 声明对半精度浮点数组进行堆排序的函数
NPY_NO_EXPORT int
aheapsort_half(void *vv, npy_intp *tosort, npy_intp n, void *NPY_UNUSED(varr))
{
    return aheapsort_<npy::half_tag>((npy_half *)vv, tosort, n);
}

# 实例化模板，对单精度浮点数组进行堆排序
template NPY_NO_EXPORT int
aheapsort_<npy::float_tag, npy_float>(npy_float *vv, npy_intp *tosort,
                                      npy_intp n);
# 声明对单精度浮点数组进行堆排序的函数
NPY_NO_EXPORT int
aheapsort_float(void *vv, npy_intp *tosort, npy_intp n, void *NPY_UNUSED(varr))
{
    return aheapsort_<npy::float_tag>((npy_float *)vv, tosort, n);
}

# 实例化模板，对双精度浮点数组进行堆排序
template NPY_NO_EXPORT int
aheapsort_<npy::double_tag, npy_double>(npy_double *vv, npy_intp *tosort,
                                        npy_intp n);
# 声明对双精度浮点数组进行堆排序的函数
NPY_NO_EXPORT int
aheapsort_double(void *vv, npy_intp *tosort, npy_intp n,
                 void *NPY_UNUSED(varr))
{
    return aheapsort_<npy::double_tag>((npy_double *)vv, tosort, n);
}

# 实例化模板，对长双精度浮点数组进行堆排序
template NPY_NO_EXPORT int
aheapsort_<npy::longdouble_tag, npy_longdouble>(npy_longdouble *vv,
                                                npy_intp *tosort, npy_intp n);
# 声明对长双精度浮点数组进行堆排序的函数
NPY_NO_EXPORT int
aheapsort_longdouble(void *vv, npy_intp *tosort, npy_intp n,
                     void *NPY_UNUSED(varr))
{
    return aheapsort_<npy::longdouble_tag>((npy_longdouble *)vv, tosort, n);
}

# 实例化模板，对复数浮点数组进行堆排序
template NPY_NO_EXPORT int
aheapsort_<npy::cfloat_tag, npy_cfloat>(npy_cfloat *vv, npy_intp *tosort,
                                        npy_intp n);
# 声明对复数浮点数组进行堆排序的函数
NPY_NO_EXPORT int
aheapsort_cfloat(void *vv, npy_intp *tosort, npy_intp n,
                 void *NPY_UNUSED(varr))
{
    return aheapsort_<npy::cfloat_tag>((npy_cfloat *)vv, tosort, n);
}

# 实例化模板，对复数双精度浮点数组进行堆排序
template NPY_NO_EXPORT int
aheapsort_<npy::cdouble_tag, npy_cdouble>(npy_cdouble *vv, npy_intp *tosort,
                                          npy_intp n);
# 声明对复数双精度浮点数组进行堆排序的函数
// 调用模板函数 aheapsort_，对 void 类型的数组进行堆排序，并返回排序后的结果
aheapsort_cdouble(void *vv, npy_intp *tosort, npy_intp n,
                  void *NPY_UNUSED(varr))
{
    // 调用特化的 aheapsort_ 函数，处理 npy_cdouble 类型的数据
    return aheapsort_<npy::cdouble_tag>((npy_cdouble *)vv, tosort, n);
}

// 显式实例化模板函数 aheapsort_，处理 npy_clongdouble 类型的数组排序
template NPY_NO_EXPORT int
aheapsort_<npy::clongdouble_tag, npy_clongdouble>(npy_clongdouble *vv,
                                                  npy_intp *tosort,
                                                  npy_intp n);
NPY_NO_EXPORT int
// 对 void 类型的数组进行堆排序，返回排序后的结果
aheapsort_clongdouble(void *vv, npy_intp *tosort, npy_intp n,
                      void *NPY_UNUSED(varr))
{
    // 调用特化的 aheapsort_ 函数，处理 npy_clongdouble 类型的数据
    return aheapsort_<npy::clongdouble_tag>((npy_clongdouble *)vv, tosort, n);
}

// 显式实例化模板函数 aheapsort_，处理 npy_datetime 类型的数组排序
template NPY_NO_EXPORT int
aheapsort_<npy::datetime_tag, npy_datetime>(npy_datetime *vv, npy_intp *tosort,
                                            npy_intp n);
NPY_NO_EXPORT int
// 对 void 类型的数组进行堆排序，返回排序后的结果
aheapsort_datetime(void *vv, npy_intp *tosort, npy_intp n,
                   void *NPY_UNUSED(varr))
{
    // 调用特化的 aheapsort_ 函数，处理 npy_datetime 类型的数据
    return aheapsort_<npy::datetime_tag>((npy_datetime *)vv, tosort, n);
}

// 显式实例化模板函数 aheapsort_，处理 npy_timedelta 类型的数组排序
template NPY_NO_EXPORT int
aheapsort_<npy::timedelta_tag, npy_timedelta>(npy_timedelta *vv,
                                              npy_intp *tosort, npy_intp n);
NPY_NO_EXPORT int
// 对 void 类型的数组进行堆排序，返回排序后的结果
aheapsort_timedelta(void *vv, npy_intp *tosort, npy_intp n,
                    void *NPY_UNUSED(varr))
{
    // 调用特化的 aheapsort_ 函数，处理 npy_timedelta 类型的数据
    return aheapsort_<npy::timedelta_tag>((npy_timedelta *)vv, tosort, n);
}

// 对 void 类型的数组进行堆排序，使用 string_heapsort_ 函数处理 npy::string_tag 类型的数据
NPY_NO_EXPORT int
heapsort_string(void *start, npy_intp n, void *varr)
{
    return string_heapsort_<npy::string_tag>((npy_char *)start, n, varr);
}

// 对 void 类型的数组进行堆排序，使用 string_heapsort_ 函数处理 npy::unicode_tag 类型的数据
NPY_NO_EXPORT int
heapsort_unicode(void *start, npy_intp n, void *varr)
{
    return string_heapsort_<npy::unicode_tag>((npy_ucs4 *)start, n, varr);
}

// 对 void 类型的数组进行堆排序，使用 string_aheapsort_ 函数处理 npy::string_tag 类型的数据
NPY_NO_EXPORT int
aheapsort_string(void *vv, npy_intp *tosort, npy_intp n, void *varr)
{
    return string_aheapsort_<npy::string_tag>((npy_char *)vv, tosort, n, varr);
}

// 对 void 类型的数组进行堆排序，使用 string_aheapsort_ 函数处理 npy::unicode_tag 类型的数据
NPY_NO_EXPORT int
aheapsort_unicode(void *vv, npy_intp *tosort, npy_intp n, void *varr)
{
    return string_aheapsort_<npy::unicode_tag>((npy_ucs4 *)vv, tosort, n,
                                               varr);
}
```