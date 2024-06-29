# `.\numpy\numpy\_core\src\umath\clip.cpp`

```py
/**
 * This module provides the inner loops for the clip ufunc
 */

// 包含必要的头文件，定义和声明
#include <type_traits>

#define _UMATHMODULE
#define _MULTIARRAYMODULE
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

// 引入 Python 头文件
#define PY_SSIZE_T_CLEAN
#include <Python.h>

// 引入 NumPy 中的特定头文件
#include "numpy/halffloat.h"
#include "numpy/ndarraytypes.h"
#include "numpy/npy_common.h"
#include "numpy/npy_math.h"
#include "numpy/utils.h"

// 引入特定的宏定义文件
#include "fast_loop_macros.h"

// 引入自定义的 NumPy 标签
#include "../common/numpy_tag.h"

// 定义模板函数 _NPY_MIN，用于整数类型的最小值比较
template <class T>
T
_NPY_MIN(T a, T b, npy::integral_tag const &)
{
    return PyArray_MIN(a, b);
}

// 定义模板函数 _NPY_MAX，用于整数类型的最大值比较
template <class T>
T
_NPY_MAX(T a, T b, npy::integral_tag const &)
{
    return PyArray_MAX(a, b);
}

// 定义针对 npy_half 类型的最小值比较函数
npy_half
_NPY_MIN(npy_half a, npy_half b, npy::half_tag const &)
{
    return npy_half_isnan(a) || npy_half_le(a, b) ? (a) : (b);
}

// 定义针对 npy_half 类型的最大值比较函数
npy_half
_NPY_MAX(npy_half a, npy_half b, npy::half_tag const &)
{
    return npy_half_isnan(a) || npy_half_ge(a, b) ? (a) : (b);
}

// 定义针对浮点数类型的最小值比较函数
template <class T>
T
_NPY_MIN(T a, T b, npy::floating_point_tag const &)
{
    return npy_isnan(a) ? (a) : PyArray_MIN(a, b);
}

// 定义针对浮点数类型的最大值比较函数
template <class T>
T
_NPY_MAX(T a, T b, npy::floating_point_tag const &)
{
    return npy_isnan(a) ? (a) : PyArray_MAX(a, b);
}

// 定义宏函数 PyArray_CLT，用于复数类型的小于比较
#define PyArray_CLT(p,q,suffix) (((npy_creal##suffix(p)==npy_creal##suffix(q)) ? (npy_cimag##suffix(p) < npy_cimag##suffix(q)) : \
                               (npy_creal##suffix(p) < npy_creal##suffix(q))))

// 定义宏函数 PyArray_CGT，用于复数类型的大于比较
#define PyArray_CGT(p,q,suffix) (((npy_creal##suffix(p)==npy_creal##suffix(q)) ? (npy_cimag##suffix(p) > npy_cimag##suffix(q)) : \
                               (npy_creal##suffix(p) > npy_creal##suffix(q))))

// 定义针对 npy_cdouble 类型的最小值比较函数
npy_cdouble
_NPY_MIN(npy_cdouble a, npy_cdouble b, npy::complex_tag const &)
{
    return npy_isnan(npy_creal(a)) || npy_isnan(npy_cimag(a)) || PyArray_CLT(a, b,)
                ? (a)
                : (b);
}

// 定义针对 npy_cfloat 类型的最小值比较函数
npy_cfloat
_NPY_MIN(npy_cfloat a, npy_cfloat b, npy::complex_tag const &)
{
    return npy_isnan(npy_crealf(a)) || npy_isnan(npy_cimagf(a)) || PyArray_CLT(a, b, f)
                ? (a)
                : (b);
}

// 定义针对 npy_clongdouble 类型的最小值比较函数
npy_clongdouble
_NPY_MIN(npy_clongdouble a, npy_clongdouble b, npy::complex_tag const &)
{
    return npy_isnan(npy_creall(a)) || npy_isnan(npy_cimagl(a)) || PyArray_CLT(a, b, l)
                ? (a)
                : (b);
}

// 定义针对 npy_cdouble 类型的最大值比较函数
npy_cdouble
_NPY_MAX(npy_cdouble a, npy_cdouble b, npy::complex_tag const &)
{
    return npy_isnan(npy_creal(a)) || npy_isnan(npy_cimag(a)) || PyArray_CGT(a, b,)
                ? (a)
                : (b);
}

// 定义针对 npy_cfloat 类型的最大值比较函数
npy_cfloat
_NPY_MAX(npy_cfloat a, npy_cfloat b, npy::complex_tag const &)
{
    return npy_isnan(npy_crealf(a)) || npy_isnan(npy_cimagf(a)) || PyArray_CGT(a, b, f)
                ? (a)
                : (b);
}

// 定义针对 npy_clongdouble 类型的最大值比较函数
npy_clongdouble
_NPY_MAX(npy_clongdouble a, npy_clongdouble b, npy::complex_tag const &)
{
    return npy_isnan(npy_creall(a)) || npy_isnan(npy_cimagl(a)) || PyArray_CGT(a, b, l)
                ? (a)
                : (b);
}

// 取消前面定义的宏函数 PyArray_CLT 和 PyArray_CGT
#undef PyArray_CLT
#undef PyArray_CGT

// 定义针对日期类型的最小值比较函数，但没有提供实现
template <class T>
T
_NPY_MIN(T a, T b, npy::date_tag const &)
{
    # 如果 a 等于 NPY_DATETIME_NAT，则返回 a，否则继续判断下一条件
    return (a) == NPY_DATETIME_NAT   ? (a)
           # 如果 b 等于 NPY_DATETIME_NAT，则返回 b，否则继续判断下一条件
           : (b) == NPY_DATETIME_NAT ? (b)
           # 如果 a 小于 b，则返回 a，否则返回 b
           : (a) < (b)               ? (a)
                                     # 如果上述所有条件都不满足，则返回 b
                                     : (b);
/* 
 * 最大值计算函数，返回两个值中的较大值，考虑特殊情况
 */
template <class T>
T
_NPY_MAX(T a, T b, npy::date_tag const &)
{
    return (a) == NPY_DATETIME_NAT   ? (a)
           : (b) == NPY_DATETIME_NAT ? (b)
           : (a) > (b)               ? (a)
                                     : (b);
}

/* 
 * 最小值计算函数，根据给定的标签调用 _NPY_MAX 函数，返回两个值中的较小值
 */
template <class Tag, class T = typename Tag::type>
T
_NPY_MIN(T const &a, T const &b)
{
    return _NPY_MIN(a, b, Tag{});
}

/* 
 * 最大值计算函数，根据给定的标签调用 _NPY_MAX 函数，返回两个值中的较大值
 */
template <class Tag, class T = typename Tag::type>
T
_NPY_MAX(T const &a, T const &b)
{
    return _NPY_MAX(a, b, Tag{});
}

/* 
 * 裁剪函数，根据给定的标签和最小/最大值对数据进行裁剪，确保数据在指定范围内
 */
template <class Tag, class T>
T
_NPY_CLIP(T x, T min, T max)
{
    return _NPY_MIN<Tag>(_NPY_MAX<Tag>((x), (min)), (max));
}

/* 
 * 针对非浮点数的裁剪函数实现，处理输入和输出数据的指针，确保数据在指定范围内
 */
template <class Tag, class T>
static inline void
_npy_clip_const_minmax_(
    char *ip, npy_intp is, char *op, npy_intp os, npy_intp n, T min_val, T max_val,
    std::false_type /* non-floating point */
)
{
    /* 连续内存，使用分支让编译器优化 */
    if (is == sizeof(T) && os == sizeof(T)) {
        for (npy_intp i = 0; i < n; i++, ip += sizeof(T), op += sizeof(T)) {
            *(T *)op = _NPY_CLIP<Tag>(*(T *)ip, min_val, max_val);
        }
    }
    else {
        for (npy_intp i = 0; i < n; i++, ip += is, op += os) {
            *(T *)op = _NPY_CLIP<Tag>(*(T *)ip, min_val, max_val);
        }
    }
}

/* 
 * 针对浮点数的裁剪函数实现，处理输入和输出数据的指针，确保数据在指定范围内，
 * 同时处理 NaN 值的情况
 */
template <class Tag, class T>
static inline void
_npy_clip_const_minmax_(
    char *ip, npy_intp is, char *op, npy_intp os, npy_intp n, T min_val, T max_val,
    std::true_type  /* floating point */
)
{
    if (!npy_isnan(min_val) && !npy_isnan(max_val)) {
        /*
         * min_val 和 max_val 都不是 NaN，因此下面的比较会正确处理 NaN 的传播
         */

        /* 连续内存，使用分支让编译器优化 */
        if (is == sizeof(T) && os == sizeof(T)) {
            for (npy_intp i = 0; i < n; i++, ip += sizeof(T), op += sizeof(T)) {
                T x = *(T *)ip;
                if (x < min_val) {
                    x = min_val;
                }
                if (x > max_val) {
                    x = max_val;
                }
                *(T *)op = x;
            }
        }
        else {
            for (npy_intp i = 0; i < n; i++, ip += is, op += os) {
                T x = *(T *)ip;
                if (x < min_val) {
                    x = min_val;
                }
                if (x > max_val) {
                    x = max_val;
                }
                *(T *)op = x;
            }
        }
    }
    else {
        /* min_val 和/或 max_val 是 NaN */
        T x = npy_isnan(min_val) ? min_val : max_val;
        for (npy_intp i = 0; i < n; i++, op += os) {
            *(T *)op = x;
        }
    }
}

/* 
 * 裁剪函数的入口点，调用裁剪函数处理输入参数和维度信息
 */
template <class Tag, class T = typename Tag::type>
static void
_npy_clip(char **args, npy_intp const *dimensions, npy_intp const *steps)
{
    npy_intp n = dimensions[0];
    // 检查步长数组中的第二个和第三个元素是否为零
    if (steps[1] == 0 && steps[2] == 0) {
        /* min and max are constant throughout the loop, the most common case */
        // 获取循环期间常量的最小和最大值
        T min_val = *(T *)args[1];  // 获取最小值
        T max_val = *(T *)args[2];  // 获取最大值

        // 调用针对常量最小和最大值的剪切函数
        _npy_clip_const_minmax_<Tag, T>(
            args[0], steps[0], args[3], steps[3], n, min_val, max_val,
            std::is_base_of<npy::floating_point_tag, Tag>{}
        );
    }
    else {
        // 设置输入和输出指针及其对应的步长
        char *ip1 = args[0], *ip2 = args[1], *ip3 = args[2], *op1 = args[3];
        npy_intp is1 = steps[0], is2 = steps[1],
                 is3 = steps[2], os1 = steps[3];
        // 迭代处理每个元素
        for (npy_intp i = 0; i < n;
             i++, ip1 += is1, ip2 += is2, ip3 += is3, op1 += os1)
        {
            // 调用通用剪切函数来计算并将结果存储到输出指针
            *(T *)op1 = _NPY_CLIP<Tag>(*(T *)ip1, *(T *)ip2, *(T *)ip3);
        }
    }
    // 清除浮点状态障碍
    npy_clear_floatstatus_barrier((char *)dimensions);
extern "C" {
// 定义一个 C 语言风格的外部函数接口块

NPY_NO_EXPORT void
BOOL_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
          void *NPY_UNUSED(func))
{
    // 调用 _npy_clip 函数处理布尔类型数据
    return _npy_clip<npy::bool_tag>(args, dimensions, steps);
}
// 定义一个不导出的函数，用于处理布尔类型数据

NPY_NO_EXPORT void
BYTE_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
          void *NPY_UNUSED(func))
{
    // 调用 _npy_clip 函数处理字节类型数据
    return _npy_clip<npy::byte_tag>(args, dimensions, steps);
}
// 定义一个不导出的函数，用于处理字节类型数据

NPY_NO_EXPORT void
UBYTE_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
           void *NPY_UNUSED(func))
{
    // 调用 _npy_clip 函数处理无符号字节类型数据
    return _npy_clip<npy::ubyte_tag>(args, dimensions, steps);
}
// 定义一个不导出的函数，用于处理无符号字节类型数据

NPY_NO_EXPORT void
SHORT_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
           void *NPY_UNUSED(func))
{
    // 调用 _npy_clip 函数处理短整型数据
    return _npy_clip<npy::short_tag>(args, dimensions, steps);
}
// 定义一个不导出的函数，用于处理短整型数据

NPY_NO_EXPORT void
USHORT_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
            void *NPY_UNUSED(func))
{
    // 调用 _npy_clip 函数处理无符号短整型数据
    return _npy_clip<npy::ushort_tag>(args, dimensions, steps);
}
// 定义一个不导出的函数，用于处理无符号短整型数据

NPY_NO_EXPORT void
INT_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
         void *NPY_UNUSED(func))
{
    // 调用 _npy_clip 函数处理整型数据
    return _npy_clip<npy::int_tag>(args, dimensions, steps);
}
// 定义一个不导出的函数，用于处理整型数据

NPY_NO_EXPORT void
UINT_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
          void *NPY_UNUSED(func))
{
    // 调用 _npy_clip 函数处理无符号整型数据
    return _npy_clip<npy::uint_tag>(args, dimensions, steps);
}
// 定义一个不导出的函数，用于处理无符号整型数据

NPY_NO_EXPORT void
LONG_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
          void *NPY_UNUSED(func))
{
    // 调用 _npy_clip 函数处理长整型数据
    return _npy_clip<npy::long_tag>(args, dimensions, steps);
}
// 定义一个不导出的函数，用于处理长整型数据

NPY_NO_EXPORT void
ULONG_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
           void *NPY_UNUSED(func))
{
    // 调用 _npy_clip 函数处理无符号长整型数据
    return _npy_clip<npy::ulong_tag>(args, dimensions, steps);
}
// 定义一个不导出的函数，用于处理无符号长整型数据

NPY_NO_EXPORT void
LONGLONG_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
              void *NPY_UNUSED(func))
{
    // 调用 _npy_clip 函数处理长长整型数据
    return _npy_clip<npy::longlong_tag>(args, dimensions, steps);
}
// 定义一个不导出的函数，用于处理长长整型数据

NPY_NO_EXPORT void
ULONGLONG_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
               void *NPY_UNUSED(func))
{
    // 调用 _npy_clip 函数处理无符号长长整型数据
    return _npy_clip<npy::ulonglong_tag>(args, dimensions, steps);
}
// 定义一个不导出的函数，用于处理无符号长长整型数据

NPY_NO_EXPORT void
HALF_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
          void *NPY_UNUSED(func))
{
    // 调用 _npy_clip 函数处理半精度浮点数据
    return _npy_clip<npy::half_tag>(args, dimensions, steps);
}
// 定义一个不导出的函数，用于处理半精度浮点数据

NPY_NO_EXPORT void
FLOAT_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
           void *NPY_UNUSED(func))
{
    // 调用 _npy_clip 函数处理单精度浮点数据
    return _npy_clip<npy::float_tag>(args, dimensions, steps);
}
// 定义一个不导出的函数，用于处理单精度浮点数据

NPY_NO_EXPORT void
DOUBLE_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
            void *NPY_UNUSED(func))
{
    // 调用 _npy_clip 函数处理双精度浮点数据
    return _npy_clip<npy::double_tag>(args, dimensions, steps);
}
// 定义一个不导出的函数，用于处理双精度浮点数据

NPY_NO_EXPORT void
LONGDOUBLE_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
                void *NPY_UNUSED(func))
{
    // 调用 _npy_clip 函数处理长双精度浮点数据
    return _npy_clip<npy::longdouble_tag>(args, dimensions, steps);
}
// 定义一个不导出的函数，用于处理长双精度浮点数据

NPY_NO_EXPORT void
# 定义 CFLOAT_clip 函数，处理复数浮点数类型（cfloat）的数组剪切操作
CFLOAT_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
            void *NPY_UNUSED(func))
{
    # 调用模板函数 _npy_clip，传入复数浮点数标签类型（cfloat_tag），处理剪切操作
    return _npy_clip<npy::cfloat_tag>(args, dimensions, steps);
}

# 定义 CDOUBLE_clip 函数，处理双精度复数浮点数类型（cdouble）的数组剪切操作
NPY_NO_EXPORT void
CDOUBLE_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
             void *NPY_UNUSED(func))
{
    # 调用模板函数 _npy_clip，传入双精度复数浮点数标签类型（cdouble_tag），处理剪切操作
    return _npy_clip<npy::cdouble_tag>(args, dimensions, steps);
}

# 定义 CLONGDOUBLE_clip 函数，处理长双精度复数浮点数类型（clongdouble）的数组剪切操作
NPY_NO_EXPORT void
CLONGDOUBLE_clip(char **args, npy_intp const *dimensions,
                 npy_intp const *steps, void *NPY_UNUSED(func))
{
    # 调用模板函数 _npy_clip，传入长双精度复数浮点数标签类型（clongdouble_tag），处理剪切操作
    return _npy_clip<npy::clongdouble_tag>(args, dimensions, steps);
}

# 定义 DATETIME_clip 函数，处理日期时间类型（datetime）的数组剪切操作
NPY_NO_EXPORT void
DATETIME_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
              void *NPY_UNUSED(func))
{
    # 调用模板函数 _npy_clip，传入日期时间标签类型（datetime_tag），处理剪切操作
    return _npy_clip<npy::datetime_tag>(args, dimensions, steps);
}

# 定义 TIMEDELTA_clip 函数，处理时间差类型（timedelta）的数组剪切操作
NPY_NO_EXPORT void
TIMEDELTA_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
               void *NPY_UNUSED(func))
{
    # 调用模板函数 _npy_clip，传入时间差标签类型（timedelta_tag），处理剪切操作
    return _npy_clip<npy::timedelta_tag>(args, dimensions, steps);
}
```