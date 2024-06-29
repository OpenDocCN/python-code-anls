# `.\numpy\numpy\_core\src\common\array_assign.c`

```
/*
 * This file implements some helper functions for the array assignment
 * routines. The actual assignment routines are in array_assign_*.c
 *
 * Written by Mark Wiebe (mwwiebe@gmail.com)
 * Copyright (c) 2011 by Enthought, Inc.
 *
 * See LICENSE.txt for the license.
 */
// 定义宏以使用 NPY API 版本，不包含废弃的 API
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
// 定义宏 _MULTIARRAYMODULE
#define _MULTIARRAYMODULE

// 包含 Python 头文件
#include <Python.h>

// 包含 NumPy 的数组类型头文件
#include <numpy/ndarraytypes.h>

// 包含 NumPy 的配置文件
#include "npy_config.h"

// 包含自定义的头文件
#include "shape.h"
#include "array_assign.h"
#include "common.h"
#include "lowlevel_strided_loops.h"
#include "mem_overlap.h"

// 该函数在 array_assign.h 中有详细的参数文档
NPY_NO_EXPORT int
broadcast_strides(int ndim, npy_intp const *shape,
                int strides_ndim, npy_intp const *strides_shape, npy_intp const *strides,
                char const *strides_name,
                npy_intp *out_strides)
{
    // 计算开始的维度索引
    int idim, idim_start = ndim - strides_ndim;

    /* Can't broadcast to fewer dimensions */
    // 如果 idim_start 小于 0，无法进行广播
    if (idim_start < 0) {
        goto broadcast_error;
    }

    /*
     * Process from the end to the start, so that 'strides' and 'out_strides'
     * can point to the same memory.
     */
    // 从后向前处理，以便 'strides' 和 'out_strides' 可以指向相同的内存
    for (idim = ndim - 1; idim >= idim_start; --idim) {
        npy_intp strides_shape_value = strides_shape[idim - idim_start];
        // 如果维度大小为 1，则输出的步长为 0
        if (strides_shape_value == 1) {
            out_strides[idim] = 0;
        }
        // 否则，维度大小必须与 shape 相同
        else if (strides_shape_value != shape[idim]) {
            goto broadcast_error;
        }
        else {
            out_strides[idim] = strides[idim - idim_start];
        }
    }

    /* New dimensions get a zero stride */
    // 新维度的步长设置为 0
    for (idim = 0; idim < idim_start; ++idim) {
        out_strides[idim] = 0;
    }

    return 0;

broadcast_error: {
        // 将 strides_shape 转换为字符串形式
        PyObject *shape1 = convert_shape_to_string(strides_ndim,
                                                   strides_shape, "");
        if (shape1 == NULL) {
            return -1;
        }

        // 将 shape 转换为字符串形式
        PyObject *shape2 = convert_shape_to_string(ndim, shape, "");
        if (shape2 == NULL) {
            Py_DECREF(shape1);
            return -1;
        }
        // 抛出值错误，指示无法广播的原因
        PyErr_Format(PyExc_ValueError,
                "could not broadcast %s from shape %S into shape %S",
                strides_name, shape1, shape2);
        Py_DECREF(shape1);
        Py_DECREF(shape2);
        return -1;
    }
}
    /*
     * 以下代码假设以下情况：
     *  * alignment 是 C 标准要求的2的幂次方。
     *  * 从指针到 uintp 的转换得到的是一个可以进行位操作的合理表示
     *    （这可能不是 C 标准要求的，但由 glibc 假定，因此应该是可以的）。
     *  * 将 stride 从 intp 转换为 uintp（以避免依赖于有符号 int 表示）保留与 alignment 的余数，
     *    因此 stride % a 与 ((unsigned intp) stride) % a 相同。这是 C 标准要求的。
     *
     *  代码检查 `data` 的最低 log2(alignment) 位和所有 `strides` 的最低 log2(alignment) 位是否为0，
     *  因为这意味着对所有整数 n，(data + n*stride) % alignment == 0。
     */
    if (alignment > 1) {
        npy_uintp align_check = (npy_uintp)data; // 将 data 转换为 uintp 类型，作为对齐检查的初始值
        int i;
    
        for (i = 0; i < ndim; i++) {
            /* 如果 shape[i] > 1，则需要考虑 strides[i] 是否为 0 */
            if (shape[i] > 1) {
                align_check |= (npy_uintp)strides[i]; // 将 strides[i] 转换为 uintp 类型，并将其加入对齐检查中
            }
            else if (shape[i] == 0) {
                /* 元素数为零的数组始终是对齐的 */
                return 1;
            }
        }
    
        return npy_is_aligned((void *)align_check, alignment); // 调用外部函数检查 align_check 是否对齐于 alignment
    }
    else if (alignment == 1) {
        return 1; // 如果 alignment 为 1，任何数据都是对齐的
    }
    else {
        /* 当 alignment == 0 时总是返回 false，表示无法对齐 */
        return 0;
    }
# 检查给定的数组对象是否是按要求对齐的
NPY_NO_EXPORT int
IsAligned(PyArrayObject *ap)
{
    return raw_array_is_aligned(PyArray_NDIM(ap), PyArray_DIMS(ap),
                                PyArray_DATA(ap), PyArray_STRIDES(ap),
                                PyArray_DESCR(ap)->alignment);
}

# 检查给定的数组对象是否是按指定的无符号整数对齐
NPY_NO_EXPORT int
IsUintAligned(PyArrayObject *ap)
{
    return raw_array_is_aligned(PyArray_NDIM(ap), PyArray_DIMS(ap),
                                PyArray_DATA(ap), PyArray_STRIDES(ap),
                                npy_uint_alignment(PyArray_ITEMSIZE(ap)));
}

# 返回数组对象是否具有重叠数据区域，1表示有重叠，0表示没有重叠
NPY_NO_EXPORT int
arrays_overlap(PyArrayObject *arr1, PyArrayObject *arr2)
{
    mem_overlap_t result;

    # 调用函数判断两个数组对象是否共享内存，以边界共享为准
    result = solve_may_share_memory(arr1, arr2, NPY_MAY_SHARE_BOUNDS);
    # 如果不共享内存，返回0；否则返回1
    if (result == MEM_OVERLAP_NO) {
        return 0;
    }
    else {
        return 1;
    }
}
```