# `.\numpy\numpy\_core\src\multiarray\array_assign_array.c`

```
/*
 * This file implements assignment from an ndarray to another ndarray.
 *
 * Written by Mark Wiebe (mwwiebe@gmail.com)
 * Copyright (c) 2011 by Enthought, Inc.
 *
 * See LICENSE.txt for the license.
 */

#define NPY_NO_DEPRECATED_API NPY_API_VERSION  // Define to avoid deprecated API usage
#define _MULTIARRAYMODULE  // Define for multiarray module
#define _UMATHMODULE       // Define for umath module

#define PY_SSIZE_T_CLEAN   // Ensures Python.h defines Py_ssize_t instead of other types
#include <Python.h>

#include "numpy/ndarraytypes.h"         // NumPy array data types
#include "numpy/npy_math.h"             // NumPy math functions
#include "npy_config.h"                 // NumPy configuration

#include "convert_datatype.h"           // Functions for converting data types
#include "methods.h"                    // NumPy array methods
#include "shape.h"                      // NumPy array shape handling
#include "lowlevel_strided_loops.h"     // Low-level strided loops for array operations

#include "array_assign.h"               // Functions for array assignment
#include "dtype_transfer.h"             // Functions for dtype transfer
#include "umathmodule.h"                // NumPy's umath module

/*
 * Check that array data is both uint-aligned and true-aligned for all array
 * elements, as required by the copy/casting code in lowlevel_strided_loops.c
 */
NPY_NO_EXPORT int
copycast_isaligned(int ndim, npy_intp const *shape,
        PyArray_Descr *dtype, char *data, npy_intp const *strides)
{
    int aligned;
    int big_aln, small_aln;

    int uint_aln = npy_uint_alignment(dtype->elsize); // Calculate uint alignment requirement
    int true_aln = dtype->alignment;                   // Get true alignment requirement

    /* uint alignment can be 0, meaning not uint alignable */
    if (uint_aln == 0) {
        return 0;  // Return false if uint alignment is zero
    }

    /*
     * As an optimization, it is unnecessary to check the alignment to the
     * smaller of (uint_aln, true_aln) if the data is aligned to the bigger of
     * the two and the big is a multiple of the small aln. We check the bigger
     * one first and only check the smaller if necessary.
     */
    if (true_aln >= uint_aln) {
        big_aln = true_aln;
        small_aln = uint_aln;
    }
    else {
        big_aln = uint_aln;
        small_aln = true_aln;
    }

    // Check alignment of array data based on the bigger alignment requirement
    aligned = raw_array_is_aligned(ndim, shape, data, strides, big_aln);
    if (aligned && big_aln % small_aln != 0) {
        aligned = raw_array_is_aligned(ndim, shape, data, strides, small_aln);
    }
    return aligned;  // Return whether data is aligned
}

/*
 * Assigns the array from 'src' to 'dst'. The strides must already have
 * been broadcast.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
raw_array_assign_array(int ndim, npy_intp const *shape,
        PyArray_Descr *dst_dtype, char *dst_data, npy_intp const *dst_strides,
        PyArray_Descr *src_dtype, char *src_data, npy_intp const *src_strides)
{
    int idim;
    npy_intp shape_it[NPY_MAXDIMS];
    npy_intp dst_strides_it[NPY_MAXDIMS];
    npy_intp src_strides_it[NPY_MAXDIMS];
    npy_intp coord[NPY_MAXDIMS];

    int aligned;

    NPY_BEGIN_THREADS_DEF;  // Macro to begin thread-safe operations

    // Check if both source and destination arrays are aligned
    aligned =
        copycast_isaligned(ndim, shape, dst_dtype, dst_data, dst_strides) &&
        copycast_isaligned(ndim, shape, src_dtype, src_data, src_strides);

    /* Use raw iteration with no heap allocation */
    // 调用 NumPy 函数准备两个原始数组的迭代器，以进行类型转换和数据拷贝
    if (PyArray_PrepareTwoRawArrayIter(
                    ndim, shape,
                    dst_data, dst_strides,
                    src_data, src_strides,
                    &ndim, shape_it,
                    &dst_data, dst_strides_it,
                    &src_data, src_strides_it) < 0) {
        // 如果准备迭代器失败，则返回错误码
        return -1;
    }

    /*
     * 检查在一维情况下的重叠情况。更高维度的数组和相反的步长会在此之前进行临时拷贝。
     */
    if (ndim == 1 && src_data < dst_data &&
                src_data + shape_it[0] * src_strides_it[0] > dst_data) {
        // 如果存在重叠，调整源数据和目标数据的位置和步长
        src_data += (shape_it[0] - 1) * src_strides_it[0];
        dst_data += (shape_it[0] - 1) * dst_strides_it[0];
        src_strides_it[0] = -src_strides_it[0];
        dst_strides_it[0] = -dst_strides_it[0];
    }

    /* 获取执行类型转换的函数 */
    NPY_cast_info cast_info;
    NPY_ARRAYMETHOD_FLAGS flags;
    if (PyArray_GetDTypeTransferFunction(aligned,
                        src_strides_it[0], dst_strides_it[0],
                        src_dtype, dst_dtype,
                        0,
                        &cast_info, &flags) != NPY_SUCCEED) {
        // 如果获取类型转换函数失败，则返回错误码
        return -1;
    }

    // 清除浮点错误状态标志，如果标志位中不包含 NPY_METH_NO_FLOATINGPOINT_ERRORS
    if (!(flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        npy_clear_floatstatus_barrier((char*)&src_data);
    }

    /* 确保元素数量超过线程化的阈值 */
    if (!(flags & NPY_METH_REQUIRES_PYAPI)) {
        // 计算需要处理的总元素个数
        npy_intp nitems = 1, i;
        for (i = 0; i < ndim; i++) {
            nitems *= shape_it[i];
        }
        // 在超过阈值时启用线程化处理
        NPY_BEGIN_THREADS_THRESHOLDED(nitems);
    }

    // 创建步长数组
    npy_intp strides[2] = {src_strides_it[0], dst_strides_it[0]};

    // 开始迭代处理数组数据
    NPY_RAW_ITER_START(idim, ndim, coord, shape_it) {
        /* 处理最内层的维度 */
        char *args[2] = {src_data, dst_data};
        // 调用类型转换函数处理当前迭代位置的数据
        if (cast_info.func(&cast_info.context,
                args, &shape_it[0], strides, cast_info.auxdata) < 0) {
            // 如果处理失败，则跳转到失败处理标签
            goto fail;
        }
    } NPY_RAW_ITER_TWO_NEXT(idim, ndim, coord, shape_it,
                            dst_data, dst_strides_it,
                            src_data, src_strides_it);

    // 结束多线程处理
    NPY_END_THREADS;
    // 释放类型转换信息结构体
    NPY_cast_info_xfree(&cast_info);

    // 如果标志位中不包含 NPY_METH_NO_FLOATINGPOINT_ERRORS
    if (!(flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        // 获取浮点错误状态，并在需要时处理浮点错误
        int fpes = npy_get_floatstatus_barrier((char*)&src_data);
        if (fpes && PyUFunc_GiveFloatingpointErrors("cast", fpes) < 0) {
            // 处理浮点错误失败时返回错误码
            return -1;
        }
    }

    // 处理成功，返回 0 表示没有错误
    return 0;
/*
 * 终止多线程执行，用于异常情况下的清理工作
 */
fail:
    NPY_END_THREADS;
    /*
     * 释放指向 cast_info 的内存
     */
    NPY_cast_info_xfree(&cast_info);
    /*
     * 返回 -1 表示函数执行失败
     */
    return -1;
}

/*
 * 将 'src' 数组根据 'wheremask' 数组中的 True 值分配给 'dst' 数组。
 * 要求 'dst' 和 'src' 的步长已经广播过。
 *
 * 成功时返回 0，失败时返回 -1。
 */
NPY_NO_EXPORT int
raw_array_wheremasked_assign_array(int ndim, npy_intp const *shape,
        PyArray_Descr *dst_dtype, char *dst_data, npy_intp const *dst_strides,
        PyArray_Descr *src_dtype, char *src_data, npy_intp const *src_strides,
        PyArray_Descr *wheremask_dtype, char *wheremask_data,
        npy_intp const *wheremask_strides)
{
    int idim;
    npy_intp shape_it[NPY_MAXDIMS];
    npy_intp dst_strides_it[NPY_MAXDIMS];
    npy_intp src_strides_it[NPY_MAXDIMS];
    npy_intp wheremask_strides_it[NPY_MAXDIMS];
    npy_intp coord[NPY_MAXDIMS];

    int aligned;

    /*
     * 定义多线程开始的宏，用于线程安全操作
     */
    NPY_BEGIN_THREADS_DEF;

    /*
     * 检查 'dst' 和 'src' 是否按字节对齐，并进行类型转换
     */
    aligned =
        copycast_isaligned(ndim, shape, dst_dtype, dst_data, dst_strides) &&
        copycast_isaligned(ndim, shape, src_dtype, src_data, src_strides);

    /*
     * 使用原始迭代，无堆内存分配
     */
    if (PyArray_PrepareThreeRawArrayIter(
                    ndim, shape,
                    dst_data, dst_strides,
                    src_data, src_strides,
                    wheremask_data, wheremask_strides,
                    &ndim, shape_it,
                    &dst_data, dst_strides_it,
                    &src_data, src_strides_it,
                    &wheremask_data, wheremask_strides_it) < 0) {
        return -1;
    }

    /*
     * 检查是否有重叠，对于一维情况进行处理，高维数组在此之前会进行临时复制
     */
    if (ndim == 1 && src_data < dst_data &&
                src_data + shape_it[0] * src_strides_it[0] > dst_data) {
        src_data += (shape_it[0] - 1) * src_strides_it[0];
        dst_data += (shape_it[0] - 1) * dst_strides_it[0];
        wheremask_data += (shape_it[0] - 1) * wheremask_strides_it[0];
        src_strides_it[0] = -src_strides_it[0];
        dst_strides_it[0] = -dst_strides_it[0];
        wheremask_strides_it[0] = -wheremask_strides_it[0];
    }

    /*
     * 获取执行类型转换的函数信息
     */
    NPY_cast_info cast_info;
    NPY_ARRAYMETHOD_FLAGS flags;
    if (PyArray_GetMaskedDTypeTransferFunction(aligned,
                        src_strides_it[0],
                        dst_strides_it[0],
                        wheremask_strides_it[0],
                        src_dtype, dst_dtype, wheremask_dtype,
                        0,
                        &cast_info, &flags) != NPY_SUCCEED) {
        return -1;
    }

    /*
     * 清除浮点状态异常，如果转换不需要 Python API
     */
    if (!(flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        npy_clear_floatstatus_barrier(src_data);
    }

    /*
     * 如果转换不需要 Python API，根据数据项数目启动线程
     */
    if (!(flags & NPY_METH_REQUIRES_PYAPI)) {
        npy_intp nitems = 1, i;
        for (i = 0; i < ndim; i++) {
            nitems *= shape_it[i];
        }
        NPY_BEGIN_THREADS_THRESHOLDED(nitems);
    }

    /*
     * 设置步长数组，用于转换函数
     */
    npy_intp strides[2] = {src_strides_it[0], dst_strides_it[0]};
    # 迭代处理数组的每个元素，使用 NPY_RAW_ITER_START 宏开始
    NPY_RAW_ITER_START(idim, ndim, coord, shape_it) {
        # 获取转换函数的指针并转换为合适的类型
        PyArray_MaskedStridedUnaryOp *stransfer;
        stransfer = (PyArray_MaskedStridedUnaryOp *)cast_info.func;

        /* 处理最内层的维度 */
        # 准备参数数组，用于传递源数据和目标数据
        char *args[2] = {src_data, dst_data};
        # 调用转换函数处理数据
        if (stransfer(&cast_info.context,
                args, &shape_it[0], strides,
                (npy_bool *)wheremask_data, wheremask_strides_it[0],
                cast_info.auxdata) < 0) {
            # 处理函数调用失败时跳转到错误处理代码块
            goto fail;
        }
    } NPY_RAW_ITER_THREE_NEXT(idim, ndim, coord, shape_it,
                            dst_data, dst_strides_it,
                            src_data, src_strides_it,
                            wheremask_data, wheremask_strides_it);

    # 结束多线程操作
    NPY_END_THREADS;

    # 释放类型转换信息的内存
    NPY_cast_info_xfree(&cast_info);

    # 如果未设置 NPY_METH_NO_FLOATINGPOINT_ERRORS 标志，则检查浮点错误
    if (!(flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        # 获取浮点错误状态
        int fpes = npy_get_floatstatus_barrier(src_data);
        # 如果存在浮点错误并且处理浮点错误函数调用失败，则返回错误
        if (fpes && PyUFunc_GiveFloatingpointErrors("cast", fpes) < 0) {
            return -1;
        }
    }

    # 执行成功返回 0
    return 0;
fail:
    // 终止多线程执行
    NPY_END_THREADS;
    // 释放分配的转换信息内存
    NPY_cast_info_xfree(&cast_info);
    // 返回错误码
    return -1;
}

/*
 * An array assignment function for copying arrays, broadcasting 'src' into
 * 'dst'. This function makes a temporary copy of 'src' if 'src' and
 * 'dst' overlap, to be able to handle views of the same data with
 * different strides.
 *
 * dst: The destination array.
 * src: The source array.
 * wheremask: If non-NULL, a boolean mask specifying where to copy.
 * casting: An exception is raised if the copy violates this
 *          casting rule.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
PyArray_AssignArray(PyArrayObject *dst, PyArrayObject *src,
                    PyArrayObject *wheremask,
                    NPY_CASTING casting)
{
    // 是否复制了源数组
    int copied_src = 0;

    // 源数组的步长
    npy_intp src_strides[NPY_MAXDIMS];

    /* Use array_assign_scalar if 'src' NDIM is 0 */
    // 如果源数组的维度为0，则使用标量赋值函数
    if (PyArray_NDIM(src) == 0) {
        return PyArray_AssignRawScalar(
                            dst, PyArray_DESCR(src), PyArray_DATA(src),
                            wheremask, casting);
    }

    /*
     * Performance fix for expressions like "a[1000:6000] += x".  In this
     * case, first an in-place add is done, followed by an assignment,
     * equivalently expressed like this:
     *
     *   tmp = a[1000:6000]   # Calls array_subscript in mapping.c
     *   np.add(tmp, x, tmp)
     *   a[1000:6000] = tmp   # Calls array_assign_subscript in mapping.c
     *
     * In the assignment the underlying data type, shape, strides, and
     * data pointers are identical, but src != dst because they are separately
     * generated slices.  By detecting this and skipping the redundant
     * copy of values to themselves, we potentially give a big speed boost.
     *
     * Note that we don't call EquivTypes, because usually the exact same
     * dtype object will appear, and we don't want to slow things down
     * with a complicated comparison.  The comparisons are ordered to
     * try and reject this with as little work as possible.
     */
    // 检测是否可以跳过冗余的数组复制操作以提升性能
    if (PyArray_DATA(src) == PyArray_DATA(dst) &&
                        PyArray_DESCR(src) == PyArray_DESCR(dst) &&
                        PyArray_NDIM(src) == PyArray_NDIM(dst) &&
                        PyArray_CompareLists(PyArray_DIMS(src),
                                             PyArray_DIMS(dst),
                                             PyArray_NDIM(src)) &&
                        PyArray_CompareLists(PyArray_STRIDES(src),
                                             PyArray_STRIDES(dst),
                                             PyArray_NDIM(src))) {
        /*printf("Redundant copy operation detected\n");*/
        // 如果数据类型、形状、步长和数据指针都相同，则跳过冗余复制操作
        return 0;
    }

    // 检查目标数组是否可写
    if (PyArray_FailUnlessWriteable(dst, "assignment destination") < 0) {
        // 如果不可写，则跳转到失败标签
        goto fail;
    }

    /* Check the casting rule */
    /*
     * 如果不能将源数组src的数据类型安全地转换为目标数组dst的数据类型，报错并跳转到fail标签处。
     */
    if (!PyArray_CanCastTypeTo(PyArray_DESCR(src),
                                PyArray_DESCR(dst), casting)) {
        npy_set_invalid_cast_error(
                PyArray_DESCR(src), PyArray_DESCR(dst), casting, NPY_FALSE);
        goto fail;
    }

    /*
     * 当目标数组dst的维度为1，并且步长指向同一个方向时，内部的最低级循环处理重叠数据的复制。
     * 对于更高维度和步长相反的一维数据，如果src和dst重叠，我们会为src创建一个临时副本。
     */
    if (((PyArray_NDIM(dst) == 1 && PyArray_NDIM(src) >= 1 &&
                    PyArray_STRIDES(dst)[0] *
                            PyArray_STRIDES(src)[PyArray_NDIM(src) - 1] < 0) ||
                    PyArray_NDIM(dst) > 1 || PyArray_HASFIELDS(dst)) &&
                    arrays_overlap(src, dst)) {
        PyArrayObject *tmp;

        /*
         * 分配一个临时的复制数组。
         */
        tmp = (PyArrayObject *)PyArray_NewLikeArray(dst,
                                        NPY_KEEPORDER, NULL, 0);
        if (tmp == NULL) {
            goto fail;
        }

        if (PyArray_AssignArray(tmp, src, NULL, NPY_UNSAFE_CASTING) < 0) {
            Py_DECREF(tmp);
            goto fail;
        }

        src = tmp;
        copied_src = 1;
    }

    /* 将src广播到dst以进行原始迭代 */
    if (PyArray_NDIM(src) > PyArray_NDIM(dst)) {
        int ndim_tmp = PyArray_NDIM(src);
        npy_intp *src_shape_tmp = PyArray_DIMS(src);
        npy_intp *src_strides_tmp = PyArray_STRIDES(src);
        /*
         * 作为向后兼容的特例，从src左侧去掉单位维度。
         */
        while (ndim_tmp > PyArray_NDIM(dst) && src_shape_tmp[0] == 1) {
            --ndim_tmp;
            ++src_shape_tmp;
            ++src_strides_tmp;
        }

        if (broadcast_strides(PyArray_NDIM(dst), PyArray_DIMS(dst),
                    ndim_tmp, src_shape_tmp,
                    src_strides_tmp, "input array",
                    src_strides) < 0) {
            goto fail;
        }
    }
    else {
        if (broadcast_strides(PyArray_NDIM(dst), PyArray_DIMS(dst),
                    PyArray_NDIM(src), PyArray_DIMS(src),
                    PyArray_STRIDES(src), "input array",
                    src_strides) < 0) {
            goto fail;
        }
    }

    /* 优化：标量布尔掩码 */
    if (wheremask != NULL &&
            PyArray_NDIM(wheremask) == 0 &&
            PyArray_DESCR(wheremask)->type_num == NPY_BOOL) {
        npy_bool value = *(npy_bool *)PyArray_DATA(wheremask);
        if (value) {
            /* 当where=True时，相当于不使用任何where条件 */
            wheremask = NULL;
        }
        else {
            /* 当where=False时，不复制任何数据 */
            return 0;
        }
    }
    # 如果 wheremask 是 NULL
    if (wheremask == NULL) {
        /* 直接赋值操作 */
        /* 使用原始数组迭代进行赋值 */
        # 调用函数进行数组赋值操作，使用了原始数组迭代的方式
        if (raw_array_assign_array(PyArray_NDIM(dst), PyArray_DIMS(dst),
                PyArray_DESCR(dst), PyArray_DATA(dst), PyArray_STRIDES(dst),
                PyArray_DESCR(src), PyArray_DATA(src), src_strides) < 0) {
            goto fail;
        }
    }
    else {
        npy_intp wheremask_strides[NPY_MAXDIMS];

        /* 将 wheremask 广播到 'dst' 以便进行原始迭代 */
        # 使用广播操作将 wheremask 广播到 'dst' 数组以便进行原始迭代
        if (broadcast_strides(PyArray_NDIM(dst), PyArray_DIMS(dst),
                    PyArray_NDIM(wheremask), PyArray_DIMS(wheremask),
                    PyArray_STRIDES(wheremask), "where mask",
                    wheremask_strides) < 0) {
            goto fail;
        }

        /* 带有 where-mask 的直接赋值操作 */
        /* 使用带有 where-mask 的原始数组迭代进行赋值 */
        # 调用函数进行带有 where-mask 的数组赋值操作，使用了原始数组迭代的方式
        if (raw_array_wheremasked_assign_array(
                PyArray_NDIM(dst), PyArray_DIMS(dst),
                PyArray_DESCR(dst), PyArray_DATA(dst), PyArray_STRIDES(dst),
                PyArray_DESCR(src), PyArray_DATA(src), src_strides,
                PyArray_DESCR(wheremask), PyArray_DATA(wheremask),
                        wheremask_strides) < 0) {
            goto fail;
        }
    }

    # 如果复制了 src，释放其引用计数
    if (copied_src) {
        Py_DECREF(src);
    }
    # 返回成功状态
    return 0;
fail:
    // 如果已经成功复制了源对象，则需要减少其引用计数
    if (copied_src) {
        Py_DECREF(src);
    }
    // 返回-1表示函数执行失败
    return -1;
}
```