# `.\numpy\numpy\_core\src\multiarray\array_assign_scalar.c`

```py
/*
 * This file implements assignment from a scalar to an ndarray.
 *
 * Written by Mark Wiebe (mwwiebe@gmail.com)
 * Copyright (c) 2011 by Enthought, Inc.
 *
 * See LICENSE.txt for the license.
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#define _UMATHMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <numpy/ndarraytypes.h>
#include "numpy/npy_math.h"

#include "npy_config.h"


#include "convert_datatype.h"
#include "methods.h"
#include "shape.h"
#include "lowlevel_strided_loops.h"

#include "array_assign.h"
#include "dtype_transfer.h"

#include "umathmodule.h"

/*
 * Assigns the scalar value to every element of the destination raw array.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
raw_array_assign_scalar(int ndim, npy_intp const *shape,
        PyArray_Descr *dst_dtype, char *dst_data, npy_intp const *dst_strides,
        PyArray_Descr *src_dtype, char *src_data)
{
    int idim;
    npy_intp shape_it[NPY_MAXDIMS], dst_strides_it[NPY_MAXDIMS];
    npy_intp coord[NPY_MAXDIMS];

    int aligned;

    NPY_BEGIN_THREADS_DEF;

    /* Check both uint and true alignment */
    aligned = raw_array_is_aligned(ndim, shape, dst_data, dst_strides,
                                   npy_uint_alignment(dst_dtype->elsize)) &&
              raw_array_is_aligned(ndim, shape, dst_data, dst_strides,
                                   dst_dtype->alignment) &&
              npy_is_aligned(src_data, npy_uint_alignment(src_dtype->elsize)) &&
              npy_is_aligned(src_data, src_dtype->alignment);

    /* Use raw iteration with no heap allocation */
    if (PyArray_PrepareOneRawArrayIter(
                    ndim, shape,
                    dst_data, dst_strides,
                    &ndim, shape_it,
                    &dst_data, dst_strides_it) < 0) {
        return -1;
    }

    /* Get the function to do the casting */
    NPY_cast_info cast_info;
    NPY_ARRAYMETHOD_FLAGS flags;
    if (PyArray_GetDTypeTransferFunction(aligned,
                        0, dst_strides_it[0],
                        src_dtype, dst_dtype,
                        0,
                        &cast_info, &flags) != NPY_SUCCEED) {
        return -1;
    }

    if (!(flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        npy_clear_floatstatus_barrier(src_data);
    }
    if (!(flags & NPY_METH_REQUIRES_PYAPI)) {
        npy_intp nitems = 1, i;
        for (i = 0; i < ndim; i++) {
            nitems *= shape_it[i];
        }
        NPY_BEGIN_THREADS_THRESHOLDED(nitems);
    }

    npy_intp strides[2] = {0, dst_strides_it[0]};

    NPY_RAW_ITER_START(idim, ndim, coord, shape_it) {
        /* Process the innermost dimension */
        char *args[2] = {src_data, dst_data};
        if (cast_info.func(&cast_info.context,
                args, &shape_it[0], strides, cast_info.auxdata) < 0) {
            goto fail;
        }
    } NPY_RAW_ITER_ONE_NEXT(idim, ndim, coord,
                            shape_it, dst_data, dst_strides_it);

// 调用宏 `NPY_RAW_ITER_ONE_NEXT`，用于迭代并处理数据数组中的元素，执行下一个迭代步骤。


    NPY_END_THREADS;

// 结束线程，用于在使用多线程执行过程中的清理工作。


    NPY_cast_info_xfree(&cast_info);

// 释放用于类型转换的信息结构体 `cast_info` 的内存。


    if (!(flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {

// 如果 `flags` 中未设置 `NPY_METH_NO_FLOATINGPOINT_ERRORS` 标志位，则执行以下操作，处理浮点数错误。


        int fpes = npy_get_floatstatus_barrier(src_data);

// 获取源数据 `src_data` 中的浮点数状态，并在此处设置一个障碍，确保获取时是稳定的状态。


        if (fpes && PyUFunc_GiveFloatingpointErrors("cast", fpes) < 0) {

// 如果浮点数状态 `fpes` 非零，并且调用 `PyUFunc_GiveFloatingpointErrors` 处理浮点数错误时返回小于零的错误码。


            return -1;
        }
    }

// 如果未发生浮点数错误或处理浮点数错误时成功，继续执行。


    return 0;

// 返回整数值 `0`，表示函数执行成功。
fail:
    NPY_END_THREADS;  # 结束线程锁定状态，确保线程安全性
    NPY_cast_info_xfree(&cast_info);  # 释放类型转换信息的内存
    return -1;  # 返回失败标志
}

/*
 * Assigns the scalar value to every element of the destination raw array
 * where the 'wheremask' value is True.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
raw_array_wheremasked_assign_scalar(int ndim, npy_intp const *shape,
        PyArray_Descr *dst_dtype, char *dst_data, npy_intp const *dst_strides,
        PyArray_Descr *src_dtype, char *src_data,
        PyArray_Descr *wheremask_dtype, char *wheremask_data,
        npy_intp const *wheremask_strides)
{
    int idim;
    npy_intp shape_it[NPY_MAXDIMS], dst_strides_it[NPY_MAXDIMS];
    npy_intp wheremask_strides_it[NPY_MAXDIMS];
    npy_intp coord[NPY_MAXDIMS];

    int aligned;

    NPY_BEGIN_THREADS_DEF;  # 定义线程开始

    /* Check both uint and true alignment */
    aligned = raw_array_is_aligned(ndim, shape, dst_data, dst_strides,
                                   npy_uint_alignment(dst_dtype->elsize)) &&
              raw_array_is_aligned(ndim, shape, dst_data, dst_strides,
                                   dst_dtype->alignment) &&
              npy_is_aligned(src_data, npy_uint_alignment(src_dtype->elsize) &&
              npy_is_aligned(src_data, src_dtype->alignment));

    /* Use raw iteration with no heap allocation */
    if (PyArray_PrepareTwoRawArrayIter(
                    ndim, shape,
                    dst_data, dst_strides,
                    wheremask_data, wheremask_strides,
                    &ndim, shape_it,
                    &dst_data, dst_strides_it,
                    &wheremask_data, wheremask_strides_it) < 0) {
        return -1;  # 准备迭代器失败，返回失败标志
    }

    /* Get the function to do the casting */
    NPY_cast_info cast_info;
    NPY_ARRAYMETHOD_FLAGS flags;
    if (PyArray_GetMaskedDTypeTransferFunction(aligned,
                        0, dst_strides_it[0], wheremask_strides_it[0],
                        src_dtype, dst_dtype, wheremask_dtype,
                        0,
                        &cast_info, &flags) != NPY_SUCCEED) {
        return -1;  # 获取类型转换函数失败，返回失败标志
    }

    if (!(flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        npy_clear_floatstatus_barrier(src_data);  # 清除浮点数状态标志
    }
    if (!(flags & NPY_METH_REQUIRES_PYAPI)) {
        npy_intp nitems = 1, i;
        for (i = 0; i < ndim; i++) {
            nitems *= shape_it[i];  # 计算数组元素总数
        }
        NPY_BEGIN_THREADS_THRESHOLDED(nitems);  # 根据元素总数设置线程阈值
    }

    npy_intp strides[2] = {0, dst_strides_it[0]};

    NPY_RAW_ITER_START(idim, ndim, coord, shape_it) {  # 进入原始迭代循环
        /* Process the innermost dimension */
        PyArray_MaskedStridedUnaryOp *stransfer;
        stransfer = (PyArray_MaskedStridedUnaryOp *)cast_info.func;  # 获取转换函数指针

        char *args[2] = {src_data, dst_data};
        if (stransfer(&cast_info.context,
                args, &shape_it[0], strides,
                (npy_bool *)wheremask_data, wheremask_strides_it[0],
                cast_info.auxdata) < 0) {
            goto fail;  # 转换失败，跳转到失败标签
        }

        /* Continue processing if successful */
    }

    NPY_END_THREADS;  # 结束线程锁定状态

    return 0;  # 返回成功标志
}
    } NPY_RAW_ITER_TWO_NEXT(idim, ndim, coord, shape_it,
                            dst_data, dst_strides_it,
                            wheremask_data, wheremask_strides_it);

这行代码调用了一个宏 `NPY_RAW_ITER_TWO_NEXT`，它在本语境中可能是一个用于迭代的宏，接受多个参数，包括迭代器状态、数据指针和步长等。


    NPY_END_THREADS;

该语句调用宏 `NPY_END_THREADS`，用于结束多线程操作。


    NPY_cast_info_xfree(&cast_info);

调用函数 `NPY_cast_info_xfree`，释放 `cast_info` 结构体或对象所占用的内存空间。


    if (!(flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {

条件判断语句，检查 `flags` 变量是否包含位掩码 `NPY_METH_NO_FLOATINGPOINT_ERRORS` 的反义。如果未包含该位掩码，则执行以下代码块。


        int fpes = npy_get_floatstatus_barrier(src_data);

声明并初始化整数变量 `fpes`，调用函数 `npy_get_floatstatus_barrier` 来获取 `src_data` 的浮点数状态。


        if (fpes && PyUFunc_GiveFloatingpointErrors("cast", fpes) < 0) {
            return -1;
        }

条件判断语句，如果 `fpes` 非零且调用 `PyUFunc_GiveFloatingpointErrors` 函数返回负值，返回 -1。


    }

条件判断语句结束。


    return 0;

函数返回整数值 0，表示成功执行。
/*
 * NPY_END_THREADS;
 * Ends any threading operations in NumPy's internal routines.
 * This macro is typically used to ensure thread safety.
 */
fail:
    NPY_END_THREADS;

    /*
     * NPY_cast_info_xfree(&cast_info);
     * Frees the memory associated with the cast information structure.
     * The cast_info structure holds details about type casting operations.
     */
    NPY_cast_info_xfree(&cast_info);

    /*
     * return -1;
     * Indicates failure by returning -1 to the caller of this function.
     */
    return -1;
}

/*
 * Assigns a scalar value specified by 'src_dtype' and 'src_data'
 * to elements of 'dst'.
 *
 * dst: The destination array.
 * src_dtype: The data type of the source scalar.
 * src_data: The memory element of the source scalar.
 * wheremask: If non-NULL, a boolean mask specifying where to copy.
 * casting: An exception is raised if the assignment violates this
 *          casting rule.
 *
 * This function is implemented in array_assign_scalar.c.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
PyArray_AssignRawScalar(PyArrayObject *dst,
                        PyArray_Descr *src_dtype, char *src_data,
                        PyArrayObject *wheremask,
                        NPY_CASTING casting)
{
    /*
     * int allocated_src_data = 0;
     * Flag indicating if memory was allocated for src_data.
     */
    int allocated_src_data = 0;

    /*
     * npy_longlong scalarbuffer[4];
     * Buffer used to store the scalar value when a temporary buffer is needed.
     * It's sized to accommodate up to 4 npy_longlong values.
     */
    npy_longlong scalarbuffer[4];

    /*
     * if (PyArray_FailUnlessWriteable(dst, "assignment destination") < 0) {
     * Checks if the destination array 'dst' is writable; raises an error if not.
     */
    if (PyArray_FailUnlessWriteable(dst, "assignment destination") < 0) {
        return -1;
    }

    /*
     * Check the casting rule:
     * if (!can_cast_scalar_to(src_dtype, src_data,
     *                         PyArray_DESCR(dst), casting)) {
     * Validates if casting of 'src_data' to 'dst' is permissible based on 'casting'.
     */
    if (!can_cast_scalar_to(src_dtype, src_data,
                            PyArray_DESCR(dst), casting)) {
        npy_set_invalid_cast_error(
                src_dtype, PyArray_DESCR(dst), casting, NPY_TRUE);
        return -1;
    }

    /*
     * Make a copy of the src data if it's a different dtype than 'dst'
     * or isn't aligned, and the destination we're copying to has
     * more than one element. To avoid having to manage object lifetimes,
     * we also skip this if 'dst' has an object dtype.
     */
    if ((!PyArray_EquivTypes(PyArray_DESCR(dst), src_dtype) ||
            !(npy_is_aligned(src_data, npy_uint_alignment(src_dtype->elsize)) &&
              npy_is_aligned(src_data, src_dtype->alignment))) &&
                    PyArray_SIZE(dst) > 1 &&
                    !PyDataType_REFCHK(PyArray_DESCR(dst))) {
        char *tmp_src_data;

        /*
         * Use a static buffer to store the aligned/cast version,
         * or allocate some memory if more space is needed.
         */
        if ((int)sizeof(scalarbuffer) >= PyArray_ITEMSIZE(dst)) {
            tmp_src_data = (char *)&scalarbuffer[0];
        }
        else {
            tmp_src_data = PyArray_malloc(PyArray_ITEMSIZE(dst));
            if (tmp_src_data == NULL) {
                PyErr_NoMemory();
                goto fail;
            }
            allocated_src_data = 1;
        }

        /*
         * if (PyDataType_FLAGCHK(PyArray_DESCR(dst), NPY_NEEDS_INIT)) {
         *     memset(tmp_src_data, 0, PyArray_ITEMSIZE(dst));
         * }
         * Initializes 'tmp_src_data' with zeros if 'dst' requires initialization.
         */
        if (PyDataType_FLAGCHK(PyArray_DESCR(dst), NPY_NEEDS_INIT)) {
            memset(tmp_src_data, 0, PyArray_ITEMSIZE(dst));
        }

        /*
         * if (PyArray_CastRawArrays(1, src_data, tmp_src_data, 0, 0,
         *                    src_dtype, PyArray_DESCR(dst), 0) != NPY_SUCCEED) {
         *     src_data = tmp_src_data;
         *     goto fail;
         * }
         * Casts 'src_data' to match the type of 'dst'; uses 'tmp_src_data' if necessary.
         */
        if (PyArray_CastRawArrays(1, src_data, tmp_src_data, 0, 0,
                            src_dtype, PyArray_DESCR(dst), 0) != NPY_SUCCEED) {
            src_data = tmp_src_data;
            goto fail;
        }

        /*
         * src_data = tmp_src_data;
         * src_dtype = PyArray_DESCR(dst);
         * Reassigns 'src_data' and 'src_dtype' to reflect the new data and type.
         */
        src_data = tmp_src_data;
        src_dtype = PyArray_DESCR(dst);
    }
    # 如果wheremask为NULL，则进行数值赋值
    if (wheremask == NULL) {
        # 使用原始数组迭代进行赋值
        if (raw_array_assign_scalar(PyArray_NDIM(dst), PyArray_DIMS(dst),
                PyArray_DESCR(dst), PyArray_DATA(dst), PyArray_STRIDES(dst),
                src_dtype, src_data) < 0) {
            goto fail;
        }
    }
    # 如果wheremask不为NULL
    else {
        # 创建用于广播的wheremask_strides数组
        npy_intp wheremask_strides[NPY_MAXDIMS];

        # 将wheremask广播到'dst'进行原始迭代
        if (broadcast_strides(PyArray_NDIM(dst), PyArray_DIMS(dst),
                    PyArray_NDIM(wheremask), PyArray_DIMS(wheremask),
                    PyArray_STRIDES(wheremask), "where mask",
                    wheremask_strides) < 0) {
            goto fail;
        }

        # 使用原始数组迭代进行带掩码的赋值
        if (raw_array_wheremasked_assign_scalar(
                PyArray_NDIM(dst), PyArray_DIMS(dst),
                PyArray_DESCR(dst), PyArray_DATA(dst), PyArray_STRIDES(dst),
                src_dtype, src_data,
                PyArray_DESCR(wheremask), PyArray_DATA(wheremask),
                wheremask_strides) < 0) {
            goto fail;
        }
    }
    
    # 如果分配了src_data，则释放它
    if (allocated_src_data) {
        PyArray_free(src_data);
    }
    
    # 返回0表示成功
    return 0;
# 如果已经分配了源数据（allocated_src_data 为真），则释放源数据内存
if (allocated_src_data) {
    PyArray_free(src_data);
}
# 返回错误代码 -1，表示操作失败
return -1;
```