# `.\numpy\numpy\_core\src\umath\reduction.c`

```py
/*
 * This file implements generic methods for computing reductions on arrays.
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

#include "npy_config.h"
#include "numpy/arrayobject.h"


#include "array_assign.h"
#include "array_coercion.h"
#include "array_method.h"
#include "ctors.h"
#include "refcount.h"

#include "numpy/ufuncobject.h"
#include "lowlevel_strided_loops.h"
#include "reduction.h"
#include "extobj.h"  /* for _check_ufunc_fperr */


/*
 * Count the number of dimensions selected in 'axis_flags'
 */
static int
count_axes(int ndim, const npy_bool *axis_flags)
{
    int idim;
    int naxes = 0;

    for (idim = 0; idim < ndim; ++idim) {
        if (axis_flags[idim]) {
            naxes++;
        }
    }
    return naxes;
}

/*
 * This function initializes a result array for a reduction operation
 * which has no identity. This means it needs to copy the first element
 * it sees along the reduction axes to result.
 *
 * If a reduction has an identity, such as 0 or 1, the result should be
 * fully initialized to the identity, because this function raises an
 * exception when there are no elements to reduce (which is appropriate if,
 * and only if, the reduction operation has no identity).
 *
 * This means it copies the subarray indexed at zero along each reduction axis
 * into 'result'.
 *
 * result  : The array into which the result is computed. This must have
 *           the same number of dimensions as 'operand', but for each
 *           axis i where 'axis_flags[i]' is True, it has a single element.
 * operand : The array being reduced.
 * axis_flags : An array of boolean flags, one for each axis of 'operand'.
 *              When a flag is True, it indicates to reduce along that axis.
 * funcname : The name of the reduction operation, for the purpose of
 *            better quality error messages. For example, "numpy.max"
 *            would be a good name for NumPy's max function.
 *
 * Returns -1 if an error occurred, and otherwise the reduce arrays size,
 * which is the number of elements already initialized.
 */
static npy_intp
PyArray_CopyInitialReduceValues(
                    PyArrayObject *result, PyArrayObject *operand,
                    const npy_bool *axis_flags, const char *funcname,
                    int keepdims)
{
    npy_intp shape[NPY_MAXDIMS], strides[NPY_MAXDIMS];
    npy_intp *shape_orig = PyArray_SHAPE(operand);
    npy_intp *strides_orig = PyArray_STRIDES(operand);
    PyArrayObject *op_view = NULL;

    int ndim = PyArray_NDIM(operand);

    /*
     * Copy the subarray of the first element along each reduction axis.
     *
     * Adjust the shape to only look at the first element along
     * any of the reduction axes. If keepdims is False remove the axes
     * entirely.
     */
    # 初始化输出维度计数器为0
    int idim_out = 0;
    # 初始化总大小为1
    npy_intp size = 1;
    # 遍历输入数组的每一个维度
    for (int idim = 0; idim < ndim; idim++) {
        # 检查当前维度是否为归约轴
        if (axis_flags[idim]) {
            # 如果归约轴上的原始形状为0，则抛出异常并返回-1
            if (NPY_UNLIKELY(shape_orig[idim] == 0)) {
                PyErr_Format(PyExc_ValueError,
                        "zero-size array to reduction operation %s "
                        "which has no identity", funcname);
                return -1;
            }
            # 如果需要保持归约后的维度，则设置当前输出维度为1，并且步长为0
            if (keepdims) {
                shape[idim_out] = 1;
                strides[idim_out] = 0;
                idim_out++;
            }
        }
        else {
            # 如果当前维度不是归约轴，则计算总大小，并将当前形状和步长复制到输出数组中
            size *= shape_orig[idim];
            shape[idim_out] = shape_orig[idim];
            strides[idim_out] = strides_orig[idim];
            idim_out++;
        }
    }

    # 获取操作数的描述符
    PyArray_Descr *descr = PyArray_DESCR(operand);
    # 增加描述符的引用计数
    Py_INCREF(descr);
    # 根据描述符创建新的数组视图
    op_view = (PyArrayObject *)PyArray_NewFromDescr(
            &PyArray_Type, descr, idim_out, shape, strides,
            PyArray_DATA(operand), 0, NULL);
    # 如果创建数组视图失败，则返回-1
    if (op_view == NULL) {
        return -1;
    }

    /*
     * 将元素复制到结果数组中以便开始操作。
     */
    # 将操作视图中的元素复制到结果数组中
    int res = PyArray_CopyInto(result, op_view);
    # 减少操作视图的引用计数
    Py_DECREF(op_view);
    # 如果复制操作失败，则返回-1
    if (res < 0) {
        return -1;
    }

    /*
     * 如果没有归约轴，则已经完成。
     * 注意，如果只有一个归约轴，原则上可以在设置迭代器之前通过移除该轴来更有效地设置迭代（简化迭代，因为`skip_first_count`（返回的大小）可以设置为0）。
     */
    # 返回总的元素个数作为归约操作的结果大小
    return size;
/*
 * This function executes all the standard NumPy reduction function
 * boilerplate code, just calling the appropriate inner loop function where
 * necessary.
 *
 * context     : The ArrayMethod context (with ufunc, method, and descriptors).
 * operand     : The array to be reduced.
 * out         : NULL, or the array into which to place the result.
 * wheremask   : Reduction mask of valid values used for `where=`.
 * axis_flags  : Flags indicating the reduction axes of 'operand'.
 * keepdims    : If true, leaves the reduction dimensions in the result
 *               with size one.
 * subok       : If true, the result uses the subclass of operand, otherwise
 *               it is always a base class ndarray.
 * initial     : Initial value, if NULL the default is fetched from the
 *               ArrayMethod (typically as the default from the ufunc).
 * loop        : `reduce_loop` from `ufunc_object.c`.  TODO: Refactor
 * buffersize  : Buffer size for the iterator. For the default, pass in 0.
 * funcname    : The name of the reduction function, for error messages.
 * errormask   : forwarded from _get_bufsize_errmask
 *
 * TODO FIXME: if you squint, this is essentially an second independent
 * implementation of generalized ufuncs with signature (i)->(), plus a few
 * extra bells and whistles. (Indeed, as far as I can tell, it was originally
 * split out to support a fancy version of count_nonzero... which is not
 * actually a reduction function at all, it's just a (i)->() function!) So
 * probably these two implementation should be merged into one. (In fact it
 * would be quite nice to support axis= and keepdims etc. for arbitrary
 * generalized ufuncs!)
 */

NPY_NO_EXPORT PyArrayObject *
PyUFunc_ReduceWrapper(PyArrayMethod_Context *context,
        PyArrayObject *operand, PyArrayObject *out, PyArrayObject *wheremask,
        npy_bool *axis_flags, int keepdims,
        PyObject *initial, PyArray_ReduceLoopFunc *loop,
        npy_intp buffersize, const char *funcname, int errormask)
{
    // Ensure the loop function is not NULL
    assert(loop != NULL);

    // Initialize the result array object
    PyArrayObject *result = NULL;

    // Number of initial elements to skip in reduction
    npy_intp skip_first_count = 0;

    /* Iterator parameters */
    NpyIter *iter = NULL;           // Numpy iterator
    PyArrayObject *op[3];           // Array operands for the iterator
    PyArray_Descr *op_dtypes[3];    // Data types of the operands
    npy_uint32 it_flags, op_flags[3];   // Iterator and operand flags

    /* Loop auxdata (must be freed on error) */
    NpyAuxData *auxdata = NULL;     // Auxiliary data used during iteration

    /* Set up the iterator */
    op[0] = out;                    // Output array
    op[1] = operand;                // Input array
    op_dtypes[0] = context->descriptors[0];  // Data type of output
    op_dtypes[1] = context->descriptors[1];  // Data type of input

    /* Buffer to use when we need an initial value */
    char *initial_buf = NULL;       // Buffer for initial value storage

    /* More than one axis means multiple orders are possible */
    if (!(context->method->flags & NPY_METH_IS_REORDERABLE)
            && count_axes(PyArray_NDIM(operand), axis_flags) > 1) {
        // Error if the reduction operation is not reorderable and more than one axis is specified
        PyErr_Format(PyExc_ValueError,
                "reduction operation '%s' is not reorderable, "
                "so at most one axis may be specified",
                funcname);
        goto fail;  // Jump to fail label in case of error
    }
    // 定义迭代器的标志位，指定缓冲、外部循环、内增长、接受零尺寸、引用允许、延迟缓冲分配、重叠时复制
    it_flags = NPY_ITER_BUFFERED |
            NPY_ITER_EXTERNAL_LOOP |
            NPY_ITER_GROWINNER |
            NPY_ITER_ZEROSIZE_OK |
            NPY_ITER_REFS_OK |
            NPY_ITER_DELAY_BUFALLOC |
            NPY_ITER_COPY_IF_OVERLAP;
    
    // 如果方法不可重新排序，则设置不反转步幅标志位
    if (!(context->method->flags & NPY_METH_IS_REORDERABLE)) {
        it_flags |= NPY_ITER_DONT_NEGATE_STRIDES;
    }
    
    // 设置第一个操作数的标志位，读写、对齐、分配、无子类型
    op_flags[0] = NPY_ITER_READWRITE |
                  NPY_ITER_ALIGNED |
                  NPY_ITER_ALLOCATE |
                  NPY_ITER_NO_SUBTYPE;
    
    // 设置第二个操作数的标志位，只读、对齐、无广播
    op_flags[1] = NPY_ITER_READONLY |
                  NPY_ITER_ALIGNED |
                  NPY_ITER_NO_BROADCAST;

    // 如果存在 where 掩码
    if (wheremask != NULL) {
        // 设置第三个操作数为 where 掩码
        op[2] = wheremask;
        /* wheremask 被保证为 NPY_BOOL 类型，因此借用其引用 */
        op_dtypes[2] = PyArray_DESCR(wheremask);
        assert(op_dtypes[2]->type_num == NPY_BOOL);
        if (op_dtypes[2] == NULL) {
            goto fail;
        }
        // 设置第三个操作数的标志位为只读
        op_flags[2] = NPY_ITER_READONLY;
    }
    
    // 设置结果数组的轴映射，默认使用操作数和 where 掩码的默认轴
    int result_axes[NPY_MAXDIMS];
    int *op_axes[3] = {result_axes, NULL, NULL};

    // 当前轴索引
    int curr_axis = 0;
    // 遍历操作数的维度
    for (int i = 0; i < PyArray_NDIM(operand); i++) {
        // 如果轴标志存在
        if (axis_flags[i]) {
            // 如果保持维度
            if (keepdims) {
                result_axes[i] = NPY_ITER_REDUCTION_AXIS(curr_axis);
                curr_axis++;
            }
            else {
                result_axes[i] = NPY_ITER_REDUCTION_AXIS(-1);
            }
        }
        else {
            result_axes[i] = curr_axis;
            curr_axis++;
        }
    }
    
    // 如果输出数组存在
    if (out != NULL) {
        /* NpyIter 在这种常见情况下不会提供良好的错误消息。 */
        // 检查输出数组的维度是否匹配当前轴数量
        if (NPY_UNLIKELY(curr_axis != PyArray_NDIM(out))) {
            if (keepdims) {
                PyErr_Format(PyExc_ValueError,
                        "output parameter for reduction operation %s has the "
                        "wrong number of dimensions: Found %d but expected %d "
                        "(must match the operand's when keepdims=True)",
                        funcname, PyArray_NDIM(out), curr_axis);
            }
            else {
                PyErr_Format(PyExc_ValueError,
                        "output parameter for reduction operation %s has the "
                        "wrong number of dimensions: Found %d but expected %d",
                        funcname, PyArray_NDIM(out), curr_axis);
            }
            goto fail;
        }
    }
    
    // 使用 NpyIter_AdvancedNew 创建高级迭代器
    iter = NpyIter_AdvancedNew(wheremask == NULL ? 2 : 3, op, it_flags,
                               NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                               op_flags,
                               op_dtypes,
                               PyArray_NDIM(operand), op_axes, NULL, buffersize);
    if (iter == NULL) {
        goto fail;
    }

    // 检查迭代器是否为空迭代
    npy_bool empty_iteration = NpyIter_GetIterSize(iter) == 0;
    // 获取迭代器的第一个操作数数组作为结果
    result = NpyIter_GetOperandArray(iter)[0];
    /*
     * Get the initial value (if it exists).  If the iteration is empty
     * then we assume the reduction is also empty.  The reason is that when
     * the outer iteration is empty we just won't use the initial value
     * in any case.  (`np.sum(np.zeros((0, 3)), axis=0)` is a length 3
     * reduction but has an empty result.)
     */
    // 检查是否存在初始值。如果迭代为空，则假设归约也为空。
    // 当外部迭代为空时，无论如何都不会使用初始值。
    // (`np.sum(np.zeros((0, 3)), axis=0)` 是一个长度为 3 的归约，但结果为空。
    if ((initial == NULL && context->method->get_reduction_initial == NULL)
            || initial == Py_None) {
        /* There is no initial value, or initial value was explicitly unset */
        // 没有初始值，或者初始值被明确地取消设置
    }
    else {
        /* Not all functions will need initialization, but init always: */
        // 并非所有函数都需要初始化，但初始化始终需要：
        // 分配初始缓冲区
        initial_buf = PyMem_Calloc(1, op_dtypes[0]->elsize);
        if (initial_buf == NULL) {
            PyErr_NoMemory();  // 分配内存失败，抛出内存错误
            goto fail;
        }
        if (initial != NULL) {
            /* must use user provided initial value */
            // 必须使用用户提供的初始值
            if (PyArray_Pack(op_dtypes[0], initial_buf, initial) < 0) {
                goto fail;
            }
        }
        else {
            /*
             * Fetch initial from ArrayMethod, we pretend the reduction is
             * empty when the iteration is.  This may be wrong, but when it is,
             * we will not need the identity as the result is also empty.
             */
            // 从 ArrayMethod 获取初始值，当迭代为空时，我们假装归约也为空。
            // 这可能是错误的，但当这种情况发生时，由于结果也为空，我们不需要标识。
            int has_initial = context->method->get_reduction_initial(
                    context, empty_iteration, initial_buf);
            if (has_initial < 0) {
                goto fail;
            }
            if (!has_initial) {
                /* We have no initial value available, free buffer to indicate */
                // 没有可用的初始值，释放缓冲区以指示
                PyMem_FREE(initial_buf);
                initial_buf = NULL;
            }
        }
    }

    PyArrayMethod_StridedLoop *strided_loop;
    NPY_ARRAYMETHOD_FLAGS flags = 0;

    int needs_api = (flags & NPY_METH_REQUIRES_PYAPI) != 0;
    needs_api |= NpyIter_IterationNeedsAPI(iter);
    if (!(flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        /* Start with the floating-point exception flags cleared */
        // 从清除浮点异常标志开始
        npy_clear_floatstatus_barrier((char*)&iter);
    }

    /*
     * Initialize the result to the reduction unit if possible,
     * otherwise copy the initial values and get a view to the rest.
     */
    // 如果可能的话，将结果初始化为归约单元，否则复制初始值并获取其余部分的视图。
    if (initial_buf != NULL) {
        /* Loop provided an identity or default value, assign to result. */
        // 循环提供了标识或默认值，将其分配给结果。
        int ret = raw_array_assign_scalar(
                PyArray_NDIM(result), PyArray_DIMS(result),
                PyArray_DESCR(result),
                PyArray_BYTES(result), PyArray_STRIDES(result),
                op_dtypes[0], initial_buf);
        if (ret < 0) {
            goto fail;
        }
    }
    else {
        /* 只能在有初始值（来自标识或参数）的情况下使用 */
        if (wheremask != NULL) {
            PyErr_Format(PyExc_ValueError,
                    "reduction operation '%s' does not have an identity, "
                    "so to use a where mask one has to specify 'initial'",
                    funcname);
            goto fail;
        }

        /*
         * 对于一维数组，skip_first_count 可以优化为 0，但无初始值的约简操作并不常见。
         * （见 CopyInitialReduceValues 中的注释）
         */
        skip_first_count = PyArray_CopyInitialReduceValues(
                result, operand, axis_flags, funcname, keepdims);
        if (skip_first_count < 0) {
            goto fail;
        }
    }

    if (!NpyIter_Reset(iter, NULL)) {
        goto fail;
    }

    /*
     * 需要确保在获取固定步长之前重置迭代器。（在此之前缓冲区信息是未初始化的。）
     */
    npy_intp fixed_strides[3];
    NpyIter_GetInnerFixedStrideArray(iter, fixed_strides);
    if (wheremask != NULL) {
        if (PyArrayMethod_GetMaskedStridedLoop(context,
                1, fixed_strides, &strided_loop, &auxdata, &flags) < 0) {
            goto fail;
        }
    }
    else {
        if (context->method->get_strided_loop(context,
                1, 0, fixed_strides, &strided_loop, &auxdata, &flags) < 0) {
            goto fail;
        }
    }

    if (!empty_iteration) {
        NpyIter_IterNextFunc *iternext;
        char **dataptr;
        npy_intp *strideptr;
        npy_intp *countptr;

        iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            goto fail;
        }
        dataptr = NpyIter_GetDataPtrArray(iter);
        strideptr = NpyIter_GetInnerStrideArray(iter);
        countptr = NpyIter_GetInnerLoopSizePtr(iter);

        if (loop(context, strided_loop, auxdata,
                iter, dataptr, strideptr, countptr, iternext,
                needs_api, skip_first_count) < 0) {
            goto fail;
        }
    }

    if (!(flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        /* 注意：即使在错误情况下，我们也可以检查浮点错误 */
        if (_check_ufunc_fperr(errormask, "reduce") < 0) {
            goto fail;
        }
    }

    if (out != NULL) {
        result = out;
    }
    Py_INCREF(result);

    if (initial_buf != NULL && PyDataType_REFCHK(PyArray_DESCR(result))) {
        PyArray_ClearBuffer(PyArray_DESCR(result), initial_buf, 0, 1, 1);
    }
    PyMem_FREE(initial_buf);
    NPY_AUXDATA_FREE(auxdata);
    if (!NpyIter_Deallocate(iter)) {
        Py_DECREF(result);
        return NULL;
    }
    return result;
fail:
    // 检查 initial_buf 是否非空，并且 op_dtypes[0] 是可引用的 PyDataType
    if (initial_buf != NULL && PyDataType_REFCHK(op_dtypes[0])) {
        // 清理 op_dtypes[0] 的缓冲区
        PyArray_ClearBuffer(op_dtypes[0], initial_buf, 0, 1, 1);
    }
    // 释放 initial_buf 占用的内存
    PyMem_FREE(initial_buf);
    // 释放所有辅助数据
    NPY_AUXDATA_FREE(auxdata);
    // 如果 iter 非空，释放迭代器资源
    if (iter != NULL) {
        NpyIter_Deallocate(iter);
    }

    // 返回 NULL，表示函数执行失败
    return NULL;
}
```