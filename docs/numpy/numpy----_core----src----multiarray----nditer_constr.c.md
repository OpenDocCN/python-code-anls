# `.\numpy\numpy\_core\src\multiarray\nditer_constr.c`

```
/*
 * This file implements the construction, copying, and destruction
 * aspects of NumPy's nditer.
 *
 * Copyright (c) 2010-2011 by Mark Wiebe (mwwiebe@gmail.com)
 * The University of British Columbia
 *
 * Copyright (c) 2011 Enthought, Inc
 *
 * See LICENSE.txt for the license.
 */

#define NPY_NO_DEPRECATED_API NPY_API_VERSION

/* Allow this .c file to include nditer_impl.h */
#define NPY_ITERATOR_IMPLEMENTATION_CODE

#include "nditer_impl.h"
#include "arrayobject.h"
#include "array_coercion.h"
#include "templ_common.h"
#include "array_assign.h"
#include "dtype_traversal.h"


/* Internal helper functions private to this file */

/* Check global flags and update iterator flags */
static int
npyiter_check_global_flags(npy_uint32 flags, npy_uint32* itflags);

/* Check operation axes validity */
static int
npyiter_check_op_axes(int nop, int oa_ndim, int **op_axes,
                        const npy_intp *itershape);

/* Calculate the number of dimensions for the iterator */
static int
npyiter_calculate_ndim(int nop, PyArrayObject **op_in,
                       int oa_ndim);

/* Check per-operation flags */
static int
npyiter_check_per_op_flags(npy_uint32 flags, npyiter_opitflags *op_itflags);

/* Prepare one operand for iteration */
static int
npyiter_prepare_one_operand(PyArrayObject **op,
                        char **op_dataptr,
                        PyArray_Descr *op_request_dtype,
                        PyArray_Descr** op_dtype,
                        npy_uint32 flags,
                        npy_uint32 op_flags, npyiter_opitflags *op_itflags);

/* Prepare all operands for iteration */
static int
npyiter_prepare_operands(int nop,
                    PyArrayObject **op_in,
                    PyArrayObject **op,
                    char **op_dataptr,
                    PyArray_Descr **op_request_dtypes,
                    PyArray_Descr **op_dtype,
                    npy_uint32 flags,
                    npy_uint32 *op_flags, npyiter_opitflags *op_itflags,
                    npy_int8 *out_maskop);

/* Check casting compatibility */
static int
npyiter_check_casting(int nop, PyArrayObject **op,
                    PyArray_Descr **op_dtype,
                    NPY_CASTING casting,
                    npyiter_opitflags *op_itflags);

/* Fill axis data for the iterator */
static int
npyiter_fill_axisdata(NpyIter *iter, npy_uint32 flags, npyiter_opitflags *op_itflags,
                    char **op_dataptr,
                    const npy_uint32 *op_flags, int **op_axes,
                    npy_intp const *itershape);

/* Get the operation axis */
static inline int
npyiter_get_op_axis(int axis, npy_bool *reduction_axis);

/* Replace axis data in the iterator */
static void
npyiter_replace_axisdata(
        NpyIter *iter, int iop, PyArrayObject *op,
        int orig_op_ndim, const int *op_axes);

/* Compute index strides for the iterator */
static void
npyiter_compute_index_strides(NpyIter *iter, npy_uint32 flags);

/* Apply forced iteration order */
static void
npyiter_apply_forced_iteration_order(NpyIter *iter, NPY_ORDER order);

/* Flip negative strides */
static void
npyiter_flip_negative_strides(NpyIter *iter);

/* Reverse axis ordering */
static void
npyiter_reverse_axis_ordering(NpyIter *iter);

/* Find the best axis ordering */
static void
npyiter_find_best_axis_ordering(NpyIter *iter);

/* Return a pointer to PyArray_Descr */
static PyArray_Descr *
    /* 定义函数：从多个数组对象创建高级迭代器，支持广播、形状和缓冲区大小控制 */
NPY_NO_EXPORT NpyIter *
NpyIter_AdvancedNew(int nop, PyArrayObject **op_in, npy_uint32 flags,
                 NPY_ORDER order, NPY_CASTING casting,
                 npy_uint32 *op_flags,
                 PyArray_Descr **op_request_dtypes,
                 int oa_ndim, int **op_axes, npy_intp *itershape,
                 npy_intp buffersize)
{
    /* 迭代器的标志 */
    npy_uint32 itflags = NPY_ITFLAG_IDENTPERM;
    /* 迭代器的维度和操作数的维度 */
    int idim, ndim;
    /* 操作数索引 */
    int iop;

    /* 正在构建的迭代器 */
    NpyIter *iter;

    /* 每个操作数的值 */
    PyArrayObject **op;
    PyArray_Descr **op_dtype;
    npyiter_opitflags *op_itflags;
    char **op_dataptr;

    /* 排列 */
    npy_int8 *perm;
    /* 缓冲区数据 */
    NpyIter_BufferData *bufferdata = NULL;
    /* 是否有任何分配 */
    int any_allocate = 0, any_missing_dtypes = 0, need_subtype = 0;

    /* 自动分配输出的子类型 */
    double subtype_priority = NPY_PRIORITY;
    PyTypeObject *subtype = &PyArray_Type;

    /* 如果定义了构建时间跟踪 */
#if NPY_IT_CONSTRUCTION_TIMING
    /* 构建时间点 */
    npy_intp c_temp,
            c_start,
            c_check_op_axes,
            c_check_global_flags,
            c_calculate_ndim,
            c_malloc,
            c_prepare_operands,
            c_fill_axisdata,
            c_compute_index_strides,
            c_apply_forced_iteration_order,
            c_find_best_axis_ordering,
            c_get_priority_subtype,
            c_find_output_common_dtype,
            c_check_casting,
            c_allocate_arrays,
            c_coalesce_axes,
            c_prepare_buffers;
#endif

    /* 如果操作数超过了最大允许数量 */
    if (nop > NPY_MAXARGS) {
        /* 报错：不能构造超过最大操作数数量的迭代器 */
        PyErr_Format(PyExc_ValueError,
            "Cannot construct an iterator with more than %d operands "
            "(%d were requested)", NPY_MAXARGS, nop);
        /* 返回空指针 */
        return NULL;
    }
    /*
     * 在1.8版本之前，如果 `oa_ndim == 0`，这意味着 `op_axes != NULL` 是一个错误。
     * 在1.8版本中，`oa_ndim == -1` 承担了这个角色，而在这种情况下，op_axes 强制成一个0维的迭代器。
     * 因此，在1.13版本之后，使用 `oa_ndim == 0` 且 `op_axes == NULL` 是一个错误（已废弃）。
     */
    if ((oa_ndim == 0) && (op_axes == NULL)) {
        PyErr_Format(PyExc_ValueError,
            "Using `oa_ndim == 0` when `op_axes` is NULL. "
            "Use `oa_ndim == -1` or the MultiNew "
            "iterator for NumPy <1.8 compatibility");
        return NULL;
    }

    /* 检查 'oa_ndim' 和 'op_axes' 是否一起使用 */
    if (!npyiter_check_op_axes(nop, oa_ndim, op_axes, itershape)) {
        return NULL;
    }

    NPY_IT_TIME_POINT(c_check_op_axes);

    /* 检查全局迭代器标志 */
    if (!npyiter_check_global_flags(flags, &itflags)) {
        return NULL;
    }

    NPY_IT_TIME_POINT(c_check_global_flags);

    /* 计算迭代器应该有多少维度 */
    ndim = npyiter_calculate_ndim(nop, op_in, oa_ndim);

    NPY_IT_TIME_POINT(c_calculate_ndim);

    /* 为迭代器分配内存空间 */
    iter = (NpyIter*)
                PyObject_Malloc(NIT_SIZEOF_ITERATOR(itflags, ndim, nop));
    if (iter == NULL) {
        return NULL;
    }

    NPY_IT_TIME_POINT(c_malloc);

    /* 填充基本数据 */
    NIT_ITFLAGS(iter) = itflags;
    NIT_NDIM(iter) = ndim;
    NIT_NOP(iter) = nop;
    NIT_MASKOP(iter) = -1;
    NIT_ITERINDEX(iter) = 0;
    memset(NIT_BASEOFFSETS(iter), 0, (nop+1)*NPY_SIZEOF_INTP);

    op = NIT_OPERANDS(iter);
    op_dtype = NIT_DTYPES(iter);
    op_itflags = NIT_OPITFLAGS(iter);
    op_dataptr = NIT_RESETDATAPTR(iter);

    /* 准备所有操作数 */
    if (!npyiter_prepare_operands(nop, op_in, op, op_dataptr,
                        op_request_dtypes, op_dtype,
                        flags,
                        op_flags, op_itflags,
                        &NIT_MASKOP(iter))) {
        PyObject_Free(iter);
        return NULL;
    }
    /* 将 resetindex 设置为零（就在 resetdataptr 之后） */
    op_dataptr[nop] = 0;

    NPY_IT_TIME_POINT(c_prepare_operands);

    /*
     * 初始化缓冲区数据（在可能释放迭代器之前，必须将缓冲区和 transferdata 设置为 NULL）。
     */
    if (itflags & NPY_ITFLAG_BUFFER) {
        bufferdata = NIT_BUFFERDATA(iter);
        NBF_SIZE(bufferdata) = 0;
        memset(NBF_BUFFERS(bufferdata), 0, nop*NPY_SIZEOF_INTP);
        memset(NBF_PTRS(bufferdata), 0, nop*NPY_SIZEOF_INTP);
        /* 确保 transferdata/auxdata 被置为 NULL */
        memset(NBF_TRANSFERINFO(bufferdata), 0, nop * sizeof(NpyIter_TransferInfo));
    }

    /* 填充 AXISDATA 数组并设置 ITERSIZE 字段 */
    // 如果填充轴数据失败，则释放迭代器并返回空指针
    if (!npyiter_fill_axisdata(iter, flags, op_itflags, op_dataptr,
                                        op_flags, op_axes, itershape)) {
        NpyIter_Deallocate(iter);
        return NULL;
    }

    // 记录时间点：填充轴数据完成
    NPY_IT_TIME_POINT(c_fill_axisdata);

    // 如果启用了缓冲并且未指定缓冲区大小，则使用默认大小
    if (itflags & NPY_ITFLAG_BUFFER) {
        /*
         * If buffering is enabled and no buffersize was given, use a default
         * chosen to be big enough to get some amortization benefits, but
         * small enough to be cache-friendly.
         */
        if (buffersize <= 0) {
            buffersize = NPY_BUFSIZE;
        }
        /* No point in a buffer bigger than the iteration size */
        if (buffersize > NIT_ITERSIZE(iter)) {
            buffersize = NIT_ITERSIZE(iter);
        }
        NBF_BUFFERSIZE(bufferdata) = buffersize;

        /*
         * Initialize for use in FirstVisit, which may be called before
         * the buffers are filled and the reduce pos is updated.
         */
        NBF_REDUCE_POS(bufferdata) = 0;
    }

    /*
     * 如果请求了索引，则计算索引的步长。
     * 注意：在改变轴顺序之前必须执行此操作。
     */
    npyiter_compute_index_strides(iter, flags);

    // 记录时间点：计算索引步长完成
    NPY_IT_TIME_POINT(c_compute_index_strides);

    // 初始化轴置换为标识顺序
    perm = NIT_PERM(iter);
    for(idim = 0; idim < ndim; ++idim) {
        perm[idim] = (npy_int8)idim;
    }

    /*
     * 如果强制指定了迭代顺序，则应用它。
     */
    npyiter_apply_forced_iteration_order(iter, order);
    itflags = NIT_ITFLAGS(iter);

    // 记录时间点：应用强制迭代顺序完成
    NPY_IT_TIME_POINT(c_apply_forced_iteration_order);

    // 设置一些已分配输出的标志
    for (iop = 0; iop < nop; ++iop) {
        if (op[iop] == NULL) {
            /* Flag this so later we can avoid flipping axes */
            any_allocate = 1;
            /* If a subtype may be used, indicate so */
            if (!(op_flags[iop] & NPY_ITER_NO_SUBTYPE)) {
                need_subtype = 1;
            }
            /*
             * If the data type wasn't provided, will need to
             * calculate it.
             */
            if (op_dtype[iop] == NULL) {
                any_missing_dtypes = 1;
            }
        }
    }

    /*
     * 如果未强制指定顺序，则重新排序轴并翻转负步长以找到最佳顺序。
     */
    if (!(itflags & NPY_ITFLAG_FORCEDORDER)) {
        if (ndim > 1) {
            npyiter_find_best_axis_ordering(iter);
        }
        /*
         * If there's an output being allocated, we must not negate
         * any strides.
         */
        if (!any_allocate && !(flags & NPY_ITER_DONT_NEGATE_STRIDES)) {
            npyiter_flip_negative_strides(iter);
        }
        itflags = NIT_ITFLAGS(iter);
    }

    // 记录时间点：找到最佳轴顺序完成
    NPY_IT_TIME_POINT(c_find_best_axis_ordering);

    // 如果需要子类型，获取优先子类型
    if (need_subtype) {
        npyiter_get_priority_subtype(nop, op, op_itflags,
                                     &subtype_priority, &subtype);
    }
    NPY_IT_TIME_POINT(c_get_priority_subtype);
    # 记录当前时间点，用于性能分析和调试

    /*
     * If an automatically allocated output didn't have a specified
     * dtype, we need to figure it out now, before allocating the outputs.
     */
    # 如果自动分配的输出没有指定数据类型，需要在分配输出之前确定数据类型

    if (any_missing_dtypes || (flags & NPY_ITER_COMMON_DTYPE)) {
        # 如果存在缺失的数据类型或者设置了共同数据类型标志

        PyArray_Descr *dtype;
        # 声明一个 NumPy 数组描述符对象指针

        int only_inputs = !(flags & NPY_ITER_COMMON_DTYPE);
        # 只有输入参数没有共同数据类型标志

        op = NIT_OPERANDS(iter);
        # 获取迭代器的操作数

        op_dtype = NIT_DTYPES(iter);
        # 获取迭代器的数据类型数组

        dtype = npyiter_get_common_dtype(nop, op,
                                    op_itflags, op_dtype,
                                    op_request_dtypes,
                                    only_inputs);
        # 调用函数获取共同的数据类型

        if (dtype == NULL) {
            NpyIter_Deallocate(iter);
            return NULL;
        }
        # 如果未能获取到共同的数据类型，释放迭代器并返回空指针

        if (flags & NPY_ITER_COMMON_DTYPE) {
            NPY_IT_DBG_PRINT("Iterator: Replacing all data types\n");
            /* Replace all the data types */
            # 调试信息：替换所有数据类型

            for (iop = 0; iop < nop; ++iop) {
                if (op_dtype[iop] != dtype) {
                    Py_XDECREF(op_dtype[iop]);
                    Py_INCREF(dtype);
                    op_dtype[iop] = dtype;
                }
            }
            # 如果设置了共同数据类型标志，替换所有操作数的数据类型为共同数据类型
        }
        else {
            NPY_IT_DBG_PRINT("Iterator: Setting unset output data types\n");
            /* Replace the NULL data types */
            # 调试信息：设置未设置的输出数据类型

            for (iop = 0; iop < nop; ++iop) {
                if (op_dtype[iop] == NULL) {
                    Py_INCREF(dtype);
                    op_dtype[iop] = dtype;
                }
            }
            # 如果没有设置共同数据类型标志，设置操作数中未设置的数据类型为共同数据类型
        }

        Py_DECREF(dtype);
        # 减少共同数据类型的引用计数
    }

    NPY_IT_TIME_POINT(c_find_output_common_dtype);
    # 记录当前时间点，用于性能分析和调试

    /*
     * All of the data types have been settled, so it's time
     * to check that data type conversions are following the
     * casting rules.
     */
    # 所有数据类型都已确定，现在是检查数据类型转换是否遵循强制转换规则的时候

    if (!npyiter_check_casting(nop, op, op_dtype, casting, op_itflags)) {
        NpyIter_Deallocate(iter);
        return NULL;
    }
    # 如果数据类型转换不符合强制转换规则，释放迭代器并返回空指针

    NPY_IT_TIME_POINT(c_check_casting);
    # 记录当前时间点，用于性能分析和调试

    /*
     * At this point, the iteration order has been finalized. so
     * any allocation of ops that were NULL, or any temporary
     * copying due to casting/byte order/alignment can be
     * done now using a memory layout matching the iterator.
     */
    # 此时，迭代顺序已经最终确定。因此，可以使用与迭代器匹配的内存布局，现在执行任何空操作的分配或由于强制转换/字节顺序/对齐而产生的临时复制。

    if (!npyiter_allocate_arrays(iter, flags, op_dtype, subtype, op_flags,
                            op_itflags, op_axes)) {
        NpyIter_Deallocate(iter);
        return NULL;
    }
    # 如果无法分配数组，释放迭代器并返回空指针

    NPY_IT_TIME_POINT(c_allocate_arrays);
    # 记录当前时间点，用于性能分析和调试

    /*
     * Finally, if a multi-index wasn't requested,
     * it may be possible to coalesce some axes together.
     */
    # 最后，如果没有请求多索引，可能可以将一些轴合并在一起。
    /*
     * 如果数组维度大于1且没有多重索引标志，执行以下操作：
     * 将迭代器的轴合并成更少的维度。
     */
    if (ndim > 1 && !(itflags & NPY_ITFLAG_HASMULTIINDEX)) {
        npyiter_coalesce_axes(iter);
        /*
         * 操作可能改变了布局，因此需要重新获取内部指针。
         */
        itflags = NIT_ITFLAGS(iter);
        ndim = NIT_NDIM(iter);
        op = NIT_OPERANDS(iter);
        op_dtype = NIT_DTYPES(iter);
        op_itflags = NIT_OPITFLAGS(iter);
        op_dataptr = NIT_RESETDATAPTR(iter);
    }

    NPY_IT_TIME_POINT(c_coalesce_axes);

    /*
     * 现在轴已经完成，检查是否可以对 iternext 函数应用单迭代优化。
     */
    if (!(itflags & NPY_ITFLAG_BUFFER)) {
        NpyIter_AxisData *axisdata = NIT_AXISDATA(iter);
        if (itflags & NPY_ITFLAG_EXLOOP) {
            if (NIT_ITERSIZE(iter) == NAD_SHAPE(axisdata)) {
                NIT_ITFLAGS(iter) |= NPY_ITFLAG_ONEITERATION;
            }
        }
        else if (NIT_ITERSIZE(iter) == 1) {
            NIT_ITFLAGS(iter) |= NPY_ITFLAG_ONEITERATION;
        }
    }

    /*
     * 如果设置了 REFS_OK 标志，则检查是否存在任何引用数组，并进行标记。
     *
     * 注意：这实际上应该是不必要的，但很有可能有人依赖它。
     * 迭代器本身不需要此API，因为它仅对类型转换/缓冲需要。
     * 但在几乎所有用例中，API将被用于进行操作。
     */
    if (flags & NPY_ITER_REFS_OK) {
        for (iop = 0; iop < nop; ++iop) {
            PyArray_Descr *rdt = op_dtype[iop];
            if ((rdt->flags & (NPY_ITEM_REFCOUNT |
                                     NPY_ITEM_IS_POINTER |
                                     NPY_NEEDS_PYAPI)) != 0) {
                /* 迭代需要API访问 */
                NIT_ITFLAGS(iter) |= NPY_ITFLAG_NEEDSAPI;
            }
        }
    }

    /* 如果设置了缓冲标志且没有延迟分配 */
    if (itflags & NPY_ITFLAG_BUFFER) {
        if (!npyiter_allocate_transfer_functions(iter)) {
            NpyIter_Deallocate(iter);
            return NULL;
        }
        if (!(itflags & NPY_ITFLAG_DELAYBUF)) {
            /* 分配缓冲区 */
            if (!npyiter_allocate_buffers(iter, NULL)) {
                NpyIter_Deallocate(iter);
                return NULL;
            }

            /* 准备下一个缓冲区并设置 iterend/size */
            if (npyiter_copy_to_buffers(iter, NULL) < 0) {
                NpyIter_Deallocate(iter);
                return NULL;
            }
        }
    }

    NPY_IT_TIME_POINT(c_prepare_buffers);
#if NPY_IT_CONSTRUCTION_TIMING
    // 如果定义了 NPY_IT_CONSTRUCTION_TIMING 宏，则打印迭代器构造时的时间信息
    printf("\nIterator construction timing:\n");
    // 打印各个阶段的时间信息
    NPY_IT_PRINT_TIME_START(c_start);
    NPY_IT_PRINT_TIME_VAR(c_check_op_axes);
    NPY_IT_PRINT_TIME_VAR(c_check_global_flags);
    NPY_IT_PRINT_TIME_VAR(c_calculate_ndim);
    NPY_IT_PRINT_TIME_VAR(c_malloc);
    NPY_IT_PRINT_TIME_VAR(c_prepare_operands);
    NPY_IT_PRINT_TIME_VAR(c_fill_axisdata);
    NPY_IT_PRINT_TIME_VAR(c_compute_index_strides);
    NPY_IT_PRINT_TIME_VAR(c_apply_forced_iteration_order);
    NPY_IT_PRINT_TIME_VAR(c_find_best_axis_ordering);
    NPY_IT_PRINT_TIME_VAR(c_get_priority_subtype);
    NPY_IT_PRINT_TIME_VAR(c_find_output_common_dtype);
    NPY_IT_PRINT_TIME_VAR(c_check_casting);
    NPY_IT_PRINT_TIME_VAR(c_allocate_arrays);
    NPY_IT_PRINT_TIME_VAR(c_coalesce_axes);
    NPY_IT_PRINT_TIME_VAR(c_prepare_buffers);
    // 打印完毕，换行
    printf("\n");
#endif

    // 返回迭代器对象
    return iter;
}

/*NUMPY_API
 * Allocate a new iterator for more than one array object, using
 * standard NumPy broadcasting rules and the default buffer size.
 */
NPY_NO_EXPORT NpyIter *
NpyIter_MultiNew(int nop, PyArrayObject **op_in, npy_uint32 flags,
                 NPY_ORDER order, NPY_CASTING casting,
                 npy_uint32 *op_flags,
                 PyArray_Descr **op_request_dtypes)
{
    // 调用 NpyIter_AdvancedNew 函数创建多个数组对象的迭代器
    return NpyIter_AdvancedNew(nop, op_in, flags, order, casting,
                            op_flags, op_request_dtypes,
                            -1, NULL, NULL, 0);
}

/*NUMPY_API
 * Allocate a new iterator for one array object.
 */
NPY_NO_EXPORT NpyIter *
NpyIter_New(PyArrayObject *op, npy_uint32 flags,
                  NPY_ORDER order, NPY_CASTING casting,
                  PyArray_Descr* dtype)
{
    /* Split the flags into separate global and op flags */
    // 将 flags 分解为全局标志和操作标志
    npy_uint32 op_flags = flags & NPY_ITER_PER_OP_FLAGS;
    flags &= NPY_ITER_GLOBAL_FLAGS;

    // 调用 NpyIter_AdvancedNew 函数创建单个数组对象的迭代器
    return NpyIter_AdvancedNew(1, &op, flags, order, casting,
                            &op_flags, &dtype,
                            -1, NULL, NULL, 0);
}

/*NUMPY_API
 * Makes a copy of the iterator
 */
NPY_NO_EXPORT NpyIter *
NpyIter_Copy(NpyIter *iter)
{
    // 获取迭代器的标志位和维度信息
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    int ndim = NIT_NDIM(iter);
    int iop, nop = NIT_NOP(iter);
    int out_of_memory = 0;

    npy_intp size;
    NpyIter *newiter;
    PyArrayObject **objects;
    PyArray_Descr **dtypes;

    /* Allocate memory for the new iterator */
    // 为新迭代器分配内存空间
    size = NIT_SIZEOF_ITERATOR(itflags, ndim, nop);
    newiter = (NpyIter*)PyObject_Malloc(size);

    /* Copy the raw values to the new iterator */
    // 将原始值复制到新迭代器中
    memcpy(newiter, iter, size);

    /* Take ownership of references to the operands and dtypes */
    // 获取对操作数和数据类型的引用的所有权
    objects = NIT_OPERANDS(newiter);
    dtypes = NIT_DTYPES(newiter);
    for (iop = 0; iop < nop; ++iop) {
        Py_INCREF(objects[iop]);
        Py_INCREF(dtypes[iop]);
    }

    /* Allocate buffers and make copies of the transfer data if necessary */
    // 如果需要，分配缓冲区并复制传输数据
    # 如果迭代器标志中包含 NPY_ITFLAG_BUFFER
    if (itflags & NPY_ITFLAG_BUFFER) {
        # 获取缓冲区数据结构和相关信息
        NpyIter_BufferData *bufferdata;
        npy_intp buffersize, itemsize;
        char **buffers;

        bufferdata = NIT_BUFFERDATA(newiter);
        buffers = NBF_BUFFERS(bufferdata);
        buffersize = NBF_BUFFERSIZE(bufferdata);
        NpyIter_TransferInfo *transferinfo = NBF_TRANSFERINFO(bufferdata);

        # 遍历每个操作数
        for (iop = 0; iop < nop; ++iop) {
            # 如果当前缓冲区不为 NULL
            if (buffers[iop] != NULL) {
                # 如果发生内存不足错误
                if (out_of_memory) {
                    # 置当前缓冲区为 NULL，无需清理
                    buffers[iop] = NULL;
                }
                else {
                    # 分配当前缓冲区所需大小的内存
                    itemsize = dtypes[iop]->elsize;
                    buffers[iop] = PyArray_malloc(itemsize*buffersize);
                    # 如果内存分配失败，则设置内存不足错误标志
                    if (buffers[iop] == NULL) {
                        out_of_memory = 1;
                    }
                    else {
                        # 如果数据类型需要初始化，用零填充缓冲区
                        if (PyDataType_FLAGCHK(dtypes[iop], NPY_NEEDS_INIT)) {
                            memset(buffers[iop], '\0', itemsize*buffersize);
                        }
                    }
                }
            }

            # 如果读取函数不为 NULL
            if (transferinfo[iop].read.func != NULL) {
                # 如果发生内存不足错误
                if (out_of_memory) {
                    # 置读取函数为 NULL，无需清理
                    transferinfo[iop].read.func = NULL;  /* No cleanup */
                }
                else {
                    # 复制读取函数信息，处理内存不足情况
                    if (NPY_cast_info_copy(&transferinfo[iop].read,
                                           &transferinfo[iop].read) < 0) {
                        out_of_memory = 1;
                    }
                }
            }

            # 如果写入函数不为 NULL
            if (transferinfo[iop].write.func != NULL) {
                # 如果发生内存不足错误
                if (out_of_memory) {
                    # 置写入函数为 NULL，无需清理
                    transferinfo[iop].write.func = NULL;  /* No cleanup */
                }
                else {
                    # 复制写入函数信息，处理内存不足情况
                    if (NPY_cast_info_copy(&transferinfo[iop].write,
                                           &transferinfo[iop].write) < 0) {
                        out_of_memory = 1;
                    }
                }
            }

            # 如果清理函数不为 NULL
            if (transferinfo[iop].clear.func != NULL) {
                # 如果发生内存不足错误
                if (out_of_memory) {
                    # 置清理函数为 NULL，无需清理
                    transferinfo[iop].clear.func = NULL;  /* No cleanup */
                }
                else {
                    # 复制清理函数信息，处理内存不足情况
                    if (NPY_traverse_info_copy(&transferinfo[iop].clear,
                                               &transferinfo[iop].clear) < 0) {
                        out_of_memory = 1;
                    }
                }
            }
        }

        /* 初始化缓冲区到当前迭代索引 */
        # 如果没有内存不足错误且缓冲区大小大于 0
        if (!out_of_memory && NBF_SIZE(bufferdata) > 0) {
            # 跳转到当前迭代索引处
            npyiter_goto_iterindex(newiter, NIT_ITERINDEX(newiter));

            /* 准备下一个缓冲区并设置迭代结束标志和大小 */
            npyiter_copy_to_buffers(newiter, NULL);
        }
    }

    # 如果发生内存不足错误
    if (out_of_memory) {
        # 释放迭代器内存并设置无内存错误
        NpyIter_Deallocate(newiter);
        PyErr_NoMemory();
        return NULL;
    }

    # 返回新迭代器
    return newiter;
/*NUMPY_API
 * Deallocate an iterator.
 *
 * To correctly work when an error is in progress, we have to check
 * `PyErr_Occurred()`. This is necessary when buffers are not finalized
 * or WritebackIfCopy is used. We could avoid that check by exposing a new
 * function which is passed in whether or not a Python error is already set.
 */
NPY_NO_EXPORT int
NpyIter_Deallocate(NpyIter *iter)
{
    int success = PyErr_Occurred() == NULL;  // 检查当前是否有 Python 异常

    npy_uint32 itflags;  // 迭代器的标志位
    /*int ndim = NIT_NDIM(iter);*/  // 未使用，注释掉

    int iop, nop;  // 操作数和操作数数量
    PyArray_Descr **dtype;  // 数据类型的数组指针
    PyArrayObject **object;  // 数组对象的数组指针
    npyiter_opitflags *op_itflags;  // 操作标志位的数组指针

    if (iter == NULL) {  // 如果迭代器为空，则直接返回成功状态
        return success;
    }

    itflags = NIT_ITFLAGS(iter);  // 获取迭代器的标志位
    nop = NIT_NOP(iter);  // 获取迭代器的操作数数量
    dtype = NIT_DTYPES(iter);  // 获取迭代器的数据类型数组指针
    object = NIT_OPERANDS(iter);  // 获取迭代器的数组对象数组指针
    op_itflags = NIT_OPITFLAGS(iter);  // 获取迭代器的操作标志位数组指针

    /* Deallocate any buffers and buffering data */
    if (itflags & NPY_ITFLAG_BUFFER) {  // 如果迭代器标志位包含缓冲区标志
        /* Ensure no data is held by the buffers before they are cleared */
        if (success) {  // 如果成功状态为真
            if (npyiter_copy_from_buffers(iter) < 0) {  // 从缓冲区复制数据到数组对象失败
                success = NPY_FAIL;  // 设置成功状态为失败
            }
        }
        else {  // 如果成功状态为假
            npyiter_clear_buffers(iter);  // 清空迭代器的缓冲区数据
        }

        NpyIter_BufferData *bufferdata = NIT_BUFFERDATA(iter);  // 获取迭代器的缓冲区数据
        char **buffers;  // 缓冲区数组指针

        /* buffers */
        buffers = NBF_BUFFERS(bufferdata);  // 获取缓冲区数据的缓冲区数组指针
        for (iop = 0; iop < nop; ++iop, ++buffers) {  // 遍历每个操作数
            PyArray_free(*buffers);  // 释放每个缓冲区的内存
        }

        NpyIter_TransferInfo *transferinfo = NBF_TRANSFERINFO(bufferdata);  // 获取缓冲区数据的传输信息
        /* read bufferdata */
        for (iop = 0; iop < nop; ++iop, ++transferinfo) {  // 遍历每个操作数
            NPY_cast_info_xfree(&transferinfo->read);  // 释放读取传输信息的内存
            NPY_cast_info_xfree(&transferinfo->write);  // 释放写入传输信息的内存
            NPY_traverse_info_xfree(&transferinfo->clear);  // 释放清除传输信息的内存
        }
    }

    /*
     * Deallocate all the dtypes and objects that were iterated and resolve
     * any writeback buffers created by the iterator.
     */
    for (iop = 0; iop < nop; ++iop, ++dtype, ++object) {  // 遍历每个操作数
        if (op_itflags[iop] & NPY_OP_ITFLAG_HAS_WRITEBACK) {  // 如果操作标志包含写回标志
            if (success && PyArray_ResolveWritebackIfCopy(*object) < 0) {  // 如果成功状态为真且解析写回失败
                success = 0;  // 设置成功状态为失败
            }
            else {  // 否则
                PyArray_DiscardWritebackIfCopy(*object);  // 放弃写回拷贝
            }
        }
        Py_XDECREF(*dtype);  // 释放数据类型对象引用
        Py_XDECREF(*object);  // 释放数组对象引用
    }

    /* Deallocate the iterator memory */
    PyObject_Free(iter);  // 释放迭代器内存
    return success;  // 返回操作成功状态
}


/* Checks 'flags' for (C|F)_ORDER_INDEX, MULTI_INDEX, and EXTERNAL_LOOP,
 * setting the appropriate internal flags in 'itflags'.
 *
 * Returns 1 on success, 0 on error.
 */
static int
npyiter_check_global_flags(npy_uint32 flags, npy_uint32* itflags)
{
    if ((flags & NPY_ITER_PER_OP_FLAGS) != 0) {  // 如果传入的标志包含操作数标志
        PyErr_SetString(PyExc_ValueError,
                    "A per-operand flag was passed as a global flag "
                    "to the iterator constructor");  // 抛出值错误异常
        return 0;  // 返回错误状态
    }

    /* Check for an index */
    // 检查是否存在索引，设置相应的内部标志位到 `itflags` 中
    // 成功返回 1，错误返回 0
    // 没有代码，注释结束
}
    # 检查是否设置了 C_INDEX 或 F_INDEX 标志位
    if (flags & (NPY_ITER_C_INDEX | NPY_ITER_F_INDEX)) {
        # 如果同时设置了 C_INDEX 和 F_INDEX 标志位，则抛出数值错误异常
        if ((flags & (NPY_ITER_C_INDEX | NPY_ITER_F_INDEX)) ==
                    (NPY_ITER_C_INDEX | NPY_ITER_F_INDEX)) {
            PyErr_SetString(PyExc_ValueError,
                    "Iterator flags C_INDEX and "
                    "F_INDEX cannot both be specified");
            return 0;
        }
        # 标记迭代器具有索引
        (*itflags) |= NPY_ITFLAG_HASINDEX;
    }
    /* Check if a multi-index was requested */
    # 检查是否请求了多重索引
    if (flags & NPY_ITER_MULTI_INDEX) {
        /*
         * This flag primarily disables dimension manipulations that
         * would produce an incorrect multi-index.
         */
        # 此标志主要禁用可能产生不正确多重索引的维度操作
        (*itflags) |= NPY_ITFLAG_HASMULTIINDEX;
    }
    /* Check if the caller wants to handle inner iteration */
    # 检查调用者是否想处理内部迭代
    if (flags & NPY_ITER_EXTERNAL_LOOP) {
        # 如果迭代器已跟踪索引或多重索引，则不能使用 EXTERNAL_LOOP 标志位
        if ((*itflags) & (NPY_ITFLAG_HASINDEX | NPY_ITFLAG_HASMULTIINDEX)) {
            PyErr_SetString(PyExc_ValueError,
                    "Iterator flag EXTERNAL_LOOP cannot be used "
                    "if an index or multi-index is being tracked");
            return 0;
        }
        # 标记迭代器使用外部循环
        (*itflags) |= NPY_ITFLAG_EXLOOP;
    }
    /* Ranged */
    # 区间迭代标志
    if (flags & NPY_ITER_RANGED) {
        # 标记迭代器具有区间
        (*itflags) |= NPY_ITFLAG_RANGE;
        # 如果同时使用 RANGED 和 EXTERNAL_LOOP，但未使用 BUFFERED，则抛出数值错误异常
        if ((flags & NPY_ITER_EXTERNAL_LOOP) &&
                                    !(flags & NPY_ITER_BUFFERED)) {
            PyErr_SetString(PyExc_ValueError,
                    "Iterator flag RANGED cannot be used with "
                    "the flag EXTERNAL_LOOP unless "
                    "BUFFERED is also enabled");
            return 0;
        }
    }
    /* Buffering */
    # 缓冲标志
    if (flags & NPY_ITER_BUFFERED) {
        # 标记迭代器使用缓冲
        (*itflags) |= NPY_ITFLAG_BUFFER;
        # 如果同时使用 GROWINNER 标志，则标记迭代器增长内部迭代器
        if (flags & NPY_ITER_GROWINNER) {
            (*itflags) |= NPY_ITFLAG_GROWINNER;
        }
        # 如果使用 DELAY_BUFALLOC 标志，则标记迭代器延迟缓冲分配
        if (flags & NPY_ITER_DELAY_BUFALLOC) {
            (*itflags) |= NPY_ITFLAG_DELAYBUF;
        }
    }

    # 返回成功标志
    return 1;
static int
npyiter_calculate_ndim(int nop, PyArrayObject **op_in,
                       int oa_ndim)
{
    /* 如果使用了 'op_axes'，则强制使用 'oa_ndim' */
    if (oa_ndim >= 0 ) {
        // 如果 'oa_ndim' 大于等于零，则直接返回它作为迭代器的维度
        return oa_ndim;
    }
    /* 否则取操作数中的最大 'ndim' */
    else {
        // 初始化变量 ndim 为 0， iop 为循环计数器
        int ndim = 0, iop;

        // 循环遍历操作数数组 op_in
        for (iop = 0; iop < nop; ++iop) {
            // 检查当前操作数是否为非空
            if (op_in[iop] != NULL) {
                // 获取当前操作数的维度
                int ondim = PyArray_NDIM(op_in[iop]);
                // 如果当前操作数的维度大于 ndim，则更新 ndim
                if (ondim > ndim) {
                    ndim = ondim;
                }
            }
        }

        // 返回最大的维度 ndim
        return ndim;
    }
/*
 * 检查每个操作数的输入标志，并填充op_itflags。
 *
 * 在成功时返回1，在失败时返回0。
 */
static int
npyiter_check_per_op_flags(npy_uint32 op_flags, npyiter_opitflags *op_itflags)
{
    // 检查是否存在全局迭代器标志作为操作数标志传递给迭代器构造函数
    if ((op_flags & NPY_ITER_GLOBAL_FLAGS) != 0) {
        PyErr_SetString(PyExc_ValueError,
                    "A global iterator flag was passed as a per-operand flag "
                    "to the iterator constructor");
        return 0;
    }

    /* 检查读写标志 */
    if (op_flags & NPY_ITER_READONLY) {
        /* 读写标志是互斥的 */
        if (op_flags & (NPY_ITER_READWRITE|NPY_ITER_WRITEONLY)) {
            PyErr_SetString(PyExc_ValueError,
                    "Only one of the iterator flags READWRITE, "
                    "READONLY, and WRITEONLY may be "
                    "specified for an operand");
            return 0;
        }

        *op_itflags = NPY_OP_ITFLAG_READ;
    }
    else if (op_flags & NPY_ITER_READWRITE) {
        /* 读写标志是互斥的 */
        if (op_flags & NPY_ITER_WRITEONLY) {
            PyErr_SetString(PyExc_ValueError,
                    "Only one of the iterator flags READWRITE, "
                    "READONLY, and WRITEONLY may be "
                    "specified for an operand");
            return 0;
        }

        *op_itflags = NPY_OP_ITFLAG_READ|NPY_OP_ITFLAG_WRITE;
    }
    else if(op_flags & NPY_ITER_WRITEONLY) {
        *op_itflags = NPY_OP_ITFLAG_WRITE;
    }
    else {
        PyErr_SetString(PyExc_ValueError,
                "None of the iterator flags READWRITE, "
                "READONLY, or WRITEONLY were "
                "specified for an operand");
        return 0;
    }

    /* 检查临时拷贝标志 */
    if (((*op_itflags) & NPY_OP_ITFLAG_WRITE) &&
                (op_flags & (NPY_ITER_COPY |
                           NPY_ITER_UPDATEIFCOPY)) == NPY_ITER_COPY) {
        PyErr_SetString(PyExc_ValueError,
                "If an iterator operand is writeable, must use "
                "the flag UPDATEIFCOPY instead of "
                "COPY");
        return 0;
    }

    /* 检查写入掩码操作数的标志 */
    if (op_flags & NPY_ITER_WRITEMASKED) {
        if (!((*op_itflags) & NPY_OP_ITFLAG_WRITE)) {
            PyErr_SetString(PyExc_ValueError,
                "The iterator flag WRITEMASKED may only "
                "be used with READWRITE or WRITEONLY");
            return 0;
        }
        if ((op_flags & NPY_ITER_ARRAYMASK) != 0) {
            PyErr_SetString(PyExc_ValueError,
                "The iterator flag WRITEMASKED may not "
                "be used together with ARRAYMASK");
            return 0;
        }
        *op_itflags |= NPY_OP_ITFLAG_WRITEMASKED;
    }
    # 检查是否设置了 NPY_ITER_VIRTUAL 标志位
    if ((op_flags & NPY_ITER_VIRTUAL) != 0):
        # 如果设置了 NPY_ITER_VIRTUAL 标志位，再检查是否没有设置 NPY_ITER_READWRITE 标志位
        if ((op_flags & NPY_ITER_READWRITE) == 0):
            # 如果没有设置 NPY_ITER_READWRITE 标志位，则抛出值错误异常
            PyErr_SetString(PyExc_ValueError,
                "The iterator flag VIRTUAL should be "
                "be used together with READWRITE")
            # 返回 0 表示操作失败
            return 0
        # 如果两个标志位都设置了，将 NPY_OP_ITFLAG_VIRTUAL 添加到 op_itflags 中
        *op_itflags |= NPY_OP_ITFLAG_VIRTUAL
    
    # 返回 1 表示操作成功
    return 1
/*
 * Prepares a constructor operand.  Assumes a reference to 'op'
 * is owned, and that 'op' may be replaced.  Fills in 'op_dataptr',
 * 'op_dtype', and may modify 'op_itflags'.
 *
 * Returns 1 on success, 0 on failure.
 */
static int
npyiter_prepare_one_operand(PyArrayObject **op,
                        char **op_dataptr,
                        PyArray_Descr *op_request_dtype,
                        PyArray_Descr **op_dtype,
                        npy_uint32 flags,
                        npy_uint32 op_flags, npyiter_opitflags *op_itflags)
{
    /* NULL operands must be automatically allocated outputs */
    if (*op == NULL) {
        /* ALLOCATE or VIRTUAL should be enabled */
        if ((op_flags & (NPY_ITER_ALLOCATE|NPY_ITER_VIRTUAL)) == 0) {
            PyErr_SetString(PyExc_ValueError,
                    "Iterator operand was NULL, but neither the "
                    "ALLOCATE nor the VIRTUAL flag was specified");
            return 0;
        }

        if (op_flags & NPY_ITER_ALLOCATE) {
            /* Writing should be enabled */
            if (!((*op_itflags) & NPY_OP_ITFLAG_WRITE)) {
                PyErr_SetString(PyExc_ValueError,
                        "Automatic allocation was requested for an iterator "
                        "operand, but it wasn't flagged for writing");
                return 0;
            }
            /*
             * Reading should be disabled if buffering is enabled without
             * also enabling NPY_ITER_DELAY_BUFALLOC.  In all other cases,
             * the caller may initialize the allocated operand to a value
             * before beginning iteration.
             */
            if (((flags & (NPY_ITER_BUFFERED |
                            NPY_ITER_DELAY_BUFALLOC)) == NPY_ITER_BUFFERED) &&
                    ((*op_itflags) & NPY_OP_ITFLAG_READ)) {
                PyErr_SetString(PyExc_ValueError,
                        "Automatic allocation was requested for an iterator "
                        "operand, and it was flagged as readable, but "
                        "buffering  without delayed allocation was enabled");
                return 0;
            }

            /* If a requested dtype was provided, use it, otherwise NULL */
            Py_XINCREF(op_request_dtype);
            *op_dtype = op_request_dtype;
        }
        else {
            *op_dtype = NULL;
        }

        /* Specify bool if no dtype was requested for the mask */
        if (op_flags & NPY_ITER_ARRAYMASK) {
            if (*op_dtype == NULL) {
                *op_dtype = PyArray_DescrFromType(NPY_BOOL);
                if (*op_dtype == NULL) {
                    return 0;
                }
            }
        }

        *op_dataptr = NULL;

        return 1;
    }

    /* VIRTUAL operands must be NULL */
    # 检查 op_flags 中是否包含 NPY_ITER_VIRTUAL 标志位
    if (op_flags & NPY_ITER_VIRTUAL) {
        # 如果包含，则设置错误信息，指出使用了 VIRTUAL 标志但操作数不为 NULL
        PyErr_SetString(PyExc_ValueError,
                "Iterator operand flag VIRTUAL was specified, "
                "but the operand was not NULL");
        # 返回 0，表示出现错误
        return 0;
    }


    }
    else {
        # 如果 op_flags 中不包含 NPY_ITER_VIRTUAL 标志位，则设置错误信息
        PyErr_SetString(PyExc_ValueError,
                "Iterator inputs must be ndarrays");
        # 返回 0，表示出现错误
        return 0;
    }

    # 如果未进入上述错误情况，则返回 1，表示操作成功
    return 1;
/*
 * Process all the operands, copying new references so further processing
 * can replace the arrays if copying is necessary.
 */
static int
npyiter_prepare_operands(int nop, PyArrayObject **op_in,
                    PyArrayObject **op,
                    char **op_dataptr,
                    PyArray_Descr **op_request_dtypes,
                    PyArray_Descr **op_dtype,
                    npy_uint32 flags,
                    npy_uint32 *op_flags, npyiter_opitflags *op_itflags,
                    npy_int8 *out_maskop)
{
    int iop, i;
    npy_int8 maskop = -1;
    int any_writemasked_ops = 0;

    /*
     * Here we just prepare the provided operands.
     */
    for (iop = 0; iop < nop; ++iop) {
        // Copy the input operand into op array and increment its reference count
        op[iop] = op_in[iop];
        Py_XINCREF(op[iop]);
        op_dtype[iop] = NULL;

        /* Check the readonly/writeonly flags, and fill in op_itflags */
        // Verify per-operation flags and populate op_itflags accordingly
        if (!npyiter_check_per_op_flags(op_flags[iop], &op_itflags[iop])) {
            goto fail_iop; // Jump to failure handling for this operand
        }

        /* Extract the operand which is for masked iteration */
        // Identify the operand intended for masked iteration
        if ((op_flags[iop] & NPY_ITER_ARRAYMASK) != 0) {
            if (maskop != -1) {
                PyErr_SetString(PyExc_ValueError,
                        "Only one iterator operand may receive an "
                        "ARRAYMASK flag");
                goto fail_iop; // Multiple ARRAYMASK flags detected, handle error
            }

            maskop = iop; // Set maskop to the current operand index
            *out_maskop = iop; // Store the mask operand index in out_maskop
        }

        if (op_flags[iop] & NPY_ITER_WRITEMASKED) {
            any_writemasked_ops = 1; // Flag indicating at least one WRITEMASKED operand
        }

        /*
         * Prepare the operand.  This produces an op_dtype[iop] reference
         * on success.
         */
        // Prepare the current operand, determine its data pointer and dtype
        if (!npyiter_prepare_one_operand(&op[iop],
                        &op_dataptr[iop],
                        op_request_dtypes ? op_request_dtypes[iop] : NULL,
                        &op_dtype[iop],
                        flags,
                        op_flags[iop], &op_itflags[iop])) {
            goto fail_iop; // Jump to failure handling for this operand
        }
    }

    // Ensure consistency when WRITEMASKED is used without ARRAYMASK
    if (any_writemasked_ops && maskop < 0) {
        PyErr_SetString(PyExc_ValueError,
                "An iterator operand was flagged as WRITEMASKED, "
                "but no ARRAYMASK operand was given to supply "
                "the mask");
        goto fail_nop; // Jump to overall failure handling due to missing ARRAYMASK
    }
    else if (!any_writemasked_ops && maskop >= 0) {
        PyErr_SetString(PyExc_ValueError,
                "An iterator operand was flagged as the ARRAYMASK, "
                "but no WRITEMASKED operands were given to use "
                "the mask");
        goto fail_nop; // Jump to overall failure handling due to mismatched usage
    }

    return 1; // Success

  fail_nop:
    iop = nop - 1; // Adjust iop to the last valid index
  fail_iop:
    // Cleanup and release resources for failed operands
    for (i = 0; i < iop+1; ++i) {
        Py_XDECREF(op[i]); // Decrement reference count of operand
        Py_XDECREF(op_dtype[i]); // Decrement reference count of operand dtype
    }
    return 0; // Return failure
}
    # 根据传入的 `casting` 参数进行不同情况的匹配并返回对应的字符串表示
    switch (casting) {
        case NPY_NO_CASTING:
            return "'no'";  # 如果 `casting` 为 NPY_NO_CASTING，则返回字符串 "'no'"
        case NPY_EQUIV_CASTING:
            return "'equiv'";  # 如果 `casting` 为 NPY_EQUIV_CASTING，则返回字符串 "'equiv'"
        case NPY_SAFE_CASTING:
            return "'safe'";  # 如果 `casting` 为 NPY_SAFE_CASTING，则返回字符串 "'safe'"
        case NPY_SAME_KIND_CASTING:
            return "'same_kind'";  # 如果 `casting` 为 NPY_SAME_KIND_CASTING，则返回字符串 "'same_kind'"
        case NPY_UNSAFE_CASTING:
            return "'unsafe'";  # 如果 `casting` 为 NPY_UNSAFE_CASTING，则返回字符串 "'unsafe'"
        default:
            return "<unknown>";  # 如果 `casting` 不匹配以上任何一个值，则返回 "<unknown>"
    }
# 检查在给定操作数和数据类型描述符上的强制转换需求
static int
npyiter_check_casting(int nop, PyArrayObject **op,
                    PyArray_Descr **op_dtype,
                    NPY_CASTING casting,
                    npyiter_opitflags *op_itflags)
{
    int iop;

    # 遍历所有操作数
    for(iop = 0; iop < nop; ++iop) {
        # 调试输出：打印正在检查的操作数的强制转换情况
        NPY_IT_DBG_PRINT1("Iterator: Checking casting for operand %d\n",
                            (int)iop);
        
        # 如果开启了追踪调试，输出操作数和迭代器的数据类型描述符
#if NPY_IT_DBG_TRACING
        printf("op: ");
        if (op[iop] != NULL) {
            # 打印操作数的数据类型描述符
            PyObject_Print((PyObject *)PyArray_DESCR(op[iop]), stdout, 0);
        }
        else {
            printf("<null>");
        }
        printf(", iter: ");
        # 打印迭代器的数据类型描述符
        PyObject_Print((PyObject *)op_dtype[iop], stdout, 0);
        printf("\n");
#endif

        /* 如果操作数不为空且数据类型不等效，则需要进行强制转换 */
        if (op[iop] != NULL && !PyArray_EquivTypes(PyArray_DESCR(op[iop]),
                                                     op_dtype[iop])) {
            /* 检查读取（op -> temp）的强制转换 */
            if ((op_itflags[iop] & NPY_OP_ITFLAG_READ) &&
                        !PyArray_CanCastArrayTo(op[iop],
                                          op_dtype[iop],
                                          casting)) {
                PyErr_Format(PyExc_TypeError,
                        "Iterator operand %d dtype could not be cast from "
                        "%R to %R according to the rule %s",
                        iop, PyArray_DESCR(op[iop]), op_dtype[iop],
                        npyiter_casting_to_string(casting));
                return 0;
            }
            /* 检查写入（temp -> op）的强制转换 */
            if ((op_itflags[iop] & NPY_OP_ITFLAG_WRITE) &&
                        !PyArray_CanCastTypeTo(op_dtype[iop],
                                          PyArray_DESCR(op[iop]),
                                          casting)) {
                PyErr_Format(PyExc_TypeError,
                        "Iterator requested dtype could not be cast from "
                        "%R to %R, the operand %d dtype, "
                        "according to the rule %s",
                        op_dtype[iop], PyArray_DESCR(op[iop]), iop,
                        npyiter_casting_to_string(casting));
                return 0;
            }

            # 调试输出：因为类型不等效，设置 NPY_OP_ITFLAG_CAST
            NPY_IT_DBG_PRINT("Iterator: Setting NPY_OP_ITFLAG_CAST "
                                "because the types aren't equivalent\n");
            /* 表明此操作数需要强制转换 */
            op_itflags[iop] |= NPY_OP_ITFLAG_CAST;
        }
    }

    return 1;
}

/*
 * 检查掩码是否广播到 WRITEMASK REDUCE 操作数 'iop'，但 'iop' 没有广播到掩码。
 * 如果 'iop' 广播到掩码，则每个约简元素会有多个掩码值，这是无效的情况。
 *
 * 此检查应在填充所有操作数之后调用。
 *
 * 成功时返回 1，出错时返回 0。
 */
static int
check_mask_for_writemasked_reduction(NpyIter *iter, int iop)
{
    # 获取迭代器的标志位
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    # 获取迭代器的维度数量
    int idim, ndim = NIT_NDIM(iter);
    # 获取迭代器的运算操作数量
    int nop = NIT_NOP(iter);
    # 获取迭代器的掩码操作数量
    int maskop = NIT_MASKOP(iter);

    # 定义和初始化轴数据指针和大小
    NpyIter_AxisData *axisdata;
    npy_intp sizeof_axisdata;

    # 获取轴数据指针
    axisdata = NIT_AXISDATA(iter);
    # 获取轴数据大小
    sizeof_axisdata = NIT_AXISDATA_SIZEOF(itflags, ndim, nop);

    # 遍历每一个维度
    for(idim = 0; idim < ndim; ++idim) {
        npy_intp maskstride, istride;

        # 获取迭代器的步长
        istride = NAD_STRIDES(axisdata)[iop];
        # 获取迭代器的掩码步长
        maskstride = NAD_STRIDES(axisdata)[maskop];

        # 如果掩码步长不为0且迭代器步长为0，抛出异常并返回0
        if (maskstride != 0 && istride == 0) {
            PyErr_SetString(PyExc_ValueError,
                    "Iterator reduction operand is WRITEMASKED, "
                    "but also broadcasts to multiple mask values. "
                    "There can be only one mask value per WRITEMASKED "
                    "element.");
            return 0;
        }

        # 更新轴数据指针
        NIT_ADVANCE_AXISDATA(axisdata, 1);
    }

    # 返回1，表示没有异常
    return 1;
/*
 * 检查基于标志和读写操作数的约简是否有效。这个路径已经过时，
 * 因为通常只有特定的轴应该被约简。如果显式指定了轴，那么标志就是不必要的。
 */
static int
npyiter_check_reduce_ok_and_set_flags(
        NpyIter *iter, npy_uint32 flags, npyiter_opitflags *op_itflags,
        int iop, int maskop, int dim) {
    /* 如果可写，意味着进行约简操作 */
    if (op_itflags[iop] & NPY_OP_ITFLAG_WRITE) {
        // 如果标志中没有允许约简，则引发错误
        if (!(flags & NPY_ITER_REDUCE_OK)) {
            PyErr_Format(PyExc_ValueError,
                    "output operand requires a reduction along dimension %d, "
                    "but the reduction is not enabled. The dimension size of 1 "
                    "does not match the expected output shape.", dim);
            return 0;
        }
        // 如果操作标志没有读取权限，则引发错误
        if (!(op_itflags[iop] & NPY_OP_ITFLAG_READ)) {
            PyErr_SetString(PyExc_ValueError,
                    "output operand requires a reduction, but is flagged as "
                    "write-only, not read-write");
            return 0;
        }
        /*
         * 如果是掩码操作，不能进行约简，因为一旦掩码操作返回'True'，
         * 就可能会向数组写回一次，然后在后续的约简中，掩码操作返回'False'，
         * 表明不应该进行写回操作，这会违反严格的掩码语义。
         */
        if (iop == maskop) {
            PyErr_SetString(PyExc_ValueError,
                    "output operand requires a "
                    "reduction, but is flagged as "
                    "the ARRAYMASK operand which "
                    "is not permitted to be the "
                    "result of a reduction");
            return 0;
        }
        // 输出调试信息，指示正在进行约简操作
        NPY_IT_DBG_PRINT("Iterator: Indicating that a reduction is"
                         "occurring\n");

        // 设置迭代器标志表明正在进行约简操作
        NIT_ITFLAGS(iter) |= NPY_ITFLAG_REDUCE;
        // 设置操作标志表明正在进行约简操作
        op_itflags[iop] |= NPY_OP_ITFLAG_REDUCE;
    }
    return 1;
}

/**
 * 移除(NPY_ITER_REDUCTION_AXIS)的减少指示，并将is_forced_broadcast设置为1（如果设置）。否则设置为0。
 *
 * @param axis 要规范化的操作轴（op_axes[i]）。
 * @param reduction_axis 如果是减少轴则设置为1，否则设置为0。
 * @returns 规范化后的轴（去除减少轴标志）。
 */
static inline int
npyiter_get_op_axis(int axis, npy_bool *reduction_axis) {
    npy_bool forced_broadcast = axis >= NPY_ITER_REDUCTION_AXIS(-1);

    // 如果reduction_axis不为NULL，则根据forced_broadcast设置其值
    if (reduction_axis != NULL) {
        *reduction_axis = forced_broadcast;
    }
    // 如果是强制广播，则返回规范化后的轴（去除减少轴标志）
    if (forced_broadcast) {
        return axis - NPY_ITER_REDUCTION_AXIS(0);
    }
    // 否则直接返回原始轴值
    return axis;
}
/*
 * Fills in the AXISDATA for the 'nop' operands, broadcasting
 * the dimensions as necessary. Also fills
 * in the ITERSIZE data member.
 *
 * If op_axes is not NULL, it should point to an array of ndim-sized
 * arrays, one for each operand.
 *
 * Returns 1 on success, 0 on failure.
 */
static int
npyiter_fill_axisdata(NpyIter *iter, npy_uint32 flags, npyiter_opitflags *op_itflags,
                    char **op_dataptr,
                    const npy_uint32 *op_flags, int **op_axes,
                    npy_intp const *itershape)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);  // 获取迭代器的标志位
    int idim, ndim = NIT_NDIM(iter);  // 获取迭代器的维度信息
    int iop, nop = NIT_NOP(iter);  // 获取迭代器的操作数数量
    int maskop = NIT_MASKOP(iter);  // 获取迭代器的掩码操作数标志位

    int ondim;  // 未使用的变量
    NpyIter_AxisData *axisdata;  // 定义轴数据结构指针
    npy_intp sizeof_axisdata;  // 定义轴数据结构大小
    PyArrayObject **op = NIT_OPERANDS(iter), *op_cur;  // 获取迭代器的操作数数组

    npy_intp broadcast_shape[NPY_MAXDIMS];  // 定义广播形状数组，大小为最大维度数

    /* First broadcast the shapes together */
    if (itershape == NULL) {  // 如果外部没有指定形状
        for (idim = 0; idim < ndim; ++idim) {
            broadcast_shape[idim] = 1;  // 将广播形状初始化为1
        }
    }
    else {  // 如果外部指定了形状
        for (idim = 0; idim < ndim; ++idim) {
            broadcast_shape[idim] = itershape[idim];  // 使用外部指定的形状
            /* Negative shape entries are deduced from the operands */
            if (broadcast_shape[idim] < 0) {
                broadcast_shape[idim] = 1;  // 如果形状为负数，则设为1
            }
        }
    }
    for (iop = 0; iop < nop; ++iop) {
        # 获取当前操作数
        op_cur = op[iop];
        # 如果操作数不为空
        if (op_cur != NULL) {
            # 获取操作数的形状数组和维度数
            npy_intp *shape = PyArray_DIMS(op_cur);
            ondim = PyArray_NDIM(op_cur);

            # 如果没有指定操作轴或者当前操作的轴为NULL
            if (op_axes == NULL || op_axes[iop] == NULL) {
                /*
                 * 可能是因为正在使用操作轴，但 op_axes[iop] 为 NULL
                 */
                # 如果操作数的维度大于所允许的维度 ndim
                if (ondim > ndim) {
                    # 抛出维度超出异常
                    PyErr_SetString(PyExc_ValueError,
                            "input operand has more dimensions than allowed "
                            "by the axis remapping");
                    return 0;
                }
                # 遍历当前操作数的维度
                for (idim = 0; idim < ondim; ++idim) {
                    # 获取广播后的形状和当前操作数的形状
                    npy_intp bshape = broadcast_shape[idim+ndim-ondim];
                    npy_intp op_shape = shape[idim];

                    # 如果广播形状为1，则更新为当前操作数的形状
                    if (bshape == 1) {
                        broadcast_shape[idim+ndim-ondim] = op_shape;
                    }
                    # 否则，如果广播形状与当前操作数的形状不同且当前操作数的形状不为1，则跳转到广播错误处理
                    else if (bshape != op_shape && op_shape != 1) {
                        goto broadcast_error;
                    }
                }
            }
            # 如果有指定操作轴
            else {
                # 获取当前操作的轴数组
                int *axes = op_axes[iop];
                # 遍历迭代器的维度数
                for (idim = 0; idim < ndim; ++idim) {
                    # 获取操作轴的索引 i
                    int i = npyiter_get_op_axis(axes[idim], NULL);

                    # 如果索引 i 是有效的
                    if (i >= 0) {
                        # 如果 i 小于当前操作数的维度数
                        if (i < ondim) {
                            # 获取广播后的形状和当前操作数轴 i 的形状
                            npy_intp bshape = broadcast_shape[idim];
                            npy_intp op_shape = shape[i];

                            # 如果广播形状为1，则更新为当前操作数轴 i 的形状
                            if (bshape == 1) {
                                broadcast_shape[idim] = op_shape;
                            }
                            # 否则，如果广播形状与当前操作数轴 i 的形状不同且当前操作数轴 i 的形状不为1，则跳转到广播错误处理
                            else if (bshape != op_shape && op_shape != 1) {
                                goto broadcast_error;
                            }
                        }
                        # 否则，如果 i 不在当前操作数的维度范围内，抛出错误
                        else {
                            PyErr_Format(PyExc_ValueError,
                                    "Iterator input op_axes[%d][%d] (==%d) "
                                    "is not a valid axis of op[%d], which "
                                    "has %d dimensions ",
                                    iop, (ndim-idim-1), i,
                                    iop, ondim);
                            return 0;
                        }
                    }
                }
            }
        }
    }
    /*
     * 如果提供了形状并且有一个条目为1，则确保该条目没有通过广播进行扩展。
     */
    # 如果 itershape 不为空
    if (itershape != NULL) {
        # 遍历迭代器的维度数
        for (idim = 0; idim < ndim; ++idim) {
            # 如果 itershape 的当前维度为1，并且广播后的形状的当前维度不为1，则跳转到广播错误处理
            if (itershape[idim] == 1 && broadcast_shape[idim] != 1) {
                goto broadcast_error;
            }
        }
    }

    # 获取轴数据和轴数据大小
    axisdata = NIT_AXISDATA(iter);
    sizeof_axisdata = NIT_AXISDATA_SIZEOF(itflags, ndim, nop);
    # 如果数组维度为0，则需要填充第一个轴的axisdata，即使迭代器是0维的情况下
    if (ndim == 0) {
        # 设置axisdata的形状为1
        NAD_SHAPE(axisdata) = 1;
        # 设置axisdata的索引为0
        NAD_INDEX(axisdata) = 0;
        # 将op_dataptr指向的数据拷贝到axisdata的指针位置，共拷贝nop个字节
        memcpy(NAD_PTRS(axisdata), op_dataptr, NPY_SIZEOF_INTP*nop);
        # 将axisdata的步长数组位置的内容设置为0，共设置nop个步长
        memset(NAD_STRIDES(axisdata), 0, NPY_SIZEOF_INTP*nop);
    }

    # 现在处理操作数，填充axisdata
    }

    # 现在填充ITERSIZE成员
    # 设置ITERSIZE为1
    NIT_ITERSIZE(iter) = 1;
    # 遍历维度，计算总的迭代大小
    for (idim = 0; idim < ndim; ++idim) {
        # 如果计算NIT_ITERSIZE(iter)与broadcast_shape[idim]的乘积溢出
        if (npy_mul_sizes_with_overflow(&NIT_ITERSIZE(iter),
                    NIT_ITERSIZE(iter), broadcast_shape[idim])) {
            # 如果itflags包含NPY_ITFLAG_HASMULTIINDEX，且不包含NPY_ITFLAG_HASINDEX和NPY_ITFLAG_BUFFER
            # 则延迟大小检查，直到多索引被移除或GetIterNext被调用
            NIT_ITERSIZE(iter) = -1;
            break;
        }
        else {
            # 如果迭代器太大，抛出ValueError异常
            PyErr_SetString(PyExc_ValueError, "iterator is too large");
            return 0;
        }
    }
    # 默认迭代范围为全部数据
    NIT_ITERSTART(iter) = 0;
    NIT_ITEREND(iter) = NIT_ITERSIZE(iter);

    # 返回成功标志
    return 1;

    }
operand_different_than_broadcast: {
    /* operand shape */
    int ndims = PyArray_NDIM(op[iop]);  // 获取操作数的维度数
    npy_intp *dims = PyArray_DIMS(op[iop]);  // 获取操作数的维度数组
    PyObject *shape1 = convert_shape_to_string(ndims, dims, "");  // 将维度数组转换为字符串表示的形状
    if (shape1 == NULL) {  // 如果形状转换失败，则返回 0
        return 0;
    }

    /* Broadcast shape */
    PyObject *shape2 = convert_shape_to_string(ndim, broadcast_shape, "");  // 将广播形状转换为字符串表示的形状
    if (shape2 == NULL) {  // 如果形状转换失败，则释放 shape1 并返回 0
        Py_DECREF(shape1);
        return 0;
    }

    if (op_axes == NULL || op_axes[iop] == NULL) {
        /* operand shape not remapped */
        
        if (op_flags[iop] & NPY_ITER_READONLY) {
            PyErr_Format(PyExc_ValueError,
                "non-broadcastable operand with shape %S doesn't "
                "match the broadcast shape %S", shape1, shape2);  // 报错，显示操作数形状与广播形状不匹配
        }
        else {
            PyErr_Format(PyExc_ValueError,
                "non-broadcastable output operand with shape %S doesn't "
                "match the broadcast shape %S", shape1, shape2);  // 报错，显示输出操作数形状与广播形状不匹配
        }
        Py_DECREF(shape1);
        Py_DECREF(shape2);
        return 0;
    }
    else {
        /* operand shape remapped */

        npy_intp remdims[NPY_MAXDIMS];  // 创建重新映射的维度数组
        int *axes = op_axes[iop];  // 获取操作数的轴映射数组
        for (idim = 0; idim < ndim; ++idim) {
            npy_intp i = axes[ndim - idim - 1];  // 获取映射后的轴索引
            if (i >= 0 && i < PyArray_NDIM(op[iop])) {
                remdims[idim] = PyArray_DIM(op[iop], i);  // 填充重新映射的维度数组
            }
            else {
                remdims[idim] = -1;  // 如果映射索引不合法，填充为 -1
            }
        }

        PyObject *shape3 = convert_shape_to_string(ndim, remdims, "");  // 将重新映射的维度数组转换为字符串表示的形状
        if (shape3 == NULL) {  // 如果形状转换失败，则释放 shape1 和 shape2，并返回 0
            Py_DECREF(shape1);
            Py_DECREF(shape2);
            return 0;
        }

        if (op_flags[iop] & NPY_ITER_READONLY) {
            PyErr_Format(PyExc_ValueError,
                "non-broadcastable operand with shape %S "
                "[remapped to %S] doesn't match the broadcast shape %S",
                shape1, shape3, shape2);  // 报错，显示操作数形状经过重新映射后与广播形状不匹配
        }
        else {
            PyErr_Format(PyExc_ValueError,
                "non-broadcastable output operand with shape %S "
                "[remapped to %S] doesn't match the broadcast shape %S",
                shape1, shape3, shape2);  // 报错，显示输出操作数形状经过重新映射后与广播形状不匹配
        }
        Py_DECREF(shape1);
        Py_DECREF(shape2);
        Py_DECREF(shape3);
        return 0;
    }
}
{
    // 获取迭代器的标志位
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    // 获取迭代器的维度数目
    int idim, ndim = NIT_NDIM(iter);
    // 获取迭代器的操作数目
    int nop = NIT_NOP(iter);
    // 获取操作数组的数据指针
    char *op_dataptr = PyArray_DATA(op);

    // 迭代器的轴数据指针和相关变量声明
    NpyIter_AxisData *axisdata0, *axisdata;
    npy_intp sizeof_axisdata;
    npy_int8 *perm;
    npy_intp baseoffset = 0;

    // 获取迭代器的轴数据排列
    perm = NIT_PERM(iter);
    // 获取迭代器的初始轴数据
    axisdata0 = NIT_AXISDATA(iter);
    // 获取轴数据的大小
    sizeof_axisdata = NIT_AXISDATA_SIZEOF(itflags, ndim, nop);

    /*
     * 替换非零步幅，并计算基本数据地址。
     */
    axisdata = axisdata0;

    // 如果操作轴不为空
    if (op_axes != NULL) {
        // 遍历迭代器的维度
        for (idim = 0; idim < ndim; ++idim, NIT_ADVANCE_AXISDATA(axisdata, 1)) {
            int i;
            npy_bool axis_flipped;
            npy_intp shape;

            /* 应用排列来获取原始轴，并检查其是否翻转 */
            i = npyiter_undo_iter_axis_perm(idim, ndim, perm, &axis_flipped);

            // 获取操作数组的轴索引
            i = npyiter_get_op_axis(op_axes[i], NULL);
            // 断言索引小于原始操作数组的维度数
            assert(i < orig_op_ndim);
            if (i >= 0) {
                // 获取操作数组在指定轴上的维度
                shape = PyArray_DIM(op, i);
                // 如果维度不为1
                if (shape != 1) {
                    // 获取操作数组在指定轴上的步幅
                    npy_intp stride = PyArray_STRIDE(op, i);
                    // 如果轴被翻转
                    if (axis_flipped) {
                        // 设置轴数据的步幅为负值
                        NAD_STRIDES(axisdata)[iop] = -stride;
                        // 基本偏移增加步幅乘以形状减1
                        baseoffset += stride * (shape - 1);
                    } else {
                        // 设置轴数据的步幅为正值
                        NAD_STRIDES(axisdata)[iop] = stride;
                    }
                }
            }
        }
    } else {
        // 如果操作轴为空
        for (idim = 0; idim < ndim; ++idim, NIT_ADVANCE_AXISDATA(axisdata, 1)) {
            int i;
            npy_bool axis_flipped;
            npy_intp shape;

            // 获取未经过迭代器轴排列的原始操作数组轴索引
            i = npyiter_undo_iter_axis_perm(idim, orig_op_ndim, perm, &axis_flipped);

            // 如果索引大于等于0
            if (i >= 0) {
                // 获取操作数组在指定轴上的维度
                shape = PyArray_DIM(op, i);
                // 如果维度不为1
                if (shape != 1) {
                    // 获取操作数组在指定轴上的步幅
                    npy_intp stride = PyArray_STRIDE(op, i);
                    // 如果轴被翻转
                    if (axis_flipped) {
                        // 设置轴数据的步幅为负值
                        NAD_STRIDES(axisdata)[iop] = -stride;
                        // 基本偏移增加步幅乘以形状减1
                        baseoffset += stride * (shape - 1);
                    } else {
                        // 设置轴数据的步幅为正值
                        NAD_STRIDES(axisdata)[iop] = stride;
                    }
                }
            }
        }
    }

    // 增加基本数据地址偏移量
    op_dataptr += baseoffset;

    /* 现在基本数据指针已经计算出来，将其设置到所有需要的地方 */
    // 设置迭代器重置数据指针
    NIT_RESETDATAPTR(iter)[iop] = op_dataptr;
    // 设置迭代器基本偏移
    NIT_BASEOFFSETS(iter)[iop] = baseoffset;
    // 重新初始化轴数据
    axisdata = axisdata0;
    /* 对于0维情况，至少填充一个轴数据 */
    // 设置第一个轴数据指针
    NAD_PTRS(axisdata)[iop] = op_dataptr;
    // 逐步增加轴数据指针
    NIT_ADVANCE_AXISDATA(axisdata, 1);
    // 遍历迭代器的维度
    for (idim = 1; idim < ndim; ++idim, NIT_ADVANCE_AXISDATA(axisdata, 1)) {
        // 设置轴数据指针
        NAD_PTRS(axisdata)[iop] = op_dataptr;
    }
}
/*
 * 计算迭代器的索引步长。
 * iter: 迭代器对象
 * flags: 标志位，指示索引顺序（C顺序或Fortran顺序）
 */
npyiter_compute_index_strides(NpyIter *iter, npy_uint32 flags)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);  // 获取迭代器的标志位
    int idim, ndim = NIT_NDIM(iter);         // 获取迭代器的维度信息
    int nop = NIT_NOP(iter);                 // 获取迭代器的操作数

    npy_intp indexstride;                    // 索引步长
    NpyIter_AxisData *axisdata;              // 指向轴数据的指针
    npy_intp sizeof_axisdata;                // 轴数据的大小

    /*
     * 如果只有一个元素在迭代，则只需操作第一个轴数据，因为没有任何增量操作。
     * 这也初始化了0维情况下的数据。
     */
    if (NIT_ITERSIZE(iter) == 1) {
        if (itflags & NPY_ITFLAG_HASINDEX) {
            axisdata = NIT_AXISDATA(iter);
            NAD_PTRS(axisdata)[nop] = 0;
        }
        return;
    }

    if (flags & NPY_ITER_C_INDEX) {
        sizeof_axisdata = NIT_AXISDATA_SIZEOF(itflags, ndim, nop);
        axisdata = NIT_AXISDATA(iter);
        indexstride = 1;
        for(idim = 0; idim < ndim; ++idim, NIT_ADVANCE_AXISDATA(axisdata, 1)) {
            npy_intp shape = NAD_SHAPE(axisdata);

            if (shape == 1) {
                NAD_STRIDES(axisdata)[nop] = 0;
            }
            else {
                NAD_STRIDES(axisdata)[nop] = indexstride;
            }
            NAD_PTRS(axisdata)[nop] = 0;
            indexstride *= shape;
        }
    }
    else if (flags & NPY_ITER_F_INDEX) {
        sizeof_axisdata = NIT_AXISDATA_SIZEOF(itflags, ndim, nop);
        axisdata = NIT_INDEX_AXISDATA(NIT_AXISDATA(iter), ndim-1);
        indexstride = 1;
        for(idim = 0; idim < ndim; ++idim, NIT_ADVANCE_AXISDATA(axisdata, -1)) {
            npy_intp shape = NAD_SHAPE(axisdata);

            if (shape == 1) {
                NAD_STRIDES(axisdata)[nop] = 0;
            }
            else {
                NAD_STRIDES(axisdata)[nop] = indexstride;
            }
            NAD_PTRS(axisdata)[nop] = 0;
            indexstride *= shape;
        }
    }
}

/*
 * 如果 order 是 NPY_KEEPORDER，则让迭代器找到最佳的迭代顺序；否则强制指定顺序。
 * 在 itflags 中指示是否强制了迭代顺序。
 */
static void
npyiter_apply_forced_iteration_order(NpyIter *iter, NPY_ORDER order)
{
    /*npy_uint32 itflags = NIT_ITFLAGS(iter);*/  // 注释掉的代码，不会被执行
    int ndim = NIT_NDIM(iter);                  // 获取迭代器的维度信息
    int iop, nop = NIT_NOP(iter);               // 获取迭代器的操作数

    switch (order) {
    case NPY_CORDER:
        NIT_ITFLAGS(iter) |= NPY_ITFLAG_FORCEDORDER;  // 强制迭代顺序为 C 顺序
        break;
    case NPY_FORTRANORDER:
        NIT_ITFLAGS(iter) |= NPY_ITFLAG_FORCEDORDER;  // 强制迭代顺序为 Fortran 顺序
        /* 只有在维度大于1时才需要实际执行操作 */
        if (ndim > 1) {
            npyiter_reverse_axis_ordering(iter);  // 反转轴的顺序
        }
        break;
    # 当选项为 NPY_ANYORDER 时执行以下操作
    case NPY_ANYORDER:
        # 设置迭代器强制顺序标志位，确保按顺序迭代
        NIT_ITFLAGS(iter) |= NPY_ITFLAG_FORCEDORDER;
        # 只有在维度大于 1 时才需要实际执行操作
        if (ndim > 1) {
            # 获取操作数数组的指针数组
            PyArrayObject **op = NIT_OPERANDS(iter);
            # 默认按顺序
            int forder = 1;

            # 检查所有数组输入是否都是 Fortran（列优先）顺序
            for (iop = 0; iop < nop; ++iop, ++op) {
                # 如果当前数组不是 Fortran 顺序，则取消顺序标志
                if (*op && !PyArray_CHKFLAGS(*op, NPY_ARRAY_F_CONTIGUOUS)) {
                    forder = 0;
                    break;
                }
            }

            # 如果所有数组都是 Fortran 顺序，则反转轴顺序
            if (forder) {
                npyiter_reverse_axis_ordering(iter);
            }
        }
        break;

    # 当选项为 NPY_KEEPORDER 时不执行任何操作
    case NPY_KEEPORDER:
        # 这里不设置强制顺序标志...
        break;
/*
 * This function negates any strides in the iterator
 * which are negative. When iterating over multiple operands,
 * it flips strides only if all are negative or zero.
 */
static void
npyiter_flip_negative_strides(NpyIter *iter)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);  // 获取迭代器的标志位
    int idim, ndim = NIT_NDIM(iter);  // 获取迭代器的维度数和当前维度
    int iop, nop = NIT_NOP(iter);  // 获取迭代器的操作数数目

    npy_intp istrides, nstrides = NAD_NSTRIDES();  // 获取每个轴的步长数目
    NpyIter_AxisData *axisdata, *axisdata0;  // 定义轴数据指针
    npy_intp *baseoffsets;  // 基础偏移量指针
    npy_intp sizeof_axisdata = NIT_AXISDATA_SIZEOF(itflags, ndim, nop);  // 计算轴数据结构体大小
    int any_flipped = 0;  // 指示是否有步长被反转过的标志位

    axisdata0 = axisdata = NIT_AXISDATA(iter);  // 获取迭代器的轴数据
    baseoffsets = NIT_BASEOFFSETS(iter);  // 获取基础偏移量数组
    for (idim = 0; idim < ndim; ++idim, NIT_ADVANCE_AXISDATA(axisdata, 1)) {
        npy_intp *strides = NAD_STRIDES(axisdata);  // 获取当前轴的步长数组
        int any_negative = 0;  // 指示当前轴是否有负步长的标志位

        /*
         * Check the signs of all the operand strides.
         */
        for (iop = 0; iop < nop; ++iop) {
            if (strides[iop] < 0) {
                any_negative = 1;  // 如果有负步长则设置标志位
            }
            else if (strides[iop] != 0) {
                break;  // 如果有正步长则退出循环
            }
        }
        /*
         * If at least one stride is negative and none are positive,
         * flip all the strides for this dimension.
         */
        if (any_negative && iop == nop) {
            npy_intp shapem1 = NAD_SHAPE(axisdata) - 1;  // 获取当前轴的形状-1

            for (istrides = 0; istrides < nstrides; ++istrides) {
                npy_intp stride = strides[istrides];

                /* Adjust the base pointers to start at the end */
                baseoffsets[istrides] += shapem1 * stride;  // 调整基础偏移量
                /* Flip the stride */
                strides[istrides] = -stride;  // 反转步长为负值
            }
            /*
             * Make the perm entry negative so get_multi_index
             * knows it's flipped
             */
            NIT_PERM(iter)[idim] = -1 - NIT_PERM(iter)[idim];  // 设置轴的排列索引为负值以指示反转

            any_flipped = 1;  // 设置有步长被反转的标志位
        }
    }

    /*
     * If any strides were flipped, the base pointers were adjusted
     * in the first AXISDATA, and need to be copied to all the rest
     */
    if (any_flipped) {
        char **resetdataptr = NIT_RESETDATAPTR(iter);  // 获取重置数据指针数组

        for (istrides = 0; istrides < nstrides; ++istrides) {
            resetdataptr[istrides] += baseoffsets[istrides];  // 调整重置数据指针的偏移量
        }
        axisdata = axisdata0;
        for (idim = 0; idim < ndim; ++idim, NIT_ADVANCE_AXISDATA(axisdata, 1)) {
            char **ptrs = NAD_PTRS(axisdata);
            for (istrides = 0; istrides < nstrides; ++istrides) {
                ptrs[istrides] = resetdataptr[istrides];  // 设置轴数据指针为重置数据指针
            }
        }
        /*
         * Indicate that some of the perm entries are negative,
         * and that it's not (strictly speaking) the identity perm.
         */
        NIT_ITFLAGS(iter) = (NIT_ITFLAGS(iter) | NPY_ITFLAG_NEGPERM) & ~NPY_ITFLAG_IDENTPERM;  // 更新迭代器的标志位
    }
}

/*
 * Reverse the order of iteration over the axes in the iterator.
 */
static void
npyiter_reverse_axis_ordering(NpyIter *iter)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);  // 获取迭代器的标志位
    # 获取迭代器中的维度数
    int ndim = NIT_NDIM(iter);
    # 获取迭代器中的操作数
    int nop = NIT_NOP(iter);

    # 声明整型变量和指针变量
    npy_intp i, temp, size;
    npy_intp *first, *last;
    npy_int8 *perm;

    # 计算 AXISDATA 数组的大小并分配空间
    size = NIT_AXISDATA_SIZEOF(itflags, ndim, nop) / NPY_SIZEOF_INTP;
    # 获取 AXISDATA 数组的起始地址
    first = (npy_intp*)NIT_AXISDATA(iter);
    # 计算 AXISDATA 数组的末尾地址
    last = first + (ndim - 1) * size;

    /* This loop reverses the order of the AXISDATA array */
    # 循环：反转 AXISDATA 数组的顺序
    while (first < last) {
        for (i = 0; i < size; ++i) {
            temp = first[i];
            first[i] = last[i];
            last[i] = temp;
        }
        first += size;
        last -= size;
    }

    /* Store the perm we applied */
    # 存储应用的排列顺序
    perm = NIT_PERM(iter);
    for (i = ndim - 1; i >= 0; --i, ++perm) {
        *perm = (npy_int8)i;
    }

    # 清除标志位 NPY_ITFLAG_IDENTPERM
    NIT_ITFLAGS(iter) &= ~NPY_ITFLAG_IDENTPERM;
static inline npy_intp
intp_abs(npy_intp x)
{
    // 返回整数 x 的绝对值
    return (x < 0) ? -x : x;
}

static void
npyiter_find_best_axis_ordering(NpyIter *iter)
{
    // 获取迭代器的标志位
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    // 获取迭代器的维度数
    int idim, ndim = NIT_NDIM(iter);
    // 获取迭代器的操作数数量
    int iop, nop = NIT_NOP(iter);

    npy_intp ax_i0, ax_i1, ax_ipos;
    npy_int8 ax_j0, ax_j1;
    npy_int8 *perm;
    // 获取迭代器的轴数据
    NpyIter_AxisData *axisdata = NIT_AXISDATA(iter);
    // 计算轴数据结构体的大小
    npy_intp sizeof_axisdata = NIT_AXISDATA_SIZEOF(itflags, ndim, nop);
    // 判断是否进行了置换
    int permuted = 0;

    // 获取迭代器的轴置换数组
    perm = NIT_PERM(iter);

    /*
     * 进行自定义的稳定插入排序。注意，由于 AXISDATA 是从 C 顺序反转过来的，
     * 这里是按照从最小步幅到最大步幅的顺序进行排序。
     */
    for (ax_i0 = 1; ax_i0 < ndim; ++ax_i0) {
        npy_intp *strides0;

        /* 'ax_ipos' 是 perm[ax_i0] 将要插入的位置 */
        ax_ipos = ax_i0;
        ax_j0 = perm[ax_i0];

        // 获取轴 ax_j0 的步幅数组
        strides0 = NAD_STRIDES(NIT_INDEX_AXISDATA(axisdata, ax_j0));
        for (ax_i1 = ax_i0-1; ax_i1 >= 0; --ax_i1) {
            int ambig = 1, shouldswap = 0;
            npy_intp *strides1;

            ax_j1 = perm[ax_i1];

            // 获取轴 ax_j1 的步幅数组
            strides1 = NAD_STRIDES(NIT_INDEX_AXISDATA(axisdata, ax_j1));

            for (iop = 0; iop < nop; ++iop) {
                if (strides0[iop] != 0 && strides1[iop] != 0) {
                    if (intp_abs(strides1[iop]) <= intp_abs(strides0[iop])) {
                        /*
                         * 即使在不明确的情况下，也要设置交换，因为在不同操作数之间的冲突情况下，C 顺序优先。
                         */
                        shouldswap = 0;
                    }
                    else {
                        /* 只有在仍然不明确的情况下才设置交换 */
                        if (ambig) {
                            shouldswap = 1;
                        }
                    }

                    /*
                     * 已经进行了比较，因此不再是不明确的
                     */
                    ambig = 0;
                }
            }
            /*
             * 如果比较是明确的，则将 'ax_ipos' 移动到 'ax_i1' 或停止查找插入点
             */
            if (!ambig) {
                if (shouldswap) {
                    ax_ipos = ax_i1;
                }
                else {
                    break;
                }
            }
        }

        /* 将 perm[ax_i0] 插入到正确的位置 */
        if (ax_ipos != ax_i0) {
            for (ax_i1 = ax_i0; ax_i1 > ax_ipos; --ax_i1) {
                perm[ax_i1] = perm[ax_i1-1];
            }
            perm[ax_ipos] = ax_j0;
            permuted = 1;
        }
    }

    /* 将计算出的置换应用于 AXISDATA 数组 */
}
    if (permuted == 1) {
        npy_intp i, size = sizeof_axisdata/NPY_SIZEOF_INTP;
        NpyIter_AxisData *ad_i;

        /* Use the index as a flag, set each to 1 */
        ad_i = axisdata;
        for (idim = 0; idim < ndim; ++idim, NIT_ADVANCE_AXISDATA(ad_i, 1)) {
            NAD_INDEX(ad_i) = 1;
        }
        /* Apply the permutation by following the cycles */
        for (idim = 0; idim < ndim; ++idim) {
            ad_i = NIT_INDEX_AXISDATA(axisdata, idim);

            /* If this axis hasn't been touched yet, process it */
            if (NAD_INDEX(ad_i) == 1) {
                npy_int8 pidim = perm[idim];
                npy_intp tmp;
                NpyIter_AxisData *ad_p, *ad_q;

                if (pidim != idim) {
                    /* Follow the cycle, copying the data */
                    for (i = 0; i < size; ++i) {
                        pidim = perm[idim];
                        ad_q = ad_i;
                        tmp = *((npy_intp*)ad_q + i);
                        while (pidim != idim) {
                            ad_p = NIT_INDEX_AXISDATA(axisdata, pidim);
                            *((npy_intp*)ad_q + i) = *((npy_intp*)ad_p + i);

                            ad_q = ad_p;
                            pidim = perm[(int)pidim];
                        }
                        *((npy_intp*)ad_q + i) = tmp;
                    }
                    /* Follow the cycle again, marking it as done */
                    pidim = perm[idim];

                    while (pidim != idim) {
                        NAD_INDEX(NIT_INDEX_AXISDATA(axisdata, pidim)) = 0;
                        pidim = perm[(int)pidim];
                    }
                }
                NAD_INDEX(ad_i) = 0;
            }
        }
        /* Clear the identity perm flag */
        NIT_ITFLAGS(iter) &= ~NPY_ITFLAG_IDENTPERM;
    }



        /* 如果 permuted 等于 1，则执行以下操作 */
        if (permuted == 1) {
            /* 定义循环变量 i 和 size */
            npy_intp i, size = sizeof_axisdata/NPY_SIZEOF_INTP;
            NpyIter_AxisData *ad_i;

            /* 使用索引作为标志，将每个标志设置为 1 */
            ad_i = axisdata;
            for (idim = 0; idim < ndim; ++idim, NIT_ADVANCE_AXISDATA(ad_i, 1)) {
                NAD_INDEX(ad_i) = 1;
            }

            /* 根据排列应用置换操作 */
            for (idim = 0; idim < ndim; ++idim) {
                ad_i = NIT_INDEX_AXISDATA(axisdata, idim);

                /* 如果这个轴还没有被处理过，则处理它 */
                if (NAD_INDEX(ad_i) == 1) {
                    npy_int8 pidim = perm[idim];
                    npy_intp tmp;
                    NpyIter_AxisData *ad_p, *ad_q;

                    if (pidim != idim) {
                        /* 按照循环路径复制数据 */
                        for (i = 0; i < size; ++i) {
                            pidim = perm[idim];
                            ad_q = ad_i;
                            tmp = *((npy_intp*)ad_q + i);
                            while (pidim != idim) {
                                ad_p = NIT_INDEX_AXISDATA(axisdata, pidim);
                                *((npy_intp*)ad_q + i) = *((npy_intp*)ad_p + i);

                                ad_q = ad_p;
                                pidim = perm[(int)pidim];
                            }
                            *((npy_intp*)ad_q + i) = tmp;
                        }
                        /* 再次按照循环路径标记为已完成 */
                        pidim = perm[idim];

                        while (pidim != idim) {
                            NAD_INDEX(NIT_INDEX_AXISDATA(axisdata, pidim)) = 0;
                            pidim = perm[(int)pidim];
                        }
                    }
                    NAD_INDEX(ad_i) = 0;
                }
            }
            /* 清除身份置换标志 */
            NIT_ITFLAGS(iter) &= ~NPY_ITFLAG_IDENTPERM;
        }
/*
 * Calculates a dtype that all the types can be promoted to, using the
 * ufunc rules.  If only_inputs is 1, it leaves any operands that
 * are not read from out of the calculation.
 */
static PyArray_Descr *
npyiter_get_common_dtype(int nop, PyArrayObject **op,
                        const npyiter_opitflags *op_itflags, PyArray_Descr **op_dtype,
                        PyArray_Descr **op_request_dtypes,
                        int only_inputs)
{
    int iop;
    npy_intp narrs = 0, ndtypes = 0;
    PyArrayObject *arrs[NPY_MAXARGS];
    PyArray_Descr *dtypes[NPY_MAXARGS];
    PyArray_Descr *ret;

    NPY_IT_DBG_PRINT("Iterator: Getting a common data type from operands\n");

    // 遍历所有操作数
    for (iop = 0; iop < nop; ++iop) {
        // 检查是否存在操作数的 dtype，并且只有在 only_inputs 为 0 或者 op_itflags 表明需要读取时才考虑
        if (op_dtype[iop] != NULL &&
                    (!only_inputs || (op_itflags[iop] & NPY_OP_ITFLAG_READ))) {
            /* 如果没有请求 dtype，并且操作数是标量，将操作数本身传入数组 */
            if ((op_request_dtypes == NULL ||
                            op_request_dtypes[iop] == NULL) &&
                                            PyArray_NDIM(op[iop]) == 0) {
                arrs[narrs++] = op[iop];
            }
            /* 否则，将操作数的 dtype 传入数组 */
            else {
                dtypes[ndtypes++] = op_dtype[iop];
            }
        }
    }

    // 根据收集到的操作数和 dtype 计算共同的数据类型
    if (narrs == 0) {
        npy_intp i;
        ret = dtypes[0];
        for (i = 1; i < ndtypes; ++i) {
            // 如果所有的 dtypes 都相同，选择第一个 dtype
            if (ret != dtypes[i])
                break;
        }
        // 如果所有 dtypes 都相同，且只有一个 dtype 或者该 dtype 是本机字节顺序，增加其引用计数
        if (i == ndtypes) {
            if (ndtypes == 1 || PyArray_ISNBO(ret->byteorder)) {
                Py_INCREF(ret);
            }
            // 否则，根据本机字节顺序创建一个新的 dtype
            else {
                ret = PyArray_DescrNewByteorder(ret, NPY_NATIVE);
            }
        }
        // 否则，根据给定的数组和 dtypes 计算结果 dtype
        else {
            ret = PyArray_ResultType(narrs, arrs, ndtypes, dtypes);
        }
    }
    // 如果有操作数是标量，则根据给定的数组和 dtypes 计算结果 dtype
    else {
        ret = PyArray_ResultType(narrs, arrs, ndtypes, dtypes);
    }

    return ret;
}

/*
 * Allocates a temporary array which can be used to replace op
 * in the iteration.  Its dtype will be op_dtype.
 *
 * The result array has a memory ordering which matches the iterator,
 * which may or may not match that of op.  The parameter 'shape' may be
 * NULL, in which case it is filled in from the iterator's shape.
 *
 * This function must be called before any axes are coalesced.
 */
static PyArrayObject *
npyiter_new_temp_array(NpyIter *iter, PyTypeObject *subtype,
                npy_uint32 flags, npyiter_opitflags *op_itflags,
                int op_ndim, npy_intp const *shape,
                PyArray_Descr *op_dtype, const int *op_axes)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    int idim, ndim = NIT_NDIM(iter);
    int used_op_ndim;
    int nop = NIT_NOP(iter);

    npy_int8 *perm = NIT_PERM(iter);
    npy_intp new_shape[NPY_MAXDIMS], strides[NPY_MAXDIMS];
    npy_intp stride = op_dtype->elsize;
    NpyIter_AxisData *axisdata;
    npy_intp sizeof_axisdata;
    int i;

    PyArrayObject *ret;
    /*
     * 在这里与数组数据类型有交互，通常情况下是有效的。假设您使用一个带有输出数据类型为双精度数组的nditer。
     * 所有标量输入将导致一个形状为(2)的一维输出。在nditer中一切仍然正常工作，因为新维度始终添加到末尾，
     * 它关心开始时发生的情况。
     */

    /* 如果是标量，不需要检查轴 */
    if (op_ndim == 0) {
        // 增加引用计数以避免释放内存
        Py_INCREF(op_dtype);
        // 从描述符创建新的数组对象，没有轴
        ret = (PyArrayObject *)PyArray_NewFromDescr(subtype, op_dtype, 0,
                               NULL, NULL, NULL, 0, NULL);
        // 返回新创建的数组对象
        return ret;
    }

    // 获取轴数据
    axisdata = NIT_AXISDATA(iter);
    // 计算轴数据的大小
    sizeof_axisdata = NIT_AXISDATA_SIZEOF(itflags, ndim, nop);

    /* 初始化步幅为无效值 */
    for (i = 0; i < op_ndim; ++i) {
        strides[i] = NPY_MAX_INTP;
    }

    // 如果操作轴不为空
    if (op_axes != NULL) {
        used_op_ndim = 0;
        // 迭代每个轴
        for (idim = 0; idim < ndim; ++idim, NIT_ADVANCE_AXISDATA(axisdata, 1)) {
            npy_bool reduction_axis;

            /* 应用排列以获取原始轴 */
            i = npyiter_undo_iter_axis_perm(idim, ndim, perm, NULL);
            // 获取操作轴
            i = npyiter_get_op_axis(op_axes[i], &reduction_axis);

            /*
             * 如果 i < 0，这是一个新轴（操作数没有它），因此我们可以在这里忽略它。
             * 迭代器设置将已经确保了潜在的减少/广播是有效的。
             */
            if (i >= 0) {
                // 调试信息，设置分配的步幅
                NPY_IT_DBG_PRINT3("Iterator: Setting allocated stride %d "
                                    "for iterator dimension %d to %d\n", (int)i,
                                    (int)idim, (int)stride);
                // 增加已使用的操作轴计数
                used_op_ndim += 1;
                // 设置步幅
                strides[i] = stride;
                // 如果形状为空
                if (shape == NULL) {
                    // 如果是减少轴，长度总是1
                    if (reduction_axis) {
                        new_shape[i] = 1;
                    }
                    else {
                        // 否则获取轴的形状
                        new_shape[i] = NAD_SHAPE(axisdata);
                    }
                    // 更新步幅
                    stride *= new_shape[i];
                    // 如果 i 大于等于轴的数量，报错
                    if (i >= ndim) {
                        PyErr_Format(PyExc_ValueError,
                                "automatically allocated output array "
                                "specified with an inconsistent axis mapping; "
                                "the axis mapping cannot include dimension %d "
                                "which is too large for the iterator dimension "
                                "of %d.", i, ndim);
                        // 返回空指针，表示错误
                        return NULL;
                    }
                }
                else {
                    // 断言：如果是减少轴，形状应该是1
                    assert(!reduction_axis || shape[i] == 1);
                    // 更新步幅
                    stride *= shape[i];
                }
            }
        }
    }
    else {
        // 如果 shape 不为 NULL，则计算并设置新的 strides
        used_op_ndim = ndim;
        for (idim = 0; idim < ndim; ++idim, NIT_ADVANCE_AXISDATA(axisdata, 1)) {
            /* Apply the perm to get the original axis */
            // 根据排列 perm，找到原始的轴
            i = npyiter_undo_iter_axis_perm(idim, op_ndim, perm, NULL);

            if (i >= 0) {
                // 调试输出：设置迭代器维度 idim 的已分配步长 i 为 stride
                NPY_IT_DBG_PRINT3("Iterator: Setting allocated stride %d "
                                    "for iterator dimension %d to %d\n", (int)i,
                                    (int)idim, (int)stride);
                strides[i] = stride;
                if (shape == NULL) {
                    // 如果 shape 为空，则使用新计算的 shape
                    new_shape[i] = NAD_SHAPE(axisdata);
                    stride *= new_shape[i];
                }
                else {
                    // 否则使用给定的 shape
                    stride *= shape[i];
                }
            }
        }
    }

    if (shape == NULL) {
        /* If shape was NULL, use the shape we calculated */
        // 如果 shape 为空，则使用之前计算的 new_shape
        op_ndim = used_op_ndim;
        shape = new_shape;
        /*
         * If there's a gap in the array's dimensions, it's an error.
         * For instance, if op_axes [0, 2] is specified, there will a place
         * in the strides array where the value is not set.
         */
        for (i = 0; i < op_ndim; i++) {
            // 如果 strides 中有 NPY_MAX_INTP，表示存在维度映射缺失，报错
            if (strides[i] == NPY_MAX_INTP) {
                PyErr_Format(PyExc_ValueError,
                        "automatically allocated output array "
                        "specified with an inconsistent axis mapping; "
                        "the axis mapping is missing an entry for "
                        "dimension %d.", i);
                return NULL;
            }
        }
    }
    else if (used_op_ndim < op_ndim) {
        /*
         * If custom axes were specified, some dimensions may not have
         * been used. These are additional axes which are ignored in the
         * iterator but need to be handled here.
         */
        // 如果 op_ndim 比 used_op_ndim 大，说明有额外的维度需要处理
        npy_intp factor, itemsize, new_strides[NPY_MAXDIMS];

        /* Fill in the missing strides in C order */
        // 按照 C 顺序填充缺失的步长
        factor = 1;
        itemsize = op_dtype->elsize;
        for (i = op_ndim-1; i >= 0; --i) {
            if (strides[i] == NPY_MAX_INTP) {
                new_strides[i] = factor * itemsize;
                factor *= shape[i];
            }
        }

        /*
         * Copy the missing strides, and multiply the existing strides
         * by the calculated factor.  This way, the missing strides
         * are tighter together in memory, which is good for nested
         * loops.
         */
        // 复制缺失的步长，并且将现有的步长乘以计算得到的因子
        for (i = 0; i < op_ndim; ++i) {
            if (strides[i] == NPY_MAX_INTP) {
                strides[i] = new_strides[i];
            }
            else {
                strides[i] *= factor;
            }
        }
    }

    /* Allocate the temporary array */
    // 分配临时数组
    Py_INCREF(op_dtype);
    ret = (PyArrayObject *)PyArray_NewFromDescr(subtype, op_dtype, op_ndim,
                               shape, strides, NULL, 0, NULL);
    if (ret == NULL) {
        return NULL;
    }
    # 检查 subtype 是否与 PyArray_Type 相同，确保其未修改维度
    if (subtype != &PyArray_Type):
        """
         * TODO: dtype 可能具有子数组，这会添加新的维度到 `ret`，
         *       这通常是可以接受的，但在这个分支中会导致错误。
         """
        # 如果 `ret` 的维度与 op_ndim 不同，或者 shape 与 PyArray_DIMS(ret) 的列表不匹配
        if (PyArray_NDIM(ret) != op_ndim or
                    !PyArray_CompareLists(shape, PyArray_DIMS(ret), op_ndim)):
            # 抛出运行时错误，指出迭代器的自动输出具有修改输出维度的数组子类型
            PyErr_SetString(PyExc_RuntimeError,
                    "Iterator automatic output has an array subtype "
                    "which changed the dimensions of the output")
            # 释放 ret 对象
            Py_DECREF(ret)
            # 返回空指针
            return NULL

    # 返回 ret 对象
    return ret;
}

static int
npyiter_allocate_arrays(NpyIter *iter,
                        npy_uint32 flags,
                        PyArray_Descr **op_dtype, PyTypeObject *subtype,
                        const npy_uint32 *op_flags, npyiter_opitflags *op_itflags,
                        int **op_axes)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);  // 获取迭代器的标志位信息
    int idim, ndim = NIT_NDIM(iter);  // 获取迭代器操作的维度数目
    int iop, nop = NIT_NOP(iter);  // 获取迭代器操作的操作数目

    int check_writemasked_reductions = 0;  // 初始化检查写入屏蔽约简操作标志

    NpyIter_BufferData *bufferdata = NULL;  // 初始化缓冲数据指针为NULL
    PyArrayObject **op = NIT_OPERANDS(iter);  // 获取迭代器的操作数对象数组

    if (itflags & NPY_ITFLAG_BUFFER) {  // 如果迭代器标志中包含缓冲标志
        bufferdata = NIT_BUFFERDATA(iter);  // 获取迭代器的缓冲数据
    }

    }

    }

    if (check_writemasked_reductions) {  // 如果需要检查写入屏蔽约简操作
        for (iop = 0; iop < nop; ++iop) {  // 遍历所有操作数目
            /*
             * 检查是否有需要验证的写入屏蔽约简操作数
             * 在所有步长填充完毕后进行验证。
             */
            if ((op_itflags[iop] &
                    (NPY_OP_ITFLAG_WRITEMASKED | NPY_OP_ITFLAG_REDUCE)) ==
                        (NPY_OP_ITFLAG_WRITEMASKED | NPY_OP_ITFLAG_REDUCE)) {
                /*
                 * 如果数组掩码比此约简写入屏蔽操作的维度还要“大”，
                 * 结果将是每个约简元素超过一个掩码值，这是无效的。
                 * 此函数提供了这种情况的验证。
                 */
                if (!check_mask_for_writemasked_reduction(iter, iop)) {
                    return 0;  // 如果验证失败，返回0
                }
            }
        }
    }

    return 1;  // 默认返回1，表示分配操作数组成功
}

/*
 * 输入的 __array_priority__ 属性决定了输出数组的子类型。
 * 此函数找到优先级最高的输入数组的子类型。
 */
static void
npyiter_get_priority_subtype(int nop, PyArrayObject **op,
                            const npyiter_opitflags *op_itflags,
                            double *subtype_priority,
                            PyTypeObject **subtype)
{
    int iop;

    for (iop = 0; iop < nop; ++iop) {  // 遍历所有操作数目
        if (op[iop] != NULL && op_itflags[iop] & NPY_OP_ITFLAG_READ) {
            double priority = PyArray_GetPriority((PyObject *)op[iop], 0.0);  // 获取数组对象的优先级
            if (priority > *subtype_priority) {  // 如果优先级高于当前记录的最高优先级
                *subtype_priority = priority;  // 更新最高优先级
                *subtype = Py_TYPE(op[iop]);  // 更新子类型对象
            }
        }
    }
}

static int
npyiter_allocate_transfer_functions(NpyIter *iter)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);  // 获取迭代器的标志位信息
    /*int ndim = NIT_NDIM(iter);*/  // 注释掉的代码行，不执行
    int iop = 0, nop = NIT_NOP(iter);  // 初始化操作数索引和操作数目

    npy_intp i;
    npyiter_opitflags *op_itflags = NIT_OPITFLAGS(iter);  // 获取迭代器的操作标志
    NpyIter_BufferData *bufferdata = NIT_BUFFERDATA(iter);  // 获取迭代器的缓冲数据
    NpyIter_AxisData *axisdata = NIT_AXISDATA(iter);  // 获取迭代器的轴数据
    PyArrayObject **op = NIT_OPERANDS(iter);  // 获取迭代器的操作数对象数组
    PyArray_Descr **op_dtype = NIT_DTYPES(iter);  // 获取迭代器的操作数数据类型数组
    npy_intp *strides = NAD_STRIDES(axisdata), op_stride;  // 获取轴数据的步长数组和操作步长
}
    # 将 bufferdata 转换为 NpyIter_TransferInfo 结构体指针
    NpyIter_TransferInfo *transferinfo = NBF_TRANSFERINFO(bufferdata);

    /* combined cast flags, the new cast flags for each cast: */
    # 初始化组合的转换标志为 PyArrayMethod_MINIMAL_FLAGS
    NPY_ARRAYMETHOD_FLAGS cflags = PyArrayMethod_MINIMAL_FLAGS;
    # 新的未使用标志
    NPY_ARRAYMETHOD_FLAGS nc_flags;

    }

    /* Store the combined transfer flags on the iterator */
    # 将组合的转换标志存储到迭代器的标志中
    NIT_ITFLAGS(iter) |= cflags << NPY_ITFLAG_TRANSFERFLAGS_SHIFT;
    # 断言迭代器的标志中存储的转换标志等于 cflags
    assert(NIT_ITFLAGS(iter) >> NPY_ITFLAG_TRANSFERFLAGS_SHIFT == cflags);

    /* If any of the dtype transfer functions needed the API, flag it. */
    # 如果转换标志中包含 NPY_METH_REQUIRES_PYAPI，设置迭代器需要 API 标志
    if (cflags & NPY_METH_REQUIRES_PYAPI) {
        NIT_ITFLAGS(iter) |= NPY_ITFLAG_NEEDSAPI;
    }

    # 返回成功标志
    return 1;
fail:
    # 循环遍历从 0 到 iop+1 的范围
    for (i = 0; i < iop+1; ++i) {
        # 释放 transferinfo[iop].read 的内存
        NPY_cast_info_xfree(&transferinfo[iop].read);
        # 释放 transferinfo[iop].write 的内存
        NPY_cast_info_xfree(&transferinfo[iop].write);
    }
    # 返回 0 表示函数执行失败
    return 0;
}
# 取消定义 NPY_ITERATOR_IMPLEMENTATION_CODE
#undef NPY_ITERATOR_IMPLEMENTATION_CODE
```