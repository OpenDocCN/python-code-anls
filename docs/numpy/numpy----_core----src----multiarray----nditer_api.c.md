# `.\numpy\numpy\_core\src\multiarray\nditer_api.c`

```py
/*
 * This file implements most of the main API functions of NumPy's nditer.
 * This excludes functions specialized using the templating system.
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
#include "templ_common.h"
#include "ctors.h"
#include "refcount.h"

/* Internal helper functions private to this file */

/*
 * Check and adjust the size of the reduction buffers.
 * This function is used internally by the iterator.
 *
 * Parameters:
 * - iter: Pointer to the NpyIter struct representing the iterator.
 * - count: The current count of elements being iterated.
 * - reduce_innersize: Pointer to the size of the inner reduction buffer.
 * - reduce_outerdim: Pointer to the size of the outer reduction buffer.
 *
 * Returns:
 * - The adjusted size of the reduction buffers.
 */
static npy_intp
npyiter_checkreducesize(NpyIter *iter, npy_intp count,
                        npy_intp *reduce_innersize,
                        npy_intp *reduce_outerdim);

/*NUMPY_API
 * Removes an axis from iteration. This requires that NPY_ITER_MULTI_INDEX
 * was set for iterator creation, and does not work if buffering is
 * enabled. This function also resets the iterator to its initial state.
 *
 * Returns NPY_SUCCEED or NPY_FAIL.
 */
NPY_NO_EXPORT int
NpyIter_RemoveAxis(NpyIter *iter, int axis)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);  // Get the iterator flags
    int idim, ndim = NIT_NDIM(iter);         // Get the number of dimensions
    int iop, nop = NIT_NOP(iter);            // Get the number of operands

    int xdim = 0;                            // Initialize xdim variable
    npy_int8 *perm = NIT_PERM(iter);         // Get the permutation array
    NpyIter_AxisData *axisdata_del = NIT_AXISDATA(iter), *axisdata;  // Get axis data
    npy_intp sizeof_axisdata = NIT_AXISDATA_SIZEOF(itflags, ndim, nop);  // Calculate size of axis data

    npy_intp *baseoffsets = NIT_BASEOFFSETS(iter);     // Get base offsets
    char **resetdataptr = NIT_RESETDATAPTR(iter);      // Get reset data pointers

    if (!(itflags & NPY_ITFLAG_HASMULTIINDEX)) {
        PyErr_SetString(PyExc_RuntimeError,
                "Iterator RemoveAxis may only be called "
                "if a multi-index is being tracked");
        return NPY_FAIL;
    }
    else if (itflags & NPY_ITFLAG_HASINDEX) {
        PyErr_SetString(PyExc_RuntimeError,
                "Iterator RemoveAxis may not be called on "
                "an index is being tracked");
        return NPY_FAIL;
    }
    else if (itflags & NPY_ITFLAG_BUFFER) {
        PyErr_SetString(PyExc_RuntimeError,
                "Iterator RemoveAxis may not be called on "
                "a buffered iterator");
        return NPY_FAIL;
    }
    else if (axis < 0 || axis >= ndim) {
        PyErr_SetString(PyExc_ValueError,
                "axis out of bounds in iterator RemoveAxis");
        return NPY_FAIL;
    }

    /* Reverse axis, since the iterator treats them that way */
    axis = ndim - 1 - axis;

    /* First find the axis in question */
    
    for (idim = 0; idim < ndim; ++idim) {
        /* 如果这是我们要找的轴，并且是正向迭代，则完成 */
        if (perm[idim] == axis) {
            xdim = idim;
            break;
        }
        /* 如果这是我们要找的轴，但是是反向迭代，则需要反转该轴 */
        else if (-1 - perm[idim] == axis) {
            npy_intp *strides = NAD_STRIDES(axisdata_del);
            npy_intp shape = NAD_SHAPE(axisdata_del), offset;

            xdim = idim;

            /*
             * 调整 baseoffsets 并将 resetbaseptr 重置回该轴的起始位置。
             */
            for (iop = 0; iop < nop; ++iop) {
                offset = (shape-1)*strides[iop];
                baseoffsets[iop] += offset;
                resetdataptr[iop] += offset;
            }
            break;
        }

        NIT_ADVANCE_AXISDATA(axisdata_del, 1);
    }

    if (idim == ndim) {
        PyErr_SetString(PyExc_RuntimeError,
                "iterator perm 中的内部错误");
        return NPY_FAIL;
    }

    /* 调整排列顺序 */
    for (idim = 0; idim < ndim-1; ++idim) {
        npy_int8 p = (idim < xdim) ? perm[idim] : perm[idim+1];
        if (p >= 0) {
            if (p > axis) {
                --p;
            }
        }
        else {
            if (p < -1-axis) {
                ++p;
            }
        }
        perm[idim] = p;
    }

    /* 将所有 axisdata 结构向前移动一位 */
    axisdata = NIT_INDEX_AXISDATA(axisdata_del, 1);
    memmove(axisdata_del, axisdata, (ndim-1-xdim)*sizeof_axisdata);

    /* 调整迭代器的大小并重置 iterend */
    NIT_ITERSIZE(iter) = 1;
    axisdata = NIT_AXISDATA(iter);
    for (idim = 0; idim < ndim-1; ++idim) {
        if (npy_mul_sizes_with_overflow(&NIT_ITERSIZE(iter),
                    NIT_ITERSIZE(iter), NAD_SHAPE(axisdata))) {
            NIT_ITERSIZE(iter) = -1;
            break;
        }
        NIT_ADVANCE_AXISDATA(axisdata, 1);
    }
    NIT_ITEREND(iter) = NIT_ITERSIZE(iter);

    /* 缩小迭代器 */
    NIT_NDIM(iter) = ndim - 1;
    /* 如果现在是 0 维，则填充单例维度 */
    if (ndim == 1) {
        npy_intp *strides = NAD_STRIDES(axisdata_del);
        NAD_SHAPE(axisdata_del) = 1;
        for (iop = 0; iop < nop; ++iop) {
            strides[iop] = 0;
        }
        NIT_ITFLAGS(iter) |= NPY_ITFLAG_ONEITERATION;
    }

    return NpyIter_Reset(iter, NULL);
/*NUMPY_API
 * Removes multi-index support from an iterator.
 *
 * Returns NPY_SUCCEED or NPY_FAIL.
 */
NPY_NO_EXPORT int
NpyIter_RemoveMultiIndex(NpyIter *iter)
{
    npy_uint32 itflags;

    /* Make sure the iterator is reset */
    // 调用 NpyIter_Reset 函数重置迭代器
    if (NpyIter_Reset(iter, NULL) != NPY_SUCCEED) {
        return NPY_FAIL;
    }

    itflags = NIT_ITFLAGS(iter);
    // 检查迭代器是否具有多重索引
    if (itflags & NPY_ITFLAG_HASMULTIINDEX) {
        // 检查迭代器大小是否小于零，如果是则设置异常并返回失败
        if (NIT_ITERSIZE(iter) < 0) {
            PyErr_SetString(PyExc_ValueError, "iterator is too large");
            return NPY_FAIL;
        }

        // 清除迭代器的多重索引标志位
        NIT_ITFLAGS(iter) = itflags & ~NPY_ITFLAG_HASMULTIINDEX;
        // 执行迭代器轴数据合并
        npyiter_coalesce_axes(iter);
    }

    return NPY_SUCCEED;
}

/*NUMPY_API
 * Removes the inner loop handling (so HasExternalLoop returns true)
 */
NPY_NO_EXPORT int
NpyIter_EnableExternalLoop(NpyIter *iter)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    /*int ndim = NIT_NDIM(iter);*/
    int nop = NIT_NOP(iter);

    /* Check conditions under which this can be done */
    // 检查是否可以启用外部循环
    if (itflags & (NPY_ITFLAG_HASINDEX | NPY_ITFLAG_HASMULTIINDEX)) {
        PyErr_SetString(PyExc_ValueError,
                        "Iterator flag EXTERNAL_LOOP cannot be used "
                        "if an index or multi-index is being tracked");
        return NPY_FAIL;
    }
    if ((itflags & (NPY_ITFLAG_BUFFER | NPY_ITFLAG_RANGE | NPY_ITFLAG_EXLOOP))
        == (NPY_ITFLAG_RANGE | NPY_ITFLAG_EXLOOP)) {
        PyErr_SetString(PyExc_ValueError,
                        "Iterator flag EXTERNAL_LOOP cannot be used "
                        "with ranged iteration unless buffering is also enabled");
        return NPY_FAIL;
    }
    /* Set the flag */
    // 设置迭代器的外部循环标志位
    if (!(itflags & NPY_ITFLAG_EXLOOP)) {
        itflags |= NPY_ITFLAG_EXLOOP;
        NIT_ITFLAGS(iter) = itflags;

        /*
         * Check whether we can apply the single iteration
         * optimization to the iternext function.
         */
        // 如果不使用缓冲区，检查是否可以应用单次迭代优化
        if (!(itflags & NPY_ITFLAG_BUFFER)) {
            NpyIter_AxisData *axisdata = NIT_AXISDATA(iter);
            if (NIT_ITERSIZE(iter) == NAD_SHAPE(axisdata)) {
                NIT_ITFLAGS(iter) |= NPY_ITFLAG_ONEITERATION;
            }
        }
    }

    /* Reset the iterator */
    // 重置迭代器到初始状态
    return NpyIter_Reset(iter, NULL);
}


static char *_reset_cast_error = (
        "Iterator reset failed due to a casting failure. "
        "This error is set as a Python error.");

/*NUMPY_API
 * Resets the iterator to its initial state
 *
 * The use of errmsg is discouraged, it cannot be guaranteed that the GIL
 * will not be grabbed on casting errors even when this is passed.
 *
 * If errmsg is non-NULL, it should point to a variable which will
 * receive the error message, and no Python exception will be set.
 * This is so that the function can be called from code not holding
 * the GIL. Note that cast errors may still lead to the GIL being
 * grabbed temporarily.
 */
NPY_NO_EXPORT int
NpyIter_Reset(NpyIter *iter, char **errmsg)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    /*int ndim = NIT_NDIM(iter);*/
    // ...
    # 根据迭代器获取操作的 NOP 值
    int nop = NIT_NOP(iter);

    # 如果迭代器标志包含 NPY_ITFLAG_BUFFER
    if (itflags&NPY_ITFLAG_BUFFER) {
        # 定义缓冲区数据结构指针
        NpyIter_BufferData *bufferdata;

        # 如果延迟了缓冲区分配，现在进行分配
        if (itflags&NPY_ITFLAG_DELAYBUF) {
            # 如果无法成功分配缓冲区，设置错误消息并返回失败
            if (!npyiter_allocate_buffers(iter, errmsg)) {
                if (errmsg != NULL) {
                    *errmsg = _reset_cast_error;
                }
                return NPY_FAIL;
            }
            # 清除延迟缓冲区标志
            NIT_ITFLAGS(iter) &= ~NPY_ITFLAG_DELAYBUF;
        }
        else {
            /*
             * 如果迭代索引已经正确，不需要执行任何操作
             * （且之前未发生类型转换错误）。
             */
            # 获取缓冲区数据结构指针
            bufferdata = NIT_BUFFERDATA(iter);
            # 如果迭代索引等于迭代起始值，并且缓冲区迭代结束位置小于等于迭代结束位置，并且缓冲区大小大于零
            if (NIT_ITERINDEX(iter) == NIT_ITERSTART(iter) &&
                    NBF_BUFITEREND(bufferdata) <= NIT_ITEREND(iter) &&
                    NBF_SIZE(bufferdata) > 0) {
                return NPY_SUCCEED;
            }
            # 如果从缓冲区复制数据失败，设置错误消息并返回失败
            if (npyiter_copy_from_buffers(iter) < 0) {
                if (errmsg != NULL) {
                    *errmsg = _reset_cast_error;
                }
                return NPY_FAIL;
            }
        }
    }

    # 将迭代器移动到指定的迭代索引位置
    npyiter_goto_iterindex(iter, NIT_ITERSTART(iter));

    # 如果迭代器标志包含 NPY_ITFLAG_BUFFER
    if (itflags&NPY_ITFLAG_BUFFER) {
        # 准备下一个缓冲区并设置迭代结束位置和大小
        if (npyiter_copy_to_buffers(iter, NULL) < 0) {
            # 如果复制数据到缓冲区失败，设置错误消息并返回失败
            if (errmsg != NULL) {
                *errmsg = _reset_cast_error;
            }
            return NPY_FAIL;
        }
    }

    # 返回成功状态
    return NPY_SUCCEED;
/*NUMPY_API
 * Resets the iterator to its initial state, with new base data pointers.
 * This function requires great caution.
 *
 * If errmsg is non-NULL, it should point to a variable which will
 * receive the error message, and no Python exception will be set.
 * This is so that the function can be called from code not holding
 * the GIL. Note that cast errors may still lead to the GIL being
 * grabbed temporarily.
 */
NPY_NO_EXPORT int
NpyIter_ResetBasePointers(NpyIter *iter, char **baseptrs, char **errmsg)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    /* 获取迭代器的标志位 */
    /*int ndim = NIT_NDIM(iter);*/
    /* 获取迭代器的维度 */
    int iop, nop = NIT_NOP(iter);
    /* 初始化操作符数目，并获取迭代器的操作符数目 */

    char **resetdataptr = NIT_RESETDATAPTR(iter);
    /* 获取迭代器重置数据指针 */
    npy_intp *baseoffsets = NIT_BASEOFFSETS(iter);
    /* 获取迭代器的基本偏移量 */

    if (itflags&NPY_ITFLAG_BUFFER) {
        /* 如果设置了缓冲区标志 */
        /* 如果缓冲区分配被延迟，现在进行分配 */
        if (itflags&NPY_ITFLAG_DELAYBUF) {
            if (!npyiter_allocate_buffers(iter, errmsg)) {
                /* 如果缓冲区分配失败，则返回失败状态 */
                return NPY_FAIL;
            }
            NIT_ITFLAGS(iter) &= ~NPY_ITFLAG_DELAYBUF;
        }
        else {
            if (npyiter_copy_from_buffers(iter) < 0) {
                /* 如果从缓冲区复制数据失败 */
                if (errmsg != NULL) {
                    *errmsg = _reset_cast_error;
                }
                return NPY_FAIL;
            }
        }
    }

    /* 更新重置数据指针 */
    for (iop = 0; iop < nop; ++iop) {
        resetdataptr[iop] = baseptrs[iop] + baseoffsets[iop];
    }

    /* 将迭代器移动到起始迭代位置 */
    npyiter_goto_iterindex(iter, NIT_ITERSTART(iter));

    if (itflags&NPY_ITFLAG_BUFFER) {
        /* 如果设置了缓冲区标志 */
        /* 准备下一组缓冲区并设置迭代结束/大小 */
        if (npyiter_copy_to_buffers(iter, NULL) < 0) {
            /* 如果复制数据到缓冲区失败 */
            if (errmsg != NULL) {
                *errmsg = _reset_cast_error;
            }
            return NPY_FAIL;
        }
    }

    /* 返回成功状态 */
    return NPY_SUCCEED;
}

/*NUMPY_API
 * Resets the iterator to a new iterator index range
 *
 * If errmsg is non-NULL, it should point to a variable which will
 * receive the error message, and no Python exception will be set.
 * This is so that the function can be called from code not holding
 * the GIL. Note that cast errors may still lead to the GIL being
 * grabbed temporarily.
 */
NPY_NO_EXPORT int
NpyIter_ResetToIterIndexRange(NpyIter *iter,
                              npy_intp istart, npy_intp iend, char **errmsg)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    /* 获取迭代器的标志位 */
    /*int ndim = NIT_NDIM(iter);*/
    /* 获取迭代器的维度 */
    /*int nop = NIT_NOP(iter);*/

    if (!(itflags&NPY_ITFLAG_RANGE)) {
        /* 如果没有请求迭代范围 */
        if (errmsg == NULL) {
            PyErr_SetString(PyExc_ValueError,
                    "Cannot call ResetToIterIndexRange on an iterator without "
                    "requesting ranged iteration support in the constructor");
        }
        else {
            *errmsg = "Cannot call ResetToIterIndexRange on an iterator "
                      "without requesting ranged iteration support in the "
                    "constructor";
        }
        return NPY_FAIL;
    }
    # 检查迭代器的起始位置和结束位置是否超出有效范围
    if (istart < 0 || iend > NIT_ITERSIZE(iter)) {
        # 如果迭代器的大小小于零，表示迭代器过大，抛出异常
        if (NIT_ITERSIZE(iter) < 0) {
            # 如果错误消息为 NULL，则设置异常字符串到 ValueError
            if (errmsg == NULL) {
                PyErr_SetString(PyExc_ValueError, "iterator is too large");
            }
            # 否则，设置错误消息字符串
            else {
                *errmsg = "iterator is too large";
            }
            # 返回失败标志
            return NPY_FAIL;
        }
        # 如果超出范围，根据错误消息设置异常字符串
        if (errmsg == NULL) {
            PyErr_Format(PyExc_ValueError,
                    "Out-of-bounds range [%" NPY_INTP_FMT ", %" NPY_INTP_FMT ") passed to "
                    "ResetToIterIndexRange", istart, iend);
        }
        # 否则，设置错误消息字符串
        else {
            *errmsg = "Out-of-bounds range passed to ResetToIterIndexRange";
        }
        # 返回失败标志
        return NPY_FAIL;
    }
    # 如果结束位置小于起始位置，表示范围无效，根据错误消息设置异常字符串
    else if (iend < istart) {
        if (errmsg == NULL) {
            PyErr_Format(PyExc_ValueError,
                    "Invalid range [%" NPY_INTP_FMT ", %" NPY_INTP_FMT ") passed to ResetToIterIndexRange",
                    istart, iend);
        }
        # 否则，设置错误消息字符串
        else {
            *errmsg = "Invalid range passed to ResetToIterIndexRange";
        }
        # 返回失败标志
        return NPY_FAIL;
    }

    # 将迭代器的起始位置和结束位置设置为给定值
    NIT_ITERSTART(iter) = istart;
    NIT_ITEREND(iter) = iend;

    # 调用 NpyIter_Reset 函数重置迭代器，并传入可能存在的错误消息
    return NpyIter_Reset(iter, errmsg);
/*NUMPY_API
 * Sets the iterator to the specified multi-index, which must have the
 * correct number of entries for 'ndim'.  It is only valid
 * when NPY_ITER_MULTI_INDEX was passed to the constructor.  This operation
 * fails if the multi-index is out of bounds.
 *
 * Returns NPY_SUCCEED on success, NPY_FAIL on failure.
 */
NPY_NO_EXPORT int
NpyIter_GotoMultiIndex(NpyIter *iter, npy_intp const *multi_index)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);   // 获取迭代器的标志位
    int idim, ndim = NIT_NDIM(iter);          // 获取迭代器的维度信息
    int nop = NIT_NOP(iter);                  // 获取迭代器的操作数信息

    npy_intp iterindex, factor;               // 声明迭代器索引和因子
    NpyIter_AxisData *axisdata;               // 声明轴数据指针
    npy_intp sizeof_axisdata;                 // 声明轴数据大小
    npy_int8 *perm;                           // 声明排列数组指针

    // 检查迭代器是否支持多索引模式
    if (!(itflags & NPY_ITFLAG_HASMULTIINDEX)) {
        PyErr_SetString(PyExc_ValueError,
                "Cannot call GotoMultiIndex on an iterator without "
                "requesting a multi-index in the constructor");
        return NPY_FAIL;
    }

    // 检查迭代器是否是缓冲模式
    if (itflags & NPY_ITFLAG_BUFFER) {
        PyErr_SetString(PyExc_ValueError,
                "Cannot call GotoMultiIndex on an iterator which "
                "is buffered");
        return NPY_FAIL;
    }

    // 检查迭代器是否有外部循环标志
    if (itflags & NPY_ITFLAG_EXLOOP) {
        PyErr_SetString(PyExc_ValueError,
                "Cannot call GotoMultiIndex on an iterator which "
                "has the flag EXTERNAL_LOOP");
        return NPY_FAIL;
    }

    perm = NIT_PERM(iter);                    // 获取迭代器的排列数组
    axisdata = NIT_AXISDATA(iter);            // 获取迭代器的轴数据
    sizeof_axisdata = NIT_AXISDATA_SIZEOF(itflags, ndim, nop);  // 获取轴数据的大小

    /* Compute the iterindex corresponding to the multi-index */
    iterindex = 0;                           // 初始化迭代器索引为0
    factor = 1;                              // 初始化因子为1
    for (idim = 0; idim < ndim; ++idim) {    // 遍历迭代器的维度
        npy_int8 p = perm[idim];             // 获取当前维度的排列信息
        npy_intp i, shape;                   // 声明当前维度的索引和形状

        shape = NAD_SHAPE(axisdata);         // 获取当前轴数据的形状
        if (p < 0) {
            /* If the perm entry is negative, reverse the index */
            // 如果排列条目为负数，则反转索引
            i = shape - multi_index[ndim + p] - 1;
        } else {
            i = multi_index[ndim - p - 1];
        }

        /* Bounds-check this index */
        // 检查索引是否在合法范围内
        if (i >= 0 && i < shape) {
            iterindex += factor * i;         // 更新迭代器索引
            factor *= shape;                 // 更新因子
        } else {
            PyErr_SetString(PyExc_IndexError,
                    "Iterator GotoMultiIndex called with an out-of-bounds "
                    "multi-index");
            return NPY_FAIL;
        }

        NIT_ADVANCE_AXISDATA(axisdata, 1);   // 推进轴数据指针
    }

    // 检查迭代器索引是否在有效范围内
    if (iterindex < NIT_ITERSTART(iter) || iterindex >= NIT_ITEREND(iter)) {
        if (NIT_ITERSIZE(iter) < 0) {
            PyErr_SetString(PyExc_ValueError, "iterator is too large");
            return NPY_FAIL;
        }
        PyErr_SetString(PyExc_IndexError,
                "Iterator GotoMultiIndex called with a multi-index outside the "
                "restricted iteration range");
        return NPY_FAIL;
    }

    npyiter_goto_iterindex(iter, iterindex);  // 调用具体的迭代器索引跳转函数

    return NPY_SUCCEED;                      // 返回成功状态
}
NPY_NO_EXPORT int
NpyIter_GotoIndex(NpyIter *iter, npy_intp flat_index)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    int idim, ndim = NIT_NDIM(iter);
    int nop = NIT_NOP(iter);

    npy_intp iterindex, factor;
    NpyIter_AxisData *axisdata;
    npy_intp sizeof_axisdata;

    // 检查是否在构造函数中请求了 C 或 Fortran 索引
    if (!(itflags&NPY_ITFLAG_HASINDEX)) {
        PyErr_SetString(PyExc_ValueError,
                "Cannot call GotoIndex on an iterator without "
                "requesting a C or Fortran index in the constructor");
        return NPY_FAIL;
    }

    // 检查迭代器是否是缓冲的
    if (itflags&NPY_ITFLAG_BUFFER) {
        PyErr_SetString(PyExc_ValueError,
                "Cannot call GotoIndex on an iterator which "
                "is buffered");
        return NPY_FAIL;
    }

    // 检查迭代器是否带有 EXTERNAL_LOOP 标志
    if (itflags&NPY_ITFLAG_EXLOOP) {
        PyErr_SetString(PyExc_ValueError,
                "Cannot call GotoIndex on an iterator which "
                "has the flag EXTERNAL_LOOP");
        return NPY_FAIL;
    }

    // 检查 flat_index 是否在有效范围内
    if (flat_index < 0 || flat_index >= NIT_ITERSIZE(iter)) {
        PyErr_SetString(PyExc_IndexError,
                "Iterator GotoIndex called with an out-of-bounds "
                "index");
        return NPY_FAIL;
    }

    axisdata = NIT_AXISDATA(iter);
    sizeof_axisdata = NIT_AXISDATA_SIZEOF(itflags, ndim, nop);

    /* 计算对应于 flat_index 的 iterindex */
    iterindex = 0;
    factor = 1;
    for (idim = 0; idim < ndim; ++idim) {
        npy_intp i, shape, iterstride;

        iterstride = NAD_STRIDES(axisdata)[nop];
        shape = NAD_SHAPE(axisdata);

        /* 从 flat_index 中提取索引 */
        if (iterstride == 0) {
            i = 0;
        }
        else if (iterstride < 0) {
            i = shape - (flat_index/(-iterstride))%shape - 1;
        }
        else {
            i = (flat_index/iterstride)%shape;
        }

        /* 将其对 iterindex 的贡献加到 iterindex 中 */
        iterindex += factor * i;
        factor *= shape;

        NIT_ADVANCE_AXISDATA(axisdata, 1);
    }


    // 检查 iterindex 是否在受限迭代范围之外
    if (iterindex < NIT_ITERSTART(iter) || iterindex >= NIT_ITEREND(iter)) {
        PyErr_SetString(PyExc_IndexError,
                "Iterator GotoIndex called with an index outside the "
                "restricted iteration range.");
        return NPY_FAIL;
    }

    // 跳转到 iterindex 处的迭代位置
    npyiter_goto_iterindex(iter, iterindex);

    return NPY_SUCCEED;
}

/*NUMPY_API
 * Sets the iterator position to the specified iterindex,
 * which matches the iteration order of the iterator.
 *
 * Returns NPY_SUCCEED on success, NPY_FAIL on failure.
 */
NPY_NO_EXPORT int
NpyIter_GotoIterIndex(NpyIter *iter, npy_intp iterindex)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    /*int ndim = NIT_NDIM(iter);*/
    int iop, nop = NIT_NOP(iter);

    // 检查迭代器是否带有 EXTERNAL_LOOP 标志
    if (itflags&NPY_ITFLAG_EXLOOP) {
        PyErr_SetString(PyExc_ValueError,
                "Cannot call GotoIterIndex on an iterator which "
                "has the flag EXTERNAL_LOOP");
        return NPY_FAIL;
    }
    # 检查 iterindex 是否在指定的迭代范围之外
    if (iterindex < NIT_ITERSTART(iter) || iterindex >= NIT_ITEREND(iter)) {
        # 如果迭代器大小为负数，抛出值错误异常
        if (NIT_ITERSIZE(iter) < 0) {
            PyErr_SetString(PyExc_ValueError, "iterator is too large");
            return NPY_FAIL;
        }
        # 抛出索引错误异常，说明 iterindex 超出迭代范围
        PyErr_SetString(PyExc_IndexError,
                "Iterator GotoIterIndex called with an iterindex outside the "
                "iteration range.");
        return NPY_FAIL;
    }

    # 如果设置了 NPY_ITFLAG_BUFFER 标志
    if (itflags&NPY_ITFLAG_BUFFER) {
        # 获取缓冲区数据结构
        NpyIter_BufferData *bufferdata = NIT_BUFFERDATA(iter);
        npy_intp bufiterend, size;

        # 获取缓冲区大小和缓冲区迭代结束位置
        size = NBF_SIZE(bufferdata);
        bufiterend = NBF_BUFITEREND(bufferdata);
        
        /* Check if the new iterindex is already within the buffer */
        # 检查新的 iterindex 是否已经在缓冲区内
        if (!(itflags&NPY_ITFLAG_REDUCE) && iterindex < bufiterend &&
                                        iterindex >= bufiterend - size) {
            npy_intp *strides, delta;
            char **ptrs;

            # 获取缓冲区的步幅和指针数组
            strides = NBF_STRIDES(bufferdata);
            ptrs = NBF_PTRS(bufferdata);
            # 计算 iterindex 和当前迭代器的差值，调整指针数组的位置
            delta = iterindex - NIT_ITERINDEX(iter);

            for (iop = 0; iop < nop; ++iop) {
                # 根据步幅调整指针数组的位置
                ptrs[iop] += delta * strides[iop];
            }

            # 更新迭代器的当前索引为 iterindex
            NIT_ITERINDEX(iter) = iterindex;
        }
        /* Start the buffer at the provided iterindex */
        else {
            /* Write back to the arrays */
            # 将缓冲区的数据写回到数组中
            if (npyiter_copy_from_buffers(iter) < 0) {
                return NPY_FAIL;
            }

            # 将迭代器的当前索引移动到 iterindex
            npyiter_goto_iterindex(iter, iterindex);

            /* Prepare the next buffers and set iterend/size */
            # 准备下一轮的缓冲区数据，并设置迭代器的结束位置和大小
            if (npyiter_copy_to_buffers(iter, NULL) < 0) {
                return NPY_FAIL;
            }
        }
    }
    else {
        # 如果没有设置 NPY_ITFLAG_BUFFER 标志，直接将迭代器移动到 iterindex
        npyiter_goto_iterindex(iter, iterindex);
    }

    # 执行成功，返回成功标志
    return NPY_SUCCEED;
/*NUMPY_API
 * 获取当前迭代的索引
 */
NPY_NO_EXPORT npy_intp
NpyIter_GetIterIndex(NpyIter *iter)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);  // 获取迭代器的标志位
    int idim, ndim = NIT_NDIM(iter);  // 获取迭代器的维度数和操作数的数量
    int nop = NIT_NOP(iter);  // 获取迭代器的操作数数量

    /* 只有在设置了 NPY_ITER_RANGED 或 NPY_ITER_BUFFERED 标志时才使用 iterindex */
    if (itflags & (NPY_ITFLAG_RANGE | NPY_ITFLAG_BUFFER)) {
        return NIT_ITERINDEX(iter);  // 返回迭代器的当前索引
    }
    else {
        npy_intp iterindex;
        NpyIter_AxisData *axisdata;
        npy_intp sizeof_axisdata;

        iterindex = 0;
        if (ndim == 0) {
            return 0;  // 如果维度数为0，直接返回0
        }
        sizeof_axisdata = NIT_AXISDATA_SIZEOF(itflags, ndim, nop);  // 计算轴数据结构的大小
        axisdata = NIT_INDEX_AXISDATA(NIT_AXISDATA(iter), ndim - 1);  // 获取最后一个轴数据

        for (idim = ndim - 2; idim >= 0; --idim) {
            iterindex += NAD_INDEX(axisdata);  // 累加当前轴的索引值
            NIT_ADVANCE_AXISDATA(axisdata, -1);  // 将轴数据向前移动一个元素
            iterindex *= NAD_SHAPE(axisdata);  // 乘以当前轴的形状
        }
        iterindex += NAD_INDEX(axisdata);  // 加上最后一个轴的索引

        return iterindex;  // 返回计算得到的迭代索引
    }
}

/*NUMPY_API
 * 检查缓冲区分配是否被延迟
 */
NPY_NO_EXPORT npy_bool
NpyIter_HasDelayedBufAlloc(NpyIter *iter)
{
    return (NIT_ITFLAGS(iter) & NPY_ITFLAG_DELAYBUF) != 0;  // 检查迭代器是否设置了延迟缓冲区分配标志
}

/*NUMPY_API
 * 检查迭代器是否处理内部循环
 */
NPY_NO_EXPORT npy_bool
NpyIter_HasExternalLoop(NpyIter *iter)
{
    return (NIT_ITFLAGS(iter) & NPY_ITFLAG_EXLOOP) != 0;  // 检查迭代器是否设置了外部循环处理标志
}

/*NUMPY_API
 * 检查迭代器是否跟踪多重索引
 */
NPY_NO_EXPORT npy_bool
NpyIter_HasMultiIndex(NpyIter *iter)
{
    return (NIT_ITFLAGS(iter) & NPY_ITFLAG_HASMULTIINDEX) != 0;  // 检查迭代器是否设置了多重索引跟踪标志
}

/*NUMPY_API
 * 检查迭代器是否跟踪索引
 */
NPY_NO_EXPORT npy_bool
NpyIter_HasIndex(NpyIter *iter)
{
    return (NIT_ITFLAGS(iter) & NPY_ITFLAG_HASINDEX) != 0;  // 检查迭代器是否设置了索引跟踪标志
}

/*NUMPY_API
 * 检查指定的减少操作数在迭代器指向的元素中是否第一次被访问。
 * 对于减少操作数和禁用缓冲区的情况下，该函数会给出一个合理的答案。
 * 对于有缓冲区的非减少操作数，答案可能不正确。
 *
 * 此函数仅用于 EXTERNAL_LOOP 模式，并且在未启用该模式时，结果可能不正确。
 *
 * 如果此函数返回 true，则调用者还应检查操作数的内部循环步长，
 * 因为如果该步长为 0，则仅访问最内层外部循环的第一个元素。
 *
 * 警告：出于性能原因，'iop' 没有进行边界检查，
 *       不确认 'iop' 实际上是减少操作数，也不确认启用了 EXTERNAL_LOOP 模式。
 *       这些检查应由调用者在任何内部循环之外进行。
 */
NPY_NO_EXPORT npy_bool
NpyIter_IsFirstVisit(NpyIter *iter, int iop)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);  // 获取迭代器的标志位
    int idim, ndim = NIT_NDIM(iter);  // 获取迭代器的维度数

    /* 返回迭代器是否跟踪减少操作数 'iop' 的第一次访问状态 */
    return (itflags & NPY_ITFLAG_FIRST Visit(iop)) != 0;
}
    // 计算 NIT_NOP(iter) 的值，存储在 nop 变量中
    int nop = NIT_NOP(iter);

    // 定义指向轴数据的指针及其大小
    NpyIter_AxisData *axisdata;
    npy_intp sizeof_axisdata;

    // 计算轴数据结构的大小并赋值给 sizeof_axisdata
    sizeof_axisdata = NIT_AXISDATA_SIZEOF(itflags, ndim, nop);

    // 获取迭代器的轴数据指针并赋值给 axisdata
    axisdata = NIT_AXISDATA(iter);

    // 遍历数组的每个维度
    for (idim = 0; idim < ndim; ++idim) {
        // 获取当前轴的坐标
        npy_intp coord = NAD_INDEX(axisdata);
        // 获取当前轴的步幅
        npy_intp stride = NAD_STRIDES(axisdata)[iop];

        /*
         * 如果这是一个归约维度，并且坐标不在起始位置，
         * 则肯定不是第一次访问
         */
        if (stride == 0 && coord != 0) {
            return 0;
        }

        // 将 axisdata 向前推进一个位置
        NIT_ADVANCE_AXISDATA(axisdata, 1);
    }

    /*
     * 在归约缓冲模式下，迭代器数据结构的缓冲区部分中有一个双重循环正在跟踪。
     * 我们只需要检查这两级循环的外层级别，
     * 因为要求启用 EXTERNAL_LOOP。
     */
    if (itflags & NPY_ITFLAG_BUFFER) {
        // 获取缓冲区数据指针
        NpyIter_BufferData *bufferdata = NIT_BUFFERDATA(iter);
        /* 外部归约循环 */
        if (NBF_REDUCE_POS(bufferdata) != 0 &&
                NBF_REDUCE_OUTERSTRIDES(bufferdata)[iop] == 0) {
            return 0;
        }
    }

    // 如果所有条件都通过，则返回 1
    return 1;
/*NUMPY_API
 * Whether the iteration could be done with no buffering.
 */
NPY_NO_EXPORT npy_bool
NpyIter_RequiresBuffering(NpyIter *iter)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    // 检查迭代器标志是否包含缓冲标志
    if (!(itflags&NPY_ITFLAG_BUFFER)) {
        return 0;
    }

    npyiter_opitflags *op_itflags;
    op_itflags = NIT_OPITFLAGS(iter);

    // 如果任何操作数需要类型转换，强制使用缓冲
    int iop, nop = NIT_NOP(iter);
    for (iop = 0; iop < nop; ++iop) {
        if (op_itflags[iop]&NPY_OP_ITFLAG_CAST) {
            return 1;
        }
    }

    // 不需要缓冲
    return 0;
}

/*NUMPY_API
 * Whether the iteration loop, and in particular the iternext()
 * function, needs API access.  If this is true, the GIL must
 * be retained while iterating.
 *
 * NOTE: Internally (currently), `NpyIter_GetTransferFlags` will
 *       additionally provide information on whether floating point errors
 *       may be given during casts.  The flags only require the API use
 *       necessary for buffering though.  So an iterate which does not require
 *       buffering may indicate `NpyIter_IterationNeedsAPI`, but not include
 *       the flag in `NpyIter_GetTransferFlags`.
 */
NPY_NO_EXPORT npy_bool
NpyIter_IterationNeedsAPI(NpyIter *iter)
{
    // 检查迭代是否需要 API 访问
    return (NIT_ITFLAGS(iter)&NPY_ITFLAG_NEEDSAPI) != 0;
}

/*
 * Fetch the ArrayMethod (runtime) flags for all "transfer functions' (i.e.
 * copy to buffer/casts).
 *
 * TODO: This should be public API, but that only makes sense when the
 *       ArrayMethod API is made public.
 */
NPY_NO_EXPORT int
NpyIter_GetTransferFlags(NpyIter *iter)
{
    // 获取迭代器的传输标志
    return NIT_ITFLAGS(iter) >> NPY_ITFLAG_TRANSFERFLAGS_SHIFT;
}

/*NUMPY_API
 * Gets the number of dimensions being iterated
 */
NPY_NO_EXPORT int
NpyIter_GetNDim(NpyIter *iter)
{
    // 获取正在迭代的维度数
    return NIT_NDIM(iter);
}

/*NUMPY_API
 * Gets the number of operands being iterated
 */
NPY_NO_EXPORT int
NpyIter_GetNOp(NpyIter *iter)
{
    // 获取正在迭代的操作数个数
    return NIT_NOP(iter);
}

/*NUMPY_API
 * Gets the number of elements being iterated
 */
NPY_NO_EXPORT npy_intp
NpyIter_GetIterSize(NpyIter *iter)
{
    // 获取正在迭代的元素个数
    return NIT_ITERSIZE(iter);
}

/*NUMPY_API
 * Whether the iterator is buffered
 */
NPY_NO_EXPORT npy_bool
NpyIter_IsBuffered(NpyIter *iter)
{
    // 检查迭代器是否使用了缓冲
    return (NIT_ITFLAGS(iter)&NPY_ITFLAG_BUFFER) != 0;
}

/*NUMPY_API
 * Whether the inner loop can grow if buffering is unneeded
 */
NPY_NO_EXPORT npy_bool
NpyIter_IsGrowInner(NpyIter *iter)
{
    // 检查内部循环是否可以增长，即使不需要缓冲
    return (NIT_ITFLAGS(iter)&NPY_ITFLAG_GROWINNER) != 0;
}

/*NUMPY_API
 * Gets the size of the buffer, or 0 if buffering is not enabled
 */
NPY_NO_EXPORT npy_intp
NpyIter_GetBufferSize(NpyIter *iter)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    // 检查迭代器是否启用了缓冲
    if (itflags&NPY_ITFLAG_BUFFER) {
        NpyIter_BufferData *bufferdata = NIT_BUFFERDATA(iter);
        return NBF_BUFFERSIZE(bufferdata);
    }
    else {
        return 0;
    }
}
/*NUMPY_API
 * 获取迭代器正在跟踪多索引时的广播形状，否则获取按 Fortran 顺序排列的迭代形状
 * （最快变化的索引在前）。
 *
 * 当未启用多索引时返回 Fortran 顺序的原因是，这提供了直接查看迭代器如何遍历
 * n 维空间的视图。迭代器按最快到最慢的顺序组织其内存，并且当启用多索引时，
 * 使用排列来恢复原始顺序。
 *
 * 返回 NPY_SUCCEED 或 NPY_FAIL。
 */
NPY_NO_EXPORT int
NpyIter_GetShape(NpyIter *iter, npy_intp *outshape)
{
    // 获取迭代器的标志、维度数和操作数
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    int ndim = NIT_NDIM(iter);
    int nop = NIT_NOP(iter);

    int idim, sizeof_axisdata;
    NpyIter_AxisData *axisdata;
    npy_int8 *perm;

    // 获取迭代器的轴数据和轴数据的大小
    axisdata = NIT_AXISDATA(iter);
    sizeof_axisdata = NIT_AXISDATA_SIZEOF(itflags, ndim, nop);

    // 如果迭代器具有多索引标志
    if (itflags & NPY_ITFLAG_HASMULTIINDEX) {
        perm = NIT_PERM(iter);
        // 遍历每个维度
        for(idim = 0; idim < ndim; ++idim) {
            // 使用轴的逆排列恢复轴的原始顺序
            int axis = npyiter_undo_iter_axis_perm(idim, ndim, perm, NULL);
            // 获取当前轴的形状并赋给输出形状数组
            outshape[axis] = NAD_SHAPE(axisdata);

            // 推进轴数据以准备处理下一个轴
            NIT_ADVANCE_AXISDATA(axisdata, 1);
        }
    }
    // 否则，处理没有多索引的情况
    else {
        // 遍历每个维度
        for(idim = 0; idim < ndim; ++idim) {
            // 获取当前轴的形状并赋给输出形状数组
            outshape[idim] = NAD_SHAPE(axisdata);
            // 推进轴数据以准备处理下一个轴
            NIT_ADVANCE_AXISDATA(axisdata, 1);
        }
    }

    // 返回操作成功的标志
    return NPY_SUCCEED;
}
    # 获取迭代器的标志位，表示迭代器的属性
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    # 定义循环中使用的变量：当前维度和总维度数
    int idim, ndim = NIT_NDIM(iter);
    # 获取迭代器的操作数
    int nop = NIT_NOP(iter);
    
    # 声明变量：轴数据大小的整型指针和轴数据的指针
    npy_intp sizeof_axisdata;
    NpyIter_AxisData *axisdata;
    npy_int8 *perm;
    
    # 检查迭代器是否具有多索引，若没有则抛出运行时错误并返回失败标志
    if (!(itflags&NPY_ITFLAG_HASMULTIINDEX)) {
        PyErr_SetString(PyExc_RuntimeError,
                "Iterator CreateCompatibleStrides may only be called "
                "if a multi-index is being tracked");
        return NPY_FAIL;
    }
    
    # 获取迭代器的轴数据和其大小
    axisdata = NIT_AXISDATA(iter);
    sizeof_axisdata = NIT_AXISDATA_SIZEOF(itflags, ndim, nop);
    
    # 获取迭代器的排列顺序数组
    perm = NIT_PERM(iter);
    
    # 遍历每一个维度
    for(idim = 0; idim < ndim; ++idim) {
        # 声明布尔变量和轴号，反转标志用于指示是否反转轴的步长
        npy_bool flipped;
        npy_int8 axis = npyiter_undo_iter_axis_perm(idim, ndim, perm, &flipped);
        # 如果发生了轴反转，则抛出运行时错误并返回失败标志
        if (flipped) {
            PyErr_SetString(PyExc_RuntimeError,
                    "Iterator CreateCompatibleStrides may only be called "
                    "if DONT_NEGATE_STRIDES was used to prevent reverse "
                    "iteration of an axis");
            return NPY_FAIL;
        }
        else {
            # 否则，设置输出步长数组中对应轴的步长为itemsize
            outstrides[axis] = itemsize;
        }
    
        # 计算当前轴的itemsize，并推进轴数据以便处理下一个轴
        itemsize *= NAD_SHAPE(axisdata);
        NIT_ADVANCE_AXISDATA(axisdata, 1);
    }
    
    # 成功处理完所有维度后返回成功标志
    return NPY_SUCCEED;
/*NUMPY_API
 * Get the array of data pointers (1 per object being iterated)
 *
 * This function may be safely called without holding the Python GIL.
 */
NPY_NO_EXPORT char **
NpyIter_GetDataPtrArray(NpyIter *iter)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    /*int ndim = NIT_NDIM(iter);*/
    int nop = NIT_NOP(iter);

    // 如果迭代器标志包含NPY_ITFLAG_BUFFER，则返回缓冲区数据指针数组
    if (itflags&NPY_ITFLAG_BUFFER) {
        NpyIter_BufferData *bufferdata = NIT_BUFFERDATA(iter);
        return NBF_PTRS(bufferdata);
    }
    // 否则返回轴数据的数据指针数组
    else {
        NpyIter_AxisData *axisdata = NIT_AXISDATA(iter);
        return NAD_PTRS(axisdata);
    }
}

/*NUMPY_API
 * Get the array of data pointers (1 per object being iterated),
 * directly into the arrays (never pointing to a buffer), for starting
 * unbuffered iteration. This always returns the addresses for the
 * iterator position as reset to iterator index 0.
 *
 * These pointers are different from the pointers accepted by
 * NpyIter_ResetBasePointers, because the direction along some
 * axes may have been reversed, requiring base offsets.
 *
 * This function may be safely called without holding the Python GIL.
 */
NPY_NO_EXPORT char **
NpyIter_GetInitialDataPtrArray(NpyIter *iter)
{
    /*npy_uint32 itflags = NIT_ITFLAGS(iter);*/
    /*int ndim = NIT_NDIM(iter);*/
    int nop = NIT_NOP(iter);

    // 返回迭代器重置为索引0时的数据指针数组
    return NIT_RESETDATAPTR(iter);
}

/*NUMPY_API
 * Get the array of data type pointers (1 per object being iterated)
 */
NPY_NO_EXPORT PyArray_Descr **
NpyIter_GetDescrArray(NpyIter *iter)
{
    /*npy_uint32 itflags = NIT_ITFLAGS(iter);*/
    /*int ndim = NIT_NDIM(iter);*/
    /*int nop = NIT_NOP(iter);*/

    // 返回迭代器的数据类型指针数组
    return NIT_DTYPES(iter);
}

/*NUMPY_API
 * Get the array of objects being iterated
 */
NPY_NO_EXPORT PyArrayObject **
NpyIter_GetOperandArray(NpyIter *iter)
{
    /*npy_uint32 itflags = NIT_ITFLAGS(iter);*/
    /*int ndim = NIT_NDIM(iter);*/
    int nop = NIT_NOP(iter);

    // 返回迭代器的操作数对象数组
    return NIT_OPERANDS(iter);
}

/*NUMPY_API
 * Returns a view to the i-th object with the iterator's internal axes
 */
NPY_NO_EXPORT PyArrayObject *
NpyIter_GetIterView(NpyIter *iter, npy_intp i)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    int idim, ndim = NIT_NDIM(iter);
    int nop = NIT_NOP(iter);

    npy_intp shape[NPY_MAXDIMS], strides[NPY_MAXDIMS];
    PyArrayObject *obj, *view;
    PyArray_Descr *dtype;
    char *dataptr;
    NpyIter_AxisData *axisdata;
    npy_intp sizeof_axisdata;
    int writeable;

    if (i < 0) {
        PyErr_SetString(PyExc_IndexError,
                "index provided for an iterator view was out of bounds");
        return NULL;
    }

    // 如果索引为负数，返回索引错误
    /* Don't provide views if buffering is enabled */
    if (itflags&NPY_ITFLAG_BUFFER) {
        PyErr_SetString(PyExc_ValueError,
                "cannot provide an iterator view when buffering is enabled");
        return NULL;
    }

    // 获取第i个操作数对象
    obj = NIT_OPERANDS(iter)[i];
    // 获取对象的数据类型描述符
    dtype = PyArray_DESCR(obj);
    // 检查第i个操作数是否可写
    writeable = NIT_OPITFLAGS(iter)[i]&NPY_OP_ITFLAG_WRITE;
    // 获取第i个操作数的数据指针
    dataptr = NIT_RESETDATAPTR(iter)[i];
    # 使用 NIT_AXISDATA 宏从迭代器中获取 axisdata 结构体实例
    axisdata = NIT_AXISDATA(iter);
    # 使用 NIT_AXISDATA_SIZEOF 宏计算 axisdata 结构体的大小
    sizeof_axisdata = NIT_AXISDATA_SIZEOF(itflags, ndim, nop);

    # 从 axisdata 中获取形状（shape）和步幅（strides）
    for (idim = 0; idim < ndim; ++idim) {
        # 从 axisdata 中获取形状并反向存储到 shape 数组中
        shape[ndim-idim-1] = NAD_SHAPE(axisdata);
        # 从 axisdata 中获取步幅数组，并存储到 strides 数组中的当前索引处
        strides[ndim-idim-1] = NAD_STRIDES(axisdata)[i];

        # 将 axisdata 移动到下一个位置
        NIT_ADVANCE_AXISDATA(axisdata, 1);
    }

    # 增加 dtype 的引用计数，确保它在整个视图的生命周期内有效
    Py_INCREF(dtype);
    # 创建一个新的 PyArrayObject 视图，使用给定的描述符和基础数据
    view = (PyArrayObject *)PyArray_NewFromDescrAndBase(
            &PyArray_Type, dtype,
            ndim, shape, strides, dataptr,
            writeable ? NPY_ARRAY_WRITEABLE : 0, NULL, (PyObject *)obj);

    # 返回创建的视图对象
    return view;
/*NUMPY_API
 * Get a pointer to the index, if it is being tracked
 */
NPY_NO_EXPORT npy_intp *
NpyIter_GetIndexPtr(NpyIter *iter)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    /* 获取迭代器的标志位 */
    /*int ndim = NIT_NDIM(iter);*/
    /* 获取迭代器的维度 */
    int nop = NIT_NOP(iter);
    /* 获取迭代器的操作数 */

    NpyIter_AxisData *axisdata = NIT_AXISDATA(iter);
    /* 获取迭代器的轴数据 */

    if (itflags&NPY_ITFLAG_HASINDEX) {
        /* 如果迭代器标志位指示包含索引 */
        /* 索引位于数据指针之后 */
        return (npy_intp*)NAD_PTRS(axisdata) + nop;
    }
    else {
        /* 如果迭代器标志位指示不包含索引 */
        return NULL;
    }
}

/*NUMPY_API
 * Gets an array of read flags (1 per object being iterated)
 */
NPY_NO_EXPORT void
NpyIter_GetReadFlags(NpyIter *iter, char *outreadflags)
{
    /*npy_uint32 itflags = NIT_ITFLAGS(iter);*/
    /* 获取迭代器的标志位 */
    /*int ndim = NIT_NDIM(iter);*/
    /* 获取迭代器的维度 */
    int iop, nop = NIT_NOP(iter);
    /* 获取迭代器的操作数 */

    npyiter_opitflags *op_itflags = NIT_OPITFLAGS(iter);
    /* 获取迭代器的操作标志数组 */

    for (iop = 0; iop < nop; ++iop) {
        /* 遍历每个操作 */
        outreadflags[iop] = (op_itflags[iop]&NPY_OP_ITFLAG_READ) != 0;
        /* 设置每个操作的读标志 */
    }
}

/*NUMPY_API
 * Gets an array of write flags (1 per object being iterated)
 */
NPY_NO_EXPORT void
NpyIter_GetWriteFlags(NpyIter *iter, char *outwriteflags)
{
    /*npy_uint32 itflags = NIT_ITFLAGS(iter);*/
    /* 获取迭代器的标志位 */
    /*int ndim = NIT_NDIM(iter);*/
    /* 获取迭代器的维度 */
    int iop, nop = NIT_NOP(iter);
    /* 获取迭代器的操作数 */

    npyiter_opitflags *op_itflags = NIT_OPITFLAGS(iter);
    /* 获取迭代器的操作标志数组 */

    for (iop = 0; iop < nop; ++iop) {
        /* 遍历每个操作 */
        outwriteflags[iop] = (op_itflags[iop]&NPY_OP_ITFLAG_WRITE) != 0;
        /* 设置每个操作的写标志 */
    }
}

/*NUMPY_API
 * Get the array of strides for the inner loop (when HasExternalLoop is true)
 *
 * This function may be safely called without holding the Python GIL.
 */
NPY_NO_EXPORT npy_intp *
NpyIter_GetInnerStrideArray(NpyIter *iter)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    /* 获取迭代器的标志位 */
    /*int ndim = NIT_NDIM(iter);*/
    /* 获取迭代器的维度 */
    int nop = NIT_NOP(iter);
    /* 获取迭代器的操作数 */

    if (itflags&NPY_ITFLAG_BUFFER) {
        /* 如果迭代器标志位指示使用缓冲区 */
        NpyIter_BufferData *data = NIT_BUFFERDATA(iter);
        /* 获取迭代器的缓冲区数据 */
        return NBF_STRIDES(data);
    }
    else {
        /* 如果迭代器标志位指示不使用缓冲区 */
        NpyIter_AxisData *axisdata = NIT_AXISDATA(iter);
        /* 获取迭代器的轴数据 */
        return NAD_STRIDES(axisdata);
    }
}

/*NUMPY_API
 * Gets the array of strides for the specified axis.
 * If the iterator is tracking a multi-index, gets the strides
 * for the axis specified, otherwise gets the strides for
 * the iteration axis as Fortran order (fastest-changing axis first).
 *
 * Returns NULL if an error occurs.
 */
NPY_NO_EXPORT npy_intp *
NpyIter_GetAxisStrideArray(NpyIter *iter, int axis)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    /* 获取迭代器的标志位 */
    int idim, ndim = NIT_NDIM(iter);
    /* 获取迭代器的维度 */
    int nop = NIT_NOP(iter);
    /* 获取迭代器的操作数 */

    npy_int8 *perm = NIT_PERM(iter);
    /* 获取迭代器的轴排列 */
    NpyIter_AxisData *axisdata = NIT_AXISDATA(iter);
    /* 获取迭代器的轴数据 */
    npy_intp sizeof_axisdata = NIT_AXISDATA_SIZEOF(itflags, ndim, nop);
    /* 获取轴数据结构的大小 */

    if (axis < 0 || axis >= ndim) {
        /* 如果指定的轴超出范围 */
        PyErr_SetString(PyExc_ValueError,
                "axis out of bounds in iterator GetStrideAxisArray");
        return NULL;
    }
    /* 返回轴的步长数组 */

        /*NUMPY_API
         * Get a pointer to the index, if it is being tracked
         */
        NPY_NO_EXPORT npy_intp *
        NpyIter_GetIndexPtr(NpyIter *iter)
        {
            npy_uint32 itflags = NIT_ITFLAGS(iter);
            /* 获取迭代器的标志位 */
            /*int ndim = NIT_NDIM(iter);*/
            /* 获取迭代器的维度 */
            int nop = NIT_NOP(iter);
            /* 获取迭代器的操作数 */

            NpyIter_AxisData *axisdata = NIT_AXISDATA(iter);
            /* 获取迭代器的轴数据 */

            if (itflags&NPY_ITFLAG_HASINDEX) {
                /* 如果迭代器标志位指示包含索引 */
                /* 索引位于数据指针之后 */
                return (npy_intp*)NAD_PTRS(axisdata) + nop;
            }
            else {
                /* 如果迭代器标志位指示不包含索引 */
                return NULL;
            }
        }

        /*NUMPY_API
         * Gets an array of read flags (1 per object being iterated)
         */
        NPY_NO_EXPORT void
        NpyIter_GetReadFlags(NpyIter *iter, char *outreadflags)
        {
            /*npy_uint32 itflags = NIT_ITFLAGS(iter);*/
            /* 获取迭代器的标志位 */
            /*int ndim = NIT_NDIM(iter);*/
            /* 获取迭代器的维度 */
            int iop, nop = NIT_NOP(iter);
            /* 获取迭代器的操作数 */

            npyiter_opitflags *op_itflags = NIT_OPITFLAGS(iter);
            /* 获取迭代器的操作标志数组 */

            for (iop = 0; iop < nop; ++iop) {
                /* 遍历每个操作 */
                outreadflags[iop] = (op_itflags[iop]&NPY_OP_ITFLAG_READ) != 0;
                /* 设置每个操作的读标志 */
            }
        }

        /*NUMPY_API
         * Gets an array of write flags (1 per object being iterated)
         */
        NPY_NO_EXPORT void
        NpyIter_GetWriteFlags(NpyIter *iter, char *outwriteflags)
        {
            /*npy_uint32 itflags = NIT_ITFLAGS(iter);*/
            /* 获取迭代器的标志位 */
            /*int ndim = NIT_NDIM(iter);*/
            /* 获取迭代器的维度 */
            int iop, nop = NIT_NOP(iter);
            /* 获取迭代器的操作数 */

            npyiter_opitflags *op_itflags = NIT_OPITFLAGS(iter);
            /* 获取迭代器的操作标志数组 */

            for (iop = 0; iop < nop; ++iop) {
                /* 遍历每个操作 */
                outwriteflags[iop] = (op_itflags[iop]&NPY_OP_ITFLAG_WRITE) != 0;
                /* 设置每个操作的写标志 */
            }
        }


        /*NUMPY_API
         * Get the array of strides for the inner loop (when HasExternalLoop is true)
         *
         * This function may be safely called without holding the Python GIL.
         */
        NPY_NO_EXPORT npy_intp *
        NpyIter_GetInnerStrideArray(NpyIter *iter)
        {
            npy_uint32 itflags = NIT_ITFLAGS(iter);
            /* 获取迭代器的标志位 */
            /*int ndim = NIT_NDIM(iter);*/
            /* 获取迭代器的维度 */
            int nop = NIT_NOP(iter);
            /* 获取迭代器的操作数 */

            if (itflags&NPY_ITFLAG_BUFFER) {
                /* 如果迭代器标志位指示使用缓冲区 */
                NpyIter_BufferData *data = NIT_BUFFERDATA(iter);
                /* 获取迭代器的缓冲区数据 */
                return NBF_STRIDES(data);
            }
            else {
                /* 如果迭代器标志位指示不使用缓冲区 */
                NpyIter_AxisData *axisdata = NIT_AXISDATA(iter);
                /* 获取迭代器的轴数据 */
                return NAD_STRIDES(axisdata);
            }
        }

        /*NUMPY_API
         * Gets the array of strides for the specified axis.
         * If the iterator is tracking a multi-index, gets the strides
         * for the axis specified, otherwise gets the strides for
         * the iteration axis as Fortran order (fastest-changing axis first).
         *
         * Returns NULL if an error occurs.
         */
        NPY_NO_EXPORT npy_intp *
        NpyIter_GetAxisStrideArray(NpyIter *iter, int axis)
        {
            npy_uint32 itflags = NIT_ITFLAGS(iter);
            /* 获取迭代器的标志位 */
            int idim, ndim = NIT_NDIM(iter);
            /* 获取迭代器的维度 */
            int nop = NIT_NOP(iter);
            /* 获取迭代器的操作数 */

            npy_int8 *perm = NIT_PERM(iter);
            /* 获取迭代器的轴排列 */
            NpyIter_AxisData *axisdata = NIT_AXISDATA(iter);
            /* 获取迭代器的轴数据 */
            npy_intp sizeof_axisdata = NIT_AXISDATA_SIZEOF(itflags, ndim, nop);
            /* 获取轴数据结构的大小 */

            if (axis < 0 || axis >= ndim) {
                /* 如果指定的轴超出范围 */
                PyErr_SetString(PyExc_ValueError,
                        "axis out of bounds in iterator GetStrideAxisArray");
                return NULL;
            }
            /* 返回轴的步长数组 */
    # 检查是否存在多重索引标志位
    if (itflags&NPY_ITFLAG_HASMULTIINDEX) {
        # 如果存在多重索引，则反转轴的顺序，因为迭代器会按照这种方式处理它们
        axis = ndim-1-axis;

        # 首先找到所需的轴
        for (idim = 0; idim < ndim; ++idim, NIT_ADVANCE_AXISDATA(axisdata, 1)) {
            # 检查当前维度的排列是否等于轴或者是其相反数
            if (perm[idim] == axis || -1 - perm[idim] == axis) {
                # 返回与该轴相关的步长数据
                return NAD_STRIDES(axisdata);
            }
        }
    }
    else {
        # 如果没有多重索引，则直接返回指定轴的步长数据
        return NAD_STRIDES(NIT_INDEX_AXISDATA(axisdata, axis));
    }

    # 如果以上条件都不满足，则抛出运行时错误并返回空指针
    PyErr_SetString(PyExc_RuntimeError,
            "internal error in iterator perm");
    return  NULL;
/*
 * NUMPY_API
 * 获取一个固定的步幅数组。任何在迭代过程中可能会改变的步幅都设置为 NPY_MAX_INTP。
 * 一旦迭代器准备好进行迭代，调用此函数获取在内部循环中始终保持不变的步幅，
 * 然后选择利用这些固定步幅的优化内部循环函数。
 *
 * 此函数可以在不持有 Python 全局解释器锁（GIL）的情况下安全调用。
 */
NPY_NO_EXPORT void
NpyIter_GetInnerFixedStrideArray(NpyIter *iter, npy_intp *out_strides)
{
    // 获取迭代器的标志位
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    // 获取迭代器的维度数
    int ndim = NIT_NDIM(iter);
    // 初始化循环中的操作数和操作数的数量
    int iop, nop = NIT_NOP(iter);

    // 获取迭代器的第一个轴数据
    NpyIter_AxisData *axisdata0 = NIT_AXISDATA(iter);
    // 计算轴数据结构体的大小
    npy_intp sizeof_axisdata = NIT_AXISDATA_SIZEOF(itflags, ndim, nop);
}
    # 如果itflags中包含NPY_ITFLAG_BUFFER标志
    if (itflags&NPY_ITFLAG_BUFFER) {
        # 获取迭代器的缓冲数据结构
        NpyIter_BufferData *data = NIT_BUFFERDATA(iter);
        # 获取操作迭代器的标志数组
        npyiter_opitflags *op_itflags = NIT_OPITFLAGS(iter);
        # 获取迭代器中的步长数组以及轴数据0的步长数组
        npy_intp stride, *strides = NBF_STRIDES(data),
                *ad_strides = NAD_STRIDES(axisdata0);
        # 获取迭代器中的数据类型描述符数组
        PyArray_Descr **dtypes = NIT_DTYPES(iter);

        # 遍历操作符的数量
        for (iop = 0; iop < nop; ++iop) {
            # 获取当前操作符的步长
            stride = strides[iop];

            /*
             * 操作数始终/从不缓冲的具有固定步长，
             * 当ndim为0或1时，所有内容都具有固定步长
             */
            if (ndim <= 1 || (op_itflags[iop]&
                            (NPY_OP_ITFLAG_CAST|NPY_OP_ITFLAG_BUFNEVER))) {
                # 将输出步长设置为当前步长
                out_strides[iop] = stride;
            }
            /* 如果是一个约简操作，0步长内循环可能有固定步长 */
            else if (stride == 0 && (itflags&NPY_ITFLAG_REDUCE)) {
                /* 如果是约简操作数，则步长肯定是固定的 */
                if (op_itflags[iop]&NPY_OP_ITFLAG_REDUCE) {
                    # 将输出步长设置为当前步长
                    out_strides[iop] = stride;
                }
                /*
                 * 否则，如果所有维度的步长都是0，则保证是固定步长。
                 */
                else {
                    NpyIter_AxisData *axisdata = axisdata0;
                    int idim;
                    for (idim = 0; idim < ndim; ++idim) {
                        if (NAD_STRIDES(axisdata)[iop] != 0) {
                            break;
                        }
                        NIT_ADVANCE_AXISDATA(axisdata, 1);
                    }
                    /* 如果所有步长都是0，则步长不会改变 */
                    if (idim == ndim) {
                        # 将输出步长设置为当前步长
                        out_strides[iop] = stride;
                    }
                    else {
                        # 将输出步长设置为最大整数值，表示步长可能会变化
                        out_strides[iop] = NPY_MAX_INTP;
                    }
                }
            }
            /*
             * 内循环连续数组意味着在缓冲和非缓冲之间切换时其步长不会改变
             */
            else if (ad_strides[iop] == dtypes[iop]->elsize) {
                # 将输出步长设置为轴数据0的步长数组中的步长
                out_strides[iop] = ad_strides[iop];
            }
            /*
             * 否则，如果操作数有时缓冲有时不缓冲，步长可能会改变。
             */
            else {
                # 将输出步长设置为最大整数值，表示步长可能会变化
                out_strides[iop] = NPY_MAX_INTP;
            }
        }
    }
    else {
        /* 如果没有缓冲，步长始终是固定的 */
        # 将轴数据0的步长数组复制到输出步长数组中
        memcpy(out_strides, NAD_STRIDES(axisdata0), nop*NPY_SIZEOF_INTP);
    }
/*NUMPY_API
 * 获取指向内部循环大小的指针（当 HasExternalLoop 为 true 时）
 *
 * 可以安全地在不持有 Python GIL 的情况下调用此函数。
 */
NPY_NO_EXPORT npy_intp *
NpyIter_GetInnerLoopSizePtr(NpyIter *iter)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);  // 获取迭代器的标志位
    /*int ndim = NIT_NDIM(iter);*/  // 注释掉的代码，原本用于获取迭代器的维度
    int nop = NIT_NOP(iter);  // 获取迭代器的操作数

    if (itflags & NPY_ITFLAG_BUFFER) {
        NpyIter_BufferData *data = NIT_BUFFERDATA(iter);  // 获取迭代器的缓冲数据
        return &NBF_SIZE(data);  // 返回缓冲数据的大小指针
    }
    else {
        NpyIter_AxisData *axisdata = NIT_AXISDATA(iter);  // 获取迭代器的轴数据
        return &NAD_SHAPE(axisdata);  // 返回轴数据的形状指针
    }
}

/*NUMPY_API
 * 用于调试打印迭代器信息
 */
NPY_NO_EXPORT void
NpyIter_DebugPrint(NpyIter *iter)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);  // 获取迭代器的标志位
    int idim, ndim = NIT_NDIM(iter);  // 获取迭代器的维度
    int iop, nop = NIT_NOP(iter);  // 获取迭代器的操作数

    NpyIter_AxisData *axisdata;
    npy_intp sizeof_axisdata;

    NPY_ALLOW_C_API_DEF
    NPY_ALLOW_C_API

    printf("\n------ BEGIN ITERATOR DUMP ------\n");
    printf("| Iterator Address: %p\n", (void *)iter);  // 打印迭代器地址
    printf("| ItFlags: ");
    if (itflags & NPY_ITFLAG_IDENTPERM)
        printf("IDENTPERM ");  // 如果标志位包含 IDENTPERM，则打印
    if (itflags & NPY_ITFLAG_NEGPERM)
        printf("NEGPERM ");  // 如果标志位包含 NEGPERM，则打印
    if (itflags & NPY_ITFLAG_HASINDEX)
        printf("HASINDEX ");  // 如果标志位包含 HASINDEX，则打印
    if (itflags & NPY_ITFLAG_HASMULTIINDEX)
        printf("HASMULTIINDEX ");  // 如果标志位包含 HASMULTIINDEX，则打印
    if (itflags & NPY_ITFLAG_FORCEDORDER)
        printf("FORCEDORDER ");  // 如果标志位包含 FORCEDORDER，则打印
    if (itflags & NPY_ITFLAG_EXLOOP)
        printf("EXLOOP ");  // 如果标志位包含 EXLOOP，则打印
    if (itflags & NPY_ITFLAG_RANGE)
        printf("RANGE ");  // 如果标志位包含 RANGE，则打印
    if (itflags & NPY_ITFLAG_BUFFER)
        printf("BUFFER ");  // 如果标志位包含 BUFFER，则打印
    if (itflags & NPY_ITFLAG_GROWINNER)
        printf("GROWINNER ");  // 如果标志位包含 GROWINNER，则打印
    if (itflags & NPY_ITFLAG_ONEITERATION)
        printf("ONEITERATION ");  // 如果标志位包含 ONEITERATION，则打印
    if (itflags & NPY_ITFLAG_DELAYBUF)
        printf("DELAYBUF ");  // 如果标志位包含 DELAYBUF，则打印
    if (itflags & NPY_ITFLAG_NEEDSAPI)
        printf("NEEDSAPI ");  // 如果标志位包含 NEEDSAPI，则打印
    if (itflags & NPY_ITFLAG_REDUCE)
        printf("REDUCE ");  // 如果标志位包含 REDUCE，则打印
    if (itflags & NPY_ITFLAG_REUSE_REDUCE_LOOPS)
        printf("REUSE_REDUCE_LOOPS ");  // 如果标志位包含 REUSE_REDUCE_LOOPS，则打印

    printf("\n");
    printf("| NDim: %d\n", ndim);  // 打印迭代器的维度
    printf("| NOp: %d\n", nop);  // 打印迭代器的操作数
    if (NIT_MASKOP(iter) >= 0) {
        printf("| MaskOp: %d\n", (int)NIT_MASKOP(iter));  // 如果迭代器的掩码操作大于等于零，则打印
    }
    printf("| IterSize: %d\n", (int)NIT_ITERSIZE(iter));  // 打印迭代器的大小
    printf("| IterStart: %d\n", (int)NIT_ITERSTART(iter));  // 打印迭代器的起始位置
    printf("| IterEnd: %d\n", (int)NIT_ITEREND(iter));  // 打印迭代器的结束位置
    printf("| IterIndex: %d\n", (int)NIT_ITERINDEX(iter));  // 打印迭代器的当前索引
    printf("| Iterator SizeOf: %d\n", (int)NIT_SIZEOF_ITERATOR(itflags, ndim, nop));  // 打印迭代器的大小
    printf("| BufferData SizeOf: %d\n", (int)NIT_BUFFERDATA_SIZEOF(itflags, ndim, nop));  // 打印缓冲数据的大小
    printf("| AxisData SizeOf: %d\n", (int)NIT_AXISDATA_SIZEOF(itflags, ndim, nop));  // 打印轴数据的大小
    printf("|\n");

    printf("| Perm: ");
    for (idim = 0; idim < ndim; ++idim) {
        printf("%d ", (int)NIT_PERM(iter)[idim]);  // 打印迭代器的排列顺序
    }
    printf("\n");
    printf("| DTypes: ");
}
    // 打印 NIT_DTYPES(iter) 中指针的十六进制地址
    for (iop = 0; iop < nop; ++iop) {
        printf("%p ", (void *)NIT_DTYPES(iter)[iop]);
    }
    printf("\n");

    // 打印 NIT_DTYPES(iter) 中每个元素的 PyObject 表示，或者打印 "(nil)" 如果是空指针
    printf("| DTypes: ");
    for (iop = 0; iop < nop; ++iop) {
        if (NIT_DTYPES(iter)[iop] != NULL)
            PyObject_Print((PyObject*)NIT_DTYPES(iter)[iop], stdout, 0);
        else
            printf("(nil) ");
        printf(" ");
    }
    printf("\n");

    // 打印 NIT_RESETDATAPTR(iter) 中每个指针的十六进制地址
    printf("| InitDataPtrs: ");
    for (iop = 0; iop < nop; ++iop) {
        printf("%p ", (void *)NIT_RESETDATAPTR(iter)[iop]);
    }
    printf("\n");

    // 打印 NIT_BASEOFFSETS(iter) 中每个元素的整数值
    printf("| BaseOffsets: ");
    for (iop = 0; iop < nop; ++iop) {
        printf("%i ", (int)NIT_BASEOFFSETS(iter)[iop]);
    }
    printf("\n");

    // 如果 itflags 包含 NPY_ITFLAG_HASINDEX 标志，则打印 NIT_RESETDATAPTR(iter)[nop] 强制转换为 npy_intp 后的整数值
    if (itflags&NPY_ITFLAG_HASINDEX) {
        printf("| InitIndex: %d\n",
                        (int)(npy_intp)NIT_RESETDATAPTR(iter)[nop]);
    }

    // 打印 NIT_OPERANDS(iter) 中每个指针的十六进制地址
    printf("| Operands: ");
    for (iop = 0; iop < nop; ++iop) {
        printf("%p ", (void *)NIT_OPERANDS(iter)[iop]);
    }
    printf("\n");

    // 打印 NIT_OPERANDS(iter) 中每个操作数的 dtype 对象，或者打印 "(nil)" 如果是空指针
    printf("| Operand DTypes: ");
    for (iop = 0; iop < nop; ++iop) {
        PyArray_Descr *dtype;
        if (NIT_OPERANDS(iter)[iop] != NULL) {
            dtype = PyArray_DESCR(NIT_OPERANDS(iter)[iop]);
            if (dtype != NULL)
                PyObject_Print((PyObject *)dtype, stdout, 0);
            else
                printf("(nil) ");
        }
        else {
            printf("(op nil) ");
        }
        printf(" ");
    }
    printf("\n");

    // 打印 NIT_OPITFLAGS(iter) 中每个操作的位标志，展示哪些操作被设置为特定的标志位
    printf("| OpItFlags:\n");
    for (iop = 0; iop < nop; ++iop) {
        printf("|   Flags[%d]: ", (int)iop);
        if ((NIT_OPITFLAGS(iter)[iop])&NPY_OP_ITFLAG_READ)
            printf("READ ");
        if ((NIT_OPITFLAGS(iter)[iop])&NPY_OP_ITFLAG_WRITE)
            printf("WRITE ");
        if ((NIT_OPITFLAGS(iter)[iop])&NPY_OP_ITFLAG_CAST)
            printf("CAST ");
        if ((NIT_OPITFLAGS(iter)[iop])&NPY_OP_ITFLAG_BUFNEVER)
            printf("BUFNEVER ");
        if ((NIT_OPITFLAGS(iter)[iop])&NPY_OP_ITFLAG_ALIGNED)
            printf("ALIGNED ");
        if ((NIT_OPITFLAGS(iter)[iop])&NPY_OP_ITFLAG_REDUCE)
            printf("REDUCE ");
        if ((NIT_OPITFLAGS(iter)[iop])&NPY_OP_ITFLAG_VIRTUAL)
            printf("VIRTUAL ");
        if ((NIT_OPITFLAGS(iter)[iop])&NPY_OP_ITFLAG_WRITEMASKED)
            printf("WRITEMASKED ");
        printf("\n");
    }
    // 打印分隔符
    printf("|\n");
    # 检查是否存在 NPY_ITFLAG_BUFFER 标志位
    if (itflags&NPY_ITFLAG_BUFFER) {
        # 获取 bufferdata 结构体指针
        NpyIter_BufferData *bufferdata = NIT_BUFFERDATA(iter);
        # 获取 transferinfo 结构体指针
        NpyIter_TransferInfo *transferinfo = NBF_TRANSFERINFO(bufferdata);

        # 打印 BufferData 相关信息
        printf("| BufferData:\n");
        # 打印缓冲区大小
        printf("|   BufferSize: %d\n", (int)NBF_BUFFERSIZE(bufferdata));
        # 打印数据大小
        printf("|   Size: %d\n", (int)NBF_SIZE(bufferdata));
        # 打印 BufIterEnd 标志
        printf("|   BufIterEnd: %d\n", (int)NBF_BUFITEREND(bufferdata));

        # 如果存在 NPY_ITFLAG_REDUCE 标志位，打印 REDUCE 相关信息
        if (itflags&NPY_ITFLAG_REDUCE) {
            # 打印 REDUCE Pos
            printf("|   REDUCE Pos: %d\n", (int)NBF_REDUCE_POS(bufferdata));
            # 打印 REDUCE OuterSize
            printf("|   REDUCE OuterSize: %d\n", (int)NBF_REDUCE_OUTERSIZE(bufferdata));
            # 打印 REDUCE OuterDim
            printf("|   REDUCE OuterDim: %d\n", (int)NBF_REDUCE_OUTERDIM(bufferdata));
        }

        # 打印 Strides 数组
        printf("|   Strides: ");
        for (iop = 0; iop < nop; ++iop)
            printf("%d ", (int)NBF_STRIDES(bufferdata)[iop]);
        printf("\n");

        # 当存在 NPY_ITFLAG_EXLOOP 标志位时，打印 Fixed Strides 数组
        if (itflags&NPY_ITFLAG_EXLOOP) {
            npy_intp fixedstrides[NPY_MAXDIMS];
            printf("|   Fixed Strides: ");
            NpyIter_GetInnerFixedStrideArray(iter, fixedstrides);
            for (iop = 0; iop < nop; ++iop)
                printf("%d ", (int)fixedstrides[iop]);
            printf("\n");
        }

        # 打印 Ptrs 数组
        printf("|   Ptrs: ");
        for (iop = 0; iop < nop; ++iop)
            printf("%p ", (void *)NBF_PTRS(bufferdata)[iop]);
        printf("\n");

        # 如果存在 NPY_ITFLAG_REDUCE 标志位，打印 REDUCE Outer Strides 数组和 REDUCE Outer Ptrs 数组
        if (itflags&NPY_ITFLAG_REDUCE) {
            printf("|   REDUCE Outer Strides: ");
            for (iop = 0; iop < nop; ++iop)
                printf("%d ", (int)NBF_REDUCE_OUTERSTRIDES(bufferdata)[iop]);
            printf("\n");
            printf("|   REDUCE Outer Ptrs: ");
            for (iop = 0; iop < nop; ++iop)
                printf("%p ", (void *)NBF_REDUCE_OUTERPTRS(bufferdata)[iop]);
            printf("\n");
        }

        # 打印 ReadTransferFn 函数指针数组
        printf("|   ReadTransferFn: ");
        for (iop = 0; iop < nop; ++iop)
            printf("%p ", (void *)transferinfo[iop].read.func);
        printf("\n");

        # 打印 ReadTransferData 辅助数据指针数组
        printf("|   ReadTransferData: ");
        for (iop = 0; iop < nop; ++iop)
            printf("%p ", (void *)transferinfo[iop].read.auxdata);
        printf("\n");

        # 打印 WriteTransferFn 函数指针数组
        printf("|   WriteTransferFn: ");
        for (iop = 0; iop < nop; ++iop)
            printf("%p ", (void *)transferinfo[iop].write.func);
        printf("\n");

        # 打印 WriteTransferData 辅助数据指针数组
        printf("|   WriteTransferData: ");
        for (iop = 0; iop < nop; ++iop)
            printf("%p ", (void *)transferinfo[iop].write.auxdata);
        printf("\n");

        # 打印 Buffers 缓冲区指针数组
        printf("|   Buffers: ");
        for (iop = 0; iop < nop; ++iop)
            printf("%p ", (void *)NBF_BUFFERS(bufferdata)[iop]);
        printf("\n");

        # 打印结束符号
        printf("|\n");
    }

    # 获取 axisdata 指针
    axisdata = NIT_AXISDATA(iter);
    # 获取 axisdata 的大小
    sizeof_axisdata = NIT_AXISDATA_SIZEOF(itflags, ndim, nop);
    // 对每个维度数据进行迭代，输出 AxisData 的相关信息
    for (idim = 0; idim < ndim; ++idim, NIT_ADVANCE_AXISDATA(axisdata, 1)) {
        // 打印当前 AxisData 的索引信息
        printf("| AxisData[%d]:\n", (int)idim);
        // 打印当前 AxisData 的形状信息
        printf("|   Shape: %d\n", (int)NAD_SHAPE(axisdata));
        // 打印当前 AxisData 的索引值信息
        printf("|   Index: %d\n", (int)NAD_INDEX(axisdata));
        // 打印当前 AxisData 的步幅信息
        printf("|   Strides: ");
        for (iop = 0; iop < nop; ++iop) {
            // 依次打印每个维度的步幅值
            printf("%d ", (int)NAD_STRIDES(axisdata)[iop]);
        }
        printf("\n");
        // 如果 Iterator 标志中包含 NPY_ITFLAG_HASINDEX 标志位
        if (itflags & NPY_ITFLAG_HASINDEX) {
            // 打印 Index Stride 值
            printf("|   Index Stride: %d\n", (int)NAD_STRIDES(axisdata)[nop]);
        }
        // 打印当前 AxisData 的指针信息
        printf("|   Ptrs: ");
        for (iop = 0; iop < nop; ++iop) {
            // 依次打印每个维度的指针地址
            printf("%p ", (void *)NAD_PTRS(axisdata)[iop]);
        }
        printf("\n");
        // 如果 Iterator 标志中包含 NPY_ITFLAG_HASINDEX 标志位
        if (itflags & NPY_ITFLAG_HASINDEX) {
            // 打印 Index Value 值
            printf("|   Index Value: %d\n",
                   (int)((npy_intp *)NAD_PTRS(axisdata))[nop]);
        }
    }

    // 输出迭代器数据输出结束的标志
    printf("------- END ITERATOR DUMP -------\n");
    // 刷新标准输出缓冲区
    fflush(stdout);

    // 禁用 C API 接口
    NPY_DISABLE_C_API
/* 关闭 npyiter_coalesce_axes 函数 */
NPY_NO_EXPORT void
npyiter_coalesce_axes(NpyIter *iter)
{
    /* 获取迭代器的标志位 */
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    /* 获取迭代器的维度数量 */
    int idim, ndim = NIT_NDIM(iter);
    /* 获取迭代器的操作数数量 */
    int nop = NIT_NOP(iter);

    /* 获取第一个轴数据的步幅数量 */
    npy_intp istrides, nstrides = NAD_NSTRIDES();
    /* 获取迭代器的轴数据 */
    NpyIter_AxisData *axisdata = NIT_AXISDATA(iter);
    /* 计算轴数据结构体的大小 */
    npy_intp sizeof_axisdata = NIT_AXISDATA_SIZEOF(itflags, ndim, nop);
    /* 压缩后的轴数据 */
    NpyIter_AxisData *ad_compress = axisdata;
    /* 新的维度数量 */
    npy_intp new_ndim = 1;

    /* 在合并轴之后，清除 IDENTPERM 和 HASMULTIINDEX 标志位 */
    NIT_ITFLAGS(iter) &= ~(NPY_ITFLAG_IDENTPERM|NPY_ITFLAG_HASMULTIINDEX);

    /* 遍历迭代器的每一个轴 */
    for (idim = 0; idim < ndim-1; ++idim) {
        /* 可以合并的标志位 */
        int can_coalesce = 1;
        /* 第一个轴数据的形状 */
        npy_intp shape0 = NAD_SHAPE(ad_compress);
        /* 第二个轴数据的形状 */
        npy_intp shape1 = NAD_SHAPE(NIT_INDEX_AXISDATA(axisdata, 1));
        /* 第一个轴数据的步幅数组 */
        npy_intp *strides0 = NAD_STRIDES(ad_compress);
        /* 第二个轴数据的步幅数组 */
        npy_intp *strides1 = NAD_STRIDES(NIT_INDEX_AXISDATA(axisdata, 1));

        /* 检查所有轴是否可以合并 */
        for (istrides = 0; istrides < nstrides; ++istrides) {
            if (!((shape0 == 1 && strides0[istrides] == 0) ||
                  (shape1 == 1 && strides1[istrides] == 0)) &&
                     (strides0[istrides]*shape0 != strides1[istrides])) {
                can_coalesce = 0;
                break;
            }
        }

        /* 如果可以合并 */
        if (can_coalesce) {
            /* 第一个轴数据的步幅 */
            npy_intp *strides = NAD_STRIDES(ad_compress);

            /* 前进到下一个轴数据 */
            NIT_ADVANCE_AXISDATA(axisdata, 1);
            /* 更新合并后轴数据的形状 */
            NAD_SHAPE(ad_compress) *= NAD_SHAPE(axisdata);
            /* 更新合并后轴数据的步幅 */
            for (istrides = 0; istrides < nstrides; ++istrides) {
                if (strides[istrides] == 0) {
                    strides[istrides] = NAD_STRIDES(axisdata)[istrides];
                }
            }
        }
        else {
            /* 前进到下一个轴数据 */
            NIT_ADVANCE_AXISDATA(axisdata, 1);
            /* 前进到下一个压缩轴数据 */
            NIT_ADVANCE_AXISDATA(ad_compress, 1);
            /* 如果 ad_compress 不等于 axisdata，则复制 axisdata 到 ad_compress */
            if (ad_compress != axisdata) {
                memcpy(ad_compress, axisdata, sizeof_axisdata);
            }
            /* 增加新的维度数量 */
            ++new_ndim;
        }
    }

    /*
     * 如果轴的数量减少了，重置 perm 并压缩数据到新的布局。
     */
    if (new_ndim < ndim) {
        /* 获取 perm 数组 */
        npy_int8 *perm = NIT_PERM(iter);

        /* 重置为身份 perm */
        for (idim = 0; idim < new_ndim; ++idim) {
            perm[idim] = (npy_int8)idim;
        }
        /* 更新迭代器的维度数量 */
        NIT_NDIM(iter) = new_ndim;
    }
}

/*
 * 如果 errmsg 非空，则应指向一个变量，该变量将接收错误消息，且不会设置 Python 异常。
 * 这样可以从不持有 GIL 的代码中调用该函数。
 */
NPY_NO_EXPORT int
npyiter_allocate_buffers(NpyIter *iter, char **errmsg)
{
    /* 获取操作数的数量 */
    int iop = 0, nop = NIT_NOP(iter);

    /* 定义变量 */
    npy_intp i;
    /* 获取操作的标志位 */
    npyiter_opitflags *op_itflags = NIT_OPITFLAGS(iter);
    /* 获取缓冲区数据 */
    NpyIter_BufferData *bufferdata = NIT_BUFFERDATA(iter);
    /* 获取操作的数据类型数组 */
    PyArray_Descr **op_dtype = NIT_DTYPES(iter);
}
    // 从 bufferdata 中获取缓冲区大小并存储在 npy_intp 类型变量 buffersize 中
    npy_intp buffersize = NBF_BUFFERSIZE(bufferdata);
    // 从 bufferdata 中获取缓冲区数组的指针，并存储在 char* 类型指针 buffers 中
    char *buffer, **buffers = NBF_BUFFERS(bufferdata);

    // 对每个操作进行迭代处理
    for (iop = 0; iop < nop; ++iop) {
        // 从 op_itflags 数组中获取当前操作的迭代器标志
        npyiter_opitflags flags = op_itflags[iop];

        /*
         * 如果确定可能需要一个缓冲区，
         * 则分配一个。
         */
        if (!(flags & NPY_OP_ITFLAG_BUFNEVER)) {
            // 获取当前操作的元素大小
            npy_intp itemsize = op_dtype[iop]->elsize;
            // 使用 PyArray_malloc 分配 itemsize*buffersize 大小的内存
            buffer = PyArray_malloc(itemsize * buffersize);
            // 检查分配内存是否成功
            if (buffer == NULL) {
                // 如果内存分配失败，根据情况设置错误信息或者错误码并跳转到失败处理标签
                if (errmsg == NULL) {
                    PyErr_NoMemory();
                } else {
                    *errmsg = "out of memory";
                }
                goto fail;
            }
            // 如果操作的数据类型需要初始化，则使用 memset 初始化缓冲区
            if (PyDataType_FLAGCHK(op_dtype[iop], NPY_NEEDS_INIT)) {
                memset(buffer, '\0', itemsize * buffersize);
            }
            // 将分配的缓冲区指针存储在 buffers 数组中对应的位置
            buffers[iop] = buffer;
        }
    }

    // 函数执行成功，返回值 1 表示成功
    return 1;
fail:
    // 遍历缓冲区数组中的每一个指针
    for (i = 0; i < iop; ++i) {
        // 检查当前指针是否非空
        if (buffers[i] != NULL) {
            // 释放当前指针所指向的内存
            PyArray_free(buffers[i]);
            // 将当前指针置为 NULL，避免悬空指针
            buffers[i] = NULL;
        }
    }
    // 返回成功标志
    return 0;
}

/*
 * This sets the AXISDATA portion of the iterator to the specified
 * iterindex, updating the pointers as well.  This function does
 * no error checking.
 */
NPY_NO_EXPORT void
npyiter_goto_iterindex(NpyIter *iter, npy_intp iterindex)
{
    // 获取迭代器的标志位
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    // 获取迭代器的维度数和操作数
    int idim, ndim = NIT_NDIM(iter);
    int nop = NIT_NOP(iter);

    // 定义数据指针和轴数据结构
    char **dataptr;
    NpyIter_AxisData *axisdata;
    npy_intp sizeof_axisdata;
    npy_intp istrides, nstrides, i, shape;

    // 获取迭代器的轴数据
    axisdata = NIT_AXISDATA(iter);
    // 获取轴数据结构的大小
    sizeof_axisdata = NIT_AXISDATA_SIZEOF(itflags, ndim, nop);
    // 获取轴的步长数
    nstrides = NAD_NSTRIDES();

    // 设置迭代器的当前索引为给定的索引值
    NIT_ITERINDEX(iter) = iterindex;

    // 如果维度数为零，则将其设置为一（最小值）
    ndim = ndim ? ndim : 1;

    // 如果索引值为零
    if (iterindex == 0) {
        // 重置数据指针为初始位置
        dataptr = NIT_RESETDATAPTR(iter);

        // 遍历每个维度
        for (idim = 0; idim < ndim; ++idim) {
            char **ptrs;
            // 将当前轴数据的索引设为零
            NAD_INDEX(axisdata) = 0;
            // 获取当前轴数据的指针数组
            ptrs = NAD_PTRS(axisdata);
            // 遍历当前轴的步长数
            for (istrides = 0; istrides < nstrides; ++istrides) {
                // 将数据指针设置为当前轴数据的指针位置
                ptrs[istrides] = dataptr[istrides];
            }

            // 将轴数据向前推进一个位置
            NIT_ADVANCE_AXISDATA(axisdata, 1);
        }
    }
    else {
        /*
         * Set the multi-index, from the fastest-changing to the
         * slowest-changing.
         */
        // 重新获取轴数据
        axisdata = NIT_AXISDATA(iter);
        // 获取轴数据的形状
        shape = NAD_SHAPE(axisdata);
        // 使用索引值初始化 i
        i = iterindex;
        // 计算索引值除以轴数据形状
        iterindex /= shape;
        // 设置当前轴数据的索引值为 i 减去迭代次数乘以轴数据形状
        NAD_INDEX(axisdata) = i - iterindex * shape;
        // 遍历每个维度减一
        for (idim = 0; idim < ndim-1; ++idim) {
            // 将轴数据向前推进一个位置
            NIT_ADVANCE_AXISDATA(axisdata, 1);

            // 重新获取轴数据的形状
            shape = NAD_SHAPE(axisdata);
            // 使用索引值初始化 i
            i = iterindex;
            // 计算索引值除以轴数据形状
            iterindex /= shape;
            // 设置当前轴数据的索引值为 i 减去迭代次数乘以轴数据形状
            NAD_INDEX(axisdata) = i - iterindex * shape;
        }

        // 重置数据指针为初始位置
        dataptr = NIT_RESETDATAPTR(iter);

        /*
         * Accumulate the successive pointers with their
         * offsets in the opposite order, starting from the
         * original data pointers.
         */
        // 遍历每个维度
        for (idim = 0; idim < ndim; ++idim) {
            npy_intp *strides;
            char **ptrs;

            // 获取当前轴数据的步长数组和指针数组
            strides = NAD_STRIDES(axisdata);
            ptrs = NAD_PTRS(axisdata);

            // 初始化 i 为当前轴数据的索引
            i = NAD_INDEX(axisdata);

            // 遍历当前轴的步长数
            for (istrides = 0; istrides < nstrides; ++istrides) {
                // 将数据指针设置为原始数据指针加上索引乘以步长
                ptrs[istrides] = dataptr[istrides] + i*strides[istrides];
            }

            // 将数据指针设置为当前轴数据的指针数组
            dataptr = ptrs;

            // 将轴数据向后推进一个位置
            NIT_ADVANCE_AXISDATA(axisdata, -1);
        }
    }
}

/*
 * This gets called after the buffers have been exhausted, and
 * their data needs to be written back to the arrays.  The multi-index
 * must be positioned for the beginning of the buffer.
 */
NPY_NO_EXPORT int
npyiter_copy_from_buffers(NpyIter *iter)
{
    // 获取迭代器的标志位
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    // 获取迭代器的维度数和操作数
    int ndim = NIT_NDIM(iter);
    int iop, nop = NIT_NOP(iter);
    int maskop = NIT_MASKOP(iter);

    // 获取迭代器的操作标志位
    npyiter_opitflags *op_itflags = NIT_OPITFLAGS(iter);
    # 获取指向 NpyIter_BufferData 结构体的指针
    NpyIter_BufferData *bufferdata = NIT_BUFFERDATA(iter);
    # 获取指向 NpyIter_AxisData 结构体的指针，并初始化 reduce_outeraxisdata 为 NULL
    NpyIter_AxisData *axisdata = NIT_AXISDATA(iter),
                    *reduce_outeraxisdata = NULL;

    # 获取指向 PyArray_Descr 结构体指针数组的指针
    PyArray_Descr **dtypes = NIT_DTYPES(iter);
    # 获取数据传输大小
    npy_intp transfersize = NBF_SIZE(bufferdata);
    # 获取缓冲区步长数组和轴数据步长数组
    npy_intp *strides = NBF_STRIDES(bufferdata),
             *ad_strides = NAD_STRIDES(axisdata);
    # 计算 axisdata 结构体的大小
    npy_intp sizeof_axisdata = NIT_AXISDATA_SIZEOF(itflags, ndim, nop);
    # 获取轴数据指针数组
    char **ad_ptrs = NAD_PTRS(axisdata);
    # 获取缓冲区指针数组
    char **buffers = NBF_BUFFERS(bufferdata);
    # 初始化 buffer 指针
    char *buffer;

    # 初始化 reduce_outerdim 和 reduce_outerstrides
    npy_intp reduce_outerdim = 0;
    npy_intp *reduce_outerstrides = NULL;

    # 计算 axisdata_incr 的值
    npy_intp axisdata_incr = NIT_AXISDATA_SIZEOF(itflags, ndim, nop) /
                                NPY_SIZEOF_INTP;

    # 如果缓冲区大小为 0，则无需复制任何内容，直接返回
    if (NBF_SIZE(bufferdata) == 0) {
        return 0;
    }

    # 打印调试信息，指示正在将缓冲区复制到输出
    NPY_IT_DBG_PRINT("Iterator: Copying buffers to outputs\n");

    # 如果设置了 REDUCE 标志，则获取相关信息
    if (itflags & NPY_ITFLAG_REDUCE) {
        # 获取 reduce_outerdim 和 reduce_outerstrides
        reduce_outerdim = NBF_REDUCE_OUTERDIM(bufferdata);
        reduce_outerstrides = NBF_REDUCE_OUTERSTRIDES(bufferdata);
        # 获取 reduce_outeraxisdata 的指针
        reduce_outeraxisdata = NIT_INDEX_AXISDATA(axisdata, reduce_outerdim);
        # 调整传输大小以考虑 REDUCE 的外部尺寸
        transfersize *= NBF_REDUCE_OUTERSIZE(bufferdata);
    }

    # 获取传输信息结构体的指针
    NpyIter_TransferInfo *transferinfo = NBF_TRANSFERINFO(bufferdata);

    # 打印调试信息，指示完成将缓冲区复制到输出
    NPY_IT_DBG_PRINT("Iterator: Finished copying buffers to outputs\n");

    # 返回值为 0，表示函数执行成功
    return 0;
/*
 * This gets called after the iterator has been positioned to a multi-index
 * for the start of a buffer.  It decides which operands need a buffer,
 * and copies the data into the buffers.
 */
NPY_NO_EXPORT int
npyiter_copy_to_buffers(NpyIter *iter, char **prev_dataptrs)
{
    // 获取迭代器的标志位
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    // 获取操作数的维度
    int ndim = NIT_NDIM(iter);
    // 获取操作数的数量
    int iop, nop = NIT_NOP(iter);

    // 获取操作数的迭代器标志数组和缓冲数据结构
    npyiter_opitflags *op_itflags = NIT_OPITFLAGS(iter);
    NpyIter_BufferData *bufferdata = NIT_BUFFERDATA(iter);
    // 获取轴数据和外部减少轴数据
    NpyIter_AxisData *axisdata = NIT_AXISDATA(iter),
                    *reduce_outeraxisdata = NULL;

    // 获取数据类型数组和操作数数组
    PyArray_Descr **dtypes = NIT_DTYPES(iter);
    PyArrayObject **operands = NIT_OPERANDS(iter);
    // 获取缓冲数据和轴数据的步长数组
    npy_intp *strides = NBF_STRIDES(bufferdata),
             *ad_strides = NAD_STRIDES(axisdata);
    // 计算轴数据的大小
    npy_intp sizeof_axisdata = NIT_AXISDATA_SIZEOF(itflags, ndim, nop);
    // 获取缓冲数据和轴数据的指针数组
    char **ptrs = NBF_PTRS(bufferdata), **ad_ptrs = NAD_PTRS(axisdata);
    char **buffers = NBF_BUFFERS(bufferdata);
    // 初始化迭代器索引、迭代结束位置、传输大小和单步长大小
    npy_intp iterindex, iterend, transfersize,
            singlestridesize, reduce_innersize = 0, reduce_outerdim = 0;
    int is_onestride = 0, any_buffered = 0;

    npy_intp *reduce_outerstrides = NULL;
    char **reduce_outerptrs = NULL;

    /*
     * Have to get this flag before npyiter_checkreducesize sets
     * it for the next iteration.
     */
    // 判断是否可以重用外部减少循环结构
    npy_bool reuse_reduce_loops = (prev_dataptrs != NULL) &&
                    ((itflags&NPY_ITFLAG_REUSE_REDUCE_LOOPS) != 0);

    // 计算轴数据增量
    npy_intp axisdata_incr = NIT_AXISDATA_SIZEOF(itflags, ndim, nop) /
                                NPY_SIZEOF_INTP;

    NPY_IT_DBG_PRINT("Iterator: Copying inputs to buffers\n");

    /* Calculate the size if using any buffers */
    // 计算使用缓冲时的传输大小
    iterindex = NIT_ITERINDEX(iter);
    iterend = NIT_ITEREND(iter);
    transfersize = NBF_BUFFERSIZE(bufferdata);
    if (transfersize > iterend - iterindex) {
        transfersize = iterend - iterindex;
    }

    /* If last time around, the reduce loop structure was full, we reuse it */
    if (reuse_reduce_loops) {
        // 如果设置了重用减少循环标志，则执行以下操作
        npy_intp full_transfersize, prev_reduce_outersize;

        // 获取上一次减少循环的外部尺寸
        prev_reduce_outersize = NBF_REDUCE_OUTERSIZE(bufferdata);
        // 获取减少循环的外部步长
        reduce_outerstrides = NBF_REDUCE_OUTERSTRIDES(bufferdata);
        // 获取减少循环的外部指针
        reduce_outerptrs = NBF_REDUCE_OUTERPTRS(bufferdata);
        // 获取减少循环的外部维度
        reduce_outerdim = NBF_REDUCE_OUTERDIM(bufferdata);
        // 获取减少循环的外部轴数据
        reduce_outeraxisdata = NIT_INDEX_AXISDATA(axisdata, reduce_outerdim);
        // 获取减少循环的内部尺寸
        reduce_innersize = NBF_SIZE(bufferdata);
        // 重置减少循环的位置指针
        NBF_REDUCE_POS(bufferdata) = 0;
        /*
         * 尝试使外部尺寸尽可能大。这允许它在处理外部减少循环的最后一部分时收缩，
         * 然后在下一个外部减少循环的开始时再次增长。
         */
        NBF_REDUCE_OUTERSIZE(bufferdata) = (NAD_SHAPE(reduce_outeraxisdata) -
                                            NAD_INDEX(reduce_outeraxisdata));
        // 计算完整传输尺寸
        full_transfersize = NBF_REDUCE_OUTERSIZE(bufferdata) * reduce_innersize;
        /* 如果完整传输尺寸超过缓冲区大小，则截断传输尺寸 */
        if (full_transfersize > NBF_BUFFERSIZE(bufferdata)) {
            // 调整外部尺寸以使其适合缓冲区大小
            NBF_REDUCE_OUTERSIZE(bufferdata) = transfersize / reduce_innersize;
            transfersize = NBF_REDUCE_OUTERSIZE(bufferdata) * reduce_innersize;
        }
        else {
            transfersize = full_transfersize;
        }
        // 如果本次外部尺寸大于上次外部尺寸，则禁止重用减少循环的缓冲区
        if (prev_reduce_outersize < NBF_REDUCE_OUTERSIZE(bufferdata)) {
            /*
             * 如果上次复制的数据量较少，即使指针匹配，也可能不安全重用缓冲区。
             */
            reuse_reduce_loops = 0;
        }
        // 设置缓冲区迭代结束位置
        NBF_BUFITEREND(bufferdata) = iterindex + reduce_innersize;

        // 调试打印重用的减少传输尺寸、内部尺寸和迭代器尺寸信息
        NPY_IT_DBG_PRINT3("Reused reduce transfersize: %d innersize: %d "
                        "itersize: %d\n",
                            (int)transfersize,
                            (int)reduce_innersize,
                            (int)NpyIter_GetIterSize(iter));
        // 调试打印减少的外部尺寸信息
        NPY_IT_DBG_PRINT1("Reduced reduce outersize: %d",
                            (int)NBF_REDUCE_OUTERSIZE(bufferdata));
    }
    /*
     * 如果存在任何减少操作数，可能需要减小尺寸，以免将相同的值复制到缓冲区两次，
     * 因为缓冲没有机制来自行合并值。
     */
    else if (itflags&NPY_ITFLAG_REDUCE) {
        // 如果迭代器标志指示进行reduce操作
        NPY_IT_DBG_PRINT("Iterator: Calculating reduce loops\n");
        // 打印调试信息：计算reduce循环
        transfersize = npyiter_checkreducesize(iter, transfersize,
                                                &reduce_innersize,
                                                &reduce_outerdim);
        // 检查reduce操作的大小，并更新相关参数
        NPY_IT_DBG_PRINT3("Reduce transfersize: %d innersize: %d "
                        "itersize: %d\n",
                            (int)transfersize,
                            (int)reduce_innersize,
                            (int)NpyIter_GetIterSize(iter));
        // 打印调试信息：显示reduce操作的传输大小、内部大小和迭代器大小

        reduce_outerstrides = NBF_REDUCE_OUTERSTRIDES(bufferdata);
        // 获取缓冲数据中的reduce外部步幅
        reduce_outerptrs = NBF_REDUCE_OUTERPTRS(bufferdata);
        // 获取缓冲数据中的reduce外部指针
        reduce_outeraxisdata = NIT_INDEX_AXISDATA(axisdata, reduce_outerdim);
        // 获取axisdata中reduce操作的外部轴数据
        NBF_SIZE(bufferdata) = reduce_innersize;
        // 设置缓冲数据中的大小为reduce内部大小
        NBF_REDUCE_POS(bufferdata) = 0;
        // 设置缓冲数据中的reduce位置为0
        NBF_REDUCE_OUTERDIM(bufferdata) = reduce_outerdim;
        // 设置缓冲数据中的reduce外部维度
        NBF_BUFITEREND(bufferdata) = iterindex + reduce_innersize;
        // 设置缓冲数据中的迭代器结束位置为当前索引加上reduce内部大小
        if (reduce_innersize == 0) {
            // 如果reduce内部大小为0
            NBF_REDUCE_OUTERSIZE(bufferdata) = 0;
            // 设置缓冲数据中的reduce外部大小为0
            return 0;
            // 返回0
        }
        else {
            // 否则，如果reduce内部大小不为0
            NBF_REDUCE_OUTERSIZE(bufferdata) = transfersize/reduce_innersize;
            // 计算并设置缓冲数据中的reduce外部大小为传输大小除以reduce内部大小
        }
    }
    else {
        // 如果不是reduce操作
        NBF_SIZE(bufferdata) = transfersize;
        // 设置缓冲数据中的大小为传输大小
        NBF_BUFITEREND(bufferdata) = iterindex + transfersize;
        // 设置缓冲数据中的迭代器结束位置为当前索引加上传输大小
    }

    /* Calculate the maximum size if using a single stride and no buffers */
    // 如果使用单个步幅且无缓冲，则计算最大大小
    singlestridesize = NAD_SHAPE(axisdata)-NAD_INDEX(axisdata);
    // 计算单个步幅大小为axisdata的形状减去索引
    if (singlestridesize > iterend - iterindex) {
        // 如果单个步幅大小大于迭代结束减去当前索引
        singlestridesize = iterend - iterindex;
        // 则设置单个步幅大小为迭代结束减去当前索引
    }
    if (singlestridesize >= transfersize) {
        // 如果单个步幅大小大于等于传输大小
        is_onestride = 1;
        // 设置单步幅标志为1
    }

    NpyIter_TransferInfo *transferinfo = NBF_TRANSFERINFO(bufferdata);
    // 获取缓冲数据中的传输信息

    /*
     * If buffering wasn't needed, we can grow the inner
     * loop to as large as possible.
     *
     * TODO: Could grow REDUCE loop too with some more logic above.
     */
    // 如果不需要缓冲，我们可以尽可能扩展内部循环
    if (!any_buffered && (itflags&NPY_ITFLAG_GROWINNER) &&
                        !(itflags&NPY_ITFLAG_REDUCE)) {
        // 如果没有任何缓冲并且需要扩展内部循环且不是reduce操作
        if (singlestridesize > transfersize) {
            // 如果单个步幅大小大于传输大小
            NPY_IT_DBG_PRINT2("Iterator: Expanding inner loop size "
                    "from %d to %d since buffering wasn't needed\n",
                    (int)NBF_SIZE(bufferdata), (int)singlestridesize);
            // 打印调试信息：扩展内部循环大小
            NBF_SIZE(bufferdata) = singlestridesize;
            // 设置缓冲数据中的大小为单个步幅大小
            NBF_BUFITEREND(bufferdata) = iterindex + singlestridesize;
            // 设置缓冲数据中的迭代器结束位置为当前索引加上单个步幅大小
        }
    }

    NPY_IT_DBG_PRINT1("Any buffering needed: %d\n", any_buffered);
    // 打印调试信息：是否需要任何缓冲

    NPY_IT_DBG_PRINT1("Iterator: Finished copying inputs to buffers "
                        "(buffered size is %d)\n", (int)NBF_SIZE(bufferdata));
    // 打印调试信息：完成将输入复制到缓冲区（缓冲区大小为...）
    return 0;
    // 返回0
/**
 * This function clears any references still held by the buffers and should
 * only be used to discard buffers if an error occurred.
 *
 * @param iter Iterator object for which buffers are to be cleared
 */
NPY_NO_EXPORT void
npyiter_clear_buffers(NpyIter *iter)
{
    // Retrieve the number of operands and buffer data associated with the iterator
    int nop = iter->nop;
    NpyIter_BufferData *bufferdata = NIT_BUFFERDATA(iter);

    // If the buffers are already empty, no further action is needed
    if (NBF_SIZE(bufferdata) == 0) {
        return;
    }

    /*
     * Save and temporarily clear any current Python exception information
     * to safely perform buffer cleanup operations.
     */
    PyObject *type, *value, *traceback;
    PyErr_Fetch(&type,  &value, &traceback);

    /* Cleanup any buffers with references */
    char **buffers = NBF_BUFFERS(bufferdata);
    NpyIter_TransferInfo *transferinfo = NBF_TRANSFERINFO(bufferdata);
    PyArray_Descr **dtypes = NIT_DTYPES(iter);
    npyiter_opitflags *op_itflags = NIT_OPITFLAGS(iter);

    // Iterate over each operand to clear its associated buffer if necessary
    for (int iop = 0; iop < nop; ++iop, ++buffers) {
        // Skip operands without a clear function or not using buffers
        if (transferinfo[iop].clear.func == NULL ||
                !(op_itflags[iop] & NPY_OP_ITFLAG_USINGBUFFER)) {
            continue;
        }
        // Skip buffers that are already cleared (NULL)
        if (*buffers == 0) {
            continue;
        }
        // Calculate item size of the operand's data type
        int itemsize = dtypes[iop]->elsize;
        // Call the clear function to release the buffer
        if (transferinfo[iop].clear.func(NULL, dtypes[iop], *buffers,
                NBF_SIZE(bufferdata), itemsize,
                transferinfo[iop].clear.auxdata) < 0) {
            /* This should never fail; if it does, write an unraisable exception */
            PyErr_WriteUnraisable(NULL);
        }
    }

    /* Signal that the buffers are now empty */
    NBF_SIZE(bufferdata) = 0;
    // Restore any previous Python exception information
    PyErr_Restore(type, value, traceback);
}
*/

/*
 * This checks how much space can be buffered without encountering the
 * same value twice, or for operands whose innermost stride is zero,
 * without encountering a different value.  By reducing the buffered
 * amount to this size, reductions can be safely buffered.
 *
 * Reductions are buffered with two levels of looping, to avoid
 * frequent copying to the buffers.  The return value is the overall
 * buffer size, and when the flag NPY_ITFLAG_REDUCE is set, reduce_innersize
 * receives the size of the inner of the two levels of looping.
 *
 * The value placed in reduce_outerdim is the index into the AXISDATA
 * for where the second level of the double loop begins.
 *
 * The return value is always a multiple of the value placed in
 * reduce_innersize.
 */
static npy_intp
npyiter_checkreducesize(NpyIter *iter, npy_intp count,
                        npy_intp *reduce_innersize,
                        npy_intp *reduce_outerdim)
{
    // Retrieve flags and dimensions related to the iterator
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    int idim, ndim = NIT_NDIM(iter);
    int iop, nop = NIT_NOP(iter);

    // Variables related to axis data and iteration
    NpyIter_AxisData *axisdata;
    npy_intp sizeof_axisdata;
    npy_intp coord, shape, *strides;
    npy_intp reducespace = 1, factor;
    # 声明一个布尔变量 nonzerocoord，用于记录是否存在非零坐标
    npy_bool nonzerocoord;

    # 获取迭代器的操作标志结构体指针
    npyiter_opitflags *op_itflags = NIT_OPITFLAGS(iter);
    
    # 声明一个字符数组，用于存储第一个操作数的步长信息
    char stride0op[NPY_MAXARGS];

    # 默认情况下不进行外部轴的减少操作
    *reduce_outerdim = 0;

    # 如果数组维度为1或者元素个数为0，则无需计算任何内容，直接返回元素个数
    if (ndim == 1 || count == 0) {
        *reduce_innersize = count;
        return count;
    }

    # 计算存储 axisdata 所需的空间大小
    sizeof_axisdata = NIT_AXISDATA_SIZEOF(itflags, ndim, nop);
    axisdata = NIT_AXISDATA(iter);

    # 标记在内部循环中哪些 REDUCE 操作数的步长为0
    strides = NAD_STRIDES(axisdata);
    for (iop = 0; iop < nop; ++iop) {
        stride0op[iop] = (op_itflags[iop]&NPY_OP_ITFLAG_REDUCE) &&
                           (strides[iop] == 0);
        # 打印调试信息，指示操作数在内部循环中是否具有步长为0
        NPY_IT_DBG_PRINT2("Iterator: Operand %d has stride 0 in "
                        "the inner loop? %d\n", iop, (int)stride0op[iop]);
    }

    # 获取 axisdata 的形状和坐标信息
    shape = NAD_SHAPE(axisdata);
    coord = NAD_INDEX(axisdata);
    
    # 更新 reducespace 的值，计算剩余空间大小
    reducespace += (shape-coord-1);
    
    # 计算因子，用于后续的计算
    factor = shape;
    
    # 将 axisdata 指针向前移动一个位置
    NIT_ADVANCE_AXISDATA(axisdata, 1);

    # 根据第一个坐标初始化 nonzerocoord 变量
    nonzerocoord = (coord != 0);

    # 沿着 axisdata 前进，计算可用空间
    # 迭代每个维度，直到达到维度数或者缩减空间已满
    for (idim = 1; idim < ndim && reducespace < count;
                                ++idim, NIT_ADVANCE_AXISDATA(axisdata, 1)) {
        # 调试输出内部循环的缩减空间和计数值
        NPY_IT_DBG_PRINT2("Iterator: inner loop reducespace %d, count %d\n",
                                (int)reducespace, (int)count);

        # 获取当前轴的步长数组
        strides = NAD_STRIDES(axisdata);
        # 遍历所有操作
        for (iop = 0; iop < nop; ++iop) {
            /*
             * 如果一个缩减步长从零变为非零，或者从非零变为零，
             * 这是数据不再是同一个元素或者会重复的点，
             * 如果缓冲区从所有零的多索引开始到此点，
             * 则给出缩减内部大小。
             */
            if((stride0op[iop] && (strides[iop] != 0)) ||
                        (!stride0op[iop] &&
                         (strides[iop] == 0) &&
                         (op_itflags[iop]&NPY_OP_ITFLAG_REDUCE))) {
                # 调试输出缩减操作限制缓冲区大小到reducespace
                NPY_IT_DBG_PRINT1("Iterator: Reduce operation limits "
                                    "buffer to %d\n", (int)reducespace);
                /*
                 * 如果已经找到的元素比计数还多，或者
                 * 起始坐标不是零，则两级循环是不必要的/无法完成，因此返回。
                 */
                if (count <= reducespace) {
                    *reduce_innersize = count;
                    # 设置重用缩减循环标志并返回计数
                    NIT_ITFLAGS(iter) |= NPY_ITFLAG_REUSE_REDUCE_LOOPS;
                    return count;
                }
                else if (nonzerocoord) {
                    if (reducespace < count) {
                        count = reducespace;
                    }
                    *reduce_innersize = count;
                    # 注意：这类似于下面的（coord != 0）情况。
                    NIT_ITFLAGS(iter) &= ~NPY_ITFLAG_REUSE_REDUCE_LOOPS;
                    return count;
                }
                else {
                    *reduce_innersize = reducespace;
                    break;
                }
            }
        }
        # 如果提前跳出循环，表示找到了reduce_innersize
        if (iop != nop) {
            # 调试输出找到第一个不是缩减的维度
            NPY_IT_DBG_PRINT2("Iterator: Found first dim not "
                            "reduce (%d of %d)\n", iop, nop);
            break;
        }

        # 获取当前轴的形状和索引
        shape = NAD_SHAPE(axisdata);
        coord = NAD_INDEX(axisdata);
        # 如果索引不为零，设置非零坐标标志
        if (coord != 0) {
            nonzerocoord = 1;
        }
        # 更新缩减空间和因子
        reducespace += (shape-coord-1) * factor;
        factor *= shape;
    }

    /*
     * 如果存在任何非零坐标，缩减内部循环不适合缓冲区大小，
     * 或者缩减内部循环覆盖了整个迭代大小，则无法进行双重循环。
     */
    // 检查是否满足不重用 reduce 循环的条件：非零坐标或者计数小于减少空间的数量，或者已经达到最后一个维度
    if (nonzerocoord || count < reducespace || idim == ndim) {
        // 如果减少空间的数量小于当前计数，则将计数更新为减少空间的数量
        if (reducespace < count) {
            count = reducespace;
        }
        // 更新 reduce_innersize 指针所指向的值为当前计数
        *reduce_innersize = count;
        /* 在这种情况下，我们不能重用 reduce 循环 */
        NIT_ITFLAGS(iter) &= ~NPY_ITFLAG_REUSE_REDUCE_LOOPS;
        // 返回当前计数值
        return count;
    }

    // 从轴数据中获取坐标值
    coord = NAD_INDEX(axisdata);
    // 如果坐标值不为零
    if (coord != 0) {
        /*
         * 在这种情况下，只有在复制的数据量不超过当前轴数时才能安全地重用缓冲区，
         * 这种情况通常出现在已经启用了 reuse_reduce_loops 的情况下。
         * 当 idim 循环立即返回时，原则上是可以的。
         */
        NIT_ITFLAGS(iter) &= ~NPY_ITFLAG_REUSE_REDUCE_LOOPS;
    }
    else {
        /* 在这种情况下，我们可以重用 reduce 循环 */
        NIT_ITFLAGS(iter) |= NPY_ITFLAG_REUSE_REDUCE_LOOPS;
    }

    // 更新 reduce_innersize 指针所指向的值为减少空间的数量
    *reduce_innersize = reducespace;
    // 计算新的 count 值，即将当前计数除以减少空间的数量
    count /= reducespace;

    // 打印调试信息，显示 reduce_innersize 和计数值
    NPY_IT_DBG_PRINT2("Iterator: reduce_innersize %d count /ed %d\n",
                    (int)reducespace, (int)count);

    /*
     * 继续遍历剩余的维度。如果有两个分离的减少轴，我们可能需要再次缩短缓冲区。
     */
    // 更新 reduce_outerdim 指针所指向的值为当前维度 idim
    *reduce_outerdim = idim;
    // 重置 reducespace 为 1，重置 factor 为 1
    reducespace = 1;
    factor = 1;
    /* 指示当前级别的 REDUCE 操作数是否具有零步长 */
    strides = NAD_STRIDES(axisdata);
    // 遍历操作数，标记是否有零步长的 REDUCE 操作数
    for (iop = 0; iop < nop; ++iop) {
        stride0op[iop] = (op_itflags[iop]&NPY_OP_ITFLAG_REDUCE) &&
                           (strides[iop] == 0);
        // 打印调试信息，显示操作数是否在外部循环中具有零步长
        NPY_IT_DBG_PRINT2("Iterator: Operand %d has stride 0 in "
                        "the outer loop? %d\n", iop, (int)stride0op[iop]);
    }
    // 获取轴的形状
    shape = NAD_SHAPE(axisdata);
    // 更新 reducespace，根据坐标和因子的乘积计算
    reducespace += (shape-coord-1) * factor;
    // 更新 factor，乘以当前轴的形状
    factor *= shape;
    // 推进轴数据到下一个维度
    NIT_ADVANCE_AXISDATA(axisdata, 1);
    // 增加 idim，表示当前处理的维度
    ++idim;
    for (; idim < ndim && reducespace < count;
                                ++idim, NIT_ADVANCE_AXISDATA(axisdata, 1)) {
        # 执行外层循环，直到达到维度数目或者减少的空间小于计数
        NPY_IT_DBG_PRINT2("Iterator: outer loop reducespace %d, count %d\n",
                                (int)reducespace, (int)count);
        # 调试输出当前迭代的减少空间和计数值

        strides = NAD_STRIDES(axisdata);
        # 获取当前轴数据的步幅数组
        for (iop = 0; iop < nop; ++iop) {
            # 遍历操作数的数量
            /*
             * 如果一个减少步幅从零变为非零，或者反之，则数据将停止
             * 成为同一个元素或者重复，并且如果缓冲区从一个全零的
             * 多索引开始到这一点，给我们减少的内部大小。
             */
            if((stride0op[iop] && (strides[iop] != 0)) ||
                        (!stride0op[iop] &&
                         (strides[iop] == 0) &&
                         (op_itflags[iop]&NPY_OP_ITFLAG_REDUCE))) {
                # 如果条件满足，说明找到了减少操作的边界
                NPY_IT_DBG_PRINT1("Iterator: Reduce operation limits "
                                    "buffer to %d\n", (int)reducespace);
                # 调试输出减少操作限制缓冲区大小的消息
                /*
                 * 这终止了我们双重循环的外层级别。
                 */
                if (count <= reducespace) {
                    return count * (*reduce_innersize);
                    # 如果计数小于或等于减少的空间，则返回计数乘以减少的内部大小
                }
                else {
                    return reducespace * (*reduce_innersize);
                    # 否则返回减少的空间乘以减少的内部大小
                }
            }
        }

        shape = NAD_SHAPE(axisdata);
        # 获取当前轴数据的形状
        coord = NAD_INDEX(axisdata);
        # 获取当前轴数据的索引
        if (coord != 0) {
            nonzerocoord = 1;
            # 如果索引不为零，设置非零坐标标志为1
        }
        reducespace += (shape-coord-1) * factor;
        # 更新减少的空间，乘以形状减去索引减一再乘以因子
        factor *= shape;
        # 更新因子，乘以形状
    }

    if (reducespace < count) {
        count = reducespace;
        # 如果减少的空间小于计数，则更新计数为减少的空间
    }
    return count * (*reduce_innersize);
    # 返回最终计数乘以减少的内部大小
}

NPY_NO_EXPORT npy_bool
npyiter_has_writeback(NpyIter *iter)
{
    // 定义变量 iop 和 nop，分别表示操作数和操作标志数
    int iop, nop;
    // 声明指向操作标志的指针 op_itflags
    npyiter_opitflags *op_itflags;
    
    // 检查迭代器是否为 NULL
    if (iter == NULL) {
        // 如果迭代器为 NULL，返回 0（假）
        return 0;
    }
    
    // 获取迭代器中的操作数
    nop = NIT_NOP(iter);
    // 获取迭代器中的操作标志数组
    op_itflags = NIT_OPITFLAGS(iter);

    // 循环遍历所有操作
    for (iop = 0; iop < nop; iop++) {
        // 检查当前操作的标志是否包含写回标志 NPY_OP_ITFLAG_HAS_WRITEBACK
        if (op_itflags[iop] & NPY_OP_ITFLAG_HAS_WRITEBACK) {
            // 如果有写回标志，返回 NPY_TRUE（真）
            return NPY_TRUE;
        }
    }
    
    // 如果没有任何操作包含写回标志，则返回 NPY_FALSE（假）
    return NPY_FALSE;
}
#undef NPY_ITERATOR_IMPLEMENTATION_CODE
```