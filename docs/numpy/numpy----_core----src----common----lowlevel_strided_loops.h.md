# `.\numpy\numpy\_core\src\common\lowlevel_strided_loops.h`

```
// 检查是否定义了 NUMPY_CORE_SRC_COMMON_LOWLEVEL_STRIDED_LOOPS_H_，如果没有则定义
#ifndef NUMPY_CORE_SRC_COMMON_LOWLEVEL_STRIDED_LOOPS_H_
// 宏定义，用于导入相关头文件
#define NUMPY_CORE_SRC_COMMON_LOWLEVEL_STRIDED_LOOPS_H_
// 导入通用头文件
#include "common.h"
// 导入 numpy 配置文件
#include "npy_config.h"
// 导入数组方法头文件
#include "array_method.h"
// 导入数据类型转换头文件
#include "dtype_transfer.h"
// 导入内存重叠头文件
#include "mem_overlap.h"
// 导入映射头文件
#include "mapping.h"

/* For PyArray_ macros used below */
// 导入 ndarray 对象头文件
#include "numpy/ndarrayobject.h"

/*
 * 注意：此 API 目前应保持私有，以便进行进一步的细化。
 * 我认为 'aligned' 机制需要进行更改，例如。
 *
 * 注意：2018 年进行了更新，以区分 "true" 和 "uint" 对齐。
 */

/*
 * 此函数指针用于输入任意跨步的一维数组段并输出相同大小的任意跨步的数组段的一元操作。
 * 当步长或项目大小具有特定已知值时，它可以是完全通用的函数或专用函数。
 *
 * 一元操作的示例包括直接复制、字节交换和强制转换操作，
 *
 * 'transferdata' 参数略微特殊，遵循在 ndarraytypes.h 中定义的通用辅助数据模式
 * 使用 NPY_AUXDATA_CLONE 和 NPY_AUXDATA_FREE 处理这些数据。
 */
// TODO: FIX! That comment belongs to something now in array-method

/*
 * 这是用于指向与 PyArrayMethod_StridedLoop 完全相同的函数指针，
 * 但具有额外的掩码来控制被转换的值。
 *
 * TODO：我们应该将这个掩码“功能”移动到 ArrayMethod 本身去
 * 可能。虽然对于 NumPy 内部的事情来说，这个方法运作良好，并且暴露它应该经过深思熟虑，
 * 以便在可能的情况下有用于 NumPy 之外。
 *
 * 特别地，如果 mask[i*mask_stride] 为真，则对第 'i' 元素进行操作。
 */
typedef int (PyArray_MaskedStridedUnaryOp)(
        PyArrayMethod_Context *context, char *const *args,
        const npy_intp *dimensions, const npy_intp *strides,
        npy_bool *mask, npy_intp mask_stride,
        NpyAuxData *auxdata);

/*
 * 返回一个指向用于复制步进内存的专用函数的函数指针。如果输入出现问题，则返回 NULL。
 *
 * aligned:
 *      如果 src 和 dst 指针总是指向与 dtype->elsize 相等的 uint 对齐的位置，则为1，否则为0。
 * src_stride:
 *      如果 src 步幅总是相同的，则应为 src 步幅，否则为 NPY_MAX_INTP。
 * dst_stride:
 *      如果 dst 步幅总是相同的，则应为 dst 步幅，否则为 NPY_MAX_INTP。
 * itemsize:
 *      如果项目大小总是相同的，则应为项目大小，否则为0。
 *
 */
// 导出函数
NPY_NO_EXPORT PyArrayMethod_StridedLoop *
PyArray_GetStridedCopyFn(int aligned,
                        npy_intp src_stride, npy_intp dst_stride,
                        npy_intp itemsize);
/*
 * 返回一个指向特定函数的函数指针，用于复制和交换步进内存。假设每个元素都是单个需要交换的值。
 * 这个函数假设了步进内存是对齐的，并使用了给定的 src_stride 和 dst_stride 参数。
 * 
 * 参数和 PyArray_GetStridedCopyFn 中描述的一样。
 */
NPY_NO_EXPORT PyArrayMethod_StridedLoop *
PyArray_GetStridedCopySwapFn(int aligned,
                            npy_intp src_stride, npy_intp dst_stride,
                            npy_intp itemsize);

/*
 * 返回一个指向特定函数的函数指针，用于复制和交换步进内存。假设每个元素是一对需要交换的值。
 * 这个函数假设了步进内存是对齐的，并使用了给定的 src_stride 和 dst_stride 参数。
 * 
 * 参数和 PyArray_GetStridedCopyFn 中描述的一样。
 */
NPY_NO_EXPORT PyArrayMethod_StridedLoop *
PyArray_GetStridedCopySwapPairFn(int aligned,
                            npy_intp src_stride, npy_intp dst_stride,
                            npy_intp itemsize);

/*
 * 返回一个传输函数和传输数据对，将数据从源复制到目标，如果数据不适合，则截断它，并且如果空间过多，则用零字节填充。
 * 这个函数假设了步进内存是对齐的，并使用了给定的 src_stride 和 dst_stride 参数。
 * 
 * 参数和 PyArray_GetStridedCopyFn 中描述的一样。
 * 
 * 返回 NPY_SUCCEED 或 NPY_FAIL。
 */
NPY_NO_EXPORT int
PyArray_GetStridedZeroPadCopyFn(int aligned, int unicode_swap,
                            npy_intp src_stride, npy_intp dst_stride,
                            npy_intp src_itemsize, npy_intp dst_itemsize,
                            PyArrayMethod_StridedLoop **outstransfer,
                            NpyAuxData **outtransferdata);

/*
 * 对于内置数值类型之间的转换，返回一个函数指针，用于从 src_type_num 转换到 dst_type_num。
 * 如果不支持某个转换，则返回 NULL 而不设置 Python 异常。
 */
NPY_NO_EXPORT PyArrayMethod_StridedLoop *
PyArray_GetStridedNumericCastFn(int aligned,
                            npy_intp src_stride, npy_intp dst_stride,
                            int src_type_num, int dst_type_num);

/*
 * 获取一个操作，该操作复制给定 dtype 的元素，如果 dtype 不是 NBO 则进行字节顺序交换。
 * 
 * 返回 NPY_SUCCEED 或 NPY_FAIL。
 */
NPY_NO_EXPORT int
PyArray_GetDTypeCopySwapFn(int aligned,
                            npy_intp src_stride, npy_intp dst_stride,
                            PyArray_Descr *dtype,
                            PyArrayMethod_StridedLoop **outstransfer,
                            NpyAuxData **outtransferdata);
/*
 * If it's possible, gives back a transfer function which casts and/or
 * byte swaps data with the dtype 'src_dtype' into data with the dtype
 * 'dst_dtype'.  If the outtransferdata is populated with a non-NULL value,
 * it must be deallocated with the NPY_AUXDATA_FREE
 * function when the transfer function is no longer required.
 *
 * aligned:
 *      Should be 1 if the src and dst pointers always point to
 *      locations at which a uint of equal size to dtype->elsize
 *      would be aligned, 0 otherwise.
 * src_stride:
 *      Should be the src stride if it will always be the same,
 *      NPY_MAX_INTP otherwise.
 * dst_stride:
 *      Should be the dst stride if it will always be the same,
 *      NPY_MAX_INTP otherwise.
 * src_dtype:
 *      The data type of source data. Must not be NULL.
 * dst_dtype:
 *      The data type of destination data.  If this is NULL and
 *      move_references is 1, a transfer function which decrements
 *      source data references is produced.
 * move_references:
 *      If 0, the destination data gets new reference ownership.
 *      If 1, the references from the source data are moved to
 *      the destination data.
 * cast_info:
 *      A pointer to an (uninitialized) `NPY_cast_info` struct, the caller
 *      must call `NPY_cast_info_xfree` on it (except on error) and handle
 *      its memory livespan.
 * out_needs_api:
 *      If this is non-NULL, and the transfer function produced needs
 *      to call into the (Python) API, this gets set to 1.  This
 *      remains untouched if no API access is required.
 *
 * WARNING: If you set move_references to 1, it is best that src_stride is
 *          never zero when calling the transfer function.  Otherwise, the
 *          first destination reference will get the value and all the rest
 *          will get NULL.
 *
 * Returns NPY_SUCCEED or NPY_FAIL.
 */
NPY_NO_EXPORT int
PyArray_GetDTypeTransferFunction(int aligned,
                            npy_intp src_stride, npy_intp dst_stride,
                            PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
                            int move_references,
                            NPY_cast_info *cast_info,
                            NPY_ARRAYMETHOD_FLAGS *out_flags);

/*
 * If it's possible, gives back a transfer function which copies fields
 * from one structured dtype to another, optionally moving references.
 *
 * aligned:
 *      Should be 1 if the src and dst pointers always point to
 *      locations at which a uint of equal size to dtype->elsize
 *      would be aligned, 0 otherwise.
 * src_stride:
 *      Should be the src stride if it will always be the same,
 *      NPY_MAX_INTP otherwise.
 * dst_stride:
 *      Should be the dst stride if it will always be the same,
 *      NPY_MAX_INTP otherwise.
 * src_dtype:
 *      The data type of source data. Must not be NULL.
 * dst_dtype:
 *      The data type of destination data.  If this is NULL and
 *      move_references is 1, a transfer function which decrements
 *      source data references is produced.
 * move_references:
 *      If 0, the destination data gets new reference ownership.
 *      If 1, the references from the source data are moved to
 *      the destination data.
 * out_stransfer:
 *      A pointer to a `PyArrayMethod_StridedLoop` struct pointer,
 *      which will contain the transfer function upon successful return.
 * out_transferdata:
 *      A pointer to an `NpyAuxData` struct pointer, which will
 *      contain auxiliary data needed for the transfer function.
 * out_flags:
 *      Pointer to `NPY_ARRAYMETHOD_FLAGS` where the flags describing
 *      the transfer function will be stored.
 *
 * Returns NPY_SUCCEED or NPY_FAIL.
 */
NPY_NO_EXPORT int
get_fields_transfer_function(int aligned,
        npy_intp src_stride, npy_intp dst_stride,
        PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
        int move_references,
        PyArrayMethod_StridedLoop **out_stransfer,
        NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *out_flags);

/*
 * If it's possible, gives back a transfer function which handles
 * subarray assignment from one structured dtype to another,
 * optionally moving references.
 *
 * aligned:
 *      Should be 1 if the src and dst pointers always point to
 *      locations at which a uint of equal size to dtype->elsize
 *      would be aligned, 0 otherwise.
 * src_stride:
 *      Should be the src stride if it will always be the same,
 *      NPY_MAX_INTP otherwise.
 * dst_stride:
 *      Should be the dst stride if it will always be the same,
 *      NPY_MAX_INTP otherwise.
 * src_dtype:
 *      The data type of source data. Must not be NULL.
 * dst_dtype:
 *      The data type of destination data.  If this is NULL and
 *      move_references is 1, a transfer function which decrements
 *      source data references is produced.
 * move_references:
 *      If 0, the destination data gets new reference ownership.
 *      If 1, the references from the source data are moved to
 *      the destination data.
 * out_stransfer:
 *      A pointer to a `PyArrayMethod_StridedLoop` struct pointer,
 *      which will contain the transfer function upon successful return.
 * out_transferdata:
 *      A pointer to an `NpyAuxData` struct pointer, which will
 *      contain auxiliary data needed for the transfer function.
 * out_flags:
 *      Pointer to `NPY_ARRAYMETHOD_FLAGS` where the flags describing
 *      the transfer function will be stored.
 *
 * Returns NPY_SUCCEED or NPY_FAIL.
 */
NPY_NO_EXPORT int
get_subarray_transfer_function(int aligned,
        npy_intp src_stride, npy_intp dst_stride,
        PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
        int move_references,
        PyArrayMethod_StridedLoop **out_stransfer,
        NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *out_flags);
/*
 * This is identical to PyArray_GetDTypeTransferFunction, but returns a
 * transfer function which also takes a mask as a parameter.  The mask is used
 * to determine which values to copy, and data is transferred exactly when
 * mask[i*mask_stride] is true.
 *
 * If move_references is true, values which are not copied to the
 * destination will still have their source reference decremented.
 *
 * If mask_dtype is NPY_BOOL or NPY_UINT8, each full element is either
 * transferred or not according to the mask as described above. If
 * dst_dtype and mask_dtype are both struct dtypes, their names must
 * match exactly, and the dtype of each leaf field in mask_dtype must
 * be either NPY_BOOL or NPY_UINT8.
 */
NPY_NO_EXPORT int
PyArray_GetMaskedDTypeTransferFunction(int aligned,
                            npy_intp src_stride,
                            npy_intp dst_stride,
                            npy_intp mask_stride,
                            PyArray_Descr *src_dtype,
                            PyArray_Descr *dst_dtype,
                            PyArray_Descr *mask_dtype,
                            int move_references,
                            NPY_cast_info *cast_info,
                            NPY_ARRAYMETHOD_FLAGS *out_flags);
/*
 * Casts the specified number of elements from 'src' with data type
 * 'src_dtype' to 'dst' with 'dst_dtype'. See
 * PyArray_GetDTypeTransferFunction for more details.
 *
 * Returns NPY_SUCCEED or NPY_FAIL.
 */
NPY_NO_EXPORT int
PyArray_CastRawArrays(npy_intp count,
                      char *src, char *dst,
                      npy_intp src_stride, npy_intp dst_stride,
                      PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
                      int move_references);
/*
 * 这两个函数用于复制或转换一个n维数组的数据到/从一个一维跨步缓冲区中。
 * 这些函数仅会使用提供的dst_stride/src_stride和dst_strides[0]/src_strides[0]调用'stransfer'，
 * 因此调用者可以使用这些值来专门化函数。
 * 注意，即使ndim == 0，也需要设置所有内容，就好像ndim == 1一样。
 *
 * 返回值是无法复制的元素数量。返回值为0表示已复制所有元素，返回值大于0表示在复制'count'个元素之前到达了n维数组的末尾。
 * 返回值为负表示发生了错误。
 *
 * ndim:
 *      n维数组的维度数。
 * dst/src/mask:
 *      目标、源或掩码的起始指针。
 * dst_stride/src_stride/mask_stride:
 *      一维跨步缓冲区的跨步。
 * dst_strides/src_strides:
 *      n维数组的跨步。
 * dst_strides_inc/src_strides_inc:
 *      添加到..._strides指针以获取下一个跨步的增量。
 * coords:
 *      n维数组中的起始坐标。
 * coords_inc:
 *      添加到坐标指针以获取下一个坐标的增量。
 * shape:
 *      n维数组的形状。
 * shape_inc:
 *      添加到形状指针以获取下一个形状条目的增量。
 * count:
 *      要传输的元素数量。
 * src_itemsize:
 *      每个元素的大小。如果在不同大小的元素之间传输（例如强制转换操作），则'stransfer'函数应针对此进行专门化，
 *      在这种情况下，'stransfer'将使用此参数作为源项大小。
 * cast_info:
 *      指向NPY_cast_info结构的指针，该结构概述了执行强制转换所需的所有信息。
 */
NPY_NO_EXPORT npy_intp
PyArray_TransferNDimToStrided(npy_intp ndim,
                char *dst, npy_intp dst_stride,
                char *src, npy_intp const *src_strides, npy_intp src_strides_inc,
                npy_intp const *coords, npy_intp coords_inc,
                npy_intp const *shape, npy_intp shape_inc,
                npy_intp count, npy_intp src_itemsize,
                NPY_cast_info *cast_info);

/*
 * 将一维跨步缓冲区的数据传输到n维数组中的这个函数。
 * 这些函数仅会使用提供的dst_strides/src_strides和dst_strides_inc调用'stransfer'，
 * 因此调用者可以使用这些值来专门化函数。
 *
 * 返回值是无法复制的元素数量。返回值为0表示已复制所有元素，返回值大于0表示在复制'count'个元素之前到达了n维数组的末尾。
 * 返回值为负表示发生了错误。
 *
 * ndim:
 *      n维数组的维度数。
 * dst/src/mask:
 *      目标、源或掩码的起始指针。
 * dst_strides/src_strides:
 *      n维数组的跨步。
 * dst_strides_inc:
 *      添加到dst_strides指针以获取下一个跨步的增量。
 * src_stride:
 *      一维跨步缓冲区的跨步。
 * coords:
 *      n维数组中的起始坐标。
 * coords_inc:
 *      添加到坐标指针以获取下一个坐标的增量。
 * shape:
 *      n维数组的形状。
 * shape_inc:
 *      添加到形状指针以获取下一个形状条目的增量。
 * count:
 *      要传输的元素数量。
 * src_itemsize:
 *      每个元素的大小。如果在不同大小的元素之间传输（例如强制转换操作），则'stransfer'函数应针对此进行专门化，
 *      在这种情况下，'stransfer'将使用此参数作为源项大小。
 * cast_info:
 *      指向NPY_cast_info结构的指针，该结构概述了执行强制转换所需的所有信息。
 */
NPY_NO_EXPORT npy_intp
PyArray_TransferStridedToNDim(npy_intp ndim,
                char *dst, npy_intp const *dst_strides, npy_intp dst_strides_inc,
                char *src, npy_intp src_stride,
                npy_intp const *coords, npy_intp coords_inc,
                npy_intp const *shape, npy_intp shape_inc,
                npy_intp count, npy_intp src_itemsize,
                NPY_cast_info *cast_info);
/*
 * Transfers data from a strided array to an N-dimensional array with optional masking.
 * Coordinates, strides, shapes, and casting information are provided to control the transfer.
 * Returns void.
 */
PyArray_TransferMaskedStridedToNDim(npy_intp ndim,
                char *dst, npy_intp const *dst_strides, npy_intp dst_strides_inc,
                char *src, npy_intp src_stride,
                npy_bool *mask, npy_intp mask_stride,
                npy_intp const *coords, npy_intp coords_inc,
                npy_intp const *shape, npy_intp shape_inc,
                npy_intp count, npy_intp src_itemsize,
                NPY_cast_info *cast_info);

/*
 * Retrieves values from a simple mapping iterator at specified indices, storing results.
 * Handles alignment and casting information provided.
 * Returns an integer indicating success (0) or failure (-1).
 */
NPY_NO_EXPORT int
mapiter_trivial_get(
        PyArrayObject *self, PyArrayObject *ind, PyArrayObject *result,
        int is_aligned, NPY_cast_info *cast_info);

/*
 * Sets values into a simple mapping iterator at specified indices, using given data.
 * Handles alignment and casting information provided.
 * Returns an integer indicating success (0) or failure (-1).
 */
NPY_NO_EXPORT int
mapiter_trivial_set(
        PyArrayObject *self, PyArrayObject *ind, PyArrayObject *result,
        int is_aligned, NPY_cast_info *cast_info);

/*
 * Retrieves values from a more complex mapping iterator object.
 * Handles casting and alignment considerations, along with specified flags.
 * Returns an integer indicating success (0) or failure (-1).
 */
NPY_NO_EXPORT int
mapiter_get(
        PyArrayMapIterObject *mit, NPY_cast_info *cast_info,
        NPY_ARRAYMETHOD_FLAGS flags, int is_aligned);

/*
 * Sets values into a more complex mapping iterator object.
 * Handles casting and alignment considerations, along with specified flags.
 * Returns an integer indicating success (0) or failure (-1).
 */
NPY_NO_EXPORT int
mapiter_set(
        PyArrayMapIterObject *mit, NPY_cast_info *cast_info,
        NPY_ARRAYMETHOD_FLAGS flags, int is_aligned);

/*
 * Prepares shape and strides for a simple raw array iteration.
 * Orders strides into FORTRAN order, reverses negative strides, and coalesces axes where possible.
 * Returns 0 on success, -1 on failure.
 *
 * Intended for lightweight iteration over raw arrays without PyArrayObject buffering.
 * Used together with NPY_RAW_ITER_START and NPY_RAW_ITER_ONE_NEXT for loop handling.
 */
NPY_NO_EXPORT int
PyArray_PrepareOneRawArrayIter(int ndim, npy_intp const *shape,
                            char *data, npy_intp const *strides,
                            int *out_ndim, npy_intp *out_shape,
                            char **out_data, npy_intp *out_strides);

/*
 * Prepares shape and strides for two operands' raw array iteration.
 * Only uses strides of the first operand for dimension reordering.
 * Returns 0 on success, -1 on failure.
 *
 * Assumes operands have already been broadcasted.
 * Used with NPY_RAW_ITER_START and NPY_RAW_ITER_TWO_NEXT for loop handling.
 */
NPY_NO_EXPORT int
/*
 * 准备三个原始数组的迭代器，用于处理具有三个操作数的情况。
 * 在调用此函数之前，应该已经完成三个操作数的广播，因为ndim和shape只针对所有操作数一次性指定。
 *
 * 只使用第一个操作数的步幅来重新排序维度，不考虑所有步幅的组合，这与NpyIter对象不同。
 *
 * 您可以与NPY_RAW_ITER_START和NPY_RAW_ITER_THREE_NEXT一起使用，处理除了最内部循环之外的所有循环模板（即idim == 0的循环）。
 *
 * 成功时返回0，失败时返回-1。
 */
PyArray_PrepareThreeRawArrayIter(int ndim, npy_intp const *shape,
                            char *dataA, npy_intp const *stridesA,
                            char *dataB, npy_intp const *stridesB,
                            char *dataC, npy_intp const *stridesC,
                            int *out_ndim, npy_intp *out_shape,
                            char **out_dataA, npy_intp *out_stridesA,
                            char **out_dataB, npy_intp *out_stridesB,
                            char **out_dataC, npy_intp *out_stridesC);

/*
 * 返回从地址'addr'开始必须从'nvals'个尺寸为'esize'的元素中剥离的元素数量，
 * 以达到可块对齐。'alignment'参数传递所需的字节对齐，必须是2的幂。
 * 此函数用于为块化准备数组。参见下面'numpy_blocked_end'函数的文档，了解此函数的使用示例。
 */
static inline npy_intp
npy_aligned_block_offset(const void * addr, const npy_uintp esize,
                         const npy_uintp alignment, const npy_uintp nvals)
{
    npy_uintp offset, peel;

    // 计算地址偏移量，使其达到块对齐
    offset = (npy_uintp)addr & (alignment - 1);
    // 计算需要剥离的元素数量
    peel = offset ? (alignment - offset) / esize : 0;
    // 如果需要剥离的元素数量超过了nvals，则限制为nvals
    peel = (peel <= nvals) ? peel : nvals;
    // 确保peel在合理范围内，不超过NPY_MAX_INTP
    assert(peel <= NPY_MAX_INTP);
    return (npy_intp)peel;
}
/*
 * Calculate the upper loop bound for iterating over a raw array,
 * handling peeling by 'offset' elements and blocking to a vector
 * size of 'vsz' in bytes.
 *
 * Example usage:
 * npy_intp i;
 * double v[101];
 * npy_intp esize = sizeof(v[0]);
 * npy_intp peel = npy_aligned_block_offset(v, esize, 16, n);
 * // peel to alignment 16
 * for (i = 0; i < peel; i++)
 *   <scalar-op>
 * // simd vectorized operation
 * for (; i < npy_blocked_end(peel, esize, 16, n); i += 16 / esize)
 *   <blocked-op>
 * // handle scalar rest
 * for (; i < n; i++)
 *   <scalar-op>
 */
static inline npy_intp
npy_blocked_end(const npy_uintp peel, const npy_uintp esize,
                const npy_uintp vsz, const npy_uintp nvals)
{
    // Calculate the difference between total elements and peeled elements
    npy_uintp ndiff = nvals - peel;
    // Calculate the result for blocked end by rounding down to nearest multiple
    npy_uintp res = (ndiff - ndiff % (vsz / esize));

    // Ensure the total number of elements is at least as large as peeled elements
    assert(nvals >= peel);
    // Ensure the result fits within the maximum integer size
    assert(res <= NPY_MAX_INTP);

    // Return the calculated blocked end position
    return (npy_intp)(res);
}


/* byte swapping functions */

// Swap bytes of a 16-bit unsigned integer
static inline npy_uint16
npy_bswap2(npy_uint16 x)
{
    return ((x & 0xffu) << 8) | (x >> 8);
}

/*
 * Treat a memory area as int16 and byte-swap unaligned memory,
 * handling CPUs that don't support unaligned access.
 */
static inline void
npy_bswap2_unaligned(char * x)
{
    // Swap bytes for 16-bit unaligned memory
    char a = x[0];
    x[0] = x[1];
    x[1] = a;
}

// Swap bytes of a 32-bit unsigned integer
static inline npy_uint32
npy_bswap4(npy_uint32 x)
{
#ifdef HAVE___BUILTIN_BSWAP32
    // Use compiler's built-in function for byte swapping if available
    return __builtin_bswap32(x);
#else
    // Manually swap bytes for 32-bit unsigned integer
    return ((x & 0xffu) << 24) | ((x & 0xff00u) << 8) |
           ((x & 0xff0000u) >> 8) | (x >> 24);
#endif
}

/*
 * Byte-swap unaligned memory for 32-bit values.
 * Swaps bytes to handle CPUs that don't support unaligned access.
 */
static inline void
npy_bswap4_unaligned(char * x)
{
    // Swap bytes for 32-bit unaligned memory
    char a = x[0];
    x[0] = x[3];
    x[3] = a;
    a = x[1];
    x[1] = x[2];
    x[2] = a;
}

// Swap bytes of a 64-bit unsigned integer
static inline npy_uint64
npy_bswap8(npy_uint64 x)
{
#ifdef HAVE___BUILTIN_BSWAP64
    // Use compiler's built-in function for byte swapping if available
    return __builtin_bswap64(x);
#else
    // Manually swap bytes for 64-bit unsigned integer
    return ((x & 0xffULL) << 56) |
           ((x & 0xff00ULL) << 40) |
           ((x & 0xff0000ULL) << 24) |
           ((x & 0xff000000ULL) << 8) |
           ((x & 0xff00000000ULL) >> 8) |
           ((x & 0xff0000000000ULL) >> 24) |
           ((x & 0xff000000000000ULL) >> 40) |
           (x >> 56);
#endif
}

/*
 * Byte-swap unaligned memory for 64-bit values.
 * Swaps bytes to handle CPUs that don't support unaligned access.
 */
static inline void
npy_bswap8_unaligned(char * x)
{
    // Swap bytes for 64-bit unaligned memory
    char a = x[0]; x[0] = x[7]; x[7] = a;
    a = x[1]; x[1] = x[6]; x[6] = a;
    a = x[2]; x[2] = x[5]; x[5] = a;
    a = x[3]; x[3] = x[4]; x[4] = a;
}


/* Start raw iteration */

// Initialize raw iteration over an array
#define NPY_RAW_ITER_START(idim, ndim, coord, shape) \
        // Initialize coordinates to zero
        memset((coord), 0, (ndim) * sizeof(coord[0])); \
        // Start raw iteration loop
        do {

// Move to the next n-dimensional coordinate for one raw array
#define NPY_RAW_ITER_ONE_NEXT(idim, ndim, coord, shape, data, strides) \
            // Loop through dimensions starting from 1
            for ((idim) = 1; (idim) < (ndim); ++(idim)) { \
                // Check if current coordinate exceeds shape limit
                if (++(coord)[idim] == (shape)[idim]) { \
                    // Reset coordinate and adjust data pointer for overflow
                    (coord)[idim] = 0; \
                    (data) -= ((shape)[idim] - 1) * (strides)[idim]; \
                } \
                else { \
                    // Move data pointer and break the loop
                    (data) += (strides)[idim]; \
                    break; \
                } \
            } \
        } while ((idim) < (ndim))
/* Increment to the next n-dimensional coordinate for two raw arrays */
#define NPY_RAW_ITER_TWO_NEXT(idim, ndim, coord, shape, \
                              dataA, stridesA, dataB, stridesB) \
            for ((idim) = 1; (idim) < (ndim); ++(idim)) { \
                // 检查当前维度坐标是否需要进位
                if (++(coord)[idim] == (shape)[idim]) { \
                    // 如果需要进位，重置当前维度坐标为0
                    (coord)[idim] = 0; \
                    // 调整数据指针A和B以反映下一个坐标的数据位置
                    (dataA) -= ((shape)[idim] - 1) * (stridesA)[idim]; \
                    (dataB) -= ((shape)[idim] - 1) * (stridesB)[idim]; \
                } \
                else { \
                    // 如果不需要进位，增加数据指针A和B以跳到下一个坐标位置
                    (dataA) += (stridesA)[idim]; \
                    (dataB) += (stridesB)[idim]; \
                    break; \
                } \
            } \
        } while ((idim) < (ndim))

/* Increment to the next n-dimensional coordinate for three raw arrays */
#define NPY_RAW_ITER_THREE_NEXT(idim, ndim, coord, shape, \
                              dataA, stridesA, \
                              dataB, stridesB, \
                              dataC, stridesC) \
            for ((idim) = 1; (idim) < (ndim); ++(idim)) { \
                // 检查当前维度坐标是否需要进位
                if (++(coord)[idim] == (shape)[idim]) { \
                    // 如果需要进位，重置当前维度坐标为0
                    (coord)[idim] = 0; \
                    // 调整数据指针A、B和C以反映下一个坐标的数据位置
                    (dataA) -= ((shape)[idim] - 1) * (stridesA)[idim]; \
                    (dataB) -= ((shape)[idim] - 1) * (stridesB)[idim]; \
                    (dataC) -= ((shape)[idim] - 1) * (stridesC)[idim]; \
                } \
                else { \
                    // 如果不需要进位，增加数据指针A、B和C以跳到下一个坐标位置
                    (dataA) += (stridesA)[idim]; \
                    (dataB) += (stridesB)[idim]; \
                    (dataC) += (stridesC)[idim]; \
                    break; \
                } \
            } \
        } while ((idim) < (ndim))

/* Increment to the next n-dimensional coordinate for four raw arrays */
#define NPY_RAW_ITER_FOUR_NEXT(idim, ndim, coord, shape, \
                              dataA, stridesA, \
                              dataB, stridesB, \
                              dataC, stridesC, \
                              dataD, stridesD) \
            for ((idim) = 1; (idim) < (ndim); ++(idim)) { \
                // 检查当前维度坐标是否需要进位
                if (++(coord)[idim] == (shape)[idim]) { \
                    // 如果需要进位，重置当前维度坐标为0
                    (coord)[idim] = 0; \
                    // 调整数据指针A、B、C和D以反映下一个坐标的数据位置
                    (dataA) -= ((shape)[idim] - 1) * (stridesA)[idim]; \
                    (dataB) -= ((shape)[idim] - 1) * (stridesB)[idim]; \
                    (dataC) -= ((shape)[idim] - 1) * (stridesC)[idim]; \
                    (dataD) -= ((shape)[idim] - 1) * (stridesD)[idim]; \
                } \
                else { \
                    // 如果不需要进位，增加数据指针A、B、C和D以跳到下一个坐标位置
                    (dataA) += (stridesA)[idim]; \
                    (dataB) += (stridesB)[idim]; \
                    (dataC) += (stridesC)[idim]; \
                    (dataD) += (stridesD)[idim]; \
                    break; \
                } \
            } \
        } while ((idim) < (ndim))
/*
 *            TRIVIAL ITERATION
 *
 * In some cases when the iteration order isn't important, iteration over
 * arrays is trivial.  This is the case when:
 *   * The array has 0 or 1 dimensions.
 *   * The array is C or Fortran contiguous.
 * Use of an iterator can be skipped when this occurs.  These macros assist
 * in detecting and taking advantage of the situation.  Note that it may
 * be worthwhile to further check if the stride is a contiguous stride
 * and take advantage of that.
 *
 * Here is example code for a single array:
 *
 *      if (PyArray_TRIVIALLY_ITERABLE(self)) {
 *          char *data;
 *          npy_intp count, stride;
 *
 *          PyArray_PREPARE_TRIVIAL_ITERATION(self, count, data, stride);
 *
 *          while (count--) {
 *              // Use the data pointer
 *
 *              data += stride;
 *          }
 *      }
 *      else {
 *          // Create iterator, etc...
 *      }
 *
 */

/*
 * Note: Equivalently iterable macro requires one of arr1 or arr2 be
 *       trivially iterable to be valid.
 */

/**
 * Determine whether an array is safe for trivial iteration.
 *
 * This macro checks if the array meets conditions for trivial iteration:
 * - Has 0 or 1 dimensions
 * - Is C contiguous or Fortran contiguous
 */
#define PyArray_TRIVIALLY_ITERABLE(arr) ( \
                    PyArray_NDIM(arr) <= 1 || \
                    PyArray_CHKFLAGS(arr, NPY_ARRAY_C_CONTIGUOUS) || \
                    PyArray_CHKFLAGS(arr, NPY_ARRAY_F_CONTIGUOUS) \
                    )

/**
 * Calculate the stride for trivial iteration over a single array or pair of arrays.
 *
 * This macro computes the stride to be used for iterating over an array or pair of arrays:
 * - If size is 1, returns 0 (no iteration needed)
 * - If array has 1 dimension, returns its stride
 * - If array has more dimensions, returns item size
 */
#define PyArray_TRIVIAL_PAIR_ITERATION_STRIDE(size, arr) ( \
        assert(PyArray_TRIVIALLY_ITERABLE(arr)), \
        size == 1 ? 0 : ((PyArray_NDIM(arr) == 1) ? \
                             PyArray_STRIDE(arr, 0) : PyArray_ITEMSIZE(arr)))

/**
 * Check if two arrays can be iterated over trivially and safely.
 *
 * This function checks conditions under which two arrays can be safely iterated over:
 * - Both arrays are read-only
 * - Arrays do not share overlapping memory
 * - Strides match and one array's base address is before the other's
 *   to ensure correct data dependency
 *
 * @param arr1 First array object
 * @param arr2 Second array object
 * @param arr1_read Flag indicating if arr1 is read-only
 * @param arr2_read Flag indicating if arr2 is read-only
 * @return 1 if trivial iteration is safe, 0 otherwise
 */
static inline int
PyArray_EQUIVALENTLY_ITERABLE_OVERLAP_OK(PyArrayObject *arr1, PyArrayObject *arr2,
                                         int arr1_read, int arr2_read)
{
    npy_intp size1, size2, stride1, stride2;
    int arr1_ahead = 0, arr2_ahead = 0;

    if (arr1_read && arr2_read) {
        return 1;
    }

    size1 = PyArray_SIZE(arr1);
    stride1 = PyArray_TRIVIAL_PAIR_ITERATION_STRIDE(size1, arr1);

    /*
     * arr1 == arr2 is common for in-place operations, so we fast-path it here.
     * TODO: The stride1 != 0 check rejects broadcast arrays.  This may affect
     *       self-overlapping arrays, but seems only necessary due to
     *       `try_trivial_single_output_loop` not rejecting broadcast outputs.
     */
    if (arr1 == arr2 && stride1 != 0) {
        return 1;
    }

    if (solve_may_share_memory(arr1, arr2, 1) == 0) {
        return 1;
    }
    /*
     * 获取 arr2 的大小（元素个数）
     * PyArray_SIZE 是一个宏，用于获取数组的大小
     */
    size2 = PyArray_SIZE(arr2);

    /*
     * 计算 arr2 的迭代步长（stride）
     * PyArray_TRIVIAL_PAIR_ITERATION_STRIDE 是一个宏，用于计算数组的迭代步长
     */
    stride2 = PyArray_TRIVIAL_PAIR_ITERATION_STRIDE(size2, arr2);

    /*
     * 如果 arr1 的步长大于 0，则判断 arr1 是否在 arr2 的前面
     * 否则，如果 arr1 的步长小于 0，则判断 arr1 是否在 arr2 的前面
     * 这里通过比较字节数组的起始地址来判断数组的相对位置
     */
    if (stride1 > 0) {
        arr1_ahead = (stride1 >= stride2 &&
                      PyArray_BYTES(arr1) >= PyArray_BYTES(arr2));
    }
    else if (stride1 < 0) {
        arr1_ahead = (stride1 <= stride2 &&
                      PyArray_BYTES(arr1) <= PyArray_BYTES(arr2));
    }

    /*
     * 如果 arr2 的步长大于 0，则判断 arr2 是否在 arr1 的前面
     * 否则，如果 arr2 的步长小于 0，则判断 arr2 是否在 arr1 的前面
     * 这里通过比较字节数组的起始地址来判断数组的相对位置
     */
    if (stride2 > 0) {
        arr2_ahead = (stride2 >= stride1 &&
                      PyArray_BYTES(arr2) >= PyArray_BYTES(arr1));
    }
    else if (stride2 < 0) {
        arr2_ahead = (stride2 <= stride1 &&
                      PyArray_BYTES(arr2) <= PyArray_BYTES(arr1));
    }

    /*
     * 返回两个条件的逻辑与结果：
     * - 如果 arr1 未读取或者 arr1 在 arr2 前面，则返回 true
     * - 如果 arr2 未读取或者 arr2 在 arr1 前面，则返回 true
     * 否则返回 false
     */
    return (!arr1_read || arr1_ahead) && (!arr2_read || arr2_ahead);
#endif  /* NUMPY_CORE_SRC_COMMON_LOWLEVEL_STRIDED_LOOPS_H_ */



#endif  /* NUMPY_CORE_SRC_COMMON_LOWLEVEL_STRIDED_LOOPS_H_ */



#define PyArray_EQUIVALENTLY_ITERABLE_BASE(arr1, arr2) (            \
                        PyArray_NDIM(arr1) == PyArray_NDIM(arr2) && \
                        PyArray_CompareLists(PyArray_DIMS(arr1), \
                                             PyArray_DIMS(arr2), \
                                             PyArray_NDIM(arr1)) && \
                        (PyArray_FLAGS(arr1)&(NPY_ARRAY_C_CONTIGUOUS| \
                                      NPY_ARRAY_F_CONTIGUOUS)) & \
                                (PyArray_FLAGS(arr2)&(NPY_ARRAY_C_CONTIGUOUS| \
                                              NPY_ARRAY_F_CONTIGUOUS)) \
                        )



// 定义宏 PyArray_EQUIVALENTLY_ITERABLE_BASE，用于比较两个 NumPy 数组是否可以等效迭代
#define PyArray_EQUIVALENTLY_ITERABLE_BASE(arr1, arr2) (            \
                        PyArray_NDIM(arr1) == PyArray_NDIM(arr2) && \
                        PyArray_CompareLists(PyArray_DIMS(arr1), \
                                             PyArray_DIMS(arr2), \
                                             PyArray_NDIM(arr1)) && \
                        (PyArray_FLAGS(arr1)&(NPY_ARRAY_C_CONTIGUOUS| \
                                      NPY_ARRAY_F_CONTIGUOUS)) & \
                                (PyArray_FLAGS(arr2)&(NPY_ARRAY_C_CONTIGUOUS| \
                                              NPY_ARRAY_F_CONTIGUOUS)) \
                        )



#define PyArray_EQUIVALENTLY_ITERABLE(arr1, arr2, arr1_read, arr2_read) ( \
                        PyArray_EQUIVALENTLY_ITERABLE_BASE(arr1, arr2) && \
                        PyArray_EQUIVALENTLY_ITERABLE_OVERLAP_OK( \
                            arr1, arr2, arr1_read, arr2_read))



// 定义宏 PyArray_EQUIVALENTLY_ITERABLE，用于比较两个 NumPy 数组是否可以等效迭代，并且允许重叠
#define PyArray_EQUIVALENTLY_ITERABLE(arr1, arr2, arr1_read, arr2_read) ( \
                        PyArray_EQUIVALENTLY_ITERABLE_BASE(arr1, arr2) && \
                        PyArray_EQUIVALENTLY_ITERABLE_OVERLAP_OK( \
                            arr1, arr2, arr1_read, arr2_read))



#define PyArray_PREPARE_TRIVIAL_ITERATION(arr, count, data, stride) \
                    count = PyArray_SIZE(arr); \
                    data = PyArray_BYTES(arr); \
                    stride = ((PyArray_NDIM(arr) == 0) ? 0 : \
                                    ((PyArray_NDIM(arr) == 1) ? \
                                            PyArray_STRIDE(arr, 0) : \
                                            PyArray_ITEMSIZE(arr)));



// 定义宏 PyArray_PREPARE_TRIVIAL_ITERATION，用于准备简单迭代的参数
#define PyArray_PREPARE_TRIVIAL_ITERATION(arr, count, data, stride) \
                    count = PyArray_SIZE(arr); \
                    data = PyArray_BYTES(arr); \
                    stride = ((PyArray_NDIM(arr) == 0) ? 0 : \
                                    ((PyArray_NDIM(arr) == 1) ? \
                                            PyArray_STRIDE(arr, 0) : \
                                            PyArray_ITEMSIZE(arr)));



#define PyArray_PREPARE_TRIVIAL_PAIR_ITERATION(arr1, arr2, \
                                        count, \
                                        data1, data2, \
                                        stride1, stride2) { \
                    npy_intp size1 = PyArray_SIZE(arr1); \
                    npy_intp size2 = PyArray_SIZE(arr2); \
                    count = ((size1 > size2) || size1 == 0) ? size1 : size2; \
                    data1 = PyArray_BYTES(arr1); \
                    data2 = PyArray_BYTES(arr2); \
                    stride1 = PyArray_TRIVIAL_PAIR_ITERATION_STRIDE(size1, arr1); \
                    stride2 = PyArray_TRIVIAL_PAIR_ITERATION_STRIDE(size2, arr2); \
                }



// 定义宏 PyArray_PREPARE_TRIVIAL_PAIR_ITERATION，用于准备简单对成对迭代的参数
#define PyArray_PREPARE_TRIVIAL_PAIR_ITERATION(arr1, arr2, \
                                        count, \
                                        data1, data2, \
                                        stride1, stride2) { \
                    npy_intp size1 = PyArray_SIZE(arr1); \
                    npy_intp size2 = PyArray_SIZE(arr2); \
                    count = ((size1 > size2) || size1 == 0) ? size1 : size2; \
                    data1 = PyArray_BYTES(arr1); \
                    data2 = PyArray_BYTES(arr2); \
                    stride1 = PyArray_TRIVIAL_PAIR_ITERATION_STRIDE(size1, arr1); \
                    stride2 = PyArray_TRIVIAL_PAIR_ITERATION_STRIDE(size2, arr2); \
                }



#endif  /* NUMPY_CORE_SRC_COMMON_LOWLEVEL_STRIDED_LOOPS_H_ */



#endif  /* NUMPY_CORE_SRC_COMMON_LOWLEVEL_STRIDED_LOOPS_H_ */
```