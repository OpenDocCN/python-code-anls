# `.\numpy\numpy\_core\src\multiarray\dtype_transfer.h`

```
#ifndef NUMPY_CORE_SRC_MULTIARRAY_DTYPE_TRANSFER_H_
#define NUMPY_CORE_SRC_MULTIARRAY_DTYPE_TRANSFER_H_

#include "array_method.h"


/*
 * More than for most functions, cast information needs to be stored in
 * a few places.  Most importantly, in many cases we need to chain or wrap
 * casts (e.g. structured dtypes).
 *
 * This struct provides a place to store all necessary information as
 * compact as possible.  It must be used with the inline functions below
 * to ensure correct setup and teardown.
 *
 * In general, the casting machinery currently handles the correct set up
 * of the struct.
 */
typedef struct {
    PyArrayMethod_StridedLoop *func;    /* Function pointer to strided loop function */
    NpyAuxData *auxdata;                /* Auxiliary data for method */
    PyArrayMethod_Context context;      /* Context for method */
    /* Storage to be linked from "context" */
    PyArray_Descr *descriptors[2];      /* Array of two descriptors */
} NPY_cast_info;


/*
 * Create a new cast-info struct with cast_info->context.descriptors linked.
 * Compilers should inline this to ensure the whole struct is not actually
 * copied.
 * If set up otherwise, func must be NULL'ed to indicate no-cleanup necessary.
 */
static inline void
NPY_cast_info_init(NPY_cast_info *cast_info)
{
    cast_info->func = NULL;  /* Mark as uninitialized. */
    /*
     * Support for auxdata being unchanged, in the future, we might add
     * a scratch space to `NPY_cast_info` and link to that instead.
     */
    cast_info->auxdata = NULL;
    cast_info->context.descriptors = cast_info->descriptors;

    // TODO: Delete this again probably maybe create a new minimal init macro
    cast_info->context.caller = NULL;
}


/*
 * Free's all references and data held inside the struct (not the struct).
 * First checks whether `cast_info.func == NULL`, and assume it is
 * uninitialized in that case.
 */
static inline void
NPY_cast_info_xfree(NPY_cast_info *cast_info)
{
    if (cast_info->func == NULL) {
        return;  /* If uninitialized, return early */
    }
    assert(cast_info->context.descriptors == cast_info->descriptors);  /* Ensure descriptors are correctly set */
    NPY_AUXDATA_FREE(cast_info->auxdata);  /* Free auxiliary data */
    Py_DECREF(cast_info->descriptors[0]);  /* Decrement reference count of descriptor 0 */
    Py_XDECREF(cast_info->descriptors[1]);  /* Decrement reference count of descriptor 1 */
    Py_XDECREF(cast_info->context.method);  /* Decrement reference count of method */
    cast_info->func = NULL;  /* Mark as uninitialized */
}


/*
 * Move the data from `original` to `cast_info`. Original is cleared
 * (its func set to NULL).
 */
static inline void
NPY_cast_info_move(NPY_cast_info *cast_info, NPY_cast_info *original)
{
    *cast_info = *original;  /* Copy contents of original to cast_info */
    /* Fix internal pointer: */
    cast_info->context.descriptors = cast_info->descriptors;
    /* Mark original to not be cleaned up: */
    original->func = NULL;
}

/*
 * Finalize a copy (INCREF+auxdata clone). This assumes a previous `memcpy`
 * of the struct.
 * NOTE: It is acceptable to call this with the same struct if the struct
 *       has been filled by a valid memcpy from an initialized one.
 */
static inline int
NPY_cast_info_copy(NPY_cast_info *cast_info, NPY_cast_info *original)
{
    cast_info->context.descriptors = cast_info->descriptors;  /* Link descriptors */

    assert(original->func != NULL);  /* Ensure original is initialized */
    cast_info->func = original->func;  /* Copy function pointer from original */
    # 将原始结构体的第一个描述符复制给转换后的结构体的第一个描述符
    cast_info->descriptors[0] = original->descriptors[0];
    # 增加第一个描述符的引用计数，确保其内存不会在不需要时被释放
    Py_XINCREF(cast_info->descriptors[0]);
    
    # 将原始结构体的第二个描述符复制给转换后的结构体的第二个描述符
    cast_info->descriptors[1] = original->descriptors[1];
    # 增加第二个描述符的引用计数，确保其内存不会在不需要时被释放
    Py_XINCREF(cast_info->descriptors[1]);
    
    # 将原始结构体的调用者信息复制给转换后的结构体的调用者信息
    cast_info->context.caller = original->context.caller;
    # 增加调用者信息的引用计数，确保其内存不会在不需要时被释放
    Py_XINCREF(cast_info->context.caller);
    
    # 将原始结构体的方法信息复制给转换后的结构体的方法信息
    cast_info->context.method = original->context.method;
    # 增加方法信息的引用计数，确保其内存不会在不需要时被释放
    Py_XINCREF(cast_info->context.method);
    
    # 如果原始结构体的辅助数据为空，则设置转换后的结构体的辅助数据为空，并返回成功状态
    if (original->auxdata == NULL) {
        cast_info->auxdata = NULL;
        return 0;
    }
    
    # 复制原始结构体的辅助数据给转换后的结构体的辅助数据
    cast_info->auxdata = NPY_AUXDATA_CLONE(original->auxdata);
    # 如果辅助数据复制失败，则返回错误状态
    if (NPY_UNLIKELY(cast_info->auxdata == NULL)) {
        /* 无需清理，除了辅助数据之外的所有内容都已经正确初始化。 */
        return -1;
    }
    
    # 返回成功状态
    return 0;
# 结束之前的 C 函数定义
}

# 以下是一系列 C 函数的定义，这些函数是 NumPy 的内部函数，用于处理数组的数据类型转换和操作

# 定义一个 C 函数 _strided_to_strided_move_references，用于处理按步跨的数据到步跨的数据的移动参考
NPY_NO_EXPORT int
_strided_to_strided_move_references(
        PyArrayMethod_Context *NPY_UNUSED(context), char *const *args,
        const npy_intp *dimensions, const npy_intp *strides,
        NpyAuxData *NPY_UNUSED(auxdata));

# 定义一个 C 函数 _strided_to_strided_copy_references，用于处理按步跨的数据到步跨的数据的复制参考
NPY_NO_EXPORT int
_strided_to_strided_copy_references(
        PyArrayMethod_Context *NPY_UNUSED(context), char *const *args,
        const npy_intp *dimensions, const npy_intp *strides,
        NpyAuxData *NPY_UNUSED(auxdata));


# 定义一个 C 函数 any_to_object_get_loop，用于获取将任何类型数据转换为对象数据类型的循环方法
NPY_NO_EXPORT int
any_to_object_get_loop(
        PyArrayMethod_Context *context,
        int aligned, int move_references,
        const npy_intp *strides,
        PyArrayMethod_StridedLoop **out_loop,
        NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags);

# 定义一个 C 函数 object_to_any_get_loop，用于获取从对象数据类型转换为任何类型数据的循环方法
NPY_NO_EXPORT int
object_to_any_get_loop(
        PyArrayMethod_Context *context,
        int NPY_UNUSED(aligned), int move_references,
        const npy_intp *NPY_UNUSED(strides),
        PyArrayMethod_StridedLoop **out_loop,
        NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags);

# 定义一个 C 函数 wrap_aligned_transferfunction，用于封装对齐的传输功能
NPY_NO_EXPORT int
wrap_aligned_transferfunction(
        int aligned, int must_wrap,
        npy_intp src_stride, npy_intp dst_stride,
        PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
        PyArray_Descr *src_wrapped_dtype, PyArray_Descr *dst_wrapped_dtype,
        PyArrayMethod_StridedLoop **out_stransfer,
        NpyAuxData **out_transferdata, int *out_needs_api);

# 定义一个 C 函数 get_nbo_cast_datetime_transfer_function，用于获取网络字节顺序下日期时间转换函数
NPY_NO_EXPORT int
get_nbo_cast_datetime_transfer_function(int aligned,
        PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
        PyArrayMethod_StridedLoop **out_stransfer,
        NpyAuxData **out_transferdata);

# 定义一个 C 函数 get_nbo_datetime_to_string_transfer_function，用于获取网络字节顺序下日期时间到字符串转换函数
NPY_NO_EXPORT int
get_nbo_datetime_to_string_transfer_function(
        PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
        PyArrayMethod_StridedLoop **out_stransfer,
        NpyAuxData **out_transferdata);

# 定义一个 C 函数 get_nbo_string_to_datetime_transfer_function，用于获取网络字节顺序下字符串到日期时间转换函数
NPY_NO_EXPORT int
get_nbo_string_to_datetime_transfer_function(
        PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
        PyArrayMethod_StridedLoop **out_stransfer,
        NpyAuxData **out_transferdata);

# 定义一个 C 函数 get_datetime_to_unicode_transfer_function，用于获取日期时间到 Unicode 字符串转换函数
NPY_NO_EXPORT int
get_datetime_to_unicode_transfer_function(int aligned,
        npy_intp src_stride, npy_intp dst_stride,
        PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
        PyArrayMethod_StridedLoop **out_stransfer,
        NpyAuxData **out_transferdata,
        int *out_needs_api);

# 定义一个 C 函数 get_unicode_to_datetime_transfer_function，用于获取 Unicode 字符串到日期时间转换函数
NPY_NO_EXPORT int
get_unicode_to_datetime_transfer_function(int aligned,
        npy_intp src_stride, npy_intp dst_stride,
        PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
        PyArrayMethod_StridedLoop **out_stransfer,
        NpyAuxData **out_transferdata,
        int *out_needs_api);

# 定义一个 C 函数，用于创建一个包装器，封装 copyswapn 或者旧式转换函数
# 函数名未提供完整，需要在继续的代码中查看
NPY_NO_EXPORT int
// 定义一个函数，用于获取包装的遗留转换函数。
// 参数解释：
//   - aligned: 对齐标志，表示是否对齐内存
//   - src_stride: 源数组的步幅（stride）
//   - dst_stride: 目标数组的步幅（stride）
//   - src_dtype: 源数据类型描述符指针
//   - dst_dtype: 目标数据类型描述符指针
//   - move_references: 移动引用标志，表示是否需要移动引用
//   - out_stransfer: 传出参数，用于存储返回的转换方法的指针
//   - out_transferdata: 传出参数，用于存储返回的辅助数据的指针
//   - out_needs_api: 传出参数，用于存储返回的是否需要 API 的标志
//   - allow_wrapped: 允许使用包装函数的标志

get_wrapped_legacy_cast_function(int aligned,
        npy_intp src_stride, npy_intp dst_stride,
        PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
        int move_references,
        PyArrayMethod_StridedLoop **out_stransfer,
        NpyAuxData **out_transferdata,
        int *out_needs_api, int allow_wrapped);


这段代码是一个函数声明，定义了一个名为 `get_wrapped_legacy_cast_function` 的函数，该函数有多个参数和传出参数，用于获取包装的遗留转换函数。
```