# `.\numpy\numpy\_core\src\multiarray\dtype_transfer.c`

```
/*
 * This file contains low-level loops for data type transfers.
 * In particular the function PyArray_GetDTypeTransferFunction is
 * implemented here.
 *
 * Copyright (c) 2010 by Mark Wiebe (mwwiebe@gmail.com)
 * The University of British Columbia
 *
 * See LICENSE.txt for the license.
 *
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#define _UMATHMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include "numpy/arrayobject.h"
#include "numpy/npy_math.h"

#include "lowlevel_strided_loops.h"


#include "convert_datatype.h"
#include "ctors.h"
#include "_datetime.h"
#include "datetime_strings.h"
#include "descriptor.h"
#include "array_assign.h"

#include "shape.h"
#include "dtype_transfer.h"
#include "dtype_traversal.h"
#include "alloc.h"
#include "dtypemeta.h"
#include "array_method.h"
#include "array_coercion.h"

#include "umathmodule.h"

#define NPY_LOWLEVEL_BUFFER_BLOCKSIZE  128

/********** PRINTF DEBUG TRACING **************/
#define NPY_DT_DBG_TRACING 0
/* Tracing incref/decref can be very noisy */
#define NPY_DT_REF_DBG_TRACING 0

#if NPY_DT_REF_DBG_TRACING
#define NPY_DT_DBG_REFTRACE(msg, ref) \
    printf("%-12s %20p %s%d%s\n", msg, ref, \
                        ref ? "(refcnt " : "", \
                        ref ? (int)ref->ob_refcnt : 0, \
                        ref ? ((ref->ob_refcnt <= 0) ? \
                                        ") <- BIG PROBLEM!!!!" : ")") : ""); \
    fflush(stdout);
#else
#define NPY_DT_DBG_REFTRACE(msg, ref)
#endif
/**********************************************/

#if NPY_DT_DBG_TRACING
/*
 * Thin wrapper around print that ignores exceptions
 */
static void
_safe_print(PyObject *obj)
{
    if (PyObject_Print(obj, stdout, 0) < 0) {
        PyErr_Clear();
        printf("<error during print>");
    }
}
#endif


/*************************** COPY REFERENCES *******************************/

/* Moves references from src to dst */
NPY_NO_EXPORT int
_strided_to_strided_move_references(
        PyArrayMethod_Context *NPY_UNUSED(context), char *const *args,
        const npy_intp *dimensions, const npy_intp *strides,
        NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    char *src = args[0], *dst = args[1];
    npy_intp src_stride = strides[0], dst_stride = strides[1];

    PyObject *src_ref = NULL, *dst_ref = NULL;
    while (N > 0) {
        // Copy the reference from src to src_ref and dst to dst_ref
        memcpy(&src_ref, src, sizeof(src_ref));
        memcpy(&dst_ref, dst, sizeof(dst_ref));

        /* Release the reference in dst */
        NPY_DT_DBG_REFTRACE("dec dst ref", dst_ref);
        Py_XDECREF(dst_ref);
        /* Move the reference */
        NPY_DT_DBG_REFTRACE("move src ref", src_ref);
        memcpy(dst, &src_ref, sizeof(src_ref));
        /* Set the source reference to NULL */
        src_ref = NULL;
        memcpy(src, &src_ref, sizeof(src_ref));

        // Move to the next element in the arrays
        src += src_stride;
        dst += dst_stride;
        --N;
    }
    return 0;
}

/* Copies references from src to dst */
NPY_NO_EXPORT int
_strided_to_strided_copy_references(
        PyArrayMethod_Context *NPY_UNUSED(context), char *const *args,
        const npy_intp *dimensions, const npy_intp *strides,
        NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    char *src = args[0], *dst = args[1];
    npy_intp src_stride = strides[0], dst_stride = strides[1];

    PyObject *src_ref = NULL;
    while (N > 0) {
        // Copy the reference from src to src_ref
        memcpy(&src_ref, src, sizeof(src_ref));

        /* Move the reference */
        NPY_DT_DBG_REFTRACE("copy src ref", src_ref);
        memcpy(dst, &src_ref, sizeof(src_ref));

        // Move to the next element in the arrays
        src += src_stride;
        dst += dst_stride;
        --N;
    }
    return 0;
}
# 定义一个不导出的整型函数 _strided_to_strided_copy_references，接收参数为 PyArrayMethod_Context 结构体指针 context，字符指针数组 args，以及两个整型数组 dimensions 和 strides，还有一个未使用的 NpyAuxData 指针 auxdata
NPY_NO_EXPORT int
_strided_to_strided_copy_references(
        PyArrayMethod_Context *NPY_UNUSED(context), char *const *args,
        const npy_intp *dimensions, const npy_intp *strides,
        NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];  # 从 dimensions 数组中获取第一个元素，用于循环次数 N
    char *src = args[0], *dst = args[1];  # 从 args 数组中获取源地址 src 和目标地址 dst
    npy_intp src_stride = strides[0], dst_stride = strides[1];  # 从 strides 数组中获取源步长 src_stride 和目标步长 dst_stride

    PyObject *src_ref = NULL, *dst_ref = NULL;  # 声明源引用 src_ref 和目标引用 dst_ref，初始化为 NULL
    while (N > 0) {  # 进入循环，循环次数为 N
        memcpy(&src_ref, src, sizeof(src_ref));  # 将 src 指向的内存中的数据拷贝到 src_ref 中
        memcpy(&dst_ref, dst, sizeof(dst_ref));  # 将 dst 指向的内存中的数据拷贝到 dst_ref 中

        /* 复制引用 */
        NPY_DT_DBG_REFTRACE("copy src ref", src_ref);  # 调试信息，跟踪源引用 src_ref
        memcpy(dst, &src_ref, sizeof(src_ref));  # 将 src_ref 的地址拷贝到 dst 指向的内存中
        /* 认领引用 */
        Py_XINCREF(src_ref);  # 增加源引用 src_ref 的引用计数
        /* 释放目标中的引用 */
        NPY_DT_DBG_REFTRACE("dec dst ref", dst_ref);  # 调试信息，跟踪目标引用 dst_ref
        Py_XDECREF(dst_ref);  # 释放目标引用 dst_ref

        src += src_stride;  # 源地址指针向后移动 src_stride 步长
        dst += dst_stride;  # 目标地址指针向后移动 dst_stride 步长
        --N;  # N 减1，准备下一轮循环
    }
    return 0;  # 返回值为 0，表示执行成功
}

/************************** ANY TO OBJECT *********************************/

# 定义一个结构体 _any_to_object_auxdata，包含 NpyAuxData 基类，PyArray_GetItemFunc 指针 getitem，PyArrayObject_fields 结构体 arr_fields，以及 NPY_traverse_info 结构体 decref_src
typedef struct {
    NpyAuxData base;
    PyArray_GetItemFunc *getitem;
    PyArrayObject_fields arr_fields;
    NPY_traverse_info decref_src;
} _any_to_object_auxdata;

# 函数 _any_to_object_auxdata_free，用于释放 _any_to_object_auxdata 结构体占用的内存，参数为 NpyAuxData 指针 auxdata
static void
_any_to_object_auxdata_free(NpyAuxData *auxdata)
{
    _any_to_object_auxdata *data = (_any_to_object_auxdata *)auxdata;  # 将 auxdata 强制类型转换为 _any_to_object_auxdata 结构体指针 data

    Py_DECREF(data->arr_fields.descr);  # 释放 arr_fields 中 descr 成员的引用计数
    NPY_traverse_info_xfree(&data->decref_src);  # 释放 decref_src 中的内存
    PyMem_Free(data);  # 释放整个结构体占用的内存
}

# 函数 _any_to_object_auxdata_clone，用于克隆 _any_to_object_auxdata 结构体，参数为 NpyAuxData 指针 auxdata
static NpyAuxData *
_any_to_object_auxdata_clone(NpyAuxData *auxdata)
{
    _any_to_object_auxdata *data = (_any_to_object_auxdata *)auxdata;  # 将 auxdata 强制类型转换为 _any_to_object_auxdata 结构体指针 data

    _any_to_object_auxdata *res = PyMem_Malloc(sizeof(_any_to_object_auxdata));  # 分配一个新的 _any_to_object_auxdata 结构体的内存空间

    res->base = data->base;  # 复制 base 成员的值
    res->getitem = data->getitem;  # 复制 getitem 成员的值
    res->arr_fields = data->arr_fields;  # 复制 arr_fields 成员的值
    Py_INCREF(res->arr_fields.descr);  # 增加 arr_fields 中 descr 成员的引用计数

    if (data->decref_src.func != NULL) {  # 如果 decref_src 中 func 成员不为空
        if (NPY_traverse_info_copy(&res->decref_src, &data->decref_src) < 0) {  # 复制 decref_src 中的数据到 res->decref_src
            NPY_AUXDATA_FREE((NpyAuxData *)res);  # 如果复制失败，释放 res 的内存
            return NULL;  # 返回空指针，表示复制失败
        }
    }
    else {
        res->decref_src.func = NULL;  # 否则将 res->decref_src.func 置为 NULL
    }
    return (NpyAuxData *)res;  # 返回克隆后的 NpyAuxData 指针
}

# 定义一个函数 _strided_to_strided_any_to_object，接收参数为 PyArrayMethod_Context 结构体指针 context，字符指针数组 args，以及两个整型数组 dimensions 和 strides，还有一个 NpyAuxData 指针 auxdata
static int
_strided_to_strided_any_to_object(
        PyArrayMethod_Context *NPY_UNUSED(context), char *const *args,
        const npy_intp *dimensions, const npy_intp *strides,
        NpyAuxData *auxdata)
{
    npy_intp N = dimensions[0];  # 从 dimensions 数组中获取第一个元素，用于循环次数 N
    char *src = args[0], *dst = args[1];  # 从 args 数组中获取源地址 src 和目标地址 dst
    npy_intp src_stride = strides[0], dst_stride = strides[1];  # 从 strides 数组中获取源步长 src_stride 和目标步长 dst_stride

    _any_to_object_auxdata *data = (_any_to_object_auxdata *)auxdata;  # 将 auxdata 强制类型转换为 _any_to_object_auxdata 结构体指针 data

    PyObject *dst_ref = NULL;  # 声明目标引用 dst_ref，初始化为 NULL
    char *orig_src = src;  # 保存源地址的初始值到 orig_src
    while (N > 0) {  # 进入循环，循环次数为 N
        memcpy(&dst_ref, dst, sizeof(dst_ref));  # 将 dst 指向的内存中的数据拷贝到 dst_ref 中
        Py_XDECREF(dst_ref);  # 释放目标引用 dst_ref
        dst_ref = data->getitem(src, &data->arr_fields);  # 调用 data->getitem 函数获取目标引用 dst_ref
        memcpy(dst, &dst_ref, sizeof(PyObject *));  # 将 dst_ref 的地址拷贝到 dst 指向的内存中

        if (dst_ref == NULL) {  # 如果 dst_ref 为空
            return -1;  # 返回 -1，表示执行失败
        }
        src += src_stride;  # 源地址指针向后移动 src_stride 步长
        dst += dst_stride;  # 目标地址指针向后移动 dst_stride 步长
        --N;  # N 减1，准备下一轮循环
    }
    # 检查 `data->decref_src.func` 是否不为 NULL
    if (data->decref_src.func != NULL) {
        # 如果需要，清空输入缓冲区 (`move_references`)
        if (data->decref_src.func(NULL, data->decref_src.descr,
                orig_src, N, src_stride, data->decref_src.auxdata) < 0) {
            # 如果清空输入缓冲区操作失败，返回 -1
            return -1;
        }
    }
    # 操作成功，返回 0
    return 0;
    *flags = NPY_METH_REQUIRES_PYAPI;  /* 设置方法标志为需要 Python API */

    *out_loop = _strided_to_strided_any_to_object;  /* 将循环函数指针设置为特定函数 */

    *out_transferdata = PyMem_Malloc(sizeof(_any_to_object_auxdata));  /* 分配传输数据的内存 */

    if (*out_transferdata == NULL) {
        return -1;  /* 如果内存分配失败，返回错误码 */
    }

    _any_to_object_auxdata *data = (_any_to_object_auxdata *)*out_transferdata;  /* 将分配的内存转换为特定结构体指针 */

    data->base.free = &_any_to_object_auxdata_free;  /* 设置数据结构体的释放函数 */
    data->base.clone = &_any_to_object_auxdata_clone;  /* 设置数据结构体的克隆函数 */
    data->arr_fields.base = NULL;  /* 初始化数据结构体中的基本字段 */
    Py_SET_TYPE(&data->arr_fields, NULL);  /* 设置数据结构体的类型为 NULL */
    data->arr_fields.descr = context->descriptors[0];  /* 将描述符指定给数据结构体 */
    Py_INCREF(data->arr_fields.descr);  /* 增加描述符的引用计数 */
    data->arr_fields.flags = aligned ? NPY_ARRAY_ALIGNED : 0;  /* 根据对齐情况设置标志位 */
    data->arr_fields.nd = 0;  /* 初始化数据结构体的维度为 0 */

    data->getitem = PyDataType_GetArrFuncs(context->descriptors[0])->getitem;  /* 获取获取元素函数 */
    NPY_traverse_info_init(&data->decref_src);  /* 初始化遍历信息对象 */

    if (move_references && PyDataType_REFCHK(context->descriptors[0])) {
        NPY_ARRAYMETHOD_FLAGS clear_flags;  /* 定义清除标志位对象 */

        if (PyArray_GetClearFunction(
                aligned, strides[0], context->descriptors[0],
                &data->decref_src, &clear_flags) < 0)  {
            NPY_AUXDATA_FREE(*out_transferdata);  /* 如果获取清除函数失败，释放分配的内存 */
            *out_transferdata = NULL;  /* 将传输数据指针置为空 */
            return -1;  /* 返回错误码 */
        }
        *flags = PyArrayMethod_COMBINED_FLAGS(*flags, clear_flags);  /* 组合标志位 */
    }

    return 0;  /* 返回成功状态 */
}


注释：
这段代码定义了一个函数 `any_to_object_get_loop`，用于处理从任意类型到对象类型的转换。函数主要功能是设置一些标志位、函数指针以及分配和初始化传输数据结构体 `_any_to_object_auxdata`。
    # 当 N 大于 0 时执行循环，处理每个元素
    while (N > 0) {
        # 将 src 指针处的数据复制到 src_ref 变量中，大小为 src_ref 的大小
        memcpy(&src_ref, src, sizeof(src_ref));
        # 根据 data 结构的描述符将 src_ref 或者 Py_None 打包到 dst 指向的位置
        if (PyArray_Pack(data->descr, dst, src_ref ? src_ref : Py_None) < 0) {
            # 如果打包操作失败，返回 -1 表示出错
            return -1;
        }

        # 如果 data 结构指示移动引用并且 src_ref 不为空
        if (data->move_references && src_ref != NULL) {
            # 减少 src_ref 的引用计数
            Py_DECREF(src_ref);
            # 清空 src 指向的内存，大小为 src_ref 的大小
            memset(src, 0, sizeof(src_ref));
        }

        # 减少剩余处理次数 N
        N--;
        # 更新 dst 指针到下一个元素的位置
        dst += dst_stride;
        # 更新 src 指针到下一个元素的位置
        src += src_stride;
    }
    # 循环结束，返回 0 表示正常执行完成
    return 0;
}


NPY_NO_EXPORT int
object_to_any_get_loop(
        PyArrayMethod_Context *context,
        int NPY_UNUSED(aligned), int move_references,
        const npy_intp *NPY_UNUSED(strides),
        PyArrayMethod_StridedLoop **out_loop,
        NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags)
{
    *flags = NPY_METH_REQUIRES_PYAPI;

    /* NOTE: auxdata is only really necessary to flag `move_references` */
    // 分配内存以存储辅助数据对象
    _object_to_any_auxdata *data = PyMem_Malloc(sizeof(*data));
    if (data == NULL) {
        return -1;
    }
    // 设置释放函数和克隆函数
    data->base.free = &_object_to_any_auxdata_free;
    data->base.clone = &_object_to_any_auxdata_clone;

    // 增加对目标描述符的引用计数
    Py_INCREF(context->descriptors[1]);
    // 将描述符设置为辅助数据的一部分
    data->descr = context->descriptors[1];
    // 设置是否需要移动引用标志
    data->move_references = move_references;
    *out_transferdata = (NpyAuxData *)data;
    // 将循环函数设置为指定的函数指针
    *out_loop = &strided_to_strided_object_to_any;
    return 0;
}

/************************** ZERO-PADDED COPY ******************************/

/*
 * Does a strided to strided zero-padded copy for the case where
 * dst_itemsize > src_itemsize
 */
static int
_strided_to_strided_zero_pad_copy(
        PyArrayMethod_Context *context, char *const *args,
        const npy_intp *dimensions, const npy_intp *strides,
        NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    char *src = args[0], *dst = args[1];
    npy_intp src_stride = strides[0], dst_stride = strides[1];
    npy_intp src_itemsize = context->descriptors[0]->elsize;
    npy_intp dst_itemsize = context->descriptors[1]->elsize;

    // 计算需要填充零的字节数
    npy_intp zero_size = dst_itemsize - src_itemsize;

    // 循环执行数据复制和填充零操作
    while (N > 0) {
        // 复制源数据到目标
        memcpy(dst, src, src_itemsize);
        // 填充零到目标的剩余空间
        memset(dst + src_itemsize, 0, zero_size);
        // 移动到下一个源和目标数据位置
        src += src_stride;
        dst += dst_stride;
        --N;
    }
    return 0;
}

/*
 * Does a strided to strided zero-padded copy for the case where
 * dst_itemsize < src_itemsize
 */
static int
_strided_to_strided_truncate_copy(
        PyArrayMethod_Context *context, char *const *args,
        const npy_intp *dimensions, const npy_intp *strides,
        NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    char *src = args[0], *dst = args[1];
    npy_intp src_stride = strides[0], dst_stride = strides[1];
    npy_intp dst_itemsize = context->descriptors[1]->elsize;

    // 循环执行截断复制操作
    while (N > 0) {
        // 复制源数据到目标，只复制目标可以容纳的大小
        memcpy(dst, src, dst_itemsize);
        // 移动到下一个源和目标数据位置
        src += src_stride;
        dst += dst_stride;
        --N;
    }
    return 0;
}

/*
 * Does a strided to strided zero-padded or truncated copy for the case where
 * unicode swapping is needed.
 */
static int
_strided_to_strided_unicode_copyswap(
        PyArrayMethod_Context *context, char *const *args,
        const npy_intp *dimensions, const npy_intp *strides,
        NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    char *src = args[0], *dst = args[1];
    npy_intp src_stride = strides[0], dst_stride = strides[1];
    npy_intp src_itemsize = context->descriptors[0]->elsize;
    // 获取目标数组元素的字节大小，通过上下文中第二个描述符的元素大小获取
    npy_intp dst_itemsize = context->descriptors[1]->elsize;

    // 计算零填充的大小，目标元素大小减去源元素大小
    npy_intp zero_size = dst_itemsize - src_itemsize;

    // 确定复制大小为源元素大小或目标元素大小，取较小者
    npy_intp copy_size = zero_size > 0 ? src_itemsize : dst_itemsize;

    // 目标数组的字符数，每个字符占四个字节
    npy_intp characters = dst_itemsize / 4;
    
    // 循环变量
    int i;

    // 当剩余复制次数大于零时，执行循环
    while (N > 0) {
        // 将源数据复制到目标数据
        memcpy(dst, src, copy_size);

        // 如果存在零填充，将目标数据中源数据后面的部分置零
        if (zero_size > 0) {
            memset(dst + src_itemsize, 0, zero_size);
        }

        // 指向目标数据的指针
        _dst = dst;

        // 对每个字符进行字节交换，每次跳过4个字节
        for (i = 0; i < characters; i++) {
            npy_bswap4_unaligned(_dst);
            _dst += 4;
        }

        // 更新源数据和目标数据的位置
        src += src_stride;
        dst += dst_stride;

        // 减少剩余复制次数
        --N;
    }

    // 返回成功标志
    return 0;
/*************************** WRAP DTYPE COPY/SWAP *************************/
/* Wraps the dtype copy swap function */

/* Defines a structure for wrapping copy swap data */
typedef struct {
    NpyAuxData base;                // Base structure for auxiliary data
    PyArray_CopySwapNFunc *copyswapn; // Pointer to the copy swap function
    int swap;                       // Flag indicating if swap is needed
    PyArrayObject *arr;             // Pointer to the array object
} _wrap_copy_swap_data;

/* Frees the memory allocated for wrap copy swap data */
static void _wrap_copy_swap_data_free(NpyAuxData *data)
{
    _wrap_copy_swap_data *d = (_wrap_copy_swap_data *)data;
    Py_DECREF(d->arr);              // Decrements the reference count of arr
    PyMem_Free(data);               // Frees the memory allocated for data
}

/* Clones wrap copy swap data */
static NpyAuxData *_wrap_copy_swap_data_clone(NpyAuxData *data)
{
    _wrap_copy_swap_data *newdata =
        (_wrap_copy_swap_data *)PyMem_Malloc(sizeof(_wrap_copy_swap_data));
    if (newdata == NULL) {
        return NULL;
    }

    memcpy(newdata, data, sizeof(_wrap_copy_swap_data));  // Copies data into newdata
    Py_INCREF(newdata->arr);        // Increments the reference count of arr

    return (NpyAuxData *)newdata;   // Returns cloned auxiliary data
}

/* Wraps a function for strided to strided copy swap */
static int
_strided_to_strided_wrap_copy_swap(
        PyArrayMethod_Context *NPY_UNUSED(context), char *const *args,
        const npy_intp *dimensions, const npy_intp *strides,
        NpyAuxData *auxdata)
{
    npy_intp N = dimensions[0];     // Number of elements
    char *src = args[0], *dst = args[1];  // Source and destination pointers
    npy_intp src_stride = strides[0], dst_stride = strides[1];  // Strides

    _wrap_copy_swap_data *d = (_wrap_copy_swap_data *)auxdata;  // Cast auxiliary data

    /* Calls the copy swap function stored in d->copyswapn */
    d->copyswapn(dst, dst_stride, src, src_stride, N, d->swap, d->arr);
    return 0;                       // Returns success
}

/*
 * This function is used only via `get_wrapped_legacy_cast_function`
 * when we wrap a legacy DType (or explicitly fall back to the legacy
 * wrapping) for an internal cast.
 */
static int
wrap_copy_swap_function(
        PyArray_Descr *dtype, int should_swap,
        PyArrayMethod_StridedLoop **out_stransfer,
        NpyAuxData **out_transferdata)
{
    /* Allocate memory for wrap copy swap data */
    _wrap_copy_swap_data *data = PyMem_Malloc(sizeof(_wrap_copy_swap_data));
    if (data == NULL) {
        return NPY_FAIL;            // Memory allocation failed
    }

    *out_transferdata = NULL;       // Initialize output auxiliary data

    /* Set swap flag based on should_swap */
    data->swap = should_swap;

    /* Set copyswapn function based on dtype and swap flag */
    if (should_swap) {
        data->copyswapn = dtype->f->copyswapn;
    } else {
        data->copyswapn = dtype->f->copyswap;
    }

    /* Set array object to NULL initially */
    data->arr = NULL;

    /* Allocate memory for _wrap_copy_swap_data succeeded */
    *out_stransfer = &_strided_to_strided_wrap_copy_swap;
    *out_transferdata = (NpyAuxData *)data;

    return NPY_SUCCEED;             // Return success
}
    # 如果数据指针为 NULL，则表示内存分配失败，触发 Python 异常并清理传出参数，返回失败状态
    if (data == NULL) {
        PyErr_NoMemory();
        *out_stransfer = NULL;
        *out_transferdata = NULL;
        return NPY_FAIL;
    }

    # 将自定义数据结构中的函数指针设置为特定函数，用于释放数据
    data->base.free = &_wrap_copy_swap_data_free;
    # 将自定义数据结构中的函数指针设置为特定函数，用于克隆数据
    data->base.clone = &_wrap_copy_swap_data_clone;
    # 获取指定数据类型的 copyswapn 函数，并保存到自定义数据结构中
    data->copyswapn = PyDataType_GetArrFuncs(dtype)->copyswapn;
    # 设置是否需要进行字节交换的标志
    data->swap = should_swap;

    """
     * TODO: This is a hack so the copyswap functions have an array.
     *       The copyswap functions shouldn't need that.
     """
    # 增加数据类型的引用计数，确保在函数结束前不会被释放
    Py_INCREF(dtype);
    # 创建一个具有指定数据类型和形状的新的 PyArrayObject 对象
    npy_intp shape = 1;
    data->arr = (PyArrayObject *)PyArray_NewFromDescr_int(
            &PyArray_Type, dtype,
            1, &shape, NULL, NULL,
            0, NULL, NULL,
            _NPY_ARRAY_ENSURE_DTYPE_IDENTITY);
    # 如果创建 PyArrayObject 对象失败，则释放之前分配的内存并返回失败状态
    if (data->arr == NULL) {
        PyMem_Free(data);
        return NPY_FAIL;
    }

    # 设置传出参数，指向一个特定的函数，用于执行数据交换和拷贝操作
    *out_stransfer = &_strided_to_strided_wrap_copy_swap;
    # 将自定义数据结构转换为 NpyAuxData 类型，并设置传出参数
    *out_transferdata = (NpyAuxData *)data;

    # 返回成功状态
    return NPY_SUCCEED;
/*************************** DTYPE CAST FUNCTIONS *************************/

/* 定义一个结构体 _strided_cast_data，用于存储结构化的数据，包含辅助数据基类、类型转换函数指针、输入和输出数组对象、API 需求标志 */
typedef struct {
    NpyAuxData base;
    PyArray_VectorUnaryFunc *castfunc;  // 指向数组向量一元函数的指针
    PyArrayObject *aip, *aop;  // 输入和输出的 PyArrayObject 对象指针
    npy_bool needs_api;  // 表示是否需要调用 Python C API
} _strided_cast_data;

/* 释放 strided cast 数据的函数 */
static void _strided_cast_data_free(NpyAuxData *data)
{
    _strided_cast_data *d = (_strided_cast_data *)data;
    Py_DECREF(d->aip);  // 减少输入数组对象的引用计数
    Py_DECREF(d->aop);  // 减少输出数组对象的引用计数
    PyMem_Free(data);  // 释放内存
}

/* 复制 strided cast 数据的函数 */
static NpyAuxData *_strided_cast_data_clone(NpyAuxData *data)
{
    _strided_cast_data *newdata =
            (_strided_cast_data *)PyMem_Malloc(sizeof(_strided_cast_data));  // 分配新的 _strided_cast_data 结构体内存空间
    if (newdata == NULL) {
        return NULL;  // 分配失败返回空指针
    }

    memcpy(newdata, data, sizeof(_strided_cast_data));  // 复制数据
    Py_INCREF(newdata->aip);  // 增加输入数组对象的引用计数
    Py_INCREF(newdata->aop);  // 增加输出数组对象的引用计数

    return (NpyAuxData *)newdata;  // 返回复制后的新数据
}

/* 执行对齐的 strided 到 strided 类型转换的函数 */
static int
_aligned_strided_to_strided_cast(
        PyArrayMethod_Context *NPY_UNUSED(context), char *const *args,
        const npy_intp *dimensions, const npy_intp *strides,
        NpyAuxData *auxdata)
{
    npy_intp N = dimensions[0];  // 获取数组的维度
    char *src = args[0], *dst = args[1];  // 获取输入和输出的数据指针
    npy_intp src_stride = strides[0], dst_stride = strides[1];  // 获取输入和输出的步幅

    _strided_cast_data *d = (_strided_cast_data *)auxdata;  // 强制转换为 _strided_cast_data 结构体指针
    PyArray_VectorUnaryFunc *castfunc = d->castfunc;  // 获取类型转换函数指针
    PyArrayObject *aip = d->aip, *aop = d->aop;  // 获取输入和输出的数组对象指针
    npy_bool needs_api = d->needs_api;  // 获取 API 需求标志

    while (N > 0) {
        castfunc(src, dst, 1, aip, aop);  // 执行类型转换函数
        /*
         * 由于通用函数中的错误处理不完善（在编写时，调用该函数之前可能已经存在错误。
         * 在大多数 NumPy 历史中，这些检查完全缺失，所以暂时地这样（直到通用函数被修复）应该是可以的。
         */
        if (needs_api && PyErr_Occurred()) {  // 如果需要 API 并且发生了错误
            return -1;  // 返回 -1 表示出错
        }
        dst += dst_stride;  // 移动到下一个输出数据位置
        src += src_stride;  // 移动到下一个输入数据位置
        --N;  // 减少剩余处理的元素个数
    }
    return 0;  // 成功返回 0
}

/* 这个函数要求 src 是 NPY_OBJECT 类型 */
static int
_aligned_strided_to_strided_cast_decref_src(
        PyArrayMethod_Context *NPY_UNUSED(context), char *const *args,
        const npy_intp *dimensions, const npy_intp *strides,
        NpyAuxData *auxdata)
{
    npy_intp N = dimensions[0];  // 获取数组的维度
    char *src = args[0], *dst = args[1];  // 获取输入和输出的数据指针
    npy_intp src_stride = strides[0], dst_stride = strides[1];  // 获取输入和输出的步幅

    _any_to_object_auxdata *data = (_any_to_object_auxdata *)auxdata;  // 强制转换为 _any_to_object_auxdata 结构体指针
    _strided_cast_data *d = (_strided_cast_data *)data;  // 强制转换为 _strided_cast_data 结构体指针
    PyArray_VectorUnaryFunc *castfunc = d->castfunc;  // 获取类型转换函数指针
    PyArrayObject *aip = d->aip, *aop = d->aop;  // 获取输入和输出的数组对象指针
    npy_bool needs_api = d->needs_api;  // 获取 API 需求标志
    PyObject *src_ref;  // Python 对象引用

    /* 此处未完，但是由于要求不省略任何部分，因此需要继续完整地添加注释 */
    # 当 N 大于 0 时执行循环
    while (N > 0) {
        # 调用 castfunc 对数据进行类型转换
        castfunc(src, dst, 1, aip, aop);
        
        /*
         * 查看 `_aligned_strided_to_strided_cast` 中的注释，可能在调用 `castfunc` 前设置错误。
         */
        # 如果需要 API 并且出现了异常，立即返回 -1
        if (needs_api && PyErr_Occurred()) {
            return -1;
        }
        
        /* 转换完成后，减少源对象的引用计数并将其设置为 NULL */
        # 复制源对象的引用，减少其引用计数
        memcpy(&src_ref, src, sizeof(src_ref));
        Py_XDECREF(src_ref);
        # 将源对象指针所指向的内存区域清零
        memset(src, 0, sizeof(PyObject *));
        # 调试信息：减少源对象引用计数，从对象到非对象
        NPY_DT_DBG_REFTRACE("dec src ref (cast object -> not object)", src_ref);

        # 更新目标指针和源指针以指向下一个元素
        dst += dst_stride;
        src += src_stride;
        # 减少剩余处理次数 N
        --N;
    }
    # 循环结束后返回 0
    return 0;
}

static int
_aligned_contig_to_contig_cast(
        PyArrayMethod_Context *NPY_UNUSED(context), char * const*args,
        const npy_intp *dimensions, const npy_intp *NPY_UNUSED(strides),
        NpyAuxData *auxdata)
{
    // 获取输入的数组维度
    npy_intp N = dimensions[0];
    // 获取输入源和目标数组的指针
    char *src = args[0], *dst = args[1];

    // 从辅助数据中获取类型转换所需的信息
    _strided_cast_data *d = (_strided_cast_data *)auxdata;
    // 检查是否需要调用 Python API
    npy_bool needs_api = d->needs_api;

    // 调用类型转换函数进行数据转换
    d->castfunc(src, dst, N, d->aip, d->aop);
    /*
     * 查看 `_aligned_strided_to_strided_cast` 中的注释，
     * 在调用 `castfunc` 之前可能已经设置了错误状态。
     */
    if (needs_api && PyErr_Occurred()) {
        return -1;
    }
    // 返回成功状态
    return 0;
}


/*
 * 执行 datetime->datetime, timedelta->timedelta,
 * datetime->ascii 或 ascii->datetime 转换
 */
typedef struct {
    NpyAuxData base;
    /* 转换比例 */
    npy_int64 num, denom;
    /* 对于 datetime -> string 转换，目标字符串的长度 */
    npy_intp src_itemsize, dst_itemsize;
    /*
     * 大小为 'src_itemsize + 1' 的缓冲区，
     * 当输入字符串长度正好为 src_itemsize 且没有 NULL 结尾时使用。
     */
    char *tmp_buffer;
    /*
     * 处理 Months 或 Years 等非线性单位时的元数据，
     * 与其他单位相比表现不同。
     */
    PyArray_DatetimeMetaData src_meta, dst_meta;
} _strided_datetime_cast_data;

/* 释放 strided datetime 转换数据的内存 */
static void _strided_datetime_cast_data_free(NpyAuxData *data)
{
    _strided_datetime_cast_data *d = (_strided_datetime_cast_data *)data;
    PyMem_Free(d->tmp_buffer);
    PyMem_Free(data);
}

/* 复制 strided datetime 转换数据的函数 */
static NpyAuxData *_strided_datetime_cast_data_clone(NpyAuxData *data)
{
    _strided_datetime_cast_data *newdata =
            (_strided_datetime_cast_data *)PyMem_Malloc(
                                        sizeof(_strided_datetime_cast_data));
    if (newdata == NULL) {
        return NULL;
    }

    // 复制数据内容
    memcpy(newdata, data, sizeof(_strided_datetime_cast_data));
    // 如果有临时缓冲区，则分配新的缓冲区
    if (newdata->tmp_buffer != NULL) {
        newdata->tmp_buffer = PyMem_Malloc(newdata->src_itemsize + 1);
        if (newdata->tmp_buffer == NULL) {
            PyMem_Free(newdata);
            return NULL;
        }
    }

    return (NpyAuxData *)newdata;
}

static int
_strided_to_strided_datetime_general_cast(
        PyArrayMethod_Context *NPY_UNUSED(context), char * const*args,
        const npy_intp *dimensions, const npy_intp *strides,
        NpyAuxData *auxdata)
{
    // 获取输入的数组维度
    npy_intp N = dimensions[0];
    // 获取输入源和目标数组的指针及其步幅
    char *src = args[0], *dst = args[1];
    npy_intp src_stride = strides[0], dst_stride = strides[1];

    // 从辅助数据中获取类型转换所需的信息
    _strided_datetime_cast_data *d = (_strided_datetime_cast_data *)auxdata;
    npy_int64 dt;
    npy_datetimestruct dts;
    // 当 N 大于 0 时，循环执行以下操作
    while (N > 0) {
        // 从 src 处复制 sizeof(dt) 大小的数据到 dt 中
        memcpy(&dt, src, sizeof(dt));

        // 将 dt 转换为日期时间结构体 dts，使用 d->src_meta 描述元数据
        if (NpyDatetime_ConvertDatetime64ToDatetimeStruct(&d->src_meta, dt, &dts) < 0) {
            // 转换失败时返回 -1
            return -1;
        }
        else {
            // 将日期时间结构体 dts 转换为 datetime64 格式，使用 d->dst_meta 描述目标元数据
            if (NpyDatetime_ConvertDatetimeStructToDatetime64(&d->dst_meta, &dts, &dt) < 0) {
                // 转换失败时返回 -1
                return -1;
            }
        }

        // 将 dt 的数据复制到 dst 处，大小为 sizeof(dt)
        memcpy(dst, &dt, sizeof(dt));

        // 移动 dst 指针到下一个位置，以 dst_stride 为步长
        dst += dst_stride;

        // 移动 src 指针到下一个位置，以 src_stride 为步长
        src += src_stride;

        // N 减少 1
        --N;
    }
    // 循环结束后返回 0，表示成功执行
    return 0;
static int
_strided_to_strided_datetime_cast(
        PyArrayMethod_Context *NPY_UNUSED(context), char * const*args,
        const npy_intp *dimensions, const npy_intp *strides,
        NpyAuxData *auxdata)
{
    npy_intp N = dimensions[0];  # 获取数组的第一个维度大小
    char *src = args[0], *dst = args[1];  # 获取输入和输出数组的起始指针
    npy_intp src_stride = strides[0], dst_stride = strides[1];  # 获取输入和输出数组的步长

    _strided_datetime_cast_data *d = (_strided_datetime_cast_data *)auxdata;  # 将辅助数据转换为 datetime 转换数据结构
    npy_int64 num = d->num, denom = d->denom;  # 获取转换的分子和分母
    npy_int64 dt;  # 用于存储 datetime 数据

    while (N > 0) {  # 迭代处理数组中的每个元素
        memcpy(&dt, src, sizeof(dt));  # 从源数组复制一个 datetime 数据到 dt

        if (dt != NPY_DATETIME_NAT) {  # 如果 dt 不是自然时间
            /* 应用缩放 */
            if (dt < 0) {
                dt = (dt * num - (denom - 1)) / denom;  # 负数情况下的缩放转换
            }
            else {
                dt = dt * num / denom;  # 正数情况下的缩放转换
            }
        }

        memcpy(dst, &dt, sizeof(dt));  # 将处理后的 dt 数据复制到目标数组

        dst += dst_stride;  # 更新目标数组的指针位置
        src += src_stride;  # 更新源数组的指针位置
        --N;  # 减少处理的剩余元素数量
    }
    return 0;  # 返回处理成功
}

static int
_aligned_strided_to_strided_datetime_cast(
        PyArrayMethod_Context *NPY_UNUSED(context), char *const *args,
        const npy_intp *dimensions, const npy_intp *strides,
        NpyAuxData *auxdata)
{
    npy_intp N = dimensions[0];  # 获取数组的第一个维度大小
    char *src = args[0], *dst = args[1];  # 获取输入和输出数组的起始指针
    npy_intp src_stride = strides[0], dst_stride = strides[1];  # 获取输入和输出数组的步长

    _strided_datetime_cast_data *d = (_strided_datetime_cast_data *)auxdata;  # 将辅助数据转换为 datetime 转换数据结构
    npy_int64 num = d->num, denom = d->denom;  # 获取转换的分子和分母
    npy_int64 dt;  # 用于存储 datetime 数据

    while (N > 0) {  # 迭代处理数组中的每个元素
        dt = *(npy_int64 *)src;  # 从源数组中读取一个 datetime 数据到 dt

        if (dt != NPY_DATETIME_NAT) {  # 如果 dt 不是自然时间
            /* 应用缩放 */
            if (dt < 0) {
                dt = (dt * num - (denom - 1)) / denom;  # 负数情况下的缩放转换
            }
            else {
                dt = dt * num / denom;  # 正数情况下的缩放转换
            }
        }

        *(npy_int64 *)dst = dt;  # 将处理后的 dt 数据写入目标数组

        dst += dst_stride;  # 更新目标数组的指针位置
        src += src_stride;  # 更新源数组的指针位置
        --N;  # 减少处理的剩余元素数量
    }
    return 0;  # 返回处理成功
}

static int
_strided_to_strided_datetime_to_string(
        PyArrayMethod_Context *NPY_UNUSED(context), char *const *args,
        const npy_intp *dimensions, const npy_intp *strides,
        NpyAuxData *auxdata)
{
    npy_intp N = dimensions[0];  # 获取数组的第一个维度大小
    char *src = args[0], *dst = args[1];  # 获取输入和输出数组的起始指针
    npy_intp src_stride = strides[0], dst_stride = strides[1];  # 获取输入和输出数组的步长

    _strided_datetime_cast_data *d = (_strided_datetime_cast_data *)auxdata;  # 将辅助数据转换为 datetime 转换数据结构
    npy_intp dst_itemsize = d->dst_itemsize;  # 获取目标字符串的字节大小
    npy_int64 dt;  # 用于存储 datetime 数据
    npy_datetimestruct dts;  # 用于存储 datetime 结构体

    while (N > 0) {  # 迭代处理数组中的每个元素
        memcpy(&dt, src, sizeof(dt));  # 从源数组复制一个 datetime 数据到 dt

        if (NpyDatetime_ConvertDatetime64ToDatetimeStruct(&d->src_meta,
                                               dt, &dts) < 0) {
            return -1;  # 如果转换失败，则返回错误
        }

        /* 将目标初始化为全零 */
        memset(dst, 0, dst_itemsize);  # 将目标数组初始化为全零

        if (NpyDatetime_MakeISO8601Datetime(&dts, dst, dst_itemsize,
                                0, 0, d->src_meta.base, -1,
                                NPY_UNSAFE_CASTING) < 0) {
            return -1;  # 如果生成 ISO8601 格式的日期时间失败，则返回错误
        }

        dst += dst_stride;  # 更新目标数组的指针位置
        src += src_stride;  # 更新源数组的指针位置
        --N;  # 减少处理的剩余元素数量
    }
    # 返回整数值 0
    return 0;
}

static int
_strided_to_strided_string_to_datetime(
        PyArrayMethod_Context *context, char *const *args,
        const npy_intp *dimensions, const npy_intp *strides,
        NpyAuxData *auxdata)
{
    npy_intp N = dimensions[0];  # 获取维度数组的第一个元素作为循环次数 N
    char *src = args[0], *dst = args[1];  # 获取参数数组中的第一个和第二个元素，分别作为源和目标字符串的起始地址
    npy_intp src_itemsize = context->descriptors[0]->elsize;  # 源字符串的元素大小
    npy_intp src_stride = strides[0], dst_stride = strides[1];  # 源字符串和目标字符串的步长

    _strided_datetime_cast_data *d = (_strided_datetime_cast_data *)auxdata;  # 强制转换辅助数据为 _strided_datetime_cast_data 结构体指针
    npy_datetimestruct dts;  # 创建 npy_datetimestruct 结构体变量
    char *tmp_buffer = d->tmp_buffer;  # 辅助数据中的临时缓冲区地址
    char *tmp;  # 临时指针变量

    while (N > 0) {  # 循环，直到 N 减至 0
        npy_int64 dt = ~NPY_DATETIME_NAT;  # 初始化 dt 变量为 NPY_DATETIME_NAT 的按位取反

        /* Replicating strnlen with memchr, because Mac OS X lacks it */
        tmp = memchr(src, '\0', src_itemsize);  # 使用 memchr 复制 strnlen 函数的功能，查找源字符串中的 '\0' 字符位置

        /* If the string is all full, use the buffer */
        if (tmp == NULL) {  # 如果未找到 '\0' 字符
            memcpy(tmp_buffer, src, src_itemsize);  # 将源字符串复制到临时缓冲区
            tmp_buffer[src_itemsize] = '\0';  # 在缓冲区末尾添加 '\0' 结束符

            if (NpyDatetime_ParseISO8601Datetime(
                    tmp_buffer, src_itemsize,
                    d->dst_meta.base, NPY_SAME_KIND_CASTING,
                    &dts, NULL, NULL) < 0) {
                return -1;  # 解析 ISO8601 格式的日期时间字符串，如果失败则返回 -1
            }
        }
        /* Otherwise parse the data in place */
        else {  # 如果找到 '\0' 字符
            if (NpyDatetime_ParseISO8601Datetime(
                    src, tmp - src,
                    d->dst_meta.base, NPY_SAME_KIND_CASTING,
                    &dts, NULL, NULL) < 0) {
                return -1;  # 在原地解析日期时间数据，如果失败则返回 -1
            }
        }

        /* Convert to the datetime */
        if (dt != NPY_DATETIME_NAT &&
                NpyDatetime_ConvertDatetimeStructToDatetime64(&d->dst_meta,
                                               &dts, &dt) < 0) {
            return -1;  # 将日期时间结构体转换为 datetime64 格式，如果失败则返回 -1
        }

        memcpy(dst, &dt, sizeof(dt));  # 将转换后的 datetime64 数据复制到目标字符串

        dst += dst_stride;  # 移动目标字符串指针
        src += src_stride;  # 移动源字符串指针
        --N;  # 减少循环次数计数
    }
    return 0;  # 返回成功
}

/*
 * Assumes src_dtype and dst_dtype are both datetimes or both timedeltas
 */
NPY_NO_EXPORT int
get_nbo_cast_datetime_transfer_function(int aligned,
                            PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
                            PyArrayMethod_StridedLoop **out_stransfer,
                            NpyAuxData **out_transferdata)
{
    PyArray_DatetimeMetaData *src_meta, *dst_meta;
    npy_int64 num = 0, denom = 0;
    _strided_datetime_cast_data *data;

    src_meta = get_datetime_metadata_from_dtype(src_dtype);  # 从源数据类型中获取日期时间元数据
    if (src_meta == NULL) {
        return NPY_FAIL;  # 如果获取失败，则返回失败标志
    }
    dst_meta = get_datetime_metadata_from_dtype(dst_dtype);  # 从目标数据类型中获取日期时间元数据
    if (dst_meta == NULL) {
        return NPY_FAIL;  # 如果获取失败，则返回失败标志
    }

    get_datetime_conversion_factor(src_meta, dst_meta, &num, &denom);  # 获取日期时间转换因子

    if (num == 0) {
        return NPY_FAIL;  # 如果转换因子为零，则返回失败标志
    }

    /* Allocate the data for the casting */
    data = (_strided_datetime_cast_data *)PyMem_Malloc(
                                    sizeof(_strided_datetime_cast_data));  # 分配用于类型转换的数据空间
    /*
     * 如果数据指针为空，表示内存分配失败，
     * 触发 Python 错误处理并将输出指针置为空，
     * 返回失败状态码。
     */
    if (data == NULL) {
        PyErr_NoMemory();
        *out_stransfer = NULL;
        *out_transferdata = NULL;
        return NPY_FAIL;
    }

    /*
     * 设置数据结构中的释放函数和克隆函数，
     * 设置分子和分母值，以及临时缓冲区为空。
     */
    data->base.free = &_strided_datetime_cast_data_free;
    data->base.clone = &_strided_datetime_cast_data_clone;
    data->num = num;
    data->denom = denom;
    data->tmp_buffer = NULL;

    /*
     * 处理日期时间类型的特殊情况（但不包括时间间隔），
     * 特别处理非线性单位（年和月）。对于时间间隔，
     * 使用平均年和月值。
     */
    if (src_dtype->type_num == NPY_DATETIME &&
            (src_meta->base == NPY_FR_Y ||
             src_meta->base == NPY_FR_M ||
             dst_meta->base == NPY_FR_Y ||
             dst_meta->base == NPY_FR_M)) {
        // 复制源元数据和目标元数据到数据结构
        memcpy(&data->src_meta, src_meta, sizeof(data->src_meta));
        memcpy(&data->dst_meta, dst_meta, sizeof(data->dst_meta));
        // 设置输出转换函数为通用日期时间转换函数
        *out_stransfer = &_strided_to_strided_datetime_general_cast;
    }
    else if (aligned) {
        // 如果对齐标志为真，设置输出转换函数为对齐日期时间转换函数
        *out_stransfer = &_aligned_strided_to_strided_datetime_cast;
    }
    else {
        // 否则，设置输出转换函数为普通日期时间转换函数
        *out_stransfer = &_strided_to_strided_datetime_cast;
    }

    // 将数据结构转换为通用数据附加对象，并赋给输出转换数据指针
    *out_transferdata = (NpyAuxData *)data;
#if NPY_DT_DBG_TRACING
    # 在调试跟踪模式下打印消息，显示源数据类型到目标数据类型的转换过程信息
    printf("Dtype transfer from ");
    _safe_print((PyObject *)src_dtype);  // 打印源数据类型对象的信息
    printf(" to ");
    _safe_print((PyObject *)dst_dtype);  // 打印目标数据类型对象的信息
    printf("\n");
#endif

    // 返回成功状态码
    return NPY_SUCCEED;
}

// 获取从NBO日期时间到字符串转换的传输函数
NPY_NO_EXPORT int
get_nbo_datetime_to_string_transfer_function(
        PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
        PyArrayMethod_StridedLoop **out_stransfer, NpyAuxData **out_transferdata)
{
    PyArray_DatetimeMetaData *src_meta;
    _strided_datetime_cast_data *data;

    // 从源数据类型获取日期时间的元数据
    src_meta = get_datetime_metadata_from_dtype(src_dtype);
    if (src_meta == NULL) {
        return NPY_FAIL;
    }

    /* 分配用于转换的数据 */
    data = (_strided_datetime_cast_data *)PyMem_Malloc(
                                    sizeof(_strided_datetime_cast_data));
    if (data == NULL) {
        PyErr_NoMemory();  // 内存分配失败时设置内存错误异常
        *out_stransfer = NULL;
        *out_transferdata = NULL;
        return NPY_FAIL;
    }
    data->base.free = &_strided_datetime_cast_data_free;  // 设置数据释放函数
    data->base.clone = &_strided_datetime_cast_data_clone;  // 设置数据克隆函数
    data->dst_itemsize = dst_dtype->elsize;  // 设置目标数据类型的字节大小
    data->tmp_buffer = NULL;

    // 复制源日期时间元数据到数据结构中
    memcpy(&data->src_meta, src_meta, sizeof(data->src_meta));

    // 设置输出的传输函数为日期时间到字符串的转换函数
    *out_stransfer = &_strided_to_strided_datetime_to_string;
    *out_transferdata = (NpyAuxData *)data;

#if NPY_DT_DBG_TRACING
    // 在调试跟踪模式下打印消息，显示源数据类型到目标数据类型的转换过程信息
    printf("Dtype transfer from ");
    _safe_print((PyObject *)src_dtype);  // 打印源数据类型对象的信息
    printf(" to ");
    _safe_print((PyObject *)dst_dtype);  // 打印目标数据类型对象的信息
    printf("\n");
#endif

    // 返回成功状态码
    return NPY_SUCCEED;
}


// 获取从日期时间到Unicode字符串的传输函数
NPY_NO_EXPORT int
get_datetime_to_unicode_transfer_function(int aligned,
                            npy_intp src_stride, npy_intp dst_stride,
                            PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
                            PyArrayMethod_StridedLoop **out_stransfer,
                            NpyAuxData **out_transferdata,
                            int *out_needs_api)
{
    PyArray_Descr *str_dtype;

    // 获取一个适配于UNICODE的ASCII字符串数据类型
    str_dtype = PyArray_DescrNewFromType(NPY_STRING);
    if (str_dtype == NULL) {
        return NPY_FAIL;
    }
    str_dtype->elsize = dst_dtype->elsize / 4;  // 设置字符串数据类型的字节大小

    // 确保源数据类型不是交换的
    assert(PyDataType_ISNOTSWAPPED(src_dtype));

    // 获取NBO日期时间到字符串对齐连续函数
    if (get_nbo_datetime_to_string_transfer_function(
            src_dtype, str_dtype,
            out_stransfer, out_transferdata) != NPY_SUCCEED) {
        Py_DECREF(str_dtype);
        return NPY_FAIL;
    }

    // 封装对齐的传输函数
    int res = wrap_aligned_transferfunction(
            aligned, 0,  /* 不需要保证连续性 */
            src_stride, dst_stride,
            src_dtype, dst_dtype,
            src_dtype, str_dtype,
            out_stransfer, out_transferdata, out_needs_api);
    Py_DECREF(str_dtype);
    if (res < 0) {
        return NPY_FAIL;
    }
    return NPY_SUCCEED;


注释：


    // 返回 NPY_SUCCEED，表示函数成功执行
}

NPY_NO_EXPORT int
get_nbo_string_to_datetime_transfer_function(
        PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
        PyArrayMethod_StridedLoop **out_stransfer, NpyAuxData **out_transferdata)
{
    // 从目标数据类型中获取日期时间元数据
    PyArray_DatetimeMetaData *dst_meta;
    dst_meta = get_datetime_metadata_from_dtype(dst_dtype);
    if (dst_meta == NULL) {
        return NPY_FAIL;
    }

    /* Allocate the data for the casting */
    // 分配日期时间转换数据的内存
    _strided_datetime_cast_data *data;
    data = (_strided_datetime_cast_data *)PyMem_Malloc(
                                    sizeof(_strided_datetime_cast_data));
    if (data == NULL) {
        PyErr_NoMemory();
        *out_stransfer = NULL;
        *out_transferdata = NULL;
        return NPY_FAIL;
    }
    data->base.free = &_strided_datetime_cast_data_free;
    data->base.clone = &_strided_datetime_cast_data_clone;
    data->src_itemsize = src_dtype->elsize;
    data->tmp_buffer = PyMem_Malloc(data->src_itemsize + 1);
    if (data->tmp_buffer == NULL) {
        PyErr_NoMemory();
        PyMem_Free(data);
        *out_stransfer = NULL;
        *out_transferdata = NULL;
        return NPY_FAIL;
    }

    // 复制目标日期时间元数据
    memcpy(&data->dst_meta, dst_meta, sizeof(data->dst_meta));

    // 设置输出转换函数和转换数据
    *out_stransfer = &_strided_to_strided_string_to_datetime;
    *out_transferdata = (NpyAuxData *)data;

#if NPY_DT_DBG_TRACING
    // 调试输出转换的数据类型信息
    printf("Dtype transfer from ");
    _safe_print((PyObject *)src_dtype);
    printf(" to ");
    _safe_print((PyObject *)dst_dtype);
    printf("\n");
#endif

    return NPY_SUCCEED;
}

NPY_NO_EXPORT int
get_unicode_to_datetime_transfer_function(int aligned,
                            npy_intp src_stride, npy_intp dst_stride,
                            PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
                            PyArrayMethod_StridedLoop **out_stransfer,
                            NpyAuxData **out_transferdata,
                            int *out_needs_api)
{
    // 获取一个适配UNICODE数据类型的ASCII字符串数据类型
    PyArray_Descr *str_dtype;
    str_dtype = PyArray_DescrNewFromType(NPY_STRING);
    if (str_dtype == NULL) {
        return NPY_FAIL;
    }
    assert(src_dtype->type_num == NPY_UNICODE);
    str_dtype->elsize = src_dtype->elsize / 4;

    // 获取字符串到NBO日期时间对齐的函数
    if (get_nbo_string_to_datetime_transfer_function(
            str_dtype, dst_dtype,
            out_stransfer, out_transferdata) != NPY_SUCCEED) {
        Py_DECREF(str_dtype);
        return NPY_FAIL;
    }

    // 包装对齐的转换函数
    int res = wrap_aligned_transferfunction(
            aligned, 0,  /* no need to ensure contiguous */
            src_stride, dst_stride,
            src_dtype, dst_dtype,
            str_dtype, dst_dtype,
            out_stransfer, out_transferdata, out_needs_api);
    Py_DECREF(str_dtype);

    if (res < 0) {
        return NPY_FAIL;
    }
    return NPY_SUCCEED;
}

NPY_NO_EXPORT int
# 定义一个函数，获取向后兼容数据类型转换函数
get_legacy_dtype_cast_function(
        int aligned, npy_intp src_stride, npy_intp dst_stride,
        PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
        int move_references,
        PyArrayMethod_StridedLoop **out_stransfer, NpyAuxData **out_transferdata,
        int *out_needs_api, int *out_needs_wrap)
{
    # 声明_strided_cast_data类型的指针变量data，用于存储转换数据
    _strided_cast_data *data;
    # 声明一个指向单目函数的指针变量castfunc，用于存储类型转换函数
    PyArray_VectorUnaryFunc *castfunc;
    # 声明一个临时的数据类型描述符指针变量tmp_dtype
    PyArray_Descr *tmp_dtype;
    # 初始化shape变量为1，用于描述数据的形状
    npy_intp shape = 1;
    # 获取源数据类型元素的字节大小
    npy_intp src_itemsize = src_dtype->elsize;
    # 获取目标数据类型元素的字节大小
    npy_intp dst_itemsize = dst_dtype->elsize;

    # 检查是否需要包装数据，根据对齐、字节序等条件判断
    *out_needs_wrap = !aligned ||
                      !PyArray_ISNBO(src_dtype->byteorder) ||
                      !PyArray_ISNBO(dst_dtype->byteorder);

    /* Check the data types whose casting functions use API calls */
    # 检查源数据类型和目标数据类型是否需要调用API函数来进行转换
    switch (src_dtype->type_num) {
        case NPY_OBJECT:
        case NPY_STRING:
        case NPY_UNICODE:
        case NPY_VOID:
            if (out_needs_api) {
                *out_needs_api = 1;
            }
            break;
    }
    switch (dst_dtype->type_num) {
        case NPY_OBJECT:
        case NPY_STRING:
        case NPY_UNICODE:
        case NPY_VOID:
            if (out_needs_api) {
                *out_needs_api = 1;
            }
            break;
    }

    # 如果源数据类型或目标数据类型需要使用Python API，则设置out_needs_api标志
    if (PyDataType_FLAGCHK(src_dtype, NPY_NEEDS_PYAPI) ||
            PyDataType_FLAGCHK(dst_dtype, NPY_NEEDS_PYAPI)) {
        if (out_needs_api) {
            *out_needs_api = 1;
        }
    }

    # 获取数据类型转换函数castfunc，将源数据类型转换为目标数据类型
    castfunc = PyArray_GetCastFunc(src_dtype, dst_dtype->type_num);
    if (!castfunc) {
        *out_stransfer = NULL;
        *out_transferdata = NULL;
        return NPY_FAIL;
    }

    # 分配用于类型转换的数据结构_strided_cast_data
    data = (_strided_cast_data *)PyMem_Malloc(sizeof(_strided_cast_data));
    if (data == NULL) {
        PyErr_NoMemory();
        *out_stransfer = NULL;
        *out_transferdata = NULL;
        return NPY_FAIL;
    }
    # 初始化_strided_cast_data结构体的成员函数和属性
    data->base.free = &_strided_cast_data_free;
    data->base.clone = &_strided_cast_data_clone;
    data->castfunc = castfunc;
    data->needs_api = *out_needs_api;
    /*
     * TODO: This is a hack so the cast functions have an array.
     *       The cast functions shouldn't need that.  Also, since we
     *       always handle byte order conversions, this array should
     *       have native byte order.
     */
    # 如果源数据类型的字节顺序是本机字节顺序，则使用其本身作为tmp_dtype
    if (PyArray_ISNBO(src_dtype->byteorder)) {
        tmp_dtype = src_dtype;
        Py_INCREF(tmp_dtype);
    }
    else {
        # 否则，创建一个新的数据类型描述符tmp_dtype，并指定为本机字节顺序
        tmp_dtype = PyArray_DescrNewByteorder(src_dtype, NPY_NATIVE);
        if (tmp_dtype == NULL) {
            PyMem_Free(data);
            return NPY_FAIL;
        }
    }
    # 使用数据类型描述符tmp_dtype创建一个新的PyArrayObject对象，存储在data->aip中
    data->aip = (PyArrayObject *)PyArray_NewFromDescr_int(
            &PyArray_Type, tmp_dtype,
            1, &shape, NULL, NULL,
            0, NULL, NULL,
            _NPY_ARRAY_ENSURE_DTYPE_IDENTITY);
    if (data->aip == NULL) {
        PyMem_Free(data);
        return NPY_FAIL;
    }
    """
    如果目标数据类型的字节顺序是本地字节顺序，使用目标数据类型作为临时数据类型。
    这是一个暂时的解决方案，因为类型转换函数本应不需要这样的数组。
    此外，由于我们始终处理字节顺序转换，这个数组应该是本地字节顺序的。

    如果条件不满足（目标数据类型字节顺序不是本地字节顺序），
    创建一个新的目标数据类型，使用本地字节顺序，并将其赋给临时数据类型。
    如果创建失败，释放之前分配的资源并返回失败。

    创建一个新的 PyArrayObject，使用临时数据类型和给定的形状信息。
    如果创建失败，释放之前分配的资源并返回失败。

    如果需要移动引用并且源数据类型是 NPY_OBJECT 类型，
    设置输出转换函数为 _aligned_strided_to_strided_cast_decref_src。

    否则，根据源和目标的步长情况来选择合适的转换函数：
    - 如果源和目标的步长都是连续的，或者调用者需要包装返回值（*out_needs_wrap=True），
      则设置输出转换函数为 _aligned_contig_to_contig_cast。
    - 否则，设置输出转换函数为 _aligned_strided_to_strided_cast。

    将数据结构中的转换数据设置为之前分配的数据结构，并返回成功状态。
    """
/* 结构体定义，表示复制从一个元素到N个连续元素 */
typedef struct {
    NpyAuxData base;                // 基本辅助数据结构
    npy_intp N;                     // 要复制的连续元素的数量
    NPY_cast_info wrapped;          // 包装的类型转换信息
    /* 如果finish->func非NULL，源需要进行减少引用计数 */
    NPY_traverse_info decref_src;   // 遍历信息，用于源数据的引用计数减少
} _one_to_n_data;

/* 数据释放函数 */
static void _one_to_n_data_free(NpyAuxData *data)
{
    _one_to_n_data *d = (_one_to_n_data *)data;
    NPY_cast_info_xfree(&d->wrapped);           // 释放包装的类型转换信息
    NPY_traverse_info_xfree(&d->decref_src);    // 释放遍历信息
    PyMem_Free(data);                           // 释放内存
}

/* 数据复制函数 */
static NpyAuxData *_one_to_n_data_clone(NpyAuxData *data)
{
    _one_to_n_data *d = (_one_to_n_data *)data;
    _one_to_n_data *newdata;

    /* 分配数据，并填充它 */
    newdata = (_one_to_n_data *)PyMem_Malloc(sizeof(_one_to_n_data));
    if (newdata == NULL) {
        return NULL;
    }
    newdata->base.free = &_one_to_n_data_free;    // 设置释放函数
    newdata->base.clone = &_one_to_n_data_clone;  // 设置复制函数
    newdata->N = d->N;                            // 复制N的值

    /* 初始化遍历信息，以防错误或未使用 */
    NPY_traverse_info_init(&newdata->decref_src);

    /* 复制包装的类型转换信息 */
    if (NPY_cast_info_copy(&newdata->wrapped, &d->wrapped) < 0) {
        _one_to_n_data_free((NpyAuxData *)newdata);
        return NULL;
    }

    /* 如果源数据的减少引用计数函数为空，直接返回新数据 */
    if (d->decref_src.func == NULL) {
        return (NpyAuxData *)newdata;
    }

    /* 复制遍历信息 */
    if (NPY_traverse_info_copy(&newdata->decref_src, &d->decref_src) < 0) {
        _one_to_n_data_free((NpyAuxData *)newdata);
        return NULL;
    }

    return (NpyAuxData *)newdata;
}

/* 实现从一个步进数组到另一个步进数组的一到N复制 */
static int
_strided_to_strided_one_to_n(
        PyArrayMethod_Context *NPY_UNUSED(context), char *const *args,
        const npy_intp *dimensions, const npy_intp *strides,
        NpyAuxData *auxdata)
{
    npy_intp N = dimensions[0];                     // 获取维度大小N
    char *src = args[0], *dst = args[1];            // 获取源和目标数组的指针
    npy_intp src_stride = strides[0], dst_stride = strides[1];   // 获取源和目标数组的步进

    _one_to_n_data *d = (_one_to_n_data *)auxdata;  // 获取辅助数据结构

    const npy_intp subN = d->N;                     // 获取子数组的大小N
    npy_intp sub_strides[2] = {0, d->wrapped.descriptors[1]->elsize};  // 子数组的步进设置

    /* 循环复制 */
    while (N > 0) {
        char *sub_args[2] = {src, dst};             // 子数组参数
        /* 调用包装的函数复制数据 */
        if (d->wrapped.func(&d->wrapped.context,
                sub_args, &subN, sub_strides, d->wrapped.auxdata) < 0) {
            return -1;                              // 复制失败返回-1
        }

        src += src_stride;                          // 更新源数组指针
        dst += dst_stride;                          // 更新目标数组指针
        --N;                                        // 减少N
    }
    return 0;                                       // 返回成功
}

/* 实现从一个步进数组到另一个步进数组的一到N复制，包含结束处理 */
static int
_strided_to_strided_one_to_n_with_finish(
        PyArrayMethod_Context *NPY_UNUSED(context), char *const *args,
        const npy_intp *dimensions, const npy_intp *strides,
        NpyAuxData *auxdata)
{
    npy_intp N = dimensions[0];                     // 获取维度大小N
    char *src = args[0], *dst = args[1];            // 获取源和目标数组的指针
    npy_intp src_stride = strides[0], dst_stride = strides[1];   // 获取源和目标数组的步进

    _one_to_n_data *d = (_one_to_n_data *)auxdata;  // 获取辅助数据结构

    const npy_intp subN = d->N;                     // 获取子数组的大小N
    const npy_intp one_item = 1, zero_stride = 0;   // 单个元素和零步进
    npy_intp sub_strides[2] = {0, d->wrapped.descriptors[1]->elsize};  // 子数组的步进设置

    /* 循环复制 */
    while (N > 0) {
        char *sub_args[2] = {src, dst};             // 子数组参数
        /* 调用包装的函数复制数据 */
        if (d->wrapped.func(&d->wrapped.context,
                sub_args, &subN, sub_strides, d->wrapped.auxdata) < 0) {
            return -1;                              // 复制失败返回-1
        }

        src += src_stride;                          // 更新源数组指针
        dst += dst_stride;                          // 更新目标数组指针
        --N;                                        // 减少N
    }
    return 0;                                       // 返回成功
}
    // 当 N 大于 0 时执行循环
    while (N > 0) {
        // 创建包含两个元素的字符指针数组 sub_args，分别指向 src 和 dst
        char *sub_args[2] = {src, dst};
        // 调用 wrapped 结构体中的 func 函数，传递参数给函数进行处理
        // 如果返回值小于 0，表示执行失败，返回 -1
        if (d->wrapped.func(&d->wrapped.context,
                sub_args, &subN, sub_strides, d->wrapped.auxdata) < 0) {
            return -1;
        }

        // 调用 decref_src 结构体中的 func 函数，进行资源的减少引用计数
        // 如果返回值小于 0，表示执行失败，返回 -1
        if (d->decref_src.func(NULL, d->decref_src.descr,
                src, one_item, zero_stride, d->decref_src.auxdata) < 0) {
            return -1;
        }

        // 更新 src 和 dst 的位置，移动到下一个元素的位置
        src += src_stride;
        dst += dst_stride;
        // N 减少 1，表示处理了一个元素
        --N;
    }
    // 循环结束后返回 0，表示成功执行
    return 0;
    # 分配内存以存储 _one_to_n_data 结构体
    data = PyMem_Malloc(sizeof(_one_to_n_data));
    if (data == NULL) {
        PyErr_NoMemory();
        return NPY_FAIL;
    }

    # 设置 _one_to_n_data 结构体的 free 和 clone 函数指针
    data->base.free = &_one_to_n_data_free;
    data->base.clone = &_one_to_n_data_clone;
    # 设置数据个数 N
    data->N = N;
    # 初始化 decref_src 成员
    NPY_traverse_info_init(&data->decref_src);  /* In case of error */

    /*
     * 设置 move_references 为 0，由包装的传输函数处理
     * 将 src_stride 设置为零，因为是从一个到多个的复制
     * 将 dst_stride 设置为连续的，因为子数组始终是连续的
     */
    if (PyArray_GetDTypeTransferFunction(aligned,
                    0, dst_dtype->elsize,
                    src_dtype, dst_dtype,
                    0,
                    &data->wrapped,
                    out_flags) != NPY_SUCCEED) {
        NPY_AUXDATA_FREE((NpyAuxData *)data);
        return NPY_FAIL;
    }

    # 如果需要 DECREF 源对象，则设置 src_dtype
    if (move_references && PyDataType_REFCHK(src_dtype)) {
        # 获取清除函数及相关标志
        NPY_ARRAYMETHOD_FLAGS clear_flags;
        if (PyArray_GetClearFunction(
                aligned, src_stride, src_dtype,
                &data->decref_src, &clear_flags) < 0) {
            NPY_AUXDATA_FREE((NpyAuxData *)data);
            return NPY_FAIL;
        }
        # 更新 out_flags
        *out_flags = PyArrayMethod_COMBINED_FLAGS(*out_flags, clear_flags);
    }

    # 根据 decref_src.func 是否为空，设置 out_stransfer 指针
    if (data->decref_src.func == NULL) {
        *out_stransfer = &_strided_to_strided_one_to_n;
    }
    else {
        *out_stransfer = &_strided_to_strided_one_to_n_with_finish;
    }
    # 设置 out_transferdata
    *out_transferdata = (NpyAuxData *)data;

    return NPY_SUCCEED;
}



/**************************** COPY N TO N CONTIGUOUS ************************/

# 将 N 个连续元素复制到 N 个连续元素
typedef struct {
    NpyAuxData base;
    NPY_cast_info wrapped;
    npy_intp N;
    npy_intp strides[2];  /* avoid look up on the dtype (dst can be NULL) */
} _n_to_n_data;

# 传输数据释放函数
static void _n_to_n_data_free(NpyAuxData *data)
{
    # 强制转换数据类型为 _n_to_n_data
    _n_to_n_data *d = (_n_to_n_data *)data;
    # 释放 wrapped 成员
    NPY_cast_info_xfree(&d->wrapped);
    # 释放内存
    PyMem_Free(data);
}

# 传输数据复制函数
static NpyAuxData *_n_to_n_data_clone(NpyAuxData *data)
{
    # 强制转换数据类型为 _n_to_n_data
    _n_to_n_data *d = (_n_to_n_data *)data;
    _n_to_n_data *newdata;

    # 分配新的 _n_to_n_data 结构体
    newdata = (_n_to_n_data *)PyMem_Malloc(sizeof(_n_to_n_data));
    if (newdata == NULL) {
        return NULL;
    }
    # 将原数据复制到新数据结构体中
    *newdata = *d;
    // 如果调用 NPY_cast_info_copy 函数返回值小于 0，则表示复制失败
    if (NPY_cast_info_copy(&newdata->wrapped, &d->wrapped) < 0) {
        // 如果复制失败，释放 newdata 对象的资源
        _n_to_n_data_free((NpyAuxData *)newdata);
    }

    // 返回一个指向 newdata 的 NpyAuxData 指针类型的指针
    return (NpyAuxData *)newdata;
/*
 * Closing brace indicating the end of a previous function or code block.
 */

static int
_strided_to_strided_1_to_1(
        PyArrayMethod_Context *NPY_UNUSED(context), char *const *args,
        const npy_intp *dimensions, const npy_intp *strides,
        NpyAuxData *auxdata)
{
    // Casts the auxdata to _n_to_n_data structure
    _n_to_n_data *d = (_n_to_n_data *)auxdata;
    // Calls a wrapped function with provided arguments, dimensions, strides, and auxdata
    return d->wrapped.func(&d->wrapped.context,
            args, dimensions, strides, d->wrapped.auxdata);
}

static int
_strided_to_strided_n_to_n(
        PyArrayMethod_Context *NPY_UNUSED(context), char *const *args,
        const npy_intp *dimensions, const npy_intp *strides,
        NpyAuxData *auxdata)
{
    // Retrieves the number of elements from dimensions
    npy_intp N = dimensions[0];
    // Retrieves pointers to source and destination arrays
    char *src = args[0], *dst = args[1];
    // Retrieves strides for source and destination arrays
    npy_intp src_stride = strides[0], dst_stride = strides[1];

    // Casts the auxdata to _n_to_n_data structure
    _n_to_n_data *d = (_n_to_n_data *)auxdata;
    // Retrieves subN from the _n_to_n_data structure
    npy_intp subN = d->N;

    // Loops through N elements
    while (N > 0) {
        // Creates an array of pointers to source and destination arrays
        char *sub_args[2] = {src, dst};
        // Calls a wrapped function with sub_args, subN, d->strides, and d->wrapped.auxdata
        if (d->wrapped.func(&d->wrapped.context,
                sub_args, &subN, d->strides, d->wrapped.auxdata) < 0) {
            return -1;
        }
        // Moves source and destination pointers by their respective strides
        src += src_stride;
        dst += dst_stride;
        // Decrements N
        --N;
    }
    return 0;
}

static int
_contig_to_contig_n_to_n(
        PyArrayMethod_Context *NPY_UNUSED(context), char *const *args,
        const npy_intp *dimensions, const npy_intp *NPY_UNUSED(strides),
        NpyAuxData *auxdata)
{
    // Retrieves the number of elements from dimensions
    npy_intp N = dimensions[0];
    // Retrieves pointers to source and destination arrays
    char *src = args[0], *dst = args[1];

    // Casts the auxdata to _n_to_n_data structure
    _n_to_n_data *d = (_n_to_n_data *)auxdata;
    // Calculates subN as the product of N and d->N
    npy_intp subN = N * d->N;

    // Creates an array of pointers to source and destination arrays
    char *sub_args[2] = {src, dst};
    // Calls a wrapped function with sub_args, subN, d->strides, and d->wrapped.auxdata
    if (d->wrapped.func(&d->wrapped.context,
            sub_args, &subN, d->strides, d->wrapped.auxdata) < 0) {
        return -1;
    }
    return 0;
}


/*
 * Note that this function is currently both used for structured dtype
 * casting as well as a decref function (with `dst_dtype == NULL`)
 */
static int
get_n_to_n_transfer_function(int aligned,
                            npy_intp src_stride, npy_intp dst_stride,
                            PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
                            int move_references,
                            npy_intp N,
                            PyArrayMethod_StridedLoop **out_stransfer,
                            NpyAuxData **out_transferdata,
                            NPY_ARRAYMETHOD_FLAGS *out_flags)
{
    // Allocates memory for _n_to_n_data structure
    _n_to_n_data *data = PyMem_Malloc(sizeof(_n_to_n_data));
    if (data == NULL) {
        // Raises a memory error if allocation fails
        PyErr_NoMemory();
        return NPY_FAIL;
    }
    // Initializes base.free and base.clone function pointers
    data->base.free = &_n_to_n_data_free;
    data->base.clone = &_n_to_n_data_clone;
    // Sets N to the provided value
    data->N = N;

    // Checks if N is not equal to 1
    if (N != 1) {
        // Sets src_stride to src_dtype->elsize and dst_stride to dst_dtype->elsize if dst_dtype is not NULL
        src_stride = src_dtype->elsize;
        dst_stride = dst_dtype != NULL ? dst_dtype->elsize : 0;
        // Stores src_stride and dst_stride in data->strides for later access
        data->strides[0] = src_stride;
        data->strides[1] = dst_stride;
    }
}
    /*
     * 如果数据已对齐，设置 src_stride 和 dst_stride 为连续的，因为
     * 子数组始终是连续的。
     */
    if (PyArray_GetDTypeTransferFunction(aligned,
                    src_stride, dst_stride,
                    src_dtype, dst_dtype,
                    move_references,
                    &data->wrapped,
                    out_flags) != NPY_SUCCEED) {
        NPY_AUXDATA_FREE((NpyAuxData *)data);
        return NPY_FAIL;
    }

    if (N == 1) {
        /*
         * 不需要包装，可以直接复制。原则上这一步可以完全优化掉，
         * 但需要替换上下文（以使用未打包的数据类型）。
         */
        *out_stransfer = &_strided_to_strided_1_to_1;
    }
    else if (src_stride == N * src_stride &&
             dst_stride == N * dst_stride) {
        /* 子数组可以合并（可能非常罕见） */
        *out_stransfer = &_contig_to_contig_n_to_n;
    }
    else {
        *out_stransfer = &_strided_to_strided_n_to_n;
    }
    *out_transferdata = (NpyAuxData *)data;

    return NPY_SUCCEED;
/********************** COPY WITH SUBARRAY BROADCAST ************************/

/* 定义一个结构体用于存储子数组广播时的偏移量和计数信息 */
typedef struct {
    npy_intp offset, count;
} _subarray_broadcast_offsetrun;

/* 定义一个结构体用于存储子数组广播时的数据传输信息 */
typedef struct {
    NpyAuxData base;                /* 基础辅助数据结构 */
    NPY_cast_info wrapped;          /* 封装的数据转换信息 */
    NPY_traverse_info decref_src;   /* 源数据遍历信息（应该考虑废弃该用例） */
    NPY_traverse_info decref_dst;   /* 目标数据遍历信息（应该考虑废弃该用例） */
    npy_intp src_N, dst_N;          /* 源和目标数据的维度数 */
    npy_intp run_count;             /* 数据传输的运行长度编码表示 */
    _subarray_broadcast_offsetrun offsetruns[];  /* 偏移量和计数信息数组 */
} _subarray_broadcast_data;

/* 释放子数组广播数据的函数 */
static void _subarray_broadcast_data_free(NpyAuxData *data)
{
    _subarray_broadcast_data *d = (_subarray_broadcast_data *)data;
    NPY_cast_info_xfree(&d->wrapped);               /* 释放封装的数据转换信息 */
    NPY_traverse_info_xfree(&d->decref_src);       /* 释放源数据遍历信息 */
    NPY_traverse_info_xfree(&d->decref_dst);       /* 释放目标数据遍历信息 */
    PyMem_Free(data);                               /* 释放整个辅助数据结构内存 */
}

/* 复制子数组广播数据的函数 */
static NpyAuxData *_subarray_broadcast_data_clone(NpyAuxData *data)
{
    _subarray_broadcast_data *d = (_subarray_broadcast_data *)data;

    npy_intp offsetruns_size = d->run_count * sizeof(_subarray_broadcast_offsetrun);
    npy_intp structsize = sizeof(_subarray_broadcast_data) + offsetruns_size;

    /* 分配内存并填充数据 */
    _subarray_broadcast_data *newdata = PyMem_Malloc(structsize);
    if (newdata == NULL) {
        return NULL;
    }
    newdata->base.free = &_subarray_broadcast_data_free;
    newdata->base.clone = &_subarray_broadcast_data_clone;
    newdata->src_N = d->src_N;
    newdata->dst_N = d->dst_N;
    newdata->run_count = d->run_count;
    memcpy(newdata->offsetruns, d->offsetruns, offsetruns_size);

    NPY_traverse_info_init(&newdata->decref_src);
    NPY_traverse_info_init(&newdata->decref_dst);

    /* 复制封装的数据转换信息 */
    if (NPY_cast_info_copy(&newdata->wrapped, &d->wrapped) < 0) {
        _subarray_broadcast_data_free((NpyAuxData *)newdata);
        return NULL;
    }
    /* 复制源数据遍历信息 */
    if (d->decref_src.func != NULL) {
        if (NPY_traverse_info_copy(&newdata->decref_src, &d->decref_src) < 0) {
            _subarray_broadcast_data_free((NpyAuxData *) newdata);
            return NULL;
        }
    }
    /* 复制目标数据遍历信息 */
    if (d->decref_dst.func != NULL) {
        if (NPY_traverse_info_copy(&newdata->decref_dst, &d->decref_dst) < 0) {
            _subarray_broadcast_data_free((NpyAuxData *) newdata);
            return NULL;
        }
    }

    return (NpyAuxData *)newdata;
}

/* 处理从分层数组到分层数组的子数组广播的函数 */
static int
_strided_to_strided_subarray_broadcast(
        PyArrayMethod_Context *NPY_UNUSED(context), char *const *args,
        const npy_intp *dimensions, const npy_intp *strides,
        NpyAuxData *auxdata)
{
    npy_intp N = dimensions[0];     /* 获取第一维度的大小 */
    char *src = args[0], *dst = args[1];    /* 获取源和目标数据的指针 */
    npy_intp src_stride = strides[0], dst_stride = strides[1];   /* 获取源和目标数据的步长 */

    _subarray_broadcast_data *d = (_subarray_broadcast_data *)auxdata;   /* 将辅助数据转换为子数组广播数据结构 */
    npy_intp run, run_count = d->run_count;    /* 获取运行长度和运行数量 */
    npy_intp loop_index, offset, count;

    /* 
       以下是处理子数组广播的核心算法逻辑，具体实现依赖于实际的数据处理需求，
       包括从源到目标的数据传输和偏移量的应用。
    */
    // 获取源数组元素大小（以字节为单位）
    npy_intp src_subitemsize = d->wrapped.descriptors[0]->elsize;
    // 获取目标数组元素大小（以字节为单位）
    npy_intp dst_subitemsize = d->wrapped.descriptors[1]->elsize;
    
    // 创建包含源数组和目标数组的子步长数组
    npy_intp sub_strides[2] = {src_subitemsize, dst_subitemsize};
    
    // 当还有剩余元素未处理时执行循环
    while (N > 0) {
        // 初始化循环内部索引
        loop_index = 0;
        // 遍历偏移量运行列表中的每个运行
        for (run = 0; run < run_count; ++run) {
            // 获取当前运行的偏移量和计数
            offset = d->offsetruns[run].offset;
            count = d->offsetruns[run].count;
            // 计算目标数组的指针位置
            char *dst_ptr = dst + loop_index * dst_subitemsize;
            // 构建源数组和目标数组的子参数数组
            char *sub_args[2] = {src + offset, dst_ptr};
            // 如果偏移量不为-1，则调用包装函数处理子数组区域
            if (offset != -1) {
                // 调用封装函数，将其结果存储在目标数组中
                if (d->wrapped.func(&d->wrapped.context,
                                    sub_args, &count, sub_strides, d->wrapped.auxdata) < 0) {
                    // 如果调用返回负值，表示出错，返回-1
                    return -1;
                }
            } else {
                // 如果偏移量为-1，使用memset将目标数组区域填充为0
                memset(dst_ptr, 0, count * dst_subitemsize);
            }
            // 更新循环内部索引
            loop_index += count;
        }
    
        // 更新源数组和目标数组的全局指针位置
        src += src_stride;
        dst += dst_stride;
        // 减少剩余元素计数
        --N;
    }
    
    // 处理完所有元素后返回0
    return 0;
# 定义一个静态函数 `_strided_to_strided_subarray_broadcast_withrefs`，处理带有引用的广播子数组之间的转换
static int
_strided_to_strided_subarray_broadcast_withrefs(
        PyArrayMethod_Context *NPY_UNUSED(context), char *const *args,
        const npy_intp *dimensions, const npy_intp *strides,
        NpyAuxData *auxdata)
{
    # 获取第一个维度的大小
    npy_intp N = dimensions[0];
    # 获取输入和输出数组的起始位置
    char *src = args[0], *dst = args[1];
    # 获取输入和输出数组的步长
    npy_intp src_stride = strides[0], dst_stride = strides[1];

    # 从辅助数据中获取广播数据结构的指针
    _subarray_broadcast_data *d = (_subarray_broadcast_data *)auxdata;
    # 初始化循环索引、运行计数和其他变量
    npy_intp run, run_count = d->run_count;
    npy_intp loop_index, offset, count;

    # 获取源子项大小和目标子项大小
    npy_intp src_subitemsize = d->wrapped.descriptors[0]->elsize;
    npy_intp dst_subitemsize = d->wrapped.descriptors[1]->elsize;

    # 设置子数组的步长数组
    npy_intp sub_strides[2] = {src_subitemsize, dst_subitemsize};

    # 开始主循环，处理每个元素
    while (N > 0) {
        # 初始化循环索引
        loop_index = 0;
        # 遍历广播的运行
        for (run = 0; run < run_count; ++run) {
            # 获取偏移量和计数
            offset = d->offsetruns[run].offset;
            count = d->offsetruns[run].count;
            # 计算目标数组的指针位置
            char *dst_ptr = dst + loop_index * dst_subitemsize;
            # 设置子数组的参数数组
            char *sub_args[2] = {src + offset, dst_ptr};
            # 如果偏移量不为 -1，则调用封装的函数处理子数组数据
            if (offset != -1) {
                if (d->wrapped.func(&d->wrapped.context,
                        sub_args, &count, sub_strides, d->wrapped.auxdata) < 0) {
                    return -1;
                }
            }
            # 否则，如果存在减少引用的函数，则执行相应操作；否则，将目标数组置零
            else {
                if (d->decref_dst.func != NULL) {
                    if (d->decref_dst.func(NULL, d->decref_dst.descr,
                            dst_ptr, count, dst_subitemsize,
                            d->decref_dst.auxdata) < 0) {
                        return -1;
                    }
                }
                memset(dst_ptr, 0, count * dst_subitemsize);
            }
            # 更新循环索引
            loop_index += count;
        }

        # 如果存在减少源引用的函数，则执行相应操作
        if (d->decref_src.func != NULL) {
            if (d->decref_src.func(NULL, d->decref_src.descr,
                    src, d->src_N, src_subitemsize,
                    d->decref_src.auxdata) < 0) {
                return -1;
            }
        }

        # 更新输入和输出数组的位置
        src += src_stride;
        dst += dst_stride;
        # 更新剩余的元素数量
        --N;
    }
    # 返回成功状态
    return 0;
}
    # 分配内存以存储子数组广播数据，并进行空指针检查
    data = (_subarray_broadcast_data *)PyMem_Malloc(structsize);
    if (data == NULL) {
        PyErr_NoMemory();  # 如果分配失败，抛出内存错误异常
        return NPY_FAIL;   # 返回失败状态
    }
    # 设置数据结构中的释放函数和克隆函数
    data->base.free = &_subarray_broadcast_data_free;
    data->base.clone = &_subarray_broadcast_data_clone;
    # 设置源数组和目标数组的维度大小
    data->src_N = src_size;
    data->dst_N = dst_size;

    # 初始化需要递减引用计数的源数组和目标数组信息
    NPY_traverse_info_init(&data->decref_src);
    NPY_traverse_info_init(&data->decref_dst);

    """
     * move_references is set to 0, handled in the wrapping transfer fn,
     * src_stride and dst_stride are set to contiguous, as N will always
     * be 1 when it's called.
     """
    # 在包装的传输函数中处理 move_references 被设置为 0，设置 src_stride 和 dst_stride 为连续的，
    # 因为 N 会在调用时总是为 1。
    if (PyArray_GetDTypeTransferFunction(aligned,
                    src_dtype->elsize, dst_dtype->elsize,
                    src_dtype, dst_dtype,
                    0,
                    &data->wrapped,
                    out_flags) != NPY_SUCCEED) {
        NPY_AUXDATA_FREE((NpyAuxData *)data);
        return NPY_FAIL;
    }

    # 如果源对象需要 DECREF 引用计数递减
    if (move_references && PyDataType_REFCHK(src_dtype)) {
        if (PyArray_GetClearFunction(aligned,
                        src_dtype->elsize, src_dtype,
                        &data->decref_src, out_flags) < 0) {
            NPY_AUXDATA_FREE((NpyAuxData *)data);
            return NPY_FAIL;
        }
    }

    # 如果目标对象需要 DECREF 引用计数递减并将其设置为 NULL
    if (PyDataType_REFCHK(dst_dtype)) {
        if (PyArray_GetClearFunction(aligned,
                        dst_dtype->elsize, dst_dtype,
                        &data->decref_dst, out_flags) < 0) {
            NPY_AUXDATA_FREE((NpyAuxData *)data);
            return NPY_FAIL;
        }
    }

    # 计算广播的维度并设置偏移量运行
    _subarray_broadcast_offsetrun *offsetruns = data->offsetruns;
    ndim = (src_shape.len > dst_shape.len) ? src_shape.len : dst_shape.len;
    for (loop_index = 0; loop_index < dst_size; ++loop_index) {
        npy_intp src_factor = 1;

        dst_index = loop_index;
        src_index = 0;
        for (i = ndim-1; i >= 0; --i) {
            npy_intp coord = 0, shape;

            /* Get the dst coord of this index for dimension i */
            // 如果当前维度 i 在目标形状的有效范围内
            if (i >= ndim - dst_shape.len) {
                shape = dst_shape.ptr[i-(ndim-dst_shape.len)];
                coord = dst_index % shape;  // 计算目标索引在当前维度的坐标
                dst_index /= shape;  // 更新目标索引到下一个维度的坐标
            }

            /* Translate it into a src coord and update src_index */
            // 如果当前维度 i 在源形状的有效范围内
            if (i >= ndim - src_shape.len) {
                shape = src_shape.ptr[i-(ndim-src_shape.len)];
                if (shape == 1) {
                    coord = 0;  // 对于形状为1的维度，坐标为0
                }
                else {
                    if (coord < shape) {
                        src_index += src_factor*coord;  // 计算源索引
                        src_factor *= shape;  // 更新源索引的因子
                    }
                    else {
                        /* Out of bounds, flag with -1 */
                        src_index = -1;  // 如果坐标超出边界，则标记为 -1
                        break;
                    }
                }
            }
        }
        /* Set the offset */
        // 设置偏移量
        if (src_index == -1) {
            offsetruns[loop_index].offset = -1;  // 如果源索引为 -1，表示超出边界
        }
        else {
            offsetruns[loop_index].offset = src_index;  // 设置有效的源索引作为偏移量
        }
    }

    /* Run-length encode the result */
    // 对结果进行行程长度编码
    run = 0;
    run_size = 1;
    for (loop_index = 1; loop_index < dst_size; ++loop_index) {
        if (offsetruns[run].offset == -1) {
            /* Stop the run when there's a valid index again */
            // 当再次出现有效索引时停止当前运行
            if (offsetruns[loop_index].offset != -1) {
                offsetruns[run].count = run_size;
                run++;
                run_size = 1;
                offsetruns[run].offset = offsetruns[loop_index].offset;
            }
            else {
                run_size++;
            }
        }
        else {
            /* Stop the run when there's a valid index again */
            // 当再次出现有效索引时停止当前运行
            if (offsetruns[loop_index].offset !=
                            offsetruns[loop_index-1].offset + 1) {
                offsetruns[run].count = run_size;
                run++;
                run_size = 1;
                offsetruns[run].offset = offsetruns[loop_index].offset;
            }
            else {
                run_size++;
            }
        }
    }
    offsetruns[run].count = run_size;
    run++;
    data->run_count = run;

    /* Multiply all the offsets by the src item size */
    // 将所有偏移量乘以源数据类型的元素大小
    while (run--) {
        if (offsetruns[run].offset != -1) {
            offsetruns[run].offset *= src_dtype->elsize;
        }
    }

    if (data->decref_src.func == NULL &&
            data->decref_dst.func == NULL) {
        *out_stransfer = &_strided_to_strided_subarray_broadcast;
    }
    else {
        *out_stransfer = &_strided_to_strided_subarray_broadcast_withrefs;
    }
    *out_transferdata = (NpyAuxData *)data;

    return NPY_SUCCEED;
/*
 * Handles subarray transfer function selection based on source and destination
 * subarray shapes and sizes, considering alignment and memory strides.
 * At least one of the subarrays must be non-NULL to call this function.
 */
NPY_NO_EXPORT int
get_subarray_transfer_function(int aligned,
                            npy_intp src_stride, npy_intp dst_stride,
                            PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
                            int move_references,
                            PyArrayMethod_StridedLoop **out_stransfer,
                            NpyAuxData **out_transferdata,
                            NPY_ARRAYMETHOD_FLAGS *out_flags)
{
    PyArray_Dims src_shape = {NULL, -1}, dst_shape = {NULL, -1};
    npy_intp src_size = 1, dst_size = 1;

    /* Get the shapes and sizes of the source subarray */
    if (PyDataType_HASSUBARRAY(src_dtype)) {
        if (!(PyArray_IntpConverter(PyDataType_SUBARRAY(src_dtype)->shape,
                                            &src_shape))) {
            PyErr_SetString(PyExc_ValueError,
                    "invalid subarray shape");
            return NPY_FAIL;
        }
        src_size = PyArray_MultiplyList(src_shape.ptr, src_shape.len);
        src_dtype = PyDataType_SUBARRAY(src_dtype)->base;
    }

    /* Get the shapes and sizes of the destination subarray */
    if (PyDataType_HASSUBARRAY(dst_dtype)) {
        if (!(PyArray_IntpConverter(PyDataType_SUBARRAY(dst_dtype)->shape,
                                            &dst_shape))) {
            npy_free_cache_dim_obj(src_shape);
            PyErr_SetString(PyExc_ValueError,
                    "invalid subarray shape");
            return NPY_FAIL;
        }
        dst_size = PyArray_MultiplyList(dst_shape.ptr, dst_shape.len);
        dst_dtype = PyDataType_SUBARRAY(dst_dtype)->base;
    }

    /*
     * If the source and destination sizes are both 1 or their shapes are
     * identical, optimize for a direct transfer.
     */
    if ((dst_size == 1 && src_size == 1) || (
            src_shape.len == dst_shape.len && PyArray_CompareLists(
                    src_shape.ptr, dst_shape.ptr, src_shape.len))) {

        npy_free_cache_dim_obj(src_shape);
        npy_free_cache_dim_obj(dst_shape);

        return get_n_to_n_transfer_function(aligned,
                        src_stride, dst_stride,
                        src_dtype, dst_dtype,
                        move_references,
                        src_size,
                        out_stransfer, out_transferdata,
                        out_flags);
    }

    /*
     * If the source size is 1, broadcast the source value to all destination
     * values.
     */
    else if (src_size == 1) {
        npy_free_cache_dim_obj(src_shape);
        npy_free_cache_dim_obj(dst_shape);

        return get_one_to_n_transfer_function(aligned,
                src_stride, dst_stride,
                src_dtype, dst_dtype,
                move_references,
                dst_size,
                out_stransfer, out_transferdata,
                out_flags);
    }

    /*
     * Handle the transfer of subarrays with broadcasting, truncating, and
     * zero-padding as necessary.
     */
    // 如果不是最简形式，调用函数计算子数组的广播传输函数
    else {
        int ret = get_subarray_broadcast_transfer_function(aligned,
                        src_stride, dst_stride,
                        src_dtype, dst_dtype,
                        src_size, dst_size,
                        src_shape, dst_shape,
                        move_references,
                        out_stransfer, out_transferdata,
                        out_flags);

        // 释放缓存中的源形状对象
        npy_free_cache_dim_obj(src_shape);
        // 释放缓存中的目标形状对象
        npy_free_cache_dim_obj(dst_shape);
        // 返回计算结果
        return ret;
    }
}

/**************************** COPY FIELDS *******************************/

/* 定义单个字段传输的结构体 */
typedef struct {
    npy_intp src_offset, dst_offset;    // 源偏移量和目标偏移量
    NPY_cast_info info;                 // 转换信息
} _single_field_transfer;

/* 定义字段传输数据的结构体 */
typedef struct {
    NpyAuxData base;                    // 基础辅助数据
    npy_intp field_count;               // 字段数量
    NPY_traverse_info decref_src;       // 引用计数信息
    _single_field_transfer fields[];    // 单个字段传输结构体数组（柔性数组成员）
} _field_transfer_data;


/* 释放传输数据的函数 */
static void _field_transfer_data_free(NpyAuxData *data)
{
    _field_transfer_data *d = (_field_transfer_data *)data;
    NPY_traverse_info_xfree(&d->decref_src);  // 释放引用计数信息

    for (npy_intp i = 0; i < d->field_count; ++i) {
        NPY_cast_info_xfree(&d->fields[i].info);  // 释放每个字段的转换信息
    }
    PyMem_Free(d);  // 释放整个数据结构内存
}

/* 复制传输数据的函数 */
static NpyAuxData *_field_transfer_data_clone(NpyAuxData *data)
{
    _field_transfer_data *d = (_field_transfer_data *)data;

    npy_intp field_count = d->field_count;
    npy_intp structsize = sizeof(_field_transfer_data) +
                    field_count * sizeof(_single_field_transfer);

    /* 分配内存并填充数据 */
    _field_transfer_data *newdata = PyMem_Malloc(structsize);
    if (newdata == NULL) {
        return NULL;
    }
    newdata->base = d->base;  // 复制基础数据
    newdata->field_count = 0;
    if (NPY_traverse_info_copy(&newdata->decref_src, &d->decref_src) < 0) {
        PyMem_Free(newdata);
        return NULL;
    }

    /* 复制所有字段的传输数据 */
    for (npy_intp i = 0; i < field_count; ++i) {
        if (NPY_cast_info_copy(&newdata->fields[i].info, &d->fields[i].info) < 0) {
            NPY_AUXDATA_FREE((NpyAuxData *)newdata);
            return NULL;
        }
        newdata->fields[i].src_offset = d->fields[i].src_offset;  // 复制源偏移量
        newdata->fields[i].dst_offset = d->fields[i].dst_offset;  // 复制目标偏移量
        newdata->field_count++;
    }

    return (NpyAuxData *)newdata;
}


static int
_strided_to_strided_field_transfer(
        PyArrayMethod_Context *NPY_UNUSED(context), char *const *args,
        const npy_intp *dimensions, const npy_intp *strides,
        NpyAuxData *auxdata)
{
    npy_intp N = dimensions[0];
    char *src = args[0], *dst = args[1];
    npy_intp src_stride = strides[0], dst_stride = strides[1];

    _field_transfer_data *d = (_field_transfer_data *)auxdata;
    npy_intp i, field_count = d->field_count;
    const npy_intp blocksize = NPY_LOWLEVEL_BUFFER_BLOCKSIZE;

    /* 按块进行传输 */
    # 进入无限循环，执行以下操作直到条件不再满足
    for (;;) {
        # 如果 N 大于 blocksize，执行以下操作
        if (N > blocksize) {
            # 遍历所有字段
            for (i = 0; i < field_count; ++i) {
                # 获取当前字段的信息
                _single_field_transfer field = d->fields[i];
                # 准备参数数组，指向源和目标内存中字段的位置
                char *fargs[2] = {src + field.src_offset, dst + field.dst_offset};
                # 调用字段信息中指定的函数处理数据传输
                if (field.info.func(&field.info.context,
                        fargs, &blocksize, strides, field.info.auxdata) < 0) {
                    # 如果函数调用失败，返回错误
                    return -1;
                }
            }
            # 如果存在源数据减引用函数，并调用成功
            if (d->decref_src.func != NULL && d->decref_src.func(
                    NULL, d->decref_src.descr, src, blocksize, src_stride,
                    d->decref_src.auxdata) < 0) {
                # 如果调用失败，返回错误
                return -1;
            }
            # 减少剩余未处理的数据块大小 N
            N -= NPY_LOWLEVEL_BUFFER_BLOCKSIZE;
            # 调整源和目标指针位置，以处理下一个数据块
            src += NPY_LOWLEVEL_BUFFER_BLOCKSIZE * src_stride;
            dst += NPY_LOWLEVEL_BUFFER_BLOCKSIZE * dst_stride;
        }
        # 如果 N 小于或等于 blocksize，执行以下操作
        else {
            # 遍历所有字段
            for (i = 0; i < field_count; ++i) {
                # 获取当前字段的信息
                _single_field_transfer field = d->fields[i];
                # 准备参数数组，指向源和目标内存中字段的位置
                char *fargs[2] = {src + field.src_offset, dst + field.dst_offset};
                # 调用字段信息中指定的函数处理数据传输
                if (field.info.func(&field.info.context,
                        fargs, &N, strides, field.info.auxdata) < 0) {
                    # 如果函数调用失败，返回错误
                    return -1;
                }
            }
            # 如果存在源数据减引用函数，并调用成功
            if (d->decref_src.func != NULL && d->decref_src.func(
                    NULL, d->decref_src.descr, src, blocksize, src_stride,
                    d->decref_src.auxdata) < 0) {
                # 如果调用失败，返回错误
                return -1;
            }
            # 处理完所有数据后，返回成功
            return 0;
        }
    }
/*
 * 处理字段传输。要调用此函数，至少一个dtype必须具有字段。不处理对象<->结构的转换
 */
NPY_NO_EXPORT int
get_fields_transfer_function(int NPY_UNUSED(aligned),
                            npy_intp src_stride, npy_intp dst_stride,
                            PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
                            int move_references,
                            PyArrayMethod_StridedLoop **out_stransfer,
                            NpyAuxData **out_transferdata,
                            NPY_ARRAYMETHOD_FLAGS *out_flags)
{
    // 声明变量
    PyObject *key, *tup, *title;
    // 声明结构字段的数据类型变量
    PyArray_Descr *src_fld_dtype, *dst_fld_dtype;
    // 循环变量
    npy_int i;
    // 结构体大小
    size_t structsize;
    // 字段计数
    Py_ssize_t field_count;
    // 源偏移量、目标偏移量
    int src_offset, dst_offset;
    // 字段传输数据的结构体
    _field_transfer_data *data;

    /*
     * 有三种情况需要处理：1. src 是非结构化的，
     * 2. dst 是非结构化的，或者 3. 两者都是结构化的。
     */

    /* 1. src 是非结构化的。将 src 的值复制到 dst 的所有字段中 */
    // 检查源数据类型是否有字段，如果没有则处理单字段转移情况
    if (!PyDataType_HASFIELDS(src_dtype)) {
        // 获取目标数据类型的字段数量
        field_count = PyTuple_GET_SIZE(PyDataType_NAMES(dst_dtype));

        /* 分配字段传输数据结构并填充 */
        // 计算数据结构的大小，包括字段传输数据和单字段传输数据
        structsize = sizeof(_field_transfer_data) +
                        (field_count + 1) * sizeof(_single_field_transfer);
        // 分配内存空间
        data = PyMem_Malloc(structsize);
        if (data == NULL) {
            // 分配内存失败时设置内存错误并返回失败状态
            PyErr_NoMemory();
            return NPY_FAIL;
        }
        // 设置数据结构的释放函数和克隆函数
        data->base.free = &_field_transfer_data_free;
        data->base.clone = &_field_transfer_data_clone;
        // 初始化字段计数器
        data->field_count = 0;
        // 初始化源数据类型的减引用信息
        NPY_traverse_info_init(&data->decref_src);

        // 设置输出标志为最小的数组方法标志
        *out_flags = PyArrayMethod_MINIMAL_FLAGS;
        // 遍历目标数据类型的字段
        for (i = 0; i < field_count; ++i) {
            // 获取字段名
            key = PyTuple_GET_ITEM(PyDataType_NAMES(dst_dtype), i);
            // 从目标数据类型的字段字典中获取字段元组
            tup = PyDict_GetItem(PyDataType_FIELDS(dst_dtype), key);
            // 解析字段元组，获取目标字段的数据类型、偏移量和标题
            if (!PyArg_ParseTuple(tup, "Oi|O", &dst_fld_dtype,
                                                    &dst_offset, &title)) {
                // 解析失败时释放内存并返回失败状态
                PyMem_Free(data);
                return NPY_FAIL;
            }
            // 获取数据类型传输函数和字段标志
            NPY_ARRAYMETHOD_FLAGS field_flags;
            if (PyArray_GetDTypeTransferFunction(0,
                                    src_stride, dst_stride,
                                    src_dtype, dst_fld_dtype,
                                    0,
                                    &data->fields[i].info,
                                    &field_flags) != NPY_SUCCEED) {
                // 获取传输函数失败时释放辅助数据并返回失败状态
                NPY_AUXDATA_FREE((NpyAuxData *)data);
                return NPY_FAIL;
            }
            // 合并字段标志到输出标志中
            *out_flags = PyArrayMethod_COMBINED_FLAGS(*out_flags, field_flags);
            // 设置源和目标的偏移量，并增加字段计数
            data->fields[i].src_offset = 0;
            data->fields[i].dst_offset = dst_offset;
            data->field_count++;
        }

        /*
         * 如果需要在源数据中进行减引用，添加清除函数。
         */
        // 如果需要移动引用并且源数据类型需要减引用检查
        if (move_references && PyDataType_REFCHK(src_dtype)) {
            // 获取清除函数和清除标志
            NPY_ARRAYMETHOD_FLAGS clear_flags;
            if (PyArray_GetClearFunction(
                    0, src_stride, src_dtype, &data->decref_src,
                    &clear_flags) < 0) {
                // 获取清除函数失败时释放辅助数据并返回失败状态
                NPY_AUXDATA_FREE((NpyAuxData *)data);
                return NPY_FAIL;
            }
            // 合并清除标志到输出标志中
            *out_flags = PyArrayMethod_COMBINED_FLAGS(*out_flags, clear_flags);
        }

        // 设置输出传输函数和传输数据
        *out_stransfer = &_strided_to_strided_field_transfer;
        *out_transferdata = (NpyAuxData *)data;

        // 返回成功状态
        return NPY_SUCCEED;
    }

    /* 2. dst is non-structured. Allow transfer from single-field src to dst */
    
    # 如果目标数据类型不是结构化的，则执行以下操作
    if (!PyDataType_HASFIELDS(dst_dtype)):

        # 如果源数据类型的字段数量不为1，则抛出值错误异常
        if (PyTuple_GET_SIZE(PyDataType_NAMES(src_dtype)) != 1):
            PyErr_SetString(PyExc_ValueError,
                    "Can't cast from structure to non-structure, except if the "
                    "structure only has a single field.")
            return NPY_FAIL

        /* Allocate the field-data structure and populate it */
        # 计算结构体大小并分配内存，包括基本数据结构和单个字段传输信息
        structsize = sizeof(_field_transfer_data) +
                        1 * sizeof(_single_field_transfer)
        # 分配内存并检查是否成功
        data = PyMem_Malloc(structsize)
        if (data == NULL):
            PyErr_NoMemory()
            return NPY_FAIL
        # 设置基础数据结构的释放和克隆函数
        data->base.free = &_field_transfer_data_free
        data->base.clone = &_field_transfer_data_clone
        # 初始化减少源引用的信息
        NPY_traverse_info_init(&data->decref_src)

        # 获取源数据类型的字段名元组的第一个元素作为键
        key = PyTuple_GET_ITEM(PyDataType_NAMES(src_dtype), 0)
        # 从源数据类型的字段字典中获取与键对应的值
        tup = PyDict_GetItem(PyDataType_FIELDS(src_dtype), key)
        # 解析字段元组，获取源字段数据类型、偏移量和标题信息
        if (!PyArg_ParseTuple(tup, "Oi|O",
                              &src_fld_dtype, &src_offset, &title)):
            # 解析失败时释放分配的内存并返回失败状态
            PyMem_Free(data)
            return NPY_FAIL

        # 获取数据类型传输函数，并配置字段传输信息
        if (PyArray_GetDTypeTransferFunction(0,
                                             src_stride, dst_stride,
                                             src_fld_dtype, dst_dtype,
                                             move_references,
                                             &data->fields[0].info,
                                             out_flags) != NPY_SUCCEED):
            # 获取传输函数失败时释放分配的内存并返回失败状态
            PyMem_Free(data)
            return NPY_FAIL
        # 设置源偏移和目标偏移，并更新字段计数
        data->fields[0].src_offset = src_offset
        data->fields[0].dst_offset = 0
        data->field_count = 1

        # 设置输出参数为字段到字段的传输函数和传输数据
        *out_stransfer = &_strided_to_strided_field_transfer
        *out_transferdata = (NpyAuxData *)data

        # 返回成功状态
        return NPY_SUCCEED

    /* 3. Otherwise both src and dst are structured arrays */
    # 否则，如果源和目标数据类型都是结构化数组

    # 获取目标数据类型的字段数量
    field_count = PyTuple_GET_SIZE(PyDataType_NAMES(dst_dtype))

    # 检查源数据类型和目标数据类型的字段数量是否相同
    if (PyTuple_GET_SIZE(PyDataType_NAMES(src_dtype)) != field_count):
        # 如果字段数量不同，抛出值错误异常并返回失败状态
        PyErr_SetString(PyExc_ValueError, "structures must have the same size")
        return NPY_FAIL

    /* Allocate the field-data structure and populate it */
    # 计算结构体大小并分配内存，包括基本数据结构和多个字段传输信息
    structsize = sizeof(_field_transfer_data) +
                    field_count * sizeof(_single_field_transfer)
    # 分配内存并检查是否成功
    data = PyMem_Malloc(structsize)
    if (data == NULL):
        PyErr_NoMemory()
        return NPY_FAIL
    # 设置基础数据结构的释放和克隆函数
    data->base.free = &_field_transfer_data_free
    data->base.clone = &_field_transfer_data_clone
    data->field_count = 0
    # 初始化减少源引用的信息
    NPY_traverse_info_init(&data->decref_src)

    # 设置输出标志为最小化的数组方法标志
    *out_flags = PyArrayMethod_MINIMAL_FLAGS
    /* set up the transfer function for each field */
    // 遍历字段数目次数的循环，用于处理每个字段的数据类型转换和传输设置
    for (i = 0; i < field_count; ++i) {
        // 获取目标数据类型的第i个字段名
        key = PyTuple_GET_ITEM(PyDataType_NAMES(dst_dtype), i);
        // 获取目标数据类型的第i个字段的元组描述
        tup = PyDict_GetItem(PyDataType_FIELDS(dst_dtype), key);
        // 解析元组描述，获取目标字段的数据类型、偏移量和可选标题
        if (!PyArg_ParseTuple(tup, "Oi|O", &dst_fld_dtype,
                                                &dst_offset, &title)) {
            // 解析失败时释放已分配的辅助数据并返回失败标志
            NPY_AUXDATA_FREE((NpyAuxData *)data);
            return NPY_FAIL;
        }
        // 获取源数据类型的第i个字段名
        key = PyTuple_GET_ITEM(PyDataType_NAMES(src_dtype), i);
        // 获取源数据类型的第i个字段的元组描述
        tup = PyDict_GetItem(PyDataType_FIELDS(src_dtype), key);
        // 解析元组描述，获取源字段的数据类型、偏移量和可选标题
        if (!PyArg_ParseTuple(tup, "Oi|O", &src_fld_dtype,
                                                &src_offset, &title)) {
            // 解析失败时释放已分配的辅助数据并返回失败标志
            NPY_AUXDATA_FREE((NpyAuxData *)data);
            return NPY_FAIL;
        }

        // 获取字段转换函数的传输标志和信息，进行数据类型转换设置
        NPY_ARRAYMETHOD_FLAGS field_flags;
        if (PyArray_GetDTypeTransferFunction(0,
                                             src_stride, dst_stride,
                                             src_fld_dtype, dst_fld_dtype,
                                             move_references,
                                             &data->fields[i].info,
                                             &field_flags) != NPY_SUCCEED) {
            // 如果获取转换函数失败，释放辅助数据并返回失败标志
            NPY_AUXDATA_FREE((NpyAuxData *)data);
            return NPY_FAIL;
        }
        // 合并当前字段的传输标志到输出标志中
        *out_flags = PyArrayMethod_COMBINED_FLAGS(*out_flags, field_flags);
        // 设置数据结构中当前字段的源偏移量和目标偏移量
        data->fields[i].src_offset = src_offset;
        data->fields[i].dst_offset = dst_offset;
        // 增加处理过的字段计数
        data->field_count++;
    }

    // 将输出的传输函数设置为 strided-to-strided 字段传输函数
    *out_stransfer = &_strided_to_strided_field_transfer;
    // 将输出的辅助数据设置为当前数据结构的辅助数据
    *out_transferdata = (NpyAuxData *)data;

    // 返回成功标志
    return NPY_SUCCEED;
/************************* MASKED TRANSFER WRAPPER *************************/

/* 结构体定义：用于封装的转移数据 */
typedef struct {
    NpyAuxData base;                    // 基础结构体
    NPY_cast_info wrapped;              // 被封装的转移函数信息（可能直接存储）
    NPY_traverse_info decref_src;       // 源对象的减引用函数（如果需要）
} _masked_wrapper_transfer_data;

/* 转移数据的释放函数 */
static void
_masked_wrapper_transfer_data_free(NpyAuxData *data)
{
    _masked_wrapper_transfer_data *d = (_masked_wrapper_transfer_data *)data;
    NPY_cast_info_xfree(&d->wrapped);   // 释放包装的转移函数信息
    NPY_traverse_info_xfree(&d->decref_src);  // 释放源对象减引用函数信息
    PyMem_Free(data);                   // 释放数据内存
}

/* 转移数据的复制函数 */
static NpyAuxData *
_masked_wrapper_transfer_data_clone(NpyAuxData *data)
{
    _masked_wrapper_transfer_data *d = (_masked_wrapper_transfer_data *)data;
    _masked_wrapper_transfer_data *newdata;

    /* 分配新的数据并填充 */
    newdata = PyMem_Malloc(sizeof(*newdata));
    if (newdata == NULL) {
        return NULL;
    }
    newdata->base = d->base;

    if (NPY_cast_info_copy(&newdata->wrapped, &d->wrapped) < 0) {
        PyMem_Free(newdata);
        return NULL;
    }
    if (d->decref_src.func != NULL) {
        if (NPY_traverse_info_copy(&newdata->decref_src, &d->decref_src) < 0) {
            NPY_AUXDATA_FREE((NpyAuxData *)newdata);
            return NULL;
        }
    }

    return (NpyAuxData *)newdata;
}

/* 处理带有掩码的封装器的清除函数 */
static int
_strided_masked_wrapper_clear_function(
        PyArrayMethod_Context *NPY_UNUSED(context), char *const *args,
        const npy_intp *dimensions, const npy_intp *strides,
        npy_bool *mask, npy_intp mask_stride,
        NpyAuxData *auxdata)
{
    npy_intp N = dimensions[0];         // 数组维度
    char *src = args[0], *dst = args[1]; // 源和目标数组的起始地址
    npy_intp src_stride = strides[0], dst_stride = strides[1]; // 源和目标数组的步长

    _masked_wrapper_transfer_data *d = (_masked_wrapper_transfer_data *)auxdata; // 转移数据结构体指针
    npy_intp subloopsize;

    while (N > 0) {
        /* 跳过掩码值，仍然调用减引用以移动引用 */
        mask = (npy_bool*)npy_memchr((char *)mask, 0, mask_stride, N,
                                     &subloopsize, 1);
        if (d->decref_src.func(NULL, d->decref_src.descr,
                src, subloopsize, src_stride, d->decref_src.auxdata) < 0) {
            return -1;
        }
        dst += subloopsize * dst_stride;
        src += subloopsize * src_stride;
        N -= subloopsize;
        if (N <= 0) {
            break;
        }

        /* 处理非掩码值 */
        mask = (npy_bool*)npy_memchr((char *)mask, 0, mask_stride, N,
                                     &subloopsize, 0);
        char *wrapped_args[2] = {src, dst};
        if (d->wrapped.func(&d->wrapped.context,
                wrapped_args, &subloopsize, strides, d->wrapped.auxdata) < 0) {
            return -1;
        }
        dst += subloopsize * dst_stride;
        src += subloopsize * src_stride;
        N -= subloopsize;
    }
    return 0;
}
/*
 * 实现一个带有遮罩处理的数据传输函数
 */
_strided_masked_wrapper_transfer_function(
        PyArrayMethod_Context *NPY_UNUSED(context), char *const *args,
        const npy_intp *dimensions, const npy_intp *strides,
        npy_bool *mask, npy_intp mask_stride,
        NpyAuxData *auxdata)
{
    // 获取数组的长度
    npy_intp N = dimensions[0];
    // 源数组和目标数组的起始位置
    char *src = args[0], *dst = args[1];
    // 源数组和目标数组的步长
    npy_intp src_stride = strides[0], dst_stride = strides[1];

    // 获取传输数据的包装结构体
    _masked_wrapper_transfer_data *d = (_masked_wrapper_transfer_data *)auxdata;
    // 子循环的大小
    npy_intp subloopsize;

    // 循环直到处理完所有元素
    while (N > 0) {
        /* 跳过遮罩值为真的元素 */
        mask = (npy_bool*)npy_memchr((char *)mask, 0, mask_stride, N,
                                     &subloopsize, 1);
        // 更新目标数组的位置
        dst += subloopsize * dst_stride;
        // 更新源数组的位置
        src += subloopsize * src_stride;
        // 更新剩余处理的元素个数
        N -= subloopsize;
        // 如果剩余元素个数小于等于0，则退出循环
        if (N <= 0) {
            break;
        }

        /* 处理遮罩值为假的元素 */
        mask = (npy_bool*)npy_memchr((char *)mask, 0, mask_stride, N,
                                     &subloopsize, 0);
        // 创建包含源数组和目标数组的参数数组
        char *wrapped_args[2] = {src, dst};
        // 调用包装函数处理数据
        if (d->wrapped.func(&d->wrapped.context,
                wrapped_args, &subloopsize, strides, d->wrapped.auxdata) < 0) {
            // 如果处理函数返回小于0的值，表示出错，直接返回
            return -1;
        }
        // 更新目标数组的位置
        dst += subloopsize * dst_stride;
        // 更新源数组的位置
        src += subloopsize * src_stride;
        // 更新剩余处理的元素个数
        N -= subloopsize;
    }
    // 处理完成，返回0
    return 0;
}


/*
 * 空操作函数（目前仅用于清理目的）
 */
static int
_cast_no_op(
        PyArrayMethod_Context *NPY_UNUSED(context),
        char *const *NPY_UNUSED(args), const npy_intp *NPY_UNUSED(dimensions),
        const npy_intp *NPY_UNUSED(strides), NpyAuxData *NPY_UNUSED(auxdata))
{
    /* 什么也不做 */
    return 0;
}


/*
 * ********************* 通用多步转换 ************************
 *
 * 当解析描述符需要多个转换步骤时，使用的新的通用多步转换函数。
 */

typedef struct {
    NpyAuxData base;
    /* 主转换的信息 */
    NPY_cast_info main;
    /* 输入准备转换的信息 */
    NPY_cast_info from;
    /* 输出最终化转换的信息 */
    NPY_cast_info to;
    // 源缓冲区和目标缓冲区
    char *from_buffer;
    char *to_buffer;
} _multistep_castdata;


/* 零填充数据复制函数 */
static void
_multistep_cast_auxdata_free(NpyAuxData *auxdata)
{
    // 获取多步转换数据结构体
    _multistep_castdata *data = (_multistep_castdata *)auxdata;
    // 释放主转换信息
    NPY_cast_info_xfree(&data->main);
    // 如果存在输入准备转换信息，则释放
    if (data->from.func != NULL) {
        NPY_cast_info_xfree(&data->from);
    }
    // 如果存在输出最终化转换信息，则释放
    if (data->to.func != NULL) {
        NPY_cast_info_xfree(&data->to);
    }
    // 释放内存
    PyMem_Free(data);
}


/*
 * 克隆多步转换辅助数据
 */
static NpyAuxData *
_multistep_cast_auxdata_clone(NpyAuxData *auxdata_old);


/*
 * 克隆多步转换辅助数据（整数版）
 */
static NpyAuxData *
_multistep_cast_auxdata_clone_int(_multistep_castdata *castdata, int move_info)
{
    // 将结构体大小向上舍入到16字节边界，以适应缓冲区
    Py_ssize_t datasize = (sizeof(_multistep_castdata) + 15) & ~0xf;

    // 设置源缓冲区的偏移量
    Py_ssize_t from_buffer_offset = datasize;
    /* 检查是否存在源数据转换函数 */
    if (castdata->from.func != NULL) {
        /* 计算源数据项大小 */
        Py_ssize_t src_itemsize = castdata->main.context.descriptors[0]->elsize;
        /* 增加数据大小以适应源数据项大小的缓冲块 */
        datasize += NPY_LOWLEVEL_BUFFER_BLOCKSIZE * src_itemsize;
        /* 将数据大小调整为16字节的倍数 */
        datasize = (datasize + 15) & ~0xf;
    }
    /* 记录目标缓冲区的偏移量 */
    Py_ssize_t to_buffer_offset = datasize;
    /* 检查是否存在目标数据转换函数 */
    if (castdata->to.func != NULL) {
        /* 计算目标数据项大小 */
        Py_ssize_t dst_itemsize = castdata->main.context.descriptors[1]->elsize;
        /* 增加数据大小以适应目标数据项大小的缓冲块 */
        datasize += NPY_LOWLEVEL_BUFFER_BLOCKSIZE * dst_itemsize;
    }

    /* 分配数据大小的内存空间 */
    char *char_data = PyMem_Malloc(datasize);
    /* 内存分配失败时返回空指针 */
    if (char_data == NULL) {
        return NULL;
    }

    /* 将分配的内存解释为_multistep_castdata结构 */
    _multistep_castdata *newdata = (_multistep_castdata *)char_data;

    /* 设置基本信息的释放和克隆函数 */
    newdata->base.free = &_multistep_cast_auxdata_free;
    newdata->base.clone = &_multistep_cast_auxdata_clone;
    
    /* 设置源和目标缓冲区的起始位置 */
    newdata->from_buffer = char_data + from_buffer_offset;
    newdata->to_buffer = char_data + to_buffer_offset;

    /* 初始化函数指针为NULL，以确保在出错时不进行清理操作 */
    newdata->from.func = NULL;
    newdata->to.func = NULL;

    /* 如果存在移动信息，则移动主转换信息 */
    if (move_info) {
        NPY_cast_info_move(&newdata->main, &castdata->main);
    }
    /* 否则，复制主转换信息 */
    else if (NPY_cast_info_copy(&newdata->main, &castdata->main) < 0) {
        /* 复制失败时跳转到错误处理 */
        goto fail;
    }

    /* 如果存在源数据转换函数 */
    if (castdata->from.func != NULL) {
        /* 如果存在移动信息，则移动源转换信息 */
        if (move_info) {
            NPY_cast_info_move(&newdata->from, &castdata->from);
        }
        /* 否则，复制源转换信息 */
        else if (NPY_cast_info_copy(&newdata->from, &castdata->from) < 0) {
            /* 复制失败时跳转到错误处理 */
            goto fail;
        }

        /* 如果需要初始化源数据缓冲区，则将其清零 */
        if (PyDataType_FLAGCHK(newdata->main.descriptors[0], NPY_NEEDS_INIT)) {
            memset(newdata->from_buffer, 0, to_buffer_offset - from_buffer_offset);
        }
    }
    
    /* 如果存在目标数据转换函数 */
    if (castdata->to.func != NULL) {
        /* 如果存在移动信息，则移动目标转换信息 */
        if (move_info) {
            NPY_cast_info_move(&newdata->to, &castdata->to);
        }
        /* 否则，复制目标转换信息 */
        else if (NPY_cast_info_copy(&newdata->to, &castdata->to) < 0) {
            /* 复制失败时跳转到错误处理 */
            goto fail;
        }

        /* 如果需要初始化目标数据缓冲区，则将其清零 */
        if (PyDataType_FLAGCHK(newdata->main.descriptors[1], NPY_NEEDS_INIT)) {
            memset(newdata->to_buffer, 0, datasize - to_buffer_offset);
        }
    }

    /* 返回转换后的新数据 */
    return (NpyAuxData *)newdata;

  fail:
    /* 出错时释放分配的内存空间 */
    NPY_AUXDATA_FREE((NpyAuxData *)newdata);
    /* 返回空指针 */
    return NULL;
}

/*
 * Clone auxiliary data for multistep casting.
 * This function creates a new copy of the given auxiliary data object.
 */
static NpyAuxData *
_multistep_cast_auxdata_clone(NpyAuxData *auxdata_old)
{
    // Delegate cloning operation to an internal function
    return _multistep_cast_auxdata_clone_int(
            (_multistep_castdata *)auxdata_old, 0);
}

/*
 * Perform a strided to strided multistep cast operation.
 * This function converts data from source to destination using auxiliary data.
 * It handles data in chunks of NPY_LOWLEVEL_BUFFER_BLOCKSIZE size until completed.
 */
static int
_strided_to_strided_multistep_cast(
        /* The context is always stored explicitly in auxdata */
        PyArrayMethod_Context *NPY_UNUSED(context), char *const *args,
        const npy_intp *dimensions, const npy_intp *strides,
        NpyAuxData *auxdata)
{
    npy_intp N = dimensions[0]; // Number of elements in the first dimension
    char *src = args[0], *dst = args[1]; // Source and destination pointers
    _multistep_castdata *castdata = (_multistep_castdata *)auxdata; // Cast data from auxiliary data
    npy_intp src_stride = strides[0], dst_stride = strides[1]; // Source and destination strides

    char *main_src, *main_dst; // Pointers for main source and destination
    npy_intp main_src_stride, main_dst_stride; // Strides for main source and destination

    npy_intp block_size = NPY_LOWLEVEL_BUFFER_BLOCKSIZE; // Block size for processing

    // Process data in chunks until all elements are converted
    while (N > 0) {
        if (block_size > N) {
            block_size = N; // Adjust block size if it exceeds remaining elements
        }

        // Determine main source and its stride based on cast function availability
        if (castdata->from.func != NULL) {
            npy_intp out_stride = castdata->from.descriptors[1]->elsize; // Output stride
            char *const data[2] = {src, castdata->from_buffer}; // Data pointers for cast operation
            npy_intp strides[2] = {src_stride, out_stride}; // Strides for cast operation
            if (castdata->from.func(&castdata->from.context,
                    data, &block_size,
                    strides,
                    castdata->from.auxdata) != 0) {
                /* TODO: Internal buffer may require cleanup on error. */
                return -1; // Return on error during casting
            }
            main_src = castdata->from_buffer; // Update main source pointer
            main_src_stride = out_stride; // Update main source stride
        }
        else {
            main_src = src; // Use original source if no custom cast function
            main_src_stride = src_stride; // Use original source stride
        }

        // Determine main destination based on cast function availability
        if (castdata->to.func != NULL) {
            main_dst = castdata->to_buffer; // Use temporary buffer for destination
            main_dst_stride = castdata->main.descriptors[1]->elsize; // Main destination stride
        }
        else {
            main_dst = dst; // Use original destination if no custom cast function
            main_dst_stride = dst_stride; // Use original destination stride
        }

        // Perform main casting operation using main source and destination
        char *const data[2] = {main_src, main_dst}; // Data pointers for main casting
        npy_intp strides[2] = {main_src_stride, main_dst_stride}; // Strides for main casting
        if (castdata->main.func(&castdata->main.context,
                data, &block_size,
                strides,
                castdata->main.auxdata) != 0) {
            /* TODO: Internal buffer may require cleanup on error. */
            return -1; // Return on error during main casting
        }

        // Perform final conversion to destination if needed
        if (castdata->to.func != NULL) {
            char *const data[2] = {main_dst, dst}; // Data pointers for final conversion
            npy_intp strides[2] = {main_dst_stride, dst_stride}; // Strides for final conversion
            if (castdata->to.func(&castdata->to.context,
                    data, &block_size,
                    strides,
                    castdata->to.auxdata) != 0) {
                return -1; // Return on error during final conversion
            }
        }

        N -= block_size; // Decrement remaining elements count
        src += block_size * src_stride; // Move source pointer forward
        dst += block_size * dst_stride; // Move destination pointer forward
    }
    return 0; // Successful completion of the cast operation
}

/*
 * Initialize most of a cast-info structure, this step does not fetch the
 * transferfunction and transferdata.
 */
static inline int
/*
 * 初始化类型转换信息结构体。
 * 
 * Parameters:
 * - cast_info: 指向类型转换信息的指针
 * - casting: 指向类型转换方式的指针
 * - view_offset: 视图偏移的指针
 * - src_dtype: 源数据类型描述符
 * - dst_dtype: 目标数据类型描述符
 * - main_step: 主步骤标志，用于标识是否为主要转换步骤
 * 
 * Returns:
 * - 成功返回 0，失败返回 -1
 */
init_cast_info(
        NPY_cast_info *cast_info, NPY_CASTING *casting, npy_intp *view_offset,
        PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype, int main_step)
{
    // 获取类型转换实现的 Python 方法对象
    PyObject *meth = PyArray_GetCastingImpl(
            NPY_DTYPE(src_dtype), NPY_DTYPE(dst_dtype));
    if (meth == NULL) {
        return -1;
    }
    if (meth == Py_None) {
        // 处理获取到的 None 对象，释放引用并设置类型错误异常
        Py_DECREF(Py_None);
        PyErr_Format(PyExc_TypeError,
                "Cannot cast data from %S to %S.", src_dtype, dst_dtype);
        return -1;
    }
    
    /* Initialize the context and related data */
    // 初始化类型转换信息结构体
    NPY_cast_info_init(cast_info);
    cast_info->auxdata = NULL;

    // 设置调用者和方法
    cast_info->context.caller = NULL;
    cast_info->context.method = (PyArrayMethodObject *)meth;

    // 准备类型描述符数组和输入描述符数组
    PyArray_DTypeMeta *dtypes[2] = {NPY_DTYPE(src_dtype), NPY_DTYPE(dst_dtype)};
    PyArray_Descr *in_descr[2] = {src_dtype, dst_dtype};

    // 解析描述符以获取类型转换方式
    *casting = cast_info->context.method->resolve_descriptors(
            cast_info->context.method, dtypes,
            in_descr, cast_info->descriptors, view_offset);
    if (NPY_UNLIKELY(*casting < 0)) {
        if (!PyErr_Occurred()) {
            // 若未设置异常，则设置类型错误异常
            PyErr_Format(PyExc_TypeError,
                    "Cannot cast array data from %R to %R.", src_dtype, dst_dtype);
        }
        Py_DECREF(meth);
        return -1;
    }

    // 确保描述符有效性
    assert(PyArray_DescrCheck(cast_info->descriptors[0]));
    assert(PyArray_DescrCheck(cast_info->descriptors[1]));

    // 对于非主要步骤的辅助类型转换，确保不会递归解析
    if (!main_step && NPY_UNLIKELY(src_dtype != cast_info->descriptors[0] ||
                                   dst_dtype != cast_info->descriptors[1])) {
        /*
         * We currently do not resolve recursively, but require a non
         * main cast (within the same DType) to be done in a single step.
         * This could be expanded at some point if the need arises.
         */
        PyErr_Format(PyExc_RuntimeError,
                "Required internal cast from %R to %R was not done in a single "
                "step (a secondary cast must currently be between instances of "
                "the same DType class and such a cast must currently return "
                "the input descriptors unmodified).",
                src_dtype, dst_dtype);
        NPY_cast_info_xfree(cast_info);
        return -1;
    }

    return 0;
}


/*
 * 在 ArrayMethod.get_loop(...) 失败时清理类型转换信息，确保引用正确释放。
 * 注意：此函数仅在特定情况下使用，用于处理类型转换信息的异常情况。
 * 
 * Parameters:
 * - cast_info: 指向类型转换信息的指针
 */
static void
_clear_cast_info_after_get_loop_failure(NPY_cast_info *cast_info)
{
    /* As public API we could choose to clear auxdata != NULL */
    // 断言辅助数据为空，通常用于公共 API 的清理
    assert(cast_info->auxdata == NULL);
    /* Set func to be non-null so that `NPY_cats_info_xfree` does not skip */
    // 设置 func 非空，以确保 NPY_cast_info_xfree 不会跳过类型转换信息的释放
    cast_info->func = &_cast_no_op;
    NPY_cast_info_xfree(cast_info);
}
/*
 * Helper for PyArray_GetDTypeTransferFunction, which fetches a single
 * transfer function from each casting implementation (ArrayMethod).
 * May set the transfer function to NULL when the cast can be achieved using
 * a view.
 * TODO: Expand the view functionality for general offsets, not just 0:
 *       Partial casts could be skipped also for `view_offset != 0`.
 *
 * The `out_needs_api` flag must be initialized.
 *
 * NOTE: In theory, casting errors here could be slightly misleading in case
 *       of a multi-step casting scenario. It should be possible to improve
 *       this in the future.
 *
 * Note about `move_references`: Move references means stealing of
 * references. It is useful to clear buffers immediately. No matter the
 * input, all copies from a buffer must use `move_references`. Move references
 * is thus used:
 *   * For the added initial "from" cast if it was passed in.
 *   * Always in the main step if a "from" cast is made (it casts from a buffer).
 *   * Always for the "to" cast, as it always casts from a buffer to the output.
 *
 * Returns -1 on failure, 0 on success.
 */
static int
define_cast_for_descrs(
        int aligned,
        npy_intp src_stride, npy_intp dst_stride,
        PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
        int move_references,
        NPY_cast_info *cast_info, NPY_ARRAYMETHOD_FLAGS *out_flags)
{
    assert(dst_dtype != NULL);  /* Was previously used for decref */

    /* Storage for all cast info in case multi-step casting is necessary */
    _multistep_castdata castdata;
    /* Initialize funcs to NULL to simplify cleanup on error. */
    castdata.main.func = NULL;
    castdata.to.func = NULL;
    castdata.from.func = NULL;
    /* `view_offset` passed to `init_cast_info` but unused for the main cast */
    npy_intp view_offset = NPY_MIN_INTP;
    NPY_CASTING casting = -1;
    *out_flags = PyArrayMethod_MINIMAL_FLAGS;

    if (init_cast_info(
            cast_info, &casting, &view_offset, src_dtype, dst_dtype, 1) < 0) {
        return -1;
    }

    /*
     * Both input and output must be wrapped in case they may be unaligned
     * and the method does not support unaligned data.
     * NOTE: It is probable that most/all legacy loops actually do support
     *       unaligned output, we could move the wrapping there if we wanted
     *       to. It probably isn't speed relevant though and they should be
     *       deleted in any case.
     */
    int must_wrap = (!aligned &&
        (cast_info->context.method->flags & NPY_METH_SUPPORTS_UNALIGNED) == 0);

    /*
     * Wrap the input with an additional cast if necessary.
     */
    # 检查是否需要进行类型转换或者包装
    if (NPY_UNLIKELY(src_dtype != cast_info->descriptors[0] || must_wrap)) {
        # 初始化转换信息的辅助结构
        NPY_CASTING from_casting = -1;
        npy_intp from_view_offset = NPY_MIN_INTP;
        /* 如果转换函数不支持输入类型，则必要时进行包装 */
        if (init_cast_info(
                &castdata.from, &from_casting, &from_view_offset,
                src_dtype, cast_info->descriptors[0], 0) < 0) {
            goto fail;  # 转换初始化失败，跳转到错误处理标签
        }
        # 选择更安全的转换方式
        casting = PyArray_MinCastSafety(casting, from_casting);

        /* 准备实际的转换（如果有必要）： */
        if (from_view_offset == 0 && !must_wrap) {
            /* 如果视图偏移为零且无需包装，则跳过此步骤 */
            castdata.from.func = &_cast_no_op;  /* 避免为NULL */
            NPY_cast_info_xfree(&castdata.from);  # 释放转换信息的内存
        }
        else {
            /* 获取转换函数并设置 */
            PyArrayMethod_Context *context = &castdata.from.context;
            npy_intp strides[2] = {src_stride, cast_info->descriptors[0]->elsize};
            NPY_ARRAYMETHOD_FLAGS flags;
            if (context->method->get_strided_loop(
                    context, aligned, move_references, strides,
                    &castdata.from.func, &castdata.from.auxdata, &flags) < 0) {
                _clear_cast_info_after_get_loop_failure(&castdata.from);  # 获取循环失败后清理转换信息
                goto fail;  # 转换函数获取失败，跳转到错误处理标签
            }
            assert(castdata.from.func != NULL);  # 确保转换函数不为空

            *out_flags = PyArrayMethod_COMBINED_FLAGS(*out_flags, flags);
            /* 主转换现在使用缓冲输入： */
            src_stride = strides[1];  # 更新源数据的步长为目标类型的字节数
            move_references = 1;  /* 主转换必须清除缓冲区 */
        }
    }
    /*
     * 如果需要，使用额外的转换包装输出。
     */
    if (NPY_UNLIKELY(dst_dtype != cast_info->descriptors[1] || must_wrap)) {
        // 检查目标数据类型是否需要转换，或者必须包装
        NPY_CASTING to_casting = -1;
        npy_intp to_view_offset = NPY_MIN_INTP;
        /* Cast function may not support the output, wrap if necessary */
        // 如果初始化类型转换信息失败，则跳转到错误处理
        if (init_cast_info(
                &castdata.to, &to_casting, &to_view_offset,
                cast_info->descriptors[1], dst_dtype,  0) < 0) {
            goto fail;
        }
        // 更新类型转换方式
        casting = PyArray_MinCastSafety(casting, to_casting);

        /* Prepare the actual cast (if necessary): */
        // 准备实际的类型转换（如果需要的话）
        if (to_view_offset == 0 && !must_wrap) {
            /* This step is not necessary and can be skipped. */
            // 如果视图偏移为0且不需要包装，则跳过此步骤
            castdata.to.func = &_cast_no_op;  /* avoid NULL */
            NPY_cast_info_xfree(&castdata.to);
        }
        else {
            /* Fetch the cast function and set up */
            // 获取转换函数并设置
            PyArrayMethod_Context *context = &castdata.to.context;
            npy_intp strides[2] = {cast_info->descriptors[1]->elsize, dst_stride};
            NPY_ARRAYMETHOD_FLAGS flags;
            if (context->method->get_strided_loop(
                    context, aligned, 1 /* clear buffer */, strides,
                    &castdata.to.func, &castdata.to.auxdata, &flags) < 0) {
                _clear_cast_info_after_get_loop_failure(&castdata.to);
                goto fail;
            }
            assert(castdata.to.func != NULL);

            *out_flags = PyArrayMethod_COMBINED_FLAGS(*out_flags, flags);
            /* The main cast now uses a buffered input: */
            dst_stride = strides[0];
            if (castdata.from.func != NULL) {
                /* Both input and output are wrapped, now always aligned */
                aligned = 1;
            }
        }
    }

    /* Fetch the main cast function (with updated values) */
    // 获取主要的类型转换函数（带有更新的值）
    PyArrayMethod_Context *context = &cast_info->context;
    npy_intp strides[2] = {src_stride, dst_stride};
    NPY_ARRAYMETHOD_FLAGS flags;
    if (context->method->get_strided_loop(
            context, aligned, move_references, strides,
            &cast_info->func, &cast_info->auxdata, &flags) < 0) {
        _clear_cast_info_after_get_loop_failure(cast_info);
        goto fail;
    }

    *out_flags = PyArrayMethod_COMBINED_FLAGS(*out_flags, flags);

    if (castdata.from.func == NULL && castdata.to.func == NULL) {
        /* Most of the time, there will be only one step required. */
        // 大多数情况下，只需要一个步骤
        return 0;
    }
    /* The full cast passed in is only the "main" step, copy cast_info there */
    // 传入的完整类型转换仅是主要步骤，将其复制到 cast_info 中
    NPY_cast_info_move(&castdata.main, cast_info);
    Py_INCREF(src_dtype);
    cast_info->descriptors[0] = src_dtype;
    Py_INCREF(dst_dtype);
    cast_info->descriptors[1] = dst_dtype;
    cast_info->context.method = NULL;

    cast_info->func = &_strided_to_strided_multistep_cast;
    cast_info->auxdata = _multistep_cast_auxdata_clone_int(&castdata, 1);
    if (cast_info->auxdata == NULL) {
        PyErr_NoMemory();
        goto fail;
    }

    return 0;

  fail:
    // 释放 castdata 结构中 main 字段所指向的内存
    NPY_cast_info_xfree(&castdata.main);
    // 释放 castdata 结构中 from 字段所指向的内存
    NPY_cast_info_xfree(&castdata.from);
    // 释放 castdata 结构中 to 字段所指向的内存
    NPY_cast_info_xfree(&castdata.to);
    // 返回 -1，表示函数执行失败
    return -1;
/*
 * NPY_NO_EXPORT int
 * PyArray_GetDTypeTransferFunction(int aligned,
 *                            npy_intp src_stride, npy_intp dst_stride,
 *                            PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
 *                            int move_references,
 *                            NPY_cast_info *cast_info,
 *                            NPY_ARRAYMETHOD_FLAGS *out_flags)
 *
 * This function defines and retrieves a dtype transfer function based on the input parameters.
 * It calls `define_cast_for_descrs` to set up casting information and retrieve the transfer function.
 * If successful, it returns NPY_SUCCEED; otherwise, it returns NPY_FAIL.
 */
NPY_NO_EXPORT int
PyArray_GetDTypeTransferFunction(int aligned,
                            npy_intp src_stride, npy_intp dst_stride,
                            PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
                            int move_references,
                            NPY_cast_info *cast_info,
                            NPY_ARRAYMETHOD_FLAGS *out_flags)
{
    if (define_cast_for_descrs(aligned,
            src_stride, dst_stride,
            src_dtype, dst_dtype, move_references,
            cast_info, out_flags) < 0) {
        return NPY_FAIL;
    }

    return NPY_SUCCEED;
}


/*
 * Internal wrapping of casts that have to be performed in a "single"
 * function (i.e. not by the generic multi-step-cast), but rely on it
 * internally. There are only two occasions where this is used:
 *
 * 1. Void advertises that it handles unaligned casts, but has to wrap the
 *    legacy cast which (probably) does not.
 * 2. Datetime to unicode casts are implemented via bytes "U" vs. "S". If
 *    we relax the chaining rules to allow "recursive" cast chaining where
 *    `resolve_descriptors` can return a descriptor with a different type,
 *    this would become unnecessary.
 *  3. Time <-> Time casts, which currently must support byte swapping, but
 *     have a non-trivial inner-loop (due to units) which does not support
 *     it.
 *
 * When wrapping is performed (guaranteed for `aligned == 0` and if the
 * wrapped dtype is not identical to the input dtype), the wrapped transfer
 * function can assume a contiguous input.
 * Otherwise use `must_wrap` to ensure that wrapping occurs, which guarantees
 * a contiguous, aligned, call of the wrapped function.
 */
NPY_NO_EXPORT int
wrap_aligned_transferfunction(
        int aligned, int must_wrap,
        npy_intp src_stride, npy_intp dst_stride,
        PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
        PyArray_Descr *src_wrapped_dtype, PyArray_Descr *dst_wrapped_dtype,
        PyArrayMethod_StridedLoop **out_stransfer,
        NpyAuxData **out_transferdata, int *out_needs_api)
{
    must_wrap = must_wrap | !aligned;

    _multistep_castdata castdata;
    NPY_cast_info_init(&castdata.main);
    NPY_cast_info_init(&castdata.from);
    NPY_cast_info_init(&castdata.to);

    /* Finalize the existing cast information: */
    castdata.main.func = *out_stransfer;
    *out_stransfer = NULL;
    castdata.main.auxdata = *out_transferdata;
    *out_transferdata = NULL;
    castdata.main.context.method = NULL;
    /* These are always legacy casts that only support native-byte-order: */
    Py_INCREF(src_wrapped_dtype);
    castdata.main.descriptors[0] = src_wrapped_dtype;
    if (castdata.main.descriptors[0] == NULL) {
        castdata.main.descriptors[1] = NULL;
        goto fail;
    }
    Py_INCREF(dst_wrapped_dtype);
    castdata.main.descriptors[1] = dst_wrapped_dtype;
    if (castdata.main.descriptors[1] == NULL) {
        goto fail;
    }
    /*
     * 如果必须进行包装（must_wrap 为真）或者源数据类型不匹配，则执行以下步骤：
     * 获取类型转换函数，并填充转换数据结构。此处需要确保源数据类型对齐。
     * 如果获取类型转换函数失败，则跳转到失败处理逻辑。
     * 如果转换函数需要 Python API 支持，则设置 out_needs_api 为 1。
     */
    if (must_wrap || src_wrapped_dtype != src_dtype) {
        NPY_ARRAYMETHOD_FLAGS flags;
        if (PyArray_GetDTypeTransferFunction(aligned,
                src_stride, castdata.main.descriptors[0]->elsize,
                src_dtype, castdata.main.descriptors[0], 0,
                &castdata.from, &flags) != NPY_SUCCEED) {
            goto fail;
        }
        if (flags & NPY_METH_REQUIRES_PYAPI) {
            *out_needs_api = 1;
        }
    }

    /*
     * 如果必须进行包装（must_wrap 为真）或者目标数据类型不匹配，则执行以下步骤：
     * 获取类型转换函数，并填充转换数据结构。此处需要确保目标数据类型对齐。
     * 如果获取类型转换函数失败，则跳转到失败处理逻辑。
     * 如果转换函数需要 Python API 支持，则设置 out_needs_api 为 1。
     */
    if (must_wrap || dst_wrapped_dtype != dst_dtype) {
        NPY_ARRAYMETHOD_FLAGS flags;
        if (PyArray_GetDTypeTransferFunction(aligned,
                castdata.main.descriptors[1]->elsize, dst_stride,
                castdata.main.descriptors[1], dst_dtype,
                1,  /* clear buffer if it includes references */
                &castdata.to, &flags) != NPY_SUCCEED) {
            goto fail;
        }
        if (flags & NPY_METH_REQUIRES_PYAPI) {
            *out_needs_api = 1;
        }
    }

    /*
     * 克隆转换数据结构以便传出，并检查内存分配情况。
     * 如果内存分配失败，则设置错误信息并跳转到失败处理逻辑。
     */
    *out_transferdata = _multistep_cast_auxdata_clone_int(&castdata, 1);
    if (*out_transferdata == NULL) {
        PyErr_NoMemory();
        goto fail;
    }

    /*
     * 设置传出参数 out_stransfer 为指向 _strided_to_strided_multistep_cast 函数的指针，
     * 表示成功完成多步转换。
     */
    *out_stransfer = &_strided_to_strided_multistep_cast;

    // 返回 0 表示成功
    return 0;

  fail:
    /*
     * 失败处理逻辑：释放已分配的转换数据结构资源，并返回 -1 表示失败。
     */
    NPY_cast_info_xfree(&castdata.main);
    NPY_cast_info_xfree(&castdata.from);
    NPY_cast_info_xfree(&castdata.to);

    return -1;
/*
 * This function wraps the legacy casts stored on the PyDataType_GetArrFuncs(`dtype)->cast`
 * or registered with `PyArray_RegisterCastFunc`.
 * For casts between two dtypes with the same type (within DType casts)
 * it also wraps the `copyswapn` function.
 *
 * This function is called from `ArrayMethod.get_loop()` when a specialized
 * cast function is missing.
 *
 * In general, the legacy cast functions do not support unaligned access,
 * so an ArrayMethod using this must signal that.  In a few places we do
 * signal support for unaligned access (or byte swapping).
 * In this case `allow_wrapped=1` will wrap it into an additional multi-step
 * cast as necessary.
 */
NPY_NO_EXPORT int
get_wrapped_legacy_cast_function(int aligned,
        npy_intp src_stride, npy_intp dst_stride,
        PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
        int move_references,
        PyArrayMethod_StridedLoop **out_stransfer,
        NpyAuxData **out_transferdata,
        int *out_needs_api, int allow_wrapped)
{
    /* Note: We ignore `needs_wrap`; needs-wrap is handled by another cast */
    int needs_wrap = 0;

    if (src_dtype->type_num == dst_dtype->type_num) {
        /*
         * This is a cast within the same dtype. For legacy user-dtypes,
         * it is always valid to handle this using the copy swap function.
         */
        // 调用wrap_copy_swap_function处理相同dtype内的转换
        return wrap_copy_swap_function(src_dtype,
                PyDataType_ISNOTSWAPPED(src_dtype) !=
                PyDataType_ISNOTSWAPPED(dst_dtype),
                out_stransfer, out_transferdata);
    }

    if (get_legacy_dtype_cast_function(
            aligned,
            src_stride, dst_stride,
            src_dtype, dst_dtype,
            move_references,
            out_stransfer,
            out_transferdata,
            out_needs_api,
            &needs_wrap) != NPY_SUCCEED) {
        // 如果获取legacy dtype转换函数失败，返回-1
        return -1;
    }
    if (!needs_wrap) {
        // 如果不需要包装，返回0
        return 0;
    }
    if (NPY_UNLIKELY(!allow_wrapped)) {
        /*
         * Legacy casts do not support unaligned which requires wrapping.
         * However, normally we ensure that wrapping happens before calling
         * this function, so this path should never happen.
         */
        // 抛出运行时错误，表明需要包装，但未正确标记支持非对齐数据访问
        PyErr_Format(PyExc_RuntimeError,
                "Internal NumPy error, casting %S to %S required wrapping, "
                "probably it incorrectly flagged support for unaligned data. "
                "(aligned passed to discovery is %d)",
                src_dtype, dst_dtype, aligned);
        goto fail;
    }

    /*
     * If we are here, use the legacy code to wrap the above cast (which
     * does not support unaligned data) into copyswapn.
     */
    // 确保src_dtype和dst_dtype为规范的dtype
    PyArray_Descr *src_wrapped_dtype = NPY_DT_CALL_ensure_canonical(src_dtype);
    if (src_wrapped_dtype == NULL) {
        goto fail;
    }
    PyArray_Descr *dst_wrapped_dtype = NPY_DT_CALL_ensure_canonical(dst_dtype);
    if (dst_wrapped_dtype == NULL) {
        goto fail;
    }
    # 调用 wrap_aligned_transferfunction 函数执行对齐转换操作
    int res = wrap_aligned_transferfunction(
            aligned, 1,  /* We assume wrapped is contiguous here */
            src_stride, dst_stride,
            src_dtype, dst_dtype,
            src_wrapped_dtype, dst_wrapped_dtype,
            out_stransfer, out_transferdata, out_needs_api);
    # 减少源封装数据类型对象的引用计数
    Py_DECREF(src_wrapped_dtype);
    # 减少目标封装数据类型对象的引用计数
    Py_DECREF(dst_wrapped_dtype);
    # 返回 wrap_aligned_transferfunction 的结果
    return res;

  fail:
    # 如果操作失败，释放 out_transferdata 的辅助数据
    NPY_AUXDATA_FREE(*out_transferdata);
    # 将 out_transferdata 置为 NULL
    *out_transferdata = NULL;
    # 返回 -1 表示操作失败
    return -1;
/* 结束上一个函数的定义，开始定义下一个不导出的整型函数 PyArray_GetMaskedDTypeTransferFunction */
NPY_NO_EXPORT int
PyArray_GetMaskedDTypeTransferFunction(int aligned,  // 标志是否对齐的整数
                            npy_intp src_stride,  // 源数组步长
                            npy_intp dst_stride,  // 目标数组步长
                            npy_intp mask_stride,  // 掩码数组步长
                            PyArray_Descr *src_dtype,  // 源数组的数据类型描述符
                            PyArray_Descr *dst_dtype,  // 目标数组的数据类型描述符
                            PyArray_Descr *mask_dtype,  // 掩码数组的数据类型描述符
                            int move_references,  // 是否移动引用
                            NPY_cast_info *cast_info,  // 转换信息结构体指针
                            NPY_ARRAYMETHOD_FLAGS *out_flags)  // 输出的数组方法标志指针
{
    NPY_cast_info_init(cast_info);  // 初始化转换信息结构体

    // 检查掩码数组的数据类型是否为布尔类型或无符号整数类型
    if (mask_dtype->type_num != NPY_BOOL &&
                            mask_dtype->type_num != NPY_UINT8) {
        PyErr_SetString(PyExc_TypeError,
                "Only bool and uint8 masks are supported.");  // 设置类型错误异常
        return NPY_FAIL;  // 返回失败标志
    }

    /* 创建包装函数的辅助数据 */
    _masked_wrapper_transfer_data *data;
    data = PyMem_Malloc(sizeof(_masked_wrapper_transfer_data));  // 分配内存给辅助数据结构体
    if (data == NULL) {
        PyErr_NoMemory();  // 分配内存失败，抛出内存异常
        return NPY_FAIL;  // 返回失败标志
    }
    data->base.free = &_masked_wrapper_transfer_data_free;  // 设置释放函数
    data->base.clone = &_masked_wrapper_transfer_data_clone;  // 设置克隆函数

    /* 回退到包装非掩码传输函数 */
    assert(dst_dtype != NULL);  // 断言目标数据类型不为空
    // 获取非掩码传输函数，如果失败则释放内存并返回失败标志
    if (PyArray_GetDTypeTransferFunction(aligned,
                                src_stride, dst_stride,
                                src_dtype, dst_dtype,
                                move_references,
                                &data->wrapped,
                                out_flags) != NPY_SUCCEED) {
        PyMem_Free(data);  // 释放内存
        return NPY_FAIL;  // 返回失败标志
    }

    /* 如果源对象需要 DECREF，则获取处理该操作的函数 */
    if (move_references && PyDataType_REFCHK(src_dtype)) {
        NPY_ARRAYMETHOD_FLAGS clear_flags;
        // 获取清理函数，处理失败则释放辅助数据并返回失败标志
        if (PyArray_GetClearFunction(
                aligned, src_stride, src_dtype,
                &data->decref_src, &clear_flags) < 0) {
            NPY_AUXDATA_FREE((NpyAuxData *)data);  // 释放辅助数据内存
            return NPY_FAIL;  // 返回失败标志
        }
        *out_flags = PyArrayMethod_COMBINED_FLAGS(*out_flags, clear_flags);  // 更新输出标志
        cast_info->func = (PyArrayMethod_StridedLoop *)
                &_strided_masked_wrapper_clear_function;  // 设置函数指针为清理函数
    }
    else {
        NPY_traverse_info_init(&data->decref_src);  // 初始化遍历信息结构体
        cast_info->func = (PyArrayMethod_StridedLoop *)
                &_strided_masked_wrapper_transfer_function;  // 设置函数指针为传输函数
    }
    cast_info->auxdata = (NpyAuxData *)data;  // 设置辅助数据指针
    /* 上下文几乎不使用，但为了清理而清除 */
    Py_INCREF(src_dtype);  // 增加源数据类型的引用计数
    cast_info->descriptors[0] = src_dtype;  // 设置描述符数组中的第一个描述符
    Py_INCREF(dst_dtype);  // 增加目标数据类型的引用计数
    cast_info->descriptors[1] = dst_dtype;  // 设置描述符数组中的第二个描述符
    cast_info->context.caller = NULL;  // 设置上下文调用者为空
    cast_info->context.method = NULL;  // 设置上下文方法为空

    return NPY_SUCCEED;  // 返回成功标志
}

/* 不导出的整型函数 PyArray_GetMaskedDTypeTransferFunction 结束 */
/*
 * 将原始数组进行类型转换，支持各种数据对齐方式和数据类型间的转换。
 * 如果目标数组的步长为零且转换元素数大于1，则报错。
 * 如果转换元素数为零，则直接返回成功。
 */
PyArray_CastRawArrays(npy_intp count,
                      char *src, char *dst,
                      npy_intp src_stride, npy_intp dst_stride,
                      PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
                      int move_references)
{
    int aligned;

    /* 确保转换操作合理 */
    if (dst_stride == 0 && count > 1) {
        PyErr_SetString(PyExc_ValueError,
                    "NumPy CastRawArrays cannot do a reduction");
        return NPY_FAIL;
    }
    else if (count == 0) {
        return NPY_SUCCEED;
    }

    /* 检查数据对齐性，包括元素大小和对齐方式 */
    aligned = raw_array_is_aligned(1, &count, dst, &dst_stride,
                                   npy_uint_alignment(dst_dtype->elsize)) &&
              raw_array_is_aligned(1, &count, dst, &dst_stride,
                                   dst_dtype->alignment) &&
              raw_array_is_aligned(1, &count, src, &src_stride,
                                   npy_uint_alignment(src_dtype->elsize)) &&
              raw_array_is_aligned(1, &count, src, &src_stride,
                                   src_dtype->alignment);

    /* 获取执行类型转换的函数 */
    NPY_cast_info cast_info;
    NPY_ARRAYMETHOD_FLAGS flags;
    if (PyArray_GetDTypeTransferFunction(aligned,
                        src_stride, dst_stride,
                        src_dtype, dst_dtype,
                        move_references,
                        &cast_info,
                        &flags) != NPY_SUCCEED) {
        return NPY_FAIL;
    }

    /* 清除浮点错误状态 */
    if (!(flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        npy_clear_floatstatus_barrier((char*)&cast_info);
    }

    /* 执行类型转换 */
    char *args[2] = {src, dst};
    npy_intp strides[2] = {src_stride, dst_stride};
    cast_info.func(&cast_info.context, args, &count, strides, cast_info.auxdata);

    /* 清理资源 */
    NPY_cast_info_xfree(&cast_info);

    /* 检查是否需要 Python API，并检查是否有错误发生 */
    if (flags & NPY_METH_REQUIRES_PYAPI && PyErr_Occurred()) {
        return NPY_FAIL;
    }

    /* 检查浮点错误状态 */
    if (!(flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        int fpes = npy_get_floatstatus_barrier(*args);
        if (fpes && PyUFunc_GiveFloatingpointErrors("cast", fpes) < 0) {
            return NPY_FAIL;
        }
    }

    /* 返回操作成功 */
    return NPY_SUCCEED;
}
# 定义一个函数 PyArray_PrepareOneRawArrayIter，准备一个迭代器用于处理单个原始数组
PyArray_PrepareOneRawArrayIter(int ndim, npy_intp const *shape,
                            char *data, npy_intp const *strides,
                            int *out_ndim, npy_intp *out_shape,
                            char **out_data, npy_intp *out_strides)
{
    # 创建一个结构体数组，用于存储排序后的步长及其索引
    npy_stride_sort_item strideperm[NPY_MAXDIMS];
    int i, j;

    /* Special case 0 and 1 dimensions */
    # 特殊情况：处理维度为 0 和 1 的情况
    if (ndim == 0) {
        *out_ndim = 1;
        *out_data = data;
        out_shape[0] = 1;
        out_strides[0] = 0;
        return 0;
    }
    else if (ndim == 1) {
        # 处理维度为 1 的情况
        npy_intp stride_entry = strides[0], shape_entry = shape[0];
        *out_ndim = 1;
        out_shape[0] = shape[0];
        /* Always make a positive stride */
        # 总是确保步长为正数
        if (stride_entry >= 0) {
            *out_data = data;
            out_strides[0] = stride_entry;
        }
        else {
            *out_data = data + stride_entry * (shape_entry - 1);
            out_strides[0] = -stride_entry;
        }
        return 0;
    }

    /* Sort the axes based on the destination strides */
    # 根据目标步长对轴进行排序
    PyArray_CreateSortedStridePerm(ndim, strides, strideperm);
    for (i = 0; i < ndim; ++i) {
        int iperm = strideperm[ndim - i - 1].perm;
        out_shape[i] = shape[iperm];
        out_strides[i] = strides[iperm];
    }

    /* Reverse any negative strides */
    # 反转任何负步长
    for (i = 0; i < ndim; ++i) {
        npy_intp stride_entry = out_strides[i], shape_entry = out_shape[i];

        if (stride_entry < 0) {
            data += stride_entry * (shape_entry - 1);
            out_strides[i] = -stride_entry;
        }
        /* Detect 0-size arrays here */
        # 在这里检测大小为 0 的数组
        if (shape_entry == 0) {
            *out_ndim = 1;
            *out_data = data;
            out_shape[0] = 0;
            out_strides[0] = 0;
            return 0;
        }
    }

    /* Coalesce any dimensions where possible */
    # 尽可能合并维度
    i = 0;
    for (j = 1; j < ndim; ++j) {
        if (out_shape[i] == 1) {
            /* Drop axis i */
            out_shape[i] = out_shape[j];
            out_strides[i] = out_strides[j];
        }
        else if (out_shape[j] == 1) {
            /* Drop axis j */
        }
        else if (out_strides[i] * out_shape[i] == out_strides[j]) {
            /* Coalesce axes i and j */
            out_shape[i] *= out_shape[j];
        }
        else {
            /* Can't coalesce, go to next i */
            ++i;
            out_shape[i] = out_shape[j];
            out_strides[i] = out_strides[j];
        }
    }
    ndim = i+1;

#if 0
    /* DEBUG */
    {
        printf("raw iter ndim %d\n", ndim);
        printf("shape: ");
        for (i = 0; i < ndim; ++i) {
            printf("%d ", (int)out_shape[i]);
        }
        printf("\n");
        printf("strides: ");
        for (i = 0; i < ndim; ++i) {
            printf("%d ", (int)out_strides[i]);
        }
        printf("\n");
    }
#endif

    *out_data = data;
    *out_ndim = ndim;
    return 0;
}
/*
 * The same as PyArray_PrepareOneRawArrayIter, but for two
 * operands instead of one. Any broadcasting of the two operands
 * should have already been done before calling this function,
 * as the ndim and shape is only specified once for both operands.
 *
 * Only the strides of the first operand are used to reorder
 * the dimensions, no attempt to consider all the strides together
 * is made, as is done in the NpyIter object.
 *
 * You can use this together with NPY_RAW_ITER_START and
 * NPY_RAW_ITER_TWO_NEXT to handle the looping boilerplate of everything
 * but the innermost loop (which is for idim == 0).
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
PyArray_PrepareTwoRawArrayIter(int ndim, npy_intp const *shape,
                            char *dataA, npy_intp const *stridesA,
                            char *dataB, npy_intp const *stridesB,
                            int *out_ndim, npy_intp *out_shape,
                            char **out_dataA, npy_intp *out_stridesA,
                            char **out_dataB, npy_intp *out_stridesB)
{
    npy_stride_sort_item strideperm[NPY_MAXDIMS];
    int i, j;

    /* Special case 0 and 1 dimensions */
    if (ndim == 0) {
        // Handle case when ndim is 0 by setting minimal dimensions and strides
        *out_ndim = 1;
        *out_dataA = dataA;
        *out_dataB = dataB;
        out_shape[0] = 1;
        out_stridesA[0] = 0;
        out_stridesB[0] = 0;
        return 0;
    }
    else if (ndim == 1) {
        // Handle case when ndim is 1 by setting dimensions and strides for both operands
        npy_intp stride_entryA = stridesA[0], stride_entryB = stridesB[0];
        npy_intp shape_entry = shape[0];
        *out_ndim = 1;
        out_shape[0] = shape[0];
        /* Always make a positive stride for the first operand */
        if (stride_entryA >= 0) {
            *out_dataA = dataA;
            *out_dataB = dataB;
            out_stridesA[0] = stride_entryA;
            out_stridesB[0] = stride_entryB;
        }
        else {
            // Adjust data pointer and stride if stride is negative for operand A
            *out_dataA = dataA + stride_entryA * (shape_entry - 1);
            *out_dataB = dataB + stride_entryB * (shape_entry - 1);
            out_stridesA[0] = -stride_entryA;
            out_stridesB[0] = -stride_entryB;
        }
        return 0;
    }

    /* Sort the axes based on the destination strides */
    // Sort dimensions based on the strides of operand A
    PyArray_CreateSortedStridePerm(ndim, stridesA, strideperm);
    for (i = 0; i < ndim; ++i) {
        int iperm = strideperm[ndim - i - 1].perm;
        out_shape[i] = shape[iperm];
        out_stridesA[i] = stridesA[iperm];
        out_stridesB[i] = stridesB[iperm];
    }

    /* Reverse any negative strides of operand A */
    // Reverse negative strides of operand A if any exist
    // (additional handling not shown in this snippet)
    
    // Return success
    return 0;
}
    for (i = 0; i < ndim; ++i) {
        // 获取当前维度的步长和形状信息
        npy_intp stride_entryA = out_stridesA[i];
        npy_intp stride_entryB = out_stridesB[i];
        npy_intp shape_entry = out_shape[i];

        // 如果步长为负数，调整数据指针以确保数据连续访问
        if (stride_entryA < 0) {
            dataA += stride_entryA * (shape_entry - 1);
            dataB += stride_entryB * (shape_entry - 1);
            // 更新步长为正数
            out_stridesA[i] = -stride_entryA;
            out_stridesB[i] = -stride_entryB;
        }
        /* 检测数组是否为0大小 */
        if (shape_entry == 0) {
            // 如果数组大小为0，返回最小维度信息
            *out_ndim = 1;
            *out_dataA = dataA;
            *out_dataB = dataB;
            out_shape[0] = 0;
            out_stridesA[0] = 0;
            out_stridesB[0] = 0;
            return 0;
        }
    }

    /* 合并尽可能合并的维度 */
    i = 0;
    for (j = 1; j < ndim; ++j) {
        if (out_shape[i] == 1) {
            /* 删除轴 i */
            out_shape[i] = out_shape[j];
            out_stridesA[i] = out_stridesA[j];
            out_stridesB[i] = out_stridesB[j];
        }
        else if (out_shape[j] == 1) {
            /* 删除轴 j */
            // 无需操作，跳过此轴
        }
        else if (out_stridesA[i] * out_shape[i] == out_stridesA[j] &&
                    out_stridesB[i] * out_shape[i] == out_stridesB[j]) {
            /* 合并轴 i 和 j */
            out_shape[i] *= out_shape[j];
        }
        else {
            /* 无法合并，继续到下一个 i */
            ++i;
            out_shape[i] = out_shape[j];
            out_stridesA[i] = out_stridesA[j];
            out_stridesB[i] = out_stridesB[j];
        }
    }
    ndim = i+1;

    // 更新输出数据的指针和维度信息
    *out_dataA = dataA;
    *out_dataB = dataB;
    *out_ndim = ndim;
    return 0;
/*
 * The same as PyArray_PrepareOneRawArrayIter, but for three
 * operands instead of one. Any broadcasting of the three operands
 * should have already been done before calling this function,
 * as the ndim and shape is only specified once for all operands.
 *
 * Only the strides of the first operand are used to reorder
 * the dimensions, no attempt to consider all the strides together
 * is made, as is done in the NpyIter object.
 *
 * You can use this together with NPY_RAW_ITER_START and
 * NPY_RAW_ITER_THREE_NEXT to handle the looping boilerplate of everything
 * but the innermost loop (which is for idim == 0).
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
PyArray_PrepareThreeRawArrayIter(int ndim, npy_intp const *shape,
                            char *dataA, npy_intp const *stridesA,
                            char *dataB, npy_intp const *stridesB,
                            char *dataC, npy_intp const *stridesC,
                            int *out_ndim, npy_intp *out_shape,
                            char **out_dataA, npy_intp *out_stridesA,
                            char **out_dataB, npy_intp *out_stridesB,
                            char **out_dataC, npy_intp *out_stridesC)
{
    // 用于存储排序后的步幅信息的数组
    npy_stride_sort_item strideperm[NPY_MAXDIMS];
    int i, j;

    /* Special case 0 and 1 dimensions */
    // 处理特殊情况：0 维和 1 维
    if (ndim == 0) {
        // 将输出维度设置为 1
        *out_ndim = 1;
        // 直接使用传入的数据指针和步幅信息
        *out_dataA = dataA;
        *out_dataB = dataB;
        *out_dataC = dataC;
        // 输出形状设置为长度为 1 的数组
        out_shape[0] = 1;
        // 步幅都设置为 0
        out_stridesA[0] = 0;
        out_stridesB[0] = 0;
        out_stridesC[0] = 0;
        // 返回成功状态
        return 0;
    }
    else if (ndim == 1) {
        // 对于一维情况
        npy_intp stride_entryA = stridesA[0];
        npy_intp stride_entryB = stridesB[0];
        npy_intp stride_entryC = stridesC[0];
        npy_intp shape_entry = shape[0];
        // 输出维度设置为 1
        *out_ndim = 1;
        // 输出形状为传入的形状
        out_shape[0] = shape[0];
        // 对于第一个操作数，始终保证步幅是正数
        if (stride_entryA >= 0) {
            *out_dataA = dataA;
            *out_dataB = dataB;
            *out_dataC = dataC;
            out_stridesA[0] = stride_entryA;
            out_stridesB[0] = stride_entryB;
            out_stridesC[0] = stride_entryC;
        }
        else {
            // 如果步幅为负数，根据形状调整数据指针位置
            *out_dataA = dataA + stride_entryA * (shape_entry - 1);
            *out_dataB = dataB + stride_entryB * (shape_entry - 1);
            *out_dataC = dataC + stride_entryC * (shape_entry - 1);
            out_stridesA[0] = -stride_entryA;
            out_stridesB[0] = -stride_entryB;
            out_stridesC[0] = -stride_entryC;
        }
        // 返回成功状态
        return 0;
    }

    /* Sort the axes based on the destination strides */
    // 根据目标步幅对轴进行排序
    PyArray_CreateSortedStridePerm(ndim, stridesA, strideperm);
    for (i = 0; i < ndim; ++i) {
        // 获取排序后的轴索引
        int iperm = strideperm[ndim - i - 1].perm;
        // 输出形状和步幅按照排序后的轴信息设置
        out_shape[i] = shape[iperm];
        out_stridesA[i] = stridesA[iperm];
        out_stridesB[i] = stridesB[iperm];
        out_stridesC[i] = stridesC[iperm];
    }
    /* 反转任何负步长的操作数 A */
    for (i = 0; i < ndim; ++i) {
        // 获取当前维度的步长
        npy_intp stride_entryA = out_stridesA[i];
        npy_intp stride_entryB = out_stridesB[i];
        npy_intp stride_entryC = out_stridesC[i];
        npy_intp shape_entry = out_shape[i];

        // 如果步长为负数，则调整数据指针和步长为正数
        if (stride_entryA < 0) {
            dataA += stride_entryA * (shape_entry - 1);
            dataB += stride_entryB * (shape_entry - 1);
            dataC += stride_entryC * (shape_entry - 1);
            out_stridesA[i] = -stride_entryA;
            out_stridesB[i] = -stride_entryB;
            out_stridesC[i] = -stride_entryC;
        }
        /* 检测是否存在大小为 0 的数组 */
        if (shape_entry == 0) {
            // 设置输出参数指示为 1 维
            *out_ndim = 1;
            // 返回当前数据指针的位置
            *out_dataA = dataA;
            *out_dataB = dataB;
            *out_dataC = dataC;
            // 设置输出形状为 0
            out_shape[0] = 0;
            // 设置步长为 0
            out_stridesA[0] = 0;
            out_stridesB[0] = 0;
            out_stridesC[0] = 0;
            // 返回 0 表示成功处理
            return 0;
        }
    }

    /* 在可能的情况下合并任何维度 */
    i = 0;
    for (j = 1; j < ndim; ++j) {
        if (out_shape[i] == 1) {
            /* 删除轴 i */
            out_shape[i] = out_shape[j];
            out_stridesA[i] = out_stridesA[j];
            out_stridesB[i] = out_stridesB[j];
            out_stridesC[i] = out_stridesC[j];
        }
        else if (out_shape[j] == 1) {
            /* 删除轴 j */
            // 不做任何操作
        }
        else if (out_stridesA[i] * out_shape[i] == out_stridesA[j] &&
                    out_stridesB[i] * out_shape[i] == out_stridesB[j] &&
                    out_stridesC[i] * out_shape[i] == out_stridesC[j]) {
            /* 合并轴 i 和 j */
            out_shape[i] *= out_shape[j];
        }
        else {
            /* 无法合并，继续下一个 i */
            ++i;
            out_shape[i] = out_shape[j];
            out_stridesA[i] = out_stridesA[j];
            out_stridesB[i] = out_stridesB[j];
            out_stridesC[i] = out_stridesC[j];
        }
    }
    ndim = i+1;

    // 更新输出参数
    *out_dataA = dataA;
    *out_dataB = dataB;
    *out_dataC = dataC;
    *out_ndim = ndim;
    // 返回 0 表示成功处理
    return 0;
}
```