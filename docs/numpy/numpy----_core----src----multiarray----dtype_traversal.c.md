# `.\numpy\numpy\_core\src\multiarray\dtype_traversal.c`

```py
/*
 * This file is similar to the low-level loops for data type transfer
 * in `dtype_transfer.c` but for those which only require visiting
 * a single array (and mutating it in-place).
 *
 * As of writing, it is only used for CLEARing, which means mainly
 * Python object DECREF/dealloc followed by NULL'ing the data
 * (to support double clearing and ensure data is again in a usable state).
 * However, memory initialization and traverse follows similar
 * protocols (although traversal needs additional arguments).
 */

// 定义宏，指定不使用已弃用的 NumPy API 版本
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
// 定义宏，标记为 NumPy 的多维数组模块
#define _MULTIARRAYMODULE
// 定义宏，标记为 NumPy 的数学运算模块
#define _UMATHMODULE

// 清除 Python.h 头文件可能定义的不必要符号
#define PY_SSIZE_T_CLEAN
// 包含 Python 的头文件
#include <Python.h>
// 包含结构成员的头文件
#include <structmember.h>

// 包含 NumPy 的核心数据类型定义
#include "numpy/ndarraytypes.h"
// 包含 NumPy 的数组对象定义
#include "numpy/arrayobject.h"

// 包含数组分配相关功能的头文件
#include "alloc.h"
// 包含数组方法相关功能的头文件
#include "array_method.h"
// 包含数据类型元信息相关功能的头文件
#include "dtypemeta.h"
// 包含数据类型遍历功能的头文件
#include "dtype_traversal.h"


/* Buffer size with the same use case as the one in dtype_transfer.c */
// 定义缓冲区块大小，与 dtype_transfer.c 中的用例相同
#define NPY_LOWLEVEL_BUFFER_BLOCKSIZE  128


// 定义一个函数指针类型，用于获取遍历函数
typedef int get_traverse_func_function(
        void *traverse_context, const PyArray_Descr *dtype, int aligned,
        npy_intp stride, NPY_traverse_info *clear_info,
        NPY_ARRAYMETHOD_FLAGS *flags);

/*
 * Generic Clear function helpers:
 */

// 静态函数，用于获取清除函数
static int
get_clear_function(
        void *traverse_context, const PyArray_Descr *dtype, int aligned,
        npy_intp stride, NPY_traverse_info *clear_info,
        NPY_ARRAYMETHOD_FLAGS *flags)
{
    // 初始化遍历信息结构体
    NPY_traverse_info_init(clear_info);
    // 设置数组方法最小标志
    *flags = PyArrayMethod_MINIMAL_FLAGS;

    // 获取清除循环函数指针
    PyArrayMethod_GetTraverseLoop *get_clear = NPY_DT_SLOTS(NPY_DTYPE(dtype))->get_clear_loop;
    if (get_clear == NULL) {
        // 若获取失败，抛出运行时错误
        PyErr_Format(PyExc_RuntimeError,
                "Internal error, `get_clear_loop` not set for the DType '%S'",
                dtype);
        return -1;
    }

    // 调用获取的清除循环函数
    if (get_clear(traverse_context, dtype, aligned, stride,
                  &clear_info->func, &clear_info->auxdata, flags) <  0) {
        // 若调用失败，确保在非调试模式下清理
        assert(clear_info->func == NULL);
        clear_info->func = NULL;
        return -1;
    }
    // 增加数据类型的引用计数
    Py_INCREF(dtype);
    // 设置遍历信息结构体的数据类型描述符
    clear_info->descr = dtype;

    return 0;
}

/*
 * Helper to set up a strided loop used for clearing.  Clearing means
 * deallocating any references (e.g. via Py_DECREF) and resetting the data
 * back into a usable/initialized state (e.g. by NULLing any references).
 *
 * The function will error when called on a dtype which does not have
 * references (and thus the get_clear_loop slot NULL).
 * Note that old-style user-dtypes use the "void" version.
 *
 * NOTE: This function may have a use for a `traverse_context` at some point
 *       but right now, it is always NULL and only exists to allow adding it
 *       in the future without changing the strided-loop signature.
 */
// 导出的函数，用于设置用于清除的步进循环
NPY_NO_EXPORT int
PyArray_GetClearFunction(
        int aligned, npy_intp stride, PyArray_Descr *dtype,
        NPY_traverse_info *clear_info, NPY_ARRAYMETHOD_FLAGS *flags)
{
    // 调用通用的清空函数获取函数指针并返回
    return get_clear_function(NULL, dtype, aligned, stride, clear_info, flags);
}


/*
 * Generic zerofill/fill function helper:
 */

static int
get_zerofill_function(
        void *traverse_context, const PyArray_Descr *dtype, int aligned,
        npy_intp stride, NPY_traverse_info *zerofill_info,
        NPY_ARRAYMETHOD_FLAGS *flags)
{
    // 初始化 zerofill_info 结构
    NPY_traverse_info_init(zerofill_info);
    /* 填充代码检查例如浮点标志 */
    // 设置标志为最小的数组方法标志
    *flags = PyArrayMethod_MINIMAL_FLAGS;

    // 获取填充零循环的函数指针
    PyArrayMethod_GetTraverseLoop *get_zerofill = NPY_DT_SLOTS(NPY_DTYPE(dtype))->get_fill_zero_loop;
    if (get_zerofill == NULL) {
        /* 允许为 NULL（并在此处接受它） */
        return 0;
    }

    // 调用获取填充零循环的函数
    if (get_zerofill(traverse_context, dtype, aligned, stride,
                     &zerofill_info->func, &zerofill_info->auxdata, flags) <  0) {
        /* 调用方应清理，但确保在调试模式外 */
        assert(zerofill_info->func == NULL);
        zerofill_info->func = NULL;
        return -1;
    }
    if (zerofill_info->func == NULL) {
        /* 填充零也可能返回 func=NULL 而没有错误 */
        return 0;
    }

    // 增加 dtype 的引用计数
    Py_INCREF(dtype);
    zerofill_info->descr = dtype;

    return 0;
}


/****************** Python Object clear ***********************/

static int
clear_object_strided_loop(
        void *NPY_UNUSED(traverse_context), const PyArray_Descr *NPY_UNUSED(descr),
        char *data, npy_intp size, npy_intp stride,
        NpyAuxData *NPY_UNUSED(auxdata))
{
    // 初始化一个空指针
    PyObject *aligned_copy = NULL;
    while (size > 0) {
        /* 释放 src 中的引用并将其设置为 NULL */
        memcpy(&aligned_copy, data, sizeof(PyObject *));
        Py_XDECREF(aligned_copy);
        // 将 data 所指向的内存清零
        memset(data, 0, sizeof(PyObject *));

        data += stride;
        --size;
    }
    return 0;
}


NPY_NO_EXPORT int
npy_get_clear_object_strided_loop(
        void *NPY_UNUSED(traverse_context), const PyArray_Descr *NPY_UNUSED(descr),
        int NPY_UNUSED(aligned), npy_intp NPY_UNUSED(fixed_stride),
        PyArrayMethod_TraverseLoop **out_loop, NpyAuxData **out_auxdata,
        NPY_ARRAYMETHOD_FLAGS *flags)
{
    // 设置标志为需要 Python API 并且不会产生浮点错误
    *flags = NPY_METH_REQUIRES_PYAPI|NPY_METH_NO_FLOATINGPOINT_ERRORS;
    // 将清空对象的循环指针设置为 out_loop
    *out_loop = &clear_object_strided_loop;
    return 0;
}


/**************** Python Object zero fill *********************/

static int
fill_zero_object_strided_loop(
        void *NPY_UNUSED(traverse_context), const PyArray_Descr *NPY_UNUSED(descr),
        char *data, npy_intp size, npy_intp stride,
        NpyAuxData *NPY_UNUSED(auxdata))
{
    // 创建一个值为 0 的 PyLong 对象
    PyObject *zero = PyLong_FromLong(0);
    while (size--) {
        // 增加零对象的引用计数
        Py_INCREF(zero);
        // 假设 data 没有预先存在的对象，将 zero 的值拷贝到 data 所指向的内存
        memcpy(data, &zero, sizeof(zero));
        data += stride;
    }
    // 减少零对象的引用计数
    Py_DECREF(zero);
    # 返回整数值 0，结束函数并将该值返回给调用者
    return 0;
}

NPY_NO_EXPORT int
npy_object_get_fill_zero_loop(void *NPY_UNUSED(traverse_context),
                              const PyArray_Descr *NPY_UNUSED(descr),
                              int NPY_UNUSED(aligned),
                              npy_intp NPY_UNUSED(fixed_stride),
                              PyArrayMethod_TraverseLoop **out_loop,
                              NpyAuxData **NPY_UNUSED(out_auxdata),
                              NPY_ARRAYMETHOD_FLAGS *flags)
{
    // 设置方法标志，指定需要 Python C API 支持和无浮点错误
    *flags = NPY_METH_REQUIRES_PYAPI | NPY_METH_NO_FLOATINGPOINT_ERRORS;
    // 指定输出循环为 fill_zero_object_strided_loop
    *out_loop = &fill_zero_object_strided_loop;
    return 0;
}

/**************** Structured DType generic functionality ***************/

/*
 * Note that legacy user dtypes also make use of this.  Someone managed to
 * hack objects into them by adding a field that contains objects and this
 * remains (somewhat) valid.
 * (Unlike our voids, those fields must be hardcoded probably, but...)
 *
 * The below functionality mirrors the casting functionality relatively
 * closely.
 */

typedef struct {
    npy_intp src_offset;
    NPY_traverse_info info;
} single_field_traverse_data;

typedef struct {
    NpyAuxData base;
    npy_intp field_count;
    single_field_traverse_data fields[];
} fields_traverse_data;


/* traverse data free function */
static void
fields_traverse_data_free(NpyAuxData *data)
{
    // 将输入数据转换为 fields_traverse_data 类型
    fields_traverse_data *d = (fields_traverse_data *)data;

    // 释放所有字段遍历信息占用的内存
    for (npy_intp i = 0; i < d->field_count; ++i) {
        NPY_traverse_info_xfree(&d->fields[i].info);
    }
    // 释放 fields_traverse_data 结构体占用的内存
    PyMem_Free(d);
}


/* traverse data copy function (untested due to no direct use currently) */
static NpyAuxData *
fields_traverse_data_clone(NpyAuxData *data)
{
    // 将输入数据转换为 fields_traverse_data 类型
    fields_traverse_data *d = (fields_traverse_data *)data;

    // 获取字段数量和结构体大小
    npy_intp field_count = d->field_count;
    npy_intp structsize = sizeof(fields_traverse_data) +
                    field_count * sizeof(single_field_traverse_data);

    /* 分配数据并进行填充 */
    // 分配新的 fields_traverse_data 结构体内存空间
    fields_traverse_data *newdata = PyMem_Malloc(structsize);
    if (newdata == NULL) {
        return NULL;
    }
    // 复制基础信息
    newdata->base = d->base;
    newdata->field_count = 0;

    // 复制所有字段的遍历数据
    single_field_traverse_data *in_field = d->fields;
    single_field_traverse_data *new_field = newdata->fields;

    for (; newdata->field_count < field_count;
                newdata->field_count++, in_field++, new_field++) {
        new_field->src_offset = in_field->src_offset;

        // 复制遍历信息，如果失败则释放内存并返回 NULL
        if (NPY_traverse_info_copy(&new_field->info, &in_field->info) < 0) {
            fields_traverse_data_free((NpyAuxData *)newdata);
            return NULL;
        }
    }

    return (NpyAuxData *)newdata;
}


static int
traverse_fields_function(
        void *traverse_context, const _PyArray_LegacyDescr *NPY_UNUSED(descr),
        char *data, npy_intp N, npy_intp stride,
        NpyAuxData *auxdata)
{
    // 将输入数据转换为 fields_traverse_data 类型
    fields_traverse_data *d = (fields_traverse_data *)auxdata;
    # 定义变量 i 和 field_count，并初始化为结构体 d 的字段数
    npy_intp i, field_count = d->field_count;

    # 为了更好地利用内存缓存，按块大小进行遍历
    const npy_intp blocksize = NPY_LOWLEVEL_BUFFER_BLOCKSIZE;

    # 无限循环，用于处理数据块遍历
    for (;;) {
        # 如果剩余数据量 N 大于块大小，则按块大小进行遍历处理
        if (N > blocksize) {
            # 遍历结构体 d 的每个字段
            for (i = 0; i < field_count; ++i) {
                # 获取当前字段的信息
                single_field_traverse_data field = d->fields[i];
                # 调用字段信息中指定的函数处理数据块
                if (field.info.func(traverse_context,
                        field.info.descr, data + field.src_offset,
                        blocksize, stride, field.info.auxdata) < 0) {
                    return -1;  # 处理失败，返回错误
                }
            }
            # 更新剩余数据量 N 和数据指针 data
            N -= blocksize;
            data += blocksize * stride;
        }
        else {
            # 否则，处理剩余的所有数据
            for (i = 0; i < field_count; ++i) {
                # 获取当前字段的信息
                single_field_traverse_data field = d->fields[i];
                # 调用字段信息中指定的函数处理剩余数据
                if (field.info.func(traverse_context,
                        field.info.descr, data + field.src_offset,
                        N, stride, field.info.auxdata) < 0) {
                    return -1;  # 处理失败，返回错误
                }
            }
            return 0;  # 处理成功，返回 0
        }
    }
}

/* 
   Traverse a function that retrieves information from a structured NumPy array dtype.
   Args:
       traverse_context: Context for traversal.
       dtype: Legacy descriptor of the NumPy array.
       aligned: Alignment requirement (unused).
       stride: Stride between elements.
       out_func: Pointer to store the traversal loop function.
       out_auxdata: Pointer to store auxiliary data.
       flags: Flags indicating array method properties.
       get_traverse_func: Function pointer to retrieve traversal function.

   Returns:
       int: Status code, 0 for success, -1 for failure.
*/
static int
get_fields_traverse_function(
        void *traverse_context, const _PyArray_LegacyDescr *dtype, int NPY_UNUSED(aligned),
        npy_intp stride, PyArrayMethod_TraverseLoop **out_func,
        NpyAuxData **out_auxdata, NPY_ARRAYMETHOD_FLAGS *flags,
        get_traverse_func_function *get_traverse_func)
{
    PyObject *names, *key, *tup, *title;
    PyArray_Descr *fld_dtype;
    npy_int i, structsize;
    Py_ssize_t field_count;

    names = dtype->names;
    field_count = PyTuple_GET_SIZE(names);

    /* Over-allocating here: less fields may be used */
    structsize = (sizeof(fields_traverse_data) +
                    field_count * sizeof(single_field_traverse_data));
    /* Allocate the data and populate it */
    fields_traverse_data *data = PyMem_Malloc(structsize);
    if (data == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    data->base.free = &fields_traverse_data_free;
    data->base.clone = &fields_traverse_data_clone;
    data->field_count = 0;

    single_field_traverse_data *field = data->fields;
    for (i = 0; i < field_count; ++i) {
        int offset;

        key = PyTuple_GET_ITEM(names, i);
        tup = PyDict_GetItem(dtype->fields, key);
        if (!PyArg_ParseTuple(tup, "Oi|O", &fld_dtype, &offset, &title)) {
            NPY_AUXDATA_FREE((NpyAuxData *)data);
            return -1;
        }
        if (get_traverse_func == &get_clear_function
                && !PyDataType_REFCHK(fld_dtype)) {
            /* No need to do clearing (could change to use NULL return) */
            continue;
        }
        NPY_ARRAYMETHOD_FLAGS clear_flags;
        if (get_traverse_func(
                traverse_context, fld_dtype, 0,
                stride, &field->info, &clear_flags) < 0) {
            NPY_AUXDATA_FREE((NpyAuxData *)data);
            return -1;
        }
        if (field->info.func == NULL) {
            /* zerofill allows NULL func as "default" memset to zero */
            continue;
        }
        *flags = PyArrayMethod_COMBINED_FLAGS(*flags, clear_flags);
        field->src_offset = offset;
        data->field_count++;
        field++;
    }

    *out_func = (PyArrayMethod_TraverseLoop *)&traverse_fields_function;
    *out_auxdata = (NpyAuxData *)data;

    return 0;
}

/* 
   Structure to hold auxiliary data for traversing subarrays.
*/
typedef struct {
    NpyAuxData base;
    npy_intp count;
    NPY_traverse_info info;
} subarray_traverse_data;

/* 
   Free function for subarray traverse data.
   Args:
       data: Pointer to the auxiliary data to free.
*/
static void
subarray_traverse_data_free(NpyAuxData *data)
{
    subarray_traverse_data *d = (subarray_traverse_data *)data;

    NPY_traverse_info_xfree(&d->info);
    PyMem_Free(d);
}

/* 
   Clone function for subarray traverse data.
   Args:
       data: Pointer to the auxiliary data to clone.
   Returns:
       NpyAuxData*: Pointer to the cloned auxiliary data.
*/
static NpyAuxData *
subarray_traverse_data_clone(NpyAuxData *data)
{
    subarray_traverse_data *d = (subarray_traverse_data *)data;

    /* Allocate the data and populate it */
    subarray_traverse_data *newdata = PyMem_Malloc(sizeof(subarray_traverse_data));
    if (newdata == NULL) {
        return NULL;
    }
    newdata->base = d->base;
    // More initialization could go here if necessary

    return (NpyAuxData *)newdata;
}
    # 将结构体 d 中的 count 成员赋值给 newdata 的 count 成员
    newdata->count = d->count;

    # 复制结构体 d 中的 info 成员到 newdata 的 info 成员
    # 如果复制操作失败（返回值小于 0），释放 newdata 分配的内存并返回 NULL
    if (NPY_traverse_info_copy(&newdata->info, &d->info) < 0) {
        PyMem_Free(newdata);
        return NULL;
    }

    # 返回指向 newdata 的指针类型强制转换为 NpyAuxData 指针类型
    return (NpyAuxData *)newdata;
static int
traverse_subarray_func(
        void *traverse_context, const PyArray_Descr *NPY_UNUSED(descr),
        char *data, npy_intp N, npy_intp stride,
        NpyAuxData *auxdata)
{
    // 将辅助数据转换为子数组遍历数据结构
    subarray_traverse_data *subarr_data = (subarray_traverse_data *)auxdata;

    // 获取子数组遍历函数和子数组描述符
    PyArrayMethod_TraverseLoop *func = subarr_data->info.func;
    const PyArray_Descr *sub_descr = subarr_data->info.descr;
    npy_intp sub_N = subarr_data->count;
    NpyAuxData *sub_auxdata = subarr_data->info.auxdata;
    npy_intp sub_stride = sub_descr->elsize;

    // 遍历主数组中的每个元素
    while (N--) {
        // 调用子数组遍历函数处理当前元素
        if (func(traverse_context, sub_descr, data,
                 sub_N, sub_stride, sub_auxdata) < 0) {
            return -1;
        }
        // 移动到下一个主数组元素的位置
        data += stride;
    }
    return 0;
}



static int
get_subarray_traverse_func(
        void *traverse_context, const PyArray_Descr *dtype, int aligned,
        npy_intp size, npy_intp stride, PyArrayMethod_TraverseLoop **out_func,
        NpyAuxData **out_auxdata, NPY_ARRAYMETHOD_FLAGS *flags,
        get_traverse_func_function *get_traverse_func)
{
    // 分配子数组遍历的辅助数据结构
    subarray_traverse_data *auxdata = PyMem_Malloc(sizeof(subarray_traverse_data));
    if (auxdata == NULL) {
        PyErr_NoMemory();
        return -1;
    }

    // 设置子数组遍历数据结构的基本信息和回调函数
    auxdata->count = size;
    auxdata->base.free = &subarray_traverse_data_free;
    auxdata->base.clone = &subarray_traverse_data_clone;

    // 获取子数组遍历函数的具体实现
    if (get_traverse_func(
            traverse_context, dtype, aligned,
            dtype->elsize, &auxdata->info, flags) < 0) {
        PyMem_Free(auxdata);
        return -1;
    }

    // 如果子数组遍历函数为空，则设置输出函数为空并返回
    if (auxdata->info.func == NULL) {
        /* zerofill allows func to be NULL, in which we need not do anything */
        PyMem_Free(auxdata);
        *out_func = NULL;
        *out_auxdata = NULL;
        return 0;
    }

    // 设置输出函数和辅助数据
    *out_func = &traverse_subarray_func;
    *out_auxdata = (NpyAuxData *)auxdata;

    return 0;
}



static int
clear_no_op(
        void *NPY_UNUSED(traverse_context), const PyArray_Descr *NPY_UNUSED(descr),
        char *NPY_UNUSED(data), npy_intp NPY_UNUSED(size),
        npy_intp NPY_UNUSED(stride), NpyAuxData *NPY_UNUSED(auxdata))
{
    // 这是一个空操作函数，不做任何事情，直接返回
    return 0;
}

NPY_NO_EXPORT int
npy_get_clear_void_and_legacy_user_dtype_loop(
        void *traverse_context, const _PyArray_LegacyDescr *dtype, int aligned,
        npy_intp stride, PyArrayMethod_TraverseLoop **out_func,
        NpyAuxData **out_auxdata, NPY_ARRAYMETHOD_FLAGS *flags)
{
    /*
     * 如果数据类型没有引用，那么直接返回空操作函数。
     * 这条路径通常不会被执行，但当包含引用的数据类型被切片后不包含引用时会出现。
     */
    if (!PyDataType_REFCHK((PyArray_Descr *)dtype)) {
        *out_func = &clear_no_op;
        return 0;
    }
    # 如果数据类型中存在子数组
    if (dtype->subarray != NULL) {
        # 定义一个数组形状对象，初始为 NULL，长度为 -1
        PyArray_Dims shape = {NULL, -1};
        npy_intp size;

        # 尝试将子数组的形状转换为 PyArray_Dims 结构
        if (!(PyArray_IntpConverter(dtype->subarray->shape, &shape))) {
            # 如果转换失败，设置异常并返回 -1
            PyErr_SetString(PyExc_ValueError,
                    "invalid subarray shape");
            return -1;
        }
        # 计算数组的总大小
        size = PyArray_MultiplyList(shape.ptr, shape.len);
        # 释放 shape 对象所占用的内存
        npy_free_cache_dim_obj(shape);

        # 获取子数组遍历的函数指针，并检查返回值
        if (get_subarray_traverse_func(
                traverse_context, dtype->subarray->base, aligned, size, stride,
                out_func, out_auxdata, flags, &get_clear_function) < 0) {
            # 如果获取函数指针失败，返回 -1
            return -1;
        }

        # 成功获取函数指针，返回 0
        return 0;
    }
    # 如果数据类型中有结构字段
    /* If there are fields, need to do each field */
    else if (PyDataType_HASFIELDS(dtype)) {
        # 获取字段遍历的函数指针，并检查返回值
        if (get_fields_traverse_function(
                traverse_context, dtype, aligned, stride,
                out_func, out_auxdata, flags, &get_clear_function) < 0) {
            # 如果获取函数指针失败，返回 -1
            return -1;
        }
        # 成功获取函数指针，返回 0
        return 0;
    }
    # 如果数据类型是 NPY_VOID 类型
    else if (dtype->type_num == NPY_VOID) {
        /* 
         * Void dtypes can have "ghosts" of objects marking the dtype because
         * holes (or the raw bytes if fields are gone) may include objects.
         * Paths that need those flags should probably be considered incorrect.
         * But as long as this can happen (a V8 that indicates references)
         * we need to make it a no-op here.
         */
        # 将 out_func 设置为 clear_no_op 函数的指针
        *out_func = &clear_no_op;
        # 返回 0，表示操作成功
        return 0;
    }

    # 如果以上条件都不满足，抛出运行时错误
    PyErr_Format(PyExc_RuntimeError,
            "Internal error, tried to fetch clear function for the "
            "user dtype '%S' without fields or subarray (legacy support).",
            dtype);
    # 返回 -1，表示操作失败
    return -1;
/**************** Structured DType zero fill ***************/

/*
 * Function: zerofill_fields_function
 * ------------------------
 * Zero-fills fields of a structured data array.
 * 
 * traverse_context: Context for traversal (typically unused).
 * descr: Description of the structured data type.
 * data: Pointer to the data buffer.
 * N: Number of elements to process.
 * stride: Stride between elements.
 * auxdata: Additional data for traversal (typically unused).
 * 
 * Returns:
 *     0 on success, -1 on failure.
 */
static int
zerofill_fields_function(
        void *traverse_context, const _PyArray_LegacyDescr *descr,
        char *data, npy_intp N, npy_intp stride,
        NpyAuxData *auxdata)
{
    npy_intp itemsize = descr->elsize;

    /*
     * TODO: We could optimize this by chunking, but since we currently memset
     *       each element always, just loop manually.
     */
    while (N--) {
        // Zero-fill the current element in the data buffer
        memset(data, 0, itemsize);
        // Traverse the fields of the structured data
        if (traverse_fields_function(
                traverse_context, descr, data, 1, stride, auxdata) < 0) {
            return -1;
        }
        data += stride;  // Move to the next element
    }
    return 0;  // Successful completion
}

/*
 * Function: npy_get_zerofill_void_and_legacy_user_dtype_loop
 * ------------------------
 * Retrieves the traversal loop for zero-filling structured data types.
 * 
 * traverse_context: Context for traversal.
 * dtype: Description of the structured data type.
 * aligned: Flag indicating whether data is aligned.
 * stride: Stride between elements.
 * out_func: Pointer to the function pointer that will receive the traversal loop.
 * out_auxdata: Pointer to receive auxiliary data.
 * flags: Flags for array method.
 * 
 * Returns:
 *     0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
npy_get_zerofill_void_and_legacy_user_dtype_loop(
        void *traverse_context, const _PyArray_LegacyDescr *dtype, int aligned,
        npy_intp stride, PyArrayMethod_TraverseLoop **out_func,
        NpyAuxData **out_auxdata, NPY_ARRAYMETHOD_FLAGS *flags)
{
    if (dtype->subarray != NULL) {
        PyArray_Dims shape = {NULL, -1};
        npy_intp size;

        // Convert subarray shape to PyArray_Dims
        if (!(PyArray_IntpConverter(dtype->subarray->shape, &shape))) {
            PyErr_SetString(PyExc_ValueError,
                    "invalid subarray shape");
            return -1;
        }
        size = PyArray_MultiplyList(shape.ptr, shape.len);
        npy_free_cache_dim_obj(shape);

        // Get traversal function for subarray
        if (get_subarray_traverse_func(
                traverse_context, dtype->subarray->base, aligned, size, stride,
                out_func, out_auxdata, flags, &get_zerofill_function) < 0) {
            return -1;
        }

        return 0;
    }
    // If the dtype has fields, handle each field separately
    else if (PyDataType_HASFIELDS(dtype)) {
        // Get traversal function for fields
        if (get_fields_traverse_function(
                traverse_context, dtype, aligned, stride,
                out_func, out_auxdata, flags, &get_zerofill_function) < 0) {
            return -1;
        }
        // If no fields are present, return NULL for zerofill
        if (((fields_traverse_data *)*out_auxdata)->field_count == 0) {
            NPY_AUXDATA_FREE(*out_auxdata);
            *out_auxdata = NULL;
            *out_func = NULL;
            return 0;
        }
        // Use zerofill_fields_function for structured paths
        *out_func = (PyArrayMethod_TraverseLoop *)&zerofill_fields_function;
        return 0;
    }

    // For user-defined dtypes, no action is needed
    *out_auxdata = NULL;
    *out_func = NULL;
    return 0;
}
```