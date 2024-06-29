# `.\numpy\numpy\_core\src\multiarray\methods.c`

```
/*
 * 定义 NPY_NO_DEPRECATED_API 来避免使用已弃用的 NumPy API 版本
 * 定义 _MULTIARRAYMODULE 用于多维数组模块
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

/*
 * 清除 PY_SSIZE_T_CLEAN 宏定义，确保 Python.h 中定义 ssize_t
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

/*
 * 引入 NumPy 相关头文件
 */
#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"

/*
 * 引入 NumPy 内部模块头文件
 */
#include "arrayobject.h"
#include "arrayfunction_override.h"
#include "npy_argparse.h"
#include "npy_config.h"
#include "npy_pycompat.h"
#include "npy_import.h"
#include "ufunc_override.h"
#include "array_coercion.h"
#include "common.h"
#include "templ_common.h" /* 用于 npy_mul_sizes_with_overflow */
#include "ctors.h"
#include "calculation.h"
#include "convert_datatype.h"
#include "descriptor.h"
#include "dtypemeta.h"
#include "item_selection.h"
#include "conversion_utils.h"
#include "shape.h"
#include "strfuncs.h"
#include "array_assign.h"
#include "npy_dlpack.h"
#include "npy_static_data.h"
#include "multiarraymodule.h"

/*
 * 引入其他头文件
 */
#include "methods.h"
#include "alloc.h"

#include <stdarg.h>


/*
 * NpyArg_ParseKeywords
 *
 * 用于不需要 args 参数的情况下提供与 PyArg_ParseTupleAndKeywords 类似的关键字解析功能的实用函数
 */
static int
NpyArg_ParseKeywords(PyObject *keys, const char *format, char **kwlist, ...)
{
    PyObject *args = PyTuple_New(0);
    int ret;
    va_list va;

    if (args == NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                "Failed to allocate new tuple");
        return 0;
    }
    va_start(va, kwlist);
    ret = PyArg_VaParseTupleAndKeywords(args, keys, format, kwlist, va);
    va_end(va);
    Py_DECREF(args);
    return ret;
}


/*
 * npy_forward_method
 *
 * 将方法调用转发到 Python 函数，同时添加 self 参数：
 * callable(self, ...)
 */
static PyObject *
npy_forward_method(
        PyObject *callable, PyObject *self,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    PyObject *args_buffer[NPY_MAXARGS];
    /* 理论上 NPY_MAXARGS 足够用 */
    PyObject **new_args = args_buffer;

    /*
     * `PY_VECTORCALL_ARGUMENTS_OFFSET` 似乎从未设置，可能 `args[-1]`
     * 总是 `self`，但除非 Python 文档中明确说明，不要依赖于此。
     */
    npy_intp len_kwargs = kwnames != NULL ? PyTuple_GET_SIZE(kwnames) : 0;
    size_t original_arg_size = (len_args + len_kwargs) * sizeof(PyObject *);

    if (NPY_UNLIKELY(len_args + len_kwargs > NPY_MAXARGS)) {
        new_args = (PyObject **)PyMem_MALLOC(original_arg_size + sizeof(PyObject *));
        if (new_args == NULL) {
            /*
             * 如果分配内存失败，Python 将使用 `PY_VECTORCALL_ARGUMENTS_OFFSET`，
             * 我们可能需要为此添加一个快速路径（希望几乎总是）。
             */
            return PyErr_NoMemory();
        }
    }

    new_args[0] = self;
    memcpy(&new_args[1], args, original_arg_size);
    PyObject *res = PyObject_Vectorcall(callable, new_args, len_args+1, kwnames);

    if (NPY_UNLIKELY(len_args + len_kwargs > NPY_MAXARGS)) {
        PyMem_FREE(new_args);
    }
    return res;
}
/*
 * Forwards an ndarray method to the function numpy._core._methods.<name>(...),
 * caching the callable in a local static variable. Note that the
 * initialization is not thread-safe, but relies on the CPython GIL to
 * be correct.
 */
#define NPY_FORWARD_NDARRAY_METHOD(name)                                \
    npy_cache_import(                                                   \
            "numpy._core._methods", #name,                              \
            &npy_thread_unsafe_state.name);                         \
    if (npy_thread_unsafe_state.name == NULL) {                     \
        return NULL;                                                    \
    }                                                                   \
    return npy_forward_method(npy_thread_unsafe_state.name,         \
                              (PyObject *)self, args, len_args, kwnames)

/*
 * Implements the ndarray 'take' method in Python/C API. This method retrieves
 * elements from the array based on specified indices along a given axis,
 * handling optional parameters like output array, axis, and clipping mode.
 */
static PyObject *
array_take(PyArrayObject *self,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    int dimension = NPY_RAVEL_AXIS;  // Default axis for flattening
    PyObject *indices;               // Indices array
    PyArrayObject *out = NULL;       // Output array
    NPY_CLIPMODE mode = NPY_RAISE;   // Clipping mode
    NPY_PREPARE_ARGPARSER;           // Prepares argument parsing context

    if (npy_parse_arguments("take", args, len_args, kwnames,
            "indices", NULL, &indices,
            "|axis", &PyArray_AxisConverter, &dimension,
            "|out", &PyArray_OutputConverter, &out,
            "|mode", &PyArray_ClipmodeConverter, &mode,
            NULL, NULL, NULL) < 0) {
        return NULL;  // Argument parsing failed
    }

    PyObject *ret = PyArray_TakeFrom(self, indices, dimension, out, mode);

    /* this matches the unpacking behavior of ufuncs */
    if (out == NULL) {
        return PyArray_Return((PyArrayObject *)ret);  // Return as ndarray
    }
    else {
        return ret;  // Return directly
    }
}

/*
 * Implements the ndarray 'fill' method in Python/C API. This method fills
 * the array with a scalar value provided as argument.
 */
static PyObject *
array_fill(PyArrayObject *self, PyObject *args)
{
    PyObject *obj;  // Scalar value to fill with
    if (!PyArg_ParseTuple(args, "O:fill", &obj)) {
        return NULL;  // Parsing argument tuple failed
    }
    if (PyArray_FillWithScalar(self, obj) < 0) {
        return NULL;  // Filling array failed
    }
    Py_RETURN_NONE;  // Return None
}

/*
 * Implements the ndarray 'put' method in Python/C API. This method puts values
 * into the array at specified indices, handling optional parameters like
 * values array and clipping mode.
 */
static PyObject *
array_put(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *indices, *values;      // Indices and values arrays
    NPY_CLIPMODE mode = NPY_RAISE;   // Clipping mode
    static char *kwlist[] = {"indices", "values", "mode", NULL};  // Keyword list

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|O&:put", kwlist,
                                     &indices,
                                     &values,
                                     PyArray_ClipmodeConverter, &mode))
        return NULL;  // Parsing argument tuple failed
    return PyArray_PutTo(self, values, indices, mode);  // Put values into array
}

/*
 * Implements the ndarray 'reshape' method in Python/C API. This method reshapes
 * the array according to new dimensions, with optional parameters for order
 * and copy mode.
 */
static PyObject *
array_reshape(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
    static char *keywords[] = {"order", "copy", NULL};  // Keyword list
    PyArray_Dims newshape;  // New shape dimensions
    PyObject *ret;          // Return value
    NPY_ORDER order = NPY_CORDER;  // Default order
    NPY_COPYMODE copy = NPY_COPY_IF_NEEDED;  // Default copy mode
    Py_ssize_t n = PyTuple_Size(args);  // Number of arguments in tuple
    // 解析关键字参数 kwds，接受两个可选参数，分别转换为 order 和 copy
    if (!NpyArg_ParseKeywords(kwds, "|$O&O&", keywords,
                PyArray_OrderConverter, &order,
                PyArray_CopyConverter, &copy)) {
        // 解析失败则返回 NULL
        return NULL;
    }

    // 如果 n 小于等于 1
    if (n <= 1) {
        // 如果 n 不等于 0 并且 args 中的第一个参数是 Py_None
        if (n != 0 && PyTuple_GET_ITEM(args, 0) == Py_None) {
            // 返回一个基于 self 的视图（view），无需新的形状和步幅参数
            return PyArray_View(self, NULL, NULL);
        }
        // 如果无法解析 args 为一个整数数组，返回 NULL
        if (!PyArg_ParseTuple(args, "O&:reshape", PyArray_IntpConverter,
                              &newshape)) {
            return NULL;
        }
    }
    else {
        // 如果 n 大于 1，尝试解析 args 为整数数组 newshape
        if (!PyArray_IntpConverter(args, &newshape)) {
            // 如果未设置错误信息，则设置类型错误信息
            if (!PyErr_Occurred()) {
                PyErr_SetString(PyExc_TypeError,
                                "invalid shape");
            }
            // 跳转到失败处理块
            goto fail;
        }
    }
    // 调用 _reshape_with_copy_arg 函数进行数组重塑，返回结果给 ret
    ret = _reshape_with_copy_arg(self, &newshape, order, copy);
    // 释放 newshape 中的缓存对象
    npy_free_cache_dim_obj(newshape);
    // 返回重塑后的结果 ret
    return ret;

 fail:
    // 失败时释放 newshape 中的缓存对象，返回 NULL
    npy_free_cache_dim_obj(newshape);
    return NULL;
static PyObject *
array_squeeze(PyArrayObject *self,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    PyObject *axis_in = NULL;  // 定义用于接收轴参数的变量
    npy_bool axis_flags[NPY_MAXDIMS];  // 定义用于存储轴标志的数组
    NPY_PREPARE_ARGPARSER;  // 宏，准备参数解析器

    if (npy_parse_arguments("squeeze", args, len_args, kwnames,
            "|axis", NULL, &axis_in,  // 解析参数中的可选轴参数
            NULL, NULL, NULL) < 0) {  // 如果解析失败，返回空
        return NULL;
    }

    if (axis_in == NULL || axis_in == Py_None) {  // 如果未指定轴参数或者为None
        return PyArray_Squeeze(self);  // 调用NumPy函数去除数组中的单维度
    }
    else {
        if (PyArray_ConvertMultiAxis(axis_in, PyArray_NDIM(self),
                                            axis_flags) != NPY_SUCCEED) {
            return NULL;  // 如果无法转换多个轴参数，返回空
        }

        return PyArray_SqueezeSelected(self, axis_flags);  // 根据指定的轴参数去除数组中的单维度
    }
}

static PyObject *
array_view(PyArrayObject *self,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    PyObject *out_dtype = NULL;  // 用于接收dtype参数的变量
    PyObject *out_type = NULL;  // 用于接收type参数的变量
    PyArray_Descr *dtype = NULL;  // NumPy数组描述符指针
    NPY_PREPARE_ARGPARSER;  // 宏，准备参数解析器

    if (npy_parse_arguments("view", args, len_args, kwnames,
            "|dtype", NULL, &out_dtype,  // 解析参数中的可选dtype参数
            "|type", NULL, &out_type,  // 解析参数中的可选type参数
            NULL, NULL, NULL) < 0) {  // 如果解析失败，返回空
        return NULL;
    }

    /* If user specified a positional argument, guess whether it
       represents a type or a dtype for backward compatibility. */
    if (out_dtype) {  // 如果指定了dtype参数
        /* type specified? */
        if (PyType_Check(out_dtype) &&
            PyType_IsSubtype((PyTypeObject *)out_dtype,
                             &PyArray_Type)) {  // 如果dtype是有效的NumPy数组子类
            if (out_type) {  // 如果同时指定了type参数，报错
                PyErr_SetString(PyExc_ValueError,
                                "Cannot specify output type twice.");
                return NULL;
            }
            out_type = out_dtype;  // 将dtype参数赋给type参数
            out_dtype = NULL;  // 清空dtype参数
        }
    }

    if ((out_type) && (!PyType_Check(out_type) ||
                       !PyType_IsSubtype((PyTypeObject *)out_type,
                                         &PyArray_Type))) {
        PyErr_SetString(PyExc_ValueError,
                        "Type must be a sub-type of ndarray type");
        return NULL;  // 如果指定的type参数不是有效的NumPy数组子类，报错
    }

    if ((out_dtype) &&
        (PyArray_DescrConverter(out_dtype, &dtype) == NPY_FAIL)) {
        return NULL;  // 如果无法将dtype参数转换为NumPy数组描述符，返回空
    }

    return PyArray_View(self, dtype, (PyTypeObject*)out_type);  // 返回数组的视图对象
}

static PyObject *
array_argmax(PyArrayObject *self,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    int axis = NPY_RAVEL_AXIS;  // 轴的默认值
    PyArrayObject *out = NULL;  // 输出数组的指针
    npy_bool keepdims = NPY_FALSE;  // 是否保持维度标志
    NPY_PREPARE_ARGPARSER;  // 宏，准备参数解析器

    if (npy_parse_arguments("argmax", args, len_args, kwnames,
            "|axis", &PyArray_AxisConverter, &axis,  // 解析可选的轴参数
            "|out", &PyArray_OutputConverter, &out,  // 解析可选的输出参数
            "$keepdims", &PyArray_BoolConverter, &keepdims,  // 解析可选的keepdims参数
            NULL, NULL, NULL) < 0) {
        return NULL;  // 如果解析失败，返回空
    }

    PyObject *ret = _PyArray_ArgMaxWithKeepdims(self, axis, out, keepdims);

    /* this matches the unpacking behavior of ufuncs */
    // 这与ufunc的解包行为相匹配，暂无其他注释
    # 如果 out 是 NULL，则将 ret 转换为 PyArrayObject 对象后返回
    if (out == NULL) {
        return PyArray_Return((PyArrayObject *)ret);
    }
    # 否则直接返回 ret
    else {
        return ret;
    }
static PyObject *
array_argmin(PyArrayObject *self,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    // 设置默认的轴为扁平化后的轴
    int axis = NPY_RAVEL_AXIS;
    // 输出数组对象初始化为空
    PyArrayObject *out = NULL;
    // 是否保持维度参数初始化为假
    npy_bool keepdims = NPY_FALSE;
    // 准备解析参数
    NPY_PREPARE_ARGPARSER;
    // 解析参数，根据结果进行相应处理
    if (npy_parse_arguments("argmin", args, len_args, kwnames,
            "|axis", &PyArray_AxisConverter, &axis,
            "|out", &PyArray_OutputConverter, &out,
            "$keepdims", &PyArray_BoolConverter, &keepdims,
            NULL, NULL, NULL) < 0) {
        // 解析失败，返回空指针
        return NULL;
    }

    // 调用具体的函数计算最小值的索引
    PyObject *ret = _PyArray_ArgMinWithKeepdims(self, axis, out, keepdims);

    /* this matches the unpacking behavior of ufuncs */
    // 根据输出情况选择返回结果
    if (out == NULL) {
        // 如果输出为空，则转换为数组对象并返回
        return PyArray_Return((PyArrayObject *)ret);
    }
    else {
        // 否则直接返回结果对象
        return ret;
    }
}

static PyObject *
array_max(PyArrayObject *self,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    // 转发到 _amax 函数处理
    NPY_FORWARD_NDARRAY_METHOD(_amax);
}

static PyObject *
array_min(PyArrayObject *self,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    // 转发到 _amin 函数处理
    NPY_FORWARD_NDARRAY_METHOD(_amin);
}

static PyObject *
array_swapaxes(PyArrayObject *self, PyObject *args)
{
    // 定义两个轴变量
    int axis1, axis2;

    // 解析传入的参数元组，获取两个轴的值
    if (!PyArg_ParseTuple(args, "ii:swapaxes", &axis1, &axis2)) {
        // 解析失败，返回空指针
        return NULL;
    }
    // 调用 NumPy 库函数交换数组轴
    return PyArray_SwapAxes(self, axis1, axis2);
}


/*NUMPY_API
  从数组的每个元素中获取字节的子集
  typed 参数的引用被窃取，不得为空
*/
NPY_NO_EXPORT PyObject *
PyArray_GetField(PyArrayObject *self, PyArray_Descr *typed, int offset)
{
    PyObject *ret = NULL;
    PyObject *safe;
    int self_elsize, typed_elsize;

    // 检查 self 是否为空
    if (self == NULL) {
        // 设置错误信息并返回空指针
        PyErr_SetString(PyExc_ValueError,
            "self is NULL in PyArray_GetField");
        return NULL;
    }

    // 检查 typed 是否为空
    if (typed == NULL) {
        // 设置错误信息并返回空指针
        PyErr_SetString(PyExc_ValueError,
            "typed is NULL in PyArray_GetField");
        return NULL;
    }

    /* 检查是否重新解释包含对象的内存 */
    if (_may_have_objects(PyArray_DESCR(self)) || _may_have_objects(typed)) {
        // 导入线程安全状态
        npy_cache_import("numpy._core._internal", "_getfield_is_safe",
                         &npy_thread_unsafe_state._getfield_is_safe);
        // 检查线程安全状态是否为 NULL
        if (npy_thread_unsafe_state._getfield_is_safe == NULL) {
            // 释放 typed 并返回空指针
            Py_DECREF(typed);
            return NULL;
        }

        /* 只返回 True 或抛出异常 */
        // 调用 _getfield_is_safe 函数进行安全性检查
        safe = PyObject_CallFunction(npy_thread_unsafe_state._getfield_is_safe,
                                     "OOi", PyArray_DESCR(self),
                                     typed, offset);
        // 检查安全性检查返回值是否为空
        if (safe == NULL) {
            // 释放 typed 并返回空指针
            Py_DECREF(typed);
            return NULL;
        }
        // 释放 safe 对象引用
        Py_DECREF(safe);
    }

    // 获取 self 和 typed 的元素大小
    self_elsize = PyArray_ITEMSIZE(self);
    typed_elsize = typed->elsize;

    /* 检查值是否有效 */
    // 继续执行后续操作
    # 如果新类型的元素大小大于原始类型的元素大小，抛出值错误异常
    if (typed_elsize > self_elsize) {
        PyErr_SetString(PyExc_ValueError, "new type is larger than original type");
        Py_DECREF(typed);
        return NULL;
    }
    # 如果偏移量为负数，抛出值错误异常
    if (offset < 0) {
        PyErr_SetString(PyExc_ValueError, "offset is negative");
        Py_DECREF(typed);
        return NULL;
    }
    # 如果偏移量超过了原始类型大小减去新类型大小，抛出值错误异常
    if (offset > self_elsize - typed_elsize) {
        PyErr_SetString(PyExc_ValueError, "new type plus offset is larger than original type");
        Py_DECREF(typed);
        return NULL;
    }

    # 调用 PyArray_NewFromDescr_int 函数创建一个新的数组对象
    ret = PyArray_NewFromDescr_int(
            Py_TYPE(self), typed,
            PyArray_NDIM(self), PyArray_DIMS(self), PyArray_STRIDES(self),
            PyArray_BYTES(self) + offset,
            PyArray_FLAGS(self) & ~NPY_ARRAY_F_CONTIGUOUS,
            (PyObject *)self, (PyObject *)self,
            _NPY_ARRAY_ALLOW_EMPTY_STRING);
    # 返回创建的新数组对象
    return ret;
/*NUMPY_API*/
# 定义一个 API 函数，用于对数组进行字节交换
NPY_NO_EXPORT PyObject *
PyArray_Byteswap(PyArrayObject *self, npy_bool inplace)
{
    PyArrayObject *ret;  // 定义一个 PyArrayObject 指针 ret
    npy_intp size;  // 定义一个数组大小的整数型变量 size
    PyArray_CopySwapNFunc *copyswapn;  // 定义一个指向 PyArray_CopySwapNFunc 函数的指针 copyswapn
    PyArrayIterObject *it;  // 定义一个 PyArrayIterObject 迭代器对象指针 it

    // 获取数组的拷贝和交换函数，并赋给 copyswapn
    copyswapn = PyDataType_GetArrFuncs(PyArray_DESCR(self))->copyswapn;
    # 如果 inplace 参数为真
    if (inplace) {
        # 确保数组可写，如果不可写则返回空指针
        if (PyArray_FailUnlessWriteable(self, "array to be byte-swapped") < 0) {
            return NULL;
        }
        # 获取数组的总元素个数
        size = PyArray_SIZE(self);
        # 如果数组是一段连续的内存块
        if (PyArray_ISONESEGMENT(self)) {
            # 对整个数组执行字节交换
            copyswapn(PyArray_DATA(self), PyArray_ITEMSIZE(self), NULL, -1, size, 1, self);
        }
        else { /* 使用迭代器 */
            # 设定轴为 -1
            int axis = -1;
            npy_intp stride;
            # 获取除指定轴外的所有轴的迭代器对象
            it = (PyArrayIterObject *)                      \
                PyArray_IterAllButAxis((PyObject *)self, &axis);
            # 获取指定轴的步长
            stride = PyArray_STRIDES(self)[axis];
            # 获取指定轴的长度
            size = PyArray_DIMS(self)[axis];
            # 遍历迭代器对象，对每个子数组执行字节交换
            while (it->index < it->size) {
                copyswapn(it->dataptr, stride, NULL, -1, size, 1, self);
                PyArray_ITER_NEXT(it);
            }
            # 释放迭代器对象
            Py_DECREF(it);
        }

        # 增加数组的引用计数
        Py_INCREF(self);
        # 返回字节交换后的数组对象
        return (PyObject *)self;
    }
    else {
        PyObject *new;
        # 使用 PyArray_NewCopy 创建数组的深拷贝，如果失败则返回空指针
        if ((ret = (PyArrayObject *)PyArray_NewCopy(self,-1)) == NULL) {
            return NULL;
        }
        # 对新创建的数组执行字节交换
        new = PyArray_Byteswap(ret, NPY_TRUE);
        # 减少新数组对象的引用计数
        Py_DECREF(new);
        # 返回字节交换后的新数组对象
        return (PyObject *)ret;
    }
static PyObject *
array_tofile(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
    int own;
    PyObject *file;
    char *sep = "";
    char *format = "";
    static char *kwlist[] = {"file", "sep", "format", NULL};

    // 解析传入的参数，其中file是必选参数，sep和format是可选参数，默认为空字符串
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|ss:tofile", kwlist,
                                     &file,
                                     &sep,
                                     &format)) {
        return NULL;
    }

    // 将file对象转换为文件系统路径对象
    file = NpyPath_PathlikeToFspath(file);
    if (file == NULL) {
        return NULL;
    }
    
    // 检查file对象是否为字节对象或Unicode对象，若是则尝试以"wb"模式打开文件
    if (PyBytes_Check(file) || PyUnicode_Check(file)) {
        // 使用npy_PyFile_OpenFile函数打开文件，返回的文件对象赋值给file变量
        Py_SETREF(file, npy_PyFile_OpenFile(file, "wb"));
        if (file == NULL) {
            return NULL;
        }
        own = 1;  // 设置own为1，表示需要管理文件对象的生命周期
    }
    else {
        own = 0;  // 文件对象的生命周期由调用者管理
    }
    # 调用 NumPy 函数将数组内容写入文件对象，返回一个整数表示操作结果
    int file_ret = PyArray_ToFileObject(self, file, sep, format);

    # 初始化关闭文件返回值为 0
    int close_ret = 0;

    # 如果需要关闭文件
    if (own) {
        // 保存当前的错误类型、错误值和错误回溯信息
        PyObject *err_type, *err_value, *err_traceback;
        PyErr_Fetch(&err_type, &err_value, &err_traceback);
        
        // 调用 NumPy 的函数关闭文件，更新关闭文件的返回值
        close_ret = npy_PyFile_CloseFile(file);
        
        // 将之前保存的错误类型、错误值和错误回溯信息恢复并抛出异常
        npy_PyErr_ChainExceptions(err_type, err_value, err_traceback);
    }

    // 减少文件对象的引用计数，释放其内存
    Py_DECREF(file);

    // 如果写入文件操作或者关闭文件操作中有任何一个失败，则返回空值对象
    if (file_ret || close_ret) {
        return NULL;
    }
    
    // 操作成功完成，返回空值对象
    Py_RETURN_NONE;
static PyObject *
array_toscalar(PyArrayObject *self, PyObject *args)
{
    // 定义一个多维数组索引数组
    npy_intp multi_index[NPY_MAXDIMS];
    // 获取参数元组的大小
    int n = PyTuple_GET_SIZE(args);
    // 获取数组的维度数
    int idim, ndim = PyArray_NDIM(self);

    /* 如果参数数量为1且第一个参数是元组，则将参数指向该元组 */
    if (n == 1 && PyTuple_Check(PyTuple_GET_ITEM(args, 0))) {
        args = PyTuple_GET_ITEM(args, 0);
        n = PyTuple_GET_SIZE(args);
    }

    // 如果参数数量为0
    if (n == 0) {
        // 如果数组大小为1，将多维数组索引全部设为0
        if (PyArray_SIZE(self) == 1) {
            for (idim = 0; idim < ndim; ++idim) {
                multi_index[idim] = 0;
            }
        }
        // 否则，抛出值错误异常，说明只能将大小为1的数组转换为标量
        else {
            PyErr_SetString(PyExc_ValueError,
                    "can only convert an array of size 1 to a Python scalar");
            return NULL;
        }
    }
    // 如果参数数量为1且数组维度数不为1，处理C顺序的平铺索引
    else if (n == 1 && ndim != 1) {
        // 获取数组形状和大小
        npy_intp *shape = PyArray_SHAPE(self);
        npy_intp value, size = PyArray_SIZE(self);

        // 将第一个参数转换为整数索引
        value = PyArray_PyIntAsIntp(PyTuple_GET_ITEM(args, 0));
        if (error_converting(value)) {
            return NULL;
        }

        // 检查并调整索引值
        if (check_and_adjust_index(&value, size, -1, NULL) < 0) {
            return NULL;
        }

        /* 将平铺索引转换为多维索引 */
        for (idim = ndim-1; idim >= 0; --idim) {
            multi_index[idim] = value % shape[idim];
            value /= shape[idim];
        }
    }
    // 如果参数数量等于数组维度数，处理多维索引元组
    else if (n == ndim) {
        npy_intp value;

        for (idim = 0; idim < ndim; ++idim) {
            // 将每个参数转换为整数索引
            value = PyArray_PyIntAsIntp(PyTuple_GET_ITEM(args, idim));
            if (error_converting(value)) {
                return NULL;
            }
            multi_index[idim] = value;
        }
    }
    // 参数数量与数组维度数不符，抛出值错误异常
    else {
        PyErr_SetString(PyExc_ValueError,
                "incorrect number of indices for array");
        return NULL;
    }

    // 返回数组指定多维索引的元素
    return PyArray_MultiIndexGetItem(self, multi_index);
}

static PyObject *
array_astype(PyArrayObject *self,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    /*
     * TODO: UNSAFE default for compatibility, I think
     *       switching to SAME_KIND by default would be good.
     */
    // 数据类型信息结构体，默认使用UNSAFE类型转换
    npy_dtype_info dt_info = {NULL, NULL};
    NPY_CASTING casting = NPY_UNSAFE_CASTING;
    NPY_ORDER order = NPY_KEEPORDER;
    NPY_ASTYPECOPYMODE forcecopy = 1;
    int subok = 1;

    // 准备解析器参数
    NPY_PREPARE_ARGPARSER;
    // 解析函数参数
    if (npy_parse_arguments("astype", args, len_args, kwnames,
            // 解析dtype参数
            "dtype", &PyArray_DTypeOrDescrConverterRequired, &dt_info,
            // 解析order参数
            "|order", &PyArray_OrderConverter, &order,
            // 解析casting参数
            "|casting", &PyArray_CastingConverter, &casting,
            // 解析subok参数
            "|subok", &PyArray_PythonPyIntFromInt, &subok,
            // 解析copy参数
            "|copy", &PyArray_AsTypeCopyConverter, &forcecopy,
            NULL, NULL, NULL) < 0) {
        // 解析失败时释放内存并返回空指针
        Py_XDECREF(dt_info.descr);
        Py_XDECREF(dt_info.dtype);
        return NULL;
    }

    /* 如果不是具体的dtype实例，则为数组找到最佳的dtype实例 */

    // （此处未提供完整代码）
}
    // 定义一个指向 PyArray_Descr 结构体的指针变量
    PyArray_Descr *dtype;

    // 调用 PyArray_AdaptDescriptorToArray 函数，将 self 对象适配为描述符数组，返回描述符指针给 dtype，并释放 dt_info.descr 引用计数
    dtype = PyArray_AdaptDescriptorToArray(self, dt_info.dtype, dt_info.descr);
    Py_XDECREF(dt_info.descr);  // 释放 dt_info.descr 的引用计数
    Py_DECREF(dt_info.dtype);   // 释放 dt_info.dtype 的引用计数
    if (dtype == NULL) {  // 如果 dtype 为空，则返回空指针
        return NULL;
    }

    /*
     * 如果内存布局匹配，并且数据类型等效，
     * 当 subok 为 False 时不是子类型，
     * 如果转换允许视图，则可以跳过复制。
     */
    if (forcecopy != NPY_AS_TYPE_COPY_ALWAYS &&  // 如果不是强制复制
                    (order == NPY_KEEPORDER ||  // 保持原序
                    (order == NPY_ANYORDER &&    // 任意顺序
                        (PyArray_IS_C_CONTIGUOUS(self) ||  // 是 C 连续数组
                        PyArray_IS_F_CONTIGUOUS(self))) ||  // 是 Fortran 连续数组
                    (order == NPY_CORDER &&     // C 顺序
                        PyArray_IS_C_CONTIGUOUS(self)) ||  // 是 C 连续数组
                    (order == NPY_FORTRANORDER &&  // Fortran 顺序
                        PyArray_IS_F_CONTIGUOUS(self))) &&  // 是 Fortran 连续数组
                (subok || PyArray_CheckExact(self))) {  // 如果是子类型允许或者精确匹配 self
        npy_intp view_offset;
        npy_intp is_safe = PyArray_SafeCast(dtype, PyArray_DESCR(self),
                                             &view_offset, NPY_NO_CASTING, 1);  // 安全地进行类型转换，并获取视图偏移量
        if (is_safe && (view_offset != NPY_MIN_INTP)) {  // 如果转换安全且视图偏移量不是最小整数值
            Py_DECREF(dtype);  // 释放 dtype 的引用计数
            Py_INCREF(self);   // 增加 self 的引用计数
            return (PyObject *)self;  // 返回 self 对象的 Python 对象
        }
    }

    // 如果无法将 self 转换为 dtype 所需的类型，则设置类型转换错误并返回空指针
    if (!PyArray_CanCastArrayTo(self, dtype, casting)) {
        PyErr_Clear();  // 清除当前异常状态
        npy_set_invalid_cast_error(
                PyArray_DESCR(self), dtype, casting, PyArray_NDIM(self) == 0);  // 设置无效类型转换错误
        Py_DECREF(dtype);  // 释放 dtype 的引用计数
        return NULL;
    }

    PyArrayObject *ret;  // 定义 PyArrayObject 结构体指针变量 ret

    /* This steals the reference to dtype */
    Py_INCREF(dtype);  // 增加 dtype 的引用计数
    ret = (PyArrayObject *)PyArray_NewLikeArray(
                                self, order, dtype, subok);  // 创建一个类似 self 的新数组 ret，使用 order、dtype 和 subok 参数
    if (ret == NULL) {  // 如果创建失败，则释放 dtype 的引用计数并返回空指针
        Py_DECREF(dtype);  // 释放 dtype 的引用计数
        return NULL;
    }

    // 减少维度数，再次删除子数组维度
    int out_ndim = PyArray_NDIM(ret);  // 获取 ret 的维度数
    PyArray_Descr *out_descr = PyArray_DESCR(ret);  // 获取 ret 的描述符
    if (out_ndim != PyArray_NDIM(self)) {  // 如果 ret 的维度数不等于 self 的维度数
        ((PyArrayObject_fields *)ret)->nd = PyArray_NDIM(self);  // 设置 ret 的维度数为 self 的维度数
        ((PyArrayObject_fields *)ret)->descr = dtype;  // 设置 ret 的描述符为 dtype
    }
    int success = PyArray_CopyInto(ret, self);  // 将 self 复制到 ret 中

    Py_DECREF(dtype);  // 释放 dtype 的引用计数
    ((PyArrayObject_fields *)ret)->nd = out_ndim;  // 恢复 ret 的维度数
    ((PyArrayObject_fields *)ret)->descr = out_descr;  // 恢复 ret 的描述符

    if (success < 0) {  // 如果复制失败，则释放 ret 并返回空指针
        Py_DECREF(ret);  // 释放 ret 的引用计数
        return NULL;
    }

    return (PyObject *)ret;  // 返回 ret 的 Python 对象
/* 默认子类型实现 */

static PyObject *
array_finalizearray(PyArrayObject *self, PyObject *obj)
{
    /* 返回 None 对象，表示无需特殊处理 */
    Py_RETURN_NONE;
}


static PyObject *
array_wraparray(PyArrayObject *self, PyObject *args)
{
    PyArrayObject *arr;
    PyObject *obj;

    /* 检查参数个数，只接受一个参数 */
    if (PyTuple_Size(args) < 1) {
        PyErr_SetString(PyExc_TypeError,
                        "only accepts 1 argument");
        return NULL;
    }
    obj = PyTuple_GET_ITEM(args, 0);
    if (obj == NULL) {
        return NULL;
    }
    /* 检查参数是否为 ndarray 对象 */
    if (!PyArray_Check(obj)) {
        PyErr_SetString(PyExc_TypeError,
                        "can only be called with ndarray object");
        return NULL;
    }
    arr = (PyArrayObject *)obj;

    /* 如果 self 和 arr 的类型不同 */
    if (Py_TYPE(self) != Py_TYPE(arr)) {
        /* 获取 arr 的描述符 */
        PyArray_Descr *dtype = PyArray_DESCR(arr);
        Py_INCREF(dtype);
        /* 创建一个新的 ndarray 对象，以 arr 作为基础 */
        return PyArray_NewFromDescrAndBase(
                Py_TYPE(self),
                dtype,
                PyArray_NDIM(arr),
                PyArray_DIMS(arr),
                PyArray_STRIDES(arr), PyArray_DATA(arr),
                PyArray_FLAGS(arr), (PyObject *)self, obj);
    }
    else {
        /*
         * 例如，当从 Python 调用时，类型可能已经正确。
         * 典型的 ufunc 路径之前通过 __array_prepare__ 到达此处。
         */
        Py_INCREF(arr);
        return (PyObject *)arr;
    }
}


static PyObject *
array_getarray(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
    PyArray_Descr *newtype = NULL;
    NPY_COPYMODE copy = NPY_COPY_IF_NEEDED;
    static char *kwlist[] = {"dtype", "copy", NULL};
    PyObject *ret;

    /* 解析参数，接受 dtype 和 copy 作为可选参数 */
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O&$O&:__array__", kwlist,
                                     PyArray_DescrConverter, &newtype,
                                     PyArray_CopyConverter, &copy)) {
        Py_XDECREF(newtype);
        return NULL;
    }

    /* 如果 self 不是精确的 PyArray_Type */
    if (!PyArray_CheckExact(self)) {
        PyArrayObject *new;

        /* 增加对 self 的描述符的引用计数 */
        Py_INCREF(PyArray_DESCR(self));
        /* 从描述符和 self 创建一个新的 ndarray 对象 */
        new = (PyArrayObject *)PyArray_NewFromDescrAndBase(
                &PyArray_Type,
                PyArray_DESCR(self),
                PyArray_NDIM(self),
                PyArray_DIMS(self),
                PyArray_STRIDES(self),
                PyArray_DATA(self),
                PyArray_FLAGS(self),
                NULL,
                (PyObject *)self
        );
        if (new == NULL) {
            return NULL;
        }
        self = new;
    }
    else {
        Py_INCREF(self);
    }

    /* 如果需要复制数组数据 */
    if (copy == NPY_COPY_ALWAYS) {
        /* 如果未提供 newtype，则使用 self 的描述符 */
        if (newtype == NULL) {
            newtype = PyArray_DESCR(self);
        }
        /* 将 self 转换为指定类型的 ndarray */
        ret = PyArray_CastToType(self, newtype, 0);
        Py_DECREF(self);
        return ret;
    } else { // copy == NPY_COPY_IF_NEEDED || copy == NPY_COPY_NEVER
        // 如果 copy 参数是 NPY_COPY_IF_NEEDED 或者 NPY_COPY_NEVER
        if (newtype == NULL || PyArray_EquivTypes(PyArray_DESCR(self), newtype)) {
            // 如果 newtype 为 NULL 或者 self 的类型等效于 newtype 的类型，返回 self 对象的新引用
            return (PyObject *)self;
        }
        if (copy == NPY_COPY_IF_NEEDED) {
            // 如果需要复制，将 self 强制转换为 newtype 类型的数组
            ret = PyArray_CastToType(self, newtype, 0);
            // 减少 self 的引用计数
            Py_DECREF(self);
            // 返回转换后的新数组对象
            return ret;
        } else { // copy == NPY_COPY_NEVER
            // 如果不允许复制，设置异常为 ValueError，指定错误消息为 npy_no_copy_err_msg
            PyErr_SetString(PyExc_ValueError, npy_no_copy_err_msg);
            // 减少 self 的引用计数
            Py_DECREF(self);
            // 返回空指针，表示出错
            return NULL;
        }
    }
/*
 * 检查输入和输出参数中是否有任何非默认的 __array_ufunc__ 方法。
 * 如果有，则返回 1；如果没有，则返回 0；如果发生错误，则返回 -1。
 *
 * 此函数的主要作用是帮助 ndarray.__array_ufunc__ 确定是否可以支持一个 ufunc
 * （只有当没有操作数有重写时才是这样）。因此，与 umath/override.c 中不同，
 * 实际的重写方法并不需要，一旦找到一个重写方法就可以停止查找。
 */
static int
any_array_ufunc_overrides(PyObject *args, PyObject *kwds)
{
    int i;
    int nin, nout;
    PyObject *out_kwd_obj;
    PyObject *fast;
    PyObject **in_objs, **out_objs, *where_obj;

    /* 检查输入参数 */
    nin = PyTuple_Size(args);
    if (nin < 0) {
        return -1;
    }
    fast = PySequence_Fast(args, "Could not convert object to sequence");
    if (fast == NULL) {
        return -1;
    }
    in_objs = PySequence_Fast_ITEMS(fast);
    for (i = 0; i < nin; ++i) {
        if (PyUFunc_HasOverride(in_objs[i])) {
            Py_DECREF(fast);
            return 1;
        }
    }
    Py_DECREF(fast);

    /* 如果没有关键字参数，直接返回 0 */
    if (kwds == NULL) {
        return 0;
    }

    /* 检查输出参数 */
    nout = PyUFuncOverride_GetOutObjects(kwds, &out_kwd_obj, &out_objs);
    if (nout < 0) {
        return -1;
    }
    for (i = 0; i < nout; i++) {
        if (PyUFunc_HasOverride(out_objs[i])) {
            Py_DECREF(out_kwd_obj);
            return 1;
        }
    }
    Py_DECREF(out_kwd_obj);

    /* 检查 where 参数是否存在 */
    where_obj = PyDict_GetItemWithError(kwds, npy_interned_str.where);
    if (where_obj == NULL) {
        if (PyErr_Occurred()) {
            return -1;
        }
    } else {
        if (PyUFunc_HasOverride(where_obj)){
            return 1;
        }
    }

    return 0;
}

/*
 * ndarray 对象的 __array_ufunc__ 方法的实现。
 */
NPY_NO_EXPORT PyObject *
array_ufunc(PyArrayObject *NPY_UNUSED(self), PyObject *args, PyObject *kwds)
{
    PyObject *ufunc, *method_name, *normal_args, *ufunc_method;
    PyObject *result = NULL;
    int has_override;

    assert(PyTuple_CheckExact(args));
    assert(kwds == NULL || PyDict_CheckExact(kwds));

    /* 至少需要两个参数才能调用 __array_ufunc__ */
    if (PyTuple_GET_SIZE(args) < 2) {
        PyErr_SetString(PyExc_TypeError,
                        "__array_ufunc__ requires at least 2 arguments");
        return NULL;
    }

    /* 提取普通参数（除了第一个和第二个参数以外的参数） */
    normal_args = PyTuple_GetSlice(args, 2, PyTuple_GET_SIZE(args));
    if (normal_args == NULL) {
        return NULL;
    }

    /* 检查是否有重写方法 */
    has_override = any_array_ufunc_overrides(normal_args, kwds);
    if (has_override < 0) {
        goto cleanup;
    }
    else if (has_override) {
        result = Py_NotImplemented;
        Py_INCREF(Py_NotImplemented);
        goto cleanup;
    }

    /* 获取第一个参数（ufunc 对象）和第二个参数（方法名） */
    ufunc = PyTuple_GET_ITEM(args, 0);
    method_name = PyTuple_GET_ITEM(args, 1);

    /*
     * TODO(?): 在稍后的某个时刻调用 UFunc 代码，
     * 因为这里的参数已经被标准化，我们不必再查找 __array_ufunc__。
     */
    // 获取ufunc对象的方法method_name，并赋值给ufunc_method变量
    ufunc_method = PyObject_GetAttr(ufunc, method_name);
    // 如果获取方法失败，跳转到cleanup标签执行清理操作
    if (ufunc_method == NULL) {
        goto cleanup;
    }
    // 调用ufunc_method表示的方法，传入normal_args和kwds参数，并将结果赋值给result变量
    result = PyObject_Call(ufunc_method, normal_args, kwds);
    // 减少ufunc_method对象的引用计数，可能会释放该对象
    Py_DECREF(ufunc_method);
    cleanup:
        Py_DECREF(normal_args);
        /* 对 normal_args 进行 DECREF，减少其引用计数 */

        /* ufunc 和 method_name 是 borrowed references，无需进行 DECREF */
        return result;
    }



static PyObject *
array_function(PyArrayObject *NPY_UNUSED(self), PyObject *c_args, PyObject *c_kwds)
{
    PyObject *func, *types, *args, *kwargs, *result;
    static char *kwlist[] = {"func", "types", "args", "kwargs", NULL};

    // 解析传入的参数列表 c_args 和 c_kwds
    if (!PyArg_ParseTupleAndKeywords(
            c_args, c_kwds, "OOOO:__array_function__", kwlist,
            &func, &types, &args, &kwargs)) {
        return NULL;
    }

    // 快速创建 types 的 Python 序列对象，如果失败则返回 NULL
    types = PySequence_Fast(
        types,
        "types argument to ndarray.__array_function__ must be iterable");
    if (types == NULL) {
        return NULL;
    }

    // 调用 array_function_method_impl 函数处理函数调用逻辑
    result = array_function_method_impl(func, types, args, kwargs);
    // 减少 types 对象的引用计数
    Py_DECREF(types);
    return result;
}



static PyObject *
array_copy(PyArrayObject *self,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    NPY_ORDER order = NPY_CORDER;
    NPY_PREPARE_ARGPARSER;

    // 解析参数，设置拷贝的顺序 order
    if (npy_parse_arguments("copy", args, len_args, kwnames,
            "|order", PyArray_OrderConverter, &order,
            NULL, NULL, NULL) < 0) {
        return NULL;
    }

    // 创建并返回一个新的数组拷贝对象
    return PyArray_NewCopy(self, order);
}



/* Separate from array_copy to make __copy__ preserve Fortran contiguity. */
static PyObject *
array_copy_keeporder(PyArrayObject *self, PyObject *args)
{
    // 解析参数，确认函数被正确调用
    if (!PyArg_ParseTuple(args, ":__copy__")) {
        return NULL;
    }
    // 创建并返回一个新的数组拷贝对象，保留 Fortran 的连续性顺序
    return PyArray_NewCopy(self, NPY_KEEPORDER);
}



#include <stdio.h>
static PyObject *
array_resize(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"refcheck", NULL};
    Py_ssize_t size = PyTuple_Size(args);
    int refcheck = 1;
    PyArray_Dims newshape;
    PyObject *ret, *obj;

    // 解析关键字参数，设置 refcheck 标志位
    if (!NpyArg_ParseKeywords(kwds, "|i", kwlist,  &refcheck)) {
        return NULL;
    }

    // 如果没有传入参数则返回 None
    if (size == 0) {
        Py_RETURN_NONE;
    }
    else if (size == 1) {
        obj = PyTuple_GET_ITEM(args, 0);
        // 如果参数是 None 则返回 None
        if (obj == Py_None) {
            Py_RETURN_NONE;
        }
        // 否则将 obj 赋值给 args，准备后续处理
        args = obj;
    }

    // 解析并验证参数 args，设置新的数组形状 newshape
    if (!PyArray_IntpConverter(args, &newshape)) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_TypeError, "invalid shape");
        }
        return NULL;
    }

    // 调整数组 self 到新的形状 newshape，并返回结果
    ret = PyArray_Resize(self, &newshape, refcheck, NPY_ANYORDER);
    npy_free_cache_dim_obj(newshape);
    if (ret == NULL) {
        return NULL;
    }
    // 减少返回结果的引用计数，返回 None
    Py_DECREF(ret);
    Py_RETURN_NONE;
}



static PyObject *
array_repeat(PyArrayObject *self, PyObject *args, PyObject *kwds) {
    PyObject *repeats;
    int axis = NPY_RAVEL_AXIS;
    static char *kwlist[] = {"repeats", "axis", NULL};

    // 解析参数并设置重复次数 repeats 和轴 axis
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O&:repeat", kwlist,
                                     &repeats,
                                     PyArray_AxisConverter, &axis)) {
        return NULL;
    }
    // 调用 PyArray_Repeat 函数进行数组重复操作，返回结果作为 PyArrayObject 对象
    return PyArray_Return((PyArrayObject *)PyArray_Repeat(self, repeats, axis));
}

static PyObject *
array_choose(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
    // 定义关键字列表
    static char *keywords[] = {"out", "mode", NULL};
    // 初始化变量
    PyObject *choices;
    PyArrayObject *out = NULL;
    NPY_CLIPMODE clipmode = NPY_RAISE;
    // 获取参数元组的长度
    Py_ssize_t n = PyTuple_Size(args);

    // 根据参数个数决定如何解析参数
    if (n <= 1) {
        // 如果参数个数少于等于1，解析单一参数
        if (!PyArg_ParseTuple(args, "O:choose", &choices)) {
            return NULL;
        }
    }
    else {
        // 否则直接使用参数元组作为选择参数
        choices = args;
    }

    // 解析关键字参数，设置输出对象和裁剪模式
    if (!NpyArg_ParseKeywords(kwds, "|O&O&", keywords,
                PyArray_OutputConverter, &out,
                PyArray_ClipmodeConverter, &clipmode)) {
        return NULL;
    }

    // 调用选择函数，返回结果对象
    PyObject *ret = PyArray_Choose(self, choices, out, clipmode);

    /* this matches the unpacking behavior of ufuncs */
    // 根据输出对象是否为空决定返回值类型
    if (out == NULL) {
        return PyArray_Return((PyArrayObject *)ret);
    }
    else {
        return ret;
    }
}

static PyObject *
array_sort(PyArrayObject *self,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    // 初始化变量
    int axis = -1;
    int val;
    NPY_SORTKIND sortkind = _NPY_SORT_UNDEFINED;
    PyObject *order = NULL;
    PyArray_Descr *saved = NULL;
    PyArray_Descr *newd;
    int stable = -1;
    NPY_PREPARE_ARGPARSER;

    // 解析函数参数
    if (npy_parse_arguments("sort", args, len_args, kwnames,
            "|axis", &PyArray_PythonPyIntFromInt, &axis,
            "|kind", &PyArray_SortkindConverter, &sortkind,
            "|order", NULL, &order,
            "$stable", &PyArray_OptionalBoolConverter, &stable,
            NULL, NULL, NULL) < 0) {
        return NULL;
    }

    // 处理特殊情况：当order参数为None时，将其设为NULL
    if (order == Py_None) {
        order = NULL;
    }

    // 如果指定了order参数，进行额外处理
    if (order != NULL) {
        PyObject *new_name;
        PyObject *_numpy_internal;
        saved = PyArray_DESCR(self);
        // 如果数组没有字段，不能指定order
        if (!PyDataType_HASFIELDS(saved)) {
            PyErr_SetString(PyExc_ValueError, "Cannot specify " \
                            "order when the array has no fields.");
            return NULL;
        }
        // 导入内部模块_numpy_internal
        _numpy_internal = PyImport_ImportModule("numpy._core._internal");
        if (_numpy_internal == NULL) {
            return NULL;
        }
        // 调用_numpy_internal模块的方法_newnames
        new_name = PyObject_CallMethod(_numpy_internal, "_newnames",
                                       "OO", saved, order);
        Py_DECREF(_numpy_internal);
        if (new_name == NULL) {
            return NULL;
        }
        // 创建新的描述符对象
        newd = PyArray_DescrNew(saved);
        if (newd == NULL) {
            Py_DECREF(new_name);
            return NULL;
        }
        // 替换描述符对象的字段名
        Py_DECREF(((_PyArray_LegacyDescr *)newd)->names);
        ((_PyArray_LegacyDescr *)newd)->names = new_name;
        // 更新数组的描述符对象
        ((PyArrayObject_fields *)self)->descr = newd;
    }

    // 检查sortkind和stable参数的合法性
    if (sortkind != _NPY_SORT_UNDEFINED && stable != -1) {
        PyErr_SetString(PyExc_ValueError,
            "`kind` and `stable` parameters can't be provided at "
            "the same time. Use only one of them.");
        return NULL;
    }
    else if ((sortkind == _NPY_SORT_UNDEFINED && stable == -1) || (stable == 0)) {
        // 如果未指定排序算法和稳定性，使用快速排序算法
        sortkind = NPY_QUICKSORT;
    }
    # 如果 stable 等于 1，则设置排序类型为稳定排序
    else if (stable == 1) {
        sortkind = NPY_STABLESORT;
    }

    # 调用 PyArray_Sort 函数对数组进行排序，并将排序结果保存在 val 变量中
    val = PyArray_Sort(self, axis, sortkind);
    
    # 如果 order 不为 NULL，则恢复数组的描述符 saved，并释放之前的描述符
    if (order != NULL) {
        Py_XDECREF(PyArray_DESCR(self));
        ((PyArrayObject_fields *)self)->descr = saved;
    }
    
    # 如果排序操作返回值小于 0，表示排序出错，直接返回 NULL
    if (val < 0) {
        return NULL;
    }
    
    # 返回 Py_None，表示成功执行排序操作
    Py_RETURN_NONE;
static PyObject *
array_partition(PyArrayObject *self,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    // 默认轴参数为-1，排序方式为NPY_INTROSELECT
    int axis=-1;
    int val;
    NPY_SELECTKIND sortkind = NPY_INTROSELECT;
    PyObject *order = NULL;
    PyArray_Descr *saved = NULL;
    PyArray_Descr *newd;
    PyArrayObject * ktharray;
    PyObject * kthobj;
    NPY_PREPARE_ARGPARSER;

    // 解析参数，支持kth、axis、kind、order参数，其中kth为必选参数
    if (npy_parse_arguments("partition", args, len_args, kwnames,
            "kth", NULL, &kthobj,
            "|axis", &PyArray_PythonPyIntFromInt, &axis,
            "|kind", &PyArray_SelectkindConverter, &sortkind,
            "|order", NULL, &order,
            NULL, NULL, NULL) < 0) {
        return NULL;
    }

    // 如果order为Py_None，则将其置为NULL
    if (order == Py_None) {
        order = NULL;
    }

    // 如果order不为NULL，处理字段排序
    if (order != NULL) {
        PyObject *new_name;
        PyObject *_numpy_internal;
        // 保存当前描述器
        saved = PyArray_DESCR(self);
        // 如果当前描述器没有字段，则抛出错误
        if (!PyDataType_HASFIELDS(saved)) {
            PyErr_SetString(PyExc_ValueError, "Cannot specify " \
                            "order when the array has no fields.");
            return NULL;
        }
        // 导入numpy._core._internal模块
        _numpy_internal = PyImport_ImportModule("numpy._core._internal");
        if (_numpy_internal == NULL) {
            return NULL;
        }
        // 调用_numpy_internal的_newnames方法生成新的字段排序
        new_name = PyObject_CallMethod(_numpy_internal, "_newnames",
                                       "OO", saved, order);
        Py_DECREF(_numpy_internal);
        if (new_name == NULL) {
            return NULL;
        }
        // 创建新的描述器
        newd = PyArray_DescrNew(saved);
        if (newd == NULL) {
            Py_DECREF(new_name);
            return NULL;
        }
        // 替换描述器的字段排序
        Py_DECREF(((_PyArray_LegacyDescr *)newd)->names);
        ((_PyArray_LegacyDescr *)newd)->names = new_name;
        ((PyArrayObject_fields *)self)->descr = newd;
    }

    // 将kthobj转换为PyArrayObject类型
    ktharray = (PyArrayObject *)PyArray_FromAny(kthobj, NULL, 0, 1,
                                                NPY_ARRAY_DEFAULT, NULL);
    if (ktharray == NULL)
        return NULL;

    // 调用PyArray_Partition函数进行数组分区操作
    val = PyArray_Partition(self, ktharray, axis, sortkind);
    Py_DECREF(ktharray);

    // 如果处理了字段排序，则恢复原始描述器
    if (order != NULL) {
        Py_XDECREF(PyArray_DESCR(self));
        ((PyArrayObject_fields *)self)->descr = saved;
    }
    // 如果分区操作失败，返回NULL
    if (val < 0) {
        return NULL;
    }
    // 返回None
    Py_RETURN_NONE;
}

static PyObject *
array_argsort(PyArrayObject *self,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    // 默认轴参数为-1，排序方式为_NPY_SORT_UNDEFINED
    int axis = -1;
    NPY_SORTKIND sortkind = _NPY_SORT_UNDEFINED;
    PyObject *order = NULL, *res;
    PyArray_Descr *newd, *saved=NULL;
    int stable = -1;
    NPY_PREPARE_ARGPARSER;

    // 解析参数，支持axis、kind、order、stable参数
    if (npy_parse_arguments("argsort", args, len_args, kwnames,
            "|axis", &PyArray_AxisConverter, &axis,
            "|kind", &PyArray_SortkindConverter, &sortkind,
            "|order", NULL, &order,
            "$stable", &PyArray_OptionalBoolConverter, &stable,
            NULL, NULL, NULL) < 0) {
        return NULL;
    }
    // 如果order为Py_None，则将其置为NULL
    if (order == Py_None) {
        order = NULL;
    }
    # 检查是否传入了排序顺序的参数
    if (order != NULL) {
        PyObject *new_name;
        PyObject *_numpy_internal;
        saved = PyArray_DESCR(self);
        // 如果数组没有字段，则无法指定顺序，返回错误
        if (!PyDataType_HASFIELDS(saved)) {
            PyErr_SetString(PyExc_ValueError, "Cannot specify "
                            "order when the array has no fields.");
            return NULL;
        }
        // 导入 numpy 内部模块 _numpy_internal
        _numpy_internal = PyImport_ImportModule("numpy._core._internal");
        // 导入失败则返回空指针
        if (_numpy_internal == NULL) {
            return NULL;
        }
        // 调用 _numpy_internal 模块的 _newnames 方法获取新的字段名
        new_name = PyObject_CallMethod(_numpy_internal, "_newnames",
                                       "OO", saved, order);
        Py_DECREF(_numpy_internal);
        // 如果调用失败则返回空指针
        if (new_name == NULL) {
            return NULL;
        }
        // 复制原始描述符 saved 并将其赋值给 newd
        newd = PyArray_DescrNew(saved);
        // 如果复制失败则释放 new_name 并返回空指针
        if (newd == NULL) {
            Py_DECREF(new_name);
            return NULL;
        }
        // 释放原始描述符 newd 的 names 成员
        Py_DECREF(((_PyArray_LegacyDescr *)newd)->names);
        // 将 new_name 设置为 newd 的 names 成员
        ((_PyArray_LegacyDescr *)newd)->names = new_name;
        // 将新的描述符 newd 设置为数组对象的描述符
        ((PyArrayObject_fields *)self)->descr = newd;
    }
    // 检查排序类型 sortkind 和稳定性 stable 参数的组合
    if (sortkind != _NPY_SORT_UNDEFINED && stable != -1) {
        // 如果同时提供了 `kind` 和 `stable` 参数，则返回错误
        PyErr_SetString(PyExc_ValueError,
            "`kind` and `stable` parameters can't be provided at "
            "the same time. Use only one of them.");
        return NULL;
    }
    else if ((sortkind == _NPY_SORT_UNDEFINED && stable == -1) || (stable == 0)) {
        // 如果未指定排序类型且稳定性为 -1 或者稳定性为 0，则使用快速排序
        sortkind = NPY_QUICKSORT;
    }
    else if (stable == 1) {
        // 如果稳定性为 1，则使用稳定排序
        sortkind = NPY_STABLESORT;
    }

    // 调用 PyArray_ArgSort 对数组进行排序并返回结果
    res = PyArray_ArgSort(self, axis, sortkind);
    // 如果传入了排序顺序的参数，则恢复原始描述符
    if (order != NULL) {
        // 释放当前数组对象的描述符
        Py_XDECREF(PyArray_DESCR(self));
        // 将原始描述符 saved 重新赋值给数组对象的描述符
        ((PyArrayObject_fields *)self)->descr = saved;
    }
    // 返回排序后的数组对象
    return PyArray_Return((PyArrayObject *)res);
# 定义一个静态函数，用于对数组进行分区操作，返回分区后的索引数组
static PyObject *
array_argpartition(PyArrayObject *self,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    # 设置默认的轴参数为-1，排序类型为NPY_INTROSELECT
    int axis = -1;
    NPY_SELECTKIND sortkind = NPY_INTROSELECT;
    PyObject *order = NULL, *res;
    PyArray_Descr *newd, *saved=NULL;
    PyObject * kthobj;
    PyArrayObject * ktharray;
    NPY_PREPARE_ARGPARSER;

    # 解析传入的参数
    if (npy_parse_arguments("argpartition", args, len_args, kwnames,
            "kth", NULL, &kthobj,
            "|axis", &PyArray_AxisConverter, &axis,
            "|kind", &PyArray_SelectkindConverter, &sortkind,
            "|order", NULL, &order,
            NULL, NULL, NULL) < 0) {
        return NULL;
    }
    # 处理特殊情况：如果'order'参数为None，则将其设置为NULL
    if (order == Py_None) {
        order = NULL;
    }
    # 如果'order'不为NULL，进行额外处理
    if (order != NULL) {
        PyObject *new_name;
        PyObject *_numpy_internal;
        saved = PyArray_DESCR(self);
        # 如果数组没有字段，则报错
        if (!PyDataType_HASFIELDS(saved)) {
            PyErr_SetString(PyExc_ValueError, "Cannot specify "
                            "order when the array has no fields.");
            return NULL;
        }
        # 导入并调用numpy._core._internal模块中的'_newnames'方法
        _numpy_internal = PyImport_ImportModule("numpy._core._internal");
        if (_numpy_internal == NULL) {
            return NULL;
        }
        # 调用'_newnames'方法，生成新的字段名
        new_name = PyObject_CallMethod(_numpy_internal, "_newnames",
                                       "OO", saved, order);
        Py_DECREF(_numpy_internal);
        if (new_name == NULL) {
            return NULL;
        }
        # 创建新的描述符对象
        newd = PyArray_DescrNew(saved);
        if (newd == NULL) {
            Py_DECREF(new_name);
            return NULL;
        }
        # 释放旧的字段名并设置新的字段名
        Py_DECREF(((_PyArray_LegacyDescr *)newd)->names);
        ((_PyArray_LegacyDescr *)newd)->names = new_name;
        ((PyArrayObject_fields *)self)->descr = newd;
    }

    # 将'kthobj'转换为PyArrayObject对象
    ktharray = (PyArrayObject *)PyArray_FromAny(kthobj, NULL, 0, 1,
                                                NPY_ARRAY_DEFAULT, NULL);
    if (ktharray == NULL)
        return NULL;

    # 调用PyArray_ArgPartition函数进行分区操作
    res = PyArray_ArgPartition(self, ktharray, axis, sortkind);
    Py_DECREF(ktharray);

    # 如果'order'不为NULL，还原数组的描述符对象
    if (order != NULL) {
        Py_XDECREF(PyArray_DESCR(self));
        ((PyArrayObject_fields *)self)->descr = saved;
    }
    # 返回PyArrayObject对象的引用
    return PyArray_Return((PyArrayObject *)res);
}

# 定义一个静态函数，用于在已排序数组中搜索元素的位置
static PyObject *
array_searchsorted(PyArrayObject *self,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    PyObject *keys;
    PyObject *sorter;
    NPY_SEARCHSIDE side = NPY_SEARCHLEFT;
    NPY_PREPARE_ARGPARSER;

    sorter = NULL;
    # 解析传入的参数
    if (npy_parse_arguments("searchsorted", args, len_args, kwnames,
            "v", NULL, &keys,
            "|side", &PyArray_SearchsideConverter, &side,
            "|sorter", NULL, &sorter,
            NULL, NULL, NULL) < 0) {
        return NULL;
    }
    # 处理特殊情况：如果'sorter'参数为None，则将其设置为NULL
    if (sorter == Py_None) {
        sorter = NULL;
    }
    # 调用PyArray_SearchSorted函数进行搜索，并返回结果的PyArrayObject对象
    return PyArray_Return((PyArrayObject *)PyArray_SearchSorted(self, keys, side, sorter));
}
_deepcopy_call(char *iptr, char *optr, PyArray_Descr *dtype,
               PyObject *deepcopy, PyObject *visit)
{
    // 检查 dtype 是否具有引用计数，若没有则返回 0
    if (!PyDataType_REFCHK(dtype)) {
        return 0;
    }
    // 如果 dtype 具有字段
    else if (PyDataType_HASFIELDS(dtype)) {
        PyObject *key, *value, *title = NULL;
        PyArray_Descr *new;
        int offset, res;
        Py_ssize_t pos = 0;
        // 遍历 dtype 的字段
        while (PyDict_Next(PyDataType_FIELDS(dtype), &pos, &key, &value)) {
            // 跳过标题字段
            if (NPY_TITLE_KEY(key, value)) {
                continue;
            }
            // 解析字段值，获取新的 dtype 和偏移量
            if (!PyArg_ParseTuple(value, "Oi|O", &new, &offset,
                                  &title)) {
                return -1;
            }
            // 递归调用 _deepcopy_call 处理字段数据
            res = _deepcopy_call(iptr + offset, optr + offset, new,
                                 deepcopy, visit);
            if (res < 0) {
                return -1;
            }
        }
    }
    // 如果 dtype 没有字段
    else {
        PyObject *itemp, *otemp;
        PyObject *res;
        // 从 iptr 和 optr 处复制对象引用
        memcpy(&itemp, iptr, sizeof(itemp));
        memcpy(&otemp, optr, sizeof(otemp));
        // 如果 itemp 为空，则使用 Py_None
        if (itemp == NULL) {
            itemp = Py_None;
        }
        // 增加对象的引用计数
        Py_INCREF(itemp);
        // 调用 deepcopy 处理该参数
        /* 调用 deepcopy 处理该参数 */
        res = PyObject_CallFunctionObjArgs(deepcopy, itemp, visit, NULL);
        Py_DECREF(itemp);
        if (res == NULL) {
            return -1;
        }
        // 减少原对象的引用计数，并将结果复制到 optr 处
        Py_XDECREF(otemp);
        memcpy(optr, &res, sizeof(res));
    }
    // 返回成功
    return 0;
}


static PyObject *
array_deepcopy(PyArrayObject *self, PyObject *args)
{
    PyArrayObject *copied_array;
    PyObject *visit;
    NpyIter *iter = NULL;
    NpyIter_IterNextFunc *iternext;
    char *data;
    char **dataptr;
    npy_intp *strideptr, *innersizeptr;
    npy_intp stride, count;
    PyObject *copy, *deepcopy;
    int deepcopy_res;

    // 解析输入参数，获取 visit 对象
    if (!PyArg_ParseTuple(args, "O:__deepcopy__", &visit)) {
        return NULL;
    }
    // 复制 self 对象，保持原有顺序
    copied_array = (PyArrayObject*) PyArray_NewCopy(self, NPY_KEEPORDER);
    if (copied_array == NULL) {
        return NULL;
    }

    // 如果 self 不包含引用计数，则直接返回复制后的数组
    if (!PyDataType_REFCHK(PyArray_DESCR(self))) {
        return (PyObject *)copied_array;
    }

    // 如果数组包含对象，需要进行深度复制
    copy = PyImport_ImportModule("copy");
    if (copy == NULL) {
        Py_DECREF(copied_array);
        return NULL;
    }
    // 获取 copy 模块的 deepcopy 函数
    deepcopy = PyObject_GetAttrString(copy, "deepcopy");
    Py_DECREF(copy);
    if (deepcopy == NULL) {
        goto error;
    }
    // 创建 NpyIter 对象以便迭代数组
    iter = NpyIter_New(copied_array,
                        NPY_ITER_READWRITE |
                        NPY_ITER_EXTERNAL_LOOP |
                        NPY_ITER_REFS_OK |
                        NPY_ITER_ZEROSIZE_OK,
                        NPY_KEEPORDER, NPY_NO_CASTING,
                        NULL);
    if (iter == NULL) {
        goto error;
    }
    # 检查迭代器的大小是否不为零
    if (NpyIter_GetIterSize(iter) != 0) {
        # 获取迭代器的下一个迭代函数
        iternext = NpyIter_GetIterNext(iter, NULL);
        # 如果迭代函数为空，跳转到错误处理
        if (iternext == NULL) {
            goto error;
        }

        # 获取数据指针数组
        dataptr = NpyIter_GetDataPtrArray(iter);
        # 获取内部步长数组
        strideptr = NpyIter_GetInnerStrideArray(iter);
        # 获取内部循环大小的指针
        innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);

        # 开始迭代
        do {
            # 获取当前数据指针
            data = *dataptr;
            # 获取当前步长
            stride = *strideptr;
            # 获取当前内部循环大小
            count = *innersizeptr;
            # 在当前内部循环中进行操作
            while (count--) {
                # 对数据进行深拷贝操作
                deepcopy_res = _deepcopy_call(data, data, PyArray_DESCR(copied_array),
                                                deepcopy, visit);
                # 如果深拷贝操作返回错误，跳转到错误处理
                if (deepcopy_res == -1) {
                    goto error;
                }

                # 更新数据指针以跳转到下一个元素
                data += stride;
            }
        } while (iternext(iter));  # 继续迭代直到迭代器结束

    }

    # 释放深拷贝对象的引用
    Py_DECREF(deepcopy);
    # 如果迭代器释放失败，则释放复制数组的引用并返回空
    if (!NpyIter_Deallocate(iter)) {
        Py_DECREF(copied_array);
        return NULL;
    }
    # 返回复制的数组对象
    return (PyObject *)copied_array;

  error:
    # 在错误处理中释放深拷贝对象和复制数组的引用，并释放迭代器
    Py_DECREF(deepcopy);
    Py_DECREF(copied_array);
    NpyIter_Deallocate(iter);
    return NULL;
/* Convert Array to flat list (using getitem) */
static PyObject *
_getlist_pkl(PyArrayObject *self)
{
    PyObject *theobject;
    PyArrayIterObject *iter = NULL;
    PyObject *list;
    PyArray_GetItemFunc *getitem;

    // 获取数组元素的获取函数
    getitem = PyDataType_GetArrFuncs(PyArray_DESCR(self))->getitem;
    // 创建数组迭代器对象
    iter = (PyArrayIterObject *)PyArray_IterNew((PyObject *)self);
    if (iter == NULL) {
        return NULL;
    }
    // 创建一个空列表对象
    list = PyList_New(iter->size);
    if (list == NULL) {
        Py_DECREF(iter);
        return NULL;
    }
    // 遍历数组并将每个元素添加到列表中
    while (iter->index < iter->size) {
        theobject = getitem(iter->dataptr, self);
        PyList_SET_ITEM(list, iter->index, theobject);
        PyArray_ITER_NEXT(iter);
    }
    Py_DECREF(iter);
    return list;
}

static int
_setlist_pkl(PyArrayObject *self, PyObject *list)
{
    PyObject *theobject;
    PyArrayIterObject *iter = NULL;
    PyArray_SetItemFunc *setitem;

    // 获取数组元素的设置函数
    setitem = PyDataType_GetArrFuncs(PyArray_DESCR(self))->setitem;
    // 创建数组迭代器对象
    iter = (PyArrayIterObject *)PyArray_IterNew((PyObject *)self);
    if (iter == NULL) {
        return -1;
    }
    // 遍历数组并从列表中设置每个元素
    while(iter->index < iter->size) {
        theobject = PyList_GET_ITEM(list, iter->index);
        setitem(theobject, iter->dataptr, self);
        PyArray_ITER_NEXT(iter);
    }
    Py_XDECREF(iter);
    return 0;
}


static PyObject *
array_reduce(PyArrayObject *self, PyObject *NPY_UNUSED(args))
{
    /* version number of this pickle type. Increment if we need to
       change the format. Be sure to handle the old versions in
       array_setstate. */
    // 定义当前 pickle 格式的版本号
    const int version = 1;
    PyObject *ret = NULL, *state = NULL, *obj = NULL, *mod = NULL;
    PyObject *mybool, *thestr = NULL;
    PyArray_Descr *descr;

    /* Return a tuple of (callable object, arguments, object's state) */
    /*  We will put everything in the object's state, so that on UnPickle
        it can use the string object as memory without a copy */

    // 创建一个包含三个元素的元组对象
    ret = PyTuple_New(3);
    if (ret == NULL) {
        return NULL;
    }
    // 导入 numpy._core._multiarray_umath 模块
    mod = PyImport_ImportModule("numpy._core._multiarray_umath");
    if (mod == NULL) {
        Py_DECREF(ret);
        return NULL;
    }
    // 获取 _reconstruct 函数对象并添加到元组中
    obj = PyObject_GetAttrString(mod, "_reconstruct");
    Py_DECREF(mod);
    PyTuple_SET_ITEM(ret, 0, obj);
    // 设置元组的第二个元素，包括类型对象、参数和一个字符
    PyTuple_SET_ITEM(ret, 1,
                     Py_BuildValue("ONc",
                                   (PyObject *)Py_TYPE(self),
                                   Py_BuildValue("(N)",
                                                 PyLong_FromLong(0)),
                                   /* dummy data-type */
                                   'b'));


注释：
    /* 现在填充对象的状态。这是一个包含5个参数的元组

       1) 一个整数，表示pickle版本。
       2) 一个元组，给出数组的形状。
       3) 一个PyArray_Descr对象（带有正确的字节顺序设置）。
       4) 一个npy_bool，表示是否按Fortran顺序存储。
       5) 一个Python对象，表示数据（可以是字符串、列表或任何用户定义的对象）。

       注意，因为Python没有描述一种直接将原始数据写入pickle的机制，这里首先执行了向字符串的复制。
       这个问题在协议5中已得到解决，其中序列化的是缓冲区而不是字符串。
    */

    state = PyTuple_New(5);
    if (state == NULL) {
        Py_DECREF(ret);
        return NULL;
    }
    PyTuple_SET_ITEM(state, 0, PyLong_FromLong(version));  // 设置pickle版本号
    PyTuple_SET_ITEM(state, 1, PyObject_GetAttrString((PyObject *)self,
                                                      "shape"));  // 获取对象的形状属性
    descr = PyArray_DESCR(self);  // 获取数组的描述符
    Py_INCREF(descr);
    PyTuple_SET_ITEM(state, 2, (PyObject *)descr);  // 设置描述符到元组
    mybool = (PyArray_ISFORTRAN(self) ? Py_True : Py_False);  // 检查数组是否按Fortran顺序存储
    Py_INCREF(mybool);
    PyTuple_SET_ITEM(state, 3, mybool);  // 设置布尔值（是否按Fortran顺序）到元组
    if (PyDataType_FLAGCHK(PyArray_DESCR(self), NPY_LIST_PICKLE)) {  // 检查数组是否具有NPY_LIST_PICKLE标志
        thestr = _getlist_pkl(self);  // 如果是列表类型，获取其pickle表示
    }
    else {
        thestr = PyArray_ToString(self, NPY_ANYORDER);  // 否则，将数组转换为字符串表示
    }
    if (thestr == NULL) {
        Py_DECREF(ret);
        Py_DECREF(state);
        return NULL;
    }
    PyTuple_SET_ITEM(state, 4, thestr);  // 设置数据对象的字符串表示到元组
    PyTuple_SET_ITEM(ret, 2, state);  // 将状态元组设置为返回元组的第三个元素
    return ret;  // 返回填充好的返回元组
    /* Closing brace for the array_reduce_ex_picklebuffer function */

    /* Declare variables to hold references to Python objects */
    PyObject *numeric_mod = NULL, *from_buffer_func = NULL;
    PyObject *pickle_module = NULL, *picklebuf_class = NULL;
    PyObject *picklebuf_args = NULL;
    PyObject *buffer = NULL, *transposed_array = NULL;
    PyArray_Descr *descr = NULL;
    char order;

    /* Obtain the descriptor of the NumPy array object */
    descr = PyArray_DESCR(self);

    /* Import the 'pickle' module */
    pickle_module = PyImport_ImportModule("pickle");
    if (pickle_module == NULL){
        return NULL;
    }

    /* Get the 'PickleBuffer' class from the 'pickle' module */
    picklebuf_class = PyObject_GetAttrString(pickle_module, "PickleBuffer");
    Py_DECREF(pickle_module);
    if (picklebuf_class == NULL) {
        return NULL;
    }

    /* Construct a PickleBuffer of the array */
    if (!PyArray_IS_C_CONTIGUOUS((PyArrayObject*) self) &&
         PyArray_IS_F_CONTIGUOUS((PyArrayObject*) self)) {
        /* Handle Fortran-contiguous arrays */
        order = 'F';
        /* Transpose the array to ensure C-contiguity */
        transposed_array = PyArray_Transpose((PyArrayObject*)self, NULL);
        /* Build arguments for PickleBuffer with transposed array */
        picklebuf_args = Py_BuildValue("(N)", transposed_array);
    }
    else {
        /* Handle C-contiguous arrays */
        order = 'C';
        /* Build arguments for PickleBuffer with original array */
        picklebuf_args = Py_BuildValue("(O)", self);
    }
    if (picklebuf_args == NULL) {
        Py_DECREF(picklebuf_class);
        return NULL;
    }

    /* Create a PickleBuffer instance */
    buffer = PyObject_CallObject(picklebuf_class, picklebuf_args);
    Py_DECREF(picklebuf_class);
    Py_DECREF(picklebuf_args);
    if (buffer == NULL) {
        /* Handle case where buffer creation fails by falling back to regular __reduce_ex__ */
        PyErr_Clear();
        return array_reduce_ex_regular(self, protocol);
    }

    /* Import the '_frombuffer' function from 'numpy._core.numeric' */
    numeric_mod = PyImport_ImportModule("numpy._core.numeric");
    if (numeric_mod == NULL) {
        Py_DECREF(buffer);
        return NULL;
    }

    /* Get the '_frombuffer' function */
    from_buffer_func = PyObject_GetAttrString(numeric_mod,
                                              "_frombuffer");
    Py_DECREF(numeric_mod);
    if (from_buffer_func == NULL) {
        Py_DECREF(buffer);
        return NULL;
    }
    # 使用 Py_BuildValue 函数构建一个 Python 对象，格式为 "N(NONN)"
    return Py_BuildValue("N(NONN)",
                         from_buffer_func, buffer, (PyObject *)descr,
                         # 获取 self 对象的 "shape" 属性，并将其作为参数添加到 Python 对象中
                         PyObject_GetAttrString((PyObject *)self, "shape"),
                         # 根据 order 字符串构建一个单字符的 PyUnicode 对象
                         PyUnicode_FromStringAndSize(&order, 1));
static PyObject *
array_setstate(PyArrayObject *self, PyObject *args)
{
    PyObject *shape;  // 存储数组形状的 Python 对象
    PyArray_Descr *typecode;  // 存储数组数据类型的描述符
    int version = 1;  // 序列化版本号，默认为 1
    int is_f_order;  // 指示数组是否是 Fortran（列优先）顺序的标志
    PyObject *rawdata = NULL;  // 原始数据的 Python 对象指针
    char *datastr;  // 数据的字符串表示
    Py_ssize_t len;  // 数据字符串的长度
    npy_intp dimensions[NPY_MAXDIMS];  // 存储数组各维度大小的数组
    int nd;  // 数组的维度数
    npy_intp nbytes;  // 数组总字节数
    int overflowed;  // 标志是否发生溢出

    PyArrayObject_fields *fa = (PyArrayObject_fields *)self;  // 将 self 强制转换为 PyArrayObject_fields 结构体指针

    /* This will free any memory associated with a and
       use the string in setstate as the (writeable) memory.
    */
    // 尝试解析输入参数 args，期望格式为 "(iO!O!iO):__setstate__"
    if (!PyArg_ParseTuple(args, "(iO!O!iO):__setstate__",
                            &version,
                            &PyTuple_Type, &shape,
                            &PyArrayDescr_Type, &typecode,
                            &is_f_order,
                            &rawdata)) {
        PyErr_Clear();  // 清除异常状态
        version = 0;  // 如果解析失败，将版本号设为 0
        // 尝试解析参数，格式为 "(O!O!iO):__setstate__"
        if (!PyArg_ParseTuple(args, "(O!O!iO):__setstate__",
                            &PyTuple_Type, &shape,
                            &PyArrayDescr_Type, &typecode,
                            &is_f_order,
                            &rawdata)) {
            return NULL;  // 如果再次解析失败，返回 NULL
        }
    }

    /* If we ever need another pickle format, increment the version
       number. But we should still be able to handle the old versions.
       We've only got one right now. */
    // 检查版本号是否为已知版本（0 或 1），否则报错
    if (version != 1 && version != 0) {
        PyErr_Format(PyExc_ValueError,
                     "can't handle version %d of numpy.ndarray pickle",
                     version);
        return NULL;  // 返回错误信息
    }

    /*
     * Reassigning fa->descr messes with the reallocation strategy,
     * since fa could be a 0-d or scalar, and then
     * PyDataMem_UserFREE will be confused
     */
    // 获取数组占用的字节数，如果为零则设为一
    size_t n_tofree = PyArray_NBYTES(self);
    if (n_tofree == 0) {
        n_tofree = 1;
    }
    Py_XDECREF(PyArray_DESCR(self));  // 释放原先的描述符
    fa->descr = typecode;  // 用新的描述符替换旧的描述符
    Py_INCREF(typecode);  // 增加描述符的引用计数
    nd = PyArray_IntpFromSequence(shape, dimensions, NPY_MAXDIMS);  // 从形状元组中提取数组维度信息
    # 如果维度数小于 0，返回 NULL
    if (nd < 0) {
        return NULL;
    }

    # 定义一个布尔变量 `empty`，初始化为 False
    npy_bool empty = NPY_FALSE;
    # 初始化 `nbytes` 为 1
    nbytes = 1;

    # 遍历维度数组 `dimensions`，计算总字节数 `nbytes`
    for (int i = 0; i < nd; i++) {
        # 如果某个维度小于 0，抛出类型错误并返回 NULL
        if (dimensions[i] < 0) {
            PyErr_SetString(PyExc_TypeError,
                    "impossible dimension while unpickling array");
            return NULL;
        }
        # 如果某个维度为 0，将 `empty` 置为 True
        if (dimensions[i] == 0) {
            empty = NPY_TRUE;
        }
        # 使用 `npy_mul_sizes_with_overflow` 计算 `nbytes` 和当前维度的乘积，检查是否溢出
        overflowed = npy_mul_sizes_with_overflow(
                &nbytes, nbytes, dimensions[i]);
        # 如果溢出，返回内存错误
        if (overflowed) {
            return PyErr_NoMemory();
        }
    }

    # 计算 `nbytes` 和数组每个元素的大小的乘积，检查是否溢出
    overflowed = npy_mul_sizes_with_overflow(
            &nbytes, nbytes, PyArray_ITEMSIZE(self));
    # 如果溢出，返回内存错误
    if (overflowed) {
        return PyErr_NoMemory();
    }

    # 如果数组为空（`empty` 为 True），将 `nbytes` 设为 0
    if (empty) {
        nbytes = 0;
    }

    # 如果 `typecode` 标志包含 `NPY_LIST_PICKLE`
    if (PyDataType_FLAGCHK(typecode, NPY_LIST_PICKLE)) {
        # 检查 `rawdata` 是否为列表，若不是，抛出类型错误并返回 NULL
        if (!PyList_Check(rawdata)) {
            PyErr_SetString(PyExc_TypeError,
                            "object pickle not returning list");
            return NULL;
        }
    }
    else {
        # 增加 `rawdata` 的引用计数，以防止被释放
        Py_INCREF(rawdata);

        /* 与 Python 2 NumPy pickles 的向后兼容性 */
        # 如果 `rawdata` 是 Unicode 对象，转换为 Latin1 字符串
        if (PyUnicode_Check(rawdata)) {
            PyObject *tmp;
            tmp = PyUnicode_AsLatin1String(rawdata);
            Py_DECREF(rawdata);
            rawdata = tmp;
            # 如果转换失败，抛出更详细的值错误信息并返回 NULL
            if (tmp == NULL) {
                PyErr_SetString(PyExc_ValueError,
                                ("Failed to encode latin1 string when unpickling a Numpy array. "
                                 "pickle.load(a, encoding='latin1') is assumed."));
                return NULL;
            }
        }

        # 检查 `rawdata` 是否为字节串，若不是，抛出类型错误并返回 NULL
        if (!PyBytes_Check(rawdata)) {
            PyErr_SetString(PyExc_TypeError,
                            "pickle not returning string");
            Py_DECREF(rawdata);
            return NULL;
        }

        # 获取字节串的指针和长度
        if (PyBytes_AsStringAndSize(rawdata, &datastr, &len) < 0) {
            Py_DECREF(rawdata);
            return NULL;
        }

        # 如果长度 `len` 不等于 `nbytes`，抛出值错误并返回 NULL
        if (len != nbytes) {
            PyErr_SetString(PyExc_ValueError,
                    "buffer size does not match array size");
            Py_DECREF(rawdata);
            return NULL;
        }
    }
    if ((PyArray_FLAGS(self) & NPY_ARRAY_OWNDATA)) {
        /*
         * 如果数组标志包含 NPY_ARRAY_OWNDATA 标志位
         * 分配永远不会为0，请参见 ctors.c 中的注释，第820行
         */
        PyObject *handler = PyArray_HANDLER(self);
        // 获取数组的内存处理器
        if (handler == NULL) {
            /* 如果没有找到内存处理器，则可能发生这种情况 */
            PyErr_SetString(PyExc_RuntimeError,
                            "no memory handler found but OWNDATA flag set");
            return NULL;
        }
        // 释放数组数据内存，并使用内存处理器进行处理
        PyDataMem_UserFREE(PyArray_DATA(self), n_tofree, handler);
        // 清除数组的 NPY_ARRAY_OWNDATA 标志位
        PyArray_CLEARFLAGS(self, NPY_ARRAY_OWNDATA);
    }
    // 释放数组的基础对象引用
    Py_XDECREF(PyArray_BASE(self));
    // 将 fa 结构体中的 base 指针设置为 NULL
    fa->base = NULL;

    // 清除数组的 NPY_ARRAY_WRITEBACKIFCOPY 标志位
    PyArray_CLEARFLAGS(self, NPY_ARRAY_WRITEBACKIFCOPY);

    // 如果数组的维度不为 NULL
    if (PyArray_DIMS(self) != NULL) {
        // 释放缓存中的维度数组
        npy_free_cache_dim_array(self);
        // 将 fa 结构体中的 dimensions 指针设置为 NULL
        fa->dimensions = NULL;
    }

    // 将 fa 结构体中的 flags 设置为 NPY_ARRAY_DEFAULT
    fa->flags = NPY_ARRAY_DEFAULT;

    // 将 fa 结构体中的 nd 设置为当前数组的维度数
    fa->nd = nd;

    // 如果数组的维度数大于0
    if (nd > 0) {
        // 分配并缓存数组的维度数组，大小为 2 * nd
        fa->dimensions = npy_alloc_cache_dim(2 * nd);
        // 如果分配失败，则返回内存错误异常
        if (fa->dimensions == NULL) {
            return PyErr_NoMemory();
        }
        // 设置 fa 结构体中的 strides 指针为数组的维度数组的第 nd 个元素
        fa->strides = PyArray_DIMS(self) + nd;
        // 如果 nd 大于0，则将 dimensions 数组的前 nd 个元素复制到数组的维度数组中
        if (nd) {
            memcpy(PyArray_DIMS(self), dimensions, sizeof(npy_intp)*nd);
        }
        // 根据数组的内存布局（是否按行或按列连续），填充数组的步长数组
        _array_fill_strides(PyArray_STRIDES(self), dimensions, nd,
                               PyArray_ITEMSIZE(self),
                               (is_f_order ? NPY_ARRAY_F_CONTIGUOUS :
                                             NPY_ARRAY_C_CONTIGUOUS),
                               &(fa->flags));
    }
    // 检查 typecode 是否包含 NPY_LIST_PICKLE 标志
    if (!PyDataType_FLAGCHK(typecode, NPY_LIST_PICKLE)) {
        // 检查是否需要进行字节交换
        int swap = PyArray_ISBYTESWAPPED(self);
        /* Bytes should always be considered immutable, but we just grab the
         * pointer if they are large, to save memory. */
        // 如果数组不是按字节对齐的，或者需要交换字节序，或者长度较小于等于 1000，则不直接使用原始数据指针
        if (!IsAligned(self) || swap || (len <= 1000)) {
            // 获取数组的字节大小
            npy_intp num = PyArray_NBYTES(self);
            if (num == 0) {
                num = 1;
            }
            /* Store the handler in case the default is modified */
            // 存储当前的内存处理器，以防默认值被修改
            Py_XDECREF(fa->mem_handler);
            fa->mem_handler = PyDataMem_GetHandler();
            if (fa->mem_handler == NULL) {
                // 如果获取内存处理器失败，清理资源并返回空值
                Py_CLEAR(fa->mem_handler);
                Py_DECREF(rawdata);
                return NULL;
            }
            // 使用用户分配内存函数创建新的数据空间
            fa->data = PyDataMem_UserNEW(num, PyArray_HANDLER(self));
            if (PyArray_DATA(self) == NULL) {
                // 如果分配内存失败，清理资源并返回内存错误
                Py_CLEAR(fa->mem_handler);
                Py_DECREF(rawdata);
                return PyErr_NoMemory();
            }
            if (swap) {
                /* byte-swap on pickle-read */
                // 在 pickle 读取时进行字节交换
                npy_intp numels = PyArray_SIZE(self);
                PyDataType_GetArrFuncs(PyArray_DESCR(self))->copyswapn(PyArray_DATA(self),
                                        PyArray_ITEMSIZE(self),
                                        datastr, PyArray_ITEMSIZE(self),
                                        numels, 1, self);
                // 如果数组不是扩展类型且没有元数据，则创建一个描述符
                if (!(PyArray_ISEXTENDED(self) ||
                      PyArray_DESCR(self)->metadata ||
                      PyDataType_C_METADATA(PyArray_DESCR(self)))) {
                    fa->descr = PyArray_DescrFromType(
                                    PyArray_DESCR(self)->type_num);
                }
                else {
                    // 否则根据 typecode 创建一个新的描述符
                    fa->descr = PyArray_DescrNew(typecode);
                    if (fa->descr == NULL) {
                        // 如果创建描述符失败，清理资源并返回空值
                        Py_CLEAR(fa->mem_handler);
                        Py_DECREF(rawdata);
                        return NULL;
                    }
                    // 调整字节顺序
                    if (PyArray_DESCR(self)->byteorder == NPY_BIG) {
                        PyArray_DESCR(self)->byteorder = NPY_LITTLE;
                    }
                    else if (PyArray_DESCR(self)->byteorder == NPY_LITTLE) {
                        PyArray_DESCR(self)->byteorder = NPY_BIG;
                    }
                }
                // 释放 typecode 对象
                Py_DECREF(typecode);
            }
            else {
                // 否则直接复制数据到数组内存中
                memcpy(PyArray_DATA(self), datastr, PyArray_NBYTES(self));
            }
            // 设置数组的 OWN_DATA 标志
            PyArray_ENABLEFLAGS(self, NPY_ARRAY_OWNDATA);
            fa->base = NULL;
            // 释放原始数据对象引用
            Py_DECREF(rawdata);
        }
        else {
            /* The handlers should never be called in this case */
            // 在这种情况下，不应调用处理器
            // 清空内存处理器对象引用
            Py_XDECREF(fa->mem_handler);
            fa->mem_handler = NULL;
            // 直接使用数据指针
            fa->data = datastr;
            // 设置数组的基本对象为 rawdata
            if (PyArray_SetBaseObject(self, rawdata) < 0) {
                // 设置失败时，释放 rawdata 引用并返回空值
                Py_DECREF(rawdata);
                return NULL;
            }
        }
    }
    # 如果条件不满足进入 else 分支，否则执行以下代码块
    else:
        # 获取数组的字节大小
        npy_intp num = PyArray_NBYTES(self);
        # 如果数组大小为 0，则设置为 1
        if (num == 0) {
            num = 1;
        }

        /* 存储函数以备默认处理程序修改时使用 */
        // 释放先前的内存处理器并获取当前的内存处理器
        Py_XDECREF(fa->mem_handler);
        fa->mem_handler = PyDataMem_GetHandler();
        // 如果获取内存处理器失败，则返回空指针
        if (fa->mem_handler == NULL) {
            return NULL;
        }
        // 使用用户定义的内存分配器分配内存，并将结果存储到 fa->data 中
        fa->data = PyDataMem_UserNEW(num, PyArray_HANDLER(self));
        // 如果数据指针为空，则清理内存处理器并返回内存错误异常
        if (PyArray_DATA(self) == NULL) {
            Py_CLEAR(fa->mem_handler);
            return PyErr_NoMemory();
        }
        // 如果数组描述对象需要初始化，则使用 0 填充数据
        if (PyDataType_FLAGCHK(PyArray_DESCR(self), NPY_NEEDS_INIT)) {
            memset(PyArray_DATA(self), 0, PyArray_NBYTES(self));
        }
        // 设置数组拥有数据的标志
        PyArray_ENABLEFLAGS(self, NPY_ARRAY_OWNDATA);
        // 将 fa->base 设置为 NULL
        fa->base = NULL;
        // 将原始数据设置到列表中，如果失败则返回空指针
        if (_setlist_pkl(self, rawdata) < 0) {
            return NULL;
        }
    }

    // 更新数组的标志
    PyArray_UpdateFlags(self, NPY_ARRAY_UPDATE_ALL);

    // 返回 None 对象
    Py_RETURN_NONE;
/*NUMPY_API*/
// 定义 PyArray_Dump 函数，用于将数组对象序列化到文件中
NPY_NO_EXPORT int
PyArray_Dump(PyObject *self, PyObject *file, int protocol)
{
    PyObject *ret;
    // 缓存导入 numpy._core._methods 模块中的 _dump 函数
    npy_cache_import("numpy._core._methods", "_dump",
                     &npy_thread_unsafe_state._dump);
    // 如果 _dump 函数未找到，则返回错误
    if (npy_thread_unsafe_state._dump == NULL) {
        return -1;
    }
    // 根据 protocol 的值选择合适的参数调用 _dump 函数，并获得返回对象
    if (protocol < 0) {
        ret = PyObject_CallFunction(
                npy_thread_unsafe_state._dump, "OO", self, file);
    }
    else {
        ret = PyObject_CallFunction(
                npy_thread_unsafe_state._dump, "OOi", self, file, protocol);
    }
    // 如果调用失败，则返回错误
    if (ret == NULL) {
        return -1;
    }
    // 释放返回对象的引用计数
    Py_DECREF(ret);
    // 返回成功状态
    return 0;
}

/*NUMPY_API*/
// 定义 PyArray_Dumps 函数，用于将数组对象序列化为 Python 字符串
NPY_NO_EXPORT PyObject *
PyArray_Dumps(PyObject *self, int protocol)
{
    // 缓存导入 numpy._core._methods 模块中的 _dumps 函数
    npy_cache_import("numpy._core._methods", "_dumps",
                     &npy_thread_unsafe_state._dumps);
    // 如果 _dumps 函数未找到，则返回空对象
    if (npy_thread_unsafe_state._dumps == NULL) {
        return NULL;
    }
    // 根据 protocol 的值选择合适的参数调用 _dumps 函数，并返回其结果对象
    if (protocol < 0) {
        return PyObject_CallFunction(npy_thread_unsafe_state._dumps, "O", self);
    }
    else {
        return PyObject_CallFunction(
                npy_thread_unsafe_state._dumps, "Oi", self, protocol);
    }
}

static PyObject *
// 定义 array_dump 函数，将数组对象的 _dump 方法转发给 numpy._core._methods 模块
array_dump(PyArrayObject *self,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    // 调用 NPY_FORWARD_NDARRAY_METHOD 宏，转发 _dump 方法
    NPY_FORWARD_NDARRAY_METHOD(_dump);
}

static PyObject *
// 定义 array_dumps 函数，将数组对象的 _dumps 方法转发给 numpy._core._methods 模块
array_dumps(PyArrayObject *self,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    // 调用 NPY_FORWARD_NDARRAY_METHOD 宏，转发 _dumps 方法
    NPY_FORWARD_NDARRAY_METHOD(_dumps);
}

static PyObject *
// 定义 array_sizeof 函数，计算数组对象占用的内存大小
array_sizeof(PyArrayObject *self, PyObject *NPY_UNUSED(args))
{
    // 计算对象基本大小加上维度和步长所需的内存大小
    Py_ssize_t nbytes = Py_TYPE(self)->tp_basicsize +
        PyArray_NDIM(self) * sizeof(npy_intp) * 2;
    // 如果数组拥有数据内存，则加上数据内存的大小
    if (PyArray_CHKFLAGS(self, NPY_ARRAY_OWNDATA)) {
        nbytes += PyArray_NBYTES(self);
    }
    // 返回内存大小的 Python 长整型对象
    return PyLong_FromSsize_t(nbytes);
}

static PyObject *
// 定义 array_transpose 函数，对数组对象进行转置操作
array_transpose(PyArrayObject *self, PyObject *args)
{
    PyObject *shape = Py_None;
    Py_ssize_t n = PyTuple_Size(args);
    PyArray_Dims permute;
    PyObject *ret;

    // 根据参数数量设置转置形状
    if (n > 1) {
        shape = args;
    }
    else if (n == 1) {
        shape = PyTuple_GET_ITEM(args, 0);
    }

    // 根据形状是否为 None 选择转置方法
    if (shape == Py_None) {
        ret = PyArray_Transpose(self, NULL);
    }
    else {
        // 将形状参数转换为数组维度对象
        if (!PyArray_IntpConverter(shape, &permute)) {
            return NULL;
        }
        // 执行带有指定维度对象的转置操作
        ret = PyArray_Transpose(self, &permute);
        npy_free_cache_dim_obj(permute);
    }

    // 返回转置结果对象
    return ret;
}

#define _CHKTYPENUM(typ) ((typ) ? (typ)->type_num : NPY_NOTYPE)

static PyObject *
// 定义 array_mean 函数，将数组对象的 _mean 方法转发给 numpy._core._methods 模块
array_mean(PyArrayObject *self,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    // 调用 NPY_FORWARD_NDARRAY_METHOD 宏，转发 _mean 方法
    NPY_FORWARD_NDARRAY_METHOD(_mean);
}

static PyObject *
// 定义 array_sum 函数，将数组对象的 _sum 方法转发给 numpy._core._methods 模块
array_sum(PyArrayObject *self,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    // 调用 NPY_FORWARD_NDARRAY_METHOD 宏，转发 _sum 方法
    NPY_FORWARD_NDARRAY_METHOD(_sum);
}

static PyObject *
// 定义 array_cumsum 函数，计算数组对象的累积和
array_cumsum(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
    int axis = NPY_RAVEL_AXIS;
    // ...
}
    # 定义一个指向数组描述符的指针，并初始化为 NULL
    PyArray_Descr *dtype = NULL;
    # 定义一个指向数组对象的指针，并初始化为 NULL
    PyArrayObject *out = NULL;
    # 定义一个整型变量 rtype，用于存储类型检查后的结果
    int rtype;
    # 静态字符数组，用于解析参数时指定关键字的名称
    static char *kwlist[] = {"axis", "dtype", "out", NULL};

    # 使用 PyArg_ParseTupleAndKeywords 函数解析传入的参数和关键字参数
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O&O&O&:cumsum", kwlist,
                                     PyArray_AxisConverter, &axis,
                                     PyArray_DescrConverter2, &dtype,
                                     PyArray_OutputConverter, &out)) {
        # 如果解析失败，则释放 dtype 并返回 NULL
        Py_XDECREF(dtype);
        return NULL;
    }

    # 调用 _CHKTYPENUM 函数，检查 dtype 的类型，并将结果存储在 rtype 中
    rtype = _CHKTYPENUM(dtype);
    # 释放 dtype，因为不再需要它
    Py_XDECREF(dtype);
    # 调用 PyArray_CumSum 函数执行累积和操作，并返回其结果
    return PyArray_CumSum(self, axis, rtype, out);
static PyObject *
array_prod(PyArrayObject *self,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    // 使用宏定义调用 _prod 方法，执行数组的乘积计算
    NPY_FORWARD_NDARRAY_METHOD(_prod);
}

static PyObject *
array_cumprod(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
    int axis = NPY_RAVEL_AXIS;  // 设置默认轴为展平（ravel）轴
    PyArray_Descr *dtype = NULL;  // 初始化 dtype 为空指针
    PyArrayObject *out = NULL;  // 初始化输出数组对象为空

    int rtype;
    static char *kwlist[] = {"axis", "dtype", "out", NULL};  // 定义关键字列表

    // 解析输入参数，支持 axis、dtype、out 作为关键字参数
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O&O&O&:cumprod", kwlist,
                                     PyArray_AxisConverter, &axis,
                                     PyArray_DescrConverter2, &dtype,
                                     PyArray_OutputConverter, &out)) {
        Py_XDECREF(dtype);  // 解析失败时释放 dtype
        return NULL;  // 返回空指针
    }

    rtype = _CHKTYPENUM(dtype);  // 获取 dtype 的类型编号
    Py_XDECREF(dtype);  // 释放 dtype
    return PyArray_CumProd(self, axis, rtype, out);  // 调用 NumPy 的累积乘积函数
}


static PyObject *
array_dot(PyArrayObject *self,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    PyObject *a = (PyObject *)self, *b, *o = NULL;  // 初始化变量 a（self）、b 和 o（输出对象）
    PyArrayObject *ret;  // 定义返回的数组对象 ret
    NPY_PREPARE_ARGPARSER;  // 准备参数解析

    // 解析参数列表，支持输入数组 b 和输出数组对象 o
    if (npy_parse_arguments("dot", args, len_args, kwnames,
            "b", NULL, &b,
            "|out", NULL, &o,
            NULL, NULL, NULL) < 0) {
        return NULL;  // 解析失败时返回空指针
    }

    if (o != NULL) {  // 如果输出对象 o 不为空
        if (o == Py_None) {  // 如果 o 是 None
            o = NULL;  // 将 o 设置为 NULL
        }
        else if (!PyArray_Check(o)) {  // 如果 o 不是数组对象
            PyErr_SetString(PyExc_TypeError,
                            "'out' must be an array");  // 抛出类型错误异常
            return NULL;  // 返回空指针
        }
    }
    // 调用 NumPy 的矩阵乘积函数，返回结果数组对象
    ret = (PyArrayObject *)PyArray_MatrixProduct2(a, b, (PyArrayObject *)o);
    return PyArray_Return(ret);  // 返回结果数组对象
}


static PyObject *
array_any(PyArrayObject *self,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    // 使用宏定义调用 _any 方法，执行数组的逻辑或操作
    NPY_FORWARD_NDARRAY_METHOD(_any);
}


static PyObject *
array_all(PyArrayObject *self,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    // 使用宏定义调用 _all 方法，执行数组的逻辑与操作
    NPY_FORWARD_NDARRAY_METHOD(_all);
}

static PyObject *
array_stddev(PyArrayObject *self,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    // 使用宏定义调用 _std 方法，计算数组的标准差
    NPY_FORWARD_NDARRAY_METHOD(_std);
}

static PyObject *
array_variance(PyArrayObject *self,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    // 使用宏定义调用 _var 方法，计算数组的方差
    NPY_FORWARD_NDARRAY_METHOD(_var);
}

static PyObject *
array_compress(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
    int axis = NPY_RAVEL_AXIS;  // 设置默认轴为展平（ravel）轴
    PyObject *condition;  // 定义条件对象
    PyArrayObject *out = NULL;  // 初始化输出数组对象为空
    static char *kwlist[] = {"condition", "axis", "out", NULL};  // 定义关键字列表

    // 解析输入参数，支持 condition、axis 和 out 作为关键字参数
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O&O&:compress", kwlist,
                                     &condition,
                                     PyArray_AxisConverter, &axis,
                                     PyArray_OutputConverter, &out)) {
        return NULL;  // 解析失败时返回空指针
    }

    // 调用 NumPy 的压缩函数，返回结果对象
    PyObject *ret = PyArray_Compress(self, condition, axis, out);

    /* this matches the unpacking behavior of ufuncs */
    // 这与 ufunc 的解包行为相匹配

    // 返回结果对象
    return ret;
}
    # 如果`
    # 检查指针变量 out 是否为 NULL
    if (out == NULL) {
        # 如果 out 为 NULL，则将 ret 强制转换为 PyArrayObject 类型并返回
        return PyArray_Return((PyArrayObject *)ret);
    }
    else {
        # 如果 out 不为 NULL，则直接返回 ret
        return ret;
    }
static PyObject *
array_nonzero(PyArrayObject *self, PyObject *args)
{
    // 解析函数参数，确保没有参数传入
    if (!PyArg_ParseTuple(args, "")) {
        return NULL;
    }
    // 调用NumPy库函数，返回非零元素的索引
    return PyArray_Nonzero(self);
}


static PyObject *
array_trace(PyArrayObject *self,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    // 设置默认参数和变量
    int axis1 = 0, axis2 = 1, offset = 0;
    PyArray_Descr *dtype = NULL;
    PyArrayObject *out = NULL;
    int rtype;
    NPY_PREPARE_ARGPARSER;

    // 解析函数的关键字参数
    if (npy_parse_arguments("trace", args, len_args, kwnames,
            "|offset", &PyArray_PythonPyIntFromInt, &offset,
            "|axis1", &PyArray_PythonPyIntFromInt, &axis1,
            "|axis2", &PyArray_PythonPyIntFromInt, &axis2,
            "|dtype", &PyArray_DescrConverter2, &dtype,
            "|out", &PyArray_OutputConverter, &out,
            NULL, NULL, NULL) < 0) {
        Py_XDECREF(dtype);
        return NULL;
    }

    // 获取数据类型对应的类型号码
    rtype = _CHKTYPENUM(dtype);
    Py_XDECREF(dtype);

    // 调用NumPy库函数计算数组的迹
    PyObject *ret = PyArray_Trace(self, offset, axis1, axis2, rtype, out);

    /* this matches the unpacking behavior of ufuncs */
    // 根据输出参数的有无返回结果
    if (out == NULL) {
        return PyArray_Return((PyArrayObject *)ret);
    }
    else {
        return ret;
    }
}

#undef _CHKTYPENUM


static PyObject *
array_clip(PyArrayObject *self,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    // 调用NumPy库中的_ndarray_api方法，实现数组的剪裁操作
    NPY_FORWARD_NDARRAY_METHOD(_clip);
}


static PyObject *
array_conjugate(PyArrayObject *self, PyObject *args)
{
    // 定义输出参数
    PyArrayObject *out = NULL;
    // 解析函数参数，允许一个可选的输出参数
    if (!PyArg_ParseTuple(args, "|O&:conjugate",
                          PyArray_OutputConverter,
                          &out)) {
        return NULL;
    }
    // 调用NumPy库函数，返回数组的共轭
    return PyArray_Conjugate(self, out);
}


static PyObject *
array_diagonal(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
    // 设置默认参数和变量
    int axis1 = 0, axis2 = 1, offset = 0;
    static char *kwlist[] = {"offset", "axis1", "axis2", NULL};
    PyArrayObject *ret;

    // 解析函数参数和关键字参数
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|iii:diagonal", kwlist,
                                     &offset,
                                     &axis1,
                                     &axis2)) {
        return NULL;
    }

    // 调用NumPy库函数，返回数组的对角线元素
    ret = (PyArrayObject *)PyArray_Diagonal(self, offset, axis1, axis2);
    return PyArray_Return(ret);
}


static PyObject *
array_flatten(PyArrayObject *self,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    // 设置默认参数和变量
    NPY_ORDER order = NPY_CORDER;
    NPY_PREPARE_ARGPARSER;

    // 解析函数参数和关键字参数
    if (npy_parse_arguments("flatten", args, len_args, kwnames,
            "|order", PyArray_OrderConverter, &order,
            NULL, NULL, NULL) < 0) {
        return NULL;
    }

    // 调用NumPy库函数，返回数组按照指定顺序展开后的结果
    return PyArray_Flatten(self, order);
}


static PyObject *
array_ravel(PyArrayObject *self,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    // 设置默认参数和变量
    NPY_ORDER order = NPY_CORDER;
    NPY_PREPARE_ARGPARSER;

    // 解析函数参数和关键字参数
    // 这里省略了npy_parse_arguments函数的调用，需要在此处注释说明其作用
    # 如果调用 npy_parse_arguments 函数解析参数时出错（如参数数量不符合预期），则返回空值对象
    if (npy_parse_arguments("ravel", args, len_args, kwnames,
            "|order", PyArray_OrderConverter, &order,
            NULL, NULL, NULL) < 0) {
        return NULL;
    }
    # 调用 PyArray_Ravel 函数，对当前对象进行拉平操作，并根据给定的顺序参数进行处理
    return PyArray_Ravel(self, order);
static PyObject *
array_round(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
    // 默认精度为0
    int decimals = 0;
    // 输出数组对象，默认为NULL
    PyArrayObject *out = NULL;
    // 关键字参数列表
    static char *kwlist[] = {"decimals", "out", NULL};

    // 解析参数和关键字参数
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|iO&:round", kwlist,
                                     &decimals,
                                     // 将 out 参数转换为 PyArrayObject 类型
                                     PyArray_OutputConverter, &out)) {
        return NULL;
    }

    // 调用 PyArray_Round 函数进行数组元素的四舍五入操作
    PyObject *ret = PyArray_Round(self, decimals, out);

    /* this matches the unpacking behavior of ufuncs */
    // 如果未提供输出数组 out，则返回 ret 的数组对象
    if (out == NULL) {
        return PyArray_Return((PyArrayObject *)ret);
    }
    // 否则直接返回 ret
    else {
        return ret;
    }
}

static PyObject *
array_setflags(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
    // 设置关键字参数列表
    static char *kwlist[] = {"write", "align", "uic", NULL};
    // 写入标志，默认为 Py_None
    PyObject *write_flag = Py_None;
    // 对齐标志，默认为 Py_None
    PyObject *align_flag = Py_None;
    // 写回复制标志，默认为 Py_None
    PyObject *uic = Py_None;
    // 记录当前数组的标志
    int flagback = PyArray_FLAGS(self);

    // 将数组对象强制转换为 PyArrayObject_fields 类型
    PyArrayObject_fields *fa = (PyArrayObject_fields *)self;

    // 解析参数和关键字参数
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOO:setflags", kwlist,
                                     &write_flag,
                                     &align_flag,
                                     &uic))
        return NULL;

    // 如果指定了对齐标志
    if (align_flag != Py_None) {
        // 判断 align_flag 是否为 False
        int isnot = PyObject_Not(align_flag);
        if (isnot == -1) {
            return NULL;
        }
        // 如果 align_flag 为 False，则清除 NPY_ARRAY_ALIGNED 标志位
        if (isnot) {
            PyArray_CLEARFLAGS(self, NPY_ARRAY_ALIGNED);
        }
        // 如果 align_flag 为 True，并且数组是对齐的，则启用 NPY_ARRAY_ALIGNED 标志位
        else if (IsAligned(self)) {
            PyArray_ENABLEFLAGS(self, NPY_ARRAY_ALIGNED);
        }
        // 如果 align_flag 为 True 但数组未对齐，则抛出 ValueError 异常
        else {
            PyErr_SetString(PyExc_ValueError,
                            "cannot set aligned flag of mis-"
                            "aligned array to True");
            return NULL;
        }
    }

    // 如果指定了写回复制标志
    if (uic != Py_None) {
        // 判断 uic 是否为 True
        int istrue = PyObject_IsTrue(uic);
        if (istrue == -1) {
            return NULL;
        }
        // 如果 uic 为 True，则恢复原始标志并抛出 ValueError 异常
        if (istrue) {
            fa->flags = flagback;
            PyErr_SetString(PyExc_ValueError,
                            "cannot set WRITEBACKIFCOPY "
                            "flag to True");
            return NULL;
        }
        // 如果 uic 为 False，则清除 NPY_ARRAY_WRITEBACKIFCOPY 标志位，并释放 fa->base
        else {
            PyArray_CLEARFLAGS(self, NPY_ARRAY_WRITEBACKIFCOPY);
            Py_XDECREF(fa->base);
            fa->base = NULL;
        }
    }
    // 检查写标志是否不是 None 对象
    if (write_flag != Py_None) {
        // 将写标志转换为整数值
        int istrue = PyObject_IsTrue(write_flag);
        // 检查转换是否失败
        if (istrue == -1) {
            return NULL;
        }
        // 如果写标志为真
        else if (istrue == 1) {
            // 如果数组可写
            if (_IsWriteable(self)) {
                /*
                 * _IsWritable (和 PyArray_UpdateFlags) 允许翻转这个标志，
                 * 尽管创建数组的 C-API 用户可能有充分的理由使其不可写，因此不建议。
                 */
                // 如果数组没有基础对象，并且不拥有数据，也不可写
                if ((PyArray_BASE(self) == NULL) &&
                            !PyArray_CHKFLAGS(self, NPY_ARRAY_OWNDATA) &&
                            !PyArray_CHKFLAGS(self, NPY_ARRAY_WRITEABLE)) {
                    /* 2017-05-03, NumPy 1.17.0 */
                    // 发出警告提示，表明此操作不推荐使用
                    if (DEPRECATE("making a non-writeable array writeable "
                                  "is deprecated for arrays without a base "
                                  "which do not own their data.") < 0) {
                        return NULL;
                    }
                }
                // 启用数组的可写标志
                PyArray_ENABLEFLAGS(self, NPY_ARRAY_WRITEABLE);
                // 清除数组的写入警告标志
                PyArray_CLEARFLAGS(self, NPY_ARRAY_WARN_ON_WRITE);
            }
            // 如果数组不可写
            else {
                // 恢复原始标志并设置错误信息
                fa->flags = flagback;
                PyErr_SetString(PyExc_ValueError,
                                "cannot set WRITEABLE "
                                "flag to True of this "
                                "array");
                return NULL;
            }
        }
        // 如果写标志为假
        else {
            // 清除数组的可写标志和写入警告标志
            PyArray_CLEARFLAGS(self, NPY_ARRAY_WRITEABLE);
            PyArray_CLEARFLAGS(self, NPY_ARRAY_WARN_ON_WRITE);
        }
    }
    // 返回 Python 的 None 对象
    Py_RETURN_NONE;
static PyObject *
array_complex(PyArrayObject *self, PyObject *NPY_UNUSED(args))
{
    PyArrayObject *arr;  // 定义一个 PyArrayObject 类型的指针变量 arr
    PyArray_Descr *dtype;  // 定义一个 PyArray_Descr 类型的指针变量 dtype
    PyObject *c;  // 定义一个 PyObject 类型的指针变量 c，用于存储复数对象

    if (check_is_convertible_to_scalar(self) < 0) {  // 检查 self 是否可以转换为标量，如果不行则返回 NULL
        return NULL;
    }

    dtype = PyArray_DescrFromType(NPY_CDOUBLE);  // 根据类型 NPY_CDOUBLE 创建一个 PyArray_Descr 对象，赋值给 dtype
    if (dtype == NULL) {  // 如果创建失败则返回 NULL
        return NULL;
    }

    if (!PyArray_CanCastArrayTo(self, dtype, NPY_SAME_KIND_CASTING) &&
            !(PyArray_TYPE(self) == NPY_OBJECT)) {
        PyObject *descr = (PyObject*)PyArray_DESCR(self);  // 获取 self 的描述符对象

        Py_DECREF(dtype);  // 减少 dtype 的引用计数
        PyErr_Format(PyExc_TypeError,
                "Unable to convert %R to complex", descr);  // 报错，指示无法将 descr 转换为复数
        return NULL;
    }

    if (PyArray_TYPE(self) == NPY_OBJECT) {
        /* let python try calling __complex__ on the object. */
        PyObject *args, *res;

        Py_DECREF(dtype);  // 减少 dtype 的引用计数
        args = Py_BuildValue("(O)", *((PyObject**)PyArray_DATA(self)));  // 构建一个包含 self 数据的参数元组 args
        if (args == NULL) {  // 如果构建失败则返回 NULL
            return NULL;
        }
        res = PyComplex_Type.tp_new(&PyComplex_Type, args, NULL);  // 调用 PyComplex_Type 的 tp_new 方法创建一个复数对象 res
        Py_DECREF(args);  // 减少 args 的引用计数
        return res;  // 返回创建的复数对象
    }

    arr = (PyArrayObject *)PyArray_CastToType(self, dtype, 0);  // 将 self 转换为指定 dtype 类型的数组对象，赋值给 arr
    if (arr == NULL) {  // 如果转换失败则返回 NULL
        return NULL;
    }
    c = PyComplex_FromCComplex(*((Py_complex*)PyArray_DATA(arr)));  // 根据 arr 中的数据创建一个复数对象 c
    Py_DECREF(arr);  // 减少 arr 的引用计数
    return c;  // 返回创建的复数对象
}

static PyObject *
array_class_getitem(PyObject *cls, PyObject *args)
{
    const Py_ssize_t args_len = PyTuple_Check(args) ? PyTuple_Size(args) : 1;  // 获取参数元组 args 的长度

    if ((args_len > 2) || (args_len == 0)) {  // 如果参数长度大于2或者为0，则返回错误信息
        return PyErr_Format(PyExc_TypeError,
                            "Too %s arguments for %s",
                            args_len > 2 ? "many" : "few",
                            ((PyTypeObject *)cls)->tp_name);  // 报错，指示参数数量错误
    }
    return Py_GenericAlias(cls, args);  // 返回类的泛型别名对象
}

static PyObject *
array_array_namespace(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"api_version", NULL};  // 定义关键字参数列表

    PyObject *array_api_version = Py_None;  // 初始化 array_api_version 为 Py_None

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|$O:__array_namespace__", kwlist,
                                     &array_api_version)) {  // 解析函数参数，如果失败则返回 NULL
        return NULL;
    }

    if (array_api_version != Py_None) {  // 如果 array_api_version 不是 Py_None
        if (!PyUnicode_Check(array_api_version))  // 如果 array_api_version 不是 Unicode 类型
        {
            PyErr_Format(PyExc_ValueError,
                "Only None and strings are allowed as the Array API version, "
                "but received: %S.", array_api_version);  // 报错，指示版本类型错误
            return NULL;
        } else if (PyUnicode_CompareWithASCIIString(array_api_version, "2021.12") != 0 &&
            PyUnicode_CompareWithASCIIString(array_api_version, "2022.12") != 0)
        {
            PyErr_Format(PyExc_ValueError,
                "Version \"%U\" of the Array API Standard is not supported.",
                array_api_version);  // 报错，指示不支持的版本
            return NULL;
        }
    }

    PyObject *numpy_module = PyImport_ImportModule("numpy");  // 导入 numpy 模块
    if (numpy_module == NULL){  // 如果导入失败则返回 NULL
        return NULL;
    }

    return numpy_module;  // 返回导入的 numpy 模块对象
}
/* 定义一个函数，用于将数组对象移动到指定设备上
   参数：
   - self: 数组对象
   - args: 参数元组
   - kwds: 关键字参数字典
   返回值：
   - 成功时返回移动后的数组对象，失败时返回 NULL
*/
array_to_device(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
    // 静态字符数组，用于定义关键字列表
    static char *kwlist[] = {"", "stream", NULL};
    // 默认设备为空字符串
    char *device = "";
    // 流对象，默认为 Py_None
    PyObject *stream = Py_None;

    // 解析参数元组和关键字参数字典，期望字符串和可选对象类型的参数
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s|$O:to_device", kwlist,
                                     &device,
                                     &stream)) {
        // 解析失败时返回 NULL
        return NULL;
    }

    // 如果 stream 不是 Py_None，抛出 ValueError 异常
    if (stream != Py_None) {
        PyErr_SetString(PyExc_ValueError,
                        "The stream argument in to_device() "
                        "is not supported");
        return NULL;
    }

    // 如果设备不是 "cpu"，抛出 ValueError 异常，表明不支持的设备
    if (strcmp(device, "cpu") != 0) {
        PyErr_Format(PyExc_ValueError,
                     "Unsupported device: %s.", device);
        return NULL;
    }

    // 增加数组对象的引用计数，返回其 PyObject 类型的指针
    Py_INCREF(self);
    return (PyObject *)self;
}

// 非导出的 PyMethodDef 数组，用于定义数组类型的方法
NPY_NO_EXPORT PyMethodDef array_methods[] = {

    /* for subtypes */
    // "__array__" 方法，用 array_getarray 处理，支持位置参数和关键字参数
    {"__array__",
        (PyCFunction)array_getarray,
        METH_VARARGS | METH_KEYWORDS, NULL},
    // "__array_finalize__" 方法，用 array_finalizearray 处理，仅支持一个参数
    {"__array_finalize__",
        (PyCFunction)array_finalizearray,
        METH_O, NULL},
    // "__array_wrap__" 方法，用 array_wraparray 处理，支持位置参数
    {"__array_wrap__",
        (PyCFunction)array_wraparray,
        METH_VARARGS, NULL},
    // "__array_ufunc__" 方法，用 array_ufunc 处理，支持位置参数和关键字参数
    {"__array_ufunc__",
        (PyCFunction)array_ufunc,
        METH_VARARGS | METH_KEYWORDS, NULL},
    // "__array_function__" 方法，用 array_function 处理，支持位置参数和关键字参数
    {"__array_function__",
        (PyCFunction)array_function,
        METH_VARARGS | METH_KEYWORDS, NULL},

    /* for the sys module */
    // "__sizeof__" 方法，用 array_sizeof 处理，不接受参数
    {"__sizeof__",
        (PyCFunction) array_sizeof,
        METH_NOARGS, NULL},

    /* for the copy module */
    // "__copy__" 方法，用 array_copy_keeporder 处理，支持位置参数
    {"__copy__",
        (PyCFunction)array_copy_keeporder,
        METH_VARARGS, NULL},
    // "__deepcopy__" 方法，用 array_deepcopy 处理，支持位置参数
    {"__deepcopy__",
        (PyCFunction)array_deepcopy,
        METH_VARARGS, NULL},

    /* for Pickling */
    // "__reduce__" 方法，用 array_reduce 处理，支持位置参数
    {"__reduce__",
        (PyCFunction) array_reduce,
        METH_VARARGS, NULL},
    // "__reduce_ex__" 方法，用 array_reduce_ex 处理，支持位置参数
    {"__reduce_ex__",
        (PyCFunction) array_reduce_ex,
        METH_VARARGS, NULL},
    // "__setstate__" 方法，用 array_setstate 处理，支持位置参数
    {"__setstate__",
        (PyCFunction) array_setstate,
        METH_VARARGS, NULL},
    // "dumps" 方法，用 array_dumps 处理，快速调用和关键字参数
    {"dumps",
        (PyCFunction) array_dumps,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    // "dump" 方法，用 array_dump 处理，快速调用和关键字参数
    {"dump",
        (PyCFunction) array_dump,
        METH_FASTCALL | METH_KEYWORDS, NULL},

    // "__complex__" 方法，用 array_complex 处理，支持位置参数
    {"__complex__",
        (PyCFunction) array_complex,
        METH_VARARGS, NULL},

    // "__format__" 方法，用 array_format 处理，支持位置参数
    {"__format__",
        (PyCFunction) array_format,
        METH_VARARGS, NULL},

    /* for typing; requires python >= 3.9 */
    // "__class_getitem__" 方法，用 array_class_getitem 处理，支持类和单个参数
    {"__class_getitem__",
        (PyCFunction)array_class_getitem,
        METH_CLASS | METH_O, NULL},

    /* Original and Extended methods added 2005 */
    // "all" 方法，用 array_all 处理，快速调用和关键字参数
    {"all",
        (PyCFunction)array_all,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    // "any" 方法，用 array_any 处理，快速调用和关键字参数
    {"any",
        (PyCFunction)array_any,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    // "argmax" 方法，用 array_argmax 处理，快速调用和关键字参数
    {"argmax",
        (PyCFunction)array_argmax,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    // "argmin" 方法，用 array_argmin 处理，快速调用和关键字参数
    {"argmin",
        (PyCFunction)array_argmin,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    {"argpartition",
        (PyCFunction)array_argpartition,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    # 注册名为 "argpartition" 的函数，实现为 array_argpartition，支持快速调用和关键字参数
    {"argsort",
        (PyCFunction)array_argsort,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    # 注册名为 "argsort" 的函数，实现为 array_argsort，支持快速调用和关键字参数
    {"astype",
        (PyCFunction)array_astype,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    # 注册名为 "astype" 的函数，实现为 array_astype，支持快速调用和关键字参数
    {"byteswap",
        (PyCFunction)array_byteswap,
        METH_VARARGS | METH_KEYWORDS, NULL},
    # 注册名为 "byteswap" 的函数，实现为 array_byteswap，支持变长参数和关键字参数
    {"choose",
        (PyCFunction)array_choose,
        METH_VARARGS | METH_KEYWORDS, NULL},
    # 注册名为 "choose" 的函数，实现为 array_choose，支持变长参数和关键字参数
    {"clip",
        (PyCFunction)array_clip,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    # 注册名为 "clip" 的函数，实现为 array_clip，支持快速调用和关键字参数
    {"compress",
        (PyCFunction)array_compress,
        METH_VARARGS | METH_KEYWORDS, NULL},
    # 注册名为 "compress" 的函数，实现为 array_compress，支持变长参数和关键字参数
    {"conj",
        (PyCFunction)array_conjugate,
        METH_VARARGS, NULL},
    # 注册名为 "conj" 的函数，实现为 array_conjugate，支持变长参数
    {"conjugate",
        (PyCFunction)array_conjugate,
        METH_VARARGS, NULL},
    # 注册名为 "conjugate" 的函数，实现为 array_conjugate，支持变长参数
    {"copy",
        (PyCFunction)array_copy,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    # 注册名为 "copy" 的函数，实现为 array_copy，支持快速调用和关键字参数
    {"cumprod",
        (PyCFunction)array_cumprod,
        METH_VARARGS | METH_KEYWORDS, NULL},
    # 注册名为 "cumprod" 的函数，实现为 array_cumprod，支持变长参数和关键字参数
    {"cumsum",
        (PyCFunction)array_cumsum,
        METH_VARARGS | METH_KEYWORDS, NULL},
    # 注册名为 "cumsum" 的函数，实现为 array_cumsum，支持变长参数和关键字参数
    {"diagonal",
        (PyCFunction)array_diagonal,
        METH_VARARGS | METH_KEYWORDS, NULL},
    # 注册名为 "diagonal" 的函数，实现为 array_diagonal，支持变长参数和关键字参数
    {"dot",
        (PyCFunction)array_dot,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    # 注册名为 "dot" 的函数，实现为 array_dot，支持快速调用和关键字参数
    {"fill",
        (PyCFunction)array_fill,
        METH_VARARGS, NULL},
    # 注册名为 "fill" 的函数，实现为 array_fill，支持变长参数
    {"flatten",
        (PyCFunction)array_flatten,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    # 注册名为 "flatten" 的函数，实现为 array_flatten，支持快速调用和关键字参数
    {"getfield",
        (PyCFunction)array_getfield,
        METH_VARARGS | METH_KEYWORDS, NULL},
    # 注册名为 "getfield" 的函数，实现为 array_getfield，支持变长参数和关键字参数
    {"item",
        (PyCFunction)array_toscalar,
        METH_VARARGS, NULL},
    # 注册名为 "item" 的函数，实现为 array_toscalar，支持变长参数
    {"max",
        (PyCFunction)array_max,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    # 注册名为 "max" 的函数，实现为 array_max，支持快速调用和关键字参数
    {"mean",
        (PyCFunction)array_mean,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    # 注册名为 "mean" 的函数，实现为 array_mean，支持快速调用和关键字参数
    {"min",
        (PyCFunction)array_min,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    # 注册名为 "min" 的函数，实现为 array_min，支持快速调用和关键字参数
    {"nonzero",
        (PyCFunction)array_nonzero,
        METH_VARARGS, NULL},
    # 注册名为 "nonzero" 的函数，实现为 array_nonzero，支持变长参数
    {"partition",
        (PyCFunction)array_partition,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    # 注册名为 "partition" 的函数，实现为 array_partition，支持快速调用和关键字参数
    {"prod",
        (PyCFunction)array_prod,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    # 注册名为 "prod" 的函数，实现为 array_prod，支持快速调用和关键字参数
    {"put",
        (PyCFunction)array_put,
        METH_VARARGS | METH_KEYWORDS, NULL},
    # 注册名为 "put" 的函数，实现为 array_put，支持变长参数和关键字参数
    {"ravel",
        (PyCFunction)array_ravel,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    # 注册名为 "ravel" 的函数，实现为 array_ravel，支持快速调用和关键字参数
    {"repeat",
        (PyCFunction)array_repeat,
        METH_VARARGS | METH_KEYWORDS, NULL},
    # 注册名为 "repeat" 的函数，实现为 array_repeat，支持变长参数和关键字参数
    {"reshape",
        (PyCFunction)array_reshape,
        METH_VARARGS | METH_KEYWORDS, NULL},
    # 注册名为 "reshape" 的函数，实现为 array_reshape，支持变长参数和关键字参数
    {"resize",
        (PyCFunction)array_resize,
        METH_VARARGS | METH_KEYWORDS, NULL},
    # 注册名为 "resize" 的函数，实现为 array_resize，支持变长参数和关键字参数
    {"round",
        (PyCFunction)array_round,
        METH_VARARGS | METH_KEYWORDS, NULL},
    # 注册名为 "round" 的函数，实现为 array_round，支持变长参数和关键字参数
    {"searchsorted",
        (PyCFunction)array_searchsorted,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    # 注册名为 "searchsorted" 的函数，实现为 array_searchsorted，支持快速调用和关键字参数
    {"setfield",
        (PyCFunction)array_setfield,
        METH_VARARGS | METH_KEYWORDS, NULL},
    # 注册名为 "setfield" 的函数，实现为 array_setfield，支持变长参数和关键字参数
    {"setflags",
        (PyCFunction)array_setflags,
        METH_VARARGS | METH_KEYWORDS, NULL},
    // 定义名为 "setflags" 的方法，其实现为 array_setflags，接受变长参数和关键字参数，无文档字符串

    {"sort",
        (PyCFunction)array_sort,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    // 定义名为 "sort" 的方法，其实现为 array_sort，接受快速调用参数和关键字参数，无文档字符串

    {"squeeze",
        (PyCFunction)array_squeeze,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    // 定义名为 "squeeze" 的方法，其实现为 array_squeeze，接受快速调用参数和关键字参数，无文档字符串

    {"std",
        (PyCFunction)array_stddev,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    // 定义名为 "std" 的方法，其实现为 array_stddev，接受快速调用参数和关键字参数，无文档字符串

    {"sum",
        (PyCFunction)array_sum,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    // 定义名为 "sum" 的方法，其实现为 array_sum，接受快速调用参数和关键字参数，无文档字符串

    {"swapaxes",
        (PyCFunction)array_swapaxes,
        METH_VARARGS, NULL},
    // 定义名为 "swapaxes" 的方法，其实现为 array_swapaxes，接受变长参数，无文档字符串

    {"take",
        (PyCFunction)array_take,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    // 定义名为 "take" 的方法，其实现为 array_take，接受快速调用参数和关键字参数，无文档字符串

    {"tobytes",
        (PyCFunction)array_tobytes,
        METH_VARARGS | METH_KEYWORDS, NULL},
    // 定义名为 "tobytes" 的方法，其实现为 array_tobytes，接受变长参数和关键字参数，无文档字符串

    {"tofile",
        (PyCFunction)array_tofile,
        METH_VARARGS | METH_KEYWORDS, NULL},
    // 定义名为 "tofile" 的方法，其实现为 array_tofile，接受变长参数和关键字参数，无文档字符串

    {"tolist",
        (PyCFunction)array_tolist,
        METH_VARARGS, NULL},
    // 定义名为 "tolist" 的方法，其实现为 array_tolist，接受变长参数，无文档字符串

    {"tostring",
        (PyCFunction)array_tostring,
        METH_VARARGS | METH_KEYWORDS, NULL},
    // 定义名为 "tostring" 的方法，其实现为 array_tostring，接受变长参数和关键字参数，无文档字符串

    {"trace",
        (PyCFunction)array_trace,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    // 定义名为 "trace" 的方法，其实现为 array_trace，接受快速调用参数和关键字参数，无文档字符串

    {"transpose",
        (PyCFunction)array_transpose,
        METH_VARARGS, NULL},
    // 定义名为 "transpose" 的方法，其实现为 array_transpose，接受变长参数，无文档字符串

    {"var",
        (PyCFunction)array_variance,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    // 定义名为 "var" 的方法，其实现为 array_variance，接受快速调用参数和关键字参数，无文档字符串

    {"view",
        (PyCFunction)array_view,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    // 定义名为 "view" 的方法，其实现为 array_view，接受快速调用参数和关键字参数，无文档字符串

    // 用于库之间的数据交换
    {"__dlpack__",
        (PyCFunction)array_dlpack,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    // 定义名为 "__dlpack__" 的方法，其实现为 array_dlpack，接受快速调用参数和关键字参数，无文档字符串

    {"__dlpack_device__",
        (PyCFunction)array_dlpack_device,
        METH_NOARGS, NULL},
    // 定义名为 "__dlpack_device__" 的方法，其实现为 array_dlpack_device，不接受参数，无文档字符串

    // 用于数组 API 兼容性
    {"__array_namespace__",
        (PyCFunction)array_array_namespace,
        METH_VARARGS | METH_KEYWORDS, NULL},
    // 定义名为 "__array_namespace__" 的方法，其实现为 array_array_namespace，接受变长参数和关键字参数，无文档字符串

    {"to_device",
        (PyCFunction)array_to_device,
        METH_VARARGS | METH_KEYWORDS, NULL},
    // 定义名为 "to_device" 的方法，其实现为 array_to_device，接受变长参数和关键字参数，无文档字符串

    {NULL, NULL, 0, NULL}           /* sentinel */
    // 结束方法列表的标志
};


注释：


// 这是一个空的代码块闭合符号。在某些编程语言（如C、C++、Java等）中，用于结束一个代码块的语法结构。
// 在此处，`;`表示空语句，仅仅是为了保持代码结构的完整性，但实际上没有任何操作或逻辑效果。
// 注意：该代码块可能位于某个条件语句、循环或函数定义的结束位置，但是在当前提供的片段中，它并没有包含具体的代码内容。
```