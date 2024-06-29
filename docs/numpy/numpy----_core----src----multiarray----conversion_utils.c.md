# `.\numpy\numpy\_core\src\multiarray\conversion_utils.c`

```py
/*
 * 定义 NPY_NO_DEPRECATED_API 并设置为 NPY_API_VERSION
 * 定义 _MULTIARRAYMODULE
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

/*
 * 引入必要的头文件和模块
 */
#include <Python.h>             // Python 标准头文件
#include <structmember.h>       // 结构成员相关的头文件

#include "numpy/arrayobject.h"      // NumPy 数组对象的头文件
#include "numpy/arrayscalars.h"     // NumPy 数组标量相关的头文件
#include "numpy/npy_math.h"         // NumPy 中的数学函数头文件

#include "npy_config.h"         // NumPy 配置文件

#include "common.h"             // 公共函数和宏定义的头文件
#include "arraytypes.h"         // 数组类型相关的头文件

#include "conversion_utils.h"   // 类型转换工具函数的头文件
#include "alloc.h"              // 内存分配相关的头文件
#include "npy_buffer.h"         // NumPy 缓冲区对象的头文件
#include "npy_static_data.h"    // NumPy 静态数据的头文件
#include "multiarraymodule.h"   // NumPy 多维数组模块的头文件

/*
 * 定义静态函数 PyArray_PyIntAsInt_ErrMsg
 * 用于将 PyObject 转换为 int，出错时打印错误消息
 */
static int
PyArray_PyIntAsInt_ErrMsg(PyObject *o, const char * msg) NPY_GCC_NONNULL(2);

/*
 * 定义静态函数 PyArray_PyIntAsIntp_ErrMsg
 * 用于将 PyObject 转换为 npy_intp，出错时打印错误消息
 */
static npy_intp
PyArray_PyIntAsIntp_ErrMsg(PyObject *o, const char * msg) NPY_GCC_NONNULL(2);

/****************************************************************
 * PyArg_ParseTuple 使用的转换函数
 ****************************************************************/

/*NUMPY_API
 *
 * PyArg_ParseTuple 中使用的转换函数，用于处理 O& 参数
 *
 * 如果传入的对象是数组类型，则直接返回该对象，并增加引用计数
 * 否则，尝试将对象转换为 NPY_ARRAY_CARRAY 类型的数组对象
 * 需要在使用 PyArray_Converter 后对返回的数组对象进行 DECREF
 */
NPY_NO_EXPORT int
PyArray_Converter(PyObject *object, PyObject **address)
{
    if (PyArray_Check(object)) {
        *address = object;
        Py_INCREF(object);
        return NPY_SUCCEED;
    }
    else {
        *address = PyArray_FROM_OF(object, NPY_ARRAY_CARRAY);
        if (*address == NULL) {
            return NPY_FAIL;
        }
        return NPY_SUCCEED;
    }
}

/*NUMPY_API
 * PyArg_ParseTuple 中使用的输出数组转换函数
 */
NPY_NO_EXPORT int
PyArray_OutputConverter(PyObject *object, PyArrayObject **address)
{
    if (object == NULL || object == Py_None) {
        *address = NULL;
        return NPY_SUCCEED;
    }
    if (PyArray_Check(object)) {
        *address = (PyArrayObject *)object;
        return NPY_SUCCEED;
    }
    else {
        PyErr_SetString(PyExc_TypeError,
                        "output must be an array");
        *address = NULL;
        return NPY_FAIL;
    }
}

/*
 * 将给定的值转换为整数，替代 PyArray_PyIntAsIntp，保留了旧版本的行为
 */
static inline npy_intp
dimension_from_scalar(PyObject *ob)
{
    npy_intp value = PyArray_PyIntAsIntp(ob);

    if (error_converting(value)) {
        if (PyErr_ExceptionMatches(PyExc_OverflowError)) {
            PyErr_SetString(PyExc_ValueError,
                    "Maximum allowed dimension exceeded");
        }
        return -1;
    }
    return value;
}
/*NUMPY_API
 * 从序列中获取 intp 类型的块
 *
 * 此函数接受一个 Python 序列对象，并分配和填充一个 intp 数组以转换后的值。
 *
 * 在使用完毕后记得释放指针 seq.ptr，使用 PyDimMem_FREE(seq.ptr)**
 */
NPY_NO_EXPORT int
PyArray_IntpConverter(PyObject *obj, PyArray_Dims *seq)
{
    seq->ptr = NULL;  // 初始化指针为 NULL
    seq->len = 0;     // 初始化长度为 0

    /*
     * 当下面的弃用过期后，删除 `if` 语句，并更新 PyArray_OptionalIntpConverter 的注释。
     */
    if (obj == Py_None) {
        /* Numpy 1.20, 2020-05-31 */
        // 如果 obj 是 None，则发出弃用警告，并返回成功
        if (DEPRECATE(
                "Passing None into shape arguments as an alias for () is "
                "deprecated.") < 0){
            return NPY_FAIL;
        }
        return NPY_SUCCEED;
    }

    PyObject *seq_obj = NULL;

    /*
     * 如果 obj 是标量，则尽快跳转到 dimension_from_scalar，跳过所有无用的计算。
     */
    if (!PyLong_CheckExact(obj) && PySequence_Check(obj)) {
        // 如果 obj 不是精确的长整型且是序列，则快速获取序列对象
        seq_obj = PySequence_Fast(obj,
               "expected a sequence of integers or a single integer.");
        if (seq_obj == NULL) {
            /* 继续尝试解析为单个整数。 */
            PyErr_Clear();
        }
    }

    if (seq_obj == NULL) {
        /*
         * obj 可能是标量（如果 dimension_from_scalar 没有失败的话，在此刻还未执行检查以验证此假设）。
         */
        seq->ptr = npy_alloc_cache_dim(1);  // 分配大小为 1 的缓存维度
        if (seq->ptr == NULL) {
            PyErr_NoMemory();  // 内存分配失败的错误处理
            return NPY_FAIL;
        }
        else {
            seq->len = 1;  // 设置长度为 1

            seq->ptr[0] = dimension_from_scalar(obj);  // 将标量转换为维度值
            if (error_converting(seq->ptr[0])) {
                /*
                 * 如果发生的错误是类型错误（无法将值转换为整数），则告知用户预期的序列或整数。
                 */
                if (PyErr_ExceptionMatches(PyExc_TypeError)) {
                    PyErr_Format(PyExc_TypeError,
                            "expected a sequence of integers or a single "
                            "integer, got '%.100R'", obj);
                }
                npy_free_cache_dim_obj(*seq);  // 释放分配的缓存维度对象
                seq->ptr = NULL;  // 将指针置为 NULL
                return NPY_FAIL;
            }
        }
    }
    else {
        /*
         * `obj` is a sequence converted to the `PySequence_Fast` in `seq_obj`
         * `len` stores the size of the sequence `seq_obj`
         */
        Py_ssize_t len = PySequence_Fast_GET_SIZE(seq_obj);
        
        // Check if the length of the sequence exceeds the maximum supported dimensions
        if (len > NPY_MAXDIMS) {
            PyErr_Format(PyExc_ValueError,
                    "maximum supported dimension for an ndarray "
                    "is currently %d, found %d", NPY_MAXDIMS, len);
            Py_DECREF(seq_obj);
            return NPY_FAIL;
        }
        
        // Allocate memory for `seq->ptr` if the sequence length is greater than 0
        if (len > 0) {
            seq->ptr = npy_alloc_cache_dim(len);
            if (seq->ptr == NULL) {
                PyErr_NoMemory();
                Py_DECREF(seq_obj);
                return NPY_FAIL;
            }
        }

        // Set `seq->len` to the length of the sequence
        seq->len = len;

        // Convert Python index sequence `seq_obj` to C array `seq->ptr` of type `npy_intp`
        int nd = PyArray_IntpFromIndexSequence(seq_obj,
                (npy_intp *)seq->ptr, len);
        Py_DECREF(seq_obj);

        // Check if conversion was successful and dimensions match
        if (nd == -1 || nd != len) {
            // Free allocated memory and set `seq->ptr` to NULL on failure
            npy_free_cache_dim_obj(*seq);
            seq->ptr = NULL;
            return NPY_FAIL;
        }
    }

    // Return success status
    return NPY_SUCCEED;
/*
 * Like PyArray_IntpConverter, but leaves `seq` untouched if `None` is passed
 * rather than treating `None` as `()`.
 */
NPY_NO_EXPORT int
PyArray_OptionalIntpConverter(PyObject *obj, PyArray_Dims *seq)
{
    // 检查是否传入了 None 对象
    if (obj == Py_None) {
        // 如果是 None，直接返回成功，不做任何操作
        return NPY_SUCCEED;
    }

    // 否则调用 PyArray_IntpConverter 处理传入的对象
    return PyArray_IntpConverter(obj, seq);
}

NPY_NO_EXPORT int
PyArray_CopyConverter(PyObject *obj, NPY_COPYMODE *copymode) {
    // 检查是否传入了 None 对象
    if (obj == Py_None) {
        // 如果是 None，设置复制模式为 NPY_COPY_IF_NEEDED，并返回成功
        *copymode = NPY_COPY_IF_NEEDED;
        return NPY_SUCCEED;
    }

    int int_copymode;

    // 检查传入对象的类型是否为 _CopyMode
    if ((PyObject *)Py_TYPE(obj) == npy_static_pydata._CopyMode) {
        // 获取 _CopyMode 对象的 "value" 属性值
        PyObject* mode_value = PyObject_GetAttrString(obj, "value");
        if (mode_value == NULL) {
            return NPY_FAIL;
        }

        // 将属性值转换为整数
        int_copymode = (int)PyLong_AsLong(mode_value);
        Py_DECREF(mode_value);
        // 检查转换是否出错
        if (error_converting(int_copymode)) {
            return NPY_FAIL;
        }
    }
    else if(PyUnicode_Check(obj)) {
        // 如果传入对象是字符串，则返回错误，不允许用于 'copy' 关键字
        PyErr_SetString(PyExc_ValueError,
                        "strings are not allowed for 'copy' keyword. "
                        "Use True/False/None instead.");
        return NPY_FAIL;
    }
    else {
        // 对于其他对象，尝试使用 PyArray_BoolConverter 转换为布尔值
        npy_bool bool_copymode;
        if (!PyArray_BoolConverter(obj, &bool_copymode)) {
            return NPY_FAIL;
        }
        int_copymode = (int)bool_copymode;
    }

    // 设置复制模式，并返回成功
    *copymode = (NPY_COPYMODE)int_copymode;
    return NPY_SUCCEED;
}

NPY_NO_EXPORT int
PyArray_AsTypeCopyConverter(PyObject *obj, NPY_ASTYPECOPYMODE *copymode)
{
    int int_copymode;

    // 检查传入对象的类型是否为 _CopyMode
    if ((PyObject *)Py_TYPE(obj) == npy_static_pydata._CopyMode) {
        // 不允许使用 _CopyMode 枚举类型，返回错误
        PyErr_SetString(PyExc_ValueError,
                        "_CopyMode enum is not allowed for astype function. "
                        "Use true/false instead.");
        return NPY_FAIL;
    }
    else {
        // 否则，尝试使用 PyArray_BoolConverter 转换为布尔值
        npy_bool bool_copymode;
        if (!PyArray_BoolConverter(obj, &bool_copymode)) {
            return NPY_FAIL;
        }
        int_copymode = (int)bool_copymode;
    }

    // 设置复制模式，并返回成功
    *copymode = (NPY_ASTYPECOPYMODE)int_copymode;
    return NPY_SUCCEED;
}

/*NUMPY_API
 * Get buffer chunk from object
 *
 * this function takes a Python object which exposes the (single-segment)
 * buffer interface and returns a pointer to the data segment
 *
 * You should increment the reference count by one of buf->base
 * if you will hang on to a reference
 *
 * You only get a borrowed reference to the object. Do not free the
 * memory...
 */
NPY_NO_EXPORT int
PyArray_BufferConverter(PyObject *obj, PyArray_Chunk *buf)
{
    Py_buffer view;

    buf->ptr = NULL;
    buf->flags = NPY_ARRAY_BEHAVED;
    buf->base = NULL;
    // 检查是否传入了 None 对象
    if (obj == Py_None) {
        // 如果是 None，直接返回成功，不做任何操作
        return NPY_SUCCEED;
    }

    // 如果传入对象不是 None，则继续处理
    // (此处为示例中代码截断，未能提供完整的函数内容)
}
    /*
     * 如果无法获取对象的缓冲区视图，清除错误并标记缓冲区为不可写。
     * 然后再次尝试获取缓冲区视图，这次只要求连续且简单的缓冲区。
     */
    if (PyObject_GetBuffer(obj, &view,
                PyBUF_ANY_CONTIGUOUS|PyBUF_WRITABLE|PyBUF_SIMPLE) != 0) {
        PyErr_Clear();
        buf->flags &= ~NPY_ARRAY_WRITEABLE;
        if (PyObject_GetBuffer(obj, &view,
                PyBUF_ANY_CONTIGUOUS|PyBUF_SIMPLE) != 0) {
            return NPY_FAIL;
        }
    }

    /*
     * 将缓冲区视图的指针和长度分配给 buf 结构体。
     */
    buf->ptr = view.buf;
    buf->len = (npy_intp) view.len;

    /*
     * 在 Python 3 中，此段代码替换了被弃用的 PyObject_AsWriteBuffer 和
     * PyObject_AsReadBuffer 函数。这些函数释放了缓冲区。由提供缓冲区的对象
     * 负责在释放后保证缓冲区的有效性。
     */
    PyBuffer_Release(&view);

    /* 
     * 如果对象是内存视图类型，则将 buf 的 base 指向其基础对象。
     */
    if (PyMemoryView_Check(obj)) {
        buf->base = PyMemoryView_GET_BASE(obj);
    }
    /*
     * 如果 buf 的 base 仍然为空，则将其设置为当前对象。
     */
    if (buf->base == NULL) {
        buf->base = obj;
    }

    /*
     * 返回操作成功的标志。
     */
    return NPY_SUCCEED;
/*NUMPY_API
 * Get axis from an object (possibly None) -- a converter function,
 *
 * See also PyArray_ConvertMultiAxis, which also handles a tuple of axes.
 */
NPY_NO_EXPORT int
PyArray_AxisConverter(PyObject *obj, int *axis)
{
    // 如果对象是 None，则设置 axis 为 NPY_RAVEL_AXIS
    if (obj == Py_None) {
        *axis = NPY_RAVEL_AXIS;
    }
    // 否则，尝试将对象转换为整数作为 axis
    else {
        *axis = PyArray_PyIntAsInt_ErrMsg(obj,
                               "an integer is required for the axis");
        // 如果转换过程中出现错误，返回失败
        if (error_converting(*axis)) {
            return NPY_FAIL;
        }
    }
    // 返回成功
    return NPY_SUCCEED;
}

/*
 * Converts an axis parameter into an ndim-length C-array of
 * boolean flags, True for each axis specified.
 *
 * If obj is None or NULL, everything is set to True. If obj is a tuple,
 * each axis within the tuple is set to True. If obj is an integer,
 * just that axis is set to True.
 */
NPY_NO_EXPORT int
PyArray_ConvertMultiAxis(PyObject *axis_in, int ndim, npy_bool *out_axis_flags)
{
    /* None means all of the axes */
    // 如果 axis_in 是 None 或 NULL，则将所有轴设置为 True
    if (axis_in == Py_None || axis_in == NULL) {
        memset(out_axis_flags, 1, ndim);
        return NPY_SUCCEED;
    }
    /* A tuple of which axes */
    // 如果 axis_in 是一个元组，则根据元组中的每个轴设置对应位置为 True
    else if (PyTuple_Check(axis_in)) {
        int i, naxes;

        memset(out_axis_flags, 0, ndim);

        naxes = PyTuple_Size(axis_in);
        if (naxes < 0) {
            return NPY_FAIL;
        }
        // 遍历元组中的每个元素，将对应的轴设置为 True
        for (i = 0; i < naxes; ++i) {
            PyObject *tmp = PyTuple_GET_ITEM(axis_in, i);
            // 尝试将元组元素转换为整数轴
            int axis = PyArray_PyIntAsInt_ErrMsg(tmp,
                          "integers are required for the axis tuple elements");
            // 如果转换过程中出现错误，返回失败
            if (error_converting(axis)) {
                return NPY_FAIL;
            }
            // 检查和调整轴的有效性
            if (check_and_adjust_axis(&axis, ndim) < 0) {
                return NPY_FAIL;
            }
            // 检查是否有重复的轴值
            if (out_axis_flags[axis]) {
                PyErr_SetString(PyExc_ValueError,
                        "duplicate value in 'axis'");
                return NPY_FAIL;
            }
            // 将对应轴位置设置为 True
            out_axis_flags[axis] = 1;
        }

        return NPY_SUCCEED;
    }
    /* Try to interpret axis as an integer */
    // 否则，尝试将 axis_in 解释为一个整数轴
    else {
        int axis;

        memset(out_axis_flags, 0, ndim);

        // 尝试将 axis_in 转换为整数轴
        axis = PyArray_PyIntAsInt_ErrMsg(axis_in,
                                   "an integer is required for the axis");

        // 如果转换过程中出现错误，返回失败
        if (error_converting(axis)) {
            return NPY_FAIL;
        }
        /*
         * Special case letting axis={-1,0} slip through for scalars,
         * for backwards compatibility reasons.
         */
        // 对于零维数组（标量），允许 axis 取值为 {-1, 0}，这是为了向后兼容
        if (ndim == 0 && (axis == 0 || axis == -1)) {
            return NPY_SUCCEED;
        }

        // 检查和调整轴的有效性
        if (check_and_adjust_axis(&axis, ndim) < 0) {
            return NPY_FAIL;
        }

        // 将对应轴位置设置为 True
        out_axis_flags[axis] = 1;

        return NPY_SUCCEED;
    }
}

/*NUMPY_API
 * Convert an object to true / false
 */
NPY_NO_EXPORT int
PyArray_BoolConverter(PyObject *object, npy_bool *val)
{
    // 将对象转换为布尔值
    if (PyObject_IsTrue(object)) {
        *val = NPY_TRUE;
    }
    else {
        *val = NPY_FALSE;
    }

        *val = NPY_FALSE;
    }
    // 返回成功
    return NPY_SUCCEED;
}
    // 检查是否有 Python 异常发生
    if (PyErr_Occurred()) {
        // 如果有异常发生，则返回失败标志
        return NPY_FAIL;
    }
    // 如果没有异常发生，则返回成功标志
    return NPY_SUCCEED;
}

/*
 * Optionally convert an object to true / false
 */
NPY_NO_EXPORT int
PyArray_OptionalBoolConverter(PyObject *object, int *val)
{
    /* Leave the desired default from the caller for Py_None */
    // 如果对象是 Py_None，则使用调用者指定的默认值
    if (object == Py_None) {
        return NPY_SUCCEED;
    }
    // 如果对象能被解释为 True，则将 *val 设置为 1
    if (PyObject_IsTrue(object)) {
        *val = 1;
    }
    else {
        // 否则将 *val 设置为 0
        *val = 0;
    }
    // 如果出现错误，返回 NPY_FAIL
    if (PyErr_Occurred()) {
        return NPY_FAIL;
    }
    // 操作成功，返回 NPY_SUCCEED
    return NPY_SUCCEED;
}

static int
string_converter_helper(
    PyObject *object,
    void *out,
    int (*str_func)(char const*, Py_ssize_t, void*),
    char const *name,
    char const *message)
{
    /* allow bytes for compatibility */
    // 允许 bytes 类型以保持兼容性
    PyObject *str_object = NULL;
    // 如果对象是 bytes 类型，则转换为 Unicode
    if (PyBytes_Check(object)) {
        str_object = PyUnicode_FromEncodedObject(object, NULL, NULL);
        // 转换失败，抛出 ValueError 异常
        if (str_object == NULL) {
            PyErr_Format(PyExc_ValueError,
                "%s %s (got %R)", name, message, object);
            return NPY_FAIL;
        }
    }
    // 如果对象是 Unicode 类型，直接使用
    else if (PyUnicode_Check(object)) {
        str_object = object;
        Py_INCREF(str_object);
    }
    else {
        // 其它类型抛出 TypeError 异常
        PyErr_Format(PyExc_TypeError,
            "%s must be str, not %s", name, Py_TYPE(object)->tp_name);
        return NPY_FAIL;
    }

    Py_ssize_t length;
    // 将 Unicode 对象转换为 UTF-8 编码的 C 字符串
    char const *str = PyUnicode_AsUTF8AndSize(str_object, &length);
    if (str == NULL) {
        Py_DECREF(str_object);
        return NPY_FAIL;
    }

    // 调用指定的 str_func 处理字符串
    int ret = str_func(str, length, out);
    Py_DECREF(str_object);
    // 如果 str_func 返回负数且未设置异常，则抛出 ValueError 异常
    if (ret < 0) {
        if (!PyErr_Occurred()) {
            PyErr_Format(PyExc_ValueError,
                "%s %s (got %R)", name, message, object);
        }
        return NPY_FAIL;
    }
    // 操作成功，返回 NPY_SUCCEED
    return NPY_SUCCEED;
}

static int byteorder_parser(char const *str, Py_ssize_t length, void *data)
{
    char *endian = (char *)data;

    // 如果字符串长度小于 1，返回 -1
    if (length < 1) {
        return -1;
    }
    // 解析字节顺序字符
    else if (str[0] == NPY_BIG || str[0] == NPY_LITTLE ||
             str[0] == NPY_NATIVE || str[0] == NPY_IGNORE) {
        *endian = str[0];
        return 0;
    }
    else if (str[0] == 'b' || str[0] == 'B') {
        *endian = NPY_BIG;
        return 0;
    }
    else if (str[0] == 'l' || str[0] == 'L') {
        *endian = NPY_LITTLE;
        return 0;
    }
    else if (str[0] == 'n' || str[0] == 'N') {
        *endian = NPY_NATIVE;
        return 0;
    }
    else if (str[0] == 'i' || str[0] == 'I') {
        *endian = NPY_IGNORE;
        return 0;
    }
    else if (str[0] == 's' || str[0] == 'S') {
        *endian = NPY_SWAP;
        return 0;
    }
    else {
        // 未识别的字符，返回 -1
        return -1;
    }
}

/*NUMPY_API
 * Convert object to endian
 */
NPY_NO_EXPORT int
PyArray_ByteorderConverter(PyObject *obj, char *endian)
{
    // 调用通用字符串转换函数处理字节顺序
    return string_converter_helper(
        obj, (void *)endian, byteorder_parser, "byteorder", "not recognized");
}

static int sortkind_parser(char const *str, Py_ssize_t length, void *data)
{
    # 将 void 指针类型的 data 强制转换为 NPY_SORTKIND 指针类型，并赋值给 sortkind
    NPY_SORTKIND *sortkind = (NPY_SORTKIND *)data;

    # 如果 length 小于 1，返回错误码 -1
    if (length < 1) {
        return -1;
    }

    # 如果 str 的第一个字符是 'q' 或 'Q'，设置 sortkind 为 NPY_QUICKSORT，返回成功码 0
    if (str[0] == 'q' || str[0] == 'Q') {
        *sortkind = NPY_QUICKSORT;
        return 0;
    }
    # 如果 str 的第一个字符是 'h' 或 'H'，设置 sortkind 为 NPY_HEAPSORT，返回成功码 0
    else if (str[0] == 'h' || str[0] == 'H') {
        *sortkind = NPY_HEAPSORT;
        return 0;
    }
    # 如果 str 的第一个字符是 'm' 或 'M'，设置 sortkind 为 NPY_MERGESORT，返回成功码 0
    else if (str[0] == 'm' || str[0] == 'M') {
        /*
         * Mergesort 是 NPY_STABLESORT 的别名。
         * 这样做保持了向后兼容性，同时允许使用其他类型的稳定排序。
         */
        *sortkind = NPY_MERGESORT;
        return 0;
    }
    # 如果 str 的第一个字符是 's' 或 'S'，设置 sortkind 为 NPY_STABLESORT，返回成功码 0
    else if (str[0] == 's' || str[0] == 'S') {
        /*
         * NPY_STABLESORT 是以下之一：
         *
         *   - mergesort
         *   - timsort
         *
         * 具体使用哪种取决于数据类型。
         */
        *sortkind = NPY_STABLESORT;
        return 0;
    }
    # 如果 str 的第一个字符不符合上述任何条件，返回错误码 -1
    else {
        return -1;
    }
/*NUMPY_API
 * Convert object to sort kind
 */
NPY_NO_EXPORT int
PyArray_SortkindConverter(PyObject *obj, NPY_SORTKIND *sortkind)
{
    /* Leave the desired default from the caller for Py_None */
    if (obj == Py_None) {
        return NPY_SUCCEED;
    }
    /* 使用帮助函数进行字符串转换，将结果保存到 sortkind 指针所指向的位置 */
    return string_converter_helper(
        obj, (void *)sortkind, sortkind_parser, "sort kind",
        "must be one of 'quick', 'heap', or 'stable'");
}

/* 定义 selectkind_parser 函数，用于解析选择类型的字符串 */
static int selectkind_parser(char const *str, Py_ssize_t length, void *data)
{
    NPY_SELECTKIND *selectkind = (NPY_SELECTKIND *)data;

    if (length == 11 && strcmp(str, "introselect") == 0) {
        *selectkind = NPY_INTROSELECT;
        return 0;
    }
    else {
        return -1;
    }
}

/*NUMPY_API
 * Convert object to select kind
 */
NPY_NO_EXPORT int
PyArray_SelectkindConverter(PyObject *obj, NPY_SELECTKIND *selectkind)
{
    /* 使用帮助函数进行字符串转换，将结果保存到 selectkind 指针所指向的位置 */
    return string_converter_helper(
        obj, (void *)selectkind, selectkind_parser, "select kind",
        "must be 'introselect'");
}

/* 定义 searchside_parser 函数，用于解析搜索方向的字符串 */
static int searchside_parser(char const *str, Py_ssize_t length, void *data)
{
    NPY_SEARCHSIDE *side = (NPY_SEARCHSIDE *)data;
    int is_exact = 0;

    if (length < 1) {
        return -1;
    }
    else if (str[0] == 'l' || str[0] == 'L') {
        *side = NPY_SEARCHLEFT;
        is_exact = (length == 4 && strcmp(str, "left") == 0);
    }
    else if (str[0] == 'r' || str[0] == 'R') {
        *side = NPY_SEARCHRIGHT;
        is_exact = (length == 5 && strcmp(str, "right") == 0);
    }
    else {
        return -1;
    }

    /* 如果不是精确匹配，产生 DeprecationWarning */
    if (!is_exact) {
        /* NumPy 1.20, 2020-05-19 */
        if (DEPRECATE("inexact matches and case insensitive matches "
                      "for search side are deprecated, please use "
                      "one of 'left' or 'right' instead.") < 0) {
            return -1;
        }
    }

    return 0;
}

/*NUMPY_API
 * Convert object to searchsorted side
 */
NPY_NO_EXPORT int
PyArray_SearchsideConverter(PyObject *obj, void *addr)
{
    /* 使用帮助函数进行字符串转换，将结果保存到 addr 所指向的位置 */
    return string_converter_helper(
        obj, addr, searchside_parser, "search side",
        "must be 'left' or 'right'");
}

/* 定义 order_parser 函数，用于解析数组排序顺序的字符串 */
static int order_parser(char const *str, Py_ssize_t length, void *data)
{
    NPY_ORDER *val = (NPY_ORDER *)data;
    if (length != 1) {
        return -1;
    }
    if (str[0] == 'C' || str[0] == 'c') {
        *val = NPY_CORDER;
        return 0;
    }
    else if (str[0] == 'F' || str[0] == 'f') {
        *val = NPY_FORTRANORDER;
        return 0;
    }
    else if (str[0] == 'A' || str[0] == 'a') {
        *val = NPY_ANYORDER;
        return 0;
    }
    else if (str[0] == 'K' || str[0] == 'k') {
        *val = NPY_KEEPORDER;
        return 0;
    }
    else {
        return -1;
    }
}

/*NUMPY_API
 * Convert an object to FORTRAN / C / ANY / KEEP
 */
NPY_NO_EXPORT int
PyArray_OrderConverter(PyObject *object, NPY_ORDER *val)
{
    /* 如果 object 是 Py_None，则使用调用者期望的默认值 */
    if (object == Py_None) {
        // 如果 object 是 Py_None，则直接返回成功
        return NPY_SUCCEED;
    }
    // 否则，调用字符串转换助手函数，将 object 转换为字符串，并存储到 val 中
    return string_converter_helper(
        object, (void *)val, order_parser, "order",
        "must be one of 'C', 'F', 'A', or 'K'");
}

/* 解析剪裁模式字符串并转换为枚举值 */
static int clipmode_parser(char const *str, Py_ssize_t length, void *data)
{
    NPY_CLIPMODE *val = (NPY_CLIPMODE *)data;
    int is_exact = 0;

    // 检查字符串长度，如果小于1则返回错误
    if (length < 1) {
        return -1;
    }

    // 根据字符串首字符确定剪裁模式
    if (str[0] == 'C' || str[0] == 'c') {
        *val = NPY_CLIP;
        is_exact = (length == 4 && strcmp(str, "clip") == 0);  // 检查是否精确匹配
    }
    else if (str[0] == 'W' || str[0] == 'w') {
        *val = NPY_WRAP;
        is_exact = (length == 4 && strcmp(str, "wrap") == 0);  // 检查是否精确匹配
    }
    else if (str[0] == 'R' || str[0] == 'r') {
        *val = NPY_RAISE;
        is_exact = (length == 5 && strcmp(str, "raise") == 0);  // 检查是否精确匹配
    }
    else {
        return -1;  // 如果首字符不符合预期，则返回错误
    }

    /* 过滤掉大小写不敏感或非精确匹配的输入，并输出 DeprecationWarning */
    if (!is_exact) {
        /* Numpy 1.20, 2020-05-19 */
        if (DEPRECATE("inexact matches and case insensitive matches "
                      "for clip mode are deprecated, please use "
                      "one of 'clip', 'raise', or 'wrap' instead.") < 0) {
            return -1;
        }
    }

    return 0;  // 解析成功返回 0
}

/*NUMPY_API
 * 将对象转换为 NPY_RAISE / NPY_CLIP / NPY_WRAP
 */
NPY_NO_EXPORT int
PyArray_ClipmodeConverter(PyObject *object, NPY_CLIPMODE *val)
{
    // 如果对象为空或为 None，则设置为 NPY_RAISE
    if (object == NULL || object == Py_None) {
        *val = NPY_RAISE;
    }

    // 如果对象是字节串或 Unicode 字符串，则调用字符串转换器助手函数进行处理
    else if (PyBytes_Check(object) || PyUnicode_Check(object)) {
        return string_converter_helper(
            object, (void *)val, clipmode_parser, "clipmode",
            "must be one of 'clip', 'raise', or 'wrap'");
    }
    else {
        /* 对于传递了 `RAISE`, `WRAP`, `CLIP` 的用户 */
        int number = PyArray_PyIntAsInt(object);
        if (error_converting(number)) {
            goto fail;
        }
        if (number <= (int) NPY_RAISE
                && number >= (int) NPY_CLIP) {
            *val = (NPY_CLIPMODE) number;
        }
        else {
            PyErr_Format(PyExc_ValueError,
                    "integer clipmode must be RAISE, WRAP, or CLIP "
                    "from 'numpy._core.multiarray'");
        }
    }
    return NPY_SUCCEED;

 fail:
    PyErr_SetString(PyExc_TypeError,
                    "clipmode not understood");
    return NPY_FAIL;
}

/*NUMPY_API
 * 将对象转换为包含 n 个 NPY_CLIPMODE 值的数组。
 * 这用于需要为每个轴应用不同模式的函数，如 ravel_multi_index。
 */
NPY_NO_EXPORT int
PyArray_ConvertClipmodeSequence(PyObject *object, NPY_CLIPMODE *modes, int n)
{
    int i;
    /* 获取剪裁模式(s) */
    // 检查对象是否存在且为元组或列表
    if (object && (PyTuple_Check(object) || PyList_Check(object))) {
        // 如果对象是序列且长度不等于 n，则抛出长度错误异常并返回失败
        if (PySequence_Size(object) != n) {
            PyErr_Format(PyExc_ValueError,
                    "list of clipmodes has wrong length (%zd instead of %d)",
                    PySequence_Size(object), n);
            return NPY_FAIL;
        }

        // 遍历对象的每个元素
        for (i = 0; i < n; ++i) {
            // 获取序列中的第 i 个元素
            PyObject *item = PySequence_GetItem(object, i);
            // 如果获取元素失败，则返回失败
            if(item == NULL) {
                return NPY_FAIL;
            }

            // 尝试将元素转换为数组剪裁模式，若转换失败则释放元素并返回失败
            if(PyArray_ClipmodeConverter(item, &modes[i]) != NPY_SUCCEED) {
                Py_DECREF(item);
                return NPY_FAIL;
            }

            // 释放元素引用
            Py_DECREF(item);
        }
    }
    // 如果对象不是序列，尝试将其作为单个剪裁模式处理
    else if (PyArray_ClipmodeConverter(object, &modes[0]) == NPY_SUCCEED) {
        // 如果成功转换为剪裁模式，则将该模式应用到所有 modes 数组元素中
        for (i = 1; i < n; ++i) {
            modes[i] = modes[0];
        }
    }
    // 如果对象既不是序列也无法转换为剪裁模式，则返回失败
    else {
        return NPY_FAIL;
    }
    // 若所有操作成功完成，则返回成功
    return NPY_SUCCEED;
/* 
 * Parse and interpret a string representation of correlation mode
 * to an NPY_CORRELATEMODE value.
 */
static int correlatemode_parser(char const *str, Py_ssize_t length, void *data)
{
    NPY_CORRELATEMODE *val = (NPY_CORRELATEMODE *)data;
    int is_exact = 0;

    // Ensure the input string is not empty
    if (length < 1) {
        return -1;
    }

    // Check for case-insensitive matching for 'valid'
    if (str[0] == 'V' || str[0] == 'v') {
        *val = NPY_VALID;
        is_exact = (length == 5 && strcmp(str, "valid") == 0);
    }
    // Check for case-insensitive matching for 'same'
    else if (str[0] == 'S' || str[0] == 's') {
        *val = NPY_SAME;
        is_exact = (length == 4 && strcmp(str, "same") == 0);
    }
    // Check for case-insensitive matching for 'full'
    else if (str[0] == 'F' || str[0] == 'f') {
        *val = NPY_FULL;
        is_exact = (length == 4 && strcmp(str, "full") == 0);
    }
    else {
        // Invalid mode string
        return -1;
    }

    /* 
     * If the match was not exact (case-sensitive),
     * issue a deprecation warning.
     */
    if (!is_exact) {
        /* Numpy 1.21, 2021-01-19 */
        // Issue a deprecation warning for inexact matches
        if (DEPRECATE("inexact matches and case insensitive matches for "
                      "convolve/correlate mode are deprecated, please "
                      "use one of 'valid', 'same', or 'full' instead.") < 0) {
            return -1;
        }
    }

    // Parsing successful
    return 0;
}

/*
 * Convert an object to NPY_VALID / NPY_SAME / NPY_FULL
 */
NPY_NO_EXPORT int
PyArray_CorrelatemodeConverter(PyObject *object, NPY_CORRELATEMODE *val)
{
    // Check if the object is a Unicode string
    if (PyUnicode_Check(object)) {
        // Use string_converter_helper to convert the Unicode object
        return string_converter_helper(
            object, (void *)val, correlatemode_parser, "mode",
            "must be one of 'valid', 'same', or 'full'");
    }

    // If the object is not a Unicode string
    else {
        // Attempt conversion assuming it's an integer
        int number = PyArray_PyIntAsInt(object);
        // Handle conversion errors
        if (error_converting(number)) {
            PyErr_SetString(PyExc_TypeError,
                            "convolve/correlate mode not understood");
            return NPY_FAIL;
        }
        // Check if the integer is within the valid range for modes
        if (number <= (int) NPY_FULL && number >= (int) NPY_VALID) {
            *val = (NPY_CORRELATEMODE) number;
            return NPY_SUCCEED;
        }
        else {
            // Invalid integer mode value
            PyErr_Format(PyExc_ValueError,
                         "integer convolve/correlate mode must be 0, 1, or 2");
            return NPY_FAIL;
        }
    }
}
    # 当输入的 case 是 's' 时执行以下代码块
    case 's':
        # 如果输入长度为 6 并且字符串与 "unsafe" 完全相同
        if (length == 6 && strcmp(str, "unsafe") == 0) {
            # 设置 *casting 指针指向的值为 NPY_UNSAFE_CASTING
            *casting = NPY_UNSAFE_CASTING;
            # 返回 0 表示成功匹配并设置
            return 0;
        }
        # 如果条件不满足则跳出 switch 结构
        break;
    }
    # 默认情况返回 -1，表示未找到匹配项
    return -1;
/*NUMPY_API
 * 将任何 Python 对象 *obj* 转换为一个 NPY_CASTING 枚举。
 */
NPY_NO_EXPORT int
PyArray_CastingConverter(PyObject *obj, NPY_CASTING *casting)
{
    // 调用辅助函数进行字符串转换
    return string_converter_helper(
        obj, (void *)casting, casting_parser, "casting",
            "must be one of 'no', 'equiv', 'safe', "
            "'same_kind', or 'unsafe'");
    // 返回 0 表示成功
    return 0;
}

/*****************************
* 其他转换函数
*****************************/

static int
PyArray_PyIntAsInt_ErrMsg(PyObject *o, const char * msg)
{
    npy_intp long_value;
    /* 这里假设 NPY_SIZEOF_INTP >= NPY_SIZEOF_INT */
    long_value = PyArray_PyIntAsIntp_ErrMsg(o, msg);

#if (NPY_SIZEOF_INTP > NPY_SIZEOF_INT)
    // 如果 long_value 超出 int 的范围，则抛出 ValueError
    if ((long_value < INT_MIN) || (long_value > INT_MAX)) {
        PyErr_SetString(PyExc_ValueError, "integer won't fit into a C int");
        return -1;
    }
#endif
    // 返回 long_value 强制转换为 int 类型
    return (int) long_value;
}

/*NUMPY_API*/
NPY_NO_EXPORT int
PyArray_PyIntAsInt(PyObject *o)
{
    // 调用带错误消息的 PyArray_PyIntAsInt_ErrMsg 函数
    return PyArray_PyIntAsInt_ErrMsg(o, "an integer is required");
}

static npy_intp
PyArray_PyIntAsIntp_ErrMsg(PyObject *o, const char * msg)
{
#if (NPY_SIZEOF_LONG < NPY_SIZEOF_INTP)
    long long long_value = -1;
#else
    long long_value = -1;
#endif
    PyObject *obj, *err;

    /*
     * 更加严格，不允许 bool 值。
     * np.bool 也被禁止，因为布尔数组目前不支持索引。
     */
    // 如果 o 为空或者是 bool 值，或者是布尔类型数组，则抛出 TypeError
    if (!o || PyBool_Check(o) || PyArray_IsScalar(o, Bool)) {
        PyErr_SetString(PyExc_TypeError, msg);
        return -1;
    }

    /*
     * 因为这是通常的情况，首先检查 o 是否是整数。这是一个精确的检查，因为否则会使用 __index__ 方法。
     */
    // 如果 o 是 PyLong_CheckExact，则直接将其转换为 long_value
    if (PyLong_CheckExact(o)) {
#if (NPY_SIZEOF_LONG < NPY_SIZEOF_INTP)
        long_value = PyLong_AsLongLong(o);
#else
        long_value = PyLong_AsLong(o);
#endif
        return (npy_intp)long_value;
    }

    /*
     * 最一般的情况。PyNumber_Index(o) 包含了所有情况，包括数组。原则上，可以在停用后使用 PyIndex_AsSSize_t 替换整个函数。
     */
    // 调用 PyNumber_Index(o) 获取索引，然后将其转换为 long_value
    obj = PyNumber_Index(o);
    if (obj == NULL) {
        return -1;
    }
#if (NPY_SIZEOF_LONG < NPY_SIZEOF_INTP)
    long_value = PyLong_AsLongLong(obj);
#else
    long_value = PyLong_AsLong(obj);
#endif
    Py_DECREF(obj);

    // 如果转换失败，则设置相应错误消息
    if (error_converting(long_value)) {
        err = PyErr_Occurred();
        // 只在这里替换 TypeError，因为这是正常的错误情况。
        if (PyErr_GivenExceptionMatches(err, PyExc_TypeError)) {
            PyErr_SetString(PyExc_TypeError, msg);
        }
        return -1;
    }
    // 转换成功后检查溢出情况
    goto overflow_check; /* 防止未使用的警告 */

overflow_check:
#if (NPY_SIZEOF_LONG < NPY_SIZEOF_INTP)
  #if (NPY_SIZEOF_LONGLONG > NPY_SIZEOF_INTP)
    // 检查 long_value 是否超出 numpy.intp 类型的范围
    if ((long_value < NPY_MIN_INTP) || (long_value > NPY_MAX_INTP)) {
        // 如果超出范围，设置 OverflowError 异常并返回错误码 -1
        PyErr_SetString(PyExc_OverflowError,
                "Python int too large to convert to C numpy.intp");
        return -1;
    }
  #endif
#else
  #if (NPY_SIZEOF_LONG > NPY_SIZEOF_INTP)
    // 如果长整型的大小大于 npy_intp 的大小，则检查 long_value 是否超出 npy_intp 的范围
    if ((long_value < NPY_MIN_INTP) || (long_value > NPY_MAX_INTP)) {
        // 如果 long_value 超出范围，设置 OverflowError 并返回 -1
        PyErr_SetString(PyExc_OverflowError,
                "Python int too large to convert to C numpy.intp");
        return -1;
    }
  #endif
#endif
    // 返回 long_value
    return long_value;
}

/*NUMPY_API*/
// 将 Python 中的整数对象 o 转换为 npy_intp 类型
NPY_NO_EXPORT npy_intp
PyArray_PyIntAsIntp(PyObject *o)
{
    // 调用带错误消息的 PyArray_PyIntAsIntp_ErrMsg 函数
    return PyArray_PyIntAsIntp_ErrMsg(o, "an integer is required");
}


NPY_NO_EXPORT int
// 将 Python 对象 o 转换为 npy_intp 类型的 C 数组指针 *val
PyArray_IntpFromPyIntConverter(PyObject *o, npy_intp *val)
{
    // 调用 PyArray_PyIntAsIntp 将 o 转换为 npy_intp 类型，并将结果赋给 *val
    *val = PyArray_PyIntAsIntp(o);
    // 如果转换出错，返回 NPY_FAIL
    if (error_converting(*val)) {
        return NPY_FAIL;
    }
    // 成功转换，返回 NPY_SUCCEED
    return NPY_SUCCEED;
}


/**
 * 从整数序列中读取值并存储到数组中。
 *
 * @param  seq      使用 `PySequence_Fast` 创建的序列。
 * @param  vals     用于存储维度的数组（必须足够大以容纳 `maxvals` 个值）。
 * @param  max_vals 可以写入 `vals` 的最大维度数。
 * @return          维度数或如果发生错误则返回 -1。
 *
 * .. note::
 *
 *   与 PyArray_IntpFromSequence 相反，它使用并返回 `npy_intp`
 *      作为值的数量。
 */
NPY_NO_EXPORT npy_intp
PyArray_IntpFromIndexSequence(PyObject *seq, npy_intp *vals, npy_intp maxvals)
{
    /*
     * 首先，检查序列是否是标量整数或是否可以“转换”为标量。
     */
    Py_ssize_t nd = PySequence_Fast_GET_SIZE(seq);
    PyObject *op;
    for (Py_ssize_t i = 0; i < PyArray_MIN(nd, maxvals); i++) {
        op = PySequence_Fast_GET_ITEM(seq, i);

        // 调用 dimension_from_scalar 获取标量 op 的维度，并存储到 vals[i] 中
        vals[i] = dimension_from_scalar(op);
        // 如果转换出错，返回 -1
        if (error_converting(vals[i])) {
            return -1;
        }
    }
    // 返回维度数 nd
    return nd;
}

/*NUMPY_API
 * PyArray_IntpFromSequence
 * 返回转换的整数数目或如果发生错误则返回 -1。
 * vals 必须足够大以容纳 maxvals
 */
NPY_NO_EXPORT int
// 从 Python 对象 seq 转换为 npy_intp 数组 vals
PyArray_IntpFromSequence(PyObject *seq, npy_intp *vals, int maxvals)
{
    PyObject *seq_obj = NULL;
    // 如果 seq 不是长整数并且是序列类型
    if (!PyLong_CheckExact(seq) && PySequence_Check(seq)) {
        // 使用 PySequence_Fast 尝试快速获取序列对象 seq_obj
        seq_obj = PySequence_Fast(seq,
            "expected a sequence of integers or a single integer");
        // 如果获取失败，继续尝试作为单个整数解析
        if (seq_obj == NULL) {
            PyErr_Clear();
        }
    }

    // 如果 seq_obj 为空
    if (seq_obj == NULL) {
        // 将 seq 视为标量，并将其维度存储到 vals[0] 中
        vals[0] = dimension_from_scalar(seq);
        // 如果转换出错，抛出适当的 TypeError 异常
        if (error_converting(vals[0])) {
            if (PyErr_ExceptionMatches(PyExc_TypeError)) {
                PyErr_Format(PyExc_TypeError,
                        "expected a sequence of integers or a single "
                        "integer, got '%.100R'", seq);
            }
            return -1;
        }
        // 返回 1 表示成功转换一个值
        return 1;
    }
    else {
        int res;
        // 调用 PyArray_IntpFromIndexSequence 从 seq_obj 中获取 npy_intp 类型的值到 vals，并最多获取 maxvals 个值
        res = PyArray_IntpFromIndexSequence(seq_obj, vals, (npy_intp)maxvals);
        // 释放 seq_obj 对象的引用
        Py_DECREF(seq_obj);
        // 返回转换结果
        return res;
    }
}
/**
 * WARNING: This flag is a bad idea, but was the only way to both
 *   1) Support unpickling legacy pickles with object types.
 *   2) Deprecate (and later disable) usage of O4 and O8
 *
 * The key problem is that the pickled representation unpickles by
 * directly calling the dtype constructor, which has no way of knowing
 * that it is in an unpickle context instead of a normal context without
 * evil global state like we create here.
 */
NPY_NO_EXPORT NPY_TLS int evil_global_disable_warn_O4O8_flag = 0;

/*
 * Convert a gentype (that is actually a generic kind character) and
 * it's itemsize to a NUmPy typenumber, i.e. `itemsize=4` and `gentype='f'`
 * becomes `NPY_FLOAT32`.
 */
NPY_NO_EXPORT int
PyArray_TypestrConvert(int itemsize, int gentype)
{
    int newtype = NPY_NOTYPE;

    switch (gentype) {
        case NPY_GENBOOLLTR:
            if (itemsize == 1) {
                newtype = NPY_BOOL;
            }
            break;

        case NPY_SIGNEDLTR:
            switch(itemsize) {
                case 1:
                    newtype = NPY_INT8;
                    break;
                case 2:
                    newtype = NPY_INT16;
                    break;
                case 4:
                    newtype = NPY_INT32;
                    break;
                case 8:
                    newtype = NPY_INT64;
                    break;
#ifdef NPY_INT128
                case 16:
                    newtype = NPY_INT128;
                    break;
#endif
            }
            break;

        case NPY_UNSIGNEDLTR:
            switch(itemsize) {
                case 1:
                    newtype = NPY_UINT8;
                    break;
                case 2:
                    newtype = NPY_UINT16;
                    break;
                case 4:
                    newtype = NPY_UINT32;
                    break;
                case 8:
                    newtype = NPY_UINT64;
                    break;
#ifdef NPY_INT128
                case 16:
                    newtype = NPY_UINT128;
                    break;
#endif
            }
            break;

        case NPY_FLOATINGLTR:
            switch(itemsize) {
                case 2:
                    newtype = NPY_FLOAT16;
                    break;
                case 4:
                    newtype = NPY_FLOAT32;
                    break;
                case 8:
                    newtype = NPY_FLOAT64;
                    break;
#ifdef NPY_FLOAT80
                case 10:
                    newtype = NPY_FLOAT80;
                    break;
#endif
#ifdef NPY_FLOAT96
                case 12:
                    newtype = NPY_FLOAT96;
                    break;
#endif
#ifdef NPY_FLOAT128
                case 16:
                    newtype = NPY_FLOAT128;
                    break;
#endif
            }
            break;
    }

    // 返回对应的 NumPy 类型编号
    return newtype;
}
#endif
            }
            break;

        case NPY_COMPLEXLTR:
            switch(itemsize) {
                case 8:
                    // 设置新类型为复数类型 NPY_COMPLEX64
                    newtype = NPY_COMPLEX64;
                    break;
                case 16:
                    // 设置新类型为复数类型 NPY_COMPLEX128
                    newtype = NPY_COMPLEX128;
                    break;
#ifdef NPY_FLOAT80
                case 20:
                    // 设置新类型为复数类型 NPY_COMPLEX160（仅在 NPY_FLOAT80 定义时有效）
                    newtype = NPY_COMPLEX160;
                    break;
#endif
#ifdef NPY_FLOAT96
                case 24:
                    // 设置新类型为复数类型 NPY_COMPLEX192（仅在 NPY_FLOAT96 定义时有效）
                    newtype = NPY_COMPLEX192;
                    break;
#endif
#ifdef NPY_FLOAT128
                case 32:
                    // 设置新类型为复数类型 NPY_COMPLEX256（仅在 NPY_FLOAT128 定义时有效）
                    newtype = NPY_COMPLEX256;
                    break;
#endif
            }
            break;

        case NPY_OBJECTLTR:
            /*
             * 对于 'O4' 和 'O8'，允许通过，但会发出废弃警告。
             * 对于所有其他情况，通过不设置 newtype 来引发异常。
             */
            if (itemsize == 4 || itemsize == 8) {
                int ret = 0;
                if (evil_global_disable_warn_O4O8_flag) {
                    /* 2012-02-04, 1.7, 不确定何时可以移除此代码 */
                    // 发出废弃警告，提示使用 'O' 替代 'O4' 和 'O8'
                    ret = DEPRECATE("DType strings 'O4' and 'O8' are "
                            "deprecated because they are platform "
                            "specific. Use 'O' instead");
                }

                if (ret == 0) {
                    // 设置新类型为对象类型 NPY_OBJECT
                    newtype = NPY_OBJECT;
                }
            }
            break;

        case NPY_STRINGLTR:
            // 设置新类型为字符串类型 NPY_STRING
            newtype = NPY_STRING;
            break;

        case NPY_DEPRECATED_STRINGLTR2:
        {
            /*
             * 发出废弃警告，可能会引发异常，如果警告被配置为错误，不设置 newtype。
             */
            // 发出废弃警告，提示在 NumPy 2.0 中 'a' 数据类型别名已被废弃，使用 'S' 别名替代
            int ret = DEPRECATE("Data type alias 'a' was deprecated in NumPy 2.0. "
                                "Use the 'S' alias instead.");
            if (ret == 0) {
                // 设置新类型为字符串类型 NPY_STRING
                newtype = NPY_STRING;
            }
            break;
        }
        case NPY_UNICODELTR:
            // 设置新类型为Unicode类型 NPY_UNICODE
            newtype = NPY_UNICODE;
            break;

        case NPY_VOIDLTR:
            // 设置新类型为VOID类型 NPY_VOID
            newtype = NPY_VOID;
            break;

        case NPY_DATETIMELTR:
            if (itemsize == 8) {
                // 设置新类型为日期时间类型 NPY_DATETIME
                newtype = NPY_DATETIME;
            }
            break;

        case NPY_TIMEDELTALTR:
            if (itemsize == 8) {
                // 设置新类型为时间间隔类型 NPY_TIMEDELTA
                newtype = NPY_TIMEDELTA;
            }
            break;
    }

    // 返回确定的新数据类型
    return newtype;
}

/* Lifted from numarray */
/* TODO: not documented */
/*NUMPY_API
  PyArray_IntTupleFromIntp
*/
NPY_NO_EXPORT PyObject *
PyArray_IntTupleFromIntp(int len, npy_intp const *vals)
{
    int i;
    // 创建一个包含整数的元组对象
    PyObject *intTuple = PyTuple_New(len);

    if (!intTuple) {
        // 如果创建失败，跳转到失败处理分支
        goto fail;
    }
    // 循环遍历 vals 数组，逐个处理数组元素
    for (i = 0; i < len; i++) {
        // 根据 vals[i] 创建一个 Python int 对象
        PyObject *o = PyArray_PyIntFromIntp(vals[i]);
        // 如果创建失败，则执行失败处理逻辑
        if (!o) {
            // 释放已创建的 intTuple 对象
            Py_DECREF(intTuple);
            intTuple = NULL;
            // 跳转到失败处理标签
            goto fail;
        }
        // 将创建的 int 对象 o 放入 intTuple 元组的第 i 个位置
        PyTuple_SET_ITEM(intTuple, i, o);
    }

 fail:
    // 返回 intTuple 对象，可能为 NULL 或者包含了创建的 int 对象
    return intTuple;
}
```