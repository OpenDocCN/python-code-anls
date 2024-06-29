# `.\numpy\numpy\_core\src\multiarray\common.c`

```py
/*
 * Define NPY_NO_DEPRECATED_API to use the latest NumPy API version.
 * Define _MULTIARRAYMODULE to indicate this module includes multi-array support.
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

/*
 * Ensure PY_SSIZE_T_CLEAN is defined before including Python.h
 * to use the new Py_ssize_t based API for Python objects.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "numpy/arrayobject.h"      // Include NumPy's array object header

#include "npy_config.h"             // Include NumPy's configuration header

#include "common.h"                 // Include common utility functions

#include "abstractdtypes.h"         // Include functions for abstract data types

#include "usertypes.h"              // Include user-defined data types

#include "npy_buffer.h"             // Include NumPy's buffer support

#include "get_attr_string.h"        // Include functions for getting attributes as strings

#include "mem_overlap.h"            // Include functions for memory overlap handling

#include "array_coercion.h"         // Include functions for array coercion


/*
 * The casting to use for implicit assignment operations resulting from
 * in-place operations (like +=) and out= arguments. (Notice that this
 * variable is misnamed, but it's part of the public API so I'm not sure we
 * can just change it. Maybe someone should try and see if anyone notices.
 */
/*
 * In numpy 1.6 and earlier, this was NPY_UNSAFE_CASTING. In a future
 * release, it will become NPY_SAME_KIND_CASTING.  Right now, during the
 * transitional period, we continue to follow the NPY_UNSAFE_CASTING rules (to
 * avoid breaking people's code), but we also check for whether the cast would
 * be allowed under the NPY_SAME_KIND_CASTING rules, and if not we issue a
 * warning (that people's code will be broken in a future release.)
 */

// Set the default casting rule for assignment operations
NPY_NO_EXPORT NPY_CASTING NPY_DEFAULT_ASSIGN_CASTING = NPY_SAME_KIND_CASTING;


// Function to find a NumPy dtype corresponding to a Python scalar object
NPY_NO_EXPORT PyArray_Descr *
_array_find_python_scalar_type(PyObject *op)
{
    // Check if the Python object is a float
    if (PyFloat_Check(op)) {
        // Return the NumPy dtype descriptor for double precision floating point
        return PyArray_DescrFromType(NPY_DOUBLE);
    }
    // Check if the Python object is a complex number
    else if (PyComplex_Check(op)) {
        // Return the NumPy dtype descriptor for double precision complex
        return PyArray_DescrFromType(NPY_CDOUBLE);
    }
    // Check if the Python object is a long integer
    else if (PyLong_Check(op)) {
        // Return the NumPy dtype descriptor discovered from the Python long object
        return NPY_DT_CALL_discover_descr_from_pyobject(
                &PyArray_PyLongDType, op);
    }
    // If the object doesn't match any of the above types, return NULL
    return NULL;
}


/*
 * Get a suitable string dtype by calling `__str__`.
 * For `np.bytes_`, this assumes an ASCII encoding.
 */
NPY_NO_EXPORT PyArray_Descr *
PyArray_DTypeFromObjectStringDiscovery(
        PyObject *obj, PyArray_Descr *last_dtype, int string_type)
{
    int itemsize;

    // If the string type is byte string (NPY_STRING)
    if (string_type == NPY_STRING) {
        // Get a string representation of the object
        PyObject *temp = PyObject_Str(obj);
        if (temp == NULL) {
            return NULL;
        }
        // Calculate the length of the string in characters
        itemsize = PyUnicode_GetLength(temp);
        Py_DECREF(temp);
        if (itemsize < 0) {
            return NULL;
        }
    }
    // If the string type is Unicode string (NPY_UNICODE)
    else if (string_type == NPY_UNICODE) {
        // Get a string representation of the object
        PyObject *temp = PyObject_Str(obj);
        if (temp == NULL) {
            return NULL;
        }
        // Calculate the length of the string in characters
        itemsize = PyUnicode_GetLength(temp);
        Py_DECREF(temp);
        if (itemsize < 0) {
            return NULL;
        }
        // Convert UCS4 codepoints to bytes (UCS4 = 4 bytes per character)
        itemsize *= 4;
    }
    else {
        // If the string type is neither NPY_STRING nor NPY_UNICODE, return NULL
        return NULL;
    }

    // If the last dtype is provided and matches the current string type and size
    if (last_dtype != NULL &&
        last_dtype->type_num == string_type &&
        last_dtype->elsize >= itemsize) {
        // Return the existing dtype with incremented reference count
        Py_INCREF(last_dtype);
        return last_dtype;
    }

    // Create a new dtype descriptor for the specified string type
    PyArray_Descr *dtype = PyArray_DescrNewFromType(string_type);
    if (dtype == NULL) {
        return NULL;
    }
    // Set the size of the dtype to match the calculated itemsize
    dtype->elsize = itemsize;
    return dtype;
}
/*
 * This function extracts the dtype from a Python object and performs shape discovery.
 * It returns only the dtype and is intended to be phased out in favor of PyArray_DiscoverDTypeAndShape.
 * This function is exported from the NumPy C API and should be used with caution.
 */
NPY_NO_EXPORT int
PyArray_DTypeFromObject(PyObject *obj, int maxdims, PyArray_Descr **out_dtype)
{
    coercion_cache_obj *cache = NULL;  // Initialize coercion cache pointer
    npy_intp shape[NPY_MAXDIMS];        // Array to store shape information
    int ndim;                           // Variable to hold number of dimensions

    // Call PyArray_DiscoverDTypeAndShape to discover dtype and shape of the object
    ndim = PyArray_DiscoverDTypeAndShape(
            obj, maxdims, shape, &cache, NULL, NULL, out_dtype, 1, NULL);
    if (ndim < 0) {
        return -1;  // Return -1 on error
    }
    npy_free_coercion_cache(cache);  // Free coercion cache memory
    return 0;  // Return 0 indicating success
}


NPY_NO_EXPORT npy_bool
_IsWriteable(PyArrayObject *ap)
{
    PyObject *base = PyArray_BASE(ap);  // Get the base object of the array
    Py_buffer view;                     // Buffer view object for array data

    /*
     * Check if the array is writable based on its base and ownership flags.
     * Arrays without a base or with owned data are considered writable.
     */
    if (base == NULL || PyArray_CHKFLAGS(ap, NPY_ARRAY_OWNDATA)) {
        /*
         * Handle cases where arrays wrapped in C-data may not own their data,
         * or where WRITEBACKIFCOPY arrays own their data but have a base.
         */
        return NPY_TRUE;  // Return true indicating array is writable
    }

    /*
     * Traverse through the base objects to find the final base.
     * If a writable array is found during traversal, return true.
     */
    while (PyArray_Check(base)) {
        ap = (PyArrayObject *)base;  // Cast base to PyArrayObject
        base = PyArray_BASE(ap);     // Update base to next base object

        if (PyArray_ISWRITEABLE(ap)) {
            /*
             * If any base is writable, return true.
             * Bases are typically collapsed to the most general one.
             */
            return NPY_TRUE;
        }

        if (base == NULL || PyArray_CHKFLAGS(ap, NPY_ARRAY_OWNDATA)) {
            /* No further base to test for writeability */
            return NPY_FALSE;
        }
        assert(!PyArray_CHKFLAGS(ap, NPY_ARRAY_OWNDATA));  // Assert ownership flag is not set
    }

    // Check if the base object supports a writable buffer view
    if (PyObject_GetBuffer(base, &view, PyBUF_WRITABLE|PyBUF_SIMPLE) < 0) {
        PyErr_Clear();  // Clear any raised Python exceptions
        return NPY_FALSE;  // Return false if buffer view cannot be obtained
    }
    PyBuffer_Release(&view);  // Release the buffer view
    return NPY_TRUE;  // Return true indicating array is writable
}


/**
 * Convert an array shape to a string representation such as "(1, 2)".
 *
 * @param n - Dimensionality of the shape
 * @param vals - Pointer to shape array
 * @param ending - String to append after the shape "(1, 2)%s"
 *
 * @return Python unicode string object representing the shape
 */
NPY_NO_EXPORT PyObject *
convert_shape_to_string(npy_intp n, npy_intp const *vals, char *ending)
{
    npy_intp i;  // Loop variable

    /*
     * Convert the array shape into a string representation like "(1, 2)".
     * Append the specified ending string to the formatted shape.
     */

 * Convert an array shape to a string representation such as "(1, 2)".
 *
 * @param n - Dimensionality of the shape
 * @param vals - Pointer to shape array
 * @param ending - String to append after the shape "(1, 2)%s"
 *
 * @return Python unicode string object representing the shape
 */
NPY_NO_EXPORT PyObject *
convert_shape_to_string(npy_intp n, npy_intp const *vals, char *ending)
{
    npy_intp i;  // Loop variable

    /*
     * Convert the array shape into a string representation like "(1, 2)".
     * Append the specified ending string to the formatted shape.
     */
    /*
     * 如果值为负数，表示 "newaxis" 维度，对于打印来说可以丢弃，
     * 如果它是第一个维度。找到第一个非 "newaxis" 维度。
     */
    for (i = 0; i < n && vals[i] < 0; i++);

    // 如果所有维度都是 "newaxis"
    if (i == n) {
        // 返回一个格式化的空元组字符串
        return PyUnicode_FromFormat("()%s", ending);
    }

    // 创建一个字符串对象，表示第一个非 "newaxis" 维度
    PyObject *ret = PyUnicode_FromFormat("%" NPY_INTP_FMT, vals[i++]);
    if (ret == NULL) {
        return NULL;
    }

    // 处理剩余的维度
    for (; i < n; ++i) {
        PyObject *tmp;

        // 如果维度是 "newaxis"
        if (vals[i] < 0) {
            tmp = PyUnicode_FromString(",newaxis");
        }
        else {
            // 否则创建一个表示维度值的字符串对象
            tmp = PyUnicode_FromFormat(",%" NPY_INTP_FMT, vals[i]);
        }
        if (tmp == NULL) {
            Py_DECREF(ret);
            return NULL;
        }

        // 将当前维度字符串与之前的字符串连接起来
        Py_SETREF(ret, PyUnicode_Concat(ret, tmp));
        Py_DECREF(tmp);
        if (ret == NULL) {
            return NULL;
        }
    }

    // 最后格式化结果字符串，根据是否有多个维度选择不同的格式
    if (i == 1) {
        Py_SETREF(ret, PyUnicode_FromFormat("(%S,)%s", ret, ending));
    }
    else {
        Py_SETREF(ret, PyUnicode_FromFormat("(%S)%s", ret, ending));
    }
    return ret;
/**
 * dot_alignment_error - 用于报告数组形状不匹配错误的函数
 *
 * @param a: 第一个数组对象
 * @param i: 第一个数组中导致错误的维度索引
 * @param b: 第二个数组对象
 * @param j: 第二个数组中导致错误的维度索引
 */
NPY_NO_EXPORT void
dot_alignment_error(PyArrayObject *a, int i, PyArrayObject *b, int j)
{
    PyObject *errmsg = NULL, *format = NULL, *fmt_args = NULL,
             *i_obj = NULL, *j_obj = NULL,
             *shape1 = NULL, *shape2 = NULL,
             *shape1_i = NULL, *shape2_j = NULL;

    // 构建错误消息的格式字符串
    format = PyUnicode_FromString("shapes %s and %s not aligned:"
                                  " %d (dim %d) != %d (dim %d)");

    // 将数组形状转换为字符串表示
    shape1 = convert_shape_to_string(PyArray_NDIM(a), PyArray_DIMS(a), "");
    shape2 = convert_shape_to_string(PyArray_NDIM(b), PyArray_DIMS(b), "");

    // 创建表示错误的维度索引的对象
    i_obj = PyLong_FromLong(i);
    j_obj = PyLong_FromLong(j);

    // 获取导致错误的维度的大小并转换为 Python 对象
    shape1_i = PyLong_FromSsize_t(PyArray_DIM(a, i));
    shape2_j = PyLong_FromSsize_t(PyArray_DIM(b, j));

    // 如果创建任何对象失败，则跳转到 end 标签
    if (!format || !shape1 || !shape2 || !i_obj || !j_obj ||
            !shape1_i || !shape2_j) {
        goto end;
    }

    // 打包所有参数到元组中
    fmt_args = PyTuple_Pack(6, shape1, shape2,
                            shape1_i, i_obj, shape2_j, j_obj);
    if (fmt_args == NULL) {
        goto end;
    }

    // 格式化错误消息
    errmsg = PyUnicode_Format(format, fmt_args);
    if (errmsg != NULL) {
        // 设置值错误异常并附上错误消息
        PyErr_SetObject(PyExc_ValueError, errmsg);
    }
    else {
        // 如果无法格式化消息，设置通用错误消息
        PyErr_SetString(PyExc_ValueError, "shapes are not aligned");
    }

end:
    // 释放所有创建的 Python 对象
    Py_XDECREF(errmsg);
    Py_XDECREF(fmt_args);
    Py_XDECREF(format);
    Py_XDECREF(i_obj);
    Py_XDECREF(j_obj);
    Py_XDECREF(shape1);
    Py_XDECREF(shape2);
    Py_XDECREF(shape1_i);
    Py_XDECREF(shape2_j);
}

/**
 * _unpack_field - 解包 PyDataType_FIELDS(dtype) 元组
 *
 * @param value: 应为元组
 * @param descr: 将被设置为字段的数据类型描述符
 * @param offset: 将被设置为字段的偏移量
 *
 * @return: 失败返回 -1，成功返回 0
 */
NPY_NO_EXPORT int
_unpack_field(PyObject *value, PyArray_Descr **descr, npy_intp *offset)
{
    PyObject *off;

    // 检查元组长度，如果小于 2，返回错误
    if (PyTuple_GET_SIZE(value) < 2) {
        return -1;
    }

    // 设置描述符为元组的第一个元素
    *descr = (PyArray_Descr *)PyTuple_GET_ITEM(value, 0);
    // 设置偏移量为元组的第二个元素
    off = PyTuple_GET_ITEM(value, 1);

    // 如果偏移量是长整型，转换为 ssize_t 类型
    if (PyLong_Check(off)) {
        *offset = PyLong_AsSsize_t(off);
    }
    else {
        // 如果无法转换，设置索引错误异常
        PyErr_SetString(PyExc_IndexError, "can't convert offset");
        return -1;
    }

    return 0;
}

/**
 * _may_have_objects - 检查数据类型是否可能包含对象字段
 *
 * @param dtype: 要检查的数组数据类型描述符
 *
 * @return: 如果数据类型可能包含对象字段返回非零值，否则返回 0
 */
NPY_NO_EXPORT int
_may_have_objects(PyArray_Descr *dtype)
{
    PyArray_Descr *base = dtype;

    // 如果数据类型是子数组，则获取基础数据类型
    if (PyDataType_HASSUBARRAY(dtype)) {
        base = ((_PyArray_LegacyDescr *)dtype)->subarray->base;
    }

    // 检查数据类型是否有字段或者是否标记为可能包含对象
    return (PyDataType_HASFIELDS(base) ||
            PyDataType_FLAGCHK(base, NPY_ITEM_HASOBJECT));
}
/*
 * 创建一个新的空数组，尺寸为传入的大小，考虑到ap1和ap2的优先级。
 *
 * 如果`out`非空，则检查与ap1和ap2的内存重叠情况，并可能返回一个updateifcopy临时数组。
 * 如果`result`非空，则递增引用并将要返回的输出数组（如果`out`非空，则为`out`；否则为新分配的数组）放入*result。
 */
NPY_NO_EXPORT PyArrayObject *
new_array_for_sum(PyArrayObject *ap1, PyArrayObject *ap2, PyArrayObject* out,
                  int nd, npy_intp dimensions[], int typenum, PyArrayObject **result)
{
    PyArrayObject *out_buf;  // 声明PyArrayObject类型的指针out_buf

    if (out) {
        int d;

        /* 验证out是否可用 */
        if (PyArray_NDIM(out) != nd ||
            PyArray_TYPE(out) != typenum ||
            !PyArray_ISCARRAY(out)) {
            PyErr_SetString(PyExc_ValueError,
                "output array is not acceptable (must have the right datatype, "
                "number of dimensions, and be a C-Array)");
            return 0;  // 返回0，表示出错
        }
        for (d = 0; d < nd; ++d) {
            if (dimensions[d] != PyArray_DIM(out, d)) {
                PyErr_SetString(PyExc_ValueError,
                    "output array has wrong dimensions");
                return 0;  // 返回0，表示出错
            }
        }

        /* 检查内存重叠 */
        if (!(solve_may_share_memory(out, ap1, 1) == 0 &&
              solve_may_share_memory(out, ap2, 1) == 0)) {
            /* 分配临时输出数组 */
            out_buf = (PyArrayObject *)PyArray_NewLikeArray(out, NPY_CORDER,
                                                            NULL, 0);
            if (out_buf == NULL) {
                return NULL;  // 返回NULL，表示出错
            }

            /* 设置写回复制 */
            Py_INCREF(out);  // 递增引用计数
            if (PyArray_SetWritebackIfCopyBase(out_buf, out) < 0) {
                Py_DECREF(out);
                Py_DECREF(out_buf);
                return NULL;  // 返回NULL，表示出错
            }
        }
        else {
            Py_INCREF(out);  // 递增引用计数
            out_buf = out;
        }

        if (result) {
            Py_INCREF(out);  // 递增引用计数
            *result = out;  // 将out赋值给result指向的变量
        }

        return out_buf;  // 返回out_buf指向的数组对象
    }
    else {
        PyTypeObject *subtype;
        double prior1, prior2;
        /*
         * Need to choose an output array that can hold a sum
         * -- use priority to determine which subtype.
         */
        // 如果两个数组的类型不同，需要根据优先级选择一个能容纳和的输出数组类型
        if (Py_TYPE(ap2) != Py_TYPE(ap1)) {
            // 获取第二个数组的优先级
            prior2 = PyArray_GetPriority((PyObject *)ap2, 0.0);
            // 获取第一个数组的优先级
            prior1 = PyArray_GetPriority((PyObject *)ap1, 0.0);
            // 根据优先级选择子类型
            subtype = (prior2 > prior1 ? Py_TYPE(ap2) : Py_TYPE(ap1));
        }
        else {
            // 如果数组类型相同，则优先级相同，选择第一个数组的类型作为子类型
            prior1 = prior2 = 0.0;
            subtype = Py_TYPE(ap1);
        }

        // 创建新的数组对象作为输出缓冲区，使用选择的子类型
        out_buf = (PyArrayObject *)PyArray_New(subtype, nd, dimensions,
                                               typenum, NULL, NULL, 0, 0,
                                               (PyObject *)
                                               (prior2 > prior1 ? ap2 : ap1));

        // 如果成功创建输出缓冲区，并且结果指针有效，则增加输出缓冲区的引用计数，并设置结果指针
        if (out_buf != NULL && result) {
            Py_INCREF(out_buf);
            *result = out_buf;
        }

        // 返回输出缓冲区对象
        return out_buf;
    }
```cpp`
/* 检查一个 NumPy 数组是否可以转换为标量 */

NPY_NO_EXPORT int
check_is_convertible_to_scalar(PyArrayObject *v)
{
    // 如果数组的维度为 0，说明是标量，可以直接转换
    if (PyArray_NDIM(v) == 0) {
        return 0;
    }

    /* 移除此 if-else 块当不再需要支持该功能时 */
    // 如果数组大小为 1，即只有一个元素
    if (PyArray_SIZE(v) == 1) {
        /* Numpy 1.25.0, 2023-01-02 */
        // 发出弃用警告并返回 -1，表示转换不再支持
        if (DEPRECATE(
                "Conversion of an array with ndim > 0 to a scalar "
                "is deprecated, and will error in future. "
                "Ensure you extract a single element from your array "
                "before performing this operation. "
                "(Deprecated NumPy 1.25.)") < 0) {
            return -1;
        }
        return 0;
    } else {
        // 如果数组大小不为 1，则抛出类型错误异常
        PyErr_SetString(PyExc_TypeError,
            "only length-1 arrays can be converted to Python scalars");
        return -1;
    }

    // 由于之前的 return 语句会中断函数执行，因此这部分代码不会被执行到
    // 抛出类型错误异常，指示只有 0 维数组才能转换为 Python 标量
    PyErr_SetString(PyExc_TypeError,
            "only 0-dimensional arrays can be converted to Python scalars");
    return -1;
}
```