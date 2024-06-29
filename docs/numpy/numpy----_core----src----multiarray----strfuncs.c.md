# `.\numpy\numpy\_core\src\multiarray\strfuncs.c`

```
/*
 * 定义以避免使用已弃用的 NumPy API 版本
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

/*
 * 定义用于多维数组模块的标志
 */
#define _MULTIARRAYMODULE

/*
 * 清除 PY_SSIZE_T_CLEAN 宏定义，确保在包含 Python.h 之前不会定义 ssize_t
 */
#define PY_SSIZE_T_CLEAN

/*
 * 包含 Python.h，这是所有 Python C API 的核心头文件
 */
#include <Python.h>

/*
 * 包含 NumPy 提供的数组对象头文件
 */
#include "numpy/arrayobject.h"

/*
 * 包含 NumPy 兼容性模块的头文件
 */
#include "npy_pycompat.h"

/*
 * 包含 NumPy 导入模块的头文件
 */
#include "npy_import.h"

/*
 * 包含多维数组模块的头文件
 */
#include "multiarraymodule.h"

/*
 * 包含字符串处理函数的头文件
 */
#include "strfuncs.h"

/*
 * 定义一个静态函数 npy_PyErr_SetStringChained，设置一个字符串类型的异常，并链式传递之前的异常
 */
static void
npy_PyErr_SetStringChained(PyObject *type, const char *message)
{
    PyObject *exc, *val, *tb;

    /*
     * 获取当前的异常信息
     */
    PyErr_Fetch(&exc, &val, &tb);
    
    /*
     * 设置一个新的异常信息
     */
    PyErr_SetString(type, message);
    
    /*
     * 将之前捕获的异常信息链式传递
     */
    npy_PyErr_ChainExceptionsCause(exc, val, tb);
}

/*
 * NUMPY_API
 * 将数组打印函数设置为 Python 函数。
 * 该函数不会导出给外部模块使用。
 */
NPY_NO_EXPORT void
PyArray_SetStringFunction(PyObject *op, int repr)
{
    /*
     * 抛出 ValueError 异常，因为 PyArray_SetStringFunction 已被移除
     */
    PyErr_SetString(PyExc_ValueError, "PyArray_SetStringFunction was removed");
}

/*
 * NUMPY_API
 * 返回一个数组对象的字符串表示形式。
 * 该函数不会导出给外部模块使用。
 */
NPY_NO_EXPORT PyObject *
array_repr(PyArrayObject *self)
{
    /*
     * 延迟导入 numpy._core.arrayprint 模块中的 _default_array_repr 函数，
     * 避免在模块加载时引起循环导入问题。
     */
    npy_cache_import("numpy._core.arrayprint", "_default_array_repr",
                     &npy_thread_unsafe_state._default_array_repr);
    if (npy_thread_unsafe_state._default_array_repr == NULL) {
        /*
         * 如果无法配置默认的 ndarray.__repr__ 函数，则抛出 RuntimeError 异常
         */
        npy_PyErr_SetStringChained(PyExc_RuntimeError,
                "Unable to configure default ndarray.__repr__");
        return NULL;
    }
    return PyObject_CallFunctionObjArgs(
            npy_thread_unsafe_state._default_array_repr, self, NULL);
}

/*
 * NUMPY_API
 * 返回一个数组对象的字符串表示形式。
 * 该函数不会导出给外部模块使用。
 */
NPY_NO_EXPORT PyObject *
array_str(PyArrayObject *self)
{
    /*
     * 延迟导入 numpy._core.arrayprint 模块中的 _default_array_str 函数，
     * 避免在模块加载时引起循环导入问题。
     */
    npy_cache_import("numpy._core.arrayprint", "_default_array_str",
                     &npy_thread_unsafe_state._default_array_str);
    if (npy_thread_unsafe_state._default_array_str == NULL) {
        /*
         * 如果无法配置默认的 ndarray.__str__ 函数，则抛出 RuntimeError 异常
         */
        npy_PyErr_SetStringChained(PyExc_RuntimeError,
                "Unable to configure default ndarray.__str__");
        return NULL;
    }
    return PyObject_CallFunctionObjArgs(
            npy_thread_unsafe_state._default_array_str, self, NULL);
}

/*
 * NUMPY_API
 * 格式化数组对象。
 * 该函数不会导出给外部模块使用。
 */
NPY_NO_EXPORT PyObject *
array_format(PyArrayObject *self, PyObject *args)
{
    PyObject *format;
    if (!PyArg_ParseTuple(args, "O:__format__", &format))
        return NULL;

    /* 
     * 对于 0 维数组，转发到标量类型
     */
    if (PyArray_NDIM(self) == 0) {
        PyObject *item = PyArray_ToScalar(PyArray_DATA(self), self);
        PyObject *res;

        if (item == NULL) {
            return NULL;
        }
        res = PyObject_Format(item, format);
        Py_DECREF(item);
        return res;
    }
    /* 
     * 其他情况下使用内置方法
     */
    else {
        return PyObject_CallMethod(
            (PyObject *)&PyBaseObject_Type, "__format__", "OO",
            (PyObject *)self, format
        );
    }
}
```