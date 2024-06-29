# `.\numpy\numpy\_core\src\common\gil_utils.c`

```
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE


// 定义宏：禁用已废弃的 NumPy API，并使用当前版本的 API
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
// 定义宏：用于多维数组模块
#define _MULTIARRAYMODULE



#define PY_SSIZE_T_CLEAN
#include <Python.h>


// 定义宏：启用 "PY_SSIZE_T_CLEAN"，确保所有使用 Py_ssize_t 类型的 API 都能正确清除内存
#define PY_SSIZE_T_CLEAN
// 引入 Python.h 头文件，包含了 Python C API 的核心功能
#include <Python.h>



#include <numpy/ndarraytypes.h>


// 引入 numpy/ndarraytypes.h 头文件，该文件定义了 NumPy 的数组类型和相关的宏定义
#include <numpy/ndarraytypes.h>



#include <stdarg.h>


// 引入 stdarg.h 头文件，提供了支持可变参数函数的宏定义和类型
#include <stdarg.h>



NPY_NO_EXPORT void
npy_gil_error(PyObject *type, const char *format, ...)
{
    va_list va;
    va_start(va, format);
    NPY_ALLOW_C_API_DEF;
    NPY_ALLOW_C_API;
    if (!PyErr_Occurred()) {
#if !defined(PYPY_VERSION)
        PyErr_FormatV(type, format, va);
#else
        PyObject *exc_str = PyUnicode_FromFormatV(format, va);
        if (exc_str == NULL) {
            // no reason to have special handling for this error case, since
            // this function sets an error anyway
            NPY_DISABLE_C_API;
            va_end(va);
            return;
        }
        PyErr_SetObject(type, exc_str);
        Py_DECREF(exc_str);
#endif
    }
    NPY_DISABLE_C_API;
    va_end(va);
}


// 函数定义：npy_gil_error，用于在释放全局解释器锁 (GIL) 期间出现错误时报告错误
NPY_NO_EXPORT void
npy_gil_error(PyObject *type, const char *format, ...)
{
    // 定义可变参数列表
    va_list va;
    // 初始化可变参数列表
    va_start(va, format);
    // 定义并允许 NumPy C API 使用
    NPY_ALLOW_C_API_DEF;
    NPY_ALLOW_C_API;
    // 如果没有已设置的异常
    if (!PyErr_Occurred()) {
        // 如果不是 PyPy 版本
#if !defined(PYPY_VERSION)
        // 使用格式化字符串和可变参数设置异常
        PyErr_FormatV(type, format, va);
#else
        // 否则，根据格式化字符串和可变参数创建 Python Unicode 字符串对象
        PyObject *exc_str = PyUnicode_FromFormatV(format, va);
        // 如果创建失败
        if (exc_str == NULL) {
            // 没有特别处理此错误情况的原因，因为该函数无论如何都会设置错误
            NPY_DISABLE_C_API;
            // 结束可变参数列表
            va_end(va);
            // 返回
            return;
        }
        // 将 Python 对象设置为异常对象
        PyErr_SetObject(type, exc_str);
        // 释放 Python 对象的引用计数
        Py_DECREF(exc_str);
#endif
    }
    // 禁用 NumPy C API 使用
    NPY_DISABLE_C_API;
    // 结束可变参数列表
    va_end(va);
}
```