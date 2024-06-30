# `D:\src\scipysrc\scipy\scipy\_lib\_uarray\vectorcall.h`

```
#pragma once
#include <Python.h>  // 包含 Python C API 的头文件

#ifdef __cplusplus
extern "C" {
#endif

#ifdef PYPY_VERSION
#  define Q_Py_TPFLAGS_HAVE_VECTORCALL 0  // 如果是 PyPy，不支持 VECTORCALL，设为 0
#  define Q_Py_TPFLAGS_METHOD_DESCRIPTOR 0  // 如果是 PyPy，不支持方法描述符，设为 0
#  define Q_PY_VECTORCALL_ARGUMENTS_OFFSET \
    ((size_t)1 << (8 * sizeof(size_t) - 1))  // 如果是 PyPy，设定 VECTORCALL 参数偏移
#else
#  define Q_Py_TPFLAGS_HAVE_VECTORCALL Py_TPFLAGS_HAVE_VECTORCALL  // 否则使用 Python 标志中定义的 VECTORCALL 支持
#  define Q_Py_TPFLAGS_METHOD_DESCRIPTOR Py_TPFLAGS_METHOD_DESCRIPTOR  // 否则使用 Python 标志中定义的方法描述符支持
#  define Q_PY_VECTORCALL_ARGUMENTS_OFFSET PY_VECTORCALL_ARGUMENTS_OFFSET  // 否则使用 Python 定义的 VECTORCALL 参数偏移
#endif

Py_ssize_t Q_PyVectorcall_NARGS(size_t n);  // 声明 Q_PyVectorcall_NARGS 函数原型

PyObject * Q_PyObject_Vectorcall(
    PyObject * callable, PyObject * const * args, size_t nargsf,
    PyObject * kwnames);  // 声明 Q_PyObject_Vectorcall 函数原型

PyObject * Q_PyObject_VectorcallDict(
    PyObject * callable, PyObject * const * args, size_t nargsf,
    PyObject * kwdict);  // 声明 Q_PyObject_VectorcallDict 函数原型

PyObject * Q_PyObject_VectorcallMethod(
    PyObject * name, PyObject * const * args, size_t nargsf, PyObject * kwdict);  // 声明 Q_PyObject_VectorcallMethod 函数原型

#ifdef __cplusplus
} // extern "C"
#endif
```