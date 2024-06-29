# `D:\src\scipysrc\matplotlib\src\mplutils.h`

```
/* -*- mode: c++; c-basic-offset: 4 -*- */

/* Small utilities that are shared by most extension modules. */

#ifndef MPLUTILS_H
#define MPLUTILS_H
#define PY_SSIZE_T_CLEAN

#include <Python.h>

#ifdef _POSIX_C_SOURCE
#    undef _POSIX_C_SOURCE
#endif
#ifndef _AIX
#ifdef _XOPEN_SOURCE
#    undef _XOPEN_SOURCE
#endif
#endif

// Prevent multiple conflicting definitions of swab from stdlib.h and unistd.h
#if defined(__sun) || defined(sun)
#if defined(_XPG4)
#undef _XPG4
#endif
#if defined(_XPG3)
#undef _XPG3
#endif
#endif

// 定义了一个内联函数，将 double 类型的浮点数四舍五入为整数
inline int mpl_round_to_int(double v)
{
    return (int)(v + ((v >= 0.0) ? 0.5 : -0.5));
}

// 定义了一个内联函数，将 double 类型的浮点数四舍五入为另一个 double 类型的浮点数
inline double mpl_round(double v)
{
    return (double)mpl_round_to_int(v);
}

// 'kind' 用于表示路径的类型代码，作为枚举常量
enum {
    STOP = 0,
    MOVETO = 1,
    LINETO = 2,
    CURVE3 = 3,
    CURVE4 = 4,
    CLOSEPOLY = 0x4f
};

// 定义了一个内联函数，用于准备并将类型添加到指定的 Python 模块中
inline int prepare_and_add_type(PyTypeObject *type, PyObject *module)
{
    // 检查并准备要添加的类型对象
    if (PyType_Ready(type)) {
        return -1;
    }
    // 从类型对象的名称中获取最后一个点之后的部分
    char const* ptr = strrchr(type->tp_name, '.');
    if (!ptr) {
        // 如果找不到点，则抛出值错误异常
        PyErr_SetString(PyExc_ValueError, "tp_name should be a qualified name");
        return -1;
    }
    // 将类型对象添加到指定的 Python 模块中
    if (PyModule_AddObject(module, ptr + 1, (PyObject *)type)) {
        return -1;
    }
    return 0;
}

#ifdef __cplusplus  // not for macosx.m
// 检查数组的形状是否为 (N, d1) 的内联函数模板
template<typename T>
inline bool check_trailing_shape(T array, char const* name, long d1)
{
    // 如果数组的第二维不等于 d1，则抛出值错误异常
    if (array.shape(1) != d1) {
        PyErr_Format(PyExc_ValueError,
                     "%s must have shape (N, %ld), got (%ld, %ld)",
                     name, d1, array.shape(0), array.shape(1));
        return false;
    }
    return true;
}

// 检查数组的形状是否为 (N, d1, d2) 的内联函数模板
template<typename T>
inline bool check_trailing_shape(T array, char const* name, long d1, long d2)
{
    // 如果数组的第二维或第三维不符合要求，则抛出值错误异常
    if (array.shape(1) != d1 || array.shape(2) != d2) {
        PyErr_Format(PyExc_ValueError,
                     "%s must have shape (N, %ld, %ld), got (%ld, %ld, %ld)",
                     name, d1, d2, array.shape(0), array.shape(1), array.shape(2));
        return false;
    }
    return true;
}
#endif

#endif
```