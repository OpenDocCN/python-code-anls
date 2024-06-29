# `.\numpy\numpy\_core\src\common\ufunc_override.c`

```py
/*
 * 定义宏，禁用所有已弃用的 NumPy API，使用当前版本的 API
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

/*
 * 定义宏，启用多维数组模块
 */
#define _MULTIARRAYMODULE

/*
 * 包含必要的头文件，用于 NumPy 的数组类型定义
 */
#include "numpy/ndarraytypes.h"
#include "npy_pycompat.h"
#include "get_attr_string.h"
#include "npy_import.h"
#include "ufunc_override.h"
#include "scalartypes.h"
#include "npy_static_data.h"

/*
 * 检查对象是否在其类上定义了 __array_ufunc__ 方法，并且它不是默认的情况下，
 * 即对象不是 ndarray，并且其 __array_ufunc__ 与 ndarray 的不同。
 *
 * 如果存在并且不同于 ndarray 的 __array_ufunc__，返回类型(obj).__array_ufunc__ 的新引用；
 * 否则返回 NULL。
 */
NPY_NO_EXPORT PyObject *
PyUFuncOverride_GetNonDefaultArrayUfunc(PyObject *obj)
{
    PyObject *cls_array_ufunc;

    /* 快速返回 ndarray */
    if (PyArray_CheckExact(obj)) {
        return NULL;
    }
    
    /* 快速返回 numpy 标量类型 */
    if (is_anyscalar_exact(obj)) {
        return NULL;
    }

    /*
     * 类是否定义了 __array_ufunc__？（注意，LookupSpecial 对基本的 Python 类型有快速返回，所以这里不必担心）
     */
    cls_array_ufunc = PyArray_LookupSpecial(obj, npy_interned_str.array_ufunc);
    if (cls_array_ufunc == NULL) {
        if (PyErr_Occurred()) {
            PyErr_Clear(); /* TODO[gh-14801]: 在属性访问期间传播崩溃？ */
        }
        return NULL;
    }

    /* 如果与 ndarray.__array_ufunc__ 相同，则忽略 */
    if (cls_array_ufunc == npy_static_pydata.ndarray_array_ufunc) {
        Py_DECREF(cls_array_ufunc);
        return NULL;
    }

    return cls_array_ufunc;
}

/*
 * 检查对象是否在其类上定义了 __array_ufunc__ 方法，并且它不是默认的情况下，
 * 即对象不是 ndarray，并且其 __array_ufunc__ 与 ndarray 的不同。
 *
 * 如果是，返回 1；否则返回 0。
 */
NPY_NO_EXPORT int
PyUFunc_HasOverride(PyObject * obj)
{
    PyObject *method = PyUFuncOverride_GetNonDefaultArrayUfunc(obj);
    if (method) {
        Py_DECREF(method);
        return 1;
    }
    else {
        return 0;
    }
}

/*
 * 从 kwds 中获取可能的 out 参数，并返回其中包含的输出数量：
 * 如果是一个元组，则返回其中元素的数量，否则返回 1。
 * out 参数本身作为 out_kwd_obj 返回，并且输出作为 out_objs 数组（作为借用引用）返回。
 *
 * 如果没有找到输出，则返回 0；如果 kwds 不是字典，则返回 -1（并设置错误）。
 */
NPY_NO_EXPORT int
PyUFuncOverride_GetOutObjects(PyObject *kwds, PyObject **out_kwd_obj, PyObject ***out_objs)
{
    if (kwds == NULL) {
        Py_INCREF(Py_None);
        *out_kwd_obj = Py_None;
        return 0;
    }

    if (!PyDict_CheckExact(kwds)) {
        PyErr_SetString(PyExc_TypeError,
                        "Internal Numpy error: call to PyUFuncOverride_GetOutObjects "
                        "with non-dict kwds");
        *out_kwd_obj = NULL;
        return -1;
    }

    int result = PyDict_GetItemStringRef(kwds, "out", out_kwd_obj);
    // 如果结果为 -1，则返回 -1
    if (result == -1) {
        return -1;
    }
    // 如果结果为 0，则增加 Py_None 的引用计数，将其赋给 out_kwd_obj，然后返回 0
    else if (result == 0) {
        Py_INCREF(Py_None);
        *out_kwd_obj = Py_None;
        return 0;
    }
    // 如果 out_kwd_obj 是 PyTuple 类型的对象
    if (PyTuple_CheckExact(*out_kwd_obj)) {
        /*
         * C-API 建议在调用任何 PySequence_Fast* 函数之前调用 PySequence_Fast。
         * 这对于 PyPy 是必需的。
         */
        // 声明 PyObject 指针 seq，用 PySequence_Fast 快速转换 *out_kwd_obj
        PyObject *seq;
        seq = PySequence_Fast(*out_kwd_obj,
                              "Could not convert object to sequence");
        // 如果转换失败，清除 *out_kwd_obj，并返回 -1
        if (seq == NULL) {
            Py_CLEAR(*out_kwd_obj);
            return -1;
        }
        // 获取 PySequence_Fast 转换后的对象数组指针，并赋给 *out_objs
        *out_objs = PySequence_Fast_ITEMS(seq);
        // 将 seq 的引用计数设置给 *out_kwd_obj，并返回 seq 的大小
        Py_SETREF(*out_kwd_obj, seq);
        return PySequence_Fast_GET_SIZE(seq);
    }
    // 如果 out_kwd_obj 不是 PyTuple 类型的对象
    else {
        // 将 out_kwd_obj 的地址赋给 *out_objs，并返回 1
        *out_objs = out_kwd_obj;
        return 1;
    }
}


注释：

# 这是一个单独的右花括号，用于结束某个代码块或函数定义的主体部分
```