# `.\numpy\numpy\_core\src\common\get_attr_string.h`

```
#ifndef NUMPY_CORE_SRC_COMMON_GET_ATTR_STRING_H_
#define NUMPY_CORE_SRC_COMMON_GET_ATTR_STRING_H_

#include <Python.h>
#include "ufunc_object.h"

// 定义内联函数，用于检查是否为基本的 Python 类型
static inline npy_bool
_is_basic_python_type(PyTypeObject *tp)
{
    return (
        /* 基本的数值类型 */
        tp == &PyBool_Type ||
        tp == &PyLong_Type ||
        tp == &PyFloat_Type ||
        tp == &PyComplex_Type ||

        /* 基本的序列类型 */
        tp == &PyList_Type ||
        tp == &PyTuple_Type ||
        tp == &PyDict_Type ||
        tp == &PySet_Type ||
        tp == &PyFrozenSet_Type ||
        tp == &PyUnicode_Type ||
        tp == &PyBytes_Type ||

        /* 其他内建类型 */
        tp == &PySlice_Type ||
        tp == Py_TYPE(Py_None) ||
        tp == Py_TYPE(Py_Ellipsis) ||
        tp == Py_TYPE(Py_NotImplemented) ||

        /* TODO: ndarray，但是在此处我们无法看到 PyArray_Type */

        /* 结尾的哨兵，用于吸收末尾的 || */
        NPY_FALSE
    );
}


/*
 * 查找特殊方法，遵循 Python 的查找方式，查找类型对象而不是实例本身。
 *
 * 假设特殊方法是特定于 numpy 的，因此不查看内建类型。但会检查基本的 ndarray 和 numpy 标量类型。
 *
 * 未来可以更像 _Py_LookupSpecial 的实现。
 */
static inline PyObject *
PyArray_LookupSpecial(PyObject *obj, PyObject *name_unicode)
{
    PyTypeObject *tp = Py_TYPE(obj);

    /* 不需要在简单类型上检查特殊属性 */
    if (_is_basic_python_type(tp)) {
        return NULL;
    }
    
    // 获取类型对象 tp 的特定属性
    PyObject *res = PyObject_GetAttr((PyObject *)tp, name_unicode);

    // 处理属性获取异常
    if (res == NULL && PyErr_ExceptionMatches(PyExc_AttributeError)) {
        PyErr_Clear();
    }

    return res;
}


/*
 * PyArray_LookupSpecial_OnInstance:
 *
 * 实现了不正确的特殊方法查找规则，违反了 Python 的约定，查找实例而不是类型。
 *
 * 为了向后兼容而保留。未来应该弃用此功能。
 */
static inline PyObject *
PyArray_LookupSpecial_OnInstance(PyObject *obj, PyObject *name_unicode)
{
    PyTypeObject *tp = Py_TYPE(obj);

    /* 不需要在简单类型上检查特殊属性 */
    if (_is_basic_python_type(tp)) {
        return NULL;
    }

    // 获取实例对象 obj 的特定属性
    PyObject *res = PyObject_GetAttr(obj, name_unicode);

    // 处理属性获取异常
    if (res == NULL && PyErr_ExceptionMatches(PyExc_AttributeError)) {
        PyErr_Clear();
    }

    return res;
}

#endif  /* NUMPY_CORE_SRC_COMMON_GET_ATTR_STRING_H_ */
```