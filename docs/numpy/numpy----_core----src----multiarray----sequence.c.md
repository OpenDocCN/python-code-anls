# `.\numpy\numpy\_core\src\multiarray\sequence.c`

```
/* 定义以 NPY_API_VERSION 为基准的 NumPy 废弃 API 版本 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

/* 定义 _MULTIARRAYMODULE，可能用于标识多维数组模块 */
#define _MULTIARRAYMODULE

/* 清除 PY_SSIZE_T_CLEAN，确保 Py_ssize_t 被定义正确 */
#define PY_SSIZE_T_CLEAN

/* 引入 Python.h 头文件，提供 Python C API 的核心功能 */
#include <Python.h>

/* 引入 structmember.h 头文件，用于定义结构体成员和属性 */
#include <structmember.h>

/* 引入 NumPy 数组对象的头文件 */
#include "numpy/arrayobject.h"

/* 引入 NumPy 数组标量的头文件 */
#include "numpy/arrayscalars.h"

/* 引入 NumPy 配置文件的头文件 */
#include "npy_config.h"

/* 引入公共功能的头文件 */
#include "common.h"

/* 引入映射功能的头文件 */
#include "mapping.h"

/* 引入序列功能的头文件 */
#include "sequence.h"

/* 引入计算功能的头文件 */
#include "calculation.h"

/*************************************************************************
 ****************   实现序列协议 Implement Sequence Protocol **************************
 *************************************************************************/

/* 
   一些内容在 array_as_mapping 协议中也有重复。但是
   我们在这里填写它，以便 PySequence_XXXX 调用按预期工作
*/

/* 
   检查数组是否包含指定元素 el。
   相当于 (self == el).any() 的操作
*/
static int
array_contains(PyArrayObject *self, PyObject *el)
{
    int ret;
    PyObject *res, *any;

    /* 确保将 self 和 el 作为 PyObject 比较，并返回比较结果 */
    res = PyArray_EnsureAnyArray(PyObject_RichCompare((PyObject *)self,
                                                      el, Py_EQ));
    if (res == NULL) {
        return -1;
    }

    /* 对 res 应用 any 函数，检查是否存在非零元素 */
    any = PyArray_Any((PyArrayObject *)res, NPY_RAVEL_AXIS, NULL);
    Py_DECREF(res);
    if (any == NULL) {
        return -1;
    }

    /* 检查 any 对象是否为真 */
    ret = PyObject_IsTrue(any);
    Py_DECREF(any);
    return ret;
}

/* 
   尝试对数组进行连接操作时，抛出类型错误。
   注意：在 PyPy 上运行时，不会抛出此错误。
*/
static PyObject *
array_concat(PyObject *self, PyObject *other)
{
    PyErr_SetString(PyExc_TypeError,
            "Concatenation operation is not implemented for NumPy arrays, "
            "use np.concatenate() instead. Please do not rely on this error; "
            "it may not be given on all Python implementations.");
    return NULL;
}

/* 定义序列协议方法 */
NPY_NO_EXPORT PySequenceMethods array_as_sequence = {
    (lenfunc)array_length,                  /* sq_length */
    (binaryfunc)array_concat,               /* sq_concat for operator.concat */
    (ssizeargfunc)NULL,                     /* sq_repeat */
    (ssizeargfunc)array_item,               /* sq_item */
    (ssizessizeargfunc)NULL,                /* sq_slice */
    (ssizeobjargproc)array_assign_item,     /* sq_ass_item */
    (ssizessizeobjargproc)NULL,             /* sq_ass_slice */
    (objobjproc) array_contains,            /* sq_contains */
    (binaryfunc) NULL,                      /* sq_inplace_concat */
    (ssizeargfunc)NULL                      /* sq_inplace_repeat */
};
```