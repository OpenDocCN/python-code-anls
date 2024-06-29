# `.\numpy\numpy\_core\src\common\npy_dlpack.h`

```
// 引入 Python.h 头文件，用于与 Python 解释器交互
#include "Python.h"
// 引入 dlpack.h 头文件，这是 DLPack 库的头文件
#include "dlpack/dlpack.h"

#ifndef NPY_DLPACK_H
#define NPY_DLPACK_H

// Array API 规范的一部分，定义 DLPack Capsule 的名称
#define NPY_DLPACK_CAPSULE_NAME "dltensor"
#define NPY_DLPACK_VERSIONED_CAPSULE_NAME "dltensor_versioned"
#define NPY_DLPACK_USED_CAPSULE_NAME "used_dltensor"
#define NPY_DLPACK_VERSIONED_USED_CAPSULE_NAME "used_dltensor_versioned"

// NumPy 内部使用的 Capsule 名称，用于存储基本对象
// 因为需要释放对原始 Capsule 的引用
#define NPY_DLPACK_INTERNAL_CAPSULE_NAME "numpy_dltensor"
#define NPY_DLPACK_VERSIONED_INTERNAL_CAPSULE_NAME "numpy_dltensor_versioned"

// Array API 规范，导出函数，从 PyArrayObject 创建 DLPack 对象
NPY_NO_EXPORT PyObject *
array_dlpack(PyArrayObject *self, PyObject *const *args, Py_ssize_t len_args,
             PyObject *kwnames);

// Array API 规范，导出函数，返回与 PyArrayObject 相关的设备信息
NPY_NO_EXPORT PyObject *
array_dlpack_device(PyArrayObject *self, PyObject *NPY_UNUSED(args));

// 从 DLPack 对象创建 Python 对象，这是 Array API 规范的一部分
NPY_NO_EXPORT PyObject *
from_dlpack(PyObject *NPY_UNUSED(self),
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames);

#endif
```