# `.\numpy\numpy\_core\src\multiarray\compiled_base.h`

```
#ifndef NUMPY_CORE_SRC_MULTIARRAY_COMPILED_BASE_H_
#define NUMPY_CORE_SRC_MULTIARRAY_COMPILED_BASE_H_

#include "numpy/ndarraytypes.h"

// 声明不导出的函数arr_place，用于执行某种数组操作
NPY_NO_EXPORT PyObject *
arr_place(PyObject *, PyObject *, PyObject *);

// 声明不导出的函数arr_bincount，用于计算数组中每个值的出现次数
NPY_NO_EXPORT PyObject *
arr_bincount(PyObject *, PyObject *const *, Py_ssize_t, PyObject *);

// 声明不导出的函数arr__monotonicity，用于处理数组的单调性
NPY_NO_EXPORT PyObject *
arr__monotonicity(PyObject *, PyObject *, PyObject *kwds);

// 声明不导出的函数arr_interp，用于在数组上进行插值
NPY_NO_EXPORT PyObject *
arr_interp(PyObject *, PyObject *const *, Py_ssize_t, PyObject *, PyObject *);

// 声明不导出的函数arr_interp_complex，用于在复数数组上进行插值
NPY_NO_EXPORT PyObject *
arr_interp_complex(PyObject *, PyObject *const *, Py_ssize_t, PyObject *, PyObject *);

// 声明不导出的函数arr_ravel_multi_index，用于将多维数组索引展平为一维索引
NPY_NO_EXPORT PyObject *
arr_ravel_multi_index(PyObject *, PyObject *, PyObject *);

// 声明不导出的函数arr_unravel_index，用于将一维索引展开为多维数组索引
NPY_NO_EXPORT PyObject *
arr_unravel_index(PyObject *, PyObject *, PyObject *);

// 声明不导出的函数arr_add_docstring，用于给数组对象添加文档字符串
NPY_NO_EXPORT PyObject *
arr_add_docstring(PyObject *, PyObject *const *, Py_ssize_t);

// 声明不导出的函数io_pack，用于打包数据到某种格式
NPY_NO_EXPORT PyObject *
io_pack(PyObject *, PyObject *, PyObject *);

// 声明不导出的函数io_unpack，用于从某种格式解包数据
NPY_NO_EXPORT PyObject *
io_unpack(PyObject *, PyObject *, PyObject *);

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_COMPILED_BASE_H_ */
```