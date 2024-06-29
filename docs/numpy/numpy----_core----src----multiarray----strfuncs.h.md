# `.\numpy\numpy\_core\src\multiarray\strfuncs.h`

```
#ifndef NUMPY_CORE_SRC_MULTIARRAY_STRFUNCS_H_
#define NUMPY_CORE_SRC_MULTIARRAY_STRFUNCS_H_

// 声明一个不导出的函数，用于设置对象的字符串表示函数
NPY_NO_EXPORT void
PyArray_SetStringFunction(PyObject *op, int repr);

// 声明一个不导出的函数，用于返回数组对象的字符串表示形式
NPY_NO_EXPORT PyObject *
array_repr(PyArrayObject *self);

// 声明一个不导出的函数，用于返回数组对象的字符串表示形式
NPY_NO_EXPORT PyObject *
array_str(PyArrayObject *self);

// 声明一个不导出的函数，用于根据指定格式返回数组对象的字符串表示形式
NPY_NO_EXPORT PyObject *
array_format(PyArrayObject *self, PyObject *args);

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_STRFUNCS_H_ */
```