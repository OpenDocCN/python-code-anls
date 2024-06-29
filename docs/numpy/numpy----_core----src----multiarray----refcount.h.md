# `.\numpy\numpy\_core\src\multiarray\refcount.h`

```
#ifndef NUMPY_CORE_SRC_MULTIARRAY_REFCOUNT_H_
#define NUMPY_CORE_SRC_MULTIARRAY_REFCOUNT_H_

// 声明一个不导出的函数，用于清除数组描述符指向数据的缓冲区
NPY_NO_EXPORT int
PyArray_ClearBuffer(
        PyArray_Descr *descr, char *data,
        npy_intp stride, npy_intp size, int aligned);

// 声明一个不导出的函数，用于清除整个数组对象
NPY_NO_EXPORT int
PyArray_ClearArray(PyArrayObject *arr);

// 声明一个不导出的函数，用于增加数组项的引用计数
NPY_NO_EXPORT void
PyArray_Item_INCREF(char *data, PyArray_Descr *descr);

// 声明一个不导出的函数，用于减少数组项的引用计数
NPY_NO_EXPORT void
PyArray_Item_XDECREF(char *data, PyArray_Descr *descr);

// 声明一个不导出的函数，用于增加数组对象的引用计数
NPY_NO_EXPORT int
PyArray_INCREF(PyArrayObject *mp);

// 声明一个不导出的函数，用于减少数组对象的引用计数
NPY_NO_EXPORT int
PyArray_XDECREF(PyArrayObject *mp);

// 声明一个不导出的函数，用于将数组对象的元素设置为None对象
NPY_NO_EXPORT int
PyArray_SetObjectsToNone(PyArrayObject *arr);

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_REFCOUNT_H_ */
```