# `.\numpy\numpy\_core\src\multiarray\flagsobject.h`

```py
#ifndef NUMPY_CORE_SRC_FLAGSOBJECT_H_
#define NUMPY_CORE_SRC_FLAGSOBJECT_H_

/* Array Flags Object */
// 定义了一个结构体 PyArrayFlagsObject，用于表示数组的标志信息
typedef struct PyArrayFlagsObject {
        PyObject_HEAD
        PyObject *arr;   // 指向数组对象的指针
        int flags;       // 数组的标志位
} PyArrayFlagsObject;

// 导出了 PyArrayFlags_Type 类型对象
extern NPY_NO_EXPORT PyTypeObject PyArrayFlags_Type;

// 创建并返回一个新的数组标志对象
NPY_NO_EXPORT PyObject *
PyArray_NewFlagsObject(PyObject *obj);

// 更新数组对象的标志位
NPY_NO_EXPORT void
PyArray_UpdateFlags(PyArrayObject *ret, int flagmask);

#endif  /* NUMPY_CORE_SRC_FLAGSOBJECT_H_ */
```