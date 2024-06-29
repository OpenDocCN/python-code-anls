# `.\numpy\numpy\_core\src\multiarray\convert.h`

```py
#ifndef NUMPY_CORE_SRC_MULTIARRAY_CONVERT_H_
#define NUMPY_CORE_SRC_MULTIARRAY_CONVERT_H_

NPY_NO_EXPORT int
PyArray_AssignZero(PyArrayObject *dst,
                   PyArrayObject *wheremask);



#ifndef NUMPY_CORE_SRC_MULTIARRAY_CONVERT_H_
#define NUMPY_CORE_SRC_MULTIARRAY_CONVERT_H_

// 定义了一个不导出的整型函数 PyArray_AssignZero，接受两个 PyArrayObject 类型的参数
NPY_NO_EXPORT int
PyArray_AssignZero(PyArrayObject *dst,
                   PyArrayObject *wheremask);

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_CONVERT_H_ */
```