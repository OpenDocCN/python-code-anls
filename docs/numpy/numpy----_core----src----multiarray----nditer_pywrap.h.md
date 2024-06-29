# `.\numpy\numpy\_core\src\multiarray\nditer_pywrap.h`

```py
#ifndef NUMPY_CORE_SRC_MULTIARRAY_NDITER_PYWRAP_H_
// 如果未定义 NUMPY_CORE_SRC_MULTIARRAY_NDITER_PYWRAP_H_ 宏，则开始条件编译保护

#define NUMPY_CORE_SRC_MULTIARRAY_NDITER_PYWRAP_H_

// 声明一个不导出的函数 NpyIter_NestedIters，接受三个 PyObject 类型的参数
// 第一个参数 self 没有使用，参数 args 和 kwds 分别表示位置参数和关键字参数
NPY_NO_EXPORT PyObject *
NpyIter_NestedIters(PyObject *NPY_UNUSED(self),
                    PyObject *args, PyObject *kwds);

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_NDITER_PYWRAP_H_ */
// 结束条件编译保护，确保只有在未定义 NUMPY_CORE_SRC_MULTIARRAY_NDITER_PYWRAP_H_ 宏时才会包含上述内容
```