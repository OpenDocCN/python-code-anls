# `D:\src\scipysrc\scipy\scipy\sparse\sparsetools\sparsetools.h`

```
#ifndef SPARSETOOLS_H
#define SPARSETOOLS_H

#include <Python.h>  // 包含 Python C API 的头文件
#include "numpy/ndarrayobject.h"  // 包含 NumPy 的 ndarray 对象相关头文件

#include <stdexcept>  // 包含标准异常类的头文件

#include "bool_ops.h"  // 包含布尔操作的头文件
#include "complex_ops.h"  // 包含复杂操作的头文件

typedef PY_LONG_LONG thunk_t(int I_typenum, int T_typenum, void **args);  // 定义 thunk_t 类型，表示接受特定参数并返回 PY_LONG_LONG 类型的函数指针

PyObject *
call_thunk(char ret_spec, const char *spec, thunk_t *thunk, PyObject *args);  // 函数声明：调用 thunk 函数，并指定返回类型和参数列表

#endif  // 结束 SPARSETOOLS_H 头文件的条件编译
```