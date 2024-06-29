# `.\numpy\numpy\_core\src\umath\override.h`

```py
#ifndef _NPY_UMATH_OVERRIDE_H
#define _NPY_UMATH_OVERRIDE_H

定义了 `_NPY_UMATH_OVERRIDE_H` 宏，用于防止头文件的多重包含。


#include "npy_config.h"
#include "numpy/ufuncobject.h"

包含了两个头文件：`npy_config.h` 和 `numpy/ufuncobject.h`，这些文件包含了所需的配置信息和函数声明。


NPY_NO_EXPORT int
PyUFunc_CheckOverride(PyUFuncObject *ufunc, char *method,
        PyObject *in_args, PyObject *out_args, PyObject *wheremask_obj,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames,
        PyObject **result);

声明了一个名为 `PyUFunc_CheckOverride` 的函数，其参数包括一个指向 `PyUFuncObject` 结构的指针 `ufunc`，以及一系列用于函数调用和返回结果的参数。


#endif

结束了条件编译指令，确保头文件内容在多次包含时不会重复定义。
```