# `.\numpy\numpy\_core\src\common\npy_pycompat.h`

```py
#ifndef NUMPY_CORE_SRC_COMMON_NPY_PYCOMPAT_H_
// 如果未定义宏 NUMPY_CORE_SRC_COMMON_NPY_PYCOMPAT_H_，则开始条件编译
#define NUMPY_CORE_SRC_COMMON_NPY_PYCOMPAT_H_

// 包含 numpy/npy_3kcompat.h 头文件，用于兼容 Python 3k
#include "numpy/npy_3kcompat.h"

// 包含 pythoncapi-compat/pythoncapi_compat.h 头文件，用于兼容 Python C API
#include "pythoncapi-compat/pythoncapi_compat.h"

/*
 * 在 Python 3.10a7 (或 b1) 中，当值为 NaN 时，python 开始使用其自身的哈希值。
 * 参见 https://bugs.python.org/issue43475
 */
#if PY_VERSION_HEX > 0x030a00a6
// 如果 Python 版本大于 3.10a7，则定义 Npy_HashDouble 宏为 _Py_HashDouble
#define Npy_HashDouble _Py_HashDouble
#else
// 否则，定义静态内联函数 Npy_HashDouble
static inline Py_hash_t
Npy_HashDouble(PyObject *NPY_UNUSED(identity), double val)
{
    // 返回 _Py_HashDouble 函数对 val 的哈希值
    return _Py_HashDouble(val);
}
#endif

// 结束条件编译
#endif  /* NUMPY_CORE_SRC_COMMON_NPY_PYCOMPAT_H_ */
```