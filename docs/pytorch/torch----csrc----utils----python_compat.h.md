# `.\pytorch\torch\csrc\utils\python_compat.h`

```py
#ifndef PYTHON_COMPAT
#define PYTHON_COMPAT

#include <torch/csrc/utils/pythoncapi_compat.h>

#ifdef __cplusplus
extern "C" {
#endif

// 定义用于检测Python版本的宏

#define IS_PYTHON_3_11_PLUS PY_VERSION_HEX >= 0x030B00C1
#define IS_PYTHON_3_12_PLUS PY_VERSION_HEX >= 0x030C0000
#define IS_PYTHON_3_13_PLUS PY_VERSION_HEX >= 0x030D0000
#define IS_PYTHON_3_14_PLUS PY_VERSION_HEX >= 0x030E0000

// 获取PyCodeObject结构体中的cellvars数量的函数

PYCAPI_COMPAT_STATIC_INLINE(int)
PyCode_GetNCellvars(PyCodeObject* code) {
// Python 3.11.0rc1版本开始添加了co_ncellvars字段
#if IS_PYTHON_3_11_PLUS
  return code->co_ncellvars;
#else
  return PyTuple_GET_SIZE(code->co_cellvars);
#endif
}

// 获取PyCodeObject结构体中的freevars数量的函数

PYCAPI_COMPAT_STATIC_INLINE(int)
PyCode_GetNFreevars(PyCodeObject* code) {
// Python 3.11.0rc1版本开始添加了co_nfreevars字段
#if IS_PYTHON_3_11_PLUS
  return code->co_nfreevars;
#else
  return PyTuple_GET_SIZE(code->co_freevars);
#endif
}

// CPython提供的函数，但是获取它们的头文件非常困难
extern void _PyWeakref_ClearRef(PyWeakReference* self);

#ifdef __cplusplus
}
#endif
#endif // PYTHON_COMPAT
```