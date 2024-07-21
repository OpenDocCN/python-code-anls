# `.\pytorch\torch\csrc\python_headers.h`

```py
#pragma once
// 使用 pragma once 来确保头文件只被包含一次，防止重复定义

// workaround for https://github.com/python/cpython/pull/23326
// 为了解决 Python 代码库中的一个问题，详细信息可以在指定的 GitHub 链接中找到
#include <cmath>
#include <complex>

// workaround for Python 2 issue: https://bugs.python.org/issue17120
// 解决 Python 2 中的问题，这个问题似乎也影响到了 Python 3
#pragma push_macro("_XOPEN_SOURCE")
#pragma push_macro("_POSIX_C_SOURCE")
// 取消 _XOPEN_SOURCE 和 _POSIX_C_SOURCE 的定义，以便设置适当的宏

#include <Python.h>
#include <frameobject.h>
#include <structseq.h>

#pragma pop_macro("_XOPEN_SOURCE")
#pragma pop_macro("_POSIX_C_SOURCE")
// 恢复 _XOPEN_SOURCE 和 _POSIX_C_SOURCE 的定义为之前的状态

#ifdef copysign
#undef copysign
#endif
// 如果定义了 copysign 宏，则取消它的定义，以便后面重新定义

#if PY_MAJOR_VERSION < 3
#error "Python 2 has reached end-of-life and is no longer supported by PyTorch."
#endif
// 如果 Python 主版本号小于 3，则输出错误信息，指出 Python 2 已经不再由 PyTorch 支持
```