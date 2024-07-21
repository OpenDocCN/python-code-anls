# `.\pytorch\functorch\csrc\dim\dim.h`

```
// 版权声明：Copyright (c) Facebook, Inc. and its affiliates.
// 版权所有。
//
// 此源代码使用 BSD 风格许可证授权，许可证文件位于源代码根目录中的 LICENSE 文件中。
// 使用 pragma once 确保头文件只被编译一次，避免重复包含
#pragma once
// 包含 Python.h 头文件，以便在 C/C++ 中使用 Python API
#include <Python.h>
// Dim_init 函数声明，返回一个 PyObject 指针
PyObject* Dim_init();
```