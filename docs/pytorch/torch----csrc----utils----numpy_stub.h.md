# `.\pytorch\torch\csrc\utils\numpy_stub.h`

```py
#pragma once

// 使用 `#pragma once` 预处理指令确保头文件只被包含一次，避免重复定义


#include <torch/csrc/python_headers.h>

// 包含 Torch 深度学习库的 Python 头文件，用于与 Python 解释器交互


#ifdef USE_NUMPY

// 如果定义了 `USE_NUMPY` 宏，则编译以下内容，用于支持 NumPy 数组


#if !defined(NO_IMPORT_ARRAY) && !defined(WITH_NUMPY_IMPORT_ARRAY)
#define NO_IMPORT_ARRAY
#endif

// 如果未定义 `NO_IMPORT_ARRAY` 和 `WITH_NUMPY_IMPORT_ARRAY` 宏，则定义 `NO_IMPORT_ARRAY`，防止重复导入数组


#ifndef PY_ARRAY_UNIQUE_SYMBOL
#define PY_ARRAY_UNIQUE_SYMBOL __numpy_array_api
#endif

// 如果未定义 `PY_ARRAY_UNIQUE_SYMBOL` 宏，则定义为 `__numpy_array_api`，确保 NumPy 数组在多个模块中唯一标识


#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif

// 如果未定义 `NPY_NO_DEPRECATED_API` 宏，则定义为 `NPY_1_7_API_VERSION`，禁用 NumPy 的已弃用 API


#include <numpy/arrayobject.h>

// 包含 NumPy 的数组对象头文件，提供对 NumPy 数组操作的支持


#endif // USE_NUMPY

// 结束 `USE_NUMPY` 宏的条件编译块
```