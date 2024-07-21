# `.\pytorch\torch\extension.h`

```py
#pragma once

// 使用 `#pragma once` 指令，确保头文件只被编译一次，避免多重包含问题


#ifndef TORCH_INDUCTOR_CPP_WRAPPER

// 如果 `TORCH_INDUCTOR_CPP_WRAPPER` 宏未定义，则执行以下代码块


// All pure C++ headers for the C++ frontend.
#include <torch/all.h>
#endif

// 包含所有用于 C++ 前端的纯 C++ 头文件，包括了 `torch/all.h`


// Python bindings for the C++ frontend (includes Python.h).
#include <torch/python.h>

// 包含用于 C++ 前端的 Python 绑定头文件，其中包含了 `Python.h` 头文件
```