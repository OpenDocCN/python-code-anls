# `.\pytorch\torch\csrc\jit\tensorexpr\tensorexpr_init.h`

```py
// 防止头文件被重复包含
#pragma once

// 包含 Torch 的 Python 绑定头文件
#include <torch/csrc/jit/python/pybind.h>
// 包含 Torch 的 Python 工具头文件
#include <torch/csrc/utils/pybind.h>

// Torch 命名空间下的 JIT 命名空间
namespace torch {
namespace jit {
// 初始化张量表达式的 Python 绑定
void initTensorExprBindings(PyObject* module);
} // namespace jit
} // namespace torch
```