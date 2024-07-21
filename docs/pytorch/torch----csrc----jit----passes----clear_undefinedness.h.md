# `.\pytorch\torch\csrc\jit\passes\clear_undefinedness.h`

```py
#pragma once
// 声明一个预处理指令，确保此头文件只被编译一次

#include <ATen/ATen.h>
// 包含 ATen 库的核心头文件

#include <ATen/core/ivalue.h>
// 包含 ATen 库的 IValue 类相关头文件

#include <ATen/core/jit_type.h>
// 包含 ATen 库的 JIT 类型相关头文件

#include <torch/csrc/Export.h>
// 包含 Torch 导出相关头文件

#include <torch/csrc/jit/ir/ir.h>
// 包含 Torch 的 JIT IR 相关头文件

namespace torch {
namespace jit {

// 声明 torch::jit 命名空间下的 ClearUndefinedness 函数

// Undefinedness makes argument matching fail for regular tensor operations
// if 1+ arguments are undefined or possibly undefined tensors.
// Technically, undefined tensors are **not** tensors as the regular tensor
// operations do not know how to handle them.
// However, in practice, there are guards and conversion operators that
// **always** gate regular operations if undefined tensors may be present
// Eventually, we would love to move to the world where we use optionals
// in lieu of undefined tensors.
// When this happens, this pass will be removed
TORCH_API void ClearUndefinedness(const std::shared_ptr<Graph>& graph);
// 声明 ClearUndefinedness 函数，接受一个 std::shared_ptr<Graph> 参数，并通过 TORCH_API 进行导出

} // namespace jit
} // namespace torch
```