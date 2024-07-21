# `.\pytorch\torch\csrc\jit\passes\fuse_relu.h`

```py
#pragma once

// 使用 `#pragma once` 指令确保头文件只被编译一次，避免重复包含的问题


#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>

// 包含 Torch C++ API 的头文件 `<torch/csrc/jit/api/module.h>` 和 `<torch/csrc/jit/ir/ir.h>`


namespace torch {
namespace jit {

// 定义命名空间 `torch::jit`


TORCH_API void FuseAddRelu(script::Module& module);
TORCH_API void FuseAddRelu(std::shared_ptr<Graph>& graph);

// 在 `torch::jit` 命名空间内声明两个函数 `FuseAddRelu`，一个接受 `script::Module&` 类型参数，另一个接受 `std::shared_ptr<Graph>&` 类型参数，这些函数使用了 `TORCH_API` 宏定义，表明它们是 Torch 的 API 函数


} // namespace jit
} // namespace torch

// 命名空间结束符号
```