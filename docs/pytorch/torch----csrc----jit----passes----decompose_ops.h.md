# `.\pytorch\torch\csrc\jit\passes\decompose_ops.h`

```py
#pragma once


// 使用预处理指令 #pragma once，确保当前头文件只被编译一次，避免重复包含

#include <torch/csrc/jit/ir/ir.h>


// 包含了 Torch 的 JIT 模块中的 IR 头文件，用于操作和处理图结构

namespace torch {
namespace jit {


// 声明了一个命名空间 torch::jit，用于封装 Torch 的 JIT 模块相关的功能

TORCH_API void DecomposeOps(std::shared_ptr<Graph>& graph);


// 声明了一个函数 DecomposeOps，该函数接受一个 std::shared_ptr 智能指针参数 graph，
// 用于对图结构进行操作，TORCH_API 指定了该函数为 Torch 的 API 函数

}
} // namespace torch


// 结束了命名空间 torch::jit 的定义
```