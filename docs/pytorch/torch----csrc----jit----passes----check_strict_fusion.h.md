# `.\pytorch\torch\csrc\jit\passes\check_strict_fusion.h`

```py
#pragma once


// 指令，用于确保头文件只被编译一次，防止重复包含

#include <torch/csrc/jit/ir/ir.h>


// 包含 Torch 库中的 IR 相关头文件，用于操作图形表示的中间表示（Intermediate Representation, IR）

namespace torch {
namespace jit {


// Torch 命名空间开始

TORCH_API void CheckStrictFusion(std::shared_ptr<Graph>& graph);


// 声明 CheckStrictFusion 函数，接受一个指向图形的共享指针作为参数，用于检查严格融合（fusion）的条件

} // namespace jit
} // namespace torch


// Torch 命名空间结束
```