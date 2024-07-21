# `.\pytorch\torch\csrc\jit\passes\inline_autodiff_subgraphs.h`

```
#pragma once
// 预处理指令，确保头文件只被包含一次

#include <torch/csrc/jit/ir/ir.h>
// 包含 Torch 的 JIT 模块中的 IR 相关头文件

namespace torch {
namespace jit {

TORCH_API bool canRunWithAutograd(Node* node);
// 声明一个函数 canRunWithAutograd，用于检查给定节点是否可以使用自动求导运行

TORCH_API void InlineAutodiffSubgraphs(
    std::shared_ptr<Graph>& graph,
    size_t threshold = 5);
// 声明一个函数 InlineAutodiffSubgraphs，用于内联自动微分子图到给定的计算图中，
// 可选参数 threshold 控制内联的阈值

} // namespace jit
} // namespace torch
// 命名空间声明结束
```