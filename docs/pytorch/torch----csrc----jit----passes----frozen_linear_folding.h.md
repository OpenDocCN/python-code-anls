# `.\pytorch\torch\csrc\jit\passes\frozen_linear_folding.h`

```
#pragma once
// 预处理指令，确保头文件只被包含一次

#include <torch/csrc/jit/ir/ir.h>
// 包含 Torch 深度学习框架中的 IR 头文件，用于图表达的中间表示

namespace torch {
namespace jit {

// 命名空间 torch::jit 下的内容

// 将 Linear -> BatchNormNd 融合成单个 Linear 操作，
// 通过将 BatchNorm 的权重折叠进 Linear 的权重中实现。
// 这个优化只在冻结图中有效；否则无效。
TORCH_API bool FoldFrozenLinearBatchnorm(std::shared_ptr<Graph>& graph);
// 函数声明：尝试将图中的 Linear -> BatchNormNd 合并成单个 Linear 操作，
// 并通过将 BatchNorm 的权重折叠进 Linear 的权重中来实现。

} // namespace jit
} // namespace torch
```