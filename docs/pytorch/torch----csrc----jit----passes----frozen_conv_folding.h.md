# `.\pytorch\torch\csrc\jit\passes\frozen_conv_folding.h`

```py
#pragma once


// 声明了一个预处理指令，确保此头文件在编译时只被包含一次

#include <torch/csrc/jit/ir/ir.h>


// 包含了一个外部库的头文件，用于访问 Torch 的 JIT IR（Intermediate Representation）

namespace torch {
namespace jit {


// 进入了 torch::jit 命名空间，用于组织和区分代码

// 将卷积（Convolution）和批归一化（Batchnorm）融合为单个卷积操作，
// 通过将批归一化的权重融合到卷积权重中来实现。
// 此优化只在冻结图（Frozen Graphs）上有效；否则不进行任何操作。
TORCH_API bool FoldFrozenConvBatchnorm(std::shared_ptr<Graph>& graph);

// 将卷积（Convolution）和加法（Add）/减法（Sub）融合为单个卷积操作，
// 通过将加法或减法的常量张量融合到卷积权重中来实现。
// 此优化只在冻结图（Frozen Graphs）上有效；否则不进行任何操作。
TORCH_API bool FoldFrozenConvAddOrSub(std::shared_ptr<Graph>& graph);

// 将卷积（Convolution）和乘法（Mul）/除法（Div）融合为单个卷积操作，
// 通过将乘法或除法的常量张量融合到卷积权重中来实现。
// 此优化只在冻结图（Frozen Graphs）上有效；否则不进行任何操作。
TORCH_API bool FoldFrozenConvMulOrDiv(std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
```