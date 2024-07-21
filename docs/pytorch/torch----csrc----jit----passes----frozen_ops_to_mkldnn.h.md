# `.\pytorch\torch\csrc\jit\passes\frozen_ops_to_mkldnn.h`

```py
#pragma once

// 使用 `#pragma once` 预处理指令确保头文件只被编译一次，防止多重包含问题


#include <torch/csrc/jit/ir/ir.h>

// 包含 Torch 库中的 IR 相关头文件，用于操作中间表示 (IR)


namespace torch {
namespace jit {

// 声明一个命名空间 `torch::jit`，用于包裹 Torch JIT 框架的相关代码


// Converts operators & their parameters to mkldnn if it is profitable
// Currently encompassing Conv2d and Conv3d, and Linear
// Op must be in float32 and mkldnn must be built
// This pass only works on frozen graph
TORCH_API void ConvertFrozenOpsToMKLDNN(std::shared_ptr<Graph>& graph);

// 声明函数 `ConvertFrozenOpsToMKLDNN`，将运算符及其参数转换为 MKL-DNN 格式（如果有利可图）
// 目前支持 Conv2d、Conv3d 和 Linear 操作
// 要求操作必须是 float32 类型，并且必须构建了 MKL-DNN
// 此转换仅适用于冻结图（frozen graph）


} // namespace jit
} // namespace torch

// 命名空间结束声明，结束了 `torch::jit` 命名空间以及 `torch` 命名空间
```