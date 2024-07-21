# `.\pytorch\torch\csrc\jit\passes\quantization\fusion_passes.h`

```py
#pragma once


// 使用 #pragma once 指令确保头文件只被编译一次，防止多重包含的问题



#include <torch/csrc/jit/ir/ir.h>


// 包含 Torch 的 JIT 模块中的 IR 头文件，用于操作和表示计算图的中间表示（Intermediate Representation）



namespace torch {
namespace jit {


// 声明了一个命名空间 torch::jit，用于包含 Torch 的 JIT 模块相关的内容



TORCH_API void FuseQuantizedAddRelu(std::shared_ptr<Graph>& graph);


// 声明了一个公开的函数 FuseQuantizedAddRelu，用于在给定的计算图中融合量化的加法和 ReLU 操作
// 参数是一个指向 Graph 对象的 shared_ptr，表示计算图的指针
// TORCH_API 表示该函数是 Torch 提供的 API，并且可能被外部代码调用



} // namespace jit
} // namespace torch


// 命名空间的结束声明，确保 torch::jit 命名空间的封闭
```