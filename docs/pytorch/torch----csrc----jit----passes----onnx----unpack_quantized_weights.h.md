# `.\pytorch\torch\csrc\jit\passes\onnx\unpack_quantized_weights.h`

```py
#pragma once

#include <torch/csrc/jit/api/module.h> // 包含 Torch 的模块 API 头文件
#include <torch/csrc/jit/ir/ir.h> // 包含 Torch 的 IR 头文件
#include <torch/csrc/onnx/onnx.h> // 包含 Torch 的 ONNX 头文件

#include <memory> // 包含内存管理相关的头文件

namespace torch { // Torch 命名空间
namespace jit { // JIT 命名空间

// 解压量化权重函数声明，接受图和参数字典作为参数
TORCH_API void UnpackQuantizedWeights(
    std::shared_ptr<Graph>& graph, // 图的智能指针
    std::map<std::string, IValue>& paramsDict); // 参数字典

// 插入置换操作函数声明，接受图和参数字典作为参数
TORCH_API void insertPermutes(
    std::shared_ptr<Graph>& graph, // 图的智能指针
    std::map<std::string, IValue>& paramsDict); // 参数字典

} // namespace jit
} // namespace torch
```