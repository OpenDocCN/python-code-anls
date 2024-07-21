# `.\pytorch\torch\csrc\jit\passes\erase_number_types.h`

```py
// 预处理指令，指示编译器在编译此头文件时只包含一次，避免重复包含
#pragma once

// 包含 Torch 库中的 IR 相关头文件
#include <torch/csrc/jit/ir/ir.h>

// 定义了 torch 命名空间
namespace torch {
// 定义了 jit 命名空间，包含了 JIT 编译器相关的功能
namespace jit {

// 删除 NumberType 类型信息的函数声明，用于导出到 ONNX 格式
// 该函数确保没有剩余的值具有 NumberType 类型，替换为张量类型
// 删除 NumberType 信息的具体操作包括：
// - 将 NumberType 输出改为 DynamicType
// - 将是数字的 prim::Constant 节点改为对应类型的零维张量
// - 删除 prim::TensorToNum、aten::Float、aten::Int 和 prim::NumToTensor 节点
TORCH_API void EraseNumberTypes(const std::shared_ptr<Graph>& graph);

// 在给定的基本块上删除 NumberType 信息的函数声明
// 用于逐块操作，确保基本块中没有剩余的 NumberType 类型信息
TORCH_API void EraseNumberTypesOnBlock(Block* block);

} // namespace jit
} // namespace torch
```