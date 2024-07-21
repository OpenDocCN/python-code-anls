# `.\pytorch\torch\csrc\jit\passes\utils\op_registry.h`

```
#pragma once
// 预处理指令，确保头文件只被编译一次

#include <torch/csrc/Export.h>
// 包含 Torch 导出宏的头文件

#include <torch/csrc/jit/ir/ir.h>
// 包含 Torch JIT 中 IR 操作的头文件

#include <memory>
// 包含内存管理相关的标准库头文件

namespace torch {
namespace jit {
// Torch 的 JIT 模块命名空间开始

// 从 shape_analysis.cpp 移动而来

// 要求：
//   dims           : 从第一个参数中保留
//   scalar type    : 从第一个参数中保留（不需要与其他参数匹配）
//   device         : 始终匹配并保留
//   tensor inputs  : *
//   tensor outputs : 1
// 注意：这些操作（稍作调整）是重新启动的好候选。
//     通常了解权重或偏置的类型和设备就足以推断输出类型。
std::shared_ptr<OperatorSet> nn_ops_first_input_preserving();

// 要求：
//   dims           : 从第一个参数中更改
//   scalar type    : 从第一个参数中保留
//   device         : 始终匹配并保留
//   tensor inputs  : 1
//   tensor outputs : 1
std::shared_ptr<OperatorSet> ops_one_tensor_in_shape_transform();
} // namespace jit
} // namespace torch
// Torch 的 JIT 模块命名空间结束
```