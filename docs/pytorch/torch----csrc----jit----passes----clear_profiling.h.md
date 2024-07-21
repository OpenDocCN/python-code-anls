# `.\pytorch\torch\csrc\jit\passes\clear_profiling.h`

```py
#pragma once
// 只有当该头文件还没有被包含过时，才会编译它，确保头文件只被包含一次

#include <ATen/ATen.h>
// 包含 ATen 库的头文件

#include <ATen/core/ivalue.h>
// 包含 ATen 库的 IValue 类型定义的头文件

#include <ATen/core/jit_type.h>
// 包含 ATen 库的 JIT 类型定义的头文件

#include <torch/csrc/Export.h>
// 包含 Torch 的导出宏的头文件

#include <torch/csrc/jit/ir/ir.h>
// 包含 Torch JIT 的 IR 操作相关的头文件

namespace torch {
namespace jit {

TORCH_API void unprofileGraphInputs(const std::shared_ptr<Graph>& graph);
// 声明一个函数 unprofileGraphInputs，用于取消图中所有输入的分析

TORCH_API void unprofileBlock(Block* start_block);
// 声明一个函数 unprofileBlock，用于取消块中所有节点输出的分析

// Unprofiles all the node outputs in a block.
// 取消块中所有节点输出的分析

TORCH_API void ClearProfilingInformation(const std::shared_ptr<Graph>& graph);
// 声明一个函数 ClearProfilingInformation，用于清除图中的所有分析信息

} // namespace jit
} // namespace torch
```