# `.\pytorch\torch\csrc\jit\passes\insert_guards.h`

```py
// 命令预处理器指令，指定此头文件只包含一次
#pragma once

// 包含 PyTorch 核心头文件
#include <ATen/ATen.h>
// 包含 PyTorch 核心的 IValue 类定义
#include <ATen/core/ivalue.h>
// 包含 PyTorch 核心的 JIT 类型定义
#include <ATen/core/jit_type.h>
// 包含 PyTorch 核心的执行栈定义
#include <ATen/core/stack.h>
// 包含 Torch 库导出定义
#include <torch/csrc/Export.h>
// 包含 Torch JIT 中间表示(IR)的头文件
#include <torch/csrc/jit/ir/ir.h>

// 包含标准库中的列表容器定义
#include <list>
// 包含标准库中的向量容器定义
#include <vector>

// Torch 命名空间开始
namespace torch {
// JIT 命名空间开始
namespace jit {

// 定义 TORCH_API 下的函数 InsertGuards，用于向图中插入保护节点
TORCH_API void InsertGuards(std::shared_ptr<Graph> graph);

// 定义 TORCH_API 下的函数 RemoveProfilingNodes，用于移除图中的性能分析节点
TORCH_API void RemoveProfilingNodes(const std::shared_ptr<Graph>& graph);

// JIT 命名空间结束
} // namespace jit
// Torch 命名空间结束
} // namespace torch
```