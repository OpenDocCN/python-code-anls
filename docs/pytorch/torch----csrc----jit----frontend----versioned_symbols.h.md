# `.\pytorch\torch\csrc\jit\frontend\versioned_symbols.h`

```py
#pragma once
// 预处理指令，确保头文件只被编译一次

#include <caffe2/serialize/versions.h>
// 包含 Caffe2 序列化版本的头文件

#include <torch/csrc/Export.h>
// 包含 Torch 导出相关的头文件

#include <torch/csrc/jit/api/module.h>
// 包含 Torch JIT 模块 API 的头文件

#include <cstdint>
// 包含标准整数类型的头文件

namespace torch {
namespace jit {
// 定义 torch::jit 命名空间，用于包裹 Torch JIT 相关功能

// 根据给定的名称和版本号，映射到对应版本实现的符号。
// 参见注释 [Versioned Symbols]
TORCH_API Symbol
get_symbol_for_version(const Symbol name, const uint64_t version);
// 声明函数 get_symbol_for_version，返回一个符号对象

// 将给定的节点类型映射到支持它的最小版本号。
// 参见注释 [Dynamic Versions and torch.jit.save vs. torch.save]
TORCH_API uint64_t get_min_version_for_kind(const NodeKind& kind);
// 声明函数 get_min_version_for_kind，返回一个最小版本号

} // namespace jit
} // namespace torch
// 结束 torch::jit 命名空间和 torch 命名空间
```