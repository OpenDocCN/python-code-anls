# `.\pytorch\torch\csrc\jit\passes\onnx\remove_inplace_ops_for_onnx.h`

```py
#pragma once

# 使用 `#pragma once` 预处理指令，确保头文件只被包含一次，防止多重包含导致的编译错误和重定义问题


#include <torch/csrc/jit/ir/ir.h>

# 包含 Torch 框架中提供的头文件 `torch/csrc/jit/ir/ir.h`，用于引入 IR 图相关的定义和功能


namespace torch {
namespace jit {

# 定义命名空间 `torch::jit`，用于封装 Torch 框架中的 JIT（即时编译）功能


TORCH_API void RemoveInplaceOpsForONNX(
    const std::shared_ptr<Graph>& graph,
    Module* model);

# 声明一个函数 `RemoveInplaceOpsForONNX`，其功能是从给定的图 `graph` 中移除原地操作（inplace operations），这些操作可能在导出为 ONNX 格式时会引发问题。函数的参数是一个指向 `Graph` 的共享指针 `graph` 和一个指向 `Module` 的指针 `model`。


} // namespace jit
} // namespace torch

# 命名空间闭合：结束 `torch::jit` 命名空间和 `torch` 命名空间的定义。
```