# `.\pytorch\torch\csrc\jit\passes\canonicalize_graph_fuser_ops.h`

```
#pragma once

// 使用 `#pragma once` 预处理指令，确保头文件只被包含一次，以防止多重包含带来的问题。


#include <torch/csrc/jit/ir/ir.h>

// 包含 Torch 库中的 `<torch/csrc/jit/ir/ir.h>` 头文件，这是用于 JIT（即时编译）的图形表示的相关功能。


namespace torch {
namespace jit {

// 声明一个命名空间 `torch`，在其中嵌套命名空间 `jit`，用于组织 Torch 的 JIT 模块的相关功能。


TORCH_API void CanonicalizeOps(const std::shared_ptr<Graph>& graph);

// 在 `torch::jit` 命名空间中声明一个函数 `CanonicalizeOps`，该函数接受一个指向 `Graph` 类的 `shared_ptr`，用于对操作进行规范化。


} // namespace jit
} // namespace torch

// 结束 `torch::jit` 命名空间和 `torch` 命名空间的定义。
```