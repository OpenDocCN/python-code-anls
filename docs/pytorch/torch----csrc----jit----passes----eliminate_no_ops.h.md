# `.\pytorch\torch\csrc\jit\passes\eliminate_no_ops.h`

```
#pragma once
// 使用 #pragma once 防止头文件被多次包含

#include <torch/csrc/jit/ir/ir.h>
// 包含 Torch 库中的 IR 相关头文件

namespace torch {
namespace jit {

// 在前向传播过程中移除不执行任何操作的操作（例如 aten::detach）。
// 此 Pass 作为 freeze_module 的一部分被调用。
// 该函数还接受一个自定义操作集合来消除。所有这些操作在输入中必须将其输出作为第一个输入，即 x = f(x, ...)
TORCH_API bool EliminateNoOps(
    std::shared_ptr<Graph>& graph,
    std::unordered_set<c10::Symbol> custom_ops = {});
    // 函数签名，接受一个指向图的共享指针和一个可选的自定义操作的无序集合，返回布尔值

} // namespace jit
} // namespace torch
```