# `.\pytorch\torch\csrc\jit\passes\peephole.h`

```
// 预处理指令，确保本头文件只被包含一次
#pragma once

// 包含 Torch 的 JIT 模块中的 IR 头文件
#include <torch/csrc/jit/ir/ir.h>

// Torch 的命名空间
namespace torch {
namespace jit {

// 对给定的图进行Peephole优化，返回true如果图被修改
TORCH_API bool PeepholeOptimize(
    const std::shared_ptr<Graph>& graph,
    bool disable_shape_peepholes = false);

// 对给定的基本块进行Peephole优化，返回true如果块被修改
TORCH_API bool PeepholeOptimize(
    Block* block,
    bool disable_shape_peepholes = false);

// 对给定的图执行AddMM融合操作，返回true如果图被修改
TORCH_API bool FuseAddMM(const std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
```