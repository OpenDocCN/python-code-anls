# `.\pytorch\torch\csrc\jit\passes\update_differentiable_graph_requires_grad.h`

```
#pragma once
// 预处理指令，确保本头文件只被编译一次

#include <torch/csrc/jit/ir/ir.h>
// 包含 Torch 的 IR 头文件，用于操作图结构

namespace torch {
namespace jit {

// 命名空间 torch::jit

// 不同iable 图会分离输入张量的梯度，创建和内联 differentiable 图会改变
// 图中张量的 requires_grad 属性。这个 pass 更新 prim::profiles 节点的
// requires_grad，以保持 profiled 属性的最新状态。它不会更新其它节点
// 的 grad 属性，比如图的输入，因为 grad 属性的唯一下游使用者是 profiling
// executor，它只使用 prim::profiles 的类型信息。
TORCH_API void UpdateDifferentiableGraphRequiresGrad(
    std::shared_ptr<Graph>& diff_forward_graph,
    std::optional<bool> new_requires_grad);
// 函数声明，用于更新 differentiable 图中 prim::profiles 节点的 requires_grad 属性

} // namespace jit
} // namespace torch
// 命名空间 torch::jit 结束
```