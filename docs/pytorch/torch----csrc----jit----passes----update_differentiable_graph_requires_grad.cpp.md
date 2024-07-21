# `.\pytorch\torch\csrc\jit\passes\update_differentiable_graph_requires_grad.cpp`

```
// 包含 Torch 的 JIT 框架中所需的头文件

#include <torch/csrc/jit/passes/update_differentiable_graph_requires_grad.h>

// 包含 Torch 的 JIT 框架中 IR 相关的头文件
#include <torch/csrc/jit/ir/ir.h>

// 包含 Torch 的 JIT 框架中子图工具函数的头文件
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>

// 定义了 Torch 的 JIT 命名空间
namespace torch {
namespace jit {

// 更新不同iable图中需要梯度的节点信息
static void UpdateDifferentiableGraphRequiresGrad(
    Block* block,                             // 输入参数为一个 Block 对象指针
    std::optional<bool> new_requires_grad) {  // 可选的新 requires_grad 值
  // 遍历当前 block 中的每一个节点
  for (Node* n : block->nodes()) {
    // 遍历节点 n 的每一个输入值
    for (Value* v : n->inputs()) {
      // 尝试将输入值的类型转换为 TensorType
      auto ty = v->type()->cast<TensorType>();
      // 如果成功转换为 TensorType
      if (ty) {
        // 更新该输入值的类型，设置其 requires_grad 属性为新值
        v->setType(ty->withRequiresGrad(new_requires_grad));
      }
    }
    // 如果当前节点的类型是 prim::profile
    if (n->kind() == prim::profile) {
      // 更新节点的 profiled_type 属性，设置其 requires_grad 属性为新值
      n->ty_(
          attr::profiled_type,
          n->ty(attr::profiled_type)
              ->expectRef<TensorType>()
              .withRequiresGrad(new_requires_grad));
    }
    // 递归调用，更新当前节点的每个子块
    for (Block* b : n->blocks()) {
      UpdateDifferentiableGraphRequiresGrad(b, new_requires_grad);
    }
  }
}

// 对外暴露的函数接口，用于更新不同iable图中需要梯度的节点信息
void UpdateDifferentiableGraphRequiresGrad(
    std::shared_ptr<Graph>& diff_forward_graph,  // 输入参数为 Graph 对象的智能指针
    std::optional<bool> new_requires_grad) {    // 可选的新 requires_grad 值
  // 调用内部实现函数，更新传入图的起始块
  UpdateDifferentiableGraphRequiresGrad(
      diff_forward_graph->block(), new_requires_grad);
}

} // namespace jit
} // namespace torch
```