# `.\pytorch\torch\csrc\jit\passes\inline_autodiff_subgraphs.cpp`

```py
// 包含头文件：inline_autodiff_subgraphs.h、ir.h、dead_code_elimination.h、update_differentiable_graph_requires_grad.h、subgraph_utils.h
#include <torch/csrc/jit/passes/inline_autodiff_subgraphs.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/update_differentiable_graph_requires_grad.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>

// 命名空间定义：torch::jit
namespace torch {
namespace jit {

// 检查节点是否能够与 Autograd 一起运行
bool canRunWithAutograd(Node* node) {
  auto kind = node->kind();
  // 递归遍历节点内的所有块，检查其中的节点是否都能与 Autograd 一起运行
  for (Block* block : node->blocks()) {
    if (!std::all_of(
            block->nodes().begin(), block->nodes().end(), canRunWithAutograd)) {
      return false;
    }
  }
  // 返回节点是否属于 Autograd 能够处理的类型
  return kind != prim::FusionGroup && kind != prim::CudaFusionGroup &&
      kind != prim::TypeCheck && kind != prim::TensorExprGroup &&
      kind != prim::CudaFusionGuard && kind != prim::oneDNNFusionGroup &&
      kind != prim::oneDNNFusionGuard && (kind.is_aten() || kind.is_prim());
}

// 匿名命名空间下的函数定义

// 递归计算块的节点数目
size_t blockSize(Block* block) {
  size_t num = 0;
  for (Node* n : block->nodes()) {
    for (Block* b : n->blocks()) {
      num += blockSize(b);
    }
    num++;
  }
  return num;
}

// 扫描节点并处理 Autodiff 子图
graph_node_list::iterator scanNode(Node* node, size_t threshold) {
  auto next_node = ++node->iterator();

  // 递归地处理节点内的所有块
  for (Block* block : node->blocks()) {
    InlineAutodiffSubgraphs(block, threshold);
  }

  // 如果节点不是 DifferentiableGraph 类型，则直接返回下一个节点迭代器
  if (node->kind() != prim::DifferentiableGraph) {
    return next_node;
  }

  // 获取节点的子图和子图的大小
  auto subgraph = node->g(attr::Subgraph);
  size_t subgraph_size = blockSize(subgraph->block());

  // 如果子图大小超过阈值，则直接返回下一个节点迭代器
  if (subgraph_size >= threshold) {
    return next_node;
  }

  // 检查子图内的所有节点是否都能与 Autograd 一起运行
  if (!std::all_of(
          subgraph->nodes().begin(),
          subgraph->nodes().end(),
          canRunWithAutograd)) {
    return next_node;
  }

  // 更新子图内张量的 requires_grad 属性
  UpdateDifferentiableGraphRequiresGrad(subgraph, c10::nullopt);
  // 解除节点的子图合并
  SubgraphUtils::unmergeSubgraph(node);
  return next_node;
}

// 处理块内的节点，对 Autodiff 子图进行内联处理
void InlineAutodiffSubgraphs(Block* block, size_t threshold) {
  for (auto it = block->nodes().begin(); it != block->nodes().end();) {
    it = scanNode(*it, threshold);
  }
}

} // namespace jit
} // namespace torch
```