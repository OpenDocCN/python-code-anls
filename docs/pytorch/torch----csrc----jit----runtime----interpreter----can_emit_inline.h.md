# `.\pytorch\torch\csrc\jit\runtime\interpreter\can_emit_inline.h`

```py
/*
#pragma once

#include <memory>

#include <torch/csrc/jit/ir/ir.h>

namespace torch::jit::interpreter {
/*
This is an optimization that reduces the number of store/load/move nodes needed
by recognizing that parts of the graph are simple trees like a*x + b*y. When
this happens it is possible to work directly off of the stack by emitting the
tree in a depth-first left-to-right manner:
  load a
  load x
  mul # stack now is a*x
  load b
  load y
  mul # stack now is a*x, b*y
  add

can_emit_inline_[node] == true means that this node participates as a non-root
member of one of these trees. The code emitter will not emit this node when
it is encountered in the node. Instead the node is emitted in a depth first
traversal from where it is used in a tree.

To participate in a tree a node must have a single use (otherwise it is not
tree-like) and output a single value (for simplicity.) If our IR was functional,
these would be the only constraints. However, many nodes have side effects, so
we must ensure that emitting the nodes in depth first order from the tree's root
_does not reorder the emission of the nodes_. To ensure this, we work backward
from the root of a potential tree, visiting its inputs in reverse depth first
order, while scanning the node list backward (with the block_point node). When
these traversal line up we know it is safe to emit the tree in this way. We
ignore constant nodes, which do not have side effects.
*/

// 结构体 CanEmitInline，用于确定节点是否可以内联
struct CanEmitInline {
  explicit CanEmitInline(Graph& graph) {
    // 初始化时扫描图的块以确定内联能力
    scanBlock(graph.block());
  }
  
  // 判断是否可以内联指定的值 v
  bool canInline(Value* v) {
    return v->node()->kind() != prim::Param &&
        v->node()->kind() != prim::TensorExprGroup &&
        v->node()->kind() != prim::TensorExprDynamicGroup &&
        v->node()->kind() != prim::StaticSubgraph &&
        v->node()->kind() != prim::CudaFusionGroup &&
        v->node()->kind() != prim::FusionGroup &&
        v->node()->kind() != prim::BailOut && 
        v->uses().size() == 1 && // 仅有一个使用者
        v->node()->outputs().size() == 1; // 仅输出一个值
  }

  // 返回前一个非常量节点
  Node* previousNonConstant(Node* n) {
    do {
      n = n->prev();
    } while (n->kind() == prim::Constant); // 忽略常量节点
    return n;
  }

  // 扫描值的节点，确保可以安全地内联
  Node* scanValue(Node* block_point, Value* v) {
    // 如果反向扫描的节点列表与值 v 的使用位置对齐，可以安全地以树的顺序发出
    // 继续扫描更多节点
    // 如果节点 `v` 是 `block_point` 并且可以内联化，则执行以下操作
    if (v->node() == block_point && canInline(v)) {
      // 因为我们已经内联了这个节点，所以可能可以递归地内联其输入节点，因此继续扫描它
      block_point = scanNode(v->node());
      // 标记这个节点可以内联
      can_emit_inline_[v->node()] = true;
    }
    // 如果条件不符合，说明无法内联节点 `v`，只能为其生成加载/移动指令。然而，其它输入可能仍然按树顺序出现，因此继续扫描输入
    return block_point;
  }

  // 扫描节点 `n` 及其子节点
  Node* scanNode(Node* n) {
    // 如果节点 `n` 已经确定可以内联，则不再扫描
    if (can_emit_inline_.count(n)) {
      return nullptr;
    }
    // 遍历节点 `n` 的所有块
    for (auto b : n->blocks()) {
      scanBlock(b);
    }
    // 查找节点 `n` 中第一个非常量节点
    Node* block_point = previousNonConstant(n);
    // 逆序遍历节点 `n` 的输入，并扫描每个输入节点
    for (auto it = n->inputs().rbegin(), end = n->inputs().rend(); it != end;
         ++it) {
      block_point = scanValue(block_point, *it);
    }
    return block_point;
  }

  // 扫描块 `b` 中的节点
  void scanBlock(Block* b) {
    // 扫描块的返回节点
    scanNode(b->return_node());
    // 逆序遍历块 `b` 中的所有节点
    for (auto node : b->nodes().reverse()) {
      scanNode(node);
    }
  }
  // 用于存储可以被内联的节点的哈希表
  std::unordered_map<Node*, bool> can_emit_inline_;
};

// 结束命名空间 torch::jit::interpreter
} // namespace torch::jit::interpreter
```