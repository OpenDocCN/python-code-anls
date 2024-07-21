# `.\pytorch\torch\csrc\jit\passes\add_if_then_else.cpp`

```py
#include <torch/csrc/jit/passes/add_if_then_else.h>
#include <torch/csrc/jit/runtime/graph_iterator.h>

namespace torch {
namespace jit {

namespace {

// 检查给定的基本块是否没有任何节点
bool hasNoNodes(Block* block) {
  auto nodes = block->nodes();
  // 如果节点的开始迭代器等于结束迭代器，说明没有节点
  return nodes.begin() == nodes.end();
}

// 检查节点是否具有平凡的子块
bool hasTrivialSubBlocks(Node* node) {
  const auto blocks = node->blocks();
  // 确保节点有两个子块
  TORCH_DCHECK_EQ(blocks.size(), 2);

  // 检查两个子块是否都没有节点
  return hasNoNodes(blocks[0]) && hasNoNodes(blocks[1]);
}

} // namespace

// 实现添加 if-then-else 操作的函数
bool AddIfThenElseOp(std::shared_ptr<Graph>& graph) {
  std::vector<Node*> to_replace;
  // 使用深度优先的图节点迭代器遍历图中的节点
  DepthFirstGraphNodeIterator graph_it(graph);
  for (auto* node = graph_it.next(); node != nullptr; node = graph_it.next()) {
    // 如果节点的类型不是 prim::If，则跳过
    if (node->kind() != prim::If) {
      continue;
    }
    // 如果节点的输出不止一个，则跳过
    if (node->outputs().size() != 1) {
      continue;
    }
    // 如果节点具有平凡的子块，则将其加入替换列表中
    if (hasTrivialSubBlocks(node)) {
      to_replace.push_back(node);
    }
  }

  // 遍历需要替换的节点列表
  for (auto* node : to_replace) {
    // 创建一个 prim::IfThenElse 节点
    auto* if_then_else_node = graph->create(prim::IfThenElse, 1);
    // 将原始 if 节点的输入连接到 if-then-else 节点
    if_then_else_node->addInput(node->input());
    // 将第一个子块的返回节点的输入连接到 if-then-else 节点
    auto blocks = node->blocks();
    if_then_else_node->addInput(blocks[0]->return_node()->input());
    // 将第二个子块的返回节点的输入连接到 if-then-else 节点
    if_then_else_node->addInput(blocks[1]->return_node()->input());

    // 在原始 if 节点之前插入 if-then-else 节点
    if_then_else_node->insertBefore(node);
    // 复制输出节点的元数据到 if-then-else 节点的输出节点
    if_then_else_node->output()->copyMetadata(node->output());

    // 替换原始 if 节点的所有使用为 if-then-else 节点的输出节点
    node->output()->replaceAllUsesWith(if_then_else_node->output());
    // 销毁原始 if 节点
    node->destroy();
  }
  // 返回是否替换了任何节点
  return !to_replace.empty();
}

} // namespace jit
} // namespace torch
```