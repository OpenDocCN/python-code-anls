# `.\pytorch\torch\csrc\jit\passes\constant_pooling.cpp`

```py
#include <torch/csrc/jit/passes/constant_pooling.h>

#include <ATen/core/symbol.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/node_hashing.h>
#include <unordered_set>

namespace torch {
namespace jit {

namespace {

// 类似于常见的子表达式消除（common subexpression elimination）的过程
// 将所有常量移动到图的开头，并进行去重
void ConstantPooling(
    Block* block,
    std::unordered_set<Node*, HashNode, EqualNode>& constants,
    const AliasDb& aliasDb) {
  // 遍历块中的每个节点
  for (auto it = block->nodes().begin(); it != block->nodes().end();) {
    auto node = *it;
    // 可能会将节点移动到不同的块，因此此处先增加迭代器
    ++it;
    // 如果节点包含子块，则递归处理子块
    if (!node->blocks().empty()) {
      for (auto block : node->blocks()) {
        ConstantPooling(block, constants, aliasDb);
      }
      continue;
    }

    // 如果节点不是常量节点，则继续下一个节点
    if (node->kind() != prim::Constant) {
      continue;
    }

    // 检查是否已经存在相同的常量节点
    auto subit = constants.insert(node);
    if (!subit.second) {
      auto existing = *subit.first;

      auto old_ivalue = toIValue(existing->output());
      auto new_ivalue = toIValue(node->output());

      // 如果两个值是同一个对象，则不需要考虑改变别名关系
      bool same_identity =
          (old_ivalue && new_ivalue && (old_ivalue->is(new_ivalue)));

      // 如果不是同一对象，并且不能安全地改变别名关系，则继续下一个节点
      if (!same_identity &&
          !aliasDb.safeToChangeAliasingRelationship(
              node->outputs(), existing->outputs())) {
        continue;
      }

      // 如果常量已存在，则用现有常量节点替换使用当前节点的地方，并销毁当前节点
      node->replaceAllUsesWith(existing);
      node->destroy();
      continue;
    }

    // 将常量定义移动到图的开头
    auto first_node = node->owningGraph()->block()->nodes().front();
    if (node != first_node)
      node->moveBefore(first_node);
  }
}
} // anonymous namespace

// 对外接口函数，用于进行常量池化操作
void ConstantPooling(const std::shared_ptr<Graph>& graph) {
  // 创建别名分析对象，用于分析别名关系
  AliasDb aliasDb(graph);
  // 用于存储常量节点的集合
  std::unordered_set<Node*, HashNode, EqualNode> constants;
  // 调用常量池化函数处理图的顶层块
  ConstantPooling(graph->block(), constants, aliasDb);
}
} // namespace jit
} // namespace torch
```