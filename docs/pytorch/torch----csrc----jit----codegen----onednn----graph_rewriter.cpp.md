# `.\pytorch\torch\csrc\jit\codegen\onednn\graph_rewriter.cpp`

```
// 引入头文件，包括使用的库和工具
#include <torch/csrc/jit/codegen/onednn/graph_fuser.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>

// 命名空间声明，定义嵌套的命名空间结构
namespace torch {
namespace jit {
namespace fuser {
namespace onednn {

// 清理子图函数定义
void GraphRewriter::cleanupSubgraphs() {
  // 获取当前节点的逆向迭代器
  auto curNode = *block_->nodes().rbegin();
  // 循环直到遍历完所有节点
  while (curNode != *block_->nodes().rend()) {
    // 保存前一个节点的迭代器，因为下一个块中可能会删除当前节点
    auto prevNode = curNode->prev();
    // 如果当前节点是 LLGA 子图
    if (llgaHelper_.isLlgaSubgraph(curNode)) {
      // 如果由于失败的别名检查，未能将所有分区中的节点放入子图，则取消合并子图
      llgaHelper_.unmergeIfAnyNodeIsMissing(curNode);
    }
    // 将当前节点更新为前一个节点，继续迭代
    curNode = prevNode;
  }
  // 递归处理每个节点的子块
  for (Node* n : block_->nodes()) {
    for (Block* b : n->blocks()) {
      GraphRewriter(b, graph_, aliasDb_).cleanupSubgraphs();
    }
  }
}

// 构建子图函数定义
void GraphRewriter::buildupSubgraphs() {
  // 需要多次运行重写器，以获取所有合并的机会。
  // 这是因为 moveBeforeTopologicalValid 可能会重新排序节点，使其在当前迭代点之后。
  // 为了正确考虑这些节点进行合并，需要运行该 pass，直到没有更改为止。
  //
  // 示例:
  //   c = f(a, b)
  //   d = f(c)
  //   e = f(d)  <- 迭代点在此处，向上移动
  // 在 c.moveBeforeTopologicallyValid(e) 后，我们有:
  //   c = f(a, b)
  //   e = f(d)  <- 迭代点仍在这里
  //   d = f(c)  <- 这是在另一侧移动的节点。
  // 参见 [workblocks]
  auto workblocks = buildWorkBlocks();
  // 对每个工作块进行循环处理
  for (auto& workblock : workblocks) {
    bool any_changed = true;
    // 当有更改时继续循环
    while (any_changed) {
      any_changed = false;
      // 获取工作块的末尾和开头的逆向迭代器
      auto workblock_end = workblock.end()->reverseIterator();
      auto workblock_begin = workblock.begin()->reverseIterator();
      // 对工作块中的节点进行逆序迭代处理
      for (auto it = workblock_end; it != workblock_begin;) {
        bool changed = false;
        // 扫描节点，并检查是否有更改
        std::tie(it, changed) = scanNode(*it, workblock_begin);
        any_changed |= changed;
      }
    }
  }

  // 递归构建子块的子图
  for (Node* n : block_->nodes()) {
    for (auto subBlock : n->blocks()) {
      GraphRewriter(subBlock, graph_, aliasDb_).buildupSubgraphs();
    }
  }
}

// 命名空间结束
} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch
// 构建工作块列表，用于子图重写器处理
std::vector<WorkBlock> GraphRewriter::buildWorkBlocks() {
  // [workblocks]
  // IR 中存在许多节点是不可重新排序的，如 prim::Bailout。
  // 如果节点 N 被两个不可重新排序的节点 A 和 B 包围，
  // 那么从 N 创建的融合组只能包含来自 (A, B) 的节点。
  // 从 A 到 B 的节点表示子图重写器工作的一个工作块。
  // 通过提前创建这些工作块，避免每次 scanNode 返回时重新遍历整个图块
  Node* end_bound_node = block_->return_node();
  Node* curr = end_bound_node->prev();
  std::vector<WorkBlock> worklist;
  while (curr != block_->param_node()) {
    // 如果节点具有副作用，则不能在其周围重新排序
    if (curr->hasSideEffects()) {
      worklist.emplace_back(curr, end_bound_node);
      end_bound_node = curr;
    }
    curr = curr->prev();
  }
  worklist.emplace_back(curr, end_bound_node);
  return worklist;
}

// 扫描节点，尝试合并节点并返回迭代器和合并是否成功的状态
std::pair<graph_node_list::iterator, bool> GraphRewriter::scanNode(
    Node* consumer,
    graph_node_list::iterator workblock_begin) {
  GRAPH_DEBUG("Scanning ", consumer->kind().toQualString());
  if (llgaHelper_.shouldConsiderForMerge(consumer)) {
    // 如果不应考虑将节点合并到 LLGA 子图中，则创建一个单节点子图
    if (!llgaHelper_.isLlgaSubgraph(consumer)) {
      consumer = llgaHelper_.createSingletonSubgraph(consumer, aliasDb_);
    }
    // 迭代工作块以合并由 LLGA 图助手确定的同一分区中的节点
    auto prev = ++consumer->reverseIterator();
    for (auto it = prev; it != workblock_begin; it++) {
      if (auto group = tryMerge(consumer, *it)) {
        // 成功合并后，新组的 `inputs` 可能已更改，因此重新扫描以寻找更多合并机会
        return std::make_pair(group.value()->reverseIterator(), true);
      }
    }
  }
  return std::make_pair(++consumer->reverseIterator(), false);
}

// 尝试将 `producer` 合并到 `consumer` 中。如果成功，销毁 `producer` 并返回 `consumer` 组
std::optional<Node*> GraphRewriter::tryMerge(Node* consumer, Node* producer) {
  AT_ASSERT(llgaHelper_.isLlgaSubgraph(consumer));
  // 判断是否可以合并，条件包括 LLGA 图助手判断和在拓扑上的合法移动
  bool canMerge = llgaHelper_.shouldMerge(producer, consumer) &&
      aliasDb_.moveBeforeTopologicallyValid(producer, consumer);
  if (!canMerge) {
    return c10::nullopt;
  }
  // 将节点合并到 LLGA 子图中
  llgaHelper_.mergeNodeIntoSubgraph(producer, consumer, aliasDb_);
  return consumer;
}
```