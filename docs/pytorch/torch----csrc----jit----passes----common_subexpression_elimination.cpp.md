# `.\pytorch\torch\csrc\jit\passes\common_subexpression_elimination.cpp`

```py
// 引入 Torch 的常见子表达式消除头文件
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>

// 引入 Torch 的别名分析和 IR 相关的头文件
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/node_hashing.h>
#include <torch/csrc/jit/jit_log.h>

// 引入标准库中的无序映射
#include <unordered_map>

// Torch 的命名空间开始
namespace torch {
namespace jit {
namespace {

// 定义常见子表达式消除的结构体
struct CommonSubexpressionEliminator {
  // 构造函数，接受一个图的共享指针
  CommonSubexpressionEliminator(std::shared_ptr<Graph> graph)
      : graph_(std::move(graph)) {}

  // 运行函数，接受一个用于查找父节点的函数对象
  bool run(std::function<Node*(Node*)> parent_lookup_fn) {
    // 调用内部的 run 函数来处理 graph_ 的根块
    return run(graph_->block(), std::move(parent_lookup_fn));
  }

  // 实现常见子表达式消除的函数
  // 因为节点按拓扑顺序访问，所以一次遍历就足够了
  // 如果 CSE 修改了图，则返回 true
  bool run(Block* block, std::function<Node*(Node*)> parent_lookup_fn) {
    // 使用无序集合存储子表达式节点，自定义了节点的哈希和相等比较函数
    std::unordered_set<Node*, HashNode, EqualNode> subexprs;
    // 标记是否有修改发生
    bool changed = false;
    // 遍历基本块中的所有节点
    for (auto it = block->nodes().begin(); it != block->nodes().end(); ++it) {
      auto node = *it;

      // 如果节点是 profile 类型，则不进行公共子表达式消除 (CSE)，因为有单独的处理 pass
      if (node->kind() == prim::profile) {
        GRAPH_DEBUG(
            "Profiled nodes shouldn't be CSE'ed there's a separate pass that does dedup and merging:\n",
            *node);
        continue;
      }

      // 如果节点有副作用，则跳过
      if (node->hasSideEffects()) {
        GRAPH_DEBUG("Node was skipped due to side effects:\n", *node);
        continue;
      }
      
      // 如果节点是非确定性的，则跳过
      if (node->isNondeterministic()) {
        GRAPH_DEBUG("Node was skipped due to its non determinism:\n", *node);
        continue;
      }

      // 如果节点包含子块，则遍历子块
      if (!node->blocks().empty()) {
        // 遍历子块
        for (auto block : node->blocks()) {
          // 对每个子块运行 CSE，并标记是否有变化
          changed |= run(block, [&](Node* n) {
            auto existing = subexprs.find(n);
            if (existing != subexprs.end()) {
              return *existing;
            }

            return parent_lookup_fn(n);
          });
        }

        continue;
      }

      // 如果节点被别名分析结果表示有写入，则跳过
      if (getOrCreateAliasDb().hasWriters(node)) {
        GRAPH_DEBUG("Node was skipped due to alias analysis result:\n", *node);
        // 这些节点不具备足够的信息进行 CSE
        continue;
      }

      // 在父块中检查是否存在 CSE 的机会
      auto parent_lookup = parent_lookup_fn(node);
      auto g_out = node->owningGraph()->outputs();
      if (parent_lookup != nullptr) {
        // 检查是否可以安全地改变节点输出的别名关系
        if (!getOrCreateAliasDb().safeToChangeAliasingRelationship(
                node->outputs(), parent_lookup->outputs())) {
          continue;
        }

        // 执行节点的替换，并更新相关信息
        GRAPH_UPDATE("Replacing\n", *node, "with\n", *parent_lookup);
        changed = true;
        node->replaceAllUsesWith(parent_lookup);
        it.destroyCurrent();
        continue;
      }

      // 检查是否已经存在相同的子表达式
      auto subit = subexprs.insert(node);
      if (!subit.second) {
        // 子表达式已经存在，替换节点的使用，并销毁节点
        auto existing = *subit.first;

        // 确保不引入新的别名关系
        if (getOrCreateAliasDb().mayContainAlias(
                node->outputs(), node->owningGraph()->outputs()) &&
            getOrCreateAliasDb().mayContainAlias(existing->outputs(), g_out)) {
          continue;
        }

        // 执行节点的替换，并更新相关信息
        GRAPH_UPDATE("Replacing\n", *node, "with\n", *existing);
        changed = true;
        node->replaceAllUsesWith(existing);
        // 销毁节点
        it.destroyCurrent();
      }
    }

    // 返回是否有节点被改变
    return changed;
  }

  // 获取或创建别名数据库实例
  AliasDb& getOrCreateAliasDb() {
    if (!alias_db_) {
      alias_db_ = std::make_unique<AliasDb>(graph_);
    }

    return *alias_db_;
  }

 private:
  std::unique_ptr<AliasDb> alias_db_;  // 别名数据库的唯一指针
  std::shared_ptr<Graph> graph_;       // 所操作的图的共享指针
};

} // namespace

// 函数：EliminateCommonSubexpression
// 参数：graph - 指向图对象的共享指针
bool EliminateCommonSubexpression(const std::shared_ptr<Graph>& graph) {
  // 在进行公共子表达式消除前，输出图的状态
  GRAPH_DUMP("Before CSE", graph);
  
  // 创建 CommonSubexpressionEliminator 对象，传入图对象
  CommonSubexpressionEliminator cse(graph);
  
  // 运行公共子表达式消除算法，传入一个 lambda 函数用于处理每个节点
  return cse.run([](Node*) { return nullptr; });
}
} // namespace jit
} // namespace torch
```