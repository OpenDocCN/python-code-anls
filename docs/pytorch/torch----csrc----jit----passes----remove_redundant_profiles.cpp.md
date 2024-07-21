# `.\pytorch\torch\csrc\jit\passes\remove_redundant_profiles.cpp`

```
// 包含 Torch 的 JIT 模块中的头文件，用于死代码消除和移除冗余 profile 的操作
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/remove_redundant_profiles.h>

// 包含 Torch 的 JIT 模块中的别名分析和 IR 视图的头文件
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/ir_views.h>
#include <torch/csrc/jit/jit_log.h>

// Torch 的 JIT 命名空间
namespace torch {
namespace jit {

// 移除给定块中的冗余 profile 节点
void RemoveRedundantProfiles(Block* block, AliasDb& db) {
  // 从块的最后一个节点开始逆向迭代处理
  for (auto it = block->nodes().end()->reverseIterator();
       it != block->nodes().begin();) {
    Node* n = *it;
    it++;

    // 递归处理节点中的子块
    for (Block* b : n->blocks()) {
      RemoveRedundantProfiles(b, db);
    }

    // 只检查 prim::profile 节点，不处理 prim::profile_ivalue 节点
    if (n->kind() != prim::profile ||
        n->input()->node()->kind() != prim::profile) {
      continue;
    }

    // 获取输入节点，并检查其 profiled_type 属性是否与当前节点一致
    Node* input_node = n->input()->node();
    if (input_node->ty(attr::profiled_type) != n->ty(attr::profiled_type)) {
      continue;
    }

    // 尝试将当前节点移动到输入节点之前，保持拓扑顺序
    if (!db.moveBeforeTopologicallyValid(input_node, n)) {
      continue;
    }

    // 替换当前节点的输出使用为输入节点，然后销毁当前节点
    n->output()->replaceAllUsesWith(n->input());
    n->destroy();
  }
}

// 对给定图中的所有节点执行移除冗余 profile 操作
void RemoveRedundantProfiles(std::shared_ptr<Graph>& graph) {
  AliasDb db(graph);
  RemoveRedundantProfiles(graph->block(), db);
}

} // namespace jit
} // namespace torch
```