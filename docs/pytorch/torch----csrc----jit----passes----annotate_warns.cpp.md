# `.\pytorch\torch\csrc\jit\passes\annotate_warns.cpp`

```py
#include <torch/csrc/jit/passes/annotate_warns.h>

#include <atomic>

namespace torch {
namespace jit {

// 静态函数：AnnotateWarns，用于向图中的节点添加警告注释ID
static void AnnotateWarns(Block* b) {
  // 静态原子整型变量idx，用于为每个警告节点分配唯一的ID
  static std::atomic<int64_t> idx(0);
  
  // 遍历块b中的每个节点n
  for (Node* n : b->nodes()) {
    // 遍历节点n中的每个子块child_b，并递归调用AnnotateWarns函数
    for (Block* child_b : n->blocks()) {
      AnnotateWarns(child_b);
    }

    // 如果节点n的类型不是aten::warn，则跳过当前循环
    if (n->kind() != aten::warn) {
      continue;
    }

    // 给节点n设置属性attr::warn_id，值为当前idx的值，并递增idx
    n->i_(attr::warn_id, idx);
    idx++;
  }
}

// 公共函数：AnnotateWarns，用于向图中的节点添加警告注释ID
void AnnotateWarns(const std::shared_ptr<Graph>& graph) {
  // 调用私有函数AnnotateWarns，从图的根块开始向其节点添加警告注释ID
  AnnotateWarns(graph->block());
}

} // namespace jit
} // namespace torch
```