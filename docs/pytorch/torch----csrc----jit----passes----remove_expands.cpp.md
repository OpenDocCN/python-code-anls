# `.\pytorch\torch\csrc\jit\passes\remove_expands.cpp`

```py
#include <torch/csrc/jit/passes/remove_expands.h>

namespace torch {
namespace jit {

// 递归函数，用于移除图中所有节点的展开操作
static void RemoveExpands(Block* block) {
  // 遍历当前块中的所有节点
  for (auto it = block->nodes().begin(), end = block->nodes().end(); it != end;
       ++it) {
    // 对于节点中的子块，递归调用 RemoveExpands 函数
    for (auto sub : it->blocks())
      RemoveExpands(sub);

    // 如果当前节点是展开操作且为隐式展开
    if (it->kind() == aten::expand && it->get<bool>(attr::implicit) == true) {
      // 将当前节点的输出替换为其自身的命名输入
      it->output()->replaceAllUsesWith(it->namedInput(attr::self));
      // 销毁当前节点
      it.destroyCurrent();
    }
  }
}

// 对外接口函数，从图中移除所有节点的展开操作
void RemoveExpands(const std::shared_ptr<Graph>& graph) {
  RemoveExpands(graph->block());
}

} // namespace jit
} // namespace torch
```