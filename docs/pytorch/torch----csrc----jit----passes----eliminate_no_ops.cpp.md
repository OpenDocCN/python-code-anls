# `.\pytorch\torch\csrc\jit\passes\eliminate_no_ops.cpp`

```
#include <torch/csrc/jit/passes/eliminate_no_ops.h>

#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/runtime/graph_iterator.h>

namespace torch {
namespace jit {

namespace {

// 检查节点的所有输入是否都是张量类型
bool allInputsAreTensors(Node* node) {
  for (const auto* value : node->inputs()) {
    const auto& type = value->type();
    // 如果输入不是张量类型，则返回 false
    if (!type->castRaw<TensorType>()) {
      return false;
    }
  }
  return true;
}

// 检查节点是否不能进行优化
bool cannotOptimize(Node* node) {
  const auto kind = node->kind();
  // 如果节点是 __is__ 或者 __isnot__，则尝试优化其输入是否都是张量类型
  if (kind == aten::__is__ || kind == aten::__isnot__) {
    return allInputsAreTensors(node);
  }
  return false;
}

// 检查图中是否包含不能进行优化的操作
// 某些操作会使得移除无操作变得不安全，比如 detach 操作
bool containsInvalidOp(std::shared_ptr<Graph>& graph) {
  for (auto* node : graph->nodes()) {
    if (cannotOptimize(node)) {
      return true;
    }
  }
  return false;
}

} // namespace

// 删除无操作节点的函数
bool EliminateNoOps(
    std::shared_ptr<Graph>& graph,
    std::unordered_set<c10::Symbol> custom_ops) {
  // 打印优化前的图形状态
  GRAPH_DUMP("Before EliminateNoOps: ", graph);
  // 如果图中包含不能优化的操作，则直接返回 false
  if (containsInvalidOp(graph)) {
    return false;
  }

  // 设置不执行的操作集合，例如 detach
  std::unordered_set<c10::Symbol> no_ops{aten::detach};
  no_ops.insert(custom_ops.begin(), custom_ops.end());

  bool changed = false;

  // 深度优先遍历图中的节点
  auto graph_it = DepthFirstGraphNodeIterator(graph);
  for (auto* node = graph_it.next(); node != nullptr; node = graph_it.next()) {
    // 查找当前节点是否是无操作节点
    auto it = no_ops.find(node->kind());
    if (it == no_ops.end()) {
      continue;
    }

    // 如果找到无操作节点，则进行优化替换
    changed = true;
    node->output()->replaceAllUsesWith(node->input(0));
  }

  // 如果图结构发生了改变，则进行死代码消除
  if (changed) {
    EliminateDeadCode(graph);
  }

  // 打印优化后的图形状态
  GRAPH_DUMP("After EliminateNoOps: ", graph);
  return changed;
}

} // namespace jit
} // namespace torch
```