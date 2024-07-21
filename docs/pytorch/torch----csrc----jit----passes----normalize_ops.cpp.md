# `.\pytorch\torch\csrc\jit\passes\normalize_ops.cpp`

```
// 引入 Torch 库中的 normalize_ops.h 头文件，其中包含了对操作规范化的相关功能
#include <torch/csrc/jit/passes/normalize_ops.h>

// 引入 C++ 标准库中的异常处理头文件
#include <c10/util/Exception.h>

// Torch 命名空间
namespace torch {
// Torch JIT 命名空间
namespace jit {

// 匿名命名空间，用于定义内部辅助函数和数据结构

// 函数：normalizeOpAliases
// 功能：将操作的别名映射到标准形式，以便优化操作
bool normalizeOpAliases(graph_node_list_iterator& iter) {
  // 在操作别名映射表中查找当前节点的操作别名
  auto alias = getOperatorAliasMap().find(iter->kind());
  if (alias != getOperatorAliasMap().end()) {
    // 将当前节点替换为新的操作符号
    iter->replaceWithNewSymbol(alias->second);
    // 销毁当前节点
    iter.destroyCurrent();
    return true;
  }
  return false;
}

// 函数：normalizeRSub
// 功能：规范化 rsub 操作为 sub 操作
bool normalizeRSub(graph_node_list_iterator& iter) {
  // 匹配 rsub 操作的特定模式
  if (iter->matches(
          "aten::rsub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor")) {
    // 获取操作的输入参数
    ArrayRef<Value*> args = iter->inputs();
    // 用 sub 操作替换当前 rsub 操作
    Node* newSub = iter->replaceWithNewSymbol(aten::sub);
    // 调整 sub 操作的输入顺序
    newSub->replaceInput(0, args[1]);
    newSub->replaceInput(1, args[0]);
    // 销毁当前 rsub 节点
    iter.destroyCurrent();
    return true;
  }
  return false;
}

// 函数：normalizeIsBool
// 功能：将 __is__ 和 __isnot__ 操作与布尔值比较规范化为 eq 和 ne 操作
bool normalizeIsBool(graph_node_list_iterator& iter) {
  // 获取操作的输入参数
  ArrayRef<Value*> args = iter->inputs();
  // 检查操作是否是布尔值比较且参数个数为 2
  if (args.size() == 2 && args[0]->type() == BoolType::get() &&
      args[1]->type() == BoolType::get()) {
    // 如果是 __is__ 操作，则替换为 eq 操作
    if (iter->kind() == aten::__is__) {
      iter->replaceWithNewSymbol(aten::eq);
      iter.destroyCurrent();
      return true;
    }
    // 如果是 __isnot__ 操作，则替换为 ne 操作
    if (iter->kind() == aten::__isnot__) {
      iter->replaceWithNewSymbol(aten::ne);
      iter.destroyCurrent();
      return true;
    }
  }
  return false;
}

// 函数：NormalizeOps
// 功能：规范化图中所有节点的操作
void NormalizeOps(Block* block) {
  // 遍历当前块中的所有节点
  for (auto it = block->nodes().begin(), end = block->nodes().end();
       it != end;) {
    // 递归规范化当前节点包含的子块
    for (auto sub : it->blocks()) {
      NormalizeOps(sub);
    }

    // 尝试规范化当前节点为 sub 操作
    if (normalizeRSub(it)) {
      continue;
    }

    // 尝试规范化当前节点的操作别名
    if (normalizeOpAliases(it)) {
      continue;
    }

    // 尝试规范化当前节点的布尔值比较操作
    if (normalizeIsBool(it)) {
      continue;
    }

    // 移动到下一个节点
    it++;
  }
}

} // namespace jit
} // namespace torch
```