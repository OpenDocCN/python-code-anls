# `.\pytorch\torch\csrc\jit\passes\remove_mutation.h`

```
#pragma once
// 预处理指令，确保头文件只被编译一次

#include <c10/util/Exception.h>
// 引入异常处理相关的头文件

#include <torch/csrc/Export.h>
// 引入导出相关的头文件

#include <torch/csrc/jit/ir/alias_analysis.h>
// 引入别名分析相关的头文件

#include <torch/csrc/jit/ir/ir.h>
// 引入 IR 相关的头文件

namespace torch {
namespace jit {

struct TORCH_API MutationRemover {
  MutationRemover(
      std::shared_ptr<Graph> graph,
      std::optional<std::function<bool(Node*)>> mutation_filter = c10::nullopt)
      : mutation_filter_(mutation_filter),
        aliasDb_(nullptr),
        graph_(std::move(graph)) {}
  // 构造函数，初始化 MutationRemover 结构体的成员变量

  // return true if graph is modified
  bool removeListMutation();
  // 如果图被修改则返回 true，移除列表的变异操作

  // return true if graph is modified
  bool removeTensorMutation();
  // 如果图被修改则返回 true，移除张量的变异操作

  bool isSpecialMappedOp(Node* n) {
    return n->matches("aten::zero_(Tensor(a!) self) -> Tensor(a!)") ||
        n->matches(
            "aten::fill_.Scalar(Tensor(a!) self, Scalar value) -> Tensor(a!)") ||
        n->matches(
            "aten::normal_(Tensor(a!) self, float mean=0, float std=1, *, Generator? generator=None) -> Tensor(a!)");
  }
  // 判断节点 n 是否为特定的映射操作，返回布尔值

  bool inplaceOpVariant(Node* n);
  // 判断节点 n 是否为就地操作的变体

  static bool hasSideEffectOrAlias(Value* v, AliasDb* aliasDb);
  // 判断值 v 是否具有副作用或别名，传入别名分析对象 aliasDb

 private:
  Node* createSpecialMappedOp(Node* n);
  // 创建特殊映射操作的节点 n

  bool listMutationFollowingListConstruct(Node* n);
  // 列表构造后的列表变异操作

  bool tryMakeCreationAndMutationAtomic(
      Value* mutated_value,
      Node* mutating_op);
  // 尝试使创建和变异原子化，传入变异值和变异操作节点

  bool tryMakeUnaliasedIfOutputAndMutationAtomic(
      Value* mutated_value,
      Node* mutating_op);
  // 尝试使非别名化的输出和变异原子化，传入变异值和变异操作节点

  // return true if graph is modified
  bool RemoveListMutation(Block* block);
  // 如果图被修改则返回 true，移除块中的列表变异操作

  // return true if graph is modified
  bool RemoveTensorMutation(Block* block);
  // 如果图被修改则返回 true，移除块中的张量变异操作

  AliasDb* getOrCreateAliasDb() {
    if (!aliasDb_) {
      aliasDb_ = std::make_unique<AliasDb>(graph_);
    }
    return aliasDb_.get();
  }
  // 获取或创建别名分析对象

  std::optional<std::function<bool(Node*)>> mutation_filter_;
  // 可选的变异过滤器函数

  std::unique_ptr<AliasDb> aliasDb_ = nullptr;
  // 唯一指针，用于存储别名分析对象

  std::shared_ptr<Graph> graph_;
  // 共享指针，用于存储图对象
};

// Removes list mutation with functional equivalents
// return true if graph is modified
TORCH_API bool RemoveListMutation(const std::shared_ptr<Graph>& graph);
// 使用函数等效物移除列表的变异操作，如果图被修改则返回 true

// Replaces in-place aten ops with their functional equivalents
// when it can be proven that this does not change graph semantics
// if `mutation_filter` is present, the pass will only attempt to
// remove mutation on nodes which return true for the filter
// return true if graph is modified
TORCH_API bool RemoveTensorMutation(
    const std::shared_ptr<Graph>& graph,
    std::optional<std::function<bool(Node*)>> mutation_filter = c10::nullopt);
// 将就地 aten 操作替换为其函数等效物，当可以证明这不会改变图的语义时，如果 mutation_filter 存在，则仅尝试对返回 true 的节点删除变异操作，如果图被修改则返回 true

// Replaces in-place aten activation ops with their functional equivalence
TORCH_API bool InplaceToFunctionalActivation(
    const std::shared_ptr<Graph>& graph);
// 将就地 aten 激活操作替换为其函数等效物

} // namespace jit
} // namespace torch
```