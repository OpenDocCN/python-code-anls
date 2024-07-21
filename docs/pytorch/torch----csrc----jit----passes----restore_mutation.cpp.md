# `.\pytorch\torch\csrc\jit\passes\restore_mutation.cpp`

```py
#include <ATen/core/jit_type.h>
#include <ATen/core/symbol.h>
#include <torch/csrc/jit/passes/remove_mutation.h>
#include <torch/csrc/jit/passes/restore_mutation.h>

namespace torch {
namespace jit {

// 用于将函数式操作转换为原位操作的重写器类构造函数
FunctionalToInplaceRewriter::FunctionalToInplaceRewriter(std::shared_ptr<Graph> graph)
    : aliasDb_(nullptr), graph_(std::move(graph)) {}

// 判断给定节点是否可以进行原位操作
bool FunctionalToInplaceRewriter::CanBeInplace(Node* node) {
  // 如果节点类型不在允许的激活类型升级映射中，则不能进行原位操作
  if (activation_type_promotion_mapping.find(node->kind()) ==
      activation_type_promotion_mapping.end()) {
    return false;
  }

  // 构建对应的原位操作符号
  Symbol inplace_op = Symbol::fromQualString(std::string(node->kind().toQualString()) + "_");
  if (!inplace_op) {
    return false;
  }

  // 如果允许类型升级，则进行数据类型检查
  bool check_dtype = activation_type_promotion_mapping.at(node->kind());

  // 获取输入和输出值
  Value* input = node->inputs().at(0);
  Value* output = node->outputs().at(0);
  auto inputDtype = input->type()->expect<TensorType>()->scalarType();
  auto outputDtype = output->type()->expect<TensorType>()->scalarType();

  // 对于允许类型升级的操作，确保输入和输出的数据类型相同
  if (check_dtype &&
      (!inputDtype || !outputDtype ||
       inputDtype.value() != outputDtype.value())) {
    return false;
  }

  // 如果输入的定义节点有副作用或者存在别名，则不能进行原位操作
  if (MutationRemover::hasSideEffectOrAlias(input, getOrCreateAliasDb())) {
    return false;
  }

  // 如果输入值有多于一个使用，则跳过转换
  // TODO: 使用存活性分析来处理更一般的情况
  return (input->uses().size() == 1);
}

// 递归地将函数式操作转换为原位操作
bool FunctionalToInplaceRewriter::FunctionalToInplace(Block* block) {
  bool changed = false;
  for (auto it = block->nodes().begin(); it != block->nodes().end();) {
    auto* node = *it;
    it++;

    // 递归处理节点的子块
    for (Block* sub_block : node->blocks()) {
      changed |= FunctionalToInplace(sub_block);
    }

    // 如果节点不能进行原位操作，则继续下一个节点
    if (!CanBeInplace(node)) {
      continue;
    }

    // 进行原位操作的替换
    changed = true;
    Node* inplace_node = node->replaceWithNewSymbol(
        Symbol::fromQualString(node->schema().name() + "_"));
    inplace_node->output()->replaceAllUsesWith(node->inputs().at(0));
    getOrCreateAliasDb()->replaceWithNewValue(
        node->output(), inplace_node->output());

    // 销毁原节点
    node->destroy();
  }
  return changed;
}

// 对外接口，将函数式操作转换为原位操作的入口函数
bool FunctionalToInplaceActivation(const std::shared_ptr<Graph>& graph) {
  FunctionalToInplaceRewriter rewriter(graph);
  return rewriter.FunctionalToInplace(graph->block());
}

} // namespace jit
} // namespace torch
```