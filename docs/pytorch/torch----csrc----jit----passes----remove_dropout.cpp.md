# `.\pytorch\torch\csrc\jit\passes\remove_dropout.cpp`

```
#include <torch/csrc/jit/passes/remove_dropout.h>

namespace torch {
namespace jit {

namespace {
// 判断节点是否可移除dropout操作
bool isDropoutRemovable(const Node* node) {
  // 获取节点的输入
  const auto inputs = node->inputs();
  // 确保输入的数量是3个
  TORCH_INTERNAL_ASSERT(inputs.size() == 3);
  // 获取训练输入的值
  const Value* training_input = inputs[2];
  // 尝试将训练输入值转换为 IValue 类型
  auto optional_ivalue = toIValue(training_input);
  // 如果转换失败，则返回不可移除
  if (!optional_ivalue) {
    return false;
  }
  // 获取转换后的 IValue 引用
  const IValue& val = optional_ivalue.value();
  // 确保值为布尔类型
  TORCH_INTERNAL_ASSERT(val.isBool());
  // 获取布尔值，表示是否在训练模式
  const bool is_training = val.toBool();
  // 如果不在训练模式，则可移除dropout操作
  return !is_training;
}

// 递归地在基本块中移除dropout操作
void removeDropoutImpl(Block* block) {
  // 存储将要删除的节点
  std::vector<Node*> deleted_nodes;

  // 从后向前遍历基本块中的节点
  for (auto it = block->nodes().rbegin(); it != block->nodes().rend(); it++) {
    Node* node = *it;
    // 递归处理子块中的节点
    for (auto sub_block : node->blocks()) {
      removeDropoutImpl(sub_block);
    }
    // 如果节点是dropout相关操作并且可以移除，则执行以下操作
    if ((node->kind() == c10::Symbol::fromQualString("aten::dropout") ||
         node->kind() == c10::Symbol::fromQualString("aten::dropout_") ||
         node->kind() == c10::Symbol::fromQualString("aten::feature_dropout") ||
         node->kind() ==
             c10::Symbol::fromQualString("aten::feature_dropout_")) &&
        isDropoutRemovable(node)) {
      // 获取dropout操作的输入张量
      Value* input_value = node->inputs()[0];
      // 获取dropout操作的输出张量
      Value* output_value = node->outputs()[0];
      // 用输入张量替换所有使用输出张量的地方
      output_value->replaceAllUsesWith(input_value);
      // 将节点添加到删除列表中
      deleted_nodes.push_back(node);
    }
  }
  // 删除所有标记的节点
  for (auto del_node : deleted_nodes) {
    del_node->destroy();
  }
}
} // namespace

// 公共接口，从图中移除dropout操作
void removeDropout(std::shared_ptr<Graph>& graph) {
  removeDropoutImpl(graph->block());
}

// 公共接口，从脚本模块中移除dropout操作
void removeDropout(script::Module& module) {
  // 检查模块是否处于训练模式，如果是，则抛出异常
  TORCH_CHECK(
      !module.hasattr("training") || !module.is_training(),
      "Dropout removal module in training mode is not yet supported");
  // 获取前向方法的图
  auto graph = module.get_method("forward").graph();
  // 从图中移除dropout操作
  removeDropout(graph);
}

} // namespace jit
} // namespace torch
```