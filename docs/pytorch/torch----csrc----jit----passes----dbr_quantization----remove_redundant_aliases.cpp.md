# `.\pytorch\torch\csrc\jit\passes\dbr_quantization\remove_redundant_aliases.cpp`

```py
#include <torch/csrc/jit/passes/dbr_quantization/remove_redundant_aliases.h>

#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/quantization/helper.h>
#include <torch/csrc/jit/runtime/graph_iterator.h>

namespace torch {
namespace jit {

namespace {

// 实现移除冗余别名节点的函数
void DBRQuantRemoveRedundantAliasesImpl(const Method& method) {
  // 获取方法的计算图
  auto g = method.graph();
  // 是否处于冻结状态，这里设为 false
  const bool is_frozen = false;
  // 是否下降函数调用
  const bool descend_function_calls = true;
  // 创建别名分析对象
  AliasDb alias_db(g, is_frozen, descend_function_calls);

  // 查找所有别名节点
  std::vector<Node*> alias_nodes;
  DepthFirstGraphNodeIterator it(g);
  Node* node = nullptr;
  while ((node = it.next()) != nullptr) {
    // 如果节点类型为 "aten::alias"
    if (node->kind() == Symbol::aten("alias")) {
      alias_nodes.push_back(node);
    }
  }

  // 移除别名节点，如果安全的话
  for (auto* node : alias_nodes) {
    // 在调试模式下打印节点信息
    GRAPH_DEBUG(*node);

    // 获取别名节点的输入值和输出值
    Value* input_value = node->input();
    Value* output_value = node->output();

    // 检查是否总是安全进行别名关系变更
    bool always_safe_to_mutate = alias_db.safeToChangeAliasingRelationship(
        node->inputs(), node->outputs());

    // 获取计算图的输入和输出值列表
    const auto g_in = g->inputs();
    const auto g_out = g->outputs();
    // 检查输入值和输出值是否是计算图的输入和输出
    bool is_input =
        std::find(g_in.begin(), g_in.end(), input_value) != g_in.end();
    bool is_output =
        std::find(g_out.begin(), g_out.end(), output_value) != g_out.end();
    // 假设如果输入和输出没有写入者，则认为可以安全更新别名关系
    bool input_safe_to_mutate =
        (is_input && !alias_db.hasWriters(input_value) &&
         !alias_db.hasWriters(output_value));
    bool output_safe_to_mutate =
        (is_output && !alias_db.hasWriters(input_value) &&
         !alias_db.hasWriters(output_value));

    // 如果总是安全或者输入和输出安全，执行别名节点替换操作
    if (always_safe_to_mutate || input_safe_to_mutate ||
        output_safe_to_mutate) {
      output_value->replaceAllUsesWith(input_value);
      // 销毁当前节点
      node->destroy();
    }
  }
}

} // namespace

// 对模块中所有方法应用移除冗余别名节点的函数
Module DBRQuantRemoveRedundantAliases(Module& module) {
  for (const auto& child : module.modules()) {
    for (const auto& method : child.get_methods()) {
      DBRQuantRemoveRedundantAliasesImpl(method);
    }
  }

  // 返回处理后的模块
  return module;
}

} // namespace jit
} // namespace torch
```