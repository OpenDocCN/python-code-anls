# `.\pytorch\torch\csrc\jit\passes\replacement_of_old_operators.cpp`

```
#include <torch/csrc/jit/passes/replacement_of_old_operators.h>
// 引入替换旧操作符的头文件

#include <c10/util/Exception.h>
#include <caffe2/serialize/versions.h>
#include <torch/csrc/jit/frontend/schema_matching.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/operator_upgraders/upgraders.h>
#include <torch/csrc/jit/operator_upgraders/utils.h>
#include <torch/csrc/jit/operator_upgraders/version_map.h>
#include <torch/csrc/jit/runtime/graph_iterator.h>
// 引入其他必要的头文件

#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
// 引入标准库和第三方库的头文件

namespace torch {
namespace jit {

struct OldOpsReplacerWithUpgraders {
  OldOpsReplacerWithUpgraders(std::shared_ptr<Graph> graph)
      : graph_(std::move(graph)) {}
  // 定义结构体 OldOpsReplacerWithUpgraders，接受一个图的共享指针参数

  void run() {
    // 运行替换操作
    if (!graph_->get_op_version().has_value()) {
      // 如果图中没有操作版本信息，则返回
      return;
    }

    auto current_version = graph_->get_op_version().value();
    // 获取当前操作版本号

    DepthFirstGraphNodeIterator graph_it(graph_);
    // 创建深度优先图节点迭代器，传入图对象
    Node* node = graph_it.next();
    // 获取迭代器的下一个节点
    // 当前节点不为空时执行循环
    while (node) {
      // 加载此操作的模式名称
      std::optional<std::string> schema_name = c10::nullopt;
      // 如果节点有操作模式，则获取完整的模式名称
      if (auto op_schema = node->maybeSchema()) {
        schema_name = getFullSchemaName(*op_schema);
      } else {
        // 否则获取历史模式名称
        schema_name = node->getHistoricSchemaName();
      }

      // 如果成功获取模式名称
      if (schema_name.has_value()) {
        // 检查是否因此操作符进行了版本升级
        auto version_entry =
            get_operator_version_map().find(schema_name.value());
        // 如果找到了版本信息
        if (version_entry != get_operator_version_map().end()) {
          const auto& entry = version_entry->second;
          // 查找并获取当前版本对应的升级器条目
          auto upgrader_entry = findUpgrader(entry, current_version);
          // 如果没有找到对应的升级器条目
          if (!upgrader_entry.has_value()) {
            // 检查操作符是否为当前版本的符号
            if (!isOpSymbolCurrent(schema_name.value(), current_version)) {
              // 如果没有对应的升级器，则断言失败
              TORCH_INTERNAL_ASSERT(
                  false,
                  "Upgrader must be present for ",
                  schema_name.value(),
                  ". The upgrader might have deprecated");
            }
            // 继续处理下一个节点
            node = graph_it.next();
            continue;
          }
          // 获取升级器条目的值和升级器名称
          auto upgrader_entry_val = upgrader_entry.value();
          auto upgrader_name = upgrader_entry_val.upgrader_name;
          // 查找升级器名称对应的升级器图
          auto upgrader_graph_entry = dump_upgraders_map().find(upgrader_name);
          // 断言升级器图是否存在
          TORCH_INTERNAL_ASSERT(
              upgrader_graph_entry != dump_upgraders_map().end(),
              "Corresponding upgrader graph for ",
              upgrader_name,
              " must exist.",
              " This upgrader"
              " might be deprecated.");

          // 获取升级器图
          auto upgrader_graph = upgrader_graph_entry->second;
          // 在当前节点的插入点设置保护区域
          WithInsertPoint guard(node);
          // 插入升级器图到当前图中，并获取新的输出
          auto new_outputs = insertGraph(
              *node->owningGraph(), *upgrader_graph, node->inputs());
          // 获取当前节点的旧输出
          const auto& old_outputs = node->outputs();
          // 断言新旧输出的数量相同
          TORCH_INTERNAL_ASSERT(new_outputs.size() == old_outputs.size());
          // 替换所有旧输出为新输出
          for (const auto i : c10::irange(old_outputs.size())) {
            TORCH_INTERNAL_ASSERT(
                new_outputs[i]->type() == old_outputs[i]->type())
            old_outputs[i]->replaceAllUsesWith(new_outputs[i]);
          }
          // 移除当前节点的所有输入
          node->removeAllInputs();
          // 销毁当前节点
          node->destroy();
        }
      }
      // 获取下一个节点
      node = graph_it.next();
    }

    // 更新图的操作版本
    graph_->set_op_version(caffe2::serialize::kProducedFileFormatVersion);
  }

  // 图的指针成员变量
  std::shared_ptr<Graph> graph_;
};

TORCH_API void ReplaceOldOperatorsWithUpgraders(std::shared_ptr<Graph> graph) {
  // 创建一个 OldOpsReplacerWithUpgraders 对象，传入 graph 并执行 run 方法
  OldOpsReplacerWithUpgraders(std::move(graph)).run();
}

} // namespace jit
} // namespace torch
```