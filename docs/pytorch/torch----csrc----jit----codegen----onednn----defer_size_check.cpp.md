# `.\pytorch\torch\csrc\jit\codegen\onednn\defer_size_check.cpp`

```py
// 包含头文件：torch/csrc/jit/codegen/onednn/defer_size_check.h
// 用于实现 size check 延迟的相关功能
#include <torch/csrc/jit/codegen/onednn/defer_size_check.h>

// 包含头文件：torch/csrc/jit/ir/alias_analysis.h
// 提供别名分析的相关工具和类
#include <torch/csrc/jit/ir/alias_analysis.h>

// 包含头文件：torch/csrc/jit/runtime/symbolic_shape_registry_util.h
// 提供符号形状注册工具的相关函数和类
#include <torch/csrc/jit/runtime/symbolic_shape_registry_util.h>

// 定义命名空间：torch::jit::fuser::onednn
// 定义了 SizeCheckMover 类和 DeferSizeCheck 函数
namespace torch {
namespace jit {
namespace fuser {
namespace onednn {

// 定义 SizeCheckMover 类，用于移动和延迟 size check 操作
class SizeCheckMover {
 private:
  Block* block_;  // 当前操作的节点块
  std::shared_ptr<Graph> graph_;  // 所属的计算图

 public:
  // 构造函数：初始化 SizeCheckMover 对象
  SizeCheckMover(Block* block, std::shared_ptr<Graph> graph)
      : block_(block), graph_(std::move(graph)) {}

  // 分析节点函数：检查并移动 size check 操作
  bool analyzeNode(Node* node, AliasDb& aliasDb) {
    // 如果节点不是 "aten::size(Tensor self) -> int[]" 形式，则返回 false
    if (!node->matches("aten::size(Tensor self) -> int[]"))
      return false;

    auto* input = node->input(0);  // 获取节点的输入
    auto& uses = input->uses();    // 获取输入节点的使用信息

    // 检查输入节点是否只被形状保持操作使用
    bool onlyUsedByShapePreserveOp =
        uses.size() > 1 && std::all_of(uses.begin(), uses.end(), [&](auto& u) {
          if (u.user == node) {
            return true;
          }
          // 匹配在 torch/csrc/jit/runtime/symbolic_shape_registry_util.cpp 中定义的形状保持一元操作
          OperatorMap<std::string> schemaMap = get_tensorexpr_elementwise_set();
          std::optional<std::string> mapping =
              schemaMap.find(u.user->getOperator());
          return mapping == "unary";
        });

    if (!onlyUsedByShapePreserveOp)
      return false;

    // 遍历节点的使用信息，尝试将 size check 移动到形状保持操作之后
    for (const auto& use : uses) {
      if (use.user == node)
        continue;
      auto shapePreserveOp = use.user;
      if (aliasDb.moveAfterTopologicallyValid(node, shapePreserveOp)) {
        node->replaceInputWith(input, shapePreserveOp->output(0));
        return true;
      }
    }

    return false;
  }

  // 运行函数：执行 size check 的延迟操作
  void run() {
    bool changed = true;
    while (changed) {
      changed = false;
      AliasDb aliasDb(graph_);  // 创建别名分析对象
      // 遍历当前节点块的所有节点，分析并更新节点
      for (Node* node : block_->nodes()) {
        changed |= analyzeNode(node, aliasDb);
      }
    }

    // 递归遍历当前节点块中的所有子块，执行 size check 延迟操作
    for (Node* node : block_->nodes())
      for (Block* subBlock : node->blocks())
        SizeCheckMover(subBlock, graph_).run();
  }
};

// 延迟执行 size check 的函数：调用 SizeCheckMover 类执行操作
void DeferSizeCheck(std::shared_ptr<Graph>& graph) {
  SizeCheckMover(graph->block(), graph).run();
}

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch
```