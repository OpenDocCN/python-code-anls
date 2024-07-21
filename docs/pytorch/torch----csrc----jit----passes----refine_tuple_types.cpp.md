# `.\pytorch\torch\csrc\jit\passes\refine_tuple_types.cpp`

```py
#include <torch/csrc/jit/passes/refine_tuple_types.h>
#include <torch/csrc/jit/runtime/graph_iterator.h>

#include <ATen/core/type_factory.h>

#include <utility>

namespace torch {
namespace jit {

namespace {
// 定义一个静态函数，用于访问 tuple 节点并更新其类型信息
static void VisitTupleNode(Node* node) {
  // 检查 tuple 节点的输出数量是否为1
  TORCH_CHECK(
      node->outputs().size() == 1, "Tuple must have exactly one output!");

  // 获取 tuple 节点的输出值
  Value* output = node->outputs()[0];
  // 期望输出值的类型为 TupleType，并获取其类型
  auto tuple_type = output->type()->expectRef<TupleType>();

  // 检查 tuple 类型中包含的类型数量是否与节点的输入数量相匹配
  TORCH_CHECK(
      tuple_type.containedTypes().size() == node->inputs().size(),
      "Number of contained types does not match number of inputs!");

  // 从输入值中提取更新后的类型信息
  std::vector<c10::TypePtr> types;
  for (const Value* input : node->inputs()) {
    types.push_back(input->type());
  }

  // 根据输入类型构造新的 tuple 类型
  output->setType(tuple_type.withContained(std::move(types)));
}
} // anonymous namespace

// 对图中的 tuple 节点进行类型细化
void RefineTupleTypes(std::shared_ptr<Graph>& graph) {
  // 使用深度优先图节点迭代器遍历图中的每一个节点
  DepthFirstGraphNodeIterator it(graph);
  for (auto* node = it.next(); node != nullptr; node = it.next()) {
    // 如果节点的类型为 prim::TupleConstruct，则访问该 tuple 节点
    if (node->kind() == prim::TupleConstruct) {
      VisitTupleNode(node);
    }
  }
}

} // namespace jit
} // namespace torch
```