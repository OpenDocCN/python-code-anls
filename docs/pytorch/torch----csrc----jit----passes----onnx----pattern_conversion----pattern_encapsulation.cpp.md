# `.\pytorch\torch\csrc\jit\passes\onnx\pattern_conversion\pattern_encapsulation.cpp`

```
// 引入 Torch 库中相关头文件，用于 JIT 编译器的死代码消除和 ONNX 操作支持
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/onnx.h>
#include <torch/csrc/jit/passes/onnx/pattern_conversion/common.h>
#include <torch/csrc/jit/passes/onnx/pattern_conversion/pattern_encapsulation.h>
#include <torch/csrc/jit/passes/onnx/remove_inplace_ops_for_onnx.h>

// 编辑此文件时，请务必阅读 pattern_encapsulation.h 中的“注解模式封装”说明
// see Note [Edit Pattern Encapsulation] in pattern_encapsulation.h

namespace torch {
namespace jit {

namespace {

// 为 ONNX 封装索引更新操作节点，将相关的切片和选择节点复制到占位符子块中
// 例如，对于 x[1:3, 0] = update 的 IR：
//    ...
//    %8 : Float(2, 4) = aten::slice(%0, %4, %5, %6, %7)
//    ...
//    %11 : Float(2) = aten::select(%8, %9, %10)
//    ...
//    %13 : Tensor?[] = prim::ListConstruct()
//    ...
//    %16 : Float(2) = aten::index_put(%11, %13, %14, %15)
// 单独的 aten::index_put 节点不包含任何索引 (%13 : Tensor?[] = prim::ListConstruct())。
Node* EncapsulateInplaceIndexPutForONNX(Node* index_put_node) {
  auto graph = index_put_node->owningGraph();

  // 查找与此索引更新操作节点相关联的切片和选择操作符
  std::vector<Node*> slice_and_select_nodes =
      IndexingPatternFinder::FetchSliceAndSelect(index_put_node);
  // 如果找到相关节点，则取最后一个作为最后节点；否则使用索引更新节点本身
  Node* last_node = !slice_and_select_nodes.empty()
      ? slice_and_select_nodes.back()
      : index_put_node;
  // 获取原始数据值
  Value* orig_data = last_node->input(0);

  // 创建一个特殊占位符节点的子块，并将相关节点复制到子块中
  Node* placeholder_node =
      graph->create(Symbol::fromQualString("onnx::Placeholder"));
  placeholder_node->s_(attr::name, index_put_node->kind().toUnqualString());
  placeholder_node->addInput(orig_data);

  // 构造子块
  auto subblock = placeholder_node->addBlock();
  std::unordered_map<Value*, Value*> env;

  // 逆序处理切片和选择节点
  for (auto it = slice_and_select_nodes.rbegin();
       it != slice_and_select_nodes.rend();
       ++it) {
    auto n = *it;
    // 克隆节点并将其添加到子块中
    auto cloned_n = subblock->appendNode(graph->createClone(
        n, [&](Value* v) { return env.find(v) != env.end() ? env[v] : v; }));
    // 更新环境映射
    for (size_t i = 0; i < cloned_n->outputs().size(); ++i) {
      env[n->outputs().at(i)] = cloned_n->outputs().at(i);
    }
  }

  // 克隆索引更新节点并添加到子块中
  Node* new_index_put_node =
      subblock->appendNode(graph->createClone(index_put_node, [&](Value* v) {
        return env.find(v) != env.end() ? env[v] : v;
      }));
  // 注册子块的输出节点
  for (auto o : new_index_put_node->outputs()) {
    subblock->registerOutput(o);
  }

  // 将占位符节点插入到索引更新节点之前，并复制元数据
  placeholder_node->insertBefore(index_put_node);
  placeholder_node->copyMetadata(index_put_node);
  // 用占位符节点替换索引更新节点的所有使用
  index_put_node->replaceAllUsesWith(placeholder_node);

  return placeholder_node;
}

} // namespace

// 将模式封装为子块的可选节点
std::optional<Node*> EncapsulatePatternIntoSubblock(Node* n) {
  switch (n->kind()) {
    case aten::index_put_:
    // 对于操作类型是 `aten::index_put` 的情况，调用 EncapsulateInplaceIndexPutForONNX 函数处理
    case aten::index_put: {
      return EncapsulateInplaceIndexPutForONNX(n);
    }
  }
  // 如果未匹配到任何操作类型，返回空的 optional 对象
  return c10::nullopt;
}

} // namespace jit
} // namespace torch
```