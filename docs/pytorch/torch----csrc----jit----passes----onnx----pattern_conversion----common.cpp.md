# `.\pytorch\torch\csrc\jit\passes\onnx\pattern_conversion\common.cpp`

```py
// 包含 Torch 库中 ONNX 模式转换通用功能的头文件
#include <torch/csrc/jit/passes/onnx/pattern_conversion/common.h>

// Torch 命名空间开始
namespace torch {
namespace jit {

// 检查两个节点是否来自相同的源代码位置
bool IndexingPatternFinder::IsSameSource(const Node* n, const Node* m) {
  // 获取节点 n 和 m 的源代码文本
  const auto source_n = n->sourceRange().source();
  const auto source_m = m->sourceRange().source();
  // 比较两个节点的源代码文本和起始行号是否相同
  return (
      (source_n->text_str() == source_m->text_str()) &&
      (source_n->starting_line_no() == source_m->starting_line_no()));
}

// 追溯与 index_put 节点相关的所有切片和选择节点
// 例如，对于 x[1:3, 0] = update 的 IR
//    ...
//    %8 : Float(2, 4) = aten::slice(%0, %4, %5, %6, %7)
//    ...
//    %11 : Float(2) = aten::select(%8, %9, %10)
//    ...
//    %13 : Tensor?[] = prim::ListConstruct()
//    ...
//    %16 : Float(2) = aten::index_put(%11, %13, %14, %15)
//
// 我们收集 %11 和 %8，以构造索引张量。
// 向量 slice_and_select_node 包含所有相关的切片和选择节点，顺序相反。
std::vector<Node*> IndexingPatternFinder::FetchSliceAndSelect(
    const Node* node) {
  // 初始化存储切片和选择节点的向量
  std::vector<Node*> slice_and_select_node;
  // 获取节点 node 的输入节点的源节点
  auto src_node = node->input(0)->node();
  // 迭代追溯节点链，直到没有更多源节点为止
  while (src_node) {
    // 如果源节点是切片或选择节点，并且来自同一源代码位置
    if ((src_node->kind() == aten::slice || src_node->kind() == aten::select) &&
        IsSameSource(src_node, node)) {
      // 将找到的切片或选择节点添加到向量中
      slice_and_select_node.emplace_back(src_node);
      // 继续向上追溯源节点链
      src_node = src_node->input(0)->node();
    } else {
      // 如果不是期望的节点类型或者不是同一源代码位置，结束追溯
      src_node = nullptr;
    }
  }
  // 返回收集到的切片和选择节点向量
  return slice_and_select_node;
}

} // namespace jit
} // namespace torch
```