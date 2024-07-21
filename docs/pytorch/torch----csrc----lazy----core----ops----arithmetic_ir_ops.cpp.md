# `.\pytorch\torch\csrc\lazy\core\ops\arithmetic_ir_ops.cpp`

```py
// 包含头文件 torch/csrc/lazy/core/ops/arithmetic_ir_ops.h
#include <torch/csrc/lazy/core/ops/arithmetic_ir_ops.h>

// 包含自定义的辅助函数头文件
#include <torch/csrc/lazy/core/helpers.h>

// 包含标准库中的内存管理头文件
#include <memory>

// 包含操作符构建器的头文件
#include <torch/csrc/lazy/core/ir_builder.h>

// 定义命名空间 torch::lazy
namespace torch {
namespace lazy {

// 这些操作符曾广泛用于 nativefunction 实现中，用于方便地将 aten 操作符分解为更基础的操作。
// 现在不再推荐用于此目的，但仍然在 lazy_graph_executor 中的 RNG 数学计算中使用。
// 我们可以重写这部分代码。
NodePtr operator+(const Value& node1, const Value& node2) {
  // 调用 MakeGeneric 函数，生成一个表示加法操作的 NodePtr
  return MakeGeneric(
      OpKind(at::aten::add),                  // 使用 aten::add 操作符
      {node1, node2},                         // 传入两个操作数
      GetPromotedBinaryOpShape(node1.shape(), node2.shape()));  // 获取操作的推广形状
}

NodePtr operator-(const Value& node1, const Value& node2) {
  // 调用 MakeGeneric 函数，生成一个表示减法操作的 NodePtr
  return MakeGeneric(
      OpKind(at::aten::sub),                  // 使用 aten::sub 操作符
      {node1, node2},                         // 传入两个操作数
      GetPromotedBinaryOpShape(node1.shape(), node2.shape()));  // 获取操作的推广形状
}

NodePtr operator*(const Value& node1, const Value& node2) {
  // 调用 MakeGeneric 函数，生成一个表示乘法操作的 NodePtr
  return MakeGeneric(
      OpKind(at::aten::mul),                  // 使用 aten::mul 操作符
      {node1, node2},                         // 传入两个操作数
      GetPromotedBinaryOpShape(node1.shape(), node2.shape()));  // 获取操作的推广形状
}

NodePtr operator/(const Value& node1, const Value& node2) {
  // 调用 MakeGeneric 函数，生成一个表示除法操作的 NodePtr
  return MakeGeneric(
      OpKind(at::aten::div),                  // 使用 aten::div 操作符
      {node1, node2},                         // 传入两个操作数
      GetPromotedBinaryOpShape(node1.shape(), node2.shape()));  // 获取操作的推广形状
}

} // namespace lazy
} // namespace torch
```