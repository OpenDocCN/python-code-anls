# `.\pytorch\torch\csrc\jit\ir\ir_views.h`

```py
#pragma once

#include <c10/util/irange.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

// 表示一个 If 节点的视图，用于访问条件、then 分支和 else 分支等信息
struct IfView {
  explicit IfView(Node* node) : node_(node) {
    AT_ASSERT(node->kind() == ::c10::prim::If);
  }

  // 返回条件节点的值
  Value* cond() const {
    return node_->input(0);
  }

  // 返回 then 分支的 Block
  Block* thenBlock() const {
    return node_->blocks().at(0);
  }

  // 返回 else 分支的 Block
  Block* elseBlock() const {
    return node_->blocks().at(1);
  }

  // 返回 then 分支的输出值数组
  ArrayRef<Value*> thenOutputs() const {
    return thenBlock()->outputs();
  }

  // 返回 else 分支的输出值数组
  ArrayRef<Value*> elseOutputs() const {
    return elseBlock()->outputs();
  }

  // 返回节点的所有输出值数组
  ArrayRef<Value*> outputs() const {
    return node_->outputs();
  }

  // 返回当前 If 节点
  Node* node() const {
    return node_;
  }

  // 将 If 节点及其 then、else 分支的输出值按照给定的顺序重新排列
  void permuteOutputs(const std::vector<size_t>& new_output_order) {
    node_->permuteOutputs(new_output_order);
    thenBlock()->permuteOutputs(new_output_order);
    elseBlock()->permuteOutputs(new_output_order);
  }

 private:
  Node* node_;
};

// 表示一个 Loop 节点的视图，用于访问循环体、条件、以及循环中携带的输入输出信息
struct LoopView {
  explicit LoopView(Node* node) : node_(node) {
    AT_ASSERT(
        node->kind() == ::c10::prim::Loop || node->kind() == ::c10::onnx::Loop);
  }

  // 返回循环体的 Block
  Block* bodyBlock() const {
    return node_->blocks().at(0);
  }

  // 返回循环条件的值
  Value* cond() const {
    return node_->input(0);
  }

  // 返回循环的最大迭代次数
  Value* maxTripCount() const {
    return node_->input(0);
  }

  // 返回输入条件的值
  Value* inputCond() const {
    return node_->input(1);
  }

  // 返回循环的下一个条件值
  Value* nextCond() const {
    return bodyBlock()->outputs().at(0);
  }

  // 返回当前迭代次数的值
  Value* currentTripCount() const {
    return bodyBlock()->inputs().at(0);
  }

  // 返回循环携带的输入（不包括迭代次数和条件）
  ArrayRef<Value*> carriedInputs() const {
    return node_->inputs().slice(2);
  }

  // 返回带有输入条件的循环携带的输入（不包括迭代次数）
  ArrayRef<Value*> carriedInputsWithCond() const {
    return node_->inputs().slice(1);
  }

  // 返回循环携带的输出
  ArrayRef<Value*> carriedOutputs() const {
    return node_->outputs();
  }

  // 返回循环体携带的输入（不包括迭代次数）
  ArrayRef<Value*> bodyCarriedInputs() const {
    return bodyBlock()->inputs().slice(1);
  }

  // 返回循环体携带的输出（不包括下一个条件）
  ArrayRef<Value*> bodyCarriedOutputs() const {
    return bodyBlock()->outputs().slice(1);
  }

  // 返回当前 Loop 节点
  Node* node() const {
    return node_;
  }

  // 将循环节点及其相关输入输出按照给定的顺序重新排列
  void permuteLoopCarried(const std::vector<size_t>& new_output_order) {
    node_->permuteOutputs(new_output_order);
    node_->permuteInputs(adjustIndices(2, new_output_order)); // 跳过迭代次数和条件
    auto adjusted_block_order = adjustIndices(1, new_output_order); // 跳过迭代次数
    bodyBlock()->permuteOutputs(adjusted_block_order);
    bodyBlock()->permuteInputs(adjusted_block_order);
  }

  // 替换循环的最大迭代次数输入
  void replaceMaxTripCount(Value* new_max_trip_count) {
    node_->replaceInput(0, new_max_trip_count);
  }

  // 替换循环的输入条件
  void replaceInputCondition(Value* new_input_condition) {
  // 替换节点的第二个输入为新的条件输入
  node_->replaceInput(1, new_input_condition);
}

// 我们编码循环的方式使得它们难以转换回Python语法。
// 我们必须检查条件和循环次数输入的属性，以确定它最初是哪一个。
// ModifiedLoops 既不直接映射到 For 也不映射到 While。
enum LoopType { While, For, ModifiedLoop };

LoopType loopType() {
  auto trip_count = toIValue(maxTripCount());
  auto cond_input = toIValue(inputCond());
  auto cond_next = toIValue(nextCond());

  bool condition_is_always_true =
      cond_input && cond_input->toBool() && cond_next && cond_next->toBool();
  bool trip_count_is_specified = !trip_count || // trip 不是常量
      trip_count->toInt() !=
          std::numeric_limits<int64_t>::max() || // 是常量但不是默认值
      !currentTripCount()
           ->uses()
           .empty(); // 它实际上在循环体中被使用。

  if (condition_is_always_true) {
    // 如果未指定循环次数，则这是用户编写的 while True:
    return trip_count_is_specified ? For : While;
  } else {
    if (trip_count_is_specified) {
      return ModifiedLoop;
    }
    return While;
  }
}

private:
Node* node_;

// 通过添加从 0 到 adjust 的索引，并递增所有现有输入 adjust 次，来调整 index_ordering
static std::vector<size_t> adjustIndices(
    size_t adjust,
    const std::vector<size_t>& index_ordering) {
  std::vector<size_t> adjusted;
  adjusted.reserve(adjust + index_ordering.size());
  for (const auto i : c10::irange(adjust)) {
    adjusted.push_back(i);
  }
  for (auto index : index_ordering) {
    adjusted.push_back(index + adjust);
  }
  return adjusted;
}
};
} // namespace jit
} // namespace torch
```