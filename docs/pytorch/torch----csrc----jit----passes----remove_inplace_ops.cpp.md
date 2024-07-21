# `.\pytorch\torch\csrc\jit\passes\remove_inplace_ops.cpp`

```
// 包含 Torch 库中的头文件，用于移除原地操作
#include <torch/csrc/jit/passes/remove_inplace_ops.h>
// 包含标准输入输出流库
#include <iostream>

// Torch 命名空间
namespace torch {
// Torch JIT 命名空间
namespace jit {
// 匿名命名空间，用于定义静态变量和辅助函数

// 将原地操作转换为非原地操作的映射表
static const std::unordered_map<NodeKind, NodeKind> inPlaceToOutOfPlace = {
    {aten::add_, aten::add},
    {aten::sub_, aten::sub},
    {aten::div_, aten::div},
    {aten::mul_, aten::mul},
    {aten::masked_fill_, aten::masked_fill},
    {aten::zero_, aten::zeros_like},
    {aten::fill_, aten::full_like}
};

// 期望的输入数量，用于 zeros_like 和 full_like 的特殊情况处理
static const std::unordered_map<NodeKind, int> expectedInputCount = {
    {aten::zero_, 6},
    {aten::fill_, 7}
};

// 判断节点是否为原地操作
bool isInplaceOp(const Node* node) {
  return inPlaceToOutOfPlace.count(node->kind()) != 0;
}

// 移除所有原地操作，并替换为非原地操作的等效形式
void RemoveInplaceOps(Block* block) {
  auto graph = block->owningGraph();
  auto it = block->nodes().begin();
  while (it != block->nodes().end()) {
    auto node = *it;
    ++it;
    for (auto block : node->blocks()) {
      RemoveInplaceOps(block);
    }

    if (isInplaceOp(node)) {
      // 创建替代的非原地操作节点
      auto newNode = graph->create(inPlaceToOutOfPlace.at(node->kind()));
      newNode->insertBefore(node);
      newNode->copyMetadata(node);
      
      // 复制输入节点
      for (auto input : node->inputs()) {
        newNode->addInput(input);
      }

      // 处理额外的输入节点，以匹配预期的输入数量
      int additionalInputCount = 0;
      if (expectedInputCount.find(node->kind()) != expectedInputCount.end()) {
        additionalInputCount = expectedInputCount.at(node->kind()) -
            static_cast<int>(newNode->inputs().size());
      }

      for (int i = 0; i < additionalInputCount; ++i) {
        auto noneNode = graph->createNone();
        noneNode->insertBefore(newNode);
        newNode->addInput(noneNode->output());
      }

      // 创建新的输出节点，并用它替换所有使用原节点的地方
      newNode->output()->copyMetadata(node->output());
      node->replaceAllUsesWith(newNode);
      node->inputs()[0]->replaceAllUsesAfterNodeWith(
          newNode, newNode->output());
      node->destroy();
    }
  }
}
} // namespace jit
} // namespace torch
// node with the higher data type precedence, so that both the input types
// are the same.
// An example scenario would be:
// Before:
// graph(%0 : Float),
//        %1 : Half):
//   # Should result in a Half, but after translation to out-of-place,
//   # would become a Float b/c Half+Float -> Float.
//   %4 : Float = onnx::Cast[to=1](%1)
//   %5 : Float = onnx::Add(%4, %0)
//   ...
// After:
// graph(%0 : Float),
//        %1 : Half):
//   %4 : Half = onnx::Cast[to=10](%0)
//   %5 : Half = onnx::Add(%1, %4)
//   ...

void ImplicitCastForBinaryInplaceOps(Block* b) {
  // Iterate over all nodes in the given block, including nested blocks
  for (auto it = b->nodes().begin(), end = b->nodes().end(); it != end; ++it) {
    for (auto* child_block : it->blocks()) {
      // Recursively apply the function to nested blocks
      ImplicitCastForBinaryInplaceOps(child_block);
    }

    // Check if the current node represents an inplace operation
    if ((it->kind() == aten::add_) || (it->kind() == aten::sub_) ||
        (it->kind() == aten::mul_) || (it->kind() == aten::div_)) {
      // Retrieve the original inputs to the operation
      auto originalInputs = it->inputs();
      // Skip if both inputs are the same (inplace operation)
      if (originalInputs.at(0) == originalInputs.at(1)) {
        continue;
      }

      // Retrieve the node representing the shape of the first input
      auto shape_node = originalInputs.at(0)->node();
      // Check if the shape node is a tensor shape operation
      if ((shape_node->kind() == prim::NumToTensor) &&
          (shape_node->inputs().at(0)->node()->kind() == aten::size)) {
        // Print a warning message for inplace operation on tensor shape output
        std::cerr
            << "In-place op on output of tensor.shape. See https://pytorch.org/docs/main/onnx.html#"
            << "avoid-inplace-operations-when-using-tensor-shape-in-tracing-mode"
            << std::endl;
      }

      // Retrieve the types of the first and second input tensors
      TensorTypePtr firstInp_tensor =
          originalInputs.at(0)->type()->cast<TensorType>();
      TensorTypePtr secondInp_tensor =
          originalInputs.at(1)->type()->cast<TensorType>();
      // Continue if either input type is not a tensor or scalar type is not defined
      if (!(firstInp_tensor) || !(secondInp_tensor) ||
          !(firstInp_tensor->scalarType().has_value())) {
        continue;
      }

      // Create a new node to cast the second input to the type of the first input
      auto newInputNode = it->owningGraph()->create(aten::type_as, 1);
      newInputNode->insertBefore(*it);
      newInputNode->addInput(originalInputs.at(1));
      newInputNode->addInput(originalInputs.at(0));
      // Replace the second input of the current node with the output of the new node
      it->replaceInput(1, newInputNode->outputs().at(0));
    }
  }
}

void RemoveInplaceOps(const std::shared_ptr<Graph>& graph) {
  // Start the process of removing inplace operations from the entire graph
  ImplicitCastForBinaryInplaceOps(graph->block());
  RemoveInplaceOps(graph->block());
}
} // namespace jit
} // namespace torch
```