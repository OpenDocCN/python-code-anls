# `.\pytorch\torch\csrc\jit\passes\constant_propagation.h`

```py
#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

// 对传入的图形进行常量传播，除非 ignore_custom_classes 参数为 true，否则会处理所有对象。
// 这对于防止过早融合（如 packed::linear_clamp_prepack 和 prepacked::conv2d_clamp_prepack）非常有用，
// 因为这些操作会降低有关其构造函数的信息。
// 如果此传递对图形进行了更改，则返回 true。
TORCH_API bool ConstantPropagation(
    std::shared_ptr<Graph>& graph,
    bool ignore_custom_classes = false);

// 仅对具有非别名输入和输出的操作运行常量传播。
// 如果此传递对图形进行了更改，则返回 true。
TORCH_API bool ConstantPropagationImmutableTypes(std::shared_ptr<Graph>& graph);

// 如果节点的输入是常量，则运行节点。
// 调用此函数的调用者必须自行确定是否适合常量传播，例如非确定性操作或具有副作用的操作。
// 如果指定了 ignore_custom_classes，则不运行输出用户定义类的节点。
TORCH_API std::optional<Stack> runNodeIfInputsAreConstant(
    const Node* node,
    bool ignore_custom_classes = false,
    AliasDb* db = nullptr);

} // namespace jit
} // namespace torch
```