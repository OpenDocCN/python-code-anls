# `.\pytorch\torch\csrc\jit\passes\variadic_ops.h`

```py
#pragma once
// 防止头文件被多次包含

#include <torch/csrc/jit/ir/ir.h>
// 包含 Torch 的 JIT IR 头文件

namespace torch {
namespace jit {

// 尝试用支持可变参数的操作替换接受列表输入的操作
TORCH_API bool UseVariadicOp(
    const std::shared_ptr<Graph>& graph,
    NodeKind op,
    NodeKind variadic_op);

// 尝试移除列表修改并使用支持可变参数的操作替换
TORCH_API bool RemoveListMutationAndUseVariadicOp(
    const std::shared_ptr<Graph>& graph,
    NodeKind op,
    NodeKind variadic_op);

// 便捷函数，用于将 aten::stack/aten::cat 替换为它们的可变参数版本
TORCH_API bool UseVariadicCat(const std::shared_ptr<Graph>& graph);
// 尝试移除列表修改并使用可变参数的 aten::cat 替换
TORCH_API bool RemoveListMutationAndUseVariadicCat(
    const std::shared_ptr<Graph>& graph);

// 便捷函数，用于将 aten::stack 替换为其可变参数版本
TORCH_API bool UseVariadicStack(const std::shared_ptr<Graph>& graph);
// 尝试移除列表修改并使用可变参数的 aten::stack 替换
TORCH_API bool RemoveListMutationAndUseVariadicStack(
    const std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
```