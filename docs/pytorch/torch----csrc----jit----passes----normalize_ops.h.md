# `.\pytorch\torch\csrc\jit\passes\normalize_ops.h`

```
#pragma once
// 预处理指令：#pragma once，确保头文件只被编译一次

#include <torch/csrc/jit/ir/ir.h>
// 包含 Torch 库中的 IR 相关头文件

namespace torch {
namespace jit {

// 命名空间 torch::jit，包含了 Torch 的 JIT 编译器相关实现

// This pass converts aten ops to a normalized form. It is
// run immediately after IR generation in both the tracer and compiler,
// so downstream consumers of the IR do not need handle ops in their
// pre-normalized form.
// Currently only handles normalization of op aliases.
// 这个 pass 将 aten 操作转换为标准化形式。它在 IR 生成后立即运行，
// 在追踪器和编译器中都会运行，因此 IR 的下游使用者不需要处理未标准化的操作。
// 目前仅处理操作别名的标准化。

TORCH_API void NormalizeOps(const std::shared_ptr<Graph>& graph);
// 声明 NormalizeOps 函数，该函数用于将操作标准化，参数为图的共享指针

const std::unordered_map<Symbol, Symbol>& getOperatorAliasMap();
// 声明 getOperatorAliasMap 函数，返回一个无序映射，表示操作符的别名映射表

} // namespace jit
} // namespace torch
// 命名空间 jit 和 torch，分别包含了 JIT 编译器的实现
```