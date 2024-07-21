# `.\pytorch\torch\csrc\jit\passes\canonicalize.h`

```py
#pragma once
// 预处理指令：指示编译器确保头文件只包含一次

#include <torch/csrc/jit/ir/ir.h>
// 包含 Torch 库中用于 JIT 编译的 IR 相关头文件

namespace torch {
namespace jit {

TORCH_API std::shared_ptr<Graph> Canonicalize(
    const std::shared_ptr<Graph>& graph,
    bool keep_unique_names = true);
// 函数声明：对给定的图进行规范化操作，并返回规范化后的图对象指针

TORCH_API void CanonicalizeOutputs(std::shared_ptr<Graph>& graph);
// 函数声明：规范化给定图的输出节点

TORCH_API std::optional<const Use> firstOrLastUse(Value* v, bool find_first);
// 函数声明：返回值的第一个或最后一个使用位置

TORCH_API bool isBeforeOrAfter(
    const Use& a,
    const Use& b,
    bool checking_before);
// 函数声明：检查 Use 对象 a 是否在 Use 对象 b 之前或之后，根据 checking_before 参数决定

} // namespace jit
} // namespace torch
// 命名空间结束
```