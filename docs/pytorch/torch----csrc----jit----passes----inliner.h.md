# `.\pytorch\torch\csrc\jit\passes\inliner.h`

```py
#pragma once
// 防止头文件重复包含的预处理指令

#include <torch/csrc/jit/ir/ir.h>
// 包含 Torch 深度学习框架中用于 JIT 的 IR 头文件

namespace torch {
namespace jit {

// Torch JIT 的命名空间开始

TORCH_API void Inline(Graph& graph);
// 声明了一个名为 Inline 的函数，接受一个 Graph 对象的引用作为参数，没有返回值。该函数用于进行内联函数和方法调用。

TORCH_API GraphFunction* tryToGraphFunction(Node* n);
// 声明了一个名为 tryToGraphFunction 的函数，接受一个 Node 指针作为参数，返回一个 GraphFunction 指针。
// 该函数尝试将给定节点转换为图函数对象。

} // namespace jit
} // namespace torch
// Torch JIT 的命名空间结束
```