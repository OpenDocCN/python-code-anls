# `.\pytorch\torch\csrc\jit\passes\inline_fork_wait.h`

```
#pragma once
// 使用预处理指令#pragma once，确保头文件只被编译一次

#include <torch/csrc/jit/ir/ir.h>
// 包含 Torch 深度学习库的头文件，用于访问 JIT 编译器中的 IR

namespace torch {
namespace jit {

// 声明了一个命名空间 torch::jit，用于组织 JIT 编译器相关的功能

// Inline Fork and Wait calls. This is used, for example, in ONNX export, where
// we do not support the explicit parallelism structures and would rather
// just have a flat graph. This inlines the forked section in the fork()
// callsite and replaces uses of the result of wait() calls with the values
// produced from the (now-inlined) forked section.
// 定义了一个函数 InlineForkWait，用于内联 fork 和 wait 调用。例如在 ONNX 导出中使用，
// 其中不支持显式的并行结构，而是希望得到一个扁平的图。该函数会在 fork() 调用点内联 fork 的部分，
// 并用从现在内联的 fork 部分产生的值替换 wait() 调用结果的使用。

TORCH_API void InlineForkWait(const std::shared_ptr<Graph>& graph);
// 使用 TORCH_API 标记的函数声明，表示该函数是 Torch 库的公共 API，可以在库之外调用，
// 函数名为 InlineForkWait，接受一个 std::shared_ptr<Graph> 类型的参数 graph。

} // namespace jit
} // namespace torch
```