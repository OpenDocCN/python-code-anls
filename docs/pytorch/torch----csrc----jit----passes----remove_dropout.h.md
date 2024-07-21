# `.\pytorch\torch\csrc\jit\passes\remove_dropout.h`

```
#pragma once


// 指令，确保本头文件在编译过程中只被包含一次，防止重复定义错误
#pragma once

#include <torch/csrc/jit/api/module.h>
// 引入 Torch 的模块 API 头文件

#include <torch/csrc/jit/ir/ir.h>
// 引入 Torch 的 IR（Intermediate Representation，中间表示）处理头文件

namespace torch {
namespace jit {

// Torch JIT（Just-In-Time，即时编译）命名空间开始

TORCH_API void removeDropout(std::shared_ptr<Graph>& graph);
// 声明一个名为 removeDropout 的函数，接受一个指向 Graph 的 shared_ptr 参数，并标记为 Torch API

TORCH_API void removeDropout(script::Module& module);
// 声明一个名为 removeDropout 的函数，接受一个 script::Module 类型的引用参数，并标记为 Torch API

} // namespace jit
} // namespace torch
// Torch JIT 命名空间结束
```