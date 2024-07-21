# `.\pytorch\torch\csrc\jit\passes\remove_expands.h`

```py
#pragma once

// 使用 `#pragma once` 指令，确保头文件只被编译一次，以防止多重包含的问题。


#include <torch/csrc/jit/ir/ir.h>

// 引入 Torch 库中的 `ir.h` 头文件，该文件包含了与图形（Graph）相关的中间表示（Intermediate Representation，IR）的定义。


namespace torch {
namespace jit {

// 进入 Torch 的命名空间 `torch::jit`，用于访问 Torch JIT 模块提供的功能和类。


TORCH_API void RemoveExpands(const std::shared_ptr<Graph>& graph);

// 声明一个函数 `RemoveExpands`，该函数位于 `torch::jit` 命名空间中，接受一个指向 `Graph` 对象的共享指针参数，并使用 `TORCH_API` 标记来指示该函数是库的公共接口。


} // namespace jit
} // namespace torch

// 结束命名空间声明，回到全局命名空间或者上层命名空间，本例中是 `torch` 命名空间。
```