# `.\pytorch\torch\csrc\jit\passes\lift_closures.h`

```
#pragma once

该指令告诉编译器只包含本文件一次，避免重复包含。


#include <torch/csrc/Export.h>
#include <torch/csrc/jit/ir/ir.h>

这两行代码是预处理指令，用于包含 Torch 库中的头文件 `Export.h` 和 `ir.h`，以便在编译时能够使用其中定义的函数和数据结构。


namespace torch {
namespace jit {

定义了命名空间 `torch::jit`，用于组织和限定代码中的标识符，避免命名冲突。


TORCH_API void liftClosures(const std::shared_ptr<Graph>& graph);

声明了一个函数 `liftClosures`，它接受一个名为 `graph` 的 `std::shared_ptr` 类型的参数，并且带有 `TORCH_API` 标识符，该标识符通常用于声明需要在动态链接库中导出的函数。


} // namespace jit
} // namespace torch

结束了命名空间 `torch::jit` 和 `torch` 的定义。
```