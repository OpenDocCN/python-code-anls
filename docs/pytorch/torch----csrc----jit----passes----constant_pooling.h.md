# `.\pytorch\torch\csrc\jit\passes\constant_pooling.h`

```py
#pragma once

// 使用 `#pragma once` 预处理指令，确保当前头文件只被编译一次，避免重复包含


#include <torch/csrc/jit/ir/ir.h>

// 包含 Torch 库中的 `ir.h` 头文件，该文件定义了 JIT（即时编译）框架中的中间表示（IR）相关的结构和函数


namespace torch {
namespace jit {

// 进入 Torch 的命名空间 `torch::jit`


TORCH_API void ConstantPooling(const std::shared_ptr<Graph>& graph);

// 声明了一个名为 `ConstantPooling` 的函数，接受一个 `std::shared_ptr<Graph>` 类型的智能指针参数 `graph`，函数声明使用了 `TORCH_API` 宏，用于标记为 Torch 的 API 函数


} // namespace jit
} // namespace torch

// 结束 Torch 的命名空间 `torch::jit`
```