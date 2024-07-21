# `.\pytorch\torch\csrc\jit\backends\backend_resolver.h`

```py
#pragma once

// 使用 `#pragma once` 预处理指令，确保头文件只被编译一次，提高编译效率


#include <torch/csrc/jit/frontend/resolver.h>

// 包含 Torch 框架中的 `resolver.h` 头文件，以便在本文件中使用相关解析器功能


namespace torch {
namespace jit {

// 定义命名空间 `torch::jit`，用于封装 Torch JIT 编译器的相关功能和类


// Create a Resolver for use in generating LoweredModules for specific backends.

// 创建一个 Resolver 对象，用于生成特定后端的 LoweredModules


TORCH_API std::shared_ptr<Resolver> loweredModuleResolver();

// 声明一个函数 `loweredModuleResolver()`，返回一个 `std::shared_ptr` 类型的 Resolver 指针，用于创建降级模块的解析器


} // namespace jit
} // namespace torch

// 结束 `torch::jit` 命名空间和 `torch` 命名空间
```