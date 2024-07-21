# `.\pytorch\torch\csrc\jit\passes\autocast.h`

```
#pragma once

# 预处理指令 `#pragma once`：确保当前头文件只被编译一次，避免多重包含的问题


#include <torch/csrc/jit/ir/ir.h>

# 包含 Torch 库的头文件 `ir.h`：引入了 Torch 深度学习库中与中间表示（Intermediate Representation，IR）相关的头文件


namespace torch {
namespace jit {

# 命名空间 `torch::jit` 开始：定义了 Torch 库中 JIT 模块的命名空间，用于封装与即时编译（Just-in-Time compilation，JIT）相关的功能和类


TORCH_API void Autocast(const std::shared_ptr<Graph>& graph);

# 函数声明 `Autocast`：声明了一个公开的 API 函数 `Autocast`，用于在图（Graph）级别上执行自动类型转换


TORCH_API bool setAutocastMode(bool value);

# 函数声明 `setAutocastMode`：声明了一个公开的 API 函数 `setAutocastMode`，用于设置自动类型转换的模式


TORCH_API bool autocastEnabled();

# 函数声明 `autocastEnabled`：声明了一个公开的 API 函数 `autocastEnabled`，用于检查自动类型转换是否启用


} // namespace jit
} // namespace torch

# 命名空间 `torch::jit` 结束：结束了 Torch 库中 JIT 模块的命名空间定义
```