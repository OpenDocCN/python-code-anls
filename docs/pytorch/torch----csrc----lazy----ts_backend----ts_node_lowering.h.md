# `.\pytorch\torch\csrc\lazy\ts_backend\ts_node_lowering.h`

```py
#pragma once

# 使用 pragma once 指令，确保头文件只被编译一次，防止重复包含


#include <torch/csrc/api/include/torch/jit.h>
#include <torch/csrc/lazy/backend/lowering_context.h>

# 引入 Torch 的头文件，用于 JIT 编译和懒执行相关的功能


namespace torch {
namespace lazy {

# 定义命名空间 torch::lazy，用于封装懒执行相关的功能


using TSOpVector = std::vector<torch::jit::Value*>;

# 别名定义 TSOpVector 为 std::vector<torch::jit::Value*>，用于存储 JIT 值的向量


TORCH_API TSOpVector LowerTSBuiltin(
    std::shared_ptr<torch::jit::GraphFunction> function,
    c10::Symbol sym,
    const std::vector<torch::jit::NamedValue>& arguments,
    const std::vector<torch::jit::NamedValue>& kwarguments = {});

# 声明 LowerTSBuiltin 函数，用于降低内置 TorchScript 操作，接受函数对象、符号、参数和关键字参数


} // namespace lazy
} // namespace torch

# 命名空间结束标记，分别结束了 torch::lazy 和 torch 命名空间
```