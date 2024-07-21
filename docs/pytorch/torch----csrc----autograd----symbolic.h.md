# `.\pytorch\torch\csrc\autograd\symbolic.h`

```py
#pragma once
// 预处理指令，用于确保头文件只被编译一次

#include <torch/csrc/jit/ir/ir.h>
// 包含 Torch 库中 JIT 模块的 IR 相关头文件

#include <torch/csrc/onnx/onnx.h>
// 包含 Torch 库中 ONNX 模块的相关头文件

namespace torch::autograd {
// 进入 torch::autograd 命名空间

struct SymbolicContext {
  jit::Block* block;
  // 定义结构体 SymbolicContext，包含一个指向 jit::Block 类型的指针成员变量 block
};

struct symbolic_unconvertible : public std::runtime_error {
  using std::runtime_error::runtime_error;
  // 定义一个异常类 symbolic_unconvertible，继承自 std::runtime_error，
  // 并使用基类的构造函数初始化异常消息
};

} // namespace torch::autograd
// 退出 torch::autograd 命名空间
```