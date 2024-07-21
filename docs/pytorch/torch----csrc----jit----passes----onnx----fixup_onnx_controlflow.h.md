# `.\pytorch\torch\csrc\jit\passes\onnx\fixup_onnx_controlflow.h`

```py
#pragma once
// 使用 pragma once 指令确保头文件只被编译一次

#include <torch/csrc/jit/ir/ir.h>
// 包含 Torch 深度学习框架的 JIT 模块中的 IR 头文件

namespace torch {
namespace jit {

std::vector<Value*> FixupONNXControlflowNode(Node* n, int opset_version);
// 声明一个函数 FixupONNXControlflowNode，接受一个 Node 指针和一个整数 opset_version 作为参数，并返回一个 Value* 类型的向量

void FixupONNXControlflowNodeOutputs(Node* n);
// 声明一个函数 FixupONNXControlflowNodeOutputs，接受一个 Node 指针作为参数，无返回值

} // namespace jit
} // namespace torch
// 命名空间声明结束，包含了 torch::jit 命名空间，其中定义了上述两个函数
```