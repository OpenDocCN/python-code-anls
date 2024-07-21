# `.\pytorch\torch\csrc\jit\passes\onnx\eval_peephole.h`

```py
#pragma once

# 使用 pragma once 指令，确保头文件只被编译一次，防止多重包含的问题


#include <memory>

# 包含 memory 头文件，用于使用智能指针等内存管理工具


#include <torch/csrc/jit/ir/ir.h>

# 包含 Torch 的 JIT 模块中的 ir.h 头文件，提供了 JIT IR 的相关功能和数据结构


namespace torch {
namespace jit {

# 进入 torch 命名空间，然后进入 jit 命名空间，用于组织代码结构和避免命名冲突


void EvalPeepholeONNX(
    std::shared_ptr<Graph>& g,
    std::map<std::string, IValue>& paramDict);

# 声明 EvalPeepholeONNX 函数，该函数接受一个指向 Graph 对象的共享指针 g 和一个映射 paramDict，映射键为字符串，值为 IValue 类型


} // namespace jit
} // namespace torch

# 退出 jit 和 torch 命名空间，结束命名空间的定义
```