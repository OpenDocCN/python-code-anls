# `.\pytorch\torch\csrc\jit\passes\onnx\cast_all_constant_to_floating.h`

```py
#pragma once

# 预处理指令，确保头文件只包含一次


#include <torch/csrc/jit/ir/ir.h>

# 包含 Torch 深度学习库中的 IR 头文件


#include <memory>

# 包含 C++ 标准库中的内存管理头文件


namespace torch {
namespace jit {

# 声明 Torch 命名空间和 JIT 子命名空间


// see .cpp for docs

# 用于说明需要查看具体的 .cpp 文件获取更多文档信息


TORCH_API void CastAllConstantToFloating(const std::shared_ptr<Graph>& graph);

# 声明函数 CastAllConstantToFloating，用于将所有常量转换为浮点数类型，接受一个指向 Graph 对象的共享指针参数


} // namespace jit
} // namespace torch

# 结束 Torch 的 jit 命名空间和 torch 命名空间
```