# `.\pytorch\torch\csrc\distributed\rpc\rpc.h`

```py
#pragma once
// 预处理指令：确保本文件只被编译一次，避免重复包含

#include <torch/csrc/python_headers.h>
// 包含 Torch C++ 库的 Python 头文件，用于与 Python 解释器交互

namespace torch {
namespace distributed {
namespace rpc {

PyMethodDef* python_functions();
// 声明一个函数 python_functions()，返回类型为 PyMethodDef*

} // namespace rpc
} // namespace distributed
} // namespace torch
// 命名空间 torch.distributed.rpc 中定义了一个指向 PyMethodDef 结构体的指针
```