# `.\pytorch\torch\csrc\jit\passes\device_type_analysis.h`

```
#pragma once
// 使用 #pragma once 指令确保头文件只被编译一次

#include <torch/csrc/jit/ir/ir.h>
// 包含 Torch 库中的 IR 头文件，用于处理图形数据结构

namespace torch {
namespace jit {
struct Graph;
// 声明 Graph 结构体，用于表示图形数据结构

// 在给定的图形中传播设备类型信息。
TORCH_API bool DeviceTypePropagation(std::shared_ptr<Graph>& graph);
// 声明函数 DeviceTypePropagation，接受一个指向 Graph 结构体的共享指针作为参数，并返回布尔值

} // namespace jit
} // namespace torch
// 结束 Torch 的命名空间声明
```