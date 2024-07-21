# `.\pytorch\torch\csrc\autograd\variable_info.h`

```py
#pragma once
// 预处理指令，确保头文件仅被编译一次

#include <torch/csrc/autograd/variable.h>
// 包含 Torch 的自动求导模块中的 Variable 头文件

namespace torch::autograd {
// 命名空间 torch::autograd，包含 Torch 的自动求导相关内容

struct TORCH_API VariableInfo {
  // 定义结构体 VariableInfo，用于存储变量的信息

  explicit VariableInfo();
  // 默认构造函数声明，用于创建未初始化的 VariableInfo 对象

  explicit VariableInfo(const Variable& var);
  // 构造函数声明，接受一个 Variable 对象的引用作为参数

  Variable zeros(at::OptionalDeviceGuard& device_guard) const;
  // 成员函数声明，返回一个 Variable 对象，以指定的设备保护器 device_guard 初始化

  at::Layout layout = at::Layout::Strided;
  // 成员变量，表示 Tensor 的布局，默认为 Strided

  at::Device device = at::kCPU;
  // 成员变量，表示 Tensor 的设备，默认为 CPU

  at::ScalarType scalar_type = at::kFloat;
  // 成员变量，表示 Tensor 的数据类型，默认为 Float

  std::vector<c10::SymInt> size;
  // 成员变量，存储 Tensor 的尺寸信息的向量

  bool requires_grad;
  // 成员变量，表示是否需要梯度

  bool is_empty;
  // 成员变量，表示是否为空
};

} // namespace torch::autograd
// 结束命名空间 torch::autograd
```