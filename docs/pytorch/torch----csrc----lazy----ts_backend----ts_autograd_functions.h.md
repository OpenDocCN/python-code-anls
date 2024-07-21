# `.\pytorch\torch\csrc\lazy\ts_backend\ts_autograd_functions.h`

```py
#pragma once
// 使用#pragma once指令确保头文件只被包含一次，防止重复定义错误

#include <torch/csrc/autograd/custom_function.h>
// 包含torch库的自定义函数头文件，用于定义自定义的自动求导函数

namespace torch {
namespace lazy {

struct MaxPool3dAutogradFunctionTS
    : public torch::autograd::Function<MaxPool3dAutogradFunctionTS> {
  // 定义MaxPool3dAutogradFunctionTS结构体，继承自torch自动求导函数的模板类

  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      at::Tensor self,
      at::IntArrayRef kernel_size,
      at::IntArrayRef stride,
      at::IntArrayRef padding,
      at::IntArrayRef dilation,
      bool ceil_mode);
  // 声明静态的前向传播函数forward，接收自动求导上下文、张量self、核大小、步长、填充、扩展和是否使用ceil_mode参数，并返回张量

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output);
  // 声明静态的反向传播函数backward，接收自动求导上下文和梯度输出列表，并返回变量列表
};

} // namespace lazy
} // namespace torch
```