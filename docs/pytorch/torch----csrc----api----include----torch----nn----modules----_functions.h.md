# `.\pytorch\torch\csrc\api\include\torch\nn\modules\_functions.h`

```
#pragma once
// 使用预处理指令#pragma once，确保头文件只被编译一次

#include <torch/csrc/autograd/custom_function.h>
// 引入torch自动求导模块中的custom_function.h头文件

#include <torch/csrc/autograd/variable.h>
// 引入torch自动求导模块中的variable.h头文件

#include <torch/nn/options/normalization.h>
// 引入torch神经网络模块中的normalization.h头文件

#include <torch/types.h>
// 引入torch类型定义头文件

namespace torch {
namespace nn {
namespace functions {

class CrossMapLRN2d : public torch::autograd::Function<CrossMapLRN2d> {
// 定义CrossMapLRN2d类，继承自torch自动求导模块中的Function模板类，模板参数为CrossMapLRN2d本身
 public:
  static torch::autograd::Variable forward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::Variable& input,
      const CrossMapLRN2dOptions& options);
  // 声明静态成员函数forward，接受自动求导上下文、输入变量、CrossMapLRN2dOptions参数，并返回torch的自动求导变量

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output);
  // 声明静态成员函数backward，接受自动求导上下文、梯度输出列表，并返回torch的自动求导变量列表
};

} // namespace functions
} // namespace nn
} // namespace torch
// 结束torch命名空间和相关命名空间的声明
```