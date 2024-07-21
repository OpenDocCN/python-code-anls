# `.\pytorch\torch\csrc\api\include\torch\nn\functional\linear.h`

```
#pragma once

#include <torch/types.h>

namespace torch {
namespace nn {
namespace functional {

// 使用双线性插值计算两个输入张量的输出张量
inline Tensor bilinear(
    const Tensor& input1,
    const Tensor& input2,
    const Tensor& weight,
    const Tensor& bias = Tensor()) {
  // 调用 Torch 的双线性插值函数
  return torch::bilinear(input1, input2, weight, bias);
}

// ============================================================================

// 执行线性变换操作，返回结果张量
inline Tensor linear(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias = {}) {
  // 如果输入张量维度为2且定义了偏置张量，则使用融合操作以提高性能
  if (input.dim() == 2 && bias.defined()) {
    // 执行加权矩阵乘法并加上偏置
    return torch::addmm(bias, input, weight.t());
  } else {
    // 否则，执行矩阵乘法
    auto output = input.matmul(weight.t());
    // 如果定义了偏置张量，则将偏置加到输出张量上
    if (bias.defined()) {
      output += bias;
    }
    return output;
  }
}

} // namespace functional
} // namespace nn
} // namespace torch


这段代码是 C++ 的 Torch 库中的一些线性代数操作函数。注释解释了每个函数的作用以及特定条件下的优化措施。
```