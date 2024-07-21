# `.\pytorch\torch\csrc\jit\passes\fold_linear_bn.h`

```
#pragma once
// 预处理指令，确保头文件只被包含一次

#include <torch/csrc/jit/api/module.h>
// 包含 Torch C++ API 中模块的头文件

namespace torch {
namespace jit {

struct TORCH_API LinearBNParameters {
  at::Tensor linear_w;   // 线性层的权重张量
  at::Tensor linear_b;   // 线性层的偏置张量
  at::Tensor bn_rm;      // 批归一化的运行均值张量
  at::Tensor bn_rv;      // 批归一化的运行方差张量
  double bn_eps = 0.0;   // 批归一化的 epsilon 参数，默认值为 0.0
  at::Tensor bn_w;       // 批归一化的权重张量
  at::Tensor bn_b;       // 批归一化的偏置张量
};

/**
 * Given the current weight and bias tensors of a Linear module and parameters
 * of the BatchNorm module we're folding with, compute the updated values
 * for the weight and bias.
 *
 * The function is basically copied from torch/nn/utils/fusion.py
 */
TORCH_API std::tuple<at::Tensor, at::Tensor> computeUpdatedLinearWeightAndBias(
    const LinearBNParameters& p);
// 计算更新后的线性模块权重和偏置的函数声明，接收 LinearBNParameters 结构体作为参数

} // namespace jit
} // namespace torch
```