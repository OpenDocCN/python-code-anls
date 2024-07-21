# `.\pytorch\torch\csrc\jit\passes\fold_conv_bn.h`

```py
#pragma once
// 预处理指令，确保头文件只被编译一次

#include <torch/csrc/jit/api/module.h>
// 包含 Torch C++ API 的模块头文件

namespace torch {
namespace jit {

/** \brief 将该模块及其所有子模块中的 Conv2d-BatchNorm2d 折叠成 Conv2d，
 * 默认包括 forward 方法。
 *
 * Conv2d 的权重和偏置将相应地更新。仅应在评估模式下的模块上使用。
 */
TORCH_API Module FoldConvBatchNorm(const Module& module);
// 声明折叠 Conv2d-BatchNorm2d 到 Conv2d 的函数

struct TORCH_API ConvBNParameters {
  at::Tensor conv_w;
  // Conv2d 的权重张量
  at::Tensor conv_b;
  // Conv2d 的偏置张量
  at::Tensor bn_rm;
  // BatchNorm 的 running_mean 张量
  at::Tensor bn_rv;
  // BatchNorm 的 running_var 张量
  double bn_eps = 0.0;
  // BatchNorm 的 epsilon 值，默认为 0.0
  at::Tensor bn_w;
  // BatchNorm 的权重张量
  at::Tensor bn_b;
  // BatchNorm 的偏置张量
};

/**
 * 根据 Conv 模块的当前权重和偏置张量以及与其折叠的 BatchNorm 模块的参数，
 * 计算更新后的权重和偏置值。
 *
 * 该函数基本上是从 torch/nn/utils/fusion.py 复制过来的。
 */
TORCH_API std::tuple<at::Tensor, at::Tensor> computeUpdatedConvWeightAndBias(
    const ConvBNParameters& p);
// 声明计算更新后的 Conv2d 权重和偏置的函数

} // namespace jit
} // namespace torch
```