# `.\pytorch\torch\csrc\api\include\torch\nn\options\instancenorm.h`

```py
#pragma once
// 预处理指令，确保头文件只被编译一次

#include <torch/arg.h>
// 包含 Torch 库中的 arg.h 头文件

#include <torch/csrc/Export.h>
// 包含 Torch 库中的 Export.h 头文件

#include <torch/nn/options/batchnorm.h>
// 包含 Torch 库中的 batchnorm.h 头文件

#include <torch/types.h>
// 包含 Torch 库中的 types.h 头文件

namespace torch {
namespace nn {

/// Options for the `InstanceNorm` module.
/// `InstanceNorm` 模块的选项

struct TORCH_API InstanceNormOptions {
  /* implicit */ InstanceNormOptions(int64_t num_features);
  // 构造函数，接受 num_features 参数

  /// The number of features of the input tensor.
  /// 输入张量的特征数量
  TORCH_ARG(int64_t, num_features);

  /// The epsilon value added for numerical stability.
  /// 数值稳定性所添加的 epsilon 值
  TORCH_ARG(double, eps) = 1e-5;

  /// A momentum multiplier for the mean and variance.
  /// 均值和方差的动量乘数
  TORCH_ARG(double, momentum) = 0.1;

  /// Whether to learn a scale and bias that are applied in an affine
  /// transformation on the input.
  /// 是否学习应用于输入的仿射变换的比例和偏置
  TORCH_ARG(bool, affine) = false;

  /// Whether to store and update batch statistics (mean and variance) in the
  /// module.
  /// 是否在模块中存储和更新批次统计信息（均值和方差）
  TORCH_ARG(bool, track_running_stats) = false;
};

/// Options for the `InstanceNorm1d` module.
/// `InstanceNorm1d` 模块的选项
///
/// Example:
/// ```
/// InstanceNorm1d
/// model(InstanceNorm1dOptions(4).eps(0.5).momentum(0.1).affine(false).track_running_stats(true));
/// ```py
using InstanceNorm1dOptions = InstanceNormOptions;

/// Options for the `InstanceNorm2d` module.
/// `InstanceNorm2d` 模块的选项
///
/// Example:
/// ```
/// InstanceNorm2d
/// model(InstanceNorm2dOptions(4).eps(0.5).momentum(0.1).affine(false).track_running_stats(true));
/// ```py
using InstanceNorm2dOptions = InstanceNormOptions;

/// Options for the `InstanceNorm3d` module.
/// `InstanceNorm3d` 模块的选项
///
/// Example:
/// ```
/// InstanceNorm3d
/// model(InstanceNorm3dOptions(4).eps(0.5).momentum(0.1).affine(false).track_running_stats(true));
/// ```py
using InstanceNorm3dOptions = InstanceNormOptions;

namespace functional {

/// Options for `torch::nn::functional::instance_norm`.
/// `torch::nn::functional::instance_norm` 的选项
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::instance_norm(input,
/// F::InstanceNormFuncOptions().running_mean(mean).running_var(variance).weight(weight).bias(bias).momentum(0.1).eps(1e-5));
/// ```py
struct TORCH_API InstanceNormFuncOptions {
  TORCH_ARG(Tensor, running_mean) = Tensor();
  // 运行均值的张量选项，默认为空张量

  TORCH_ARG(Tensor, running_var) = Tensor();
  // 运行方差的张量选项，默认为空张量

  TORCH_ARG(Tensor, weight) = Tensor();
  // 权重张量选项，默认为空张量

  TORCH_ARG(Tensor, bias) = Tensor();
  // 偏置张量选项，默认为空张量

  TORCH_ARG(bool, use_input_stats) = true;
  // 是否使用输入统计信息，默认为 true

  TORCH_ARG(double, momentum) = 0.1;
  // 动量参数，默认为 0.1

  TORCH_ARG(double, eps) = 1e-5;
  // epsilon 参数，默认为 1e-5
};

} // namespace functional

} // namespace nn
} // namespace torch
```