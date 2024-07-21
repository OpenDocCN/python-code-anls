# `.\pytorch\torch\csrc\api\include\torch\nn\options\normalization.h`

```
#pragma once

#include <torch/arg.h>
#include <torch/csrc/Export.h>
#include <torch/types.h>
#include <vector>

namespace torch {
namespace nn {

/// Options for the `LayerNorm` module.
///
/// Example:
/// ```
/// LayerNorm model(LayerNormOptions({2,
/// 2}).elementwise_affine(false).eps(2e-5));
/// ```
struct TORCH_API LayerNormOptions {
  /* implicit */ LayerNormOptions(std::vector<int64_t> normalized_shape);
  /// input shape from an expected input.
  TORCH_ARG(std::vector<int64_t>, normalized_shape);
  /// a value added to the denominator for numerical stability. ``Default:
  /// 1e-5``.
  TORCH_ARG(double, eps) = 1e-5;
  /// a boolean value that when set to ``true``, this module
  /// has learnable per-element affine parameters initialized to ones (for
  /// weights) and zeros (for biases). ``Default: true``.
  TORCH_ARG(bool, elementwise_affine) = true;
};

// ============================================================================

namespace functional {

/// Options for `torch::nn::functional::layer_norm`.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::layer_norm(input, F::LayerNormFuncOptions({2, 2}).eps(2e-5));
/// ```
struct TORCH_API LayerNormFuncOptions {
  /* implicit */ LayerNormFuncOptions(std::vector<int64_t> normalized_shape);
  /// input shape from an expected input.
  TORCH_ARG(std::vector<int64_t>, normalized_shape);

  TORCH_ARG(Tensor, weight) = {};

  TORCH_ARG(Tensor, bias) = {};

  /// a value added to the denominator for numerical stability. ``Default:
  /// 1e-5``.
  TORCH_ARG(double, eps) = 1e-5;
};

} // namespace functional

// ============================================================================

/// Options for the `LocalResponseNorm` module.
///
/// Example:
/// ```
/// LocalResponseNorm
/// model(LocalResponseNormOptions(2).alpha(0.0002).beta(0.85).k(2.));
/// ```
struct TORCH_API LocalResponseNormOptions {
  /* implicit */ LocalResponseNormOptions(int64_t size) : size_(size) {}
  /// amount of neighbouring channels used for normalization
  TORCH_ARG(int64_t, size);

  /// multiplicative factor. Default: 1e-4
  TORCH_ARG(double, alpha) = 1e-4;

  /// exponent. Default: 0.75
  TORCH_ARG(double, beta) = 0.75;

  /// additive factor. Default: 1
  TORCH_ARG(double, k) = 1.;
};

namespace functional {
/// Options for `torch::nn::functional::local_response_norm`.
///
/// See the documentation for `torch::nn::LocalResponseNormOptions` class to
/// learn what arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::local_response_norm(x, F::LocalResponseNormFuncOptions(2));
/// ```
using LocalResponseNormFuncOptions = LocalResponseNormOptions;
} // namespace functional

// ============================================================================

/// Options for the `CrossMapLRN2d` module.
///
/// Example:
/// ```
/// CrossMapLRN2d model(CrossMapLRN2dOptions(3).alpha(1e-5).beta(0.1).k(10));
/// ```
struct TORCH_API CrossMapLRN2dOptions {
  /* implicit */ CrossMapLRN2dOptions(int64_t size) : size_(size) {}
  /// size of the local region across channels
  TORCH_ARG(int64_t, size);

  /// multiplicative factor. Default: 1e-4
  TORCH_ARG(double, alpha) = 1e-4;

  /// exponent. Default: 0.75
  TORCH_ARG(double, beta) = 0.75;

  /// additive factor. Default: 1
  TORCH_ARG(double, k) = 1.;
};
/// 结构体 `CrossMapLRN2dOptions` 的选项定义，用于配置跨通道局部响应归一化。
struct TORCH_API CrossMapLRN2dOptions {
  /// 构造函数，根据给定的尺寸初始化选项对象
  CrossMapLRN2dOptions(int64_t size);

  /// 属性：局部归一化窗口的大小
  TORCH_ARG(int64_t, size);

  /// 属性：归一化的参数之一，默认值为 1e-4
  TORCH_ARG(double, alpha) = 1e-4;

  /// 属性：归一化的参数之一，默认值为 0.75
  TORCH_ARG(double, beta) = 0.75;

  /// 属性：归一化的参数之一，默认值为 1
  TORCH_ARG(int64_t, k) = 1;
};

// ============================================================================

namespace functional {

/// `torch::nn::functional::normalize` 函数的选项定义。
///
/// 示例：
/// ```
/// namespace F = torch::nn::functional;
/// F::normalize(input, F::NormalizeFuncOptions().p(1).dim(-1));
/// ```
struct TORCH_API NormalizeFuncOptions {
  /// 属性：范数计算中的指数值，默认为 2.0
  TORCH_ARG(double, p) = 2.0;

  /// 属性：要减少的维度，默认为 1
  TORCH_ARG(int64_t, dim) = 1;

  /// 属性：避免除以零的小值，默认为 1e-12
  TORCH_ARG(double, eps) = 1e-12;

  /// 属性：输出张量的可选参数。如果指定了 `out`，该操作将不可微分。
  TORCH_ARG(std::optional<Tensor>, out) = c10::nullopt;
};

} // namespace functional

// ============================================================================

/// `GroupNorm` 模块的选项定义。
///
/// 示例：
/// ```
/// GroupNorm model(GroupNormOptions(2, 2).eps(2e-5).affine(false));
/// ```
struct TORCH_API GroupNormOptions {
  /* implicit */ GroupNormOptions(int64_t num_groups, int64_t num_channels);

  /// 属性：将通道分成的组数
  TORCH_ARG(int64_t, num_groups);

  /// 属性：输入中预期的通道数
  TORCH_ARG(int64_t, num_channels);

  /// 属性：分母上添加的值，用于数值稳定性，默认为 1e-5
  TORCH_ARG(double, eps) = 1e-5;

  /// 属性：布尔值，如果为 `true`，此模块具有可学习的通道仿射参数，默认为 `true`
  TORCH_ARG(bool, affine) = true;
};

// ============================================================================

namespace functional {

/// `torch::nn::functional::group_norm` 函数的选项定义。
///
/// 示例：
/// ```
/// namespace F = torch::nn::functional;
/// F::group_norm(input, F::GroupNormFuncOptions(2).eps(2e-5));
/// ```
struct TORCH_API GroupNormFuncOptions {
  /* implicit */ GroupNormFuncOptions(int64_t num_groups);

  /// 属性：将通道分成的组数
  TORCH_ARG(int64_t, num_groups);

  /// 属性：权重张量
  TORCH_ARG(Tensor, weight) = {};

  /// 属性：偏置张量
  TORCH_ARG(Tensor, bias) = {};

  /// 属性：分母上添加的值，用于数值稳定性，默认为 1e-5
  TORCH_ARG(double, eps) = 1e-5;
};

} // namespace functional
```