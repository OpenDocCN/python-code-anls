# `.\pytorch\torch\csrc\api\include\torch\nn\options\dropout.h`

```py
#pragma once

#include <torch/arg.h>
#include <torch/csrc/Export.h>
#include <torch/types.h>

namespace torch {
namespace nn {

/// Options for the `Dropout` module.
///
/// Example:
/// ```
/// Dropout model(DropoutOptions().p(0.42).inplace(true));
/// ```py
struct TORCH_API DropoutOptions {
  /* implicit */ DropoutOptions(double p = 0.5);

  /// The probability of an element to be zeroed. Default: 0.5
  TORCH_ARG(double, p) = 0.5;

  /// can optionally do the operation in-place. Default: False
  TORCH_ARG(bool, inplace) = false;
};

/// Options for the `Dropout2d` module.
///
/// Example:
/// ```
/// Dropout2d model(Dropout2dOptions().p(0.42).inplace(true));
/// ```py
using Dropout2dOptions = DropoutOptions;

/// Options for the `Dropout3d` module.
///
/// Example:
/// ```
/// Dropout3d model(Dropout3dOptions().p(0.42).inplace(true));
/// ```py
using Dropout3dOptions = DropoutOptions;

/// Options for the `AlphaDropout` module.
///
/// Example:
/// ```
/// AlphaDropout model(AlphaDropoutOptions(0.2).inplace(true));
/// ```py
using AlphaDropoutOptions = DropoutOptions;

/// Options for the `FeatureAlphaDropout` module.
///
/// Example:
/// ```
/// FeatureAlphaDropout model(FeatureAlphaDropoutOptions(0.2).inplace(true));
/// ```py
using FeatureAlphaDropoutOptions = DropoutOptions;

namespace functional {

/// Options for `torch::nn::functional::dropout`.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::dropout(input, F::DropoutFuncOptions().p(0.5));
/// ```py
struct TORCH_API DropoutFuncOptions {
  /// The probability of an element to be zeroed. Default: 0.5
  TORCH_ARG(double, p) = 0.5;

  /// Whether the module is in training mode. Default: true
  TORCH_ARG(bool, training) = true;

  /// can optionally do the operation in-place. Default: False
  TORCH_ARG(bool, inplace) = false;
};

/// Options for `torch::nn::functional::dropout2d`.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::dropout2d(input, F::Dropout2dFuncOptions().p(0.5));
/// ```py
using Dropout2dFuncOptions = DropoutFuncOptions;

/// Options for `torch::nn::functional::dropout3d`.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::dropout3d(input, F::Dropout3dFuncOptions().p(0.5));
/// ```py
using Dropout3dFuncOptions = DropoutFuncOptions;

/// Options for `torch::nn::functional::alpha_dropout`.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::alpha_dropout(input, F::AlphaDropoutFuncOptions().p(0.5).training(false));
/// ```py
struct TORCH_API AlphaDropoutFuncOptions {
  /// The probability of an element to be zeroed. Default: 0.5
  TORCH_ARG(double, p) = 0.5;

  /// Whether the module is in training mode. Default: false
  TORCH_ARG(bool, training) = false;

  /// can optionally do the operation in-place. Default: False
  TORCH_ARG(bool, inplace) = false;
};

/// Options for `torch::nn::functional::feature_alpha_dropout`.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::feature_alpha_dropout(input, F::FeatureAlphaDropoutFuncOptions().p(0.5).training(false));
/// ```py
struct TORCH_API FeatureAlphaDropoutFuncOptions {
  /// The probability of an element to be zeroed. Default: 0.5
  TORCH_ARG(double, p) = 0.5;

  /// Whether the module is in training mode. Default: false
  TORCH_ARG(bool, training) = false;

  /// can optionally do the operation in-place. Default: False
  TORCH_ARG(bool, inplace) = false;
};


注释：
};

// 结束 nn 命名空间
} // namespace nn

// 结束 torch 命名空间
} // namespace torch
```