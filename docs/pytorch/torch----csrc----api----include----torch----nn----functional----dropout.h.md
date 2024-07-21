# `.\pytorch\torch\csrc\api\include\torch\nn\functional\dropout.h`

```
#pragma once

#include <torch/nn/options/dropout.h> // 引入dropout相关的选项定义

#include <utility> // 引入utility工具库

namespace torch {
namespace nn {
namespace functional {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {

// 定义了dropout函数，根据输入的参数进行dropout操作
inline Tensor dropout(Tensor input, double p, bool training, bool inplace) {
  TORCH_CHECK(
      p >= 0. && p <= 1.,
      "dropout probability has to be between 0 and 1, but got ",
      p); // 检查dropout概率p是否在合法范围内

  if (inplace) {
    return torch::dropout_(input, p, training); // 若inplace为true，则原地进行dropout操作
  } else {
    return torch::dropout(input, p, training); // 若inplace为false，则创建新的dropout后的Tensor
  }
}

} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.dropout
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::DropoutFuncOptions` class
/// to learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::dropout(input, F::DropoutFuncOptions().p(0.5));
/// ```
// dropout函数的外部接口，调用detail命名空间中的dropout函数实现
inline Tensor dropout(Tensor input, const DropoutFuncOptions& options = {}) {
  return detail::dropout(
      std::move(input), options.p(), options.training(), options.inplace());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {

// 辅助函数，用于实现多维度dropout操作
template <int64_t unbatched_dim, int64_t batched_dim>
inline Tensor _dropoutNd_helper(
    Tensor input,
    double p,
    bool training,
    bool inplace,
    const char* fn_name) {
  TORCH_CHECK(
      p >= 0. && p <= 1.,
      "dropout probability has to be between 0 and 1, but got ",
      p); // 检查dropout概率p是否在合法范围内

  auto inp_dim = input.dim();
  auto is_batched = inp_dim == batched_dim;
  if (!is_batched) {
    if (inplace) {
      input = input.unsqueeze_(0); // 若不是批处理维度，且inplace为true，则添加一个维度
    } else {
      input = input.unsqueeze(0); // 若不是批处理维度，且inplace为false，则创建一个添加维度后的新Tensor
    }
  }

  Tensor result;
  if (inplace) {
    result = torch::feature_dropout_(input, p, training); // 若inplace为true，则原地进行多维度dropout操作
  } else {
    result = torch::feature_dropout(input, p, training); // 若inplace为false，则创建新的多维度dropout后的Tensor
  }

  if (!is_batched) {
    if (inplace) {
      result = result.squeeze_(0); // 若不是批处理维度，且inplace为true，则去掉添加的维度
    } else {
      result = result.squeeze(0); // 若不是批处理维度，且inplace为false，则创建一个去掉维度后的新Tensor
    }
  }
  return result;
}

// 实现2D dropout的具体功能
inline Tensor dropout2d(Tensor input, double p, bool training, bool inplace) {
  return _dropoutNd_helper<3, 4>(
      std::move(input), p, training, inplace, "dropout2d");
}

} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.dropout2d
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::Dropout2dFuncOptions`
/// class to learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::dropout2d(input, F::Dropout2dFuncOptions().p(0.5));
/// ```
// dropout2d函数的外部接口，调用detail命名空间中的dropout2d函数实现
inline Tensor dropout2d(
    Tensor input,
    const Dropout2dFuncOptions& options = {}) {
  return detail::dropout2d(
      std::move(input), options.p(), options.training(), options.inplace());
}
/// 定义了一个内部命名空间 detail，用于实现不同类型的 dropout 函数。

/// dropout3d 函数实现了对输入 Tensor 进行 3D Dropout 操作。
/// 参数 p 控制 dropout 的概率，training 表示是否处于训练模式，inplace 表示是否原地操作。
/// 该函数调用了 _dropoutNd_helper<4, 5> 辅助函数来执行具体的操作。
/// 参考文档 https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.dropout3d 了解更多细节。
/// 参考文档 torch::nn::functional::Dropout3dFuncOptions 查看该函数支持的可选参数。
/// 示例用法：
/// ```
/// namespace F = torch::nn::functional;
/// F::dropout3d(input, F::Dropout3dFuncOptions().p(0.5));
/// ```
inline Tensor dropout3d(
    Tensor input,
    const Dropout3dFuncOptions& options = {}) {
  return detail::dropout3d(
      std::move(input), options.p(), options.training(), options.inplace());
}

/// 定义了一个内部命名空间 detail，用于实现不同类型的 dropout 函数。

/// alpha_dropout 函数实现了对输入 Tensor 进行 alpha dropout 操作。
/// 参数 p 控制 dropout 的概率，training 表示是否处于训练模式，inplace 表示是否原地操作。
/// 如果 p 不在 [0, 1] 的范围内，将抛出错误。
/// 通过 torch::alpha_dropout_ 或 torch::alpha_dropout 函数执行具体操作，取决于 inplace 参数。
/// 参考文档 https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.alpha_dropout 了解更多细节。
/// 参考文档 torch::nn::functional::AlphaDropoutFuncOptions 查看该函数支持的可选参数。
/// 示例用法：
/// ```
/// namespace F = torch::nn::functional;
/// F::alpha_dropout(input, F::AlphaDropoutFuncOptions().p(0.5).training(false));
/// ```
inline Tensor alpha_dropout(
    Tensor input,
    const AlphaDropoutFuncOptions& options = {}) {
  return detail::alpha_dropout(
      std::move(input), options.p(), options.training(), options.inplace());
}

/// 定义了一个内部命名空间 detail，用于实现不同类型的 dropout 函数。

/// feature_alpha_dropout 函数实现了对输入 Tensor 进行 feature alpha dropout 操作。
/// 参数 p 控制 dropout 的概率，training 表示是否处于训练模式，inplace 表示是否原地操作。
/// 如果 p 不在 [0, 1] 的范围内，将抛出错误。
/// 通过 torch::feature_alpha_dropout_ 或 torch::feature_alpha_dropout 函数执行具体操作，取决于 inplace 参数。
/// 参考文档 https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.feature_alpha_dropout 了解更多细节。
/// 由于该注释未完成，故未提供更多详细信息。
/// `torch::nn::functional::FeatureAlphaDropoutFuncOptions` class to learn what
/// optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::feature_alpha_dropout(input,
/// F::FeatureAlphaDropoutFuncOptions().p(0.5).training(false));
/// ```
inline Tensor feature_alpha_dropout(
    Tensor input,
    const FeatureAlphaDropoutFuncOptions& options = {}) {
  // 调用底层实现函数 `detail::feature_alpha_dropout`，传入输入张量、概率、训练标志和就地标志
  return detail::feature_alpha_dropout(
      std::move(input), options.p(), options.training(), options.inplace());
}

} // namespace functional
} // namespace nn
} // namespace torch
```