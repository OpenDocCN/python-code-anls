# `.\pytorch\torch\csrc\api\include\torch\nn\functional\padding.h`

```py
#pragma once

#include <ATen/PadNd.h>  // 导入 ATen 的 PadNd.h 文件，用于张量的填充操作
#include <torch/nn/options/padding.h>  // 导入 torch.nn.options.padding.h 文件，定义了填充操作的选项类

namespace torch {
namespace nn {
namespace functional {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor pad(
    const Tensor& input,
    IntArrayRef pad,
    PadFuncOptions::mode_t mode,
    double value) {
  // 根据 mode 参数选择填充模式的枚举值
  const auto mode_enum = [&] {
    if (std::holds_alternative<enumtype::kConstant>(mode)) {
      return at::padding_mode::constant;
    } else if (std::holds_alternative<enumtype::kReflect>(mode)) {
      return at::padding_mode::reflect;
    } else if (std::holds_alternative<enumtype::kReplicate>(mode)) {
      return at::padding_mode::replicate;
    } else if (std::holds_alternative<enumtype::kCircular>(mode)) {
      return at::padding_mode::circular;
    }
    // 如果 mode 参数未知，则抛出异常
    TORCH_CHECK(false, "Unrecognised padding mode");
  }();

  // 如果 value 参数不为零，则使用填充值
  std::optional<double> fill_value;
  if (value != 0.0) {
    fill_value = value;
  }
  // 调用 ATen 的 _pad_enum 函数进行张量填充，并返回结果张量
  return at::_pad_enum(input, pad, static_cast<int64_t>(mode_enum), fill_value);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.pad
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::PadFuncOptions` class to
/// learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::pad(input, F::PadFuncOptions({1, 2, 2, 1, 1,
/// 2}).mode(torch::kReplicate));
/// ```py
// 对输入张量进行填充操作，根据给定的选项
inline Tensor pad(const Tensor& input, const PadFuncOptions& options) {
  // 调用 detail 命名空间下的 pad 函数，传入填充选项，并返回结果张量
  return detail::pad(input, options.pad(), options.mode(), options.value());
}

} // namespace functional
} // namespace nn
} // namespace torch
```