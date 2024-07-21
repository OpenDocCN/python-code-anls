# `.\pytorch\torch\csrc\api\include\torch\nn\functional\fold.h`

```
#pragma once

#include <torch/nn/options/fold.h>

namespace torch {
namespace nn {
namespace functional {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
// 定义 fold 函数，根据输入张量和参数进行折叠操作
inline Tensor fold(
    const Tensor& input,                  // 输入张量
    ExpandingArray<2> output_size,        // 输出大小
    ExpandingArray<2> kernel_size,        // 卷积核大小
    ExpandingArray<2> dilation,           // 膨胀率
    ExpandingArray<2> padding,            // 填充
    ExpandingArray<2> stride) {           // 步幅
  // 如果输入张量是 3D 或者 2D，则调用 col2im 函数
  if (input.dim() == 3 || input.dim() == 2) {
    return torch::col2im(
        input, output_size, kernel_size, dilation, padding, stride);
  } else {
    // 否则，抛出错误，只支持未批处理的（2D）或批处理的（3D）输入张量
    TORCH_CHECK(
        false,
        "Input Error: Only unbatched (2D) or batched (3D) input Tensors are supported "
        "(got ",
        input.dim(),
        "D)");
  }
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.fold
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::FoldFuncOptions` class to
/// learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::fold(input, F::FoldFuncOptions({3, 2}, {2, 2}));
/// ```
// 折叠操作的入口函数，根据给定的选项对输入张量进行折叠
inline Tensor fold(const Tensor& input, const FoldFuncOptions& options) {
  return detail::fold(
      input,
      options.output_size(),     // 获取输出大小
      options.kernel_size(),     // 获取卷积核大小
      options.dilation(),        // 获取膨胀率
      options.padding(),         // 获取填充
      options.stride());         // 获取步幅
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
// 定义 unfold 函数，根据输入张量和参数进行展开操作
inline Tensor unfold(
    const Tensor& input,              // 输入张量
    ExpandingArray<2> kernel_size,    // 卷积核大小
    ExpandingArray<2> dilation,       // 膨胀率
    ExpandingArray<2> padding,        // 填充
    ExpandingArray<2> stride) {       // 步幅
  // 如果输入张量是 4D，则调用 im2col 函数
  if (input.dim() == 4) {
    return torch::im2col(input, kernel_size, dilation, padding, stride);
  } else {
    // 否则，抛出错误，只支持 4D 输入张量
    TORCH_CHECK(
        false,
        "Input Error: Only 4D input Tensors are supported "
        "(got ",
        input.dim(),
        "D)");
  }
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.unfold
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::UnfoldFuncOptions` class
/// to learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::unfold(input, F::UnfoldFuncOptions({2, 2}).padding(1).stride(2));
/// ```
// 展开操作的入口函数，根据给定的选项对输入张量进行展开
inline Tensor unfold(const Tensor& input, const UnfoldFuncOptions& options) {
  return detail::unfold(
      input,
      options.kernel_size(),    // 获取卷积核大小
      options.dilation(),       // 获取膨胀率
      options.padding(),        // 获取填充
      options.stride());        // 获取步幅
}

} // namespace functional
} // namespace nn
} // namespace torch
```