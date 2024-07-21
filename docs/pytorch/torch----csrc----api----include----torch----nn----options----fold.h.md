# `.\pytorch\torch\csrc\api\include\torch\nn\options\fold.h`

```py
#pragma once

#include <torch/arg.h>  // 引入 Torch 库中的参数处理模块
#include <torch/csrc/Export.h>  // 引入 Torch 库中的导出声明
#include <torch/expanding_array.h>  // 引入 Torch 库中的扩展数组处理模块
#include <torch/types.h>  // 引入 Torch 库中的类型定义

namespace torch {
namespace nn {

/// Options for the `Fold` module.
///
/// Example:
/// ```
/// Fold model(FoldOptions({8, 8}, {3, 3}).dilation(2).padding({2,
/// 1}).stride(2));
/// ```py
struct TORCH_API FoldOptions {
  FoldOptions(ExpandingArray<2> output_size, ExpandingArray<2> kernel_size)
      : output_size_(std::move(output_size)),
        kernel_size_(std::move(kernel_size)) {}

  /// describes the spatial shape of the large containing tensor of the sliding
  /// local blocks. It is useful to resolve the ambiguity when multiple input
  /// shapes map to same number of sliding blocks, e.g., with stride > 0.
  TORCH_ARG(ExpandingArray<2>, output_size);  // 定义输出张量的空间形状

  /// the size of the sliding blocks
  TORCH_ARG(ExpandingArray<2>, kernel_size);  // 定义滑动块的大小

  /// controls the spacing between the kernel points; also known as the à trous
  /// algorithm.
  TORCH_ARG(ExpandingArray<2>, dilation) = 1;  // 控制内核点之间的间距，默认为 1

  /// controls the amount of implicit zero-paddings on both sides for padding
  /// number of points for each dimension before reshaping.
  TORCH_ARG(ExpandingArray<2>, padding) = 0;  // 控制在重新调整形状之前每个维度的隐式零填充数

  /// controls the stride for the sliding blocks.
  TORCH_ARG(ExpandingArray<2>, stride) = 1;  // 控制滑动块的步幅，默认为 1
};

namespace functional {
/// Options for `torch::nn::functional::fold`.
///
/// See the documentation for `torch::nn::FoldOptions` class to learn what
/// arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::fold(input, F::FoldFuncOptions({3, 2}, {2, 2}));
/// ```py
using FoldFuncOptions = FoldOptions;  // 使用 FoldOptions 作为 fold 函数的参数选项
} // namespace functional

// ============================================================================

/// Options for the `Unfold` module.
///
/// Example:
/// ```
/// Unfold model(UnfoldOptions({2, 4}).dilation(2).padding({2, 1}).stride(2));
/// ```py
struct TORCH_API UnfoldOptions {
  UnfoldOptions(ExpandingArray<2> kernel_size)
      : kernel_size_(std::move(kernel_size)) {}

  /// the size of the sliding blocks
  TORCH_ARG(ExpandingArray<2>, kernel_size);  // 定义滑动块的大小

  /// controls the spacing between the kernel points; also known as the à trous
  /// algorithm.
  TORCH_ARG(ExpandingArray<2>, dilation) = 1;  // 控制内核点之间的间距，默认为 1

  /// controls the amount of implicit zero-paddings on both sides for padding
  /// number of points for each dimension before reshaping.
  TORCH_ARG(ExpandingArray<2>, padding) = 0;  // 控制在重新调整形状之前每个维度的隐式零填充数

  /// controls the stride for the sliding blocks.
  TORCH_ARG(ExpandingArray<2>, stride) = 1;  // 控制滑动块的步幅，默认为 1
};

namespace functional {
/// Options for `torch::nn::functional::unfold`.
///
/// See the documentation for `torch::nn::UnfoldOptions` class to learn what
/// arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::unfold(input, F::UnfoldFuncOptions({2, 2}).padding(1).stride(2));
/// ```py
using UnfoldFuncOptions = UnfoldOptions;  // 使用 UnfoldOptions 作为 unfold 函数的参数选项
} // namespace functional

} // namespace nn
} // namespace torch
```