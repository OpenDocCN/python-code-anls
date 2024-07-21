# `.\pytorch\torch\csrc\api\include\torch\nn\functional\vision.h`

```
#pragma once

#include <torch/nn/options/vision.h>
#include <torch/types.h>

namespace torch {
namespace nn {
namespace functional {

/// 生成仿射网格
/// 
/// 根据给定的仿射变换参数生成一个仿射网格，用于空间变换。
/// 
/// \param theta 仿射变换参数张量，要求是浮点类型
/// \param size 输出网格的大小，可以是4维或5维
/// \param align_corners 是否对齐角点，通常用于插值操作
/// \return 生成的仿射网格张量
inline Tensor affine_grid(
    const Tensor& theta,
    const IntArrayRef& size,
    bool align_corners = false) {
  // 强制 theta 张量使用浮点数类型
  TORCH_CHECK(
      theta.is_floating_point(),
      "Expected theta to have floating point type, but got ",
      theta.dtype());

  // 检查尺寸和形状是否匹配
  if (size.size() == 4) {
    TORCH_CHECK(
        theta.dim() == 3 && theta.size(-2) == 2 && theta.size(-1) == 3,
        "Expected a batch of 2D affine matrices of shape Nx2x3 for size ",
        size,
        ". Got ",
        theta.sizes(),
        ".");
  } else if (size.size() == 5) {
    TORCH_CHECK(
        theta.dim() == 3 && theta.size(-2) == 3 && theta.size(-1) == 4,
        "Expected a batch of 3D affine matrices of shape Nx3x4 for size ",
        size,
        ". Got ",
        theta.sizes(),
        ".");
  } else {
    TORCH_CHECK(
        false,
        "affine_grid only supports 4D and 5D sizes, ",
        "for 2D and 3D affine transforms, respectively. ",
        "Got size ",
        size);
  }

  // 检查输出大小是否非零正数
  if (*std::min_element(size.begin(), size.end()) <= 0) {
    TORCH_CHECK(false, "Expected non-zero, positive output size. Got ", size);
  }

  // 调用 PyTorch 内部的仿射网格生成器
  return torch::affine_grid_generator(theta, size, align_corners);
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {

/// 网格采样
///
/// 对输入张量进行网格采样，根据给定的采样网格和模式进行操作。
/// 
/// \param input 输入张量，要进行采样的数据
/// \param grid 采样网格张量，描述了采样的位置
/// \param mode 采样模式，可以是双线性、最近邻或双三次插值
/// \param padding_mode 填充模式，可以是零填充、边界填充或反射填充
/// \param align_corners 是否对齐角点，通常用于插值操作
/// \return 采样后的张量
inline Tensor grid_sample(
    const Tensor& input,
    const Tensor& grid,
    GridSampleFuncOptions::mode_t mode,
    GridSampleFuncOptions::padding_mode_t padding_mode,
    std::optional<bool> align_corners) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int64_t mode_enum, padding_mode_enum;

  // 根据模式类型设置枚举值
  if (std::holds_alternative<enumtype::kBilinear>(mode)) {
    mode_enum = 0;
  } else if (std::holds_alternative<enumtype::kNearest>(mode)) {
    mode_enum = 1;
  } else { /// mode == 'bicubic'
    mode_enum = 2;
  }

  // 根据填充模式类型设置枚举值
  if (std::holds_alternative<enumtype::kZeros>(padding_mode)) {
    padding_mode_enum = 0;
  } else if (std::holds_alternative<enumtype::kBorder>(padding_mode)) {
    padding_mode_enum = 1;
  } else { /// padding_mode == 'reflection'
    padding_mode_enum = 2;
  }

  // 如果未指定对齐角点的选项，默认给出警告并设置为 False
  if (!align_corners.has_value()) {
    TORCH_WARN(
        "Default grid_sample and affine_grid behavior has changed ",
        "to align_corners=False since 1.3.0. Please specify ",
        "align_corners=True if the old behavior is desired. ",
        "See the documentation of grid_sample for details.");
    align_corners = false;
  }

  // 调用 PyTorch 内部的网格采样函数
  return torch::grid_sampler(
      input, grid, mode_enum, padding_mode_enum, align_corners.value());
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// 查看此功能的详细行为文档
///
/// 详细了解该函数在功能上的准确行为，可以参考以下链接：
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.grid_sample
///
/// 引入命名空间别名 F，使得调用时更简洁，避免重复输入长的命名空间路径。
/// 示例用法展示了如何使用 `torch::nn::functional::GridSampleFuncOptions` 类，
/// 学习如何使用此功能的可选参数。
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::grid_sample(input, grid,
///                F::GridSampleFuncOptions().mode(torch::kBilinear).padding_mode(torch::kZeros).align_corners(true));
/// ```
inline Tensor grid_sample(
    const Tensor& input,
    const Tensor& grid,
    const GridSampleFuncOptions& options = {}) {
  // 调用 detail 命名空间中的 grid_sample 函数，传递输入张量、网格张量以及选项中的参数
  return detail::grid_sample(
      input,
      grid,
      options.mode(),
      options.padding_mode(),
      options.align_corners());
}

} // namespace functional
} // namespace nn
} // namespace torch
```