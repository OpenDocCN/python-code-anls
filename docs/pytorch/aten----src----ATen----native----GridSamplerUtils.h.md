# `.\pytorch\aten\src\ATen\native\GridSamplerUtils.h`

```py
#pragma once

// See NOTE: [Tensor vs. TensorBase]
// https://github.com/pytorch/pytorch/pull/66979
#include <ATen/core/TensorBase.h>
#include <ATen/native/TensorProperties.h>
#include <ATen/native/CanUse32BitIndexMath.h>

namespace at::native {

namespace detail {

enum class GridSamplerInterpolation {Bilinear, Nearest, Bicubic};
enum class GridSamplerPadding {Zeros, Border, Reflection};

} // namespace detail

using detail::GridSamplerInterpolation;
using detail::GridSamplerPadding;

// See NOTE [ grid_sampler Native Functions ].
// 检查通用的网格采样器条件
inline void check_grid_sampler_common(
  const TensorBase& input,
  const TensorBase& grid
) {
  auto input_opt = input.options();
  auto grid_opt = grid.options();

  TORCH_CHECK(
    input.defined(),
    "grid_sampler(): expected input to not be undefined");
  TORCH_CHECK(
    grid.defined(),
    "grid_sampler(): expected grid to not be undefined");
  TORCH_CHECK(
    input_opt.device() == grid_opt.device(),
    "grid_sampler(): expected input and grid to be on same device, but input "
    "is on ", input_opt.device(), " and grid is on ", grid_opt.device());
  TORCH_CHECK(
    input_opt.layout() == kStrided && grid_opt.layout() == kStrided,
    "grid_sampler(): expected input and grid to have torch.strided layout, but "
    "input has ", input_opt.layout(), " and grid has ", grid_opt.layout());
  TORCH_CHECK(
    input.size(0) == grid.size(0),
    "grid_sampler(): expected grid and input to have same batch size, but got "
    "input with sizes ", input.sizes(), " and grid with sizes ", grid.sizes());
  TORCH_CHECK(
    grid.size(-1) == input.dim() - 2,
    "grid_sampler(): expected grid to have size ", input.dim() - 2, " in last "
    "dimension, but got grid with sizes ", grid.sizes());

  for (const auto i : c10::irange(2, input.dim())) {
    TORCH_CHECK(input.size(i) > 0,
      "grid_sampler(): expected input to have non-empty spatial dimensions, "
      "but input has sizes ", input.sizes(), " with dimension ", i, " being "
      "empty");
  }
}

// See NOTE [ grid_sampler Native Functions ].
// 检查2D网格采样器条件
inline void check_grid_sampler_2d(
  const TensorBase& input,
  const TensorBase& grid
) {
  TORCH_CHECK(
    input.dim() == 4 && input.dim() == grid.dim(),
    "grid_sampler(): expected 4D input and grid with same number of "
    "dimensions, but got input with sizes ", input.sizes(),
    " and grid with sizes ", grid.sizes());
}

// See NOTE [ grid_sampler Native Functions ].
// 检查3D网格采样器条件
inline void check_grid_sampler_3d(
  const TensorBase& input,
  const TensorBase& grid,
  int64_t interpolation_mode
) {
  TORCH_CHECK(
    input.dim() == 5 && input.dim() == grid.dim(),
    "grid_sampler(): expected 5D input and grid with same number of "
    "dimensions, but got input with sizes ", input.sizes(),
    " and grid with sizes ", grid.sizes());
  TORCH_CHECK(
    !(input.dim() == 5 &&
      static_cast<GridSamplerInterpolation>(interpolation_mode) ==
        GridSamplerInterpolation::Bicubic),
    "grid_sampler(): bicubic interpolation for 3D input is not supported");
}


注释：

// 根据注释链接的说明，此处定义了与张量相关的基本操作和属性检查函数
// 包含了必要的头文件来支持这些操作和检查
#include <ATen/core/TensorBase.h>
#include <ATen/native/TensorProperties.h>
#include <ATen/native/CanUse32BitIndexMath.h>

// 进入 ATen 命名空间下的 native 命名空间
namespace at::native {

// 进入 detail 命名空间，定义了网格采样器的插值和填充方式
namespace detail {

// 定义了网格采样器的插值方式枚举
enum class GridSamplerInterpolation {Bilinear, Nearest, Bicubic};
// 定义了网格采样器的填充方式枚举
enum class GridSamplerPadding {Zeros, Border, Reflection};

} // namespace detail

// 使用 detail 命名空间下的 GridSamplerInterpolation
using detail::GridSamplerInterpolation;
// 使用 detail 命名空间下的 GridSamplerPadding
using detail::GridSamplerPadding;

// 定义了检查通用网格采样器条件的函数
// 根据输入和网格张量的属性进行检查，确保它们满足预期的条件
inline void check_grid_sampler_common(
  const TensorBase& input,
  const TensorBase& grid
) {
  auto input_opt = input.options();
  auto grid_opt = grid.options();

  // 检查输入张量是否已定义
  TORCH_CHECK(
    input.defined(),
    "grid_sampler(): expected input to not be undefined");
  // 检查网格张量是否已定义
  TORCH_CHECK(
    grid.defined(),
    "grid_sampler(): expected grid to not be undefined");
  // 检查输入和网格是否在同一设备上
  TORCH_CHECK(
    input_opt.device() == grid_opt.device(),
    "grid_sampler(): expected input and grid to be on same device, but input "
    "is on ", input_opt.device(), " and grid is on ", grid_opt.device());
  // 检查输入和网格是否具有 torch.strided 布局
  TORCH_CHECK(
    input_opt.layout() == kStrided && grid_opt.layout() == kStrided,
    "grid_sampler(): expected input and grid to have torch.strided layout, but "
    "input has ", input_opt.layout(), " and grid has ", grid_opt.layout());
  // 检查输入和网格的批处理大小是否相同
  TORCH_CHECK(
    input.size(0) == grid.size(0),
    "grid_sampler(): expected grid and input to have same batch size, but got "
    "input with sizes ", input.sizes(), " and grid with sizes ", grid.sizes());
  // 检查网格在最后一个维度上的尺寸是否正确
  TORCH_CHECK(
    grid.size(-1) == input.dim() - 2,
    "grid_sampler(): expected grid to have size ", input.dim() - 2, " in last "
    "dimension, but got grid with sizes ", grid.sizes());

  // 循环检查输入张量的空间维度是否非空
  for (const auto i : c10::irange(2, input.dim())) {
    TORCH_CHECK(input.size(i) > 0,
      "grid_sampler(): expected input to have non-empty spatial dimensions, "
      "but input has sizes ", input.sizes(), " with dimension ", i, " being "
      "empty");
  }
}

// 定义了检查2D网格采样器条件的函数
// 检查输入和网格张量是否是4D，并且具有相同的维度
inline void check_grid_sampler_2d(
  const TensorBase& input,
  const TensorBase& grid
) {
  TORCH_CHECK(
    input.dim() == 4 && input.dim() == grid.dim(),
    "grid_sampler(): expected 4D input and grid with same number of "
    "dimensions, but got input with sizes ", input.sizes(),
    " and grid with sizes ", grid.sizes());
}

// 定义了检查3D网格采样器条件的函数
// 检查输入和网格张量是否是5D，并且具有相同的维度
// 同时检查插值模式是否支持 bicubic 插值
inline void check_grid_sampler_3d(
  const TensorBase& input,
  const TensorBase& grid,
  int64_t interpolation_mode
) {
  TORCH_CHECK(
    input.dim() == 5 && input.dim() == grid.dim(),
    "grid_sampler(): expected 5D input and grid with same number of "
    "dimensions, but got input with sizes ", input.sizes(),
    " and grid with sizes ", grid.sizes());
  TORCH_CHECK(
    !(input.dim() == 5 &&
      static_cast<GridSamplerInterpolation>(interpolation_mode) ==
        GridSamplerInterpolation::Bicubic),
    "grid_sampler(): bicubic interpolation for 3D input is not supported");
}
    # 抛出异常，指示 grid_sampler 函数仅支持 4D 输入
    "grid_sampler(): bicubic interpolation only supports 4D input");
}

// 这里是 grid_sampler 原生函数的相关说明，见 NOTE [ grid_sampler Native Functions ]。
// cudnn 不支持输入大于 1024 的情况。
inline bool cond_cudnn_grid_sampler(
  const TensorBase& input,
  const TensorBase& grid
) {
  // 检查输入是否满足 cudnn 的可接受条件
  return (
    at::native::cudnn_is_acceptable(input) &&
    // 检查 grid 是否满足 cudnn 的可接受条件
    at::native::cudnn_is_acceptable(grid) &&
    // 检查是否可以使用32位索引运算处理 input
    at::native::canUse32BitIndexMath(input) &&
    // 检查是否可以使用32位索引运算处理 grid
    at::native::canUse32BitIndexMath(grid) &&
    // 检查 input 是否为四维张量
    input.dim() == 4 &&
    // 检查 input 的第二维大小是否不超过 1024
    input.sym_size(1) <= 1024);
}

} // namespace at::native
```