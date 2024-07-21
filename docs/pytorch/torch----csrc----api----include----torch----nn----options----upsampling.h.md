# `.\pytorch\torch\csrc\api\include\torch\nn\options\upsampling.h`

```py
#pragma once

#include <torch/arg.h>  // 包含 Torch 库中的参数定义
#include <torch/csrc/Export.h>  // 包含 Torch 库中的导出定义
#include <torch/enum.h>  // 包含 Torch 库中的枚举定义
#include <torch/expanding_array.h>  // 包含 Torch 库中的扩展数组定义
#include <torch/types.h>  // 包含 Torch 库中的类型定义

#include <vector>  // 包含标准库中的向量容器定义

namespace torch {
namespace nn {

/// Options for the `Upsample` module.
///
/// Example:
/// ```
/// Upsample
/// model(UpsampleOptions().scale_factor(std::vector<double>({3})).mode(torch::kLinear).align_corners(false));
/// ```py
struct TORCH_API UpsampleOptions {
  /// output spatial sizes.
  TORCH_ARG(std::optional<std::vector<int64_t>>, size) = c10::nullopt;  // 可选的输出空间大小

  /// multiplier for spatial size.
  TORCH_ARG(std::optional<std::vector<double>>, scale_factor) = c10::nullopt;  // 可选的空间尺寸乘数

  /// the upsampling algorithm: one of "nearest", "linear", "bilinear",
  /// "bicubic" and "trilinear". Default: "nearest"
  typedef std::variant<
      enumtype::kNearest,
      enumtype::kLinear,
      enumtype::kBilinear,
      enumtype::kBicubic,
      enumtype::kTrilinear>
      mode_t;
  TORCH_ARG(mode_t, mode) = torch::kNearest;  // 上采样算法，默认为最近邻

  /// if "True", the corner pixels of the input and output tensors are
  /// aligned, and thus preserving the values at those pixels. This only has
  /// effect when :attr:`mode` is "linear", "bilinear", "bicubic", or
  /// "trilinear". Default: "False"
  TORCH_ARG(std::optional<bool>, align_corners) = c10::nullopt;  // 可选的对齐角落像素
};

namespace functional {

/// Options for `torch::nn::functional::interpolate`.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::interpolate(input,
/// F::InterpolateFuncOptions().size(std::vector<int64_t>({4})).mode(torch::kNearest));
/// ```py
struct TORCH_API InterpolateFuncOptions {
  typedef std::variant<
      enumtype::kNearest,                        // 定义枚举类型 `mode_t`，包含插值方法的不同选项
      enumtype::kLinear,
      enumtype::kBilinear,
      enumtype::kBicubic,
      enumtype::kTrilinear,
      enumtype::kArea,
      enumtype::kNearestExact>
      mode_t;                                    // 声明 `mode_t` 类型用于存储插值方法

  /// output spatial sizes.
  TORCH_ARG(std::optional<std::vector<int64_t>>, size) = c10::nullopt;  // 输出空间大小的可选参数，初始值为空

  /// multiplier for spatial size.
  TORCH_ARG(std::optional<std::vector<double>>, scale_factor) = c10::nullopt;  // 空间大小的乘数，可选参数，初始值为空

  /// the upsampling algorithm: one of "nearest", "linear", "bilinear",
  /// "bicubic", "trilinear", "area", "nearest-exact". Default: "nearest"
  TORCH_ARG(mode_t, mode) = torch::kNearest;     // 上采样算法：可选值包括"nearest", "linear", "bilinear"等，默认为"nearest"

  /// Geometrically, we consider the pixels of the input and output as squares
  /// rather than points. If set to "True", the input and output tensors are
  /// aligned by the center points of their corner pixels, preserving the values
  /// at the corner pixels. If set to "False", the input and output tensors
  /// are aligned by the corner points of their corner pixels, and the
  /// interpolation uses edge value padding for out-of-boundary values, making
  /// this operation *independent* of input size when `scale_factor` is
  /// kept the same.  It is *required* when interpolating mode is "linear",
  /// "bilinear", "bicubic" or "trilinear". Default: "False"
  TORCH_ARG(std::optional<bool>, align_corners) = c10::nullopt;  // 几何上，将输入和输出的像素视为方块而不是点，对齐方式的可选参数，初始值为空

  /// recompute the scale_factor for use in the
  /// interpolation calculation.  When `scale_factor` is passed as a parameter,
  /// it is used to compute the `output_size`.  If `recompute_scale_factor` is
  /// `true` or not specified, a new `scale_factor` will be computed based on
  /// the output and input sizes for use in the interpolation computation (i.e.
  /// the computation will be identical to if the computed `output_size` were
  /// passed-in explicitly).  Otherwise, the passed-in `scale_factor` will be
  /// used in the interpolation computation.  Note that when `scale_factor` is
  /// floating-point, the recomputed scale_factor may differ from the one passed
  /// in due to rounding and precision issues.
  TORCH_ARG(std::optional<bool>, recompute_scale_factor) = c10::nullopt;  // 重新计算 `scale_factor` 以供插值计算使用的可选参数，初始值为空

  /// flag to apply anti-aliasing. Using anti-alias
  /// option together with :attr:`align_corners` equals "False", interpolation
  /// result would match Pillow result for downsampling operation. Supported
  /// modes: "bilinear". Default: "False".
  TORCH_ARG(bool, antialias) = false;             // 应用抗锯齿的标志，支持 "bilinear" 模式，默认为 false
};

} // namespace functional

} // namespace nn
} // namespace torch
```