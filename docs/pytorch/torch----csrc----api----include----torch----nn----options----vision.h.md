# `.\pytorch\torch\csrc\api\include\torch\nn\options\vision.h`

```py
#pragma once

// `#pragma once` 指令确保头文件只被编译一次，用于防止多重包含


#include <torch/arg.h>
#include <torch/csrc/Export.h>
#include <torch/enum.h>
#include <torch/types.h>

// 包含了一些 Torch 库中定义的头文件，用于本头文件中需要的类型和枚举定义


namespace torch {
namespace nn {
namespace functional {

// 声明了命名空间 `torch::nn::functional`，用于包含 Torch 中的神经网络相关的功能函数


/// Options for `torch::nn::functional::grid_sample`.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::grid_sample(input, grid,
/// F::GridSampleFuncOptions().mode(torch::kBilinear).padding_mode(torch::kZeros).align_corners(true));
/// ```py

// 对 `torch::nn::functional::grid_sample` 的选项进行了文档说明和示例，展示了如何使用这些选项


struct TORCH_API GridSampleFuncOptions {
  typedef std::variant<enumtype::kBilinear, enumtype::kNearest> mode_t;
  typedef std::
      variant<enumtype::kZeros, enumtype::kBorder, enumtype::kReflection>
          padding_mode_t;

// 定义了 `GridSampleFuncOptions` 结构体，包含了两个 typedef 定义的枚举变体，用于存储插值模式和填充模式的选项


  /// interpolation mode to calculate output values. Default: Bilinear
  TORCH_ARG(mode_t, mode) = torch::kBilinear;
  /// padding mode for outside grid values. Default: Zeros
  TORCH_ARG(padding_mode_t, padding_mode) = torch::kZeros;
  /// Specifies perspective to pixel as point. Default: false
  TORCH_ARG(std::optional<bool>, align_corners) = c10::nullopt;
};

// 定义了三个成员变量 `mode`、`padding_mode` 和 `align_corners`，分别用于设置插值模式、填充模式和是否对齐角点的选项


} // namespace functional
} // namespace nn
} // namespace torch

// 结束了命名空间 `torch::nn::functional`、`torch::nn` 和 `torch`
```