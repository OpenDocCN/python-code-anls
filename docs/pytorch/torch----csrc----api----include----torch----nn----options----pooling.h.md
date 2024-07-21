# `.\pytorch\torch\csrc\api\include\torch\nn\options\pooling.h`

```
#pragma once
// 预处理指令，确保本头文件只被编译一次

#include <torch/arg.h>
// 引入 Torch 库中的参数处理头文件

#include <torch/csrc/Export.h>
// 引入 Torch 库中的导出头文件

#include <torch/expanding_array.h>
// 引入 Torch 库中的扩展数组头文件

#include <torch/types.h>
// 引入 Torch 库中的类型定义头文件

namespace torch {
namespace nn {
// Torch 命名空间和 nn 命名空间开始

/// Options for a `D`-dimensional avgpool module.
// 用于 `D` 维度平均池化模块的选项结构体模板声明
template <size_t D>
struct AvgPoolOptions {
  AvgPoolOptions(ExpandingArray<D> kernel_size)
      : kernel_size_(kernel_size), stride_(kernel_size) {}
  // AvgPoolOptions 结构体构造函数，初始化 kernel_size 和 stride

  /// the size of the window to take an average over
  TORCH_ARG(ExpandingArray<D>, kernel_size);
  // 定义 kernel_size 参数，表示窗口大小，用于计算平均值

  /// the stride of the window. Default value is `kernel_size`
  TORCH_ARG(ExpandingArray<D>, stride);
  // 定义 stride 参数，表示窗口的步幅，默认值为 kernel_size

  /// implicit zero padding to be added on both sides
  TORCH_ARG(ExpandingArray<D>, padding) = 0;
  // 定义 padding 参数，表示在两侧隐式添加的零填充

  /// when True, will use `ceil` instead of `floor` to compute the output shape
  TORCH_ARG(bool, ceil_mode) = false;
  // 定义 ceil_mode 参数，若为 True，则使用 `ceil` 而非 `floor` 来计算输出形状

  /// when True, will include the zero-padding in the averaging calculation
  TORCH_ARG(bool, count_include_pad) = true;
  // 定义 count_include_pad 参数，若为 True，则在平均计算中包括零填充区域

  /// if specified, it will be used as divisor, otherwise size of the pooling
  /// region will be used.
  TORCH_ARG(std::optional<int64_t>, divisor_override) = c10::nullopt;
  // 定义 divisor_override 参数，若指定，则用作除数；否则使用池化区域的大小。
};

/// `AvgPoolOptions` specialized for the `AvgPool1d` module.
// 专门用于 `AvgPool1d` 模块的 `AvgPoolOptions` 特化声明
using AvgPool1dOptions = AvgPoolOptions<1>;

/// `AvgPoolOptions` specialized for the `AvgPool2d` module.
// 专门用于 `AvgPool2d` 模块的 `AvgPoolOptions` 特化声明
using AvgPool2dOptions = AvgPoolOptions<2>;

/// `AvgPoolOptions` specialized for the `AvgPool3d` module.
// 专门用于 `AvgPool3d` 模块的 `AvgPoolOptions` 特化声明
using AvgPool3dOptions = AvgPoolOptions<3>;

namespace functional {
/// Options for `torch::nn::functional::avg_pool1d`.
// 用于 `torch::nn::functional::avg_pool1d` 的选项结构体声明

/// See the documentation for `torch::nn::AvgPool1dOptions` class to learn what
/// arguments are supported.
// 参见 `torch::nn::AvgPool1dOptions` 类的文档，了解支持的参数

/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::avg_pool1d(x, F::AvgPool1dFuncOptions(3).stride(2));
/// ```
// 示例代码，展示如何使用 `avg_pool1d` 函数

using AvgPool1dFuncOptions = AvgPool1dOptions;
// 使用 `AvgPool1dOptions` 作为 `avg_pool1d` 函数的选项

} // namespace functional

namespace functional {
/// Options for `torch::nn::functional::avg_pool2d`.
// 用于 `torch::nn::functional::avg_pool2d` 的选项结构体声明

/// See the documentation for `torch::nn::AvgPool2dOptions` class to learn what
/// arguments are supported.
// 参见 `torch::nn::AvgPool2dOptions` 类的文档，了解支持的参数

/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::avg_pool2d(x, F::AvgPool2dFuncOptions(3).stride(2));
/// ```
// 示例代码，展示如何使用 `avg_pool2d` 函数

using AvgPool2dFuncOptions = AvgPool2dOptions;
// 使用 `AvgPool2dOptions` 作为 `avg_pool2d` 函数的选项

} // namespace functional

namespace functional {
/// Options for `torch::nn::functional::avg_pool3d`.
// 用于 `torch::nn::functional::avg_pool3d` 的选项结构体声明

/// See the documentation for `torch::nn::AvgPool3dOptions` class to learn what
/// arguments are supported.
// 参见 `torch::nn::AvgPool3dOptions` 类的文档，了解支持的参数

/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::avg_pool3d(x, F::AvgPool3dFuncOptions(3).stride(2));
/// ```
// 示例代码，展示如何使用 `avg_pool3d` 函数

using AvgPool3dFuncOptions = AvgPool3dOptions;
// 使用 `AvgPool3dOptions` 作为 `avg_pool3d` 函数的选项

} // namespace functional

// ============================================================================

/// Options for a `D`-dimensional maxpool module.
// 用于 `D` 维度最大池化模块的选项结构体声明
/// Options for a `D`-dimensional adaptive maxpool module.
template <typename output_size_t>
struct AdaptiveMaxPoolOptions {
  /// Constructor initializing the output size parameter.
  AdaptiveMaxPoolOptions(output_size_t output_size)
      : output_size_(output_size) {}

  /// The target output size for adaptive max pooling.
  TORCH_ARG(output_size_t, output_size);
};

/// `AdaptiveMaxPoolOptions` specialized for the `AdaptiveMaxPool1d` module.
///
/// Example:
/// ```
/// AdaptiveMaxPool1d model(AdaptiveMaxPool1dOptions(3));
/// ```
using AdaptiveMaxPool1dOptions = AdaptiveMaxPoolOptions<ExpandingArray<1>>;

/// `AdaptiveMaxPoolOptions` specialized for the `AdaptiveMaxPool2d` module.
///
/// Example:
/// ```
/// AdaptiveMaxPool2d model(AdaptiveMaxPool2dOptions({3, 2}));
/// ```
using AdaptiveMaxPool2dOptions = AdaptiveMaxPoolOptions<ExpandingArray<2>>;
/// Options for `torch::nn::functional::adaptive_avg_pool1d`.
///
/// See the documentation for `torch::nn::AdaptiveAvgPool1dOptions` class to
/// learn what arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::adaptive_avg_pool1d(x, F::AdaptiveAvgPool1dFuncOptions(3));
/// ```
using AdaptiveAvgPool1dFuncOptions = AdaptiveAvgPool1dOptions;
/// Options for a `D`-dimensional maxunpool functional.
template <size_t D>
struct MaxUnpoolOptions {
  /// Constructor initializes the options with specified kernel size and sets stride to match.
  MaxUnpoolOptions(ExpandingArray<D> kernel_size)
      : kernel_size_(kernel_size), stride_(kernel_size) {}

  /// The size of the window to take a max over.
  TORCH_ARG(ExpandingArray<D>, kernel_size);

  /// The stride of the window. Default value is `kernel_size`.
  TORCH_ARG(ExpandingArray<D>, stride);

  /// Implicit zero padding to be added on both sides. Default value is 0.
  TORCH_ARG(ExpandingArray<D>, padding) = 0;
};

/// `MaxUnpoolOptions` specialized for the `MaxUnpool1d` module.
///
/// Example:
/// ```
/// MaxUnpool1d model(MaxUnpool1dOptions(3).stride(2).padding(1));
/// ```
using MaxUnpool1dOptions = MaxUnpoolOptions<1>;

/// `MaxUnpoolOptions` specialized for the `MaxUnpool2d` module.
///
/// Example:
/// ```
/// MaxUnpool2d model(MaxUnpool2dOptions(3).stride(2).padding(1));
/// ```
using MaxUnpool2dOptions = MaxUnpoolOptions<2>;

/// `MaxUnpoolOptions` specialized for the `MaxUnpool3d` module.
///
/// Example:
/// ```
/// MaxUnpool3d model(MaxUnpool3dOptions(3).stride(2).padding(1));
/// ```
using MaxUnpool3dOptions = MaxUnpoolOptions<3>;
/// Options for configuring the behavior of the `MaxUnpoolFuncOptions` structure.
struct MaxUnpoolFuncOptions {
  // Constructor initializing `kernel_size_` and `stride_` with the given `kernel_size`.
  MaxUnpoolFuncOptions(ExpandingArray<D> kernel_size)
      : kernel_size_(kernel_size), stride_(kernel_size) {}

  /// The size of the window to take a max over.
  TORCH_ARG(ExpandingArray<D>, kernel_size);

  /// The stride of the window. Default value is `kernel_size`.
  TORCH_ARG(ExpandingArray<D>, stride);

  /// Implicit zero padding to be added on both sides.
  TORCH_ARG(ExpandingArray<D>, padding) = 0;

  /// The targeted output size.
  TORCH_ARG(std::optional<std::vector<int64_t>>, output_size) = c10::nullopt;
};

/// Specialization of `MaxUnpoolFuncOptions` for `torch::nn::functional::max_unpool1d`.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::max_unpool1d(x, indices, F::MaxUnpool1dFuncOptions(3).stride(2).padding(1));
/// ```
using MaxUnpool1dFuncOptions = MaxUnpoolFuncOptions<1>;

/// Specialization of `MaxUnpoolFuncOptions` for `torch::nn::functional::max_unpool2d`.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::max_unpool2d(x, indices, F::MaxUnpool2dFuncOptions(3).stride(2).padding(1));
/// ```
using MaxUnpool2dFuncOptions = MaxUnpoolFuncOptions<2>;

/// Specialization of `MaxUnpoolFuncOptions` for `torch::nn::functional::max_unpool3d`.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::max_unpool3d(x, indices, F::MaxUnpool3dFuncOptions(3));
/// ```
using MaxUnpool3dFuncOptions = MaxUnpoolFuncOptions<3>;

} // namespace functional

// ============================================================================

/// Options for a `D`-dimensional fractional maxpool module.
template <size_t D>
struct FractionalMaxPoolOptions {
  // Constructor initializing `kernel_size_` with the given `kernel_size`.
  FractionalMaxPoolOptions(ExpandingArray<D> kernel_size)
      : kernel_size_(kernel_size) {}

  /// The size of the window to take a max over.
  TORCH_ARG(ExpandingArray<D>, kernel_size);

  /// The target output size of the image.
  TORCH_ARG(std::optional<ExpandingArray<D>>, output_size) = c10::nullopt;

  /// If one wants to have an output size as a ratio of the input size, this
  /// option can be given. This has to be a number or tuple in the range (0, 1).
  using ExpandingArrayDouble = torch::ExpandingArray<D, double>;
  TORCH_ARG(std::optional<ExpandingArrayDouble>, output_ratio) = c10::nullopt;

  // A tensor for storing random samples.
  TORCH_ARG(torch::Tensor, _random_samples) = Tensor();
};

/// Specialization of `FractionalMaxPoolOptions` for the `FractionalMaxPool2d` module.
///
/// Example:
/// ```
/// FractionalMaxPool2d model(FractionalMaxPool2dOptions(5).output_size(1));
/// ```
using FractionalMaxPool2dOptions = FractionalMaxPoolOptions<2>;

/// Specialization of `FractionalMaxPoolOptions` for the `FractionalMaxPool3d` module.
///
/// Example:
/// ```
/// FractionalMaxPool3d model(FractionalMaxPool3dOptions(5).output_size(1));
/// ```
using FractionalMaxPool3dOptions = FractionalMaxPoolOptions<3>;

namespace functional {
/// Options for `torch::nn::functional::fractional_max_pool2d` and
/// `torch::nn::functional::fractional_max_pool2d_with_indices`
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::fractional_max_pool2d(x,
/// F::FractionalMaxPool2dFuncOptions(3).output_size(2));
/// ```
using FractionalMaxPool2dFuncOptions = FractionalMaxPool2dOptions;
} // namespace functional

namespace functional {
/// Options for `torch::nn::functional::fractional_max_pool3d` and
/// `torch::nn::functional::fractional_max_pool3d_with_indices`
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::fractional_max_pool3d(x,
/// F::FractionalMaxPool3dFuncOptions(3).output_size(2));
/// ```
using FractionalMaxPool3dFuncOptions = FractionalMaxPool3dOptions;
} // namespace functional

// ============================================================================

/// Options for a `D`-dimensional lppool module.
template <size_t D>
struct LPPoolOptions {
  LPPoolOptions(double norm_type, ExpandingArray<D> kernel_size)
      : norm_type_(norm_type),
        kernel_size_(kernel_size),
        stride_(kernel_size) {}

  TORCH_ARG(double, norm_type);

  // the size of the window to take an average over
  TORCH_ARG(ExpandingArray<D>, kernel_size);

  // the stride of the window. Default value is `kernel_size`
  TORCH_ARG(ExpandingArray<D>, stride);

  // when True, will use `ceil` instead of `floor` to compute the output shape
  TORCH_ARG(bool, ceil_mode) = false;
};

/// `LPPoolOptions` specialized for the `LPPool1d` module.
///
/// Example:
/// ```
/// LPPool1d model(LPPool1dOptions(1, 2).stride(5).ceil_mode(true));
/// ```
using LPPool1dOptions = LPPoolOptions<1>;

/// `LPPoolOptions` specialized for the `LPPool2d` module.
///
/// Example:
/// ```
/// LPPool2d model(LPPool2dOptions(1, std::vector<int64_t>({3, 4})).stride({5,
/// 6}).ceil_mode(true));
/// ```
using LPPool2dOptions = LPPoolOptions<2>;

/// `LPPoolOptions` specialized for the `LPPool3d` module.
///
/// Example:
/// ```
/// LPPool3d model(LPPool3dOptions(1, std::vector<int64_t>({3, 4, 5})).stride(
/// {5, 6, 7}).ceil_mode(true));
/// ```
using LPPool3dOptions = LPPoolOptions<3>;

namespace functional {
/// Options for `torch::nn::functional::lp_pool1d`.
///
/// See the documentation for `torch::nn::LPPool1dOptions` class to learn what
/// arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::lp_pool1d(x, F::LPPool1dFuncOptions(2, 3).stride(2));
/// ```
using LPPool1dFuncOptions = LPPool1dOptions;
} // namespace functional

namespace functional {
/// Options for `torch::nn::functional::lp_pool2d`.
///
/// See the documentation for `torch::nn::LPPool2dOptions` class to learn what
/// arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::lp_pool2d(x, F::LPPool2dFuncOptions(2, {2, 3}).stride(2));
/// ```
using LPPool2dFuncOptions = LPPool2dOptions;
} // namespace functional

namespace functional {
/// Options for `torch::nn::functional::lp_pool3d`.
///
/// See the documentation for `torch::nn::LPPool3dOptions` class to learn what
/// arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::lp_pool3d(x, F::LPPool3dFuncOptions(2, {2, 3, 4}).stride(2));
/// ```
using LPPool3dFuncOptions = LPPool3dOptions;
} // namespace functional
/// 导入 torch 库中的 nn 命名空间
namespace nn {
    /// 定义功能性操作的命名空间
    namespace functional {
        /// 使用 LPPool3dOptions 类的文档，了解支持的参数和用法
        ///
        /// 示例:
        /// ```
        /// namespace F = torch::nn::functional;
        /// F::lp_pool3d(x, F::LPPool3dFuncOptions(2, {2, 3, 4}).stride(2));
        /// ```
        using LPPool3dFuncOptions = LPPool3dOptions;
    } // namespace functional
} // namespace nn
} // namespace torch
```