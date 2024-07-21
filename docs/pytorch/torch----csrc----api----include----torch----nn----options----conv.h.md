# `.\pytorch\torch\csrc\api\include\torch\nn\options\conv.h`

```
#pragma once

#include <torch/arg.h>  // 包含 Torch 库的参数处理头文件
#include <torch/csrc/Export.h>  // 包含 Torch 库的导出头文件
#include <torch/enum.h>  // 包含 Torch 库的枚举定义头文件
#include <torch/expanding_array.h>  // 包含 Torch 库的扩展数组处理头文件
#include <torch/types.h>  // 包含 Torch 库的类型定义头文件

namespace torch {
namespace nn {

namespace detail {

typedef std::variant<  // 定义一个包含不同枚举类型的 std::variant
    enumtype::kZeros,  // 枚举类型 kZeros
    enumtype::kReflect,  // 枚举类型 kReflect
    enumtype::kReplicate,  // 枚举类型 kReplicate
    enumtype::kCircular>  // 枚举类型 kCircular
    conv_padding_mode_t;  // 使用 typedef 定义名为 conv_padding_mode_t 的类型

template <size_t D>
using conv_padding_t =  // 使用 using 定义名为 conv_padding_t 的模板别名
    std::variant<ExpandingArray<D>, enumtype::kValid, enumtype::kSame>;  // 包含扩展数组和枚举类型的 std::variant

/// Options for a `D`-dimensional convolution or convolution transpose module.
/// 用于 `D` 维卷积或转置卷积模块的选项。
template <size_t D>
struct ConvNdOptions {
  using padding_t = conv_padding_t<D>;  // 使用 conv_padding_t<D> 定义 padding_t 别名

  ConvNdOptions(
      int64_t in_channels,
      int64_t out_channels,
      ExpandingArray<D> kernel_size)
      : in_channels_(in_channels),  // 初始化成员变量 in_channels_
        out_channels_(out_channels),  // 初始化成员变量 out_channels_
        kernel_size_(std::move(kernel_size)) {}  // 初始化成员变量 kernel_size_

  /// The number of channels the input volumes will have.
  /// 输入体积的通道数。
  /// Changing this parameter after construction __has no effect__.
  TORCH_ARG(int64_t, in_channels);  // 定义 in_channels 参数，不可在构造后更改

  /// The number of output channels the convolution should produce.
  /// 卷积应生成的输出通道数。
  /// Changing this parameter after construction __has no effect__.
  TORCH_ARG(int64_t, out_channels);  // 定义 out_channels 参数，不可在构造后更改

  /// The kernel size to use.
  /// 要使用的卷积核大小。
  /// For a `D`-dim convolution, must be a single number or a list of `D`
  /// numbers.
  /// This parameter __can__ be changed after construction.
  TORCH_ARG(ExpandingArray<D>, kernel_size);  // 定义 kernel_size 参数，构造后可更改

  /// The stride of the convolution.
  /// 卷积的步长。
  /// For a `D`-dim convolution, must be a single number or a list of `D`
  /// numbers.
  /// This parameter __can__ be changed after construction.
  TORCH_ARG(ExpandingArray<D>, stride) = 1;  // 定义 stride 参数，默认为1，构造后可更改

  /// The padding to add to the input volumes.
  /// 添加到输入体积的填充。
  /// For a `D`-dim convolution, must be a single number or a list of `D`
  /// numbers.
  /// This parameter __can__ be changed after construction.
  TORCH_ARG(padding_t, padding) = 0;  // 定义 padding 参数，默认为0，构造后可更改

 public:
  decltype(auto) padding(std::initializer_list<int64_t> il) {
  // 返回整数数组引用 `il` 经过填充后的结果
  return padding(IntArrayRef{il});
}

/// 卷积操作中的膨胀率。
/// 对于 `D` 维卷积，可以是一个数字或者包含 `D` 个数字的列表。
/// 这个参数在构造后__可以__被修改。
TORCH_ARG(ExpandingArray<D>, dilation) = 1;

/// 如果为 true，则执行转置卷积（也称为反卷积）。
/// 在构造后改变这个参数__没有任何效果__。
TORCH_ARG(bool, transposed) = false;

/// 对于转置卷积，输出体积需要添加的填充。
/// 对于 `D` 维卷积，可以是一个数字或者包含 `D` 个数字的列表。
/// 这个参数在构造后__可以__被修改。
TORCH_ARG(ExpandingArray<D>, output_padding) = 0;

/// 卷积组的数量。
/// 这个参数在构造后__可以__被修改。
TORCH_ARG(int64_t, groups) = 1;

/// 是否在每次卷积操作后添加偏置。
/// 在构造后改变这个参数__没有任何效果__。
TORCH_ARG(bool, bias) = true;

/// 接受的填充模式为 `torch::kZeros`, `torch::kReflect`, `torch::kReplicate` 或 `torch::kCircular`。
/// 默认为 `torch::kZeros`。
TORCH_ARG(conv_padding_mode_t, padding_mode) = torch::kZeros;
}; // 结束 detail 命名空间

} // namespace detail 结束

// ============================================================================

/// Options for a `D`-dimensional convolution module.
template <size_t D>
struct ConvOptions {
  using padding_mode_t = detail::conv_padding_mode_t; // 定义 padding_mode_t 类型为 detail::conv_padding_mode_t
  using padding_t = detail::conv_padding_t<D>; // 定义 padding_t 类型为 detail::conv_padding_t<D>

  ConvOptions(
      int64_t in_channels,
      int64_t out_channels,
      ExpandingArray<D> kernel_size)
      : in_channels_(in_channels), // 初始化输入通道数
        out_channels_(out_channels), // 初始化输出通道数
        kernel_size_(std::move(kernel_size)) {} // 初始化卷积核大小

  /// The number of channels the input volumes will have.
  /// Changing this parameter after construction __has no effect__.
  TORCH_ARG(int64_t, in_channels); // 输入通道数，构造后不可改变

  /// The number of output channels the convolution should produce.
  /// Changing this parameter after construction __has no effect__.
  TORCH_ARG(int64_t, out_channels); // 输出通道数，构造后不可改变

  /// The kernel size to use.
  /// For a `D`-dim convolution, must be a single number or a list of `D`
  /// numbers.
  /// This parameter __can__ be changed after construction.
  TORCH_ARG(ExpandingArray<D>, kernel_size); // 卷积核大小，构造后可改变

  /// The stride of the convolution.
  /// For a `D`-dim convolution, must be a single number or a list of `D`
  /// numbers.
  /// This parameter __can__ be changed after construction.
  TORCH_ARG(ExpandingArray<D>, stride) = 1; // 卷积步长，默认为1，构造后可改变

  /// The padding to add to the input volumes.
  /// For a `D`-dim convolution, must be a single number or a list of `D`
  /// numbers.
  /// This parameter __can__ be changed after construction.
  TORCH_ARG(padding_t, padding) = 0; // 输入体积的填充量，默认为0，构造后可改变

 public:
  decltype(auto) padding(std::initializer_list<int64_t> il) {
    return padding(IntArrayRef{il}); // 设置填充参数的初始化列表
  }

  /// The kernel dilation.
  /// For a `D`-dim convolution, must be a single number or a list of `D`
  /// numbers.
  /// This parameter __can__ be changed after construction.
  TORCH_ARG(ExpandingArray<D>, dilation) = 1; // 卷积核膨胀率，默认为1，构造后可改变

  /// The number of convolution groups.
  /// This parameter __can__ be changed after construction.
  TORCH_ARG(int64_t, groups) = 1; // 卷积组数，默认为1，构造后可改变

  /// Whether to add a bias after individual applications of the kernel.
  /// Changing this parameter after construction __has no effect__.
  TORCH_ARG(bool, bias) = true; // 是否在卷积后添加偏置，默认为true，构造后不可改变

  /// Accepted values `torch::kZeros`, `torch::kReflect`, `torch::kReplicate` or
  /// `torch::kCircular`. Default: `torch::kZeros`
  TORCH_ARG(padding_mode_t, padding_mode) = torch::kZeros; // 填充模式，默认为 torch::kZeros

};

/// `ConvOptions` specialized for the `Conv1d` module.
///
/// Example:
/// ```
/// Conv1d model(Conv1dOptions(3, 2, 3).stride(1).bias(false));
/// ```
using Conv1dOptions = ConvOptions<1>; // Conv1d 模块的特化选项

/// `ConvOptions` specialized for the `Conv2d` module.
///
/// Example:
/// ```
/// Conv2d model(Conv2dOptions(3, 2, 3).stride(1).bias(false));
/// ```
using Conv2dOptions = ConvOptions<2>; // Conv2d 模块的特化选项

/// `ConvOptions` specialized for the `Conv3d` module.
///
/// Example:
/// ```
/// Conv3d model(Conv3dOptions(3, 2, 3).stride(1).bias(false));
/// ```
using Conv3dOptions = ConvOptions<3>; // Conv3d 模块的特化选项
// ============================================================================

namespace functional {

/// Options for a `D`-dimensional convolution functional.
/// Defines a struct template for convolution functional options in D dimensions.

template <size_t D>
struct ConvFuncOptions {
  using padding_t = torch::nn::detail::conv_padding_t<D>;

  /// optional bias of shape `(out_channels)`. Default: ``None``
  TORCH_ARG(torch::Tensor, bias) = Tensor();

  /// The stride of the convolving kernel.
  /// For a `D`-dim convolution, must be a single number or a list of `D`
  /// numbers.
  TORCH_ARG(ExpandingArray<D>, stride) = 1;

  /// Implicit paddings on both sides of the input.
  /// For a `D`-dim convolution, must be a single number or a list of `D`
  /// numbers.
  TORCH_ARG(padding_t, padding) = 0;

 public:
  /// Setter method for padding options using initializer list.
  decltype(auto) padding(std::initializer_list<int64_t> il) {
    return padding(IntArrayRef{il});
  }

  /// The spacing between kernel elements.
  /// For a `D`-dim convolution, must be a single number or a list of `D`
  /// numbers.
  TORCH_ARG(ExpandingArray<D>, dilation) = 1;

  /// Split input into groups, `in_channels` should be divisible by
  /// the number of groups.
  TORCH_ARG(int64_t, groups) = 1;
};

/// `ConvFuncOptions` specialized for `torch::nn::functional::conv1d`.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::conv1d(x, weight, F::Conv1dFuncOptions().stride(1));
/// ```
using Conv1dFuncOptions = ConvFuncOptions<1>;

/// `ConvFuncOptions` specialized for `torch::nn::functional::conv2d`.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::conv2d(x, weight, F::Conv2dFuncOptions().stride(1));
/// ```
using Conv2dFuncOptions = ConvFuncOptions<2>;

/// `ConvFuncOptions` specialized for `torch::nn::functional::conv3d`.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::conv3d(x, weight, F::Conv3dFuncOptions().stride(1));
/// ```
using Conv3dFuncOptions = ConvFuncOptions<3>;

} // namespace functional

// ============================================================================
/// Options structure for configuring parameters of a transposed convolution operation.
struct ConvTransposeOptions {
  using padding_mode_t = detail::conv_padding_mode_t;  // Define an alias for padding mode type

  /// Constructor initializing convolution parameters.
  ConvTransposeOptions(
      int64_t in_channels,
      int64_t out_channels,
      ExpandingArray<D> kernel_size)
      : in_channels_(in_channels),                     // Initialize input channel count
        out_channels_(out_channels),                   // Initialize output channel count
        kernel_size_(std::move(kernel_size)) {}        // Initialize kernel size array

  /// The number of channels the input volumes will have.
  /// Changing this parameter after construction __has no effect__.
  TORCH_ARG(int64_t, in_channels);

  /// The number of output channels the convolution should produce.
  /// Changing this parameter after construction __has no effect__.
  TORCH_ARG(int64_t, out_channels);

  /// The kernel size to use.
  /// For a `D`-dim convolution, must be a single number or a list of `D`
  /// numbers.
  /// This parameter __can__ be changed after construction.
  TORCH_ARG(ExpandingArray<D>, kernel_size);

  /// The stride of the convolution.
  /// For a `D`-dim convolution, must be a single number or a list of `D`
  /// numbers.
  /// This parameter __can__ be changed after construction.
  TORCH_ARG(ExpandingArray<D>, stride) = 1;

  /// The padding to add to the input volumes.
  /// For a `D`-dim convolution, must be a single number or a list of `D`
  /// numbers.
  /// This parameter __can__ be changed after construction.
  TORCH_ARG(ExpandingArray<D>, padding) = 0;

  /// For transpose convolutions, the padding to add to output volumes.
  /// For a `D`-dim convolution, must be a single number or a list of `D`
  /// numbers.
  /// This parameter __can__ be changed after construction.
  TORCH_ARG(ExpandingArray<D>, output_padding) = 0;

  /// The number of convolution groups.
  /// This parameter __can__ be changed after construction.
  TORCH_ARG(int64_t, groups) = 1;

  /// Whether to add a bias after individual applications of the kernel.
  /// Changing this parameter after construction __has no effect__.
  TORCH_ARG(bool, bias) = true;

  /// The kernel dilation.
  /// For a `D`-dim convolution, must be a single number or a list of `D`
  /// numbers.
  /// This parameter __can__ be changed after construction.
  TORCH_ARG(ExpandingArray<D>, dilation) = 1;

  /// Accepted values `torch::kZeros`, `torch::kReflect`, `torch::kReplicate` or
  /// `torch::kCircular`. Default: `torch::kZeros`
  TORCH_ARG(padding_mode_t, padding_mode) = torch::kZeros;  // Specify default padding mode.
};

/// `ConvTransposeOptions` specialized for the `ConvTranspose1d` module.
///
/// Example:
/// ```
/// ConvTranspose1d model(ConvTranspose1dOptions(3, 2,
/// 3).stride(1).bias(false));
/// ```
using ConvTranspose1dOptions = ConvTransposeOptions<1>;  // Alias for 1-dimensional transposed convolution options.

/// `ConvTransposeOptions` specialized for the `ConvTranspose2d` module.
///
/// Example:
/// ```
/// ConvTranspose2d model(ConvTranspose2dOptions(3, 2,
/// 3).stride(1).bias(false));
/// ```
using ConvTranspose2dOptions = ConvTransposeOptions<2>;  // Alias for 2-dimensional transposed convolution options.

/// `ConvTransposeOptions` specialized for the `ConvTranspose3d` module.
///
/// Example:
/// ```
/// ConvTranspose3d model(ConvTranspose3dOptions(3, 2,
/// 3).stride(1).bias(false));
/// ```
using ConvTranspose3dOptions = ConvTransposeOptions<3>;  // Alias for 3-dimensional transposed convolution options.
/// 使用 ConvTranspose3dOptions 类创建 ConvTranspose3d 模型，设置为 2x2x2 的卷积转置操作，
/// 并且不使用偏置。
using ConvTranspose3dOptions = ConvTransposeOptions<3>;

// ============================================================================

namespace functional {

/// `D` 维度卷积转置操作的选项。
template <size_t D>
struct ConvTransposeFuncOptions {
  /// 可选的偏置，形状为 `(out_channels)`。默认值：``None``
  TORCH_ARG(torch::Tensor, bias) = Tensor();

  /// 卷积核的步幅。
  /// 对于 `D` 维卷积，必须是一个数字或者 `D` 个数字的列表。
  TORCH_ARG(ExpandingArray<D>, stride) = 1;

  /// 输入两侧的隐式填充。
  /// 对于 `D` 维卷积，必须是一个数字或者 `D` 个数字的列表。
  TORCH_ARG(ExpandingArray<D>, padding) = 0;

  /// 输出形状中每个维度一侧增加的额外大小。
  /// 默认值：0
  TORCH_ARG(ExpandingArray<D>, output_padding) = 0;

  /// 将输入分成组，`in_channels` 应该能够被组的数量整除。
  TORCH_ARG(int64_t, groups) = 1;

  /// 卷积核元素之间的间距。
  /// 对于 `D` 维卷积，必须是一个数字或者 `D` 个数字的列表。
  TORCH_ARG(ExpandingArray<D>, dilation) = 1;
};

/// `ConvTransposeFuncOptions` 专门用于 `torch::nn::functional::conv_transpose1d`。
///
/// 示例：
/// ```
/// namespace F = torch::nn::functional;
/// F::conv_transpose1d(x, weight, F::ConvTranspose1dFuncOptions().stride(1));
/// ```
using ConvTranspose1dFuncOptions = ConvTransposeFuncOptions<1>;

/// `ConvTransposeFuncOptions` 专门用于 `torch::nn::functional::conv_transpose2d`。
///
/// 示例：
/// ```
/// namespace F = torch::nn::functional;
/// F::conv_transpose2d(x, weight, F::ConvTranspose2dFuncOptions().stride(1));
/// ```
using ConvTranspose2dFuncOptions = ConvTransposeFuncOptions<2>;

/// `ConvTransposeFuncOptions` 专门用于 `torch::nn::functional::conv_transpose3d`。
///
/// 示例：
/// ```
/// namespace F = torch::nn::functional;
/// F::conv_transpose3d(x, weight, F::ConvTranspose3dFuncOptions().stride(1));
/// ```
using ConvTranspose3dFuncOptions = ConvTransposeFuncOptions<3>;

} // namespace functional

} // namespace nn
} // namespace torch
```