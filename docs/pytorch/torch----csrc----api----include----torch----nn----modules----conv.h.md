# `.\pytorch\torch\csrc\api\include\torch\nn\modules\conv.h`

```
#pragma once

#include <c10/util/irange.h>  // 包含用于范围迭代的头文件
#include <c10/util/overloaded.h>  // 包含用于多态访问的头文件

#include <torch/expanding_array.h>  // 包含用于扩展数组操作的头文件
#include <torch/nn/cloneable.h>  // 包含可克隆模块的头文件
#include <torch/nn/init.h>  // 包含模型初始化相关的头文件
#include <torch/nn/modules/common.h>  // 包含通用模块的头文件
#include <torch/nn/modules/utils.h>  // 包含模块工具函数的头文件
#include <torch/nn/options/conv.h>  // 包含卷积层选项的头文件
#include <torch/nn/pimpl.h>  // 包含模块实现的头文件
#include <torch/types.h>  // 包含 Torch 类型定义的头文件

#include <torch/csrc/Export.h>  // Torch 导出相关的头文件

#include <cstddef>  // 包含标准库定义的头文件
#include <vector>  // 包含向量容器的头文件

namespace torch {
namespace nn {

/// Base class for all (dimension-specialized) convolution modules.
template <size_t D, typename Derived>
class ConvNdImpl : public torch::nn::Cloneable<Derived> {
 public:
  explicit ConvNdImpl(detail::ConvNdOptions<D> options_)
      : options(std::move(options_)) {
    // NOLINTNEXTLINE(clang-analyzer-optin.cplusplus.VirtualCall)
    reset();  // 调用重置函数进行初始化
  }

  void reset() override {
    // 检查输入通道数、分组数和输出通道数是否为正整数
    TORCH_CHECK(
        options.in_channels() > 0 && options.groups() > 0 &&
            options.out_channels() > 0,
        "in_channels, groups and out_channels must be a positive integer.");
    // 检查输入通道数是否能被分组数整除
    TORCH_CHECK(
        options.in_channels() % options.groups() == 0,
        "in_channels must be divisible by groups");
    // 检查输出通道数是否能被分组数整除
    TORCH_CHECK(
        options.out_channels() % options.groups() == 0,
        "out_channels must be divisible by groups");

    // 根据不同的填充模式进行不同的处理
    std::visit(
        c10::overloaded(
            [&](enumtype::kValid) {
              // 对于有效填充模式，初始化填充数组为零
              _reversed_padding_repeated_twice.resize(2 * D);
              std::fill_n(_reversed_padding_repeated_twice.begin(), 2 * D, 0);
            },
            [&](enumtype::kSame) {
              // 对于相同填充模式，确保所有维度上的步长为1，否则抛出异常
              for (const auto i : c10::irange(D)) {
                const auto stride = (*options.stride())[i];
                TORCH_CHECK(
                    stride == 1,
                    "padding='same' is not supported for strided convolutions");
              }

              // 初始化填充数组，根据卷积核大小、膨胀率和步长计算填充量
              _reversed_padding_repeated_twice.resize(2 * D);
              for (const auto i : c10::irange(D)) {
                const auto dilation = (*options.dilation())[i];
                const auto kernel_size = (*options.kernel_size())[i];
                const auto total_padding = dilation * (kernel_size - 1);
                auto left_pad = total_padding / 2;
                auto right_pad = total_padding - left_pad;
                _reversed_padding_repeated_twice[2 * i] = left_pad;
                _reversed_padding_repeated_twice[2 * i + 1] = right_pad;
              }
            },
            [&](const ExpandingArray<D>& pad) {
              // 对于指定填充模式，使用工具函数生成反转重复的填充数组
              _reversed_padding_repeated_twice =
                  torch::nn::modules::utils::_reverse_repeat_vector(pad, 2);
            }),
        options.padding());

    if (options.transposed()) {
      // 如果是转置卷积，则根据选项初始化权重张量的大小
      std::vector<int64_t> weight_sizes = {
          options.in_channels(), options.out_channels() / options.groups()};
      weight_sizes.insert(
          weight_sizes.end(),
          (*options.kernel_size()).begin(),
          (*options.kernel_size()).end());
      weight = this->register_parameter("weight", torch::empty(weight_sizes));
  } else {
    // 如果有偏置，则创建指定大小的张量并注册为参数
    std::vector<int64_t> weight_sizes = {
        options.out_channels(), options.in_channels() / options.groups()};
    // 将卷积核大小添加到weight_sizes中
    weight_sizes.insert(
        weight_sizes.end(),
        (*options.kernel_size()).begin(),
        (*options.kernel_size()).end());
    // 使用torch::empty创建未初始化的权重张量，并注册为模块的参数
    weight = this->register_parameter("weight", torch::empty(weight_sizes));
  }

  // 如果有偏置项，则创建指定大小的张量并注册为参数；否则，注册一个空的张量，不需要梯度
  if (options.bias()) {
    bias = this->register_parameter(
        "bias", torch::empty({options.out_channels()}));
  } else {
    this->register_parameter("bias", Tensor(), /*requires_grad=*/false);
  }

  // 调用reset_parameters函数初始化权重和偏置
  reset_parameters();
}

// 初始化权重和偏置的函数
void reset_parameters() {
  // 使用kaiming_uniform_函数初始化权重，参数a设置为std::sqrt(5)
  init::kaiming_uniform_(
      weight,
      /*a=*/std::sqrt(5)); // NOLINT(cppcoreguidelines-avoid-magic-numbers)

  // 如果偏置已定义，则使用uniform_函数初始化偏置
  if (bias.defined()) {
    auto [fan_in, fan_out] = init::_calculate_fan_in_and_fan_out(weight);
    auto bound = 1 / std::sqrt(fan_in);
    init::uniform_(bias, -bound, bound);
  }
}

/// 格式化打印Conv{1,2,3}d模块到给定的流stream中
void pretty_print(std::ostream& stream) const override {
  // 打印卷积层的类型和参数信息
  stream << "torch::nn::Conv" << D << "d"
         << "(" << options.in_channels() << ", " << options.out_channels()
         << ", kernel_size=" << options.kernel_size()
         << ", stride=" << options.stride();
  // 根据填充选项打印相应的信息
  std::visit(
      c10::overloaded(
          [&](enumtype::kValid) { stream << ", padding='valid'"; },
          [&](enumtype::kSame) { stream << ", padding='same'"; },
          [&](const ExpandingArray<D>& pad) {
            if (*pad != *ExpandingArray<D>(0)) {
              stream << ", padding=" << pad;
            }
          }),
      options.padding());
  // 如果有膨胀选项，打印膨胀信息
  if (*options.dilation() != *ExpandingArray<D>(1)) {
    stream << ", dilation=" << options.dilation();
  }
  // 如果有输出填充选项，打印输出填充信息
  if (*options.output_padding() != *ExpandingArray<D>(0)) {
    stream << ", output_padding=" << options.output_padding();
  }
  // 如果有分组卷积选项，打印分组信息
  if (options.groups() != 1) {
    stream << ", groups=" << options.groups();
  }
  // 如果没有偏置项，打印bias=false
  if (!options.bias()) {
    stream << ", bias=" << std::boolalpha << false;
  }
  // 如果填充模式不是zeros，打印填充模式信息
  if (!std::get_if<enumtype::kZeros>(&options.padding_mode())) {
    stream << ", padding_mode="
           << enumtype::get_enum_name(options.padding_mode());
  }
  stream << ")";
}

/// 创建该模块时使用的选项
// NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
detail::ConvNdOptions<D> options;

/// 学习到的核（权重）
// NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
Tensor weight;

/// 学习到的偏置。仅在`bias`选项为true时定义
// NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
Tensor bias;

protected:
// NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
std::vector<int64_t> _reversed_padding_repeated_twice;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Conv1d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies convolution over a 1-D input.
/// See https://pytorch.org/docs/main/nn.html#torch.nn.Conv1d to learn about
/// the exact behavior of this module.
///
/// See the documentation for `torch::nn::Conv1dOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// Conv1d model(Conv1dOptions(3, 2, 3).stride(1).bias(false));
/// ```
class TORCH_API Conv1dImpl : public ConvNdImpl<1, Conv1dImpl> {
 public:
  // 构造函数，初始化 Conv1dImpl 实例
  Conv1dImpl(
      int64_t input_channels,
      int64_t output_channels,
      ExpandingArray<1> kernel_size)
      : Conv1dImpl(
            Conv1dOptions(input_channels, output_channels, kernel_size)) {}
  // 显式构造函数声明
  explicit Conv1dImpl(Conv1dOptions options_);
  // 前向传播函数声明，计算输入的张量
  Tensor forward(const Tensor& input);
};

/// A `ModuleHolder` subclass for `Conv1dImpl`.
/// See the documentation for `Conv1dImpl` class to learn what methods it
/// provides, and examples of how to use `Conv1d` with
/// `torch::nn::Conv1dOptions`. See the documentation for `ModuleHolder` to
/// learn about PyTorch's module storage semantics.
TORCH_MODULE(Conv1d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Conv2d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies convolution over a 2-D input.
/// See https://pytorch.org/docs/main/nn.html#torch.nn.Conv2d to learn about
/// the exact behavior of this module.
///
/// See the documentation for `torch::nn::Conv2dOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// Conv2d model(Conv2dOptions(3, 2, 3).stride(1).bias(false));
/// ```
class TORCH_API Conv2dImpl : public ConvNdImpl<2, Conv2dImpl> {
 public:
  // 构造函数，初始化 Conv2dImpl 实例
  Conv2dImpl(
      int64_t input_channels,
      int64_t output_channels,
      ExpandingArray<2> kernel_size)
      : Conv2dImpl(
            Conv2dOptions(input_channels, output_channels, kernel_size)) {}
  // 显式构造函数声明
  explicit Conv2dImpl(Conv2dOptions options_);
  // 前向传播函数声明，计算输入的张量
  Tensor forward(const Tensor& input);

 protected:
  // 私有函数，实现卷积的前向传播
  Tensor _conv_forward(const Tensor& input, const Tensor& weight);
};

/// A `ModuleHolder` subclass for `Conv2dImpl`.
/// See the documentation for `Conv2dImpl` class to learn what methods it
/// provides, and examples of how to use `Conv2d` with
/// `torch::nn::Conv2dOptions`. See the documentation for `ModuleHolder` to
/// learn about PyTorch's module storage semantics.
TORCH_MODULE(Conv2d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Conv3d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies convolution over a 3-D input.
/// See https://pytorch.org/docs/main/nn.html#torch.nn.Conv3d to learn about
/// the exact behavior of this module.
///
/// See the documentation for `torch::nn::Conv3dOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// Conv3d model(Conv3dOptions(3, 2, 3).stride(1).bias(false));
/// ```
/// `Conv3dImpl` 类继承自 `ConvNdImpl` 类，用于实现 3 维卷积操作。
/// 该类提供了处理 3 维卷积的基本功能和配置选项。
class Conv3dImpl : public ConvNdImpl<3, Conv3dImpl> {
 public:
  /// 构造函数，接受输入通道数、输出通道数和核大小作为参数，初始化 `Conv3dImpl` 对象。
  Conv3dImpl(
      int64_t input_channels,
      int64_t output_channels,
      ExpandingArray<3> kernel_size)
      : Conv3dImpl(
            Conv3dOptions(input_channels, output_channels, kernel_size)) {}
  
  /// 显式构造函数，接受 `Conv3dOptions` 对象作为参数，初始化 `Conv3dImpl` 对象。
  explicit Conv3dImpl(Conv3dOptions options_);
  
  /// 前向传播函数，接受输入张量 `input` 作为参数，返回输出张量。
  Tensor forward(const Tensor& input);
};

/// `Conv3d` 是 `ModuleHolder` 的子类，用于持有 `Conv3dImpl` 模块。
/// 参考 `Conv3dImpl` 类的文档以了解其提供的方法，以及如何使用 `torch::nn::Conv3dOptions` 配置 `Conv3d`。
/// 参考 `ModuleHolder` 的文档以了解 PyTorch 模块存储语义。
TORCH_MODULE(Conv3d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~ ConvTranspose ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// 所有（维度特定）卷积转置模块的基类模板。
template <size_t D, typename Derived>
class ConvTransposeNdImpl : public ConvNdImpl<D, Derived> {
 public:
  /// 继承自 `ConvNdImpl` 的构造函数。
  using torch::nn::ConvNdImpl<D, Derived>::ConvNdImpl;
  
  /// 显式构造函数，接受 `detail::ConvNdOptions<D>` 对象作为参数，初始化 `ConvNdImpl<D, Derived>` 对象。
  explicit ConvTransposeNdImpl(detail::ConvNdOptions<D> options_)
      : ConvNdImpl<D, Derived>(options_) {
    // 内部断言，确保 `ConvTranspose` 的填充参数不是字符串。
    TORCH_INTERNAL_ASSERT(
        std::holds_alternative<ExpandingArray<D>>(this->options.padding()),
        "ConvTranspose padding cannot be a string");
  }

  /// 将 `ConvTranspose{1,2,3}d` 模块美化打印到给定的流 `stream` 中。
  void pretty_print(std::ostream& stream) const override {
    stream << "torch::nn::ConvTranspose" << D << "d"
           << "(" << this->options.in_channels() << ", "
           << this->options.out_channels()
           << ", kernel_size=" << this->options.kernel_size()
           << ", stride=" << this->options.stride();
    const auto& pad = padding();
    if (*pad != *ExpandingArray<D>(0)) {
      stream << ", padding=" << pad;
    }
    if (*this->options.dilation() != *ExpandingArray<D>(1)) {
      stream << ", dilation=" << this->options.dilation();
    }
    if (*this->options.output_padding() != *ExpandingArray<D>(0)) {
      stream << ", output_padding=" << this->options.output_padding();
    }
    if (this->options.groups() != 1) {
      stream << ", groups=" << this->options.groups();
    }
    if (!this->options.bias()) {
      stream << ", bias=" << std::boolalpha << false;
    }
    if (!std::get_if<enumtype::kZeros>(&this->options.padding_mode())) {
      stream << ", padding_mode="
             << enumtype::get_enum_name(this->options.padding_mode());
    }
    stream << ")";
  }

 protected:
  /// 返回填充参数的引用。
  const ExpandingArray<D>& padding() const {
    return std::get<ExpandingArray<D>>(this->options.padding());
  }

  /// 计算输出填充的大小。
  std::vector<int64_t> _output_padding(
      const Tensor& input,
      const std::optional<at::IntArrayRef>& output_size,
      const ExpandingArray<D>& stride,
      const ExpandingArray<D>& padding,
      const ExpandingArray<D>& kernel_size);
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ConvTranspose1d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// 应用于 `ConvTranspose1d` 函数的模块。
/// See https://pytorch.org/docs/main/nn.html#torch.nn.ConvTranspose1d to
/// learn about the exact behavior of this module.
///
/// See the documentation for `torch::nn::ConvTranspose1dOptions` class to learn
/// what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// ConvTranspose1d model(ConvTranspose1dOptions(3, 2,
/// 3).stride(1).bias(false));
/// ```
class TORCH_API ConvTranspose1dImpl
    : public ConvTransposeNdImpl<1, ConvTranspose1dImpl> {
 public:
  /// Constructor for ConvTranspose1dImpl class.
  /// Initializes a 1-dimensional transposed convolution module with specified
  /// input channels, output channels, and kernel size.
  ConvTranspose1dImpl(
      int64_t input_channels,  // Number of channels in input tensor
      int64_t output_channels, // Number of channels produced by the convolution
      ExpandingArray<1> kernel_size)  // Size of the convolutional kernel
      : ConvTranspose1dImpl(ConvTranspose1dOptions(
            input_channels,
            output_channels,
            kernel_size)) {}  // Delegates to ConvTranspose1dOptions constructor

  /// Explicit constructor for ConvTranspose1dImpl class.
  explicit ConvTranspose1dImpl(
      ConvTranspose1dOptions options_);  // Options object for transposed convolution

  /// Forward function for ConvTranspose1dImpl.
  /// Performs the forward pass of the transposed convolution.
  /// Takes input tensor and optional output size, returns the output tensor.
  Tensor forward(
      const Tensor& input,
      const std::optional<at::IntArrayRef>& output_size = c10::nullopt);

 protected:
  /// Macro defining default arguments for forward function.
  FORWARD_HAS_DEFAULT_ARGS({1, AnyValue(std::optional<at::IntArrayRef>())})
};

/// A `ModuleHolder` subclass for `ConvTranspose1dImpl`.
/// See the documentation for `ConvTranspose1dImpl` class to learn what methods
/// it provides, and examples of how to use `ConvTranspose1d` with
/// `torch::nn::ConvTranspose1dOptions`. See the documentation for
/// `ModuleHolder` to learn about PyTorch's module storage semantics.
TORCH_MODULE(ConvTranspose1d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ConvTranspose2d
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies the ConvTranspose2d function.
/// See https://pytorch.org/docs/main/nn.html#torch.nn.ConvTranspose2d to
/// learn about the exact behavior of this module.
///
/// See the documentation for `torch::nn::ConvTranspose2dOptions` class to learn
/// what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// ConvTranspose2d model(ConvTranspose2dOptions(3, 2,
/// 3).stride(1).bias(false));
/// ```
class TORCH_API ConvTranspose2dImpl
    : public ConvTransposeNdImpl<2, ConvTranspose2dImpl> {
 public:
  /// Constructor for ConvTranspose2dImpl class.
  /// Initializes a 2-dimensional transposed convolution module with specified
  /// input channels, output channels, and kernel size.
  ConvTranspose2dImpl(
      int64_t input_channels,  // Number of channels in input tensor
      int64_t output_channels, // Number of channels produced by the convolution
      ExpandingArray<2> kernel_size)  // Size of the convolutional kernel
      : ConvTranspose2dImpl(ConvTranspose2dOptions(
            input_channels,
            output_channels,
            kernel_size)) {}  // Delegates to ConvTranspose2dOptions constructor

  /// Explicit constructor for ConvTranspose2dImpl class.
  explicit ConvTranspose2dImpl(
      ConvTranspose2dOptions options_);  // Options object for transposed convolution

  /// Forward function for ConvTranspose2dImpl.
  /// Performs the forward pass of the transposed convolution.
  /// Takes input tensor and optional output size, returns the output tensor.
  Tensor forward(
      const Tensor& input,
      const std::optional<at::IntArrayRef>& output_size = c10::nullopt);

 protected:
  /// Macro defining default arguments for forward function.
  FORWARD_HAS_DEFAULT_ARGS({1, AnyValue(std::optional<at::IntArrayRef>())})
};

/// A `ModuleHolder` subclass for `ConvTranspose2dImpl`.
/// See the documentation for `ConvTranspose2dImpl` class to learn what methods
/// it provides, and examples of how to use `ConvTranspose2d` with
/// `torch::nn::ConvTranspose2dOptions`. See the documentation for
/// `ModuleHolder` to learn about PyTorch's module storage semantics.
TORCH_MODULE(ConvTranspose2d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ConvTranspose3d
/// 定义了 ConvTranspose3dImpl 类，继承自 ConvTransposeNdImpl<3, ConvTranspose3dImpl>
/// 这是 ConvTranspose3d 的实现，是一个三维转置卷积的模块。
/// 参考 https://pytorch.org/docs/main/nn.html#torch.nn.ConvTranspose3d 了解此模块的具体行为。
///
/// 参考 `torch::nn::ConvTranspose3dOptions` 类的文档，了解可以用于此模块的构造函数参数。
///
/// 示例:
/// ```
/// ConvTranspose3d model(ConvTranspose3dOptions(2, 2, 2).stride(1).bias(false));
/// ```
class TORCH_API ConvTranspose3dImpl : public ConvTransposeNdImpl<3, ConvTranspose3dImpl> {
 public:
  /// 构造函数，初始化 ConvTranspose3dImpl 对象
  /// 参数:
  ///   - input_channels: 输入通道数
  ///   - output_channels: 输出通道数
  ///   - kernel_size: 卷积核大小
  ConvTranspose3dImpl(int64_t input_channels, int64_t output_channels, ExpandingArray<3> kernel_size)
      : ConvTranspose3dImpl(ConvTranspose3dOptions(input_channels, output_channels, kernel_size)) {}

  /// 显式构造函数，初始化 ConvTranspose3dImpl 对象
  /// 参数:
  ///   - options_: ConvTranspose3dOptions 类的选项对象
  explicit ConvTranspose3dImpl(ConvTranspose3dOptions options_);

  /// 前向传播函数，计算 ConvTranspose3d 的输出
  /// 参数:
  ///   - input: 输入张量
  ///   - output_size: 可选参数，指定输出张量的大小，默认为自动推断
  Tensor forward(const Tensor& input, const std::optional<at::IntArrayRef>& output_size = c10::nullopt);

 protected:
  /// 默认参数的前向传播函数声明，用于指定 forward 方法的默认参数
  FORWARD_HAS_DEFAULT_ARGS({1, AnyValue(std::optional<at::IntArrayRef>())})
};

/// ConvTranspose3d 的 ModuleHolder 子类
/// 提供了 ConvTranspose3dImpl 类的模块包装器，使其能够按照 PyTorch 的模块存储语义进行使用。
/// 参考 `ConvTranspose3dImpl` 类的文档了解其提供的方法，以及使用 `torch::nn::ConvTranspose3dOptions` 使用 ConvTranspose3d 的示例。
/// 参考 `ModuleHolder` 的文档了解 PyTorch 模块存储的语义。
TORCH_MODULE(ConvTranspose3d);

} // namespace nn
} // namespace torch
```