# `.\pytorch\torch\csrc\api\include\torch\nn\modules\pooling.h`

```py
#pragma once
// 预处理指令，确保头文件只被编译一次

#include <torch/expanding_array.h>
#include <torch/nn/cloneable.h>
#include <torch/nn/functional/pooling.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/options/pooling.h>
// 包含 Torch 库的相关头文件

#include <torch/csrc/Export.h>
// 包含 Torch 的导出宏定义

namespace torch {
namespace nn {

/// Base class for all (dimension-specialized) avgpool modules.
// 所有维度特化的平均池化模块的基类模板
template <size_t D, typename Derived>
class TORCH_API AvgPoolImpl : public torch::nn::Cloneable<Derived> {
 public:
  AvgPoolImpl(ExpandingArray<D> kernel_size)
      : AvgPoolImpl(AvgPoolOptions<D>(kernel_size)) {}
  // 构造函数，使用扩展数组作为参数，并使用其它构造函数初始化

  explicit AvgPoolImpl(const AvgPoolOptions<D>& options_);
  // 显式构造函数，使用 AvgPoolOptions<D> 初始化

  void reset() override;
  // 重置函数，重载自基类 Cloneable

  /// Pretty prints the `AvgPool{1,2,3}d` module into the given `stream`.
  // 将 AvgPool{1,2,3}d 模块美观地打印到给定的流中
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  // 构造此模块时使用的选项
  AvgPoolOptions<D> options;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ AvgPool1d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies avgpool over a 1-D input.
// 对一维输入应用平均池化

/// See https://pytorch.org/docs/main/nn.html#torch.nn.AvgPool1d to learn
/// about the exact behavior of this module.
// 查看链接以了解此模块的确切行为

/// See the documentation for `torch::nn::AvgPool1dOptions` class to learn what
/// constructor arguments are supported for this module.
// 查看 torch::nn::AvgPool1dOptions 文档了解支持此模块的构造函数参数

/// Example:
/// ```
/// AvgPool1d model(AvgPool1dOptions(3).stride(2));
/// ```py
// 示例用法

class TORCH_API AvgPool1dImpl : public AvgPoolImpl<1, AvgPool1dImpl> {
 public:
  using AvgPoolImpl<1, AvgPool1dImpl>::AvgPoolImpl;
  // 使用基类构造函数的声明

  Tensor forward(const Tensor& input);
  // 前向传播函数，接受输入张量并返回输出张量
};

/// A `ModuleHolder` subclass for `AvgPool1dImpl`.
// 用于 AvgPool1dImpl 的 ModuleHolder 子类

/// See the documentation for `AvgPool1dImpl` class to learn what methods it
/// provides, and examples of how to use `AvgPool1d` with
/// `torch::nn::AvgPool1dOptions`. See the documentation for `ModuleHolder` to
/// learn about PyTorch's module storage semantics.
// 查看 AvgPool1dImpl 类的文档，了解其提供的方法和如何使用 AvgPool1d 的示例，
// 以及如何使用 ModuleHolder 了解 PyTorch 的模块存储语义

TORCH_MODULE(AvgPool1d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ AvgPool2d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies avgpool over a 2-D input.
// 对二维输入应用平均池化

/// See https://pytorch.org/docs/main/nn.html#torch.nn.AvgPool2d to learn
/// about the exact behavior of this module.
// 查看链接以了解此模块的确切行为

/// See the documentation for `torch::nn::AvgPool2dOptions` class to learn what
/// constructor arguments are supported for this module.
// 查看 torch::nn::AvgPool2dOptions 文档了解支持此模块的构造函数参数

/// Example:
/// ```
/// AvgPool2d model(AvgPool2dOptions({3, 2}).stride({2, 2}));
/// ```py
// 示例用法

class TORCH_API AvgPool2dImpl : public AvgPoolImpl<2, AvgPool2dImpl> {
 public:
  using AvgPoolImpl<2, AvgPool2dImpl>::AvgPoolImpl;
  // 使用基类构造函数的声明

  Tensor forward(const Tensor& input);
  // 前向传播函数，接受输入张量并返回输出张量
};

/// A `ModuleHolder` subclass for `AvgPool2dImpl`.
// 用于 AvgPool2dImpl 的 ModuleHolder 子类

/// See the documentation for `AvgPool2dImpl` class to learn what methods it
/// provides, and examples of how to use `AvgPool2d` with
/// `torch::nn::AvgPool2dOptions`. See the documentation for `ModuleHolder` to
/// learn about PyTorch's module storage semantics.
// 查看 AvgPool2dImpl 类的文档，了解其提供的方法和如何使用 AvgPool2d 的示例，
// 以及如何使用 ModuleHolder 了解 PyTorch 的模块存储语义

TORCH_MODULE(AvgPool2d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ AvgPool3d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies avgpool over a 3-D input.
// 对三维输入应用平均池化
/// See https://pytorch.org/docs/main/nn.html#torch.nn.AvgPool3d to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::AvgPool3dOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// AvgPool3d model(AvgPool3dOptions(5).stride(2));
/// ```py
class TORCH_API AvgPool3dImpl : public AvgPoolImpl<3, AvgPool3dImpl> {
 public:
  using AvgPoolImpl<3, AvgPool3dImpl>::AvgPoolImpl;
  /// Forward pass function that applies average pooling over a 3-D input tensor.
  /// Computes average values in each pooling window defined by the kernel size
  /// and other specified options.
  Tensor forward(const Tensor& input);
};

/// A `ModuleHolder` subclass for `AvgPool3dImpl`.
/// See the documentation for `AvgPool3dImpl` class to learn what methods it
/// provides, and examples of how to use `AvgPool3d` with
/// `torch::nn::AvgPool3dOptions`. See the documentation for `ModuleHolder` to
/// learn about PyTorch's module storage semantics.
TORCH_MODULE(AvgPool3d);

// ============================================================================

/// Base class for all (dimension-specialized) maxpool modules.
template <size_t D, typename Derived>
class TORCH_API MaxPoolImpl : public torch::nn::Cloneable<Derived> {
 public:
  /// Constructor that initializes max pooling with the specified kernel size.
  explicit MaxPoolImpl(ExpandingArray<D> kernel_size)
      : MaxPoolImpl(MaxPoolOptions<D>(kernel_size)) {}
  /// Explicit constructor with options for max pooling.
  explicit MaxPoolImpl(const MaxPoolOptions<D>& options_);

  /// Reset function to reset the internal state of the module.
  void reset() override;

  /// Pretty prints the `MaxPool{1,2,3}d` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  MaxPoolOptions<D> options;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MaxPool1d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies maxpool over a 1-D input.
/// See https://pytorch.org/docs/main/nn.html#torch.nn.MaxPool1d to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::MaxPool1dOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// MaxPool1d model(MaxPool1dOptions(3).stride(2));
/// ```py
class TORCH_API MaxPool1dImpl : public MaxPoolImpl<1, MaxPool1dImpl> {
 public:
  using MaxPoolImpl<1, MaxPool1dImpl>::MaxPoolImpl;
  /// Forward pass function that applies max pooling over a 1-D input tensor.
  /// Computes the maximum value in each pooling window defined by the kernel size
  /// and other specified options.
  Tensor forward(const Tensor& input);

  /// Returns the outputs and the indices of the max values.
  /// Useful for `torch::nn::MaxUnpool1d` later.
  std::tuple<Tensor, Tensor> forward_with_indices(const Tensor& input);
};

/// A `ModuleHolder` subclass for `MaxPool1dImpl`.
/// See the documentation for `MaxPool1dImpl` class to learn what methods it
/// provides, and examples of how to use `MaxPool1d` with
/// `torch::nn::MaxPool1dOptions`. See the documentation for `ModuleHolder` to
/// learn about PyTorch's module storage semantics.
TORCH_MODULE(MaxPool1d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MaxPool2d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies maxpool over a 2-D input.
/// See https://pytorch.org/docs/main/nn.html#torch.nn.MaxPool2d to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::MaxPool2dOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// MaxPool2d model(MaxPool2dOptions(3).stride(2));
/// ```py
/// Base class for all (dimension-specialized) adaptive maxpool modules.
template <size_t D, typename output_size_t, typename Derived>
class TORCH_API AdaptiveMaxPoolImpl : public torch::nn::Cloneable<Derived> {
 public:
  /// Constructor initializing the adaptive maxpool module with the specified output size.
  AdaptiveMaxPoolImpl(output_size_t output_size)
      : AdaptiveMaxPoolImpl(
            AdaptiveMaxPoolOptions<output_size_t>(output_size)) {}
  
  /// Explicit constructor initializing the adaptive maxpool module with the provided options.
  explicit AdaptiveMaxPoolImpl(
      const AdaptiveMaxPoolOptions<output_size_t>& options_)
      : options(options_) {}

  /// Resets the module to its initial state.
  void reset() override{};

  /// Pretty prints the `AdaptiveMaxPool{1,2,3}d` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override {
    // 将字符串 "torch::nn::AdaptiveMaxPool" 与维度 D 拼接，构建池化层名称的字符串表示
    stream << "torch::nn::AdaptiveMaxPool" << D << "d"
           << "(output_size=" << options.output_size() << ")";
  }

  // 模块构造时使用的选项
  AdaptiveMaxPoolOptions<output_size_t> options;
// };
// 这是一个注释，标记了一个单独的代码行，可能是被注释掉的或者未完成的代码片段。

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~ AdaptiveMaxPool1d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// AdaptiveMaxPool1d 类的定义，应用于对 1-D 输入执行自适应最大池化操作。
// 参见 https://pytorch.org/docs/main/nn.html#torch.nn.AdaptiveMaxPool1d 以了解此模块的确切行为。
// 
// 参见 `torch::nn::AdaptiveMaxPool1dOptions` 类的文档以了解此模块支持的构造函数参数。
// 
// 示例：
// ```
// AdaptiveMaxPool1d model(AdaptiveMaxPool1dOptions(3));
// ```py
class TORCH_API AdaptiveMaxPool1dImpl
    : public AdaptiveMaxPoolImpl<1, ExpandingArray<1>, AdaptiveMaxPool1dImpl> {
 public:
  using AdaptiveMaxPoolImpl<1, ExpandingArray<1>, AdaptiveMaxPool1dImpl>::
      AdaptiveMaxPoolImpl;

  // 前向传播函数，接受一个 Tensor 输入，返回一个 Tensor 输出
  Tensor forward(const Tensor& input);

  /// Returns the indices along with the outputs.
  /// Useful to pass to nn.MaxUnpool1d.
  // 返回带有索引的输出元组，对于 nn.MaxUnpool1d 很有用。
  std::tuple<Tensor, Tensor> forward_with_indices(const Tensor& input);
};

/// `AdaptiveMaxPool1dImpl` 的 `ModuleHolder` 子类。
/// 参见 `AdaptiveMaxPool1dImpl` 类的文档，了解它提供的方法，以及如何使用 `torch::nn::AdaptiveMaxPool1dOptions` 使用 `AdaptiveMaxPool1d` 的示例。
/// 参见 `ModuleHolder` 的文档，了解 PyTorch 的模块存储语义。
TORCH_MODULE(AdaptiveMaxPool1d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ AdaptiveMaxPool2d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// AdaptiveMaxPool2d 类的定义，应用于对 2-D 输入执行自适应最大池化操作。
// 参见 https://pytorch.org/docs/main/nn.html#torch.nn.AdaptiveMaxPool2d 以了解此模块的确切行为。
// 
// 参见 `torch::nn::AdaptiveMaxPool2dOptions` 类的文档以了解此模块支持的构造函数参数。
// 
// 示例：
// ```
// AdaptiveMaxPool2d model(AdaptiveMaxPool2dOptions({3, 2}));
// ```py
class TORCH_API AdaptiveMaxPool2dImpl : public AdaptiveMaxPoolImpl<
                                            2,
                                            ExpandingArrayWithOptionalElem<2>,
                                            AdaptiveMaxPool2dImpl> {
 public:
  using AdaptiveMaxPoolImpl<
      2,
      ExpandingArrayWithOptionalElem<2>,
      AdaptiveMaxPool2dImpl>::AdaptiveMaxPoolImpl;

  // 前向传播函数，接受一个 Tensor 输入，返回一个 Tensor 输出
  Tensor forward(const Tensor& input);

  /// Returns the indices along with the outputs.
  /// Useful to pass to nn.MaxUnpool2d.
  // 返回带有索引的输出元组，对于 nn.MaxUnpool2d 很有用。
  std::tuple<Tensor, Tensor> forward_with_indices(const Tensor& input);
};

/// `AdaptiveMaxPool2dImpl` 的 `ModuleHolder` 子类。
/// 参见 `AdaptiveMaxPool2dImpl` 类的文档，了解它提供的方法，以及如何使用 `torch::nn::AdaptiveMaxPool2dOptions` 使用 `AdaptiveMaxPool2d` 的示例。
/// 参见 `ModuleHolder` 的文档，了解 PyTorch 的模块存储语义。
TORCH_MODULE(AdaptiveMaxPool2d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ AdaptiveMaxPool3d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Applies adaptive maxpool over a 3-D input.
// 对 3-D 输入执行自适应最大池化操作。
/// See https://pytorch.org/docs/main/nn.html#torch.nn.AdaptiveMaxPool3d to
/// learn about the exact behavior of this module.
///
/// See the documentation for `torch::nn::AdaptiveMaxPool3dOptions` class to
/// learn what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// AdaptiveMaxPool3d model(AdaptiveMaxPool3dOptions(3));
/// ```py
class TORCH_API AdaptiveMaxPool3dImpl : public AdaptiveMaxPoolImpl<
                                            3,
                                            ExpandingArrayWithOptionalElem<3>,
                                            AdaptiveMaxPool3dImpl> {
 public:
  using AdaptiveMaxPoolImpl<
      3,
      ExpandingArrayWithOptionalElem<3>,
      AdaptiveMaxPool3dImpl>::AdaptiveMaxPoolImpl;

  /// Performs the forward pass of the AdaptiveMaxPool3d module.
  Tensor forward(const Tensor& input);

  /// Performs the forward pass of the AdaptiveMaxPool3d module and returns
  /// the indices along with the outputs. Useful for nn.MaxUnpool3d.
  std::tuple<Tensor, Tensor> forward_with_indices(const Tensor& input);
};

/// A `ModuleHolder` subclass for `AdaptiveMaxPool3dImpl`.
/// See the documentation for `AdaptiveMaxPool3dImpl` class to learn what
/// methods it provides, and examples of how to use `AdaptiveMaxPool3d` with
/// `torch::nn::AdaptiveMaxPool3dOptions`. See the documentation for
/// `ModuleHolder` to learn about PyTorch's module storage semantics.
TORCH_MODULE(AdaptiveMaxPool3d);

// ============================================================================

/// Base class for all (dimension-specialized) adaptive avgpool modules.
template <size_t D, typename output_size_t, typename Derived>
class TORCH_API AdaptiveAvgPoolImpl : public torch::nn::Cloneable<Derived> {
 public:
  /// Constructs an AdaptiveAvgPool module with the specified output size.
  AdaptiveAvgPoolImpl(output_size_t output_size)
      : AdaptiveAvgPoolImpl(
            AdaptiveAvgPoolOptions<output_size_t>(output_size)) {}
  explicit AdaptiveAvgPoolImpl(
      const AdaptiveAvgPoolOptions<output_size_t>& options_)
      : options(options_) {}

  void reset() override {}

  /// Pretty prints the `AdaptiveAvgPool{1,2,3}d` module into the given
  /// `stream`.
  void pretty_print(std::ostream& stream) const override {
    stream << "torch::nn::AdaptiveAvgPool" << D << "d"
           << "(output_size=" << options.output_size() << ")";
  }

  /// The options with which this `Module` was constructed.
  AdaptiveAvgPoolOptions<output_size_t> options;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~ AdaptiveAvgPool1d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies adaptive avgpool over a 1-D input.
/// See https://pytorch.org/docs/main/nn.html#torch.nn.AdaptiveAvgPool1d to
/// learn about the exact behavior of this module.
///
/// See the documentation for `torch::nn::AdaptiveAvgPool1dOptions` class to
/// learn what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// AdaptiveAvgPool1d model(AdaptiveAvgPool1dOptions(5));
/// ```py
class TORCH_API AdaptiveAvgPool1dImpl
    : public AdaptiveAvgPoolImpl<1, ExpandingArray<1>, AdaptiveAvgPool1dImpl> {


// 继承自 AdaptiveAvgPoolImpl 类模板，参数为 <1, ExpandingArray<1>, AdaptiveAvgPool1dImpl>
public:



  using AdaptiveAvgPoolImpl<1, ExpandingArray<1>, AdaptiveAvgPool1dImpl>::
      AdaptiveAvgPoolImpl;


// 使用基类 AdaptiveAvgPoolImpl 的构造函数作为本类构造函数的一部分
using AdaptiveAvgPoolImpl<1, ExpandingArray<1>, AdaptiveAvgPool1dImpl>::AdaptiveAvgPoolImpl;



  Tensor forward(const Tensor& input);


// 前向传播函数声明，接收一个 Tensor 类型的输入参数 input，并返回一个 Tensor 类型的结果
Tensor forward(const Tensor& input);
};

/// A `ModuleHolder` subclass for `AdaptiveAvgPool1dImpl`.
/// See the documentation for `AdaptiveAvgPool1dImpl` class to learn what
/// methods it provides, and examples of how to use `AdaptiveAvgPool1d` with
/// `torch::nn::AdaptiveAvgPool1dOptions`. See the documentation for
/// `ModuleHolder` to learn about PyTorch's module storage semantics.
TORCH_MODULE(AdaptiveAvgPool1d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~ AdaptiveAvgPool2d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies adaptive avgpool over a 2-D input.
/// See https://pytorch.org/docs/main/nn.html#torch.nn.AdaptiveAvgPool2d to
/// learn about the exact behavior of this module.
///
/// See the documentation for `torch::nn::AdaptiveAvgPool2dOptions` class to
/// learn what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// AdaptiveAvgPool2d model(AdaptiveAvgPool2dOptions({3, 2}));
/// ```py
class TORCH_API AdaptiveAvgPool2dImpl : public AdaptiveAvgPoolImpl<
                                            2,
                                            ExpandingArrayWithOptionalElem<2>,
                                            AdaptiveAvgPool2dImpl> {
 public:
  using AdaptiveAvgPoolImpl<
      2,
      ExpandingArrayWithOptionalElem<2>,
      AdaptiveAvgPool2dImpl>::AdaptiveAvgPoolImpl;

  /// Forward pass of the module.
  /// Computes adaptive average pooling on the input tensor.
  /// Returns the output tensor.
  Tensor forward(const Tensor& input);
};

/// A `ModuleHolder` subclass for `AdaptiveAvgPool2dImpl`.
/// See the documentation for `AdaptiveAvgPool2dImpl` class to learn what
/// methods it provides, and examples of how to use `AdaptiveAvgPool2d` with
/// `torch::nn::AdaptiveAvgPool2dOptions`. See the documentation for
/// `ModuleHolder` to learn about PyTorch's module storage semantics.
TORCH_MODULE(AdaptiveAvgPool2d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~ AdaptiveAvgPool3d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies adaptive avgpool over a 3-D input.
/// See https://pytorch.org/docs/main/nn.html#torch.nn.AdaptiveAvgPool3d to
/// learn about the exact behavior of this module.
///
/// See the documentation for `torch::nn::AdaptiveAvgPool3dOptions` class to
/// learn what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// AdaptiveAvgPool3d model(AdaptiveAvgPool3dOptions(3));
/// ```py
class TORCH_API AdaptiveAvgPool3dImpl : public AdaptiveAvgPoolImpl<
                                            3,
                                            ExpandingArrayWithOptionalElem<3>,
                                            AdaptiveAvgPool3dImpl> {
 public:
  using AdaptiveAvgPoolImpl<
      3,
      ExpandingArrayWithOptionalElem<3>,
      AdaptiveAvgPool3dImpl>::AdaptiveAvgPoolImpl;

  /// Forward pass of the module.
  /// Computes adaptive average pooling on the input tensor.
  /// Returns the output tensor.
  Tensor forward(const Tensor& input);
};

/// A `ModuleHolder` subclass for `AdaptiveAvgPool3dImpl`.
/// See the documentation for `AdaptiveAvgPool3dImpl` class to learn what
/// methods it provides, and examples of how to use `AdaptiveAvgPool3d` with
/// `torch::nn::AdaptiveAvgPool3dOptions`. See the documentation for
/// `ModuleHolder` to learn about PyTorch's module storage semantics.
TORCH_MODULE(AdaptiveAvgPool3d);
/// 定义一个 TORCH_MODULE，用于封装 AdaptiveAvgPool3d 模块。
TORCH_MODULE(AdaptiveAvgPool3d);

// ============================================================================

/// 用于所有（维度特定的）MaxUnpool模块的基类。
template <size_t D, typename Derived>
class TORCH_API MaxUnpoolImpl : public torch::nn::Cloneable<Derived> {
 public:
  /// 构造函数，根据给定的 kernel_size 创建 MaxUnpoolImpl 实例。
  MaxUnpoolImpl(ExpandingArray<D> kernel_size)
      : MaxUnpoolImpl(MaxUnpoolOptions<D>(kernel_size)) {}
  /// 显式构造函数，根据指定的选项创建 MaxUnpoolImpl 实例。
  explicit MaxUnpoolImpl(const MaxUnpoolOptions<D>& options_);

  /// 重置模块状态。
  void reset() override;

  /// 将 MaxUnpool{1,2,3}d 模块的信息打印到指定的流中。
  void pretty_print(std::ostream& stream) const override;

  /// 此模块的构造选项。
  MaxUnpoolOptions<D> options;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MaxUnpool1d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// 对1维输入应用 maxunpool。
/// 查看 https://pytorch.org/docs/main/nn.html#torch.nn.MaxUnpool1d 以了解此模块的确切行为。
///
/// 查看 torch::nn::MaxUnpool1dOptions 类的文档，了解此模块支持的构造参数。
///
/// 示例：
/// ```
/// MaxUnpool1d model(MaxUnpool1dOptions(3).stride(2).padding(1));
/// ```py
class TORCH_API MaxUnpool1dImpl : public MaxUnpoolImpl<1, MaxUnpool1dImpl> {
 public:
  using MaxUnpoolImpl<1, MaxUnpool1dImpl>::MaxUnpoolImpl;

  /// 前向传播函数，接受输入、索引和可选的输出尺寸参数，返回张量。
  Tensor forward(
      const Tensor& input,
      const Tensor& indices,
      const std::optional<std::vector<int64_t>>& output_size = c10::nullopt);

 protected:
  /// 定义前向传播函数的默认参数，第2个参数是可选的输出尺寸。
  FORWARD_HAS_DEFAULT_ARGS({2, AnyValue(std::optional<std::vector<int64_t>>())})
};

/// MaxUnpool1dImpl 的 ModuleHolder 子类。
/// 查看 MaxUnpool1dImpl 类的文档，了解它提供的方法，以及如何使用 torch::nn::MaxUnpool1dOptions 使用 MaxUnpool1d。
/// 查看 ModuleHolder 的文档，了解 PyTorch 的模块存储语义。
TORCH_MODULE(MaxUnpool1d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MaxUnpool2d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// 对2维输入应用 maxunpool。
/// 查看 https://pytorch.org/docs/main/nn.html#torch.nn.MaxUnpool2d 以了解此模块的确切行为。
///
/// 查看 torch::nn::MaxUnpool2dOptions 类的文档，了解此模块支持的构造参数。
///
/// 示例：
/// ```
/// MaxUnpool2d model(MaxUnpool2dOptions(3).stride(2).padding(1));
/// ```py
class TORCH_API MaxUnpool2dImpl : public MaxUnpoolImpl<2, MaxUnpool2dImpl> {
 public:
  using MaxUnpoolImpl<2, MaxUnpool2dImpl>::MaxUnpoolImpl;

  /// 前向传播函数，接受输入、索引和可选的输出尺寸参数，返回张量。
  Tensor forward(
      const Tensor& input,
      const Tensor& indices,
      const std::optional<std::vector<int64_t>>& output_size = c10::nullopt);

 protected:
  /// 定义前向传播函数的默认参数，第2个参数是可选的输出尺寸。
  FORWARD_HAS_DEFAULT_ARGS({2, AnyValue(std::optional<std::vector<int64_t>>())})
};

/// MaxUnpool2dImpl 的 ModuleHolder 子类。
/// 定义了一个名为 `MaxUnpool2d` 的 Torch 模块，用于执行 2D 最大反池化操作。
/// 参考 `MaxUnpool2dImpl` 类的文档以了解其提供的方法，以及如何使用 `torch::nn::MaxUnpool2dOptions`。
/// 参考 `ModuleHolder` 的文档以了解 PyTorch 模块存储语义。
TORCH_MODULE(MaxUnpool2d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MaxUnpool3d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// 在 3D 输入上应用最大反池化操作。
/// 参考 https://pytorch.org/docs/main/nn.html#torch.nn.MaxUnpool3d 以了解此模块的精确行为。
///
/// 参考 `torch::nn::MaxUnpool3dOptions` 类的文档以了解为此模块支持的构造参数。
///
/// 示例：
/// ```
/// MaxUnpool3d model(MaxUnpool3dOptions(3).stride(2).padding(1));
/// ```py
class TORCH_API MaxUnpool3dImpl : public MaxUnpoolImpl<3, MaxUnpool3dImpl> {
 public:
  using MaxUnpoolImpl<3, MaxUnpool3dImpl>::MaxUnpoolImpl;
  
  /// 执行前向传播，对输入、索引和可选的输出尺寸进行最大反池化操作。
  Tensor forward(
      const Tensor& input,
      const Tensor& indices,
      const std::optional<std::vector<int64_t>>& output_size = c10::nullopt);

 protected:
  /// 声明默认参数的前向传播方法。
  FORWARD_HAS_DEFAULT_ARGS({2, AnyValue(std::optional<std::vector<int64_t>>())})
};

/// `MaxUnpool3dImpl` 的 `ModuleHolder` 子类。
/// 参考 `MaxUnpool3dImpl` 类的文档以了解其提供的方法，以及如何使用 `torch::nn::MaxUnpool3dOptions`。
/// 参考 `ModuleHolder` 的文档以了解 PyTorch 模块存储语义。
TORCH_MODULE(MaxUnpool3d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FractionalMaxPool2d
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// 在 2D 输入上应用分数最大池化操作。
/// 参考 https://pytorch.org/docs/main/nn.html#torch.nn.FractionalMaxPool2d 以了解此模块的精确行为。
///
/// 参考 `torch::nn::FractionalMaxPool2dOptions` 类的文档以了解为此模块支持的构造参数。
///
/// 示例：
/// ```
/// FractionalMaxPool2d model(FractionalMaxPool2dOptions(5).output_size(1));
/// ```py
class TORCH_API FractionalMaxPool2dImpl
    : public torch::nn::Cloneable<FractionalMaxPool2dImpl> {
 public:
  /// 使用给定的核大小构造 `FractionalMaxPool2dImpl`。
  FractionalMaxPool2dImpl(ExpandingArray<2> kernel_size)
      : FractionalMaxPool2dImpl(FractionalMaxPool2dOptions(kernel_size)) {}
  
  /// 使用指定的选项构造 `FractionalMaxPool2dImpl`。
  explicit FractionalMaxPool2dImpl(FractionalMaxPool2dOptions options_);

  /// 重置模块状态。
  void reset() override;

  /// 将 `FractionalMaxPool2d` 模块美观地打印到给定的 `stream` 中。
  void pretty_print(std::ostream& stream) const override;

  /// 执行前向传播，对输入进行分数最大池化操作。
  Tensor forward(const Tensor& input);

  /// 返回最大值的输出和索引。
  /// 这对于稍后使用的 `torch::nn::MaxUnpool2d` 非常有用。
  std::tuple<Tensor, Tensor> forward_with_indices(const Tensor& input);

  /// 构造此 `Module` 的选项。
  FractionalMaxPool2dOptions options;

  /// 内部变量，用于存储随机样本。
  Tensor _random_samples;
};

/// `FractionalMaxPool2dImpl` 的 `ModuleHolder` 子类。
///
/// 在 `ModuleHolder` 的文档中了解 PyTorch 模块存储语义。
/// 定义一个名为 `FractionalMaxPool2d` 的 Torch 模块，用于应用于二维输入的分数最大池化操作。
/// 请参阅 `FractionalMaxPool2dImpl` 类的文档，了解其提供的方法，以及使用 `torch::nn::FractionalMaxPool2dOptions` 与 `FractionalMaxPool2d` 结合使用的示例。
/// 请参阅 `ModuleHolder` 的文档，了解 PyTorch 模块存储语义。

TORCH_MODULE(FractionalMaxPool2d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FractionalMaxPool3d
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// 在三维输入上应用分数最大池化。
/// 请参阅 https://pytorch.org/docs/main/nn.html#torch.nn.FractionalMaxPool3d 了解此模块的确切行为。

/// 请参阅 `torch::nn::FractionalMaxPool3dOptions` 类的文档，了解此模块支持的构造函数参数。
///
/// 示例：
/// ```
/// FractionalMaxPool3d model(FractionalMaxPool3dOptions(5).output_size(1));
/// ```py
class TORCH_API FractionalMaxPool3dImpl
    : public torch::nn::Cloneable<FractionalMaxPool3dImpl> {
 public:
  /// 使用给定的核大小构造 `FractionalMaxPool3dImpl` 对象。
  FractionalMaxPool3dImpl(ExpandingArray<3> kernel_size)
      : FractionalMaxPool3dImpl(FractionalMaxPool3dOptions(kernel_size)) {}

  /// 使用指定选项构造 `FractionalMaxPool3dImpl` 对象。
  explicit FractionalMaxPool3dImpl(FractionalMaxPool3dOptions options_);

  /// 重置模块状态。
  void reset() override;

  /// 将 `FractionalMaxPool3d` 模块以可读形式打印到给定的流中。
  void pretty_print(std::ostream& stream) const override;

  /// 对输入数据执行前向传播。
  Tensor forward(const Tensor& input);

  /// 返回最大值的输出及其索引。
  /// 在后续使用 `torch::nn::MaxUnpool3d` 时非常有用。
  std::tuple<Tensor, Tensor> forward_with_indices(const Tensor& input);

  /// 构建此 `Module` 的选项。
  FractionalMaxPool3dOptions options;

  /// 用于存储随机样本的张量。
  Tensor _random_samples;
};

/// `FractionalMaxPool3dImpl` 的 `ModuleHolder` 子类。
/// 请参阅 `FractionalMaxPool3dImpl` 类的文档，了解其提供的方法，以及使用 `torch::nn::FractionalMaxPool3dOptions` 与 `FractionalMaxPool3d` 结合使用的示例。
/// 请参阅 `ModuleHolder` 的文档，了解 PyTorch 模块存储语义。
TORCH_MODULE(FractionalMaxPool3d);

// ============================================================================

/// 所有（维度专用）lppool 模块的基类。
template <size_t D, typename Derived>
class TORCH_API LPPoolImpl : public torch::nn::Cloneable<Derived> {
 public:
  /// 使用给定的范数类型和核大小构造 `LPPoolImpl` 对象。
  LPPoolImpl(double norm_type, ExpandingArray<D> kernel_size)
      : LPPoolImpl(LPPoolOptions<D>(norm_type, kernel_size)) {}

  /// 使用指定选项构造 `LPPoolImpl` 对象。
  explicit LPPoolImpl(const LPPoolOptions<D>& options_);

  /// 重置模块状态。
  void reset() override;

  /// 将 `LPPool{1,2}d` 模块以可读形式打印到给定的流中。
  void pretty_print(std::ostream& stream) const override;

  /// 构建此 `Module` 的选项。
  LPPoolOptions<D> options;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LPPool1d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// 对输入数据应用 LPPool1d 函数的逐元素操作。
/// 请参阅 https://pytorch.org/docs/main/nn.html#torch.nn.LPPool1d 了解此模块的确切行为。
///
/// 请参阅 `torch::nn::LPPool1dOptions` 类的文档，了解此模块支持的构造函数参数。
///
/// 示例：
/// ```
/// LPPool1d model(LPPool1dOptions(2.0, 3).stride(2));
/// ```py
/// See the documentation for `torch::nn::LPPool1dOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// LPPool1d model(LPPool1dOptions(1, 2).stride(5).ceil_mode(true));
/// ```py
class TORCH_API LPPool1dImpl : public LPPoolImpl<1, LPPool1dImpl> {
 public:
  /// Inherit constructors from LPPoolImpl<1, LPPool1dImpl>.
  using LPPoolImpl<1, LPPool1dImpl>::LPPoolImpl;

  /// Forward function definition for LPPool1d module.
  /// Computes the forward pass of the module.
  Tensor forward(const Tensor& input);
};

/// A `ModuleHolder` subclass for `LPPool1dImpl`.
/// Provides storage and management for LPPool1dImpl module instances.
/// See the documentation for `LPPool1dImpl` class for methods it provides,
/// and examples of usage with `torch::nn::LPPool1dOptions`.
/// See `ModuleHolder` documentation for PyTorch's module storage semantics.
TORCH_MODULE(LPPool1d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LPPool2d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies the LPPool2d function element-wise.
/// See https://pytorch.org/docs/main/nn.html#torch.nn.LPPool2d to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::LPPool2dOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// LPPool2d model(LPPool2dOptions(1, std::vector<int64_t>({3, 4})).stride({5,
/// 6}).ceil_mode(true));
/// ```py
class TORCH_API LPPool2dImpl : public LPPoolImpl<2, LPPool2dImpl> {
 public:
  /// Inherit constructors from LPPoolImpl<2, LPPool2dImpl>.
  using LPPoolImpl<2, LPPool2dImpl>::LPPoolImpl;

  /// Forward function definition for LPPool2d module.
  /// Computes the forward pass of the module.
  Tensor forward(const Tensor& input);
};

/// A `ModuleHolder` subclass for `LPPool2dImpl`.
/// Provides storage and management for LPPool2dImpl module instances.
/// See the documentation for `LPPool2dImpl` class for methods it provides,
/// and examples of usage with `torch::nn::LPPool2dOptions`.
/// See `ModuleHolder` documentation for PyTorch's module storage semantics.
TORCH_MODULE(LPPool2d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LPPool3d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies the LPPool3d function element-wise.
/// See https://pytorch.org/docs/main/nn.html#torch.nn.LPPool3d to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::LPPool3dOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// LPPool3d model(LPPool3dOptions(1, std::vector<int64_t>({3, 4, 5})).stride(
/// {5, 6, 7}).ceil_mode(true));
/// ```py
class TORCH_API LPPool3dImpl : public LPPoolImpl<3, LPPool3dImpl> {
 public:
  /// Inherit constructors from LPPoolImpl<3, LPPool3dImpl>.
  using LPPoolImpl<3, LPPool3dImpl>::LPPoolImpl;

  /// Forward function definition for LPPool3d module.
  /// Computes the forward pass of the module.
  Tensor forward(const Tensor& input);
};

/// A `ModuleHolder` subclass for `LPPool3dImpl`.
/// Provides storage and management for LPPool3dImpl module instances.
/// See the documentation for `LPPool3dImpl` class for methods it provides,
/// and examples of usage with `torch::nn::LPPool3dOptions`.
/// See `ModuleHolder` documentation for PyTorch's module storage semantics.
TORCH_MODULE(LPPool3d);

} // namespace nn
} // namespace torch
```