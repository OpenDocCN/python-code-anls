# `.\pytorch\torch\csrc\api\include\torch\nn\modules\activation.h`

```
#pragma once

#include <torch/nn/cloneable.h>
#include <torch/nn/functional/activation.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/options/activation.h>

#include <torch/csrc/Export.h>

namespace torch {
namespace nn {

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ELU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies elu over a given input.
/// See https://pytorch.org/docs/main/nn.html#torch.nn.ELU to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::ELUOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// ELU model(ELUOptions().alpha(42.42).inplace(true));
/// ```
class TORCH_API ELUImpl : public torch::nn::Cloneable<ELUImpl> {
 public:
  /// Constructor for initializing ELU module with given options.
  explicit ELUImpl(const ELUOptions& options_ = {});

  /// Forward function for applying ELU activation to input tensor.
  Tensor forward(Tensor input);

  /// Reset function for resetting internal state, if any.
  void reset() override;

  /// Pretty prints the `ELU` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  ELUOptions options;
};

/// A `ModuleHolder` subclass for `ELUImpl`.
/// See the documentation for `ELUImpl` class to learn what methods it
/// provides, and examples of how to use `ELU` with `torch::nn::ELUOptions`.
/// See the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(ELU);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SELU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies the selu function element-wise.
/// See https://pytorch.org/docs/main/nn.html#torch.nn.SELU to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::SELUOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// SELU model(SELUOptions().inplace(true));
/// ```
class TORCH_API SELUImpl : public torch::nn::Cloneable<SELUImpl> {
 public:
  /// Constructor for initializing SELU module with given options.
  explicit SELUImpl(const SELUOptions& options_ = {});

  /// Forward function for applying SELU activation to input tensor.
  Tensor forward(Tensor input);

  /// Reset function for resetting internal state, if any.
  void reset() override;

  /// Pretty prints the `SELU` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  SELUOptions options;
};

/// A `ModuleHolder` subclass for `SELUImpl`.
/// See the documentation for `SELUImpl` class to learn what methods it
/// provides, and examples of how to use `SELU` with `torch::nn::SELUOptions`.
/// See the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(SELU);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Hardshrink ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies the hard shrinkage function element-wise.
/// See https://pytorch.org/docs/main/nn.html#torch.nn.Hardshrink to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::HardshrinkOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// Hardshrink model(HardshrinkOptions().lambda(0.5));
/// ```
class TORCH_API HardshrinkImpl : public torch::nn::Cloneable<HardshrinkImpl> {
 public:
  /// Constructor for initializing Hardshrink module with given options.
  explicit HardshrinkImpl(const HardshrinkOptions& options_ = {});

  /// Forward function for applying Hardshrink function to input tensor.
  Tensor forward(Tensor input);

  /// Reset function for resetting internal state, if any.
  void reset() override;

  /// Pretty prints the `Hardshrink` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  HardshrinkOptions options;
};

/// A `ModuleHolder` subclass for `HardshrinkImpl`.
/// See the documentation for `HardshrinkImpl` class to learn what methods it
/// provides, and examples of how to use `Hardshrink` with `torch::nn::HardshrinkOptions`.
/// See the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(Hardshrink);

} // namespace nn
} // namespace torch
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// Hardshrink model(HardshrinkOptions().lambda(42.42));
/// ```
class TORCH_API HardshrinkImpl : public torch::nn::Cloneable<HardshrinkImpl> {
 public:
  /// Constructor initializing Hardshrink with given options.
  explicit HardshrinkImpl(const HardshrinkOptions& options_ = {});

  /// Forward function for Hardshrink module.
  Tensor forward(const Tensor& input);

  /// Reset function override.
  void reset() override;

  /// Pretty prints the `Hardshrink` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  HardshrinkOptions options;
};

/// A `ModuleHolder` subclass for `HardshrinkImpl`.
/// See the documentation for `HardshrinkImpl` class to learn what methods it
/// provides, and examples of how to use `Hardshrink` with
/// `torch::nn::HardshrinkOptions`. See the documentation for `ModuleHolder` to
/// learn about PyTorch's module storage semantics.
TORCH_MODULE(Hardshrink);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Hardtanh ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies the HardTanh function element-wise.
/// See https://pytorch.org/docs/main/nn.html#torch.nn.Hardtanh to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::HardtanhOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// Hardtanh model(HardtanhOptions().min_val(-42.42).max_val(0.42).inplace(true));
/// ```
class TORCH_API HardtanhImpl : public torch::nn::Cloneable<HardtanhImpl> {
 public:
  /// Constructor initializing Hardtanh with given options.
  explicit HardtanhImpl(const HardtanhOptions& options_ = {});

  /// Forward function for Hardtanh module.
  Tensor forward(Tensor input);

  /// Reset function override.
  void reset() override;

  /// Pretty prints the `Hardtanh` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  HardtanhOptions options;
};

/// A `ModuleHolder` subclass for `HardtanhImpl`.
/// See the documentation for `HardtanhImpl` class to learn what methods it
/// provides, and examples of how to use `Hardtanh` with
/// `torch::nn::HardtanhOptions`. See the documentation for `ModuleHolder` to
/// learn about PyTorch's module storage semantics.
TORCH_MODULE(Hardtanh);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LeakyReLU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies the LeakyReLU function element-wise.
/// See https://pytorch.org/docs/main/nn.html#torch.nn.LeakyReLU to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::LeakyReLUOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// LeakyReLU model(LeakyReLUOptions().negative_slope(0.42).inplace(true));
/// ```
/// 定义 LeakyReLU 激活函数模块的实现，继承自 `torch::nn::Cloneable<LeakyReLUImpl>`。
/// 此类提供了 LeakyReLU 模块的前向传播功能和重置方法。
class TORCH_API LeakyReLUImpl : public torch::nn::Cloneable<LeakyReLUImpl> {
 public:
  /// 构造函数，允许指定 LeakyReLUOptions 对象的选项。
  explicit LeakyReLUImpl(const LeakyReLUOptions& options_ = {});

  /// 实现前向传播功能，接受一个 Tensor 作为输入，并返回相应的输出 Tensor。
  Tensor forward(Tensor input);

  /// 重置方法的实现，用于重置模块的状态。
  void reset() override;

  /// 将 `LeakyReLU` 模块的详细信息漂亮地打印到给定的 `stream` 中。
  void pretty_print(std::ostream& stream) const override;

  /// 此 `Module` 构造时使用的选项。
  LeakyReLUOptions options;
};

/// `LeakyReLUImpl` 的 `ModuleHolder` 子类。
/// 请参阅 `LeakyReLUImpl` 类的文档，了解其提供的方法及如何使用 `torch::nn::LeakyReLUOptions`。
/// 请参阅 `ModuleHolder` 的文档，了解 PyTorch 的模块存储语义。
TORCH_MODULE(LeakyReLU);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LogSigmoid ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// 应用 LogSigmoid 函数的元素级操作。
/// 请参阅 https://pytorch.org/docs/main/nn.html#torch.nn.LogSigmoid 了解此模块的确切行为。
class TORCH_API LogSigmoidImpl : public torch::nn::Cloneable<LogSigmoidImpl> {
 public:
  /// 实现前向传播功能，接受一个 Tensor 作为输入，并返回相应的输出 Tensor。
  Tensor forward(const Tensor& input);

  /// 重置方法的实现，用于重置模块的状态。
  void reset() override;

  /// 将 `LogSigmoid` 模块的详细信息漂亮地打印到给定的 `stream` 中。
  void pretty_print(std::ostream& stream) const override;
};

/// `LogSigmoidImpl` 的 `ModuleHolder` 子类。
/// 请参阅 `LogSigmoidImpl` 类的文档，了解其提供的方法及 PyTorch 的模块存储语义。
TORCH_MODULE(LogSigmoid);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Softmax ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// 应用 Softmax 函数。
/// 请参阅 https://pytorch.org/docs/main/nn.html#torch.nn.Softmax 了解此模块的确切行为。
///
/// 参阅 `torch::nn::SoftmaxOptions` 类的文档，了解此模块支持的构造函数参数。
///
/// 示例：
/// ```
/// Softmax model(SoftmaxOptions(1));
/// ```
class TORCH_API SoftmaxImpl : public torch::nn::Cloneable<SoftmaxImpl> {
 public:
  /// 构造函数，允许指定 SoftmaxOptions 对象的选项。
  explicit SoftmaxImpl(int64_t dim) : SoftmaxImpl(SoftmaxOptions(dim)) {}
  explicit SoftmaxImpl(const SoftmaxOptions& options_);

  /// 实现前向传播功能，接受一个 Tensor 作为输入，并返回相应的输出 Tensor。
  Tensor forward(const Tensor& input);

  /// 重置方法的实现，用于重置模块的状态。
  void reset() override;

  /// 将 `Softmax` 模块的详细信息漂亮地打印到给定的 `stream` 中。
  void pretty_print(std::ostream& stream) const override;

  /// 此 `Module` 构造时使用的选项。
  SoftmaxOptions options;
};

/// `SoftmaxImpl` 的 `ModuleHolder` 子类。
/// 请参阅 `SoftmaxImpl` 类的文档，了解其提供的方法及如何使用 `torch::nn::SoftmaxOptions`。
/// 请参阅 `ModuleHolder` 的文档，了解 PyTorch 的模块存储语义。
TORCH_MODULE(Softmax);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Softmin ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// 应用 Softmin 函数的元素级操作。
/// 请参阅 https://pytorch.org/docs/main/nn.html#torch.nn.Softmin 了解此模块的详细行为。
/// Applies the Softmin function element-wise.
/// See https://pytorch.org/docs/main/nn.html#torch.nn.Softmin to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::SoftminOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// Softmin model(SoftminOptions(1));
/// ```
class TORCH_API SoftminImpl : public torch::nn::Cloneable<SoftminImpl> {
 public:
  /// Constructor that initializes the Softmin module with a specified dimension.
  explicit SoftminImpl(int64_t dim) : SoftminImpl(SoftminOptions(dim)) {}

  /// Constructor that initializes the Softmin module with specified options.
  explicit SoftminImpl(const SoftminOptions& options_);

  /// Performs the forward pass of the Softmin module.
  Tensor forward(const Tensor& input);

  /// Resets the module state.
  void reset() override;

  /// Pretty prints the `Softmin` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// Options object storing configuration options for this module.
  SoftminOptions options;
};

/// A `ModuleHolder` subclass for `SoftminImpl`.
/// See the documentation for `SoftminImpl` class to learn what methods it
/// provides, and examples of how to use `Softmin` with
/// `torch::nn::SoftminOptions`. See the documentation for `ModuleHolder` to
/// learn about PyTorch's module storage semantics.
TORCH_MODULE(Softmin);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LogSoftmax ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies the LogSoftmax function element-wise.
/// See https://pytorch.org/docs/main/nn.html#torch.nn.LogSoftmax to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::LogSoftmaxOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// LogSoftmax model(LogSoftmaxOptions(1));
/// ```
class TORCH_API LogSoftmaxImpl : public torch::nn::Cloneable<LogSoftmaxImpl> {
 public:
  /// Constructor that initializes the LogSoftmax module with a specified dimension.
  explicit LogSoftmaxImpl(int64_t dim)
      : LogSoftmaxImpl(LogSoftmaxOptions(dim)) {}

  /// Constructor that initializes the LogSoftmax module with specified options.
  explicit LogSoftmaxImpl(const LogSoftmaxOptions& options_);

  /// Performs the forward pass of the LogSoftmax module.
  Tensor forward(const Tensor& input);

  /// Resets the module state.
  void reset() override;

  /// Pretty prints the `LogSoftmax` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// Options object storing configuration options for this module.
  LogSoftmaxOptions options;
};

/// A `ModuleHolder` subclass for `LogSoftmaxImpl`.
/// See the documentation for `LogSoftmaxImpl` class to learn what methods it
/// provides, and examples of how to use `LogSoftmax` with
/// `torch::nn::LogSoftmaxOptions`. See the documentation for `ModuleHolder` to
/// learn about PyTorch's module storage semantics.
TORCH_MODULE(LogSoftmax);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Softmax2d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies the Softmax2d function element-wise.
/// See https://pytorch.org/docs/main/nn.html#torch.nn.Softmax2d to learn
/// about the exact behavior of this module.
class TORCH_API Softmax2dImpl : public torch::nn::Cloneable<Softmax2dImpl> {
 public:
  /// Performs the forward pass of the Softmax2d module.
  Tensor forward(const Tensor& input);

  /// Resets the module state.
  void reset() override;

  /// Pretty prints the `Softmax2d` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;
};

/// A `ModuleHolder` subclass for `Softmax2dImpl`.
/// See the documentation for `Softmax2dImpl` class to learn what methods it
// 定义了一个名为 Softmax2d 的宏，可能用于注册 PyTorch 模块
TORCH_MODULE(Softmax2d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PReLU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// 应用 PReLU 函数进行逐元素操作。
/// 参考 https://pytorch.org/docs/main/nn.html#torch.nn.PReLU 了解此模块的详细行为。
///
/// 参考 torch::nn::PReLUOptions 类的文档以了解此模块支持的构造函数参数。
///
/// 示例：
/// ```
/// PReLU model(PReLUOptions().num_parameters(42));
/// ```
class TORCH_API PReLUImpl : public torch::nn::Cloneable<PReLUImpl> {
 public:
  // 默认构造函数，使用给定的选项进行初始化
  explicit PReLUImpl(const PReLUOptions& options_ = {});

  // 前向传播函数，接受输入张量并返回结果张量
  Tensor forward(const Tensor& input);

  // 重置函数，重写自父类以实现重置操作
  void reset() override;

  // 将 PReLU 模块的信息输出到指定的流中
  void pretty_print(std::ostream& stream) const override;

  // 此模块构造时使用的选项
  PReLUOptions options;

  // 学习得到的权重张量
  Tensor weight;
};

/// 用于 PReLUImpl 的 ModuleHolder 子类。
/// 参考 PReLUImpl 类的文档以了解它提供的方法，并查看如何使用 `torch::nn::PReLUOptions` 与 `PReLU`。
/// 参考 ModuleHolder 的文档了解 PyTorch 的模块存储语义。
TORCH_MODULE(PReLU);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ReLU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// 应用 ReLU 函数进行逐元素操作。
/// 参考 https://pytorch.org/docs/main/nn.html#torch.nn.ReLU 了解此模块的详细行为。
///
/// 参考 torch::nn::ReLUOptions 类的文档以了解此模块支持的构造函数参数。
///
/// 示例：
/// ```
/// ReLU model(ReLUOptions().inplace(true));
/// ```
class TORCH_API ReLUImpl : public torch::nn::Cloneable<ReLUImpl> {
 public:
  // 默认构造函数，使用给定的选项进行初始化
  explicit ReLUImpl(const ReLUOptions& options_ = {});

  // 前向传播函数，接受输入张量并返回结果张量
  Tensor forward(Tensor input);

  // 重置函数，重写自父类以实现重置操作
  void reset() override;

  // 将 ReLU 模块的信息输出到指定的流中
  void pretty_print(std::ostream& stream) const override;

  // 此模块构造时使用的选项
  ReLUOptions options;
};

/// 用于 ReLUImpl 的 ModuleHolder 子类。
/// 参考 ReLUImpl 类的文档以了解它提供的方法，并查看如何使用 `torch::nn::ReLUOptions` 与 `ReLU`。
/// 参考 ModuleHolder 的文档了解 PyTorch 的模块存储语义。
TORCH_MODULE(ReLU);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ReLU6 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// 应用 ReLU6 函数进行逐元素操作。
/// 参考 https://pytorch.org/docs/main/nn.html#torch.nn.ReLU6 了解此模块的详细行为。
///
/// 参考 torch::nn::ReLU6Options 类的文档以了解此模块支持的构造函数参数。
///
/// 示例：
/// ```
/// ReLU6 model(ReLU6Options().inplace(true));
/// ```
/// ```
/// 定义了一个名为 ReLU6Impl 的类，它是 torch::nn::Cloneable<ReLU6Impl> 的子类
class TORCH_API ReLU6Impl : public torch::nn::Cloneable<ReLU6Impl> {
 public:
  /// 构造函数，接受 ReLU6Options 参数，并初始化对象
  explicit ReLU6Impl(const ReLU6Options& options_ = {});

  /// 前向传播函数，接受一个 Tensor 类型的输入并返回一个 Tensor
  Tensor forward(Tensor input);

  /// 重置函数，重写自父类 Cloneable 的方法
  void reset() override;

  /// 将 ReLU6 模块的信息漂亮地打印到给定的流中
  void pretty_print(std::ostream& stream) const override;

  /// 构造该模块时使用的选项
  ReLU6Options options;
};

/// `ReLU6Impl` 的 `ModuleHolder` 子类
/// 参见 `ReLU6Impl` 类的文档，了解其提供的方法以及如何使用 `torch::nn::ReLU6Options` 配置 `ReLU6`
/// 参见 `ModuleHolder` 的文档，了解 PyTorch 的模块存储语义
TORCH_MODULE(ReLU6);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ RReLU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// 对输入应用 RReLU 函数，逐元素地操作
/// 参见 https://pytorch.org/docs/main/nn.html#torch.nn.RReLU 以了解该模块的确切行为
///
/// 查看 `torch::nn::RReLUOptions` 类的文档，了解支持该模块的构造函数参数
///
/// 示例：
/// ```
/// RReLU model(RReLUOptions().lower(0.24).upper(0.42).inplace(true));
/// ```
class TORCH_API RReLUImpl : public torch::nn::Cloneable<RReLUImpl> {
 public:
  /// 构造函数，接受 RReLUOptions 参数，并初始化对象
  explicit RReLUImpl(const RReLUOptions& options_ = {});

  /// 前向传播函数，接受一个 Tensor 类型的输入并返回一个 Tensor
  Tensor forward(Tensor input);

  /// 重置函数，重写自父类 Cloneable 的方法
  void reset() override;

  /// 将 RReLU 模块的信息漂亮地打印到给定的流中
  void pretty_print(std::ostream& stream) const override;

  /// 构造该模块时使用的选项
  RReLUOptions options;
};

/// `RReLUImpl` 的 `ModuleHolder` 子类
/// 参见 `RReLUImpl` 类的文档，了解其提供的方法以及如何使用 `torch::nn::RReLUOptions` 配置 `RReLU`
/// 参见 `ModuleHolder` 的文档，了解 PyTorch 的模块存储语义
TORCH_MODULE(RReLU);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CELU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// 对给定的输入应用 CELU 函数
/// 参见 https://pytorch.org/docs/main/nn.html#torch.nn.CELU 以了解该模块的确切行为
///
/// 查看 `torch::nn::CELUOptions` 类的文档，了解支持该模块的构造函数参数
///
/// 示例：
/// ```
/// CELU model(CELUOptions().alpha(42.42).inplace(true));
/// ```
class TORCH_API CELUImpl : public torch::nn::Cloneable<CELUImpl> {
 public:
  /// 构造函数，接受 CELUOptions 参数，并初始化对象
  explicit CELUImpl(const CELUOptions& options_ = {});

  /// 前向传播函数，接受一个 Tensor 类型的输入并返回一个 Tensor
  Tensor forward(Tensor input);

  /// 重置函数，重写自父类 Cloneable 的方法
  void reset() override;

  /// 将 CELU 模块的信息漂亮地打印到给定的流中
  void pretty_print(std::ostream& stream) const override;

  /// 构造该模块时使用的选项
  CELUOptions options;
};

/// `CELUImpl` 的 `ModuleHolder` 子类
/// 参见 `CELUImpl` 类的文档，了解其提供的方法以及如何使用 `torch::nn::CELUOptions` 配置 `CELU`
/// 定义了一个 `TORCH_MODULE` 宏，用于将当前类声明为一个 Torch 模块。
TORCH_MODULE(CELU);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GLU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// `GLU` 类的实现，继承自 `torch::nn::Cloneable`，实现了 `GLU` 激活函数的功能。
/// 参见 https://pytorch.org/docs/main/nn.html#torch.nn.GLU 了解该模块的详细行为。
/// 参见 `torch::nn::GLUOptions` 类的文档以了解如何使用构造函数参数。
///
/// 示例:
/// ```
/// GLU model(GLUOptions(1));
/// ```
class TORCH_API GLUImpl : public torch::nn::Cloneable<GLUImpl> {
 public:
  explicit GLUImpl(const GLUOptions& options_ = {});

  /// 前向传播函数，接收输入张量并返回处理后的张量。
  Tensor forward(const Tensor& input);

  void reset() override;

  /// 将 `GLU` 模块的信息格式化输出到指定的流中。
  void pretty_print(std::ostream& stream) const override;

  /// 构建该模块时使用的选项。
  GLUOptions options;
};

/// `ModuleHolder` 的子类，用于持有 `GLUImpl` 类。
/// 参见 `GLUImpl` 类的文档以了解其提供的方法，以及如何使用 `torch::nn::GLUOptions` 创建 `GLU` 模块。
/// 参见 `ModuleHolder` 的文档了解 PyTorch 的模块存储语义。
TORCH_MODULE(GLU);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GELU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// `GELU` 类的实现，继承自 `torch::nn::Cloneable`，实现了 `GELU` 激活函数的功能。
/// 参见 https://pytorch.org/docs/main/nn.html#torch.nn.GELU 了解该模块的详细行为。
class TORCH_API GELUImpl : public torch::nn::Cloneable<GELUImpl> {
 public:
  explicit GELUImpl(GELUOptions options_ = {});

  /// 前向传播函数，接收输入张量并返回处理后的张量。
  Tensor forward(const Tensor& input);

  void reset() override;

  /// 将 `GELU` 模块的信息格式化输出到指定的流中。
  void pretty_print(std::ostream& stream) const override;

  /// 构建该模块时使用的选项。
  GELUOptions options;
};

/// `ModuleHolder` 的子类，用于持有 `GELUImpl` 类。
/// 参见 `GELUImpl` 类的文档了解其提供的方法，或者参见 `ModuleHolder` 的文档了解 PyTorch 的模块存储语义。
TORCH_MODULE(GELU);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SiLU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// `SiLU` 类的实现，继承自 `torch::nn::Cloneable`，实现了 `SiLU` 激活函数的功能。
/// 参见 https://pytorch.org/docs/main/nn.html#torch.nn.SiLU 了解该模块的详细行为。
class TORCH_API SiLUImpl : public torch::nn::Cloneable<SiLUImpl> {
 public:
  /// 前向传播函数，接收输入张量并返回处理后的张量。
  Tensor forward(const Tensor& input);

  void reset() override;

  /// 将 `SiLU` 模块的信息格式化输出到指定的流中。
  void pretty_print(std::ostream& stream) const override;
};

/// `ModuleHolder` 的子类，用于持有 `SiLUImpl` 类。
/// 参见 `SiLUImpl` 类的文档了解其提供的方法，或者参见 `ModuleHolder` 的文档了解 PyTorch 的模块存储语义。
TORCH_MODULE(SiLU);
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Mish ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Mish 函数的实现，作用于给定的输入。
/// 查看 https://pytorch.org/docs/main/nn.html#torch.nn.Mish 以了解此模块的确切行为。
class TORCH_API MishImpl : public torch::nn::Cloneable<MishImpl> {
 public:
  // 前向传播函数，接受一个张量输入，并返回一个张量
  Tensor forward(const Tensor& input);

  // 重置函数，覆盖父类的重置行为
  void reset() override;

  /// 将 `Mish` 模块在给定的 `stream` 中漂亮地打印出来。
  void pretty_print(std::ostream& stream) const override;
};

/// `MishImpl` 的 `ModuleHolder` 子类。
/// 参见 `MishImpl` 类的文档以了解它提供了哪些方法，
/// 或者参见 `ModuleHolder` 的文档以了解 PyTorch 的模块存储语义。
TORCH_MODULE(Mish);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Sigmoid ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// 对给定的输入应用 sigmoid 函数。
/// 查看 https://pytorch.org/docs/main/nn.html#torch.nn.Sigmoid 以了解此模块的确切行为。
class TORCH_API SigmoidImpl : public torch::nn::Cloneable<SigmoidImpl> {
 public:
  // 前向传播函数，接受一个张量输入，并返回一个张量
  Tensor forward(const Tensor& input);

  // 重置函数，覆盖父类的重置行为
  void reset() override;

  /// 将 `Sigmoid` 模块在给定的 `stream` 中漂亮地打印出来。
  void pretty_print(std::ostream& stream) const override;
};

/// `SigmoidImpl` 的 `ModuleHolder` 子类。
/// 参见 `SigmoidImpl` 类的文档以了解它提供了哪些方法，
/// 或者参见 `ModuleHolder` 的文档以了解 PyTorch 的模块存储语义。
TORCH_MODULE(Sigmoid);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Softplus ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// 对给定的输入应用 softplus 函数。
/// 查看 https://pytorch.org/docs/main/nn.html#torch.nn.Softplus 以了解此模块的确切行为。
///
/// 参见 `torch::nn::SoftplusOptions` 类的文档以了解此模块支持哪些构造参数。
///
/// 示例：
/// ```
/// Softplus model(SoftplusOptions().beta(0.24).threshold(42.42));
/// ```
class TORCH_API SoftplusImpl : public torch::nn::Cloneable<SoftplusImpl> {
 public:
  // 构造函数，使用给定的选项构建 Softplus 模块
  explicit SoftplusImpl(const SoftplusOptions& options_ = {});

  // 前向传播函数，接受一个张量输入，并返回一个张量
  Tensor forward(const Tensor& input);

  // 重置函数，覆盖父类的重置行为
  void reset() override;

  /// 将 `Softplus` 模块在给定的 `stream` 中漂亮地打印出来。
  void pretty_print(std::ostream& stream) const override;

  /// 构建此 `Module` 的选项。
  SoftplusOptions options;
};

/// `SoftplusImpl` 的 `ModuleHolder` 子类。
/// 参见 `SoftplusImpl` 类的文档以了解它提供了哪些方法，
/// 以及如何使用 `torch::nn::SoftplusOptions` 与 `Softplus`。
/// 参见 `ModuleHolder` 的文档以了解 PyTorch 的模块存储语义。
TORCH_MODULE(Softplus);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Softshrink ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// 对每个元素应用软收缩函数。
/// 查看 https://pytorch.org/docs/main/nn.html#torch.nn.Softshrink 以了解
/// 此模块的确切行为。
/// Applies Tanhshrink over a given input.
/// See https://pytorch.org/docs/main/nn.html#torch.nn.Tanhshrink to learn
/// about the exact behavior of this module.
class TORCH_API TanhshrinkImpl : public torch::nn::Cloneable<TanhshrinkImpl> {
 public:
  /// Constructor for TanhshrinkImpl.
  explicit TanhshrinkImpl();

  /// Forward function for Tanhshrink.
  /// Applies Tanhshrink activation to the input tensor.
  Tensor forward(const Tensor& input);

  /// Reset function override.
  void reset() override;

  /// Pretty prints the `Tanhshrink` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;
};

/// A `ModuleHolder` subclass for `TanhshrinkImpl`.
/// See the documentation for `TanhshrinkImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(Tanhshrink);
: public torch::nn::Cloneable<MultiheadAttentionImpl> {
 public:
  MultiheadAttentionImpl(int64_t embed_dim, int64_t num_heads,
                         const MultiheadAttentionOptions& options_);

  /// Performs a forward pass of MultiheadAttention.
  ///
  /// Arguments:
  ///   input: Input tensor of shape (seq_length, batch_size, embed_dim)
  ///   key_padding_mask: Optional mask tensor of shape (batch_size, seq_length)
  ///                     containing valid positions (0) and invalid positions (1).
  ///                     Default: None.
  ///   attn_mask: Optional mask tensor of shape (batch_size * num_heads, seq_length, seq_length)
  ///              containing attention mask for previous positions (1) and future positions (0).
  ///              Default: None.
  ///
  /// Returns:
  ///   A tuple containing:
  ///     - Output tensor of shape (seq_length, batch_size, embed_dim).
  ///     - Attention weights tensor of shape (batch_size, num_heads, seq_length, seq_length).
  std::tuple<Tensor, Tensor> forward(const Tensor& input,
                                     const Tensor& key_padding_mask = {},
                                     const Tensor& attn_mask = {});

  void reset() override;

  /// Pretty prints the `MultiheadAttention` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  MultiheadAttentionOptions options;
};

/// A `ModuleHolder` subclass for `MultiheadAttentionImpl`.
/// See the documentation for `MultiheadAttentionImpl` class to learn what
/// methods it provides, and examples of how to use `MultiheadAttention` with
/// `torch::nn::MultiheadAttentionOptions`. See the documentation for `ModuleHolder` to
/// learn about PyTorch's module storage semantics.
TORCH_MODULE(MultiheadAttention);
  : public torch::nn::Cloneable<MultiheadAttentionImpl> {
 public:
  // 构造函数，初始化多头注意力机制的实现
  MultiheadAttentionImpl(int64_t embed_dim, int64_t num_heads)
      : MultiheadAttentionImpl(
            MultiheadAttentionOptions(embed_dim, num_heads)) {}

  // 显式构造函数，接受 MultiheadAttentionOptions 对象作为参数初始化
  explicit MultiheadAttentionImpl(const MultiheadAttentionOptions& options_);

  // 前向传播函数，计算多头注意力机制的前向传播
  std::tuple<Tensor, Tensor> forward(
      const Tensor& query,
      const Tensor& key,
      const Tensor& value,
      const Tensor& key_padding_mask = {},
      bool need_weights = true,
      const Tensor& attn_mask = {},
      bool average_attn_weights = true);

 protected:
  // 宏定义，指定前向传播函数的默认参数
  FORWARD_HAS_DEFAULT_ARGS(
      {3, AnyValue(Tensor())},
      {4, AnyValue(true)},
      {5, AnyValue(Tensor())},
      {6, AnyValue(true)})

 public:
  // 重置函数，重置多头注意力机制的状态
  void reset() override;

  // 内部函数，重置参数
  void _reset_parameters();

  /// The options with which this `Module` was constructed.
  // 用于构造该模块的选项
  MultiheadAttentionOptions options;

  // 是否查询、键、值使用相同的嵌入维度
  bool _qkv_same_embed_dim;

  // 输入投影的权重和偏置
  Tensor in_proj_weight;
  Tensor in_proj_bias;

  // 键的偏置和值的偏置
  Tensor bias_k;
  Tensor bias_v;

  // 输出投影层，用于投影多头注意力输出
  Linear out_proj = nullptr;

  // 查询、键、值的投影权重
  Tensor q_proj_weight;
  Tensor k_proj_weight;
  Tensor v_proj_weight;

  // 头部维度
  int64_t head_dim;
};

/// 结束了 `nn` 命名空间的定义

/// `MultiheadAttentionImpl` 的 `ModuleHolder` 子类。
/// 查看 `MultiheadAttentionImpl` 类的文档以了解它提供的方法，以及如何使用 `MultiheadAttention` 与 `torch::nn::MultiheadAttentionOptions` 的示例。
/// 查看 `ModuleHolder` 的文档以了解 PyTorch 模块存储语义。
TORCH_MODULE(MultiheadAttention);

} // namespace nn
} // namespace torch
```