# `.\pytorch\torch\csrc\api\include\torch\nn\modules\padding.h`

```py
#pragma once
// 预处理指令，确保本文件只被编译一次

#include <torch/expanding_array.h>
// 引入torch库中的expanding_array模块

#include <torch/nn/cloneable.h>
// 引入torch库中的cloneable模块，用于实现可克隆的神经网络模块

#include <torch/nn/functional/padding.h>
// 引入torch库中的padding模块，用于实现填充功能

#include <torch/csrc/Export.h>
// 引入torch库中的Export模块，处理导出符号的宏

namespace torch {
namespace nn {

/// Base class for all (dimension-specialized) ReflectionPad modules.
// 所有（针对特定维度的）反射填充模块的基类模板
template <size_t D, typename Derived>
class TORCH_API ReflectionPadImpl : public torch::nn::Cloneable<Derived> {
 public:
  // 构造函数，初始化反射填充选项
  ReflectionPadImpl(ExpandingArray<D * 2> padding)
      : ReflectionPadImpl(ReflectionPadOptions<D>(padding)) {}
  
  // 显式构造函数，根据指定选项初始化反射填充模块
  explicit ReflectionPadImpl(const ReflectionPadOptions<D>& options_);

  // 重置函数，重写自父类Cloneable的方法，但未提供具体实现
  void reset() override;

  // 前向传播函数，接受输入张量并返回处理后的张量
  Tensor forward(const Tensor& input);

  /// Pretty prints the `ReflectionPad{1,2}d` module into the given `stream`.
  // 将反射填充模块美化输出到给定流stream中
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  // 此模块构造时的选项
  ReflectionPadOptions<D> options;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ReflectionPad1d
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies ReflectionPad over a 1-D input.
// 对1维输入应用反射填充

/// See https://pytorch.org/docs/main/nn.html#torch.nn.ReflectionPad1d to
/// learn about the exact behavior of this module.
// 查看链接了解此模块的确切行为

/// See the documentation for `torch::nn::ReflectionPad1dOptions` class to learn
/// what constructor arguments are supported for this module.
// 查看torch::nn::ReflectionPad1dOptions类的文档，了解此模块支持的构造参数

/// Example:
/// ```
/// ReflectionPad1d model(ReflectionPad1dOptions({3, 1}));
/// ```py
// 示例：创建一个ReflectionPad1d模型，使用ReflectionPad1dOptions指定参数

class TORCH_API ReflectionPad1dImpl
    : public ReflectionPadImpl<1, ReflectionPad1dImpl> {
 public:
  using ReflectionPadImpl<1, ReflectionPad1dImpl>::ReflectionPadImpl;
  // 使用基类ReflectionPadImpl的构造函数

};

/// A `ModuleHolder` subclass for `ReflectionPad1dImpl`.
// ReflectionPad1dImpl的ModuleHolder子类

/// See the documentation for `ReflectionPad1dImpl` class to learn what methods
/// it provides, and examples of how to use `ReflectionPad1d` with
/// `torch::nn::ReflectionPad1dOptions`. See the documentation for
/// `ModuleHolder` to learn about PyTorch's module storage semantics.
// 查看ReflectionPad1dImpl类的文档，了解其提供的方法以及如何使用ReflectionPad1d和torch::nn::ReflectionPad1dOptions
// 查看ModuleHolder的文档，了解PyTorch的模块存储语义

TORCH_MODULE(ReflectionPad1d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ReflectionPad2d
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies ReflectionPad over a 2-D input.
// 对2维输入应用反射填充

/// See https://pytorch.org/docs/main/nn.html#torch.nn.ReflectionPad2d to
/// learn about the exact behavior of this module.
// 查看链接了解此模块的确切行为

/// See the documentation for `torch::nn::ReflectionPad2dOptions` class to learn
/// what constructor arguments are supported for this module.
// 查看torch::nn::ReflectionPad2dOptions类的文档，了解此模块支持的构造参数

/// Example:
/// ```
/// ReflectionPad2d model(ReflectionPad2dOptions({1, 1, 2, 0}));
/// ```py
// 示例：创建一个ReflectionPad2d模型，使用ReflectionPad2dOptions指定参数

class TORCH_API ReflectionPad2dImpl
    : public ReflectionPadImpl<2, ReflectionPad2dImpl> {
 public:
  using ReflectionPadImpl<2, ReflectionPad2dImpl>::ReflectionPadImpl;
  // 使用基类ReflectionPadImpl的构造函数

};

/// A `ModuleHolder` subclass for `ReflectionPad2dImpl`.
// ReflectionPad2dImpl的ModuleHolder子类

/// See the documentation for `ReflectionPad2dImpl` class to learn what methods
/// it provides, and examples of how to use `ReflectionPad2d` with
/// `torch::nn::ReflectionPad2dOptions`. See the documentation for
/// `ModuleHolder` to learn about PyTorch's module storage semantics.
// 查看ReflectionPad2dImpl类的文档，了解其提供的方法以及如何使用ReflectionPad2d和torch::nn::ReflectionPad2dOptions
// 查看ModuleHolder的文档，了解PyTorch的模块存储语义

TORCH_MODULE(ReflectionPad2d);

} // namespace nn
} // namespace torch
/// Applies ReplicationPad over a 2-D input.
/// See https://pytorch.org/docs/main/nn.html#torch.nn.ReplicationPad2d to
/// learn about the exact behavior of this module.
///
/// See the documentation for `torch::nn::ReplicationPad2dOptions` class to
/// learn what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// ReplicationPad2d model(ReplicationPad2dOptions({1, 1, 2, 2}));
/// ```py
class TORCH_API ReplicationPad2dImpl
    : public ReplicationPadImpl<2, ReplicationPad2dImpl> {
 public:
  using ReplicationPadImpl<2, ReplicationPad2dImpl>::ReplicationPadImpl;
};

/// A `ModuleHolder` subclass for `ReplicationPad2dImpl`.
/// See the documentation for `ReplicationPad2dImpl` class to learn what methods
/// it provides, and examples of how to use `ReplicationPad2d` with
/// `torch::nn::ReplicationPad2dOptions`. See the documentation for
/// `ModuleHolder` to learn about PyTorch's module storage semantics.
TORCH_MODULE(ReplicationPad2d);
/// Base class for all (dimension-specialized) ZeroPad modules.
template <size_t D, typename Derived>
class TORCH_API ZeroPadImpl : public torch::nn::Cloneable<Derived> {
 public:
  // 构造函数，接受一个 D*2 大小的数组作为参数，用于设置填充大小
  ZeroPadImpl(ExpandingArray<D * 2> padding)
      : ZeroPadImpl(ZeroPadOptions<D>(padding)) {}
  
  // 显式构造函数，接受 ZeroPadOptions<D> 对象作为参数
  explicit ZeroPadImpl(const ZeroPadOptions<D>& options_);

  // 重置函数，覆盖基类中的 reset 方法
  void reset() override;

  // 前向传播函数，接受一个 Tensor 输入并返回一个 Tensor 输出
  Tensor forward(const Tensor& input);

  /// Pretty prints the `ZeroPad{1,2}d` module into the given `stream`.
  // 在给定的流中漂亮打印 `ZeroPad{1,2}d` 模块的函数

  // 漂亮打印函数，覆盖基类中的 pretty_print 方法
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  // 此 `Module` 构造时使用的选项
  ZeroPadOptions<D> options;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ZeroPad1d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 应用于 1-D 输入的 ZeroPad 模块。
// ============================================================================

/// Base class for all (dimension-specialized) ConstantPad modules.
template <size_t D, typename Derived>
class TORCH_API ConstantPadImpl : public torch::nn::Cloneable<Derived> {
 public:
  // Constructor initializing ConstantPad options with padding and value
  ConstantPadImpl(ExpandingArray<D * 2> padding, double value)
      : ConstantPadImpl(ConstantPadOptions<D>(padding, value)) {}
  // Explicit constructor initializing ConstantPad options with given options
  explicit ConstantPadImpl(const ConstantPadOptions<D>& options_);

  // Reset function override
  void reset() override;

  // Forward function for performing the padding operation on input tensor
  Tensor forward(const Tensor& input);

  // Pretty prints the ConstantPad{1,2}d module into the given stream
  void pretty_print(std::ostream& stream) const override;

  // The options with which this Module was constructed
  ConstantPadOptions<D> options;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ConstantPad1d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies ConstantPad over a 1-D input.
/// See https://pytorch.org/docs/main/nn.html#torch.nn.ConstantPad1d to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::ConstantPad1dOptions` class to learn
/// what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// ConstantPad1d model(ConstantPad1dOptions({3, 1}, 3.5));
/// ```py
class TORCH_API ConstantPad1dImpl : public ConstantPadImpl<1, ConstantPad1dImpl> {
 public:
  using ConstantPadImpl<1, ConstantPad1dImpl>::ConstantPadImpl;
};
    // 声明一个公共类 ConstantPad1dImpl，继承自 ConstantPadImpl<1, ConstantPad1dImpl>
    // 使用 using 关键字继承 ConstantPadImpl<1, ConstantPad1dImpl> 类的构造函数和成员函数
    : public ConstantPadImpl<1, ConstantPad1dImpl> {
    public:
      // 使用基类 ConstantPadImpl<1, ConstantPad1dImpl> 的构造函数
      using ConstantPadImpl<1, ConstantPad1dImpl>::ConstantPadImpl;
};

/// A `ModuleHolder` subclass for `ConstantPad1dImpl`.
/// See the documentation for `ConstantPad1dImpl` class to learn what methods it
/// provides, and examples of how to use `ConstantPad1d` with
/// `torch::nn::ConstantPad1dOptions`. See the documentation for `ModuleHolder`
/// to learn about PyTorch's module storage semantics.
TORCH_MODULE(ConstantPad1d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ConstantPad2d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies ConstantPad over a 2-D input.
/// See https://pytorch.org/docs/main/nn.html#torch.nn.ConstantPad2d to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::ConstantPad2dOptions` class to learn
/// what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// ConstantPad2d model(ConstantPad2dOptions({3, 0, 2, 1}, 3.5));
/// ```py
class TORCH_API ConstantPad2dImpl
    : public ConstantPadImpl<2, ConstantPad2dImpl> {
 public:
  using ConstantPadImpl<2, ConstantPad2dImpl>::ConstantPadImpl;
};

/// A `ModuleHolder` subclass for `ConstantPad2dImpl`.
/// See the documentation for `ConstantPad2dImpl` class to learn what methods it
/// provides, and examples of how to use `ConstantPad2d` with
/// `torch::nn::ConstantPad2dOptions`. See the documentation for `ModuleHolder`
/// to learn about PyTorch's module storage semantics.
TORCH_MODULE(ConstantPad2d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ConstantPad3d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies ConstantPad over a 3-D input.
/// See https://pytorch.org/docs/main/nn.html#torch.nn.ConstantPad3d to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::ConstantPad3dOptions` class to learn
/// what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// ConstantPad3d model(ConstantPad3dOptions({1, 2, 1, 2, 1, 2}, 3.5));
/// ```py
class TORCH_API ConstantPad3dImpl
    : public ConstantPadImpl<3, ConstantPad3dImpl> {
 public:
  using ConstantPadImpl<3, ConstantPad3dImpl>::ConstantPadImpl;
};

/// A `ModuleHolder` subclass for `ConstantPad3dImpl`.
/// See the documentation for `ConstantPad3dImpl` class to learn what methods it
/// provides, and examples of how to use `ConstantPad3d` with
/// `torch::nn::ConstantPad3dOptions`. See the documentation for `ModuleHolder`
/// to learn about PyTorch's module storage semantics.
TORCH_MODULE(ConstantPad3d);

} // namespace nn
} // namespace torch
```