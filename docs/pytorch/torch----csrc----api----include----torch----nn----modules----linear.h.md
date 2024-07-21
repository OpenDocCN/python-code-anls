# `.\pytorch\torch\csrc\api\include\torch\nn\modules\linear.h`

```
#pragma once

#include <torch/nn/cloneable.h>  // 包含 Cloneable 类定义，用于模块克隆
#include <torch/nn/functional/linear.h>  // 包含线性函数相关的功能
#include <torch/nn/module.h>  // 包含 Module 类定义
#include <torch/nn/options/linear.h>  // 包含 LinearOptions 类定义，用于线性层选项配置
#include <torch/nn/pimpl.h>  // 包含 pimpl 模式的相关实现
#include <torch/types.h>  // 包含 Tensor 类型定义

#include <cstddef>  // 包含标准库的头文件
#include <vector>  // 包含标准库的向量类

namespace torch {
namespace nn {

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Identity ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// A placeholder identity operator that is argument-insensitive.
/// See https://pytorch.org/docs/main/generated/torch.nn.Identity.html to
/// learn about the exact behavior of this module.
/// 表示一个占位符的恒等运算符，不受参数影响

class TORCH_API IdentityImpl : public Cloneable<IdentityImpl> {
 public:
  void reset() override;  // 重置方法的声明

  /// Pretty prints the `Identity` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;  // 将 Identity 模块美化打印到给定的流中

  Tensor forward(const Tensor& input);  // 前向传播方法声明
};

/// A `ModuleHolder` subclass for `IdentityImpl`.
/// See the documentation for `IdentityImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
/// IdentityImpl 的模块持有器子类，用于存储 IdentityImpl 类的模块实例

TORCH_MODULE(Identity);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Linear ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies a linear transformation with optional bias.
/// See https://pytorch.org/docs/main/generated/torch.nn.Linear.html to learn
/// about the exact behavior of this module.
/// 应用具有可选偏置的线性变换

/// See the documentation for `torch::nn::LinearOptions` class to learn what
/// constructor arguments are supported for this module.
/// 查看 `torch::nn::LinearOptions` 类的文档，了解本模块支持的构造函数参数

/// Example:
/// ```
/// Linear model(LinearOptions(5, 2).bias(false));
/// ```
/// 示例：创建一个 Linear 模块实例，禁用偏置项

class TORCH_API LinearImpl : public Cloneable<LinearImpl> {
 public:
  LinearImpl(int64_t in_features, int64_t out_features)
      : LinearImpl(LinearOptions(in_features, out_features)) {}  // 构造函数声明，使用 LinearOptions 配置线性层

  explicit LinearImpl(const LinearOptions& options_);  // 显式构造函数声明，使用给定选项配置

  void reset() override;  // 重置方法的声明

  void reset_parameters();  // 重置参数方法的声明

  /// Pretty prints the `Linear` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;  // 将 Linear 模块美化打印到给定的流中

  /// Transforms the `input` tensor by multiplying with the `weight` and
  /// optionally adding the `bias`, if `with_bias` is true in the options.
  Tensor forward(const Tensor& input);  // 前向传播方法声明

  /// The options used to configure this module.
  LinearOptions options;  // 配置本模块所用的选项

  /// The learned weight.
  Tensor weight;  // 学习到的权重

  /// The learned bias. If `bias` is false in the `options`, this tensor is
  /// undefined.
  Tensor bias;  // 学习到的偏置，如果在选项中设置 `bias` 为 false，则此张量未定义
};

/// A `ModuleHolder` subclass for `LinearImpl`.
/// See the documentation for `LinearImpl` class to learn what methods it
/// provides, and examples of how to use `Linear` with
/// `torch::nn::LinearOptions`. See the documentation for `ModuleHolder` to
/// learn about PyTorch's module storage semantics.
/// LinearImpl 的模块持有器子类，用于存储 LinearImpl 类的模块实例

TORCH_MODULE(Linear);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Flatten ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// A placeholder for Flatten operator
/// See https://pytorch.org/docs/main/generated/torch.nn.Flatten.html to learn
/// about the exact behavior of this module.
/// 表示 Flatten 操作的占位符
/// Applies a bilinear transformation with optional bias.
/// See https://pytorch.org/docs/main/generated/torch.nn.Bilinear.html to
/// learn about the exact behavior of this module.
///
/// See the documentation for `torch::nn::BilinearOptions` class to learn what
/// constructor arguments are supported for this module.
class TORCH_API BilinearImpl : public Cloneable<BilinearImpl> {
 public:
  /// Constructs a Bilinear module with specified options.
  explicit BilinearImpl(const BilinearOptions& options_);

  /// Resets the module.
  void reset() override;

  /// Pretty prints the `Bilinear` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// Applies a bilinear transformation on the input.
  Tensor forward(const Tensor& input1, const Tensor& input2);

  /// The options used to configure this module.
  BilinearOptions options;
};

/// A `ModuleHolder` subclass for `BilinearImpl`.
/// See the documentation for `BilinearImpl` class to learn what methods it
/// provides, and examples of how to use `Bilinear` with
/// `torch::nn::BilinearOptions`. See the documentation for `ModuleHolder` to
/// learn about PyTorch's module storage semantics.
TORCH_MODULE(Bilinear);
/// BilinearImpl 类的实现，继承自 Cloneable，用于实现双线性模型。
/// 通过构造函数初始化双线性模型的参数，包括输入特征数和输出特征数。
/// 如果只提供输入特征数和输出特征数，内部会创建一个 BilinearOptions 对象并使用它来初始化。
class TORCH_API BilinearImpl : public Cloneable<BilinearImpl> {
 public:
  /// 构造函数，接受输入特征数、输出特征数，初始化 BilinearImpl 对象。
  BilinearImpl(int64_t in1_features, int64_t in2_features, int64_t out_features)
      : BilinearImpl(
            BilinearOptions(in1_features, in2_features, out_features)) {}

  /// 显式构造函数，通过 BilinearOptions 对象初始化 BilinearImpl。
  explicit BilinearImpl(const BilinearOptions& options_);

  /// 重置模型参数，继承自基类 Cloneable。
  void reset() override;

  /// 重置模型的学习参数，即权重和偏置。
  void reset_parameters();

  /// 将 Bilinear 模型的信息打印到给定的流 `stream` 中。
  void pretty_print(std::ostream& stream) const override;

  /// 前向传播函数，对输入的两个张量 input1 和 input2 执行双线性变换，
  /// 根据选项中的设置使用权重和偏置（如果存在）。
  Tensor forward(const Tensor& input1, const Tensor& input2);

  /// 用于配置该模块的选项。
  BilinearOptions options;

  /// 学习到的权重。
  Tensor weight;

  /// 学习到的偏置。如果在选项中设置了 `with_bias` 为 false，则该张量未定义。
  Tensor bias;
};

/// BilinearImpl 的 `ModuleHolder` 子类，用于管理 `BilinearImpl` 的实例。
/// 参考 `BilinearImpl` 类的文档以了解其提供的方法，以及如何使用 `torch::nn::BilinearOptions` 与 `Bilinear`。
/// 参考 `ModuleHolder` 的文档以了解 PyTorch 中的模块存储语义。
TORCH_MODULE(Bilinear);

} // namespace nn
} // namespace torch
```