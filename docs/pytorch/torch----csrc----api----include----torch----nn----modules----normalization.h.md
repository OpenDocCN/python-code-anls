# `.\pytorch\torch\csrc\api\include\torch\nn\modules\normalization.h`

```
#pragma once
// 预处理指令：确保头文件只被编译一次

#include <torch/nn/cloneable.h>
// 包含 PyTorch 的克隆相关头文件
#include <torch/nn/functional/normalization.h>
// 包含 PyTorch 的标准化功能相关头文件
#include <torch/nn/modules/_functions.h>
// 包含 PyTorch 模块功能相关的函数头文件
#include <torch/nn/options/normalization.h>
// 包含 PyTorch 标准化选项相关的头文件
#include <torch/nn/pimpl.h>
// 包含 PyTorch 的私有实现相关头文件
#include <torch/types.h>
// 包含 PyTorch 的数据类型相关头文件

#include <cstddef>
// 包含标准库的 cstddef 头文件，定义了一些常见的类型定义
#include <vector>
// 包含标准库的 vector 头文件，定义了向量容器类模板

namespace torch {
namespace nn {

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LayerNorm ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// 对输入的一个小批量数据应用层标准化，如`Layer Normalization`论文中所述。
/// 参见 https://pytorch.org/docs/main/nn.html#torch.nn.LayerNorm 了解此模块的确切行为。
///
/// 参见 `torch::nn::LayerNormOptions` 类的文档，了解此模块支持的构造函数参数。
///
/// 示例：
/// ```
/// LayerNorm model(LayerNormOptions({2,
/// 2}).elementwise_affine(false).eps(2e-5));
/// ```
class TORCH_API LayerNormImpl : public torch::nn::Cloneable<LayerNormImpl> {
 public:
  // 构造函数：根据给定的标准化形状构造 LayerNormImpl 对象
  LayerNormImpl(std::vector<int64_t> normalized_shape)
      : LayerNormImpl(LayerNormOptions(normalized_shape)) {}
  // 显式构造函数：根据给定的选项构造 LayerNormImpl 对象
  explicit LayerNormImpl(LayerNormOptions options_);

  // 重置函数，重写自父类 Cloneable 的方法
  void reset() override;

  // 重置参数函数：用于重置该模块的参数
  void reset_parameters();

  /// 将 `LayerNorm` 模块的信息漂亮地打印到给定的 `stream` 中。
  void pretty_print(std::ostream& stream) const override;

  /// 对一个小批量输入应用层标准化，如`Layer Normalization`论文中所述。
  ///
  /// 均值和标准差分别在最后的一定数量维度上分别计算，这些维度的形状由输入的 `normalized_shape` 指定。
  ///
  /// `Layer Normalization`: https://arxiv.org/abs/1607.06450
  Tensor forward(const Tensor& input);

  /// 构造此模块时使用的选项。
  LayerNormOptions options;

  /// 学习得到的权重。
  /// 如果构造时 `elementwise_affine` 选项设置为 `true`，则初始化为全1。
  Tensor weight;

  /// 学习得到的偏置。
  /// 如果构造时 `elementwise_affine` 选项设置为 `true`，则初始化为全0。
  Tensor bias;
};

/// `LayerNormImpl` 的 `ModuleHolder` 子类。
/// 参见 `LayerNormImpl` 类的文档，了解其提供的方法，以及如何使用 `LayerNorm` 和 `torch::nn::LayerNormOptions` 的示例。
/// 参见 `ModuleHolder` 的文档，了解 PyTorch 模块存储语义。
TORCH_MODULE(LayerNorm);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LocalResponseNorm
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// 对由多个输入平面组成的输入信号应用局部响应标准化，其中通道占据第二个维度。
/// 在通道间应用标准化。
/// 参见 https://pytorch.org/docs/main/nn.html#torch.nn.LocalResponseNorm 了解此模块的确切行为。
///
/// 参见 `torch::nn::LocalResponseNormOptions` 类的文档，了解此模块的行为。
/// learn what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// LocalResponseNorm
/// model(LocalResponseNormOptions(2).alpha(0.0002).beta(0.85).k(2.));
/// ```
class TORCH_API LocalResponseNormImpl
    : public Cloneable<LocalResponseNormImpl> {
 public:
  /// Constructor taking `size` argument and initializing with options.
  LocalResponseNormImpl(int64_t size)
      : LocalResponseNormImpl(LocalResponseNormOptions(size)) {}

  /// Explicit constructor initializing with provided `options_`.
  explicit LocalResponseNormImpl(const LocalResponseNormOptions& options_);

  /// Performs forward pass using input tensor `input`.
  Tensor forward(const Tensor& input);

  /// Resets the module state.
  void reset() override;

  /// Pretty prints the `LocalResponseNormImpl` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  LocalResponseNormOptions options;
};

/// A `ModuleHolder` subclass for `LocalResponseNormImpl`.
/// See the documentation for `LocalResponseNormImpl` class to learn what
/// methods it provides, and examples of how to use `LocalResponseNorm` with
/// `torch::nn::LocalResponseNormOptions`. See the documentation for
/// `ModuleHolder` to learn about PyTorch's module storage semantics.
TORCH_MODULE(LocalResponseNorm);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CrossMapLRN2d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// See the documentation for `torch::nn::CrossMapLRN2dOptions` class to learn
/// what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// CrossMapLRN2d model(CrossMapLRN2dOptions(3).alpha(1e-5).beta(0.1).k(10));
/// ```
class TORCH_API CrossMapLRN2dImpl
    : public torch::nn::Cloneable<CrossMapLRN2dImpl> {
 public:
  /// Constructor taking `size` argument and initializing with options.
  CrossMapLRN2dImpl(int64_t size)
      : CrossMapLRN2dImpl(CrossMapLRN2dOptions(size)) {}

  /// Explicit constructor initializing with provided `options_`.
  explicit CrossMapLRN2dImpl(const CrossMapLRN2dOptions& options_)
      : options(options_) {}

  /// Resets the module state.
  void reset() override;

  /// Pretty prints the `CrossMapLRN2d` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// Performs forward pass using input tensor `input`.
  torch::Tensor forward(const torch::Tensor& input);

  /// The options with which this `Module` was constructed.
  CrossMapLRN2dOptions options;
};

/// A `ModuleHolder` subclass for `CrossMapLRN2dImpl`.
/// See the documentation for `CrossMapLRN2dImpl` class to learn what methods it
/// provides, and examples of how to use `CrossMapLRN2d` with
/// `torch::nn::CrossMapLRN2dOptions`. See the documentation for `ModuleHolder`
/// to learn about PyTorch's module storage semantics.
TORCH_MODULE(CrossMapLRN2d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GroupNorm ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies Group Normalization over a mini-batch of inputs as described in
/// the paper `Group Normalization`_ .
/// See https://pytorch.org/docs/main/nn.html#torch.nn.GroupNorm to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::GroupNormOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// GroupNorm model(GroupNormOptions(2, 2).eps(2e-5).affine(false));
/// ```
/// 定义一个名为 GroupNormImpl 的类，它继承自 torch::nn::Cloneable<GroupNormImpl>
class TORCH_API GroupNormImpl : public torch::nn::Cloneable<GroupNormImpl> {
 public:
  /// 构造函数，接受两个参数：num_groups 表示分组数量，num_channels 表示通道数量
  GroupNormImpl(int64_t num_groups, int64_t num_channels)
      : GroupNormImpl(GroupNormOptions(num_groups, num_channels)) {}

  /// 显式构造函数，接受 GroupNormOptions 类型的参数 options_
  explicit GroupNormImpl(const GroupNormOptions& options_);

  /// 重置函数，继承自 Cloneable 接口，用于重置对象状态
  void reset() override;

  /// 重置参数函数，用于初始化权重和偏置
  void reset_parameters();

  /// 将 GroupNorm 模块的信息打印到指定的流对象 stream 中
  void pretty_print(std::ostream& stream) const override;

  /// 前向传播函数，接受输入张量 input，返回输出张量 Tensor
  Tensor forward(const Tensor& input);

  /// 记录 GroupNorm 模块构造时的选项
  GroupNormOptions options;

  /// 学习得到的权重张量
  Tensor weight;

  /// 学习得到的偏置张量
  Tensor bias;
};

/// `GroupNormImpl` 的 `ModuleHolder` 子类，用于管理 `GroupNormImpl` 的实例
/// 查阅 `GroupNormImpl` 类的文档以了解其提供的方法，以及如何使用 `torch::nn::GroupNormOptions` 使用 `GroupNorm`
/// 查阅 `ModuleHolder` 的文档以了解 PyTorch 的模块存储语义
TORCH_MODULE(GroupNorm);

} // namespace nn
} // namespace torch
```