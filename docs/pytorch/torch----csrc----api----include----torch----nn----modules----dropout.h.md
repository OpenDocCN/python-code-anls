# `.\pytorch\torch\csrc\api\include\torch\nn\modules\dropout.h`

```
#pragma once
// 预处理指令，确保头文件只被编译一次

#include <torch/nn/cloneable.h>
// 引入 PyTorch 中 Cloneable 类的头文件，用于支持模块的克隆

#include <torch/nn/options/dropout.h>
// 引入 PyTorch 中 DropoutOptions 类的头文件，定义了 Dropout 模块的选项

#include <torch/nn/pimpl.h>
// 引入 PyTorch 中 pimpl 模式的头文件，用于实现类的内部指针

#include <torch/types.h>
// 引入 PyTorch 中 Tensor 类型的头文件，定义了张量的相关操作

#include <torch/csrc/Export.h>
// 引入 PyTorch 的导出声明头文件，用于指定导出库的接口

#include <cstddef>
// 引入标准库头文件，定义了 size_t 类型等

#include <vector>
// 引入标准库头文件，定义了 vector 类型等

namespace torch {
namespace nn {

namespace detail {

template <typename Derived>
// 模板类 _DropoutNd，继承自 Cloneable 类
class _DropoutNd : public torch::nn::Cloneable<Derived> {
 public:
  _DropoutNd(double p) : _DropoutNd(DropoutOptions().p(p)){};
  // 构造函数，接受 dropout 概率参数 p，使用 DropoutOptions 设置选项

  explicit _DropoutNd(const DropoutOptions& options_ = {}) : options(options_) {
    // 显示声明构造函数，接受 DropoutOptions 对象作为参数，设置选项
    // NOLINTNEXTLINE(clang-analyzer-optin.cplusplus.VirtualCall)
    reset();
    // 重置函数，确保 dropout 概率在合理范围内
  }

  void reset() override {
    // 重置函数，覆盖基类方法
    TORCH_CHECK(
        options.p() >= 0. && options.p() <= 1.,
        "dropout probability has to be between 0 and 1, but got ",
        options.p());
    // 检查 dropout 概率是否在合理范围内，否则抛出异常
  }

  /// The options with which this `Module` was constructed.
  // 此模块构造时的选项
  DropoutOptions options;
};

} // namespace detail

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Dropout ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies dropout over a 1-D input.
/// See https://pytorch.org/docs/main/nn.html#torch.nn.Dropout to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::DropoutOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// Dropout model(DropoutOptions().p(0.42).inplace(true));
/// ```
// Dropout 类，应用于 1-D 输入的 dropout 操作
class TORCH_API DropoutImpl : public detail::_DropoutNd<DropoutImpl> {
 public:
  using detail::_DropoutNd<DropoutImpl>::_DropoutNd;
  // 使用基类 _DropoutNd 的构造函数

  Tensor forward(Tensor input);
  // 前向传播函数，接受输入张量并返回张量

  /// Pretty prints the `Dropout` module into the given `stream`.
  // 将 Dropout 模块美观地打印到给定流中
  void pretty_print(std::ostream& stream) const override;
  // 覆盖基类方法，实现美化打印
};

/// A `ModuleHolder` subclass for `DropoutImpl`.
/// See the documentation for `DropoutImpl` class to learn what methods it
/// provides, and examples of how to use `Dropout` with
/// `torch::nn::DropoutOptions`. See the documentation for `ModuleHolder` to
/// learn about PyTorch's module storage semantics.
// DropoutImpl 的 ModuleHolder 子类，用于存储 DropoutImpl 模块实例
TORCH_MODULE(Dropout);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Dropout2d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies dropout over a 2-D input.
/// See https://pytorch.org/docs/main/nn.html#torch.nn.Dropout2d to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::Dropout2dOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// Dropout2d model(Dropout2dOptions().p(0.42).inplace(true));
/// ```
// Dropout2d 类，应用于 2-D 输入的 dropout 操作
class TORCH_API Dropout2dImpl : public detail::_DropoutNd<Dropout2dImpl> {
 public:
  using detail::_DropoutNd<Dropout2dImpl>::_DropoutNd;
  // 使用基类 _DropoutNd 的构造函数

  Tensor forward(Tensor input);
  // 前向传播函数，接受输入张量并返回张量

  /// Pretty prints the `Dropout2d` module into the given `stream`.
  // 将 Dropout2d 模块美观地打印到给定流中
  void pretty_print(std::ostream& stream) const override;
  // 覆盖基类方法，实现美化打印
};

/// A `ModuleHolder` subclass for `Dropout2dImpl`.
/// See the documentation for `Dropout2dImpl` class to learn what methods it
/// provides, and examples of how to use `Dropout2d` with
/// `torch::nn::Dropout2dOptions`.
// Dropout2dImpl 的 ModuleHolder 子类，用于存储 Dropout2dImpl 模块实例
// 定义一个名为 `Dropout2d` 的 Torch 模块，使用了 `torch::nn::Dropout2dOptions`。
// 查看 `ModuleHolder` 的文档以了解 PyTorch 模块存储语义。
TORCH_MODULE(Dropout2d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Dropout3d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// 应用于三维输入的 dropout。
/// 查看 https://pytorch.org/docs/main/nn.html#torch.nn.Dropout3d 以了解此模块的确切行为。
///
/// 查看 `torch::nn::Dropout3dOptions` 类的文档以了解此模块支持的构造函数参数。
///
/// 示例：
/// ```
/// Dropout3d model(Dropout3dOptions().p(0.42).inplace(true));
/// ```
class TORCH_API Dropout3dImpl : public detail::_DropoutNd<Dropout3dImpl> {
 public:
  using detail::_DropoutNd<Dropout3dImpl>::_DropoutNd;

  // 对输入进行前向传播
  Tensor forward(Tensor input);

  /// 将 `Dropout3d` 模块美化打印到给定的 `stream` 中。
  void pretty_print(std::ostream& stream) const override;
};

/// 用于 `Dropout3dImpl` 的 `ModuleHolder` 子类。
/// 查看 `Dropout3dImpl` 类的文档以了解它提供的方法，以及如何使用 `torch::nn::Dropout3dOptions` 使用 `Dropout3d`。
/// 查看 `ModuleHolder` 的文档以了解 PyTorch 模块存储语义。
TORCH_MODULE(Dropout3d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ AlphaDropout ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// 对输入应用 Alpha Dropout。
/// 查看 https://pytorch.org/docs/main/nn.html#torch.nn.AlphaDropout 以了解此模块的确切行为。
///
/// 查看 `torch::nn::AlphaDropoutOptions` 类的文档以了解此模块支持的构造函数参数。
///
/// 示例：
/// ```
/// AlphaDropout model(AlphaDropoutOptions(0.2).inplace(true));
/// ```
class TORCH_API AlphaDropoutImpl : public detail::_DropoutNd<AlphaDropoutImpl> {
 public:
  using detail::_DropoutNd<AlphaDropoutImpl>::_DropoutNd;

  // 对输入进行前向传播
  Tensor forward(const Tensor& input);

  /// 将 `AlphaDropout` 模块美化打印到给定的 `stream` 中。
  void pretty_print(std::ostream& stream) const override;
};

/// 用于 `AlphaDropoutImpl` 的 `ModuleHolder` 子类。
/// 查看 `AlphaDropoutImpl` 类的文档以了解它提供的方法，以及如何使用 `torch::nn::AlphaDropoutOptions` 使用 `AlphaDropout`。
/// 查看 `ModuleHolder` 的文档以了解 PyTorch 模块存储语义。
TORCH_MODULE(AlphaDropout);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FeatureAlphaDropout
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// 查看 `torch::nn::FeatureAlphaDropoutOptions` 类的文档以了解此模块支持的构造函数参数。
///
/// 示例：
/// ```
/// FeatureAlphaDropout model(FeatureAlphaDropoutOptions(0.2).inplace(true));
/// ```
class TORCH_API FeatureAlphaDropoutImpl
    : public detail::_DropoutNd<FeatureAlphaDropoutImpl> {
 public:
  using detail::_DropoutNd<FeatureAlphaDropoutImpl>::_DropoutNd;

  // 继承自 _DropoutNd<FeatureAlphaDropoutImpl> 的构造函数的公共接口
  // 使用 using 指示符继承基类的构造函数

  // 前向传播函数，接受输入张量并返回处理后的张量
  Tensor forward(const Tensor& input);

  /// Pretty prints the `FeatureAlphaDropout` module into the given `stream`.
  // 将 `FeatureAlphaDropout` 模块美观地打印到给定的流 `stream` 中
  void pretty_print(std::ostream& stream) const override;
  // override 关键字表明该函数覆盖了基类中的同名虚函数
};

/// 结束 `nn` 命名空间的声明

/// `ModuleHolder` 的子类，用于 `FeatureAlphaDropoutImpl`。
/// 查看 `FeatureAlphaDropoutImpl` 类的文档，了解其提供的方法，以及如何使用 `FeatureAlphaDropout` 和 `torch::nn::FeatureAlphaDropoutOptions` 的示例。
/// 查看 `ModuleHolder` 的文档，了解 PyTorch 模块存储语义。
TORCH_MODULE(FeatureAlphaDropout);

} // namespace nn
} // namespace torch


这段代码主要是在 C++ 的命名空间中声明了一个 `TORCH_MODULE` 宏，用于将 C++ 类 `FeatureAlphaDropout` 与 PyTorch 的模块系统绑定。注释解释了命名空间的作用以及 `TORCH_MODULE` 的使用场景和相关文档链接。
```