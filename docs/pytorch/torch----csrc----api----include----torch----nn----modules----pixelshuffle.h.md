# `.\pytorch\torch\csrc\api\include\torch\nn\modules\pixelshuffle.h`

```py
#pragma once
// 只在编译时包含一次该头文件，避免重复定义

#include <torch/nn/cloneable.h>
// 引入 torch 中的 nn 模块中的 Cloneable 类

#include <torch/nn/functional/pixelshuffle.h>
// 引入 torch 中 nn 模块中的 functional 子模块中的 pixelshuffle 头文件

#include <torch/nn/options/pixelshuffle.h>
// 引入 torch 中 nn 模块中的 options 子模块中的 pixelshuffle 头文件

#include <torch/csrc/Export.h>
// 引入 torch 中 csrc 文件夹下的 Export 头文件

namespace torch {
namespace nn {

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PixelShuffle
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Rearranges elements in a tensor of shape :math:`(*, C \times r^2, H, W)`
/// to a tensor of shape :math:`(*, C, H \times r, W \times r)`, where r is an
/// upscale factor. See
/// https://pytorch.org/docs/main/nn.html#torch.nn.PixelShuffle to learn about
/// the exact behavior of this module.
///
/// See the documentation for `torch::nn::PixelShuffleOptions` class to learn
/// what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// PixelShuffle model(PixelShuffleOptions(5));
/// ```py
struct TORCH_API PixelShuffleImpl
    : public torch::nn::Cloneable<PixelShuffleImpl> {
  explicit PixelShuffleImpl(const PixelShuffleOptions& options_);
  // 显式构造函数，使用 PixelShuffleOptions 初始化 PixelShuffleImpl 对象

  /// Pretty prints the `PixelShuffle` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;
  // 打印 PixelShuffle 模块的详细信息到给定的流对象中

  Tensor forward(const Tensor& input);
  // 前向传播函数，接受一个输入张量，并返回一个张量作为输出

  void reset() override;
  // 重置函数，覆盖基类中的 reset 方法，重置模块的状态

  /// The options with which this `Module` was constructed.
  PixelShuffleOptions options;
  // 保存构造该模块时使用的选项

};

/// A `ModuleHolder` subclass for `PixelShuffleImpl`.
/// See the documentation for `PixelShuffleImpl` class to learn what methods it
/// provides, and examples of how to use `PixelShuffle` with
/// `torch::nn::PixelShuffleOptions`. See the documentation for `ModuleHolder`
/// to learn about PyTorch's module storage semantics.
TORCH_MODULE(PixelShuffle);
// 定义 PixelShuffle 模块的模块持有类

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PixelUnshuffle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Reverses the PixelShuffle operation by rearranging elements in a tensor of
/// shape :math:`(*, C, H \times r, W \times r)` to a tensor of shape :math:`(*,
/// C \times r^2, H, W)`, where r is a downscale factor. See
/// https://pytorch.org/docs/main/nn.html#torch.nn.PixelUnshuffle to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::PixelUnshuffleOptions` class to learn
/// what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// PixelUnshuffle model(PixelUnshuffleOptions(5));
/// ```py
struct TORCH_API PixelUnshuffleImpl
    : public torch::nn::Cloneable<PixelUnshuffleImpl> {
  explicit PixelUnshuffleImpl(const PixelUnshuffleOptions& options_);
  // 显式构造函数，使用 PixelUnshuffleOptions 初始化 PixelUnshuffleImpl 对象

  /// Pretty prints the `PixelUnshuffle` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;
  // 打印 PixelUnshuffle 模块的详细信息到给定的流对象中

  Tensor forward(const Tensor& input);
  // 前向传播函数，接受一个输入张量，并返回一个张量作为输出

  void reset() override;
  // 重置函数，覆盖基类中的 reset 方法，重置模块的状态

  /// The options with which this `Module` was constructed.
  PixelUnshuffleOptions options;
  // 保存构造该模块时使用的选项

};

/// A `ModuleHolder` subclass for `PixelUnshuffleImpl`.
/// See the documentation for `PixelUnshuffleImpl` class to learn what methods
/// it provides, and examples of how to use `PixelUnshuffle` with
/// `torch::nn::PixelUnshuffleOptions`.
/// See the documentation for `ModuleHolder` to learn about PyTorch's module storage semantics.
/// 声明一个名为 `PixelUnshuffle` 的PyTorch模块，并使用 `torch::nn::PixelUnshuffleOptions` 配置选项。
/// 参考 `ModuleHolder` 的文档可以了解PyTorch模块存储的语义。
TORCH_MODULE(PixelUnshuffle);

} // namespace nn
} // namespace torch
```