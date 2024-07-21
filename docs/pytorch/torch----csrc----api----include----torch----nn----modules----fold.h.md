# `.\pytorch\torch\csrc\api\include\torch\nn\modules\fold.h`

```
#pragma once

#include <torch/expanding_array.h>  // 引入扩展数组的头文件
#include <torch/nn/cloneable.h>     // 引入可克隆模块的头文件
#include <torch/nn/functional/fold.h>  // 引入fold功能的头文件
#include <torch/nn/options/fold.h>  // 引入fold选项的头文件
#include <torch/nn/pimpl.h>         // 引入私有实现的头文件
#include <torch/types.h>            // 引入torch数据类型的头文件

namespace torch {
namespace nn {

/// Applies fold over a 3-D input.
/// See https://pytorch.org/docs/main/nn.html#torch.nn.Fold to learn about
/// the exact behavior of this module.
///
/// See the documentation for `torch::nn::FoldOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// Fold model(FoldOptions({8, 8}, {3, 3}).dilation(2).padding({2, 1}).stride(2));
/// ```
class TORCH_API FoldImpl : public torch::nn::Cloneable<FoldImpl> {
 public:
  // 构造函数，接受两个扩展数组参数来创建FoldImpl对象
  FoldImpl(ExpandingArray<2> output_size, ExpandingArray<2> kernel_size)
      : FoldImpl(FoldOptions(output_size, kernel_size)) {}
  
  // 显式构造函数，接受FoldOptions对象作为参数
  explicit FoldImpl(const FoldOptions& options_);

  // 重置函数，重载自基类Cloneable的虚函数
  void reset() override;

  /// Pretty prints the `Fold` module into the given `stream`.
  // 将Fold模块的信息漂亮地打印到给定的流中
  void pretty_print(std::ostream& stream) const override;

  // 前向传播函数，接受Tensor作为输入，返回Tensor作为输出
  Tensor forward(const Tensor& input);

  /// The options with which this `Module` was constructed.
  // 存储构造此模块时使用的选项对象
  FoldOptions options;
};

/// A `ModuleHolder` subclass for `FoldImpl`.
/// See the documentation for `FoldImpl` class to learn what methods it
/// provides, and examples of how to use `Fold` with `torch::nn::FoldOptions`.
/// See the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(Fold);

// ============================================================================

/// Applies unfold over a 4-D input.
/// See https://pytorch.org/docs/main/nn.html#torch.nn.Unfold to learn about
/// the exact behavior of this module.
///
/// See the documentation for `torch::nn::UnfoldOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// Unfold model(UnfoldOptions({2, 4}).dilation(2).padding({2, 1}).stride(2));
/// ```
class TORCH_API UnfoldImpl : public Cloneable<UnfoldImpl> {
 public:
  // 构造函数，接受一个扩展数组参数来创建UnfoldImpl对象
  UnfoldImpl(ExpandingArray<2> kernel_size)
      : UnfoldImpl(UnfoldOptions(kernel_size)) {}
  
  // 显式构造函数，接受UnfoldOptions对象作为参数
  explicit UnfoldImpl(const UnfoldOptions& options_);

  // 重置函数，重载自基类Cloneable的虚函数
  void reset() override;

  /// Pretty prints the `Unfold` module into the given `stream`.
  // 将Unfold模块的信息漂亮地打印到给定的流中
  void pretty_print(std::ostream& stream) const override;

  // 前向传播函数，接受Tensor作为输入，返回Tensor作为输出
  Tensor forward(const Tensor& input);

  /// The options with which this `Module` was constructed.
  // 存储构造此模块时使用的选项对象
  UnfoldOptions options;
};

/// A `ModuleHolder` subclass for `UnfoldImpl`.
/// See the documentation for `UnfoldImpl` class to learn what methods it
/// provides, and examples of how to use `Unfold` with
/// `torch::nn::UnfoldOptions`. See the documentation for `ModuleHolder` to
/// learn about PyTorch's module storage semantics.
TORCH_MODULE(Unfold);

} // namespace nn
} // namespace torch
```