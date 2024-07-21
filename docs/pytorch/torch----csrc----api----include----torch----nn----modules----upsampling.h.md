# `.\pytorch\torch\csrc\api\include\torch\nn\modules\upsampling.h`

```py
#pragma once

#include <torch/nn/cloneable.h>  // 包含 Cloneable 类定义
#include <torch/nn/functional/upsampling.h>  // 包含上采样函数定义
#include <torch/nn/options/upsampling.h>  // 包含上采样选项类定义
#include <torch/nn/pimpl.h>  // 包含 pimpl 模式实现
#include <torch/types.h>  // 包含 Torch 类型定义

#include <torch/csrc/Export.h>  // Torch 导出定义

#include <cstddef>  // 包含标准库的 size_t 定义
#include <ostream>  // 包含输出流定义

namespace torch {
namespace nn {

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Upsample ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Upsamples a given multi-channel 1D (temporal), 2D (spatial) or 3D
/// (volumetric) data.
/// See https://pytorch.org/docs/main/nn.html#torch.nn.Upsample to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::UpsampleOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// Upsample
/// model(UpsampleOptions().scale_factor({3}).mode(torch::kLinear).align_corners(false));
/// ```py
class TORCH_API UpsampleImpl : public Cloneable<UpsampleImpl> {
 public:
  explicit UpsampleImpl(const UpsampleOptions& options_ = {});  // 构造函数声明

  void reset() override;  // 重置函数声明

  /// Pretty prints the `Upsample` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;  // 输出漂亮的打印信息声明

  Tensor forward(const Tensor& input);  // 前向传播函数声明

  /// The options with which this `Module` was constructed.
  UpsampleOptions options;  // 上采样选项对象
};

/// A `ModuleHolder` subclass for `UpsampleImpl`.
/// See the documentation for `UpsampleImpl` class to learn what methods it
/// provides, and examples of how to use `Upsample` with
/// `torch::nn::UpsampleOptions`. See the documentation for `ModuleHolder` to
/// learn about PyTorch's module storage semantics.
TORCH_MODULE(Upsample);  // 定义 TORCH_MODULE 宏，用于管理 UpsampleImpl 对象

} // namespace nn
} // namespace torch
```