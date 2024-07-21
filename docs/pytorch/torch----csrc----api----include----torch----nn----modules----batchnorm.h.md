# `.\pytorch\torch\csrc\api\include\torch\nn\modules\batchnorm.h`

```py
#pragma once

#include <torch/nn/cloneable.h>  // 包含用于克隆的基类
#include <torch/nn/functional/batchnorm.h>  // 包含批归一化函数的头文件
#include <torch/nn/init.h>  // 包含参数初始化函数的头文件
#include <torch/nn/options/batchnorm.h>  // 包含批归一化选项的头文件
#include <torch/nn/pimpl.h>  // 包含私有实现的头文件
#include <torch/types.h>  // 包含 Torch 张量类型的头文件

#include <cstdint>  // 包含标准整数类型的头文件

namespace torch {
namespace nn {

/// Base class for all (dimension-specialized) batchnorm and instancenorm
/// modules.
template <size_t D, typename Derived, typename DerivedOptions>
class NormImplBase : public torch::nn::Cloneable<Derived> {
 protected:
  virtual void _check_input_dim(const Tensor& input) = 0;  // 虚拟函数，用于检查输入维度

 public:
  /// Constructor initializing the options.
  NormImplBase(const DerivedOptions& options_) : options(options_) {
    // NOLINTNEXTLINE(clang-analyzer-optin.cplusplus.VirtualCall)
    reset();  // 调用 reset() 函数进行初始化
  }

  /// Reset function for initializing parameters and buffers.
  void reset() override {
    // 根据是否启用仿射变换初始化权重和偏置参数
    if (options.affine()) {
      weight = this->register_parameter(
          "weight", torch::empty({options.num_features()}));  // 注册可学习的权重参数
      bias = this->register_parameter(
          "bias", torch::empty({options.num_features()}));  // 注册可学习的偏置参数
    } else {
      weight =
          this->register_parameter("weight", Tensor(), /*requires_grad=*/false);  // 注册权重参数，不需要梯度
      bias =
          this->register_parameter("bias", Tensor(), /*requires_grad=*/false);  // 注册偏置参数，不需要梯度
    }
    // 根据是否跟踪运行时统计信息，初始化运行时统计参数
    if (options.track_running_stats()) {
      running_mean = this->register_buffer(
          "running_mean", torch::zeros({options.num_features()}));  // 注册运行时均值缓冲区
      running_var = this->register_buffer(
          "running_var", torch::ones({options.num_features()}));  // 注册运行时方差缓冲区
      num_batches_tracked = this->register_buffer(
          "num_batches_tracked", torch::tensor(0, torch::dtype(torch::kLong)));  // 注册跟踪的批次数缓冲区
    } else {
      running_mean = this->register_buffer("running_mean", Tensor());  // 注册空的运行时均值缓冲区
      running_var = this->register_buffer("running_var", Tensor());  // 注册空的运行时方差缓冲区
      num_batches_tracked =
          this->register_buffer("num_batches_tracked", Tensor());  // 注册空的跟踪批次数缓冲区
    }
    reset_parameters();  // 调用 reset_parameters() 函数重置参数
  }

  /// Function to reset running statistics (mean and variance).
  void reset_running_stats() {
    if (options.track_running_stats()) {
      running_mean.zero_();  // 将运行时均值置零
      running_var.fill_(1);  // 将运行时方差置为1
      num_batches_tracked.zero_();  // 将跟踪的批次数置零
    }
  }

  /// Function to reset parameters (weights and biases).
  void reset_parameters() {
    reset_running_stats();  // 重置运行时统计信息
    if (options.affine()) {
      torch::nn::init::ones_(weight);  // 使用初始化为1的函数初始化权重
      torch::nn::init::zeros_(bias);  // 使用初始化为0的函数初始化偏置
    }
  }

  /// The options with which this module was constructed.
  DerivedOptions options;  // 构造模块时使用的选项

  /// The learned weight.
  /// Only defined if the `affine` option was `true` upon construction.
  Tensor weight;  // 可学习的权重参数

  /// The learned bias.
  /// Only defined if the `affine` option was `true` upon construction.
  Tensor bias;  // 可学习的偏置参数

  /// The running mean.
  /// Only defined if the `track_running_stats` option was `true` upon
  /// construction.
  Tensor running_mean;  // 运行时均值

  /// The running variance.
  /// Only defined if the `track_running_stats` option was `true` upon
  /// construction.
  Tensor running_var;  // 运行时方差

  /// The number of the forward call.
  /// Only defined if the `track_running_stats` option was `true` upon
  /// construction.
  Tensor num_batches_tracked;  // 前向调用的批次数
};

}  // namespace nn
}  // namespace torch
/// Base class for all (dimension-specialized) batchnorm modules.
template <size_t D, typename Derived>
class BatchNormImplBase : public NormImplBase<D, Derived, BatchNormOptions> {
 public:
  using NormImplBase<D, Derived, BatchNormOptions>::NormImplBase;

  /// Forward pass of the batch normalization module.
  /// Computes the normalized output using the input tensor.
  Tensor forward(const Tensor& input) {
    // Check if input dimension matches expected dimension
    this->_check_input_dim(input);

    // Initialize the exponential average factor for momentum
    double exponential_average_factor;
    if (this->options.momentum() == c10::nullopt) {
      exponential_average_factor = 0.0;
    } else {
      exponential_average_factor = this->options.momentum().value();
    }

    // Update running statistics if in training mode and tracking stats
    if (this->is_training() && this->options.track_running_stats()) {
      if (this->num_batches_tracked.defined()) {
        this->num_batches_tracked += 1;
        if (this->options.momentum() == c10::nullopt) {
          // Use cumulative moving average for exponential_average_factor
          exponential_average_factor =
              1.0 / this->num_batches_tracked.template item<double>();
        } else {
          // Use user-defined momentum for exponential_average_factor
          exponential_average_factor = this->options.momentum().value();
        }
      }
    }

    // Perform batch normalization using Torch's functional API
    return torch::nn::functional::detail::batch_norm(
        input,
        this->running_mean,
        this->running_var,
        this->weight,
        this->bias,
        this->is_training() || !this->options.track_running_stats(),
        /*momentum=*/exponential_average_factor,
        this->options.eps());
  }

  /// Pretty prints the `BatchNorm{1,2,3}d` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override {
    stream << std::boolalpha << "torch::nn::BatchNorm" << D << "d("
           << this->options.num_features() << ", "
           << "eps=" << this->options.eps() << ", "
           << "momentum=";

    // Print momentum value if defined, otherwise print "None"
    if (this->options.momentum().has_value()) {
      stream << this->options.momentum().value();
    } else {
      stream << "None";
    }

    // Print other module options
    stream << ", "
           << "affine=" << this->options.affine() << ", "
           << "track_running_stats=" << this->options.track_running_stats()
           << ")";
  }
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ BatchNorm1d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies the BatchNorm1d function.
/// See https://pytorch.org/docs/main/nn.html#torch.nn.BatchNorm1d to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::BatchNorm1dOptions` class to learn
/// what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// BatchNorm1d
/// model(BatchNorm1dOptions(4).eps(0.5).momentum(0.1).affine(false).track_running_stats(true));
/// ```py
class TORCH_API BatchNorm1dImpl : public BatchNormImplBase<1, BatchNorm1dImpl> {
 protected:
  void _check_input_dim(const Tensor& input) override;
 public:
  using BatchNormImplBase<1, BatchNorm1dImpl>::BatchNormImplBase;
};

/// A `ModuleHolder` subclass for `BatchNorm1dImpl`.
/// See the documentation for `BatchNorm1dImpl` class to learn what methods it
/// 定义了一个宏 `TORCH_MODULE`，用于简化模块的声明和定义
TORCH_MODULE(BatchNorm1d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ BatchNorm2d
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// 应用 BatchNorm2d 函数，实现二维批量归一化
/// 详细行为请参见 https://pytorch.org/docs/main/nn.html#torch.nn.BatchNorm2d
///
/// 参见 `torch::nn::BatchNorm2dOptions` 类的文档，了解此模块支持的构造参数
///
/// 示例:
/// ```
/// BatchNorm2d
/// model(BatchNorm2dOptions(4).eps(0.5).momentum(0.1).affine(false).track_running_stats(true));
/// ```py
class TORCH_API BatchNorm2dImpl : public BatchNormImplBase<2, BatchNorm2dImpl> {
 protected:
  void _check_input_dim(const Tensor& input) override;

 public:
  using BatchNormImplBase<2, BatchNorm2dImpl>::BatchNormImplBase;
};

/// 用于 `BatchNorm2dImpl` 的 `ModuleHolder` 子类
/// 参见 `BatchNorm2dImpl` 类的文档，了解其提供的方法以及如何使用 `BatchNorm2d` 与 `torch::nn::BatchNorm2dOptions`
/// 参见 `ModuleHolder` 的文档，了解 PyTorch 的模块存储语义
TORCH_MODULE(BatchNorm2d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ BatchNorm3d
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// 应用 BatchNorm3d 函数，实现三维批量归一化
/// 详细行为请参见 https://pytorch.org/docs/main/nn.html#torch.nn.BatchNorm3d
///
/// 参见 `torch::nn::BatchNorm3dOptions` 类的文档，了解此模块支持的构造参数
///
/// 示例:
/// ```
/// BatchNorm3d
/// model(BatchNorm3dOptions(4).eps(0.5).momentum(0.1).affine(false).track_running_stats(true));
/// ```py
class TORCH_API BatchNorm3dImpl : public BatchNormImplBase<3, BatchNorm3dImpl> {
 protected:
  void _check_input_dim(const Tensor& input) override;

 public:
  using BatchNormImplBase<3, BatchNorm3dImpl>::BatchNormImplBase;
};

/// 用于 `BatchNorm3dImpl` 的 `ModuleHolder` 子类
/// 参见 `BatchNorm3dImpl` 类的文档，了解其提供的方法以及如何使用 `BatchNorm3d` 与 `torch::nn::BatchNorm3dOptions`
/// 参见 `ModuleHolder` 的文档，了解 PyTorch 的模块存储语义
TORCH_MODULE(BatchNorm3d);

} // namespace nn
} // namespace torch
```