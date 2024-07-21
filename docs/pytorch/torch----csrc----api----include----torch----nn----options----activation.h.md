# `.\pytorch\torch\csrc\api\include\torch\nn\options\activation.h`

```
#pragma once

#include <torch/arg.h>
#include <torch/csrc/Export.h>
#include <torch/enum.h>
#include <torch/types.h>

namespace torch {
namespace nn {

/// Options for the `ELU` module.
///
/// Example:
/// ```
/// ELU model(ELUOptions().alpha(42.42).inplace(true));
/// ```
struct TORCH_API ELUOptions {
  /// The `alpha` value for the ELU formulation. Default: 1.0
  TORCH_ARG(double, alpha) = 1.0;

  /// Can optionally do the operation in-place. Default: False
  TORCH_ARG(bool, inplace) = false;
};

namespace functional {
/// Options for `torch::nn::functional::elu`.
///
/// See the documentation for `torch::nn::ELUOptions` class to learn what
/// arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::elu(x, F::ELUFuncOptions().alpha(0.42).inplace(true));
/// ```
using ELUFuncOptions = ELUOptions;
} // namespace functional

// ============================================================================

/// Options for the `SELU` module.
///
/// Example:
/// ```
/// SELU model(SELUOptions().inplace(true));
/// ```
struct TORCH_API SELUOptions {
  /* implicit */ SELUOptions(bool inplace = false);

  /// Can optionally do the operation in-place. Default: False
  TORCH_ARG(bool, inplace);
};

namespace functional {
/// Options for `torch::nn::functional::selu`.
///
/// See the documentation for `torch::nn::SELUOptions` class to learn what
/// arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::selu(input, F::SELUFuncOptions(false));
/// ```
using SELUFuncOptions = SELUOptions;
} // namespace functional

// ============================================================================

/// Options for the `GLU` module.
///
/// Example:
/// ```
/// GLU model(GLUOptions(1));
/// ```
struct TORCH_API GLUOptions {
  /* implicit */ GLUOptions(int64_t dim = -1);

  /// The dimension on which to split the input. Default: -1
  TORCH_ARG(int64_t, dim);
};

namespace functional {
/// Options for `torch::nn::functional::glu`.
///
/// See the documentation for `torch::nn::GLUOptions` class to learn what
/// arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::glu(input, GLUFuncOptions(1));
/// ```
using GLUFuncOptions = GLUOptions;
} // namespace functional

// ============================================================================

/// Options for the `GELU` module.
///
/// Example:
/// ```
/// GELU model(GELUOptions().approximate("none"));
/// ```
struct TORCH_API GELUOptions {
  /// Specifies the approximation to apply to the output.
  TORCH_ARG(std::string, approximate) = "none";
};

namespace functional {
/// Options for `torch::nn::functional::gelu`.
///
/// See the documentation for `torch::nn::GELUOptions` class to learn what
/// arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::gelu(input, F::GELUFuncOptions().approximate("none"));
/// ```
using GELUFuncOptions = GELUOptions;
} // namespace functional
/// Options for the `Hardshrink` module.
///
/// Example:
/// ```
/// Hardshrink model(HardshrinkOptions().lambda(42.42));
/// ```
struct TORCH_API HardshrinkOptions {
  /* implicit */ HardshrinkOptions(double lambda = 0.5);

  /// the `lambda` value for the Hardshrink formulation. Default: 0.5
  TORCH_ARG(double, lambda);
};

namespace functional {
/// Options for `torch::nn::functional::hardshrink`.
///
/// See the documentation for `torch::nn::HardshrinkOptions` class to learn what
/// arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::hardshrink(x, F::HardshrinkFuncOptions().lambda(0.42));
/// ```
using HardshrinkFuncOptions = HardshrinkOptions;
} // namespace functional

// ============================================================================

/// Options for the `Hardtanh` module.
///
/// Example:
/// ```
/// Hardtanh
/// model(HardtanhOptions().min_val(-42.42).max_val(0.42).inplace(true));
/// ```
struct TORCH_API HardtanhOptions {
  /// minimum value of the linear region range. Default: -1
  TORCH_ARG(double, min_val) = -1.0;

  /// maximum value of the linear region range. Default: 1
  TORCH_ARG(double, max_val) = 1.0;

  /// can optionally do the operation in-place. Default: False
  TORCH_ARG(bool, inplace) = false;
};

namespace functional {
/// Options for `torch::nn::functional::hardtanh`.
///
/// See the documentation for `torch::nn::HardtanhOptions` class to learn what
/// arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::hardtanh(x,
/// F::HardtanhFuncOptions().min_val(-1.0).max_val(1.0).inplace(true));
/// ```
using HardtanhFuncOptions = HardtanhOptions;
} // namespace functional

// ============================================================================

/// Options for the `LeakyReLU` module.
///
/// Example:
/// ```
/// LeakyReLU model(LeakyReLUOptions().negative_slope(0.42).inplace(true));
/// ```
struct TORCH_API LeakyReLUOptions {
  /// Controls the angle of the negative slope. Default: 1e-2
  TORCH_ARG(double, negative_slope) = 1e-2;

  /// can optionally do the operation in-place. Default: False
  TORCH_ARG(bool, inplace) = false;
};

namespace functional {
/// Options for `torch::nn::functional::leaky_relu`.
///
/// See the documentation for `torch::nn::LeakyReLUOptions` class to learn what
/// arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::leaky_relu(x,
/// F::LeakyReLUFuncOptions().negative_slope(0.42).inplace(true));
/// ```
using LeakyReLUFuncOptions = LeakyReLUOptions;
} // namespace functional

// ============================================================================
/// Options for the `Softmax` module.
///
/// Example:
/// ```
/// Softmax model(SoftmaxOptions(1));
/// ```
struct TORCH_API SoftmaxOptions {
  // 构造函数，初始化 Softmax 的计算维度 dim
  SoftmaxOptions(int64_t dim);

  /// Dimension along which Softmax will be computed.
  TORCH_ARG(int64_t, dim);
};

// ============================================================================

namespace functional {

/// Options for `torch::nn::functional::softmax`.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::softmax(input, F::SoftmaxFuncOptions(1));
/// ```
struct TORCH_API SoftmaxFuncOptions {
  // 构造函数，初始化 Softmax 函数的计算维度 dim
  SoftmaxFuncOptions(int64_t dim);

  /// Dimension along which Softmax will be computed.
  TORCH_ARG(int64_t, dim);

  /// the desired data type of returned tensor.
  /// If specified, the input tensor is casted to `dtype` before the operation
  /// is performed. This is useful for preventing data type overflows. Default:
  /// None.
  TORCH_ARG(std::optional<torch::Dtype>, dtype) = c10::nullopt;
};

} // namespace functional

// ============================================================================

/// Options for the `Softmin` module.
///
/// Example:
/// ```
/// Softmin model(SoftminOptions(1));
/// ```
struct TORCH_API SoftminOptions {
  // 构造函数，初始化 Softmin 的计算维度 dim
  SoftminOptions(int64_t dim);

  /// Dimension along which Softmin will be computed.
  TORCH_ARG(int64_t, dim);
};

// ============================================================================

namespace functional {

/// Options for `torch::nn::functional::softmin`.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::softmin(input, F::SoftminFuncOptions(1));
/// ```
struct TORCH_API SoftminFuncOptions {
  // 构造函数，初始化 Softmin 函数的计算维度 dim
  SoftminFuncOptions(int64_t dim);

  /// Dimension along which Softmin will be computed.
  TORCH_ARG(int64_t, dim);

  /// the desired data type of returned tensor.
  /// If specified, the input tensor is casted to `dtype` before the operation
  /// is performed. This is useful for preventing data type overflows. Default:
  /// None.
  TORCH_ARG(std::optional<torch::Dtype>, dtype) = c10::nullopt;
};

} // namespace functional

// ============================================================================

/// Options for the `LogSoftmax` module.
///
/// Example:
/// ```
/// LogSoftmax model(LogSoftmaxOptions(1));
/// ```
struct TORCH_API LogSoftmaxOptions {
  // 构造函数，初始化 LogSoftmax 的计算维度 dim
  LogSoftmaxOptions(int64_t dim);

  /// Dimension along which LogSoftmax will be computed.
  TORCH_ARG(int64_t, dim);
};

// ============================================================================

namespace functional {

/// Options for `torch::nn::functional::log_softmax`.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::log_softmax(input, LogSoftmaxFuncOptions(1));
/// ```
// 定义了 LogSoftmaxFuncOptions 结构体，用于配置 LogSoftmax 操作的选项
struct TORCH_API LogSoftmaxFuncOptions {
  // 构造函数，接受一个整数参数 dim，表示 LogSoftmax 操作的维度
  LogSoftmaxFuncOptions(int64_t dim);

  /// Dimension along which LogSoftmax will be computed.
  // LogSoftmax 将在哪个维度上计算的参数 dim
  TORCH_ARG(int64_t, dim);

  /// the desired data type of returned tensor.
  /// If specified, the input tensor is casted to `dtype` before the operation
  /// is performed. This is useful for preventing data type overflows. Default:
  /// None.
  // 返回张量的期望数据类型
  // 如果指定，操作执行前将输入张量转换为 dtype 类型，有助于防止数据类型溢出。默认为 None
  TORCH_ARG(std::optional<torch::Dtype>, dtype) = c10::nullopt;
};

} // namespace functional

// ============================================================================

/// Options for the `PReLU` module.
///
/// Example:
/// ```
/// PReLU model(PReLUOptions().num_parameters(42));
/// ```
// 定义了 PReLUOptions 结构体，用于配置 PReLU 模块的选项
struct TORCH_API PReLUOptions {
  /// number of `a` to learn. Although it takes an int as input, there is only
  /// two values are legitimate: 1, or the number of channels at input. Default:
  /// 1
  // 要学习的参数 `a` 的数量
  // 虽然接受整数作为输入，但合法的值只有 1 或输入通道的数量。默认为 1
  TORCH_ARG(int64_t, num_parameters) = 1;

  /// the initial value of `a`. Default: 0.25
  // 参数 `a` 的初始值。默认为 0.25
  TORCH_ARG(double, init) = 0.25;
};

// ============================================================================

/// Options for the `ReLU` module.
///
/// Example:
/// ```
/// ReLU model(ReLUOptions().inplace(true));
/// ```
// 定义了 ReLUOptions 结构体，用于配置 ReLU 模块的选项
struct TORCH_API ReLUOptions {
  /* implicit */ ReLUOptions(bool inplace = false);

  /// can optionally do the operation in-place. Default: False
  // 是否可以选择原地执行操作。默认为 False
  TORCH_ARG(bool, inplace);
};

namespace functional {
/// Options for `torch::nn::functional::relu`.
///
/// See the documentation for `torch::nn::ReLUOptions` class to learn what
/// arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::relu(x, F::ReLUFuncOptions().inplace(true));
/// ```
// 使用 ReLUOptions 的别名 ReLUFuncOptions，用于配置 torch::nn::functional::relu 的选项
using ReLUFuncOptions = ReLUOptions;
} // namespace functional

// ============================================================================

/// Options for the `ReLU6` module.
///
/// Example:
/// ```
/// ReLU6 model(ReLU6Options().inplace(true));
/// ```
// 定义了 ReLU6Options 结构体，用于配置 ReLU6 模块的选项
struct TORCH_API ReLU6Options {
  /* implicit */ ReLU6Options(bool inplace = false);

  /// can optionally do the operation in-place. Default: False
  // 是否可以选择原地执行操作。默认为 False
  TORCH_ARG(bool, inplace);
};

namespace functional {
/// Options for `torch::nn::functional::relu6`.
///
/// See the documentation for `torch::nn::ReLU6Options` class to learn what
/// arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::relu6(x, F::ReLU6FuncOptions().inplace(true));
/// ```
// 使用 ReLU6Options 的别名 ReLU6FuncOptions，用于配置 torch::nn::functional::relu6 的选项
using ReLU6FuncOptions = ReLU6Options;
} // namespace functional

// ============================================================================

/// Options for the `RReLU` module.
///
/// Example:
/// ```
/// RReLU model(RReLUOptions().lower(0.24).upper(0.42).inplace(true));
/// ```
// 定义了 RReLUOptions 结构体，用于配置 RReLU 激活函数的选项
struct TORCH_API RReLUOptions {
  /// uniform 分布的下界，默认为 1/8
  TORCH_ARG(double, lower) = 1.0 / 8.0;

  /// uniform 分布的上界，默认为 1/3
  TORCH_ARG(double, upper) = 1.0 / 3.0;

  /// 是否原地操作的选项。默认为 False
  TORCH_ARG(bool, inplace) = false;
};

// ============================================================================

// functional 命名空间下的 RReLUFuncOptions 结构体，用于配置函数式 RReLU 的选项
// 示例用法见文档注释
namespace functional {

/// Options for `torch::nn::functional::rrelu`.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::rrelu(x, F::RReLUFuncOptions().lower(0.1).upper(0.4).inplace(true));
/// ```
struct TORCH_API RReLUFuncOptions {
  /// uniform 分布的下界，默认为 1/8
  TORCH_ARG(double, lower) = 1.0 / 8.0;

  /// uniform 分布的上界，默认为 1/3
  TORCH_ARG(double, upper) = 1.0 / 3.0;

  /// 是否处于训练模式的选项。默认为 False
  TORCH_ARG(bool, training) = false;

  /// 是否原地操作的选项。默认为 False
  TORCH_ARG(bool, inplace) = false;
};

} // namespace functional

// ============================================================================

/// Options for the `CELU` module.
///
/// Example:
/// ```
/// CELU model(CELUOptions().alpha(42.42).inplace(true));
/// ```
struct TORCH_API CELUOptions {
  /// CELU 函数的 alpha 参数值，默认为 1.0
  TORCH_ARG(double, alpha) = 1.0;

  /// 是否原地操作的选项。默认为 False
  TORCH_ARG(bool, inplace) = false;
};

namespace functional {
/// Options for `torch::nn::functional::celu`.
///
/// See the documentation for `torch::nn::CELUOptions` class to learn what
/// arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::celu(x, F::CELUFuncOptions().alpha(0.42).inplace(true));
/// ```
using CELUFuncOptions = CELUOptions;
} // namespace functional

// ============================================================================

/// Options for the `Softplus` module.
///
/// Example:
/// ```
/// Softplus model(SoftplusOptions().beta(0.24).threshold(42.42));
/// ```
struct TORCH_API SoftplusOptions {
  /// Softplus 函数的 beta 参数值，默认为 1
  TORCH_ARG(double, beta) = 1.0;

  /// 超过此阈值后将转换为线性函数，默认为 20
  TORCH_ARG(double, threshold) = 20.0;
};

namespace functional {
/// Options for `torch::nn::functional::softplus`.
///
/// See the documentation for `torch::nn::SoftplusOptions` class to learn what
/// arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::softplus(x, F::SoftplusFuncOptions().beta(0.5).threshold(3.0));
/// ```
using SoftplusFuncOptions = SoftplusOptions;
} // namespace functional

// ============================================================================

/// Options for the `Softshrink` module.
///
/// Example:
/// ```
/// Softshrink model(SoftshrinkOptions(42.42));
/// ```
/// Options for configuring parameters of the Softshrink function.
struct TORCH_API SoftshrinkOptions {
  /* implicit */ SoftshrinkOptions(double lambda = 0.5);

  /// Specifies the `lambda` parameter used in Softshrink. Default: 0.5
  TORCH_ARG(double, lambda);
};

namespace functional {
/// Options for using the `torch::nn::functional::softshrink` function.
///
/// Refer to the documentation of `torch::nn::SoftshrinkOptions` for supported arguments.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::softshrink(x, F::SoftshrinkFuncOptions(0.42));
/// ```
using SoftshrinkFuncOptions = SoftshrinkOptions;
} // namespace functional

// ============================================================================

/// Options for configuring parameters of the `Threshold` module.
///
/// Example:
/// ```
/// Threshold model(ThresholdOptions(42.42, 24.24).inplace(true));
/// ```
struct TORCH_API ThresholdOptions {
  ThresholdOptions(double threshold, double value)
      : threshold_(threshold), value_(value) {}

  /// Specifies the threshold value.
  TORCH_ARG(double, threshold);

  /// Specifies the value to replace with.
  TORCH_ARG(double, value);

  /// Optionally indicates whether the operation should be performed in-place. Default: False
  TORCH_ARG(bool, inplace) = false;
};

namespace functional {
/// Options for using the `torch::nn::functional::threshold` function.
///
/// Refer to the documentation of `torch::nn::ThresholdOptions` for supported arguments.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::threshold(x, F::ThresholdFuncOptions(0.5, 0.5).inplace(true));
/// ```
using ThresholdFuncOptions = ThresholdOptions;
} // namespace functional

// ============================================================================

namespace functional {
/// Options for `torch::nn::functional::gumbel_softmax` function.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::gumbel_softmax(logits, F::GumbelSoftmaxFuncOptions().hard(true).dim(-1));
/// ```
struct TORCH_API GumbelSoftmaxFuncOptions {
  /// Specifies the non-negative scalar temperature (`tau`).
  TORCH_ARG(double, tau) = 1.0;

  /// Indicates whether returned samples will be one-hot vectors (`hard`), treated as soft samples in autograd. Default: False
  TORCH_ARG(bool, hard) = false;

  /// Specifies the dimension along which softmax will be computed. Default: -1
  TORCH_ARG(int, dim) = -1;
};

} // namespace functional

// ============================================================================

/// Options for configuring parameters of the `MultiheadAttention` module.
///
/// Example:
/// ```
/// MultiheadAttention model(MultiheadAttentionOptions(20, 10).bias(false));
/// ```
// 定义了 MultiheadAttentionOptions 结构体，用于存储多头注意力机制的参数选项
struct TORCH_API MultiheadAttentionOptions {
  // 构造函数，初始化 embed_dim 和 num_heads
  MultiheadAttentionOptions(int64_t embed_dim, int64_t num_heads);

  /// total dimension of the model.
  // 模型的总维度
  TORCH_ARG(int64_t, embed_dim);

  /// parallel attention heads.
  // 并行的注意力头数
  TORCH_ARG(int64_t, num_heads);

  /// a Dropout layer on attn_output_weights. Default: 0.0.
  // 在 attn_output_weights 上添加一个 Dropout 层，默认值为 0.0
  TORCH_ARG(double, dropout) = 0.0;

  /// add bias as module parameter. Default: true.
  // 将偏置作为模块参数添加，默认值为 true
  TORCH_ARG(bool, bias) = true;

  /// add bias to the key and value sequences at dim=0.
  // 在维度为 0 的键和值序列上添加偏置
  TORCH_ARG(bool, add_bias_kv) = false;

  /// add a new batch of zeros to the key and value sequences at dim=1.
  // 在维度为 1 的键和值序列上添加一个新的零批次
  TORCH_ARG(bool, add_zero_attn) = false;

  /// total number of features in key. Default: c10::nullopt.
  // 键中的特征总数，默认值为 c10::nullopt
  TORCH_ARG(int64_t, kdim);

  /// total number of features in key. Default: c10::nullopt.
  // 值中的特征总数，默认值为 c10::nullopt
  TORCH_ARG(int64_t, vdim);
};

// ============================================================================

namespace functional {

/// Options for `torch::nn::functional::multi_head_attention_forward`
// torch::nn::functional::multi_head_attention_forward 的选项
struct TORCH_API MultiheadAttentionForwardFuncOptions {
  // 构造函数，初始化各种参数
  MultiheadAttentionForwardFuncOptions(
      int64_t embed_dim_to_check,
      int64_t num_heads,
      Tensor in_proj_weight,
      Tensor in_proj_bias,
      Tensor bias_k,
      Tensor bias_v,
      bool add_zero_attn,
      double dropout_p,
      Tensor out_proj_weight,
      Tensor out_proj_bias);

  // 检查的嵌入维度
  TORCH_ARG(int64_t, embed_dim_to_check);

  // 注意力头数
  TORCH_ARG(int64_t, num_heads);

  // 输入投影权重
  TORCH_ARG(Tensor, in_proj_weight);

  // 输入投影偏置
  TORCH_ARG(Tensor, in_proj_bias);

  // 键的偏置
  TORCH_ARG(Tensor, bias_k);

  // 值的偏置
  TORCH_ARG(Tensor, bias_v);

  // 是否在零注意力上添加
  TORCH_ARG(bool, add_zero_attn);

  // Dropout 概率
  TORCH_ARG(double, dropout_p);

  // 输出投影权重
  TORCH_ARG(Tensor, out_proj_weight);

  // 输出投影偏置
  TORCH_ARG(Tensor, out_proj_bias);

  // 是否训练，默认为 true
  TORCH_ARG(bool, training) = true;

  // 键填充掩码，默认为空
  TORCH_ARG(Tensor, key_padding_mask) = {};

  // 是否需要权重，默认为 true
  TORCH_ARG(bool, need_weights) = true;

  // 注意力掩码，默认为空
  TORCH_ARG(Tensor, attn_mask) = {};

  // 是否使用单独的投影权重，默认为 false
  TORCH_ARG(bool, use_separate_proj_weight) = false;

  // 查询投影权重，默认为空
  TORCH_ARG(Tensor, q_proj_weight) = {};

  // 键投影权重，默认为空
  TORCH_ARG(Tensor, k_proj_weight) = {};

  // 值投影权重，默认为空
  TORCH_ARG(Tensor, v_proj_weight) = {};

  // 静态键，默认为空
  TORCH_ARG(Tensor, static_k) = {};

  // 静态值，默认为空
  TORCH_ARG(Tensor, static_v) = {};

  // 平均注意力权重，默认为 true
  TORCH_ARG(bool, average_attn_weights) = true;
};

} // namespace functional

} // namespace nn
} // namespace torch
```