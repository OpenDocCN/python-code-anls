# `.\pytorch\torch\csrc\api\include\torch\nn\functional\activation.h`

```
#pragma once

// 包含 ATen 库的分发头文件，用于多种 CPU/GPU 操作的分发
#include <ATen/Dispatch.h>
// 包含 Torch 库中 dropout 相关功能的头文件
#include <torch/nn/functional/dropout.h>
// 包含 Torch 库中 linear 相关功能的头文件
#include <torch/nn/functional/linear.h>
// 包含 Torch 库中激活函数选项的头文件
#include <torch/nn/options/activation.h>
// 包含 Torch 库中 dropout 选项的头文件
#include <torch/nn/options/dropout.h>
// 包含 Torch 库中 linear 选项的头文件
#include <torch/nn/options/linear.h>
// 包含 Torch 库中 Tensor 类型定义的头文件
#include <torch/types.h>
// 包含标准库中的数值极限定义
#include <limits>
// 包含实用工具的头文件
#include <utility>

// Torch 命名空间
namespace torch {
// Torch 中 nn 命名空间
namespace nn {
// Torch 中 functional 命名空间
namespace functional {

// 防止 Doxygen 解析以下内容
#ifndef DOXYGEN_SHOULD_SKIP_THIS
// 内部细节命名空间，定义了一些私有函数
namespace detail {
// ELU 激活函数实现，根据 inplace 参数选择是否原地操作
inline Tensor elu(Tensor input, double alpha, bool inplace) {
  // 如果 inplace 参数为 true，则使用 torch::elu_ 原地操作
  if (inplace) {
    return torch::elu_(input, alpha);
  } else {
    // 否则使用 torch::elu 进行非原地操作
    return torch::elu(input, alpha);
  }
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// 查看
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.elu
/// 获取该功能的详细行为说明。
///
/// 查看 `torch::nn::functional::ELUFuncOptions` 类的文档，了解该功能支持的可选参数。
///
/// 示例:
/// ```
/// namespace F = torch::nn::functional;
/// F::elu(x, F::ELUFuncOptions().alpha(0.42).inplace(true));
/// ```
// ELU 激活函数接口，根据 ELUFuncOptions 参数调用内部实现函数
inline Tensor elu(Tensor input, const ELUFuncOptions& options = {}) {
  return detail::elu(std::move(input), options.alpha(), options.inplace());
}

// ============================================================================

// 防止 Doxygen 解析以下内容
#ifndef DOXYGEN_SHOULD_SKIP_THIS
// 内部细节命名空间，定义了一些私有函数
namespace detail {
// SELU 激活函数实现，根据 inplace 参数选择是否原地操作
inline Tensor selu(Tensor input, bool inplace) {
  // 如果 inplace 参数为 true，则使用 torch::selu_ 原地操作
  if (inplace) {
    return torch::selu_(input);
  } else {
    // 否则使用 torch::selu 进行非原地操作
    return torch::selu(input);
  }
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// 查看
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.selu
/// 获取该功能的详细行为说明。
///
/// 查看 `torch::nn::functional::SELUFuncOptions` 类的文档，了解该功能支持的可选参数。
///
/// 示例:
/// ```
/// namespace F = torch::nn::functional;
/// F::selu(input, F::SELUFuncOptions(false));
/// ```
// SELU 激活函数接口，根据 SELUFuncOptions 参数调用内部实现函数
inline Tensor selu(Tensor input, const SELUFuncOptions& options = {}) {
  return detail::selu(std::move(input), options.inplace());
}

// ============================================================================

// 防止 Doxygen 解析以下内容
#ifndef DOXYGEN_SHOULD_SKIP_THIS
// 内部细节命名空间，定义了一些私有函数
namespace detail {
// Hardshrink 函数实现，根据 lambda 参数对输入 Tensor 进行硬收缩
inline Tensor hardshrink(const Tensor& input, double lambda) {
  return torch::hardshrink(input, lambda);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// 查看
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.hardshrink
/// 获取该功能的详细行为说明。
///
/// 查看 `torch::nn::functional::HardshrinkFuncOptions` 类的文档，了解该功能支持的可选参数。
///
/// 示例:
/// ```
/// namespace F = torch::nn::functional;
/// F::hardshrink(x, F::HardshrinkFuncOptions().lambda(0.42));
/// ```
// Hardshrink 函数接口，根据 HardshrinkFuncOptions 参数调用内部实现函数
inline Tensor hardshrink(
    const Tensor& input,
    const HardshrinkFuncOptions& options = {}) {
  return detail::hardshrink(input, options.lambda());
}
// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
// 定义一个命名空间 detail，用于实现内部细节函数
namespace detail {
// 定义了一个 hardtanh 函数，用于对输入张量进行硬切线函数操作
inline Tensor hardtanh(
    Tensor input,                   // 输入张量
    double min_val,                 // 最小值
    double max_val,                 // 最大值
    bool inplace) {                 // 是否原地操作的标志

  if (inplace) {
    return torch::hardtanh_(input, min_val, max_val);  // 原地应用硬切线函数
  } else {
    return torch::hardtanh(input, min_val, max_val);   // 应用硬切线函数
  }
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// 查看此函数的详细行为，请参考：
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.hardtanh
///
/// 查看 `torch::nn::functional::HardtanhFuncOptions` 类的文档，了解此函数支持的可选参数。
///
/// 示例：
/// ```
/// namespace F = torch::nn::functional;
/// F::hardtanh(x,
/// F::HardtanhFuncOptions().min_val(-1.0).max_val(1.0).inplace(true));
/// ```
inline Tensor hardtanh(Tensor input, const HardtanhFuncOptions& options = {}) {
  return detail::hardtanh(
      std::move(input),
      options.min_val(),             // 设置最小值
      options.max_val(),             // 设置最大值
      options.inplace());            // 是否原地操作
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
// 定义一个命名空间 detail，用于实现内部细节函数
namespace detail {
// 定义了一个 leaky_relu 函数，用于对输入张量进行泄露线性整流操作
inline Tensor leaky_relu(
    Tensor input,                   // 输入张量
    double negative_slope,          // 负斜率
    bool inplace) {                 // 是否原地操作的标志

  if (inplace) {
    return torch::leaky_relu_(input, negative_slope);   // 原地应用泄露线性整流函数
  } else {
    return torch::leaky_relu(input, negative_slope);    // 应用泄露线性整流函数
  }
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// 查看此函数的详细行为，请参考：
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.leaky_relu
///
/// 查看 `torch::nn::functional::LeakyReLUFuncOptions` 类的文档，了解此函数支持的可选参数。
///
/// 示例：
/// ```
/// namespace F = torch::nn::functional;
/// F::leaky_relu(x,
/// F::LeakyReLUFuncOptions().negative_slope(0.42).inplace(true));
/// ```
inline Tensor leaky_relu(
    Tensor input,
    const LeakyReLUFuncOptions& options = {}) {
  return detail::leaky_relu(
      std::move(input),             // 移动输入张量
      options.negative_slope(),     // 设置负斜率
      options.inplace());           // 是否原地操作
}

// ============================================================================

/// 对输入张量应用 logsigmoid 函数。
inline Tensor logsigmoid(const Tensor& input) {
  return torch::log_sigmoid(input);  // 应用 log-sigmoid 函数
}

// ============================================================================
    # 如果条件成立（即 y_hard - y_soft.detach() + y_soft），执行以下操作
    ret = y_hard - y_soft.detach() + y_soft;
    # 如果条件不成立（即不满足上述条件），则执行以下操作
    } else {
        # 将 y_soft 赋值给 ret
        ret = y_soft;
    }
    # 返回 ret 变量的值作为函数的结果
    return ret;
#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
// 定义 log_softmax 函数，计算输入张量的对数 softmax 在指定维度上的结果
inline Tensor log_softmax(
    const Tensor& input,
    int64_t dim,
    std::optional<torch::Dtype> dtype) {
  Tensor ret;

  // 如果未指定 dtype，则使用默认的浮点数类型计算 softmax
  if (dtype == c10::nullopt) {
    ret = input.log_softmax(dim);
  } else {
    // 如果指定了 dtype，则使用指定的类型计算 softmax
    ret = input.log_softmax(dim, dtype);
  }

  // 返回计算结果
  return ret;
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.log_softmax
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::LogSoftmaxFuncOptions` class
/// to learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::log_softmax(input, F::LogSoftmaxFuncOptions(1));
/// ```
inline Tensor log_softmax(const Tensor& input, const LogSoftmaxFuncOptions& options) {
  return detail::log_softmax(input, options.dim(), options.dtype());
}
    // 定义一个返回类型为 Tensor 的函数 log_softmax，可以接受两种不同的参数形式
    // 如果传入的 dtype 是空的 optional，则使用 input 在维度 dim 上的 log_softmax 操作
    // 否则，使用指定的 dtype 进行 input 在维度 dim 上的 log_softmax 操作
    std::optional<torch::Dtype> dtype) {
  // 定义一个 Tensor 类型的变量 ret
  Tensor ret;

  // 根据传入的参数情况选择不同的 log_softmax 方法，并将结果赋给 ret
  if (dtype == c10::nullopt) {
    ret = input.log_softmax(dim);
  } else {
    ret = input.log_softmax(dim, dtype);
  }

  // 返回计算得到的 Tensor 结果
  return ret;
}


这段代码定义了一个函数 `log_softmax`，它可以根据输入的参数来选择不同的操作：如果 `dtype` 是空的 `optional`，则调用 `input` 的 `log_softmax` 方法；否则，使用指定的 `dtype` 调用 `input` 的 `log_softmax` 方法。最后将计算结果以 `Tensor` 类型返回。
#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
// 在指定维度上计算 log_softmax 的内部函数
inline Tensor log_softmax(const Tensor& input, int64_t dim, c10::optional<ScalarType> dtype=c10::nullopt) {
  // 使用 detail 命名空间中的 log_softmax 函数进行计算
  return torch::log_softmax(input, dim, dtype);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// 查看此函数的详细行为，请访问：
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.log_softmax
///
/// 若要了解此函数支持的可选参数，请参阅 `torch::nn::functional::LogSoftmaxFuncOptions` 类的文档。

/// 示例：
/// ```
/// namespace F = torch::nn::functional;
/// F::log_softmax(input, LogSoftmaxFuncOptions(1));
/// ```
inline Tensor log_softmax(
    const Tensor& input,
    const LogSoftmaxFuncOptions& options) {
  // 调用 detail 命名空间中的 log_softmax 函数，并返回结果
  return detail::log_softmax(input, options.dim(), options.dtype());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
// 在指定维度上执行 glu 操作的内部函数
inline Tensor glu(const Tensor& input, int64_t dim) {
  // 检查输入的维度是否为零，并抛出相应错误信息
  TORCH_CHECK(
      input.dim() != 0,
      "glu does not suppport scalars because halving size must be even");
  // 使用 torch::glu 函数进行 glu 操作
  return torch::glu(input, dim);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// 查看此函数的详细行为，请访问：
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.glu
///
/// 若要了解此函数支持的可选参数，请参阅 `torch::nn::functional::GLUFuncOptions` 类的文档。

/// 示例：
/// ```
/// namespace F = torch::nn::functional;
/// F::glu(input, GLUFuncOptions(1));
/// ```
inline Tensor glu(const Tensor& input, const GLUFuncOptions& options = {}) {
  // 调用 detail 命名空间中的 glu 函数，并返回结果
  return detail::glu(input, options.dim());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
// 执行 gelu 激活函数的内部函数
inline Tensor gelu(const Tensor& input, string approximate) {
  // 使用 torch::gelu 函数执行 gelu 激活函数
  return torch::gelu(input, approximate);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// 执行 gelu 激活函数的外部接口函数
inline Tensor gelu(const Tensor& input, const GELUFuncOptions& options = {}) {
  // 调用 detail 命名空间中的 gelu 函数，并返回结果
  return detail::gelu(input, options.approximate());
}

// ============================================================================

/// 执行 silu 激活函数的函数
inline Tensor silu(const Tensor& input) {
  // 使用 torch::silu 函数执行 silu 激活函数
  return torch::silu(input);
}

// ============================================================================

/// 执行 mish 激活函数的函数
inline Tensor mish(const Tensor& input) {
  // 使用 torch::mish 函数执行 mish 激活函数
  return torch::mish(input);
}

// ============================================================================

/// 执行 prelu 激活函数的函数
inline Tensor prelu(const Tensor& input, const Tensor& weight) {
  // 使用 torch::prelu 函数执行 prelu 激活函数
  return torch::prelu(input, weight);
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
// 在指定维度上执行 relu 激活函数的内部函数
inline Tensor relu(Tensor input, bool inplace) {
  // 根据 inplace 参数选择是否原地执行 relu 激活函数
  if (inplace) {
    return torch::relu_(input);
  } else {
    return torch::relu(input);
  }
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// 查看此函数的详细行为，请访问：
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.relu
///
/// 此处未完待续...
/// 根据传入的输入张量和可选参数选项执行 ReLU 激活函数。
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::relu(x, F::ReLUFuncOptions().inplace(true));
/// ```
inline Tensor relu(Tensor input, const ReLUFuncOptions& options = {}) {
  return detail::relu(std::move(input), options.inplace());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
/// 执行 inplace 或非 inplace 的 ReLU6 激活函数，根据参数选择相应的操作。
inline Tensor relu6(Tensor input, bool inplace) {
  if (inplace) {
    return torch::relu6_(input);
  } else {
    return torch::relu6(input);
  }
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// 根据传入的输入张量和可选参数选项执行 ReLU6 激活函数。
/// 
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::relu6(x, F::ReLU6FuncOptions().inplace(true));
/// ```
inline Tensor relu6(Tensor input, const ReLU6FuncOptions& options = {}) {
  return detail::relu6(std::move(input), options.inplace());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
/// 执行 inplace 或非 inplace 的 RReLU 激活函数，根据参数选择相应的操作。
inline Tensor rrelu(
    Tensor input,
    double lower,
    double upper,
    bool training,
    bool inplace) {
  if (inplace) {
    return torch::rrelu_(input, lower, upper, training);
  } else {
    return torch::rrelu(input, lower, upper, training);
  }
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// 根据传入的输入张量和可选参数选项执行 RReLU 激活函数。
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::rrelu(x, F::RReLUFuncOptions().lower(0.1).upper(0.4).inplace(true));
/// ```
inline Tensor rrelu(Tensor input, const RReLUFuncOptions& options = {}) {
  return detail::rrelu(
      std::move(input),
      options.lower(),
      options.upper(),
      options.training(),
      options.inplace());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
/// 执行 inplace 或非 inplace 的 CELU 激活函数，根据参数选择相应的操作。
inline Tensor celu(Tensor input, double alpha, bool inplace) {
  if (inplace) {
    return torch::celu_(input, alpha);
  } else {
    return torch::celu(input, alpha);
  }
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// 根据传入的输入张量和参数执行 CELU 激活函数。
/// 
/// See
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.celu
/// 了解此函数的确切行为。
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::celu(x, 0.1);  // 默认不支持 inplace 操作
/// ```
inline Tensor celu(Tensor input, double alpha, bool inplace = false) {
  return detail::celu(std::move(input), alpha, inplace);
}
/// 使用 CELU 函数对输入张量进行操作，可选参数详见 `torch::nn::functional::CELUFuncOptions` 类的文档。
///
/// 示例：
/// ```
/// namespace F = torch::nn::functional;
/// F::celu(x, F::CELUFuncOptions().alpha(0.42).inplace(true));
/// ```
inline Tensor celu(Tensor input, const CELUFuncOptions& options = {}) {
  // 调用 detail 命名空间中的 celu 函数，传递输入张量、alpha 和 inplace 参数
  return detail::celu(std::move(input), options.alpha(), options.inplace());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
/// 实现 softplus 函数的细节，传递输入张量及 beta、threshold 参数
inline Tensor softplus(const Tensor& input, double beta, double threshold) {
  return torch::softplus(input, beta, threshold);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// 查看 https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.softplus
/// 了解此函数的确切行为。
///
/// 详细了解 `torch::nn::functional::SoftplusFuncOptions` 类的文档以了解此函数支持的可选参数。
///
/// 示例：
/// ```
/// namespace F = torch::nn::functional;
/// F::softplus(x, F::SoftplusFuncOptions().beta(0.5).threshold(3.0));
/// ```
inline Tensor softplus(
    const Tensor& input,
    const SoftplusFuncOptions& options = {}) {
  // 调用 detail 命名空间中的 softplus 函数，传递输入张量及其选项中的 beta 和 threshold 参数
  return detail::softplus(input, options.beta(), options.threshold());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
/// 实现 softshrink 函数的细节，传递输入张量及 lambda 参数
inline Tensor softshrink(const Tensor& input, double lambda) {
  return torch::softshrink(input, lambda);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// 查看 https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.softshrink
/// 了解此函数的确切行为。
///
/// 详细了解 `torch::nn::functional::SoftshrinkFuncOptions` 类的文档以了解此函数支持的可选参数。
///
/// 示例：
/// ```
/// namespace F = torch::nn::functional;
/// F::softshrink(x, F::SoftshrinkFuncOptions(0.42));
/// ```
inline Tensor softshrink(
    const Tensor& input,
    const SoftshrinkFuncOptions& options = {}) {
  // 调用 detail 命名空间中的 softshrink 函数，传递输入张量及其选项中的 lambda 参数
  return detail::softshrink(input, options.lambda());
}

// ============================================================================

/// 实现 softsign 函数，对输入张量进行操作
inline Tensor softsign(const Tensor& input) {
  // 返回输入张量除以其绝对值加一的结果
  return input / (input.abs() + 1);
}

// ============================================================================

/// 实现 tanhshrink 函数，对输入张量进行操作
inline Tensor tanhshrink(const Tensor& input) {
  // 返回输入张量减去其双曲正切函数的结果
  return input - input.tanh();
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
/// 实现 threshold 函数的细节，传递输入张量、threshold、value 和 inplace 参数
inline Tensor threshold(
    Tensor input,
    double threshold,
    double value,
    bool inplace) {
  // 根据 inplace 参数选择性地调用 torch 库中的 threshold_ 或 threshold 函数
  if (inplace) {
    return torch::threshold_(input, threshold, value);
  } else {
    return torch::threshold(input, threshold, value);
  }
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */
/// 如果 DOXYGEN_SHOULD_SKIP_THIS 没有定义，则进入以下代码块
#ifndef DOXYGEN_SHOULD_SKIP_THIS
/// 定义了一个名为 detail 的命名空间，包含了多头注意力机制的前向传播函数
namespace detail {
    /// 实现多头注意力机制的前向传播
    ///
    /// 参数说明：
    /// - query: 查询张量
    /// - key: 键张量
    /// - value: 值张量
    /// - embed_dim_to_check: 检查嵌入维度是否匹配的参数
    /// - num_heads: 注意力头的数量
    /// - in_proj_weight: 输入投影权重张量
    /// - in_proj_bias: 输入投影偏置张量
    /// - bias_k: 键的偏置张量
    /// - bias_v: 值的偏置张量
    /// - add_zero_attn: 是否添加零注意力
    /// - dropout_p: dropout 概率
    /// - out_proj_weight: 输出投影权重张量
    /// - out_proj_bias: 输出投影偏置张量
    /// - training: 是否处于训练模式，默认为 true
    /// - key_padding_mask: 键的填充遮罩张量，默认为空张量
    /// - need_weights: 是否需要权重，默认为 true
    /// - attn_mask: 注意力遮罩张量，默认为空张量
    /// - use_separate_proj_weight: 是否使用分开的投影权重，默认为 false
    /// - q_proj_weight: 查询投影权重张量，默认为空张量
    /// - k_proj_weight: 键投影权重张量，默认为空张量
    /// - v_proj_weight: 值投影权重张量，默认为空张量
    /// - static_k: 静态键张量，默认为空张量
    /// - static_v: 静态值张量，默认为空张量
    /// - average_attn_weights: 是否平均注意力权重，默认为 true
    ///
    /// 返回：
    /// 返回一个包含两个张量的元组，分别表示注意力输出和注意力权重
    inline std::tuple<Tensor, Tensor> multi_head_attention_forward(
        const Tensor& query,
        const Tensor& key,
        const Tensor& value,
        int64_t embed_dim_to_check,
        int64_t num_heads,
        const Tensor& in_proj_weight,
        const Tensor& in_proj_bias,
        const Tensor& bias_k,
        const Tensor& bias_v,
        bool add_zero_attn,
        double dropout_p,
        const Tensor& out_proj_weight,
        const Tensor& out_proj_bias,
        bool training = true,
        const Tensor& key_padding_mask = {},
        bool need_weights = true,
        const Tensor& attn_mask = {},
        bool use_separate_proj_weight = false,
        const Tensor& q_proj_weight = {},
        const Tensor& k_proj_weight = {},
        const Tensor& v_proj_weight = {},
        const Tensor& static_k = {},
        const Tensor& static_v = {},
        bool average_attn_weights = true) {
        
        namespace F = torch::nn::functional;

        // 获取查询张量的大小
        const auto query_sizes = query.sizes();
        const auto& tgt_len = query_sizes[0];
        const auto& bsz = query_sizes[1];
        const auto& embed_dim = query_sizes[2];

        // 内部断言，确保嵌入维度与检查的维度一致
        TORCH_INTERNAL_ASSERT(embed_dim == embed_dim_to_check);
        // 内部断言，确保键张量和值张量的大小一致
        TORCH_INTERNAL_ASSERT(key.sizes() == value.sizes());

        // 计算每个头的维度
        const auto head_dim = embed_dim / num_heads;
        TORCH_CHECK(
            head_dim * num_heads == embed_dim,
            "embed_dim must be divisible by num_heads");
        // 计算缩放因子
        const auto scaling = 1 / std::sqrt(head_dim);

        // 定义查询、键、值张量
        Tensor q, k, v;
        // 如果不使用分开的投影权重
        if (!use_separate_proj_weight) {
            // 如果查询张量、键张量和值张量相等，则为自注意力
            if (torch::equal(query, key) && torch::equal(key, value)) {
                // 将查询张量通过线性变换分成三个块
                const auto chunks =
                    F::linear(query, in_proj_weight, in_proj_bias).chunk(3, /*dim=*/-1);
                // 分别赋值给查询、键、值张量
                q = chunks[0];
                k = chunks[1];
                v = chunks[2];
                ```
    } else if (torch::equal(key, value)) {
      // encoder-decoder attention
      // This is inline in_proj function with in_proj_weight and in_proj_bias
      
      // 拷贝偏置到局部变量 _b
      auto _b = in_proj_bias;
      // 设置起始和结束位置
      auto _start = 0;
      auto _end = embed_dim;
      // 从权重矩阵中切片获取需要的部分
      auto _w = in_proj_weight.slice(/*dim=*/0, _start, _end);
      // 如果偏置定义了，则也从偏置中切片获取需要的部分
      if (_b.defined()) {
        _b = _b.slice(/*dim=*/0, _start, _end);
      }
      // 使用线性函数对查询进行处理
      q = F::linear(query, _w, _b);

      // 如果键未定义，则重置键和值
      if (!key.defined()) {
        TORCH_INTERNAL_ASSERT(!value.defined());
        k.reset();
        v.reset();
      } else {
        // 再次执行 inline in_proj 函数，处理键值对
        _b = in_proj_bias;
        _start = embed_dim;
        _w = in_proj_weight.slice(/*dim=*/0, _start);
        // 如果偏置定义了，则也从偏置中切片获取需要的部分
        if (_b.defined()) {
          _b = _b.slice(/*dim=*/0, _start);
        }
        // 使用线性函数处理键，返回两个部分的块
        const auto chunks = F::linear(key, _w, _b).chunk(2, /*dim=*/-1);
        k = chunks[0];
        v = chunks[1];
      }
    } else {
      // 这是 inline in_proj 函数，处理查询
      auto _b = in_proj_bias;
      auto _start = 0;
      auto _end = embed_dim;
      auto _w = in_proj_weight.slice(/*dim=*/0, _start, _end);
      // 如果偏置定义了，则也从偏置中切片获取需要的部分
      if (_b.defined()) {
        _b = _b.slice(/*dim=*/0, _start, _end);
      }
      // 使用线性函数处理查询
      q = F::linear(query, _w, _b);

      // 这是 inline in_proj 函数，处理键
      _b = in_proj_bias;
      _start = embed_dim;
      _end = embed_dim * 2;
      _w = in_proj_weight.slice(/*dim=*/0, _start, _end);
      // 如果偏置定义了，则也从偏置中切片获取需要的部分
      if (_b.defined()) {
        _b = _b.slice(/*dim=*/0, _start, _end);
      }
      // 使用线性函数处理键
      k = F::linear(key, _w, _b);

      // 这是 inline in_proj 函数，处理值
      _b = in_proj_bias;
      _start = embed_dim * 2;
      _w = in_proj_weight.slice(/*dim=*/0, _start);
      // 如果偏置定义了，则也从偏置中切片获取需要的部分
      if (_b.defined()) {
        _b = _b.slice(0, _start);
      }
      // 使用线性函数处理值
      v = F::linear(value, _w, _b);
    }
  } else {
    // 获取非优化的查询投影权重
    const auto& q_proj_weight_non_opt = q_proj_weight;
    {
      // 检查权重的尺寸是否符合预期
      const auto sizes = q_proj_weight_non_opt.sizes();
      const auto len1 = sizes[0];
      const auto len2 = sizes[1];
      TORCH_CHECK(len1 == embed_dim && len2 == query.size(-1));
    }

    // 获取非优化的键投影权重
    const auto& k_proj_weight_non_opt = k_proj_weight;
    {
      // 检查权重的尺寸是否符合预期
      const auto sizes = k_proj_weight_non_opt.sizes();
      const auto len1 = sizes[0];
      const auto len2 = sizes[1];
      TORCH_CHECK(len1 == embed_dim && len2 == key.size(-1));
    }

    // 获取非优化的值投影权重
    const auto& v_proj_weight_non_opt = v_proj_weight;
    {
      // 检查权重的尺寸是否符合预期
      const auto sizes = v_proj_weight_non_opt.sizes();
      const auto len1 = sizes[0];
      const auto len2 = sizes[1];
      TORCH_CHECK(len1 == embed_dim && len2 == value.size(-1));
    }
    // 检查是否定义了输入投影偏置
    if (in_proj_bias.defined()) {
      // 使用非优化的权重和输入投影偏置对查询进行线性变换
      q = F::linear(
          query,
          q_proj_weight_non_opt,
          in_proj_bias.slice(/*dim=*/0, 0, embed_dim));
      // 使用非优化的权重和输入投影偏置对键进行线性变换
      k = F::linear(
          key,
          k_proj_weight_non_opt,
          in_proj_bias.slice(/*dim=*/0, embed_dim, (embed_dim * 2)));
      // 使用非优化的权重和输入投影偏置对值进行线性变换
      v = F::linear(
          value,
          v_proj_weight_non_opt,
          in_proj_bias.slice(/*dim=*/0, (embed_dim * 2)));
    } else {
      // 如果未定义输入投影偏置，则使用统一的输入投影偏置对查询、键和值进行线性变换
      q = F::linear(query, q_proj_weight_non_opt, in_proj_bias);
      k = F::linear(key, k_proj_weight_non_opt, in_proj_bias);
      v = F::linear(value, v_proj_weight_non_opt, in_proj_bias);
    }
  }
  // 对查询进行缩放
  q = q * scaling;
  // 复制注意力机制中的掩码和键值对的填充掩码
  Tensor attn_mask_ = attn_mask;
  Tensor key_padding_mask_ = key_padding_mask;
  // 如果定义了偏置键和偏置值
  if (bias_k.defined() && bias_v.defined()) {
    // 如果静态键和静态值都未定义，则将偏置键和偏置值重复添加到键和值中
    if (!static_k.defined() && !static_v.defined()) {
      k = torch::cat({k, bias_k.repeat({1, bsz, 1})});
      v = torch::cat({v, bias_v.repeat({1, bsz, 1})});
      // 如果定义了注意力掩码，则在其末尾添加一列零向量
      if (attn_mask_.defined()) {
        attn_mask_ = torch::cat(
            {attn_mask_,
             torch::zeros(
                 {attn_mask_.size(0), 1},
                 at::TensorOptions(attn_mask_.dtype())
                     .device(attn_mask_.device()))},
            /*dim=*/1);
      }
      // 如果定义了键值对的填充掩码，则在其末尾添加一列零向量
      if (key_padding_mask_.defined()) {
        key_padding_mask_ = torch::cat(
            {key_padding_mask_,
             torch::zeros(
                 {key_padding_mask_.size(0), 1},
                 at::TensorOptions(key_padding_mask_.dtype())
                     .device(key_padding_mask_.device()))},
            /*dim=*/1);
      }
    } else {
      // 如果静态键或静态值已定义，则报错，因为不能向静态键或静态值添加偏置
      TORCH_CHECK(!static_k.defined(), "bias cannot be added to static key.");
      TORCH_CHECK(!static_v.defined(), "bias cannot be added to static value.");
    }
  } else {
    // 如果未定义偏置键或偏置值，则报错
    TORCH_CHECK(!bias_k.defined());
    TORCH_CHECK(!bias_v.defined());
  }
  // 将查询张量重塑并转置，以便于后续计算
  q = q.contiguous().view({tgt_len, bsz * num_heads, head_dim}).transpose(0, 1);
  // 如果键张量已定义，则将其重塑并转置，以便于后续计算
  if (k.defined()) {
    k = k.contiguous().view({-1, bsz * num_heads, head_dim}).transpose(0, 1);
  }
  // 如果值张量已定义，则将其重塑并转置，以便于后续计算
  if (v.defined()) {
    v = v.contiguous().view({-1, bsz * num_heads, head_dim}).transpose(0, 1);
  }
  // 如果静态键已定义，则检查其形状是否符合预期
  if (static_k.defined()) {
    TORCH_CHECK(static_k.size(0) == bsz * num_heads);
    TORCH_CHECK(static_k.size(2) == head_dim);
    // 直接使用静态键
    k = static_k;
  }
  // 如果静态值已定义，则检查其形状是否符合预期
  if (static_v.defined()) {
    TORCH_CHECK(static_v.size(0) == bsz * num_heads);
    TORCH_CHECK(static_v.size(2) == head_dim);
    // 直接使用静态值
    v = static_v;
  }
  // 获取键的长度（通常是源序列的长度）
  auto src_len = k.size(1);
  // 如果定义了键值对的填充掩码，则检查其形状是否符合预期
  if (key_padding_mask_.defined()) {
    TORCH_CHECK(key_padding_mask_.size(0) == bsz);
    TORCH_CHECK(key_padding_mask_.size(1) == src_len);
  }
  // 如果需要在注意力矩阵中添加零向量
  if (add_zero_attn) {
    // 增加键张量的长度
    src_len += 1;
    // 将零向量添加到键张量中
    auto k_sizes = k.sizes().vec();
    k_sizes[1] = 1;
    k = torch::cat(
        {k,
         torch::zeros(
             k_sizes, at::TensorOptions(k.dtype()).device(k.device()))},
        /*dim=*/1);
    // 将零向量添加到值张量中
    auto v_sizes = v.sizes().vec();
    v_sizes[1] = 1;
    // 沿着第二个维度（dim=1）将张量 v 和一个与 v 相同大小的零张量连接起来，扩展 v 的大小
    v = torch::cat(
        {v,
         torch::zeros(
             v_sizes, at::TensorOptions(v.dtype()).device(v.device()))},
        /*dim=*/1);
    // 如果定义了注意力遮罩 attn_mask_
    if (attn_mask_.defined()) {
      // 在第二个维度（dim=1）将注意力遮罩 attn_mask_ 和一个额外列的零张量连接起来，扩展 attn_mask_ 的大小
      attn_mask_ = torch::cat(
          {attn_mask_,
           torch::zeros(
               {attn_mask_.size(0), 1},
               at::TensorOptions(attn_mask_.dtype())
                   .device(attn_mask_.device()))},
          /*dim=*/1);
    }
    // 如果定义了键填充遮罩 key_padding_mask_
    if (key_padding_mask_.defined()) {
      // 在第二个维度（dim=1）将键填充遮罩 key_padding_mask_ 和一个额外列的零张量连接起来，扩展 key_padding_mask_ 的大小
      key_padding_mask_ = torch::cat(
          {key_padding_mask_,
           torch::zeros(
               {key_padding_mask_.size(0), 1},
               at::TensorOptions(key_padding_mask_.dtype())
                   .device(key_padding_mask_.device()))},
          /*dim=*/1);
    }
  }
  // 计算注意力权重，q 与 k 的转置相乘
  auto attn_output_weights = torch::bmm(q, k.transpose(1, 2));
  // 检查注意力权重的维度是否符合预期
  TORCH_CHECK(
      attn_output_weights.sizes() ==
      IntArrayRef({bsz * num_heads, tgt_len, src_len}));
  // 如果定义了注意力遮罩 attn_mask_
  if (attn_mask_.defined()) {
    // 在第一维度（0维）增加一个维度来匹配注意力权重的维度，然后将注意力遮罩加到注意力权重上
    attn_mask_ = attn_mask_.unsqueeze(0);
    attn_output_weights += attn_mask_;
  }
  // 如果定义了键填充遮罩 key_padding_mask_
  if (key_padding_mask_.defined()) {
    // 重新调整注意力权重的形状以便进行遮罩填充
    attn_output_weights =
        attn_output_weights.view({bsz, num_heads, tgt_len, src_len});
    // 根据键填充遮罩进行填充操作，使用 -∞ 替换遮罩处的值
    attn_output_weights = AT_DISPATCH_FLOATING_TYPES(
        attn_output_weights.scalar_type(),
        "attn_output_weights.masked_fill",
        [&]() {
          return attn_output_weights.masked_fill(
              key_padding_mask_.unsqueeze(1).unsqueeze(2),
              -std::numeric_limits<scalar_t>::infinity());
        });
    // 恢复注意力权重的形状
    attn_output_weights =
        attn_output_weights.view({bsz * num_heads, tgt_len, src_len});
  }
  // 对注意力权重进行 softmax 归一化，沿着最后一个维度（dim=-1）
  attn_output_weights = F::softmax(attn_output_weights, /*dim=*/-1);
  // 对注意力权重进行 dropout 操作
  attn_output_weights = F::dropout(
      attn_output_weights,
      F::DropoutFuncOptions().p(dropout_p).training(training));
  // 计算注意力输出，注意力权重与 v 的乘积
  auto attn_output = torch::bmm(attn_output_weights, v);
  // 检查注意力输出的维度是否符合预期
  TORCH_CHECK(
      attn_output.sizes() == IntArrayRef({bsz * num_heads, tgt_len, head_dim}));
  // 调整注意力输出的顺序并重塑为预期的形状
  attn_output =
      attn_output.transpose(0, 1).contiguous().view({tgt_len, bsz, embed_dim});
  // 使用线性变换将注意力输出映射到期望的输出维度
  attn_output = F::linear(attn_output, out_proj_weight, out_proj_bias);
  // 如果需要返回权重
  if (need_weights) {
    // 调整注意力权重的形状
    attn_output_weights =
        attn_output_weights.view({bsz, num_heads, tgt_len, src_len});
    // 如果需要平均注意力权重
    if (average_attn_weights) {
      // 在头部维度上对注意力权重进行平均
      attn_output_weights = attn_output_weights.sum(/*dim=*/1) / num_heads;
    }
    // 返回注意力输出和注意力权重的元组
    return std::make_tuple(attn_output, attn_output_weights);
  } else {
    // 返回注意力输出和空张量的元组
    return std::make_tuple(attn_output, Tensor());
  }
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */



inline std::tuple<Tensor, Tensor> multi_head_attention_forward(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const MultiheadAttentionForwardFuncOptions& options) {
  // 调用命名空间 detail 中的 multi_head_attention_forward 函数，执行多头注意力机制的前向传播
  return detail::multi_head_attention_forward(
      query,
      key,
      value,
      // 以下为各种选项参数，传递给 multi_head_attention_forward 函数
      options.embed_dim_to_check(),
      options.num_heads(),
      options.in_proj_weight(),
      options.in_proj_bias(),
      options.bias_k(),
      options.bias_v(),
      options.add_zero_attn(),
      options.dropout_p(),
      options.out_proj_weight(),
      options.out_proj_bias(),
      options.training(),
      options.key_padding_mask(),
      options.need_weights(),
      options.attn_mask(),
      options.use_separate_proj_weight(),
      options.q_proj_weight(),
      options.k_proj_weight(),
      options.v_proj_weight(),
      options.static_k(),
      options.static_v(),
      options.average_attn_weights());
}

} // namespace functional
} // namespace nn
} // namespace torch



// 结束命名空间 torch
```