# `.\pytorch\torch\csrc\api\include\torch\nn\functional\loss.h`

```
#pragma once

#include <ATen/ExpandUtils.h>
#include <torch/nn/functional/activation.h>
#include <torch/nn/options/loss.h>

namespace torch {
namespace nn {
namespace functional {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
// 定义 l1_loss 函数，计算 L1 损失
inline Tensor l1_loss(
    const Tensor& input,  // 输入张量
    const Tensor& target,  // 目标张量
    L1LossFuncOptions::reduction_t reduction) {  // 减少选项
  return torch::l1_loss(input, target, enumtype::reduction_get_enum(reduction));  // 调用 torch 中的 l1_loss 函数
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.l1_loss
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::L1LossFuncOptions` class
/// to learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::l1_loss(input, target, F::L1LossFuncOptions(torch::kNone));
/// ```
// 定义 l1_loss 函数的外部接口
inline Tensor l1_loss(
    const Tensor& input,  // 输入张量
    const Tensor& target,  // 目标张量
    const L1LossFuncOptions& options = {}) {  // L1 损失函数的选项，默认为空选项
  return detail::l1_loss(input, target, options.reduction());  // 调用内部 detail 命名空间中的 l1_loss 函数
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
// 定义 kl_div 函数，计算 KL 散度
inline Tensor kl_div(
    const Tensor& input,  // 输入张量
    const Tensor& target,  // 目标张量
    KLDivFuncOptions::reduction_t reduction,  // 减少选项
    bool log_target = false) {  // 是否对目标张量取对数，默认为 false
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  torch::Reduction::Reduction reduction_enum;

  if (std::holds_alternative<enumtype::kMean>(reduction)) {  // 如果减少选项是均值
    TORCH_WARN(
        "reduction: 'mean' divides the total loss by both the batch size and the support size."
        "'batchmean' divides only by the batch size, and aligns with the KL div math definition."
        "'mean' will be changed to behave the same as 'batchmean' in the next major release.");
  }

  // 特殊处理 batchmean 的情况
  if (std::holds_alternative<enumtype::kBatchMean>(reduction)) {
    reduction_enum = torch::Reduction::Sum;  // 使用求和作为减少方法
  } else {
    reduction_enum = enumtype::reduction_get_enum(reduction);  // 根据枚举值获取减少方法
  }

  auto reduced = torch::kl_div(input, target, reduction_enum, log_target);  // 调用 torch 中的 kl_div 函数

  if (std::holds_alternative<enumtype::kBatchMean>(reduction) &&  // 如果减少选项是 batchmean 并且输入张量不是零维
      input.dim() != 0) {
    reduced = reduced / input.sizes()[0];  // 对结果进行除以批次大小
  }

  return reduced;  // 返回计算的 KL 散度结果
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.kl_div
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::KLDivFuncOptions` class to
/// learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::kl_div(input, target,
/// F::KLDivFuncOptions.reduction(torch::kNone).log_target(false));
/// ```
// 定义 kl_div 函数的外部接口
inline Tensor kl_div(
    const Tensor& input,  // 输入张量
    const Tensor& target,  // 目标张量
    const KLDivFuncOptions& options = {}) {  // KL 散度函数的选项，默认为空选项
    // 减少选项，默认为空选项
    // 使用传入的参数 options 创建一个常量引用对象 options，类型为 KLDivFuncOptions
    const KLDivFuncOptions& options = {}) {
    // 调用 detail 命名空间中的 kl_div 函数，传入 input 和 target 参数，
    // 并根据 options 对象中的设置来确定如何处理结果的减少(reduction)和目标对数(log_target)选项
    return detail::kl_div(
        input, target, options.reduction(), options.log_target());
// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
// 计算均方误差损失的内部函数
inline Tensor mse_loss(
    // 输入张量
    const Tensor& input,
    // 目标张量
    const Tensor& target,
    // 均方误差损失函数的降维选项
    MSELossFuncOptions::reduction_t reduction) {
  // 检查目标张量和输入张量的尺寸是否相同
  if (!(target.sizes() == input.sizes())) {
    TORCH_WARN(
        "Using a target size (",
        target.sizes(),
        ") that is different to the input size (",
        input.sizes(),
        "). ",
        "This will likely lead to incorrect results due to broadcasting. ",
        "Please ensure they have the same size.");
  }
  // 对输入张量和目标张量进行广播
  std::vector<torch::Tensor> broadcast_tensors =
      torch::broadcast_tensors({input, target});
  auto expanded_input = broadcast_tensors[0];
  auto expanded_target = broadcast_tensors[1];
  // 调用 PyTorch 提供的均方误差损失函数
  return torch::mse_loss(
      expanded_input, expanded_target, enumtype::reduction_get_enum(reduction));
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// 查看
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.mse_loss
/// 以了解此函数的确切行为。
///
/// 查看 `torch::nn::functional::MSELossFuncOptions` 类的文档，了解此函数支持的可选参数。
///
/// 示例:
/// ```
/// namespace F = torch::nn::functional;
/// F::mse_loss(input, target, F::MSELossFuncOptions(torch::kNone));
/// ```
inline Tensor mse_loss(
    // 输入张量
    const Tensor& input,
    // 目标张量
    const Tensor& target,
    // 均方误差损失函数的选项（可选）
    const MSELossFuncOptions& options = {}) {
  // 调用内部的均方误差损失函数
  return detail::mse_loss(input, target, options.reduction());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
// 计算二元交叉熵损失的内部函数
inline Tensor binary_cross_entropy(
    // 输入张量
    const Tensor& input,
    // 目标张量
    const Tensor& target,
    // 权重张量
    const Tensor& weight,
    // 二元交叉熵损失函数的降维选项
    BinaryCrossEntropyFuncOptions::reduction_t reduction) {
  // 获取降维的枚举值
  auto reduction_enum = enumtype::reduction_get_enum(reduction);

  // 检查目标张量和输入张量的尺寸是否相同
  if (target.sizes() != input.sizes()) {
    TORCH_CHECK(
        false,
        "Using a target size (",
        target.sizes(),
        ") ",
        "that is different to the input size (",
        input.sizes(),
        ") is deprecated. ",
        "Please ensure they have the same size.");
  }

  // 处理权重张量，使其与目标张量兼容
  auto weight_ = weight;
  if (weight_.defined()) {
    auto new_size = at::infer_size(target.sizes(), weight_.sizes());
    weight_ = weight_.expand(new_size);
  }

  // 调用 PyTorch 提供的二元交叉熵损失函数
  return torch::binary_cross_entropy(input, target, weight_, reduction_enum);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// 查看
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.binary_cross_entropy
/// 以了解此函数的确切行为。
///
/// 查看 `torch::nn::functional::BinaryCrossEntropyFuncOptions` 类的文档，了解此函数支持的可选参数。
///
/// 示例:
/// ```
/// namespace F = torch::nn::functional;
/// F::binary_cross_entropy(input, target,
/// ```
/// F::BinaryCrossEntropyFuncOptions().weight(weight));
/// ```
inline Tensor binary_cross_entropy(
    const Tensor& input,
    const Tensor& target,
    const BinaryCrossEntropyFuncOptions& options = {}) {
  // 调用内部函数 detail::binary_cross_entropy 处理二元交叉熵计算
  return detail::binary_cross_entropy(
      input, target, options.weight(), options.reduction());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
// 内部函数，计算 hinge embedding loss
inline Tensor hinge_embedding_loss(
    const Tensor& input,
    const Tensor& target,
    double margin,
    HingeEmbeddingLossFuncOptions::reduction_t reduction) {
  // 调用 PyTorch 提供的 hinge_embedding_loss 函数计算损失
  return torch::hinge_embedding_loss(
      input, target, margin, enumtype::reduction_get_enum(reduction));
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.hinge_embedding_loss
/// about the exact behavior of this functional.
///
/// See the documentation for
/// `torch::nn::functional::HingeEmbeddingLossFuncOptions` class to learn what
/// optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::hinge_embedding_loss(input, target,
/// F::HingeEmbeddingLossFuncOptions().margin(2));
/// ```
inline Tensor hinge_embedding_loss(
    const Tensor& input,
    const Tensor& target,
    const HingeEmbeddingLossFuncOptions& options = {}) {
  // 调用内部函数 detail::hinge_embedding_loss 处理 hinge embedding loss 计算
  return detail::hinge_embedding_loss(
      input, target, options.margin(), options.reduction());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
// 内部函数，计算 multi-margin loss
inline Tensor multi_margin_loss(
    const Tensor& input,
    const Tensor& target,
    int64_t p,
    double margin,
    const Tensor& weight,
    MultiMarginLossFuncOptions::reduction_t reduction) {
  // 检查 p 只支持 1 和 2
  TORCH_CHECK(p == 1 || p == 2, "only p == 1 and p == 2 supported");
  // 如果 weight 定义了，检查其维度是否为一维
  if (weight.defined()) {
    TORCH_CHECK(weight.dim() == 1, "weight must be one-dimensional");
  }

  // 调用 PyTorch 提供的 multi_margin_loss 函数计算损失
  return torch::multi_margin_loss(
      input,
      target,
      p,
      margin,
      weight,
      enumtype::reduction_get_enum(reduction));
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.multi_margin_loss
/// about the exact behavior of this functional.
///
/// See the documentation for
/// `torch::nn::functional::MultiMarginLossFuncOptions` class to learn what
/// optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::multi_margin_loss(input, target,
/// F::MultiMarginLossFuncOptions().margin(2).weight(weight));
/// ```
inline Tensor multi_margin_loss(
    const Tensor& input,
    const Tensor& target,
    int64_t p,
    double margin = 1.0,
    const Tensor& weight = {},
    MultiMarginLossFuncOptions::reduction_t reduction = MultiMarginLossFuncOptions::Reduction::Mean) {
  // 调用内部函数 detail::multi_margin_loss 处理 multi-margin loss 计算
  return detail::multi_margin_loss(
      input,
      target,
      p,
      margin,
      weight,
      reduction);
}
    // 定义一个常量引用 options，类型为 MultiMarginLossFuncOptions，并且默认为空对象
    const MultiMarginLossFuncOptions& options = {}) {
    // 调用 detail 命名空间中的 multi_margin_loss 函数，传入以下参数：
    // - input：输入张量
    // - target：目标张量
    // - options.p()：选项中的 p 参数值
    // - options.margin()：选项中的 margin 参数值
    // - options.weight()：选项中的 weight 参数值
    // - options.reduction()：选项中的 reduction 参数值
    // 返回 multi_margin_loss 函数的结果
    return detail::multi_margin_loss(
        input,
        target,
        options.p(),
        options.margin(),
        options.weight(),
        options.reduction());
#ifndef DOXYGEN_SHOULD_SKIP_THIS
// 如果未定义 DOXYGEN_SHOULD_SKIP_THIS 宏，则定义 detail 命名空间
namespace detail {
// 定义一个函数，计算余弦嵌入损失
inline Tensor cosine_embedding_loss(
    const Tensor& input1, // 第一个输入张量
    const Tensor& input2, // 第二个输入张量
    const Tensor& target, // 目标张量
    double margin, // 余弦嵌入损失的边界
    CosineEmbeddingLossFuncOptions::reduction_t reduction) { // 减少选项
  // 调用 PyTorch 的 cosine_embedding_loss 函数计算余弦嵌入损失
  return torch::cosine_embedding_loss(
      input1, input2, target, margin, enumtype::reduction_get_enum(reduction));
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// 查看这个函数的具体行为，请参阅
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.cosine_embedding_loss
/// 
/// 若要了解此功能支持的可选参数，请参阅 `torch::nn::functional::CosineEmbeddingLossFuncOptions` 类的文档。
/// 
/// 示例：
/// ```
/// namespace F = torch::nn::functional;
/// F::cosine_embedding_loss(input1, input2, target,
///                         F::CosineEmbeddingLossFuncOptions().margin(0.5));
/// ```
inline Tensor cosine_embedding_loss(
    const Tensor& input1, // 第一个输入张量
    const Tensor& input2, // 第二个输入张量
    const Tensor& target, // 目标张量
    const CosineEmbeddingLossFuncOptions& options = {}) { // 余弦嵌入损失函数的选项
  // 调用 detail 命名空间中的 cosine_embedding_loss 函数
  return detail::cosine_embedding_loss(
      input1, input2, target, options.margin(), options.reduction());
}

// ============================================================================

/// 计算平滑 L1 损失的函数
inline Tensor _smooth_l1_loss(
    const Tensor& input, // 输入张量
    const Tensor& target, // 目标张量
    double beta = 1.) { // 平滑 L1 损失函数的参数 beta
  // 计算输入张量和目标张量的绝对差值
  auto t = torch::abs(input - target);
  // 根据条件计算平滑 L1 损失
  return torch::where(t < beta, 0.5 * torch::pow(t, 2) / beta, t - 0.5 * beta);
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
// 平滑 L1 损失的具体实现
inline Tensor smooth_l1_loss(
    const Tensor& input, // 输入张量
    const Tensor& target, // 目标张量
    SmoothL1LossFuncOptions::reduction_t reduction, // 减少选项
    std::optional<double> beta_opt = c10::nullopt) { // 可选参数 beta
  // 如果目标张量的尺寸与输入张量不同，则发出警告
  if (target.sizes() != input.sizes()) {
    TORCH_WARN(
        "Using a target size (",
        target.sizes(),
        ") that is different to the input size (",
        input.sizes(),
        "). ",
        "This will likely lead to incorrect results due to broadcasting. ",
        "Please ensure they have the same size.");
  }
  // 确定 beta 的值，如果未提供则为默认值 1.0
  double beta = beta_opt.value_or(1.0);

  // 广播输入张量和目标张量，并计算平滑 L1 损失
  std::vector<Tensor> expanded_tensors =
      torch::broadcast_tensors({input, target});
  return torch::smooth_l1_loss(
      expanded_tensors[0],
      expanded_tensors[1],
      enumtype::reduction_get_enum(reduction),
      beta);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// 查看这个函数的具体行为，请参阅
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.smooth_l1_loss
/// 
/// 若要了解此功能支持的可选参数，请参阅 `torch::nn::functional::SmoothL1LossFuncOptions` 类的文档。
/// 
/// 示例：
/// ```
/// namespace F = torch::nn::functional;
/// F::smooth_l1_loss(input, target, F::SmoothL1LossFuncOptions(torch::kNone));
/// ```
// 计算 Smooth L1 损失函数，返回计算结果
inline Tensor smooth_l1_loss(
    const Tensor& input,
    const Tensor& target,
    const SmoothL1LossFuncOptions& options = {}) {
  // 调用详细实现函数进行计算
  return detail::smooth_l1_loss(
      input, target, options.reduction(), options.beta());
}

/// 查看关于此功能的详细行为，请访问
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.smooth_l1_loss
///
/// 示例：
/// ```
/// namespace F = torch::nn::functional;
/// F::smooth_l1_loss(input, target, /*options=*/torch::kNone, /*beta=*/0.5);
/// ```
inline Tensor smooth_l1_loss(
    const Tensor& input,
    const Tensor& target,
    const SmoothL1LossFuncOptions& options,
    double beta) {
  // 检查是否在选项中提供了 beta 参数
  TORCH_CHECK(
      options.beta() == c10::nullopt,
      "expected beta not to be provided in 'options', but got ",
      options.beta().value());
  // 调用详细实现函数进行计算
  return detail::smooth_l1_loss(input, target, options.reduction(), beta);
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
// 计算 Huber 损失函数的详细实现
inline Tensor huber_loss(
    const Tensor& input,
    const Tensor& target,
    HuberLossFuncOptions::reduction_t reduction,
    double delta = 1.) {
  // 如果目标大小与输入大小不匹配，则发出警告
  if (target.sizes() != input.sizes()) {
    TORCH_WARN(
        "Using a target size (",
        target.sizes(),
        ") that is different to the input size (",
        input.sizes(),
        "). ",
        "This will likely lead to incorrect results due to broadcasting. ",
        "Please ensure they have the same size.");
  }

  // 广播输入和目标张量，然后计算 Huber 损失
  std::vector<Tensor> expanded_tensors =
      torch::broadcast_tensors({input, target});
  return torch::huber_loss(
      expanded_tensors[0],
      expanded_tensors[1],
      enumtype::reduction_get_enum(reduction),
      delta);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// 查看关于此功能的详细行为，请访问
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.huber_loss
///
/// 查看 `torch::nn::functional::HuberLossFuncOptions` 类的文档，了解此功能支持的可选参数。
///
/// 示例：
/// ```
/// namespace F = torch::nn::functional;
/// F::huber_loss(input, target,
/// F::HuberLossFuncOptions().reduction(torch::kNone).delta(0.5));
/// ```
inline Tensor huber_loss(
    const Tensor& input,
    const Tensor& target,
    const HuberLossFuncOptions& options = {}) {
  // 调用详细实现函数进行计算
  return detail::huber_loss(
      input, target, options.reduction(), options.delta());
}

// ============================================================================
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.multilabel_margin_loss
/// 了解此函数的详细行为。
///
/// 查看 `torch::nn::functional::MultilabelMarginLossFuncOptions` 类的文档，
/// 了解此函数支持的可选参数。
///
/// 示例:
/// ```
/// namespace F = torch::nn::functional;
/// F::multilabel_margin_loss(input, target,
/// F::MultilabelMarginLossFuncOptions(torch::kNone));
/// ```
inline Tensor multilabel_margin_loss(
    const Tensor& input,
    const Tensor& target,
    const MultilabelMarginLossFuncOptions& options = {}) {
  return detail::multilabel_margin_loss(input, target, options.reduction());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
/// 使用给定的输入和目标计算软间隔损失。
/// 
/// Parameters:
/// - input: 输入张量，通常是模型的输出。
/// - target: 目标张量，通常是真实的标签。
/// - reduction: 损失的减少类型，可以是 'mean'、'sum' 或 'none' 之一。
inline Tensor soft_margin_loss(
    const Tensor& input,
    const Tensor& target,
    SoftMarginLossFuncOptions::reduction_t reduction) {
  return torch::soft_margin_loss(
      input, target, enumtype::reduction_get_enum(reduction));
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// 查看
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.soft_margin_loss
/// 了解此函数的详细行为。
///
/// 查看 `torch::nn::functional::SoftMarginLossFuncOptions` 类的文档，
/// 了解此函数支持的可选参数。
///
/// 示例:
/// ```
/// namespace F = torch::nn::functional;
/// F::soft_margin_loss(input, target,
/// F::SoftMarginLossFuncOptions(torch::kNone));
/// ```
inline Tensor soft_margin_loss(
    const Tensor& input,
    const Tensor& target,
    const SoftMarginLossFuncOptions& options = {}) {
  return detail::soft_margin_loss(input, target, options.reduction());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
/// 计算多标签软间隔损失。
/// 
/// Parameters:
/// - input: 输入张量，通常是模型的输出。
/// - target: 目标张量，通常是多标签二进制标记。
/// - weight: 权重张量，用于加权损失计算（可选）。
/// - reduction: 损失的减少类型，可以是 'mean'、'sum' 或 'none' 之一。
inline Tensor multilabel_soft_margin_loss(
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight,
    MultilabelSoftMarginLossFuncOptions::reduction_t reduction) {
  // 计算多标签软间隔损失
  auto loss =
      -(target * torch::log_sigmoid(input) +
        (1 - target) * torch::log_sigmoid(-input));
  
  // 如果定义了权重，则对损失进行加权
  if (weight.defined()) {
    loss = loss * weight;
  }

  // 计算类别的维度
  auto class_dim = input.dim() - 1;
  auto C = input.size(class_dim);
  // 对损失在类别维度上求和，然后除以类别数目，返回 N 个损失值
  loss = loss.sum(class_dim) / C;

  Tensor ret;

  // 根据减少类型选择返回的张量
  if (std::holds_alternative<enumtype::kNone>(reduction)) {
    ret = loss;
  } else if (std::holds_alternative<enumtype::kMean>(reduction)) {
    ret = loss.mean();
  } else if (std::holds_alternative<enumtype::kSum>(reduction)) {
    ret = loss.sum();
  } else {
    // 如果减少类型不合法，则抛出错误
    ret = input;
    TORCH_INTERNAL_ASSERT(
        false, enumtype::get_enum_name(reduction), " is not valid");
  }
  return ret;
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// 查看
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.multilabel_soft_margin_loss
/// 关于这个函数的详细行为，请参考文档。
///
/// 查看 `torch::nn::functional::MultilabelSoftMarginLossFuncOptions` 类的文档，
/// 了解这个函数支持的可选参数。
///
/// 示例：
/// ```
/// namespace F = torch::nn::functional;
/// F::multilabel_soft_margin_loss(input, target,
///   F::MultilabelSoftMarginLossFuncOptions().reduction(torch::kNone).weight(weight));
/// ```
inline Tensor multilabel_soft_margin_loss(
    const Tensor& input,
    const Tensor& target,
    const MultilabelSoftMarginLossFuncOptions& options = {}) {
  return detail::multilabel_soft_margin_loss(
      input, target, options.weight(), options.reduction());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
/// 实现了 Triplet Margin Loss 的细节函数
inline Tensor triplet_margin_loss(
    const Tensor& anchor,
    const Tensor& positive,
    const Tensor& negative,
    double margin,
    double p,
    double eps,
    bool swap,
    TripletMarginLossFuncOptions::reduction_t reduction) {
  return torch::triplet_margin_loss(
      anchor,
      positive,
      negative,
      margin,
      p,
      eps,
      swap,
      enumtype::reduction_get_enum(reduction));
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// 查看
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.triplet_margin_loss
/// 关于这个函数的详细行为，请参考文档。
///
/// 查看 `torch::nn::functional::TripletMarginLossFuncOptions` 类的文档，
/// 了解这个函数支持的可选参数。
///
/// 示例：
/// ```
/// namespace F = torch::nn::functional;
/// F::triplet_margin_loss(anchor, positive, negative,
///   F::TripletMarginLossFuncOptions().margin(1.0));
/// ```
inline Tensor triplet_margin_loss(
    const Tensor& anchor,
    const Tensor& positive,
    const Tensor& negative,
    const TripletMarginLossFuncOptions& options = {}) {
  return detail::triplet_margin_loss(
      anchor,
      positive,
      negative,
      options.margin(),
      options.p(),
      options.eps(),
      options.swap(),
      options.reduction());
}

// ============================================================================
#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
/// 实现了带距离函数的 Triplet Margin Loss 的细节函数
inline Tensor triplet_margin_with_distance_loss(
    const Tensor& anchor,
    const Tensor& positive,
    const Tensor& negative,
    std::optional<TripletMarginWithDistanceLossFuncOptions::distance_function_t>
        distance_function,
    double margin,
    bool swap,
    TripletMarginWithDistanceLossFuncOptions::reduction_t reduction) {
  Tensor dist_pos, dist_neg;
  if (distance_function.has_value()) {
    auto distance_function_impl = distance_function.value();
    dist_pos = distance_function_impl(anchor, positive);
    dist_neg = distance_function_impl(anchor, negative);
  } else {
    dist_pos = torch::pairwise_distance(anchor, positive);
    dist_neg = torch::pairwise_distance(anchor, negative);
  }
  // 如果使用距离函数实现，计算锚点到负样本的距离
  dist_neg = distance_function_impl(anchor, negative);
} else {
  // 使用成对距离函数计算锚点到正样本和负样本的距离
  dist_pos = pairwise_distance(anchor, positive);
  dist_neg = pairwise_distance(anchor, negative);
}

// 如果需要交换，计算交换后的负样本与正样本的距离
if (swap) {
  Tensor dist_swap;
  // 如果指定了距离函数，则使用该函数计算正样本到负样本的距离
  if (distance_function.has_value()) {
    dist_swap = distance_function.value()(positive, negative);
  } else {
    // 否则使用默认的成对距离函数计算正样本到负样本的距离
    dist_swap = pairwise_distance(positive, negative);
  }
  // 取两种计算方式中较小的距离作为负样本的距离
  dist_neg = torch::min(dist_neg, dist_swap);
}

// 计算损失，采用 hinge 损失函数并进行最小值截断
auto loss = torch::clamp_min(dist_pos - dist_neg + margin, 0);

Tensor ret;
// 根据指定的减少方式计算最终的返回值
if (std::holds_alternative<enumtype::kNone>(reduction)) {
  ret = loss;  // 如果不进行减少，则直接返回损失
} else if (std::holds_alternative<enumtype::kMean>(reduction)) {
  ret = loss.mean();  // 如果求均值减少，则返回损失的均值
} else if (std::holds_alternative<enumtype::kSum>(reduction)) {
  ret = loss.sum();  // 如果求和减少，则返回损失的总和
} else {
  ret = anchor;
  // 如果指定的减少方式不合法，则抛出断言错误
  TORCH_INTERNAL_ASSERT(
      false, enumtype::get_enum_name(reduction), " is not valid");
}
return ret;
#ifndef DOXYGEN_SHOULD_SKIP_THIS
// 在这个命名空间中定义了实现细节的函数，不需要被 Doxygen 文档化
namespace detail {
// 计算使用三元组损失函数的损失值
inline Tensor triplet_margin_with_distance_loss(
    const Tensor& anchor,                      // 锚点张量
    const Tensor& positive,                    // 正样本张量
    const Tensor& negative,                    // 负样本张量
    const TripletMarginWithDistanceLossFuncOptions& options = {}) {  // 可选参数
  return detail::triplet_margin_with_distance_loss(
      anchor,
      positive,
      negative,
      options.distance_function(),             // 距离函数选项
      options.margin(),                        // 损失函数的边界值
      options.swap(),                          // 是否进行交换
      options.reduction());                    // 损失值的缩减方式
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// 查看这个函数的详细行为：
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.triplet_margin_with_distance_loss
///
/// 查看 `torch::nn::functional::TripletMarginWithDistanceLossFuncOptions` 类的文档，
/// 了解这个函数支持的可选参数。
///
/// 示例：
/// ```
/// namespace F = torch::nn::functional;
/// F::triplet_margin_with_distance_loss(anchor, positive, negative,
/// F::TripletMarginWithDistanceLossFuncOptions().margin(1.0));
/// ```
inline Tensor triplet_margin_with_distance_loss(
    const Tensor& anchor,                      // 锚点张量
    const Tensor& positive,                    // 正样本张量
    const Tensor& negative,                    // 负样本张量
    const TripletMarginWithDistanceLossFuncOptions& options = {}) {  // 可选参数
  return detail::triplet_margin_with_distance_loss(
      anchor,
      positive,
      negative,
      options.distance_function(),             // 距离函数选项
      options.margin(),                        // 损失函数的边界值
      options.swap(),                          // 是否进行交换
      options.reduction());                    // 损失值的缩减方式
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
// 在这个命名空间中定义了实现细节的函数，不需要被 Doxygen 文档化
namespace detail {
// 计算 CTC（Connectionist Temporal Classification）损失
inline Tensor ctc_loss(
    const Tensor& log_probs,                   // 对数概率
    const Tensor& targets,                     // 目标张量
    const Tensor& input_lengths,               // 输入长度
    const Tensor& target_lengths,              // 目标长度
    int64_t blank,                             // 空白标签
    CTCLossFuncOptions::reduction_t reduction, // 损失缩减方式
    bool zero_infinity) {                      // 是否将无穷大置零
  return torch::ctc_loss(
      log_probs,
      targets,
      input_lengths,
      target_lengths,
      blank,
      enumtype::reduction_get_enum(reduction),
      zero_infinity);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// 查看这个函数的详细行为：
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.ctc_loss
///
/// 查看 `torch::nn::functional::CTCLossFuncOptions` 类的文档，
/// 了解这个函数支持的可选参数。
///
/// 示例：
/// ```
/// namespace F = torch::nn::functional;
/// F::ctc_loss(log_probs, targets, input_lengths, target_lengths,
/// F::CTCLossFuncOptions().reduction(torch::kNone));
/// ```
inline Tensor ctc_loss(
    const Tensor& log_probs,                   // 对数概率
    const Tensor& targets,                     // 目标张量
    const Tensor& input_lengths,               // 输入长度
    const Tensor& target_lengths,              // 目标长度
    const CTCLossFuncOptions& options = {}) {  // 可选参数
  return detail::ctc_loss(
      log_probs,
      targets,
      input_lengths,
      target_lengths,
      options.blank(),                         // 空白标签选项
      options.reduction(),                     // 损失缩减方式选项
      options.zero_infinity());                // 是否将无穷大置零选项
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
// 在这个命名空间中定义了实现细节的函数，不需要被 Doxygen 文档化
namespace detail {
// 计算 Poisson 损失
inline Tensor poisson_nll_loss(
    const Tensor& input,                       // 输入张量
    const Tensor& target,                      // 目标张量
    bool log_input,                            // 是否对输入取对数
    bool full,                                 // 是否使用全模式
    double eps,                                // 一个小的常量值
    PoissonNLLLossFuncOptions::reduction_t reduction) {  // 损失缩减方式
  return torch::poisson_nll_loss(
      input,
      target,
      log_input,
      full,
      eps,
      enumtype::reduction_get_enum(reduction));
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */
#ifndef DOXYGEN_SHOULD_SKIP_THIS
// 在 detail 命名空间中定义 nll_loss 函数，计算负对数似然损失
namespace detail {
// 计算负对数似然损失
inline Tensor nll_loss(
    // 输入张量，包含预测值
    const Tensor& input,
    // 目标张量，包含真实标签
    const Tensor& target,
    // 权重张量，可选参数
    const Tensor& weight,
    // 需要忽略的标签索引，可选参数
    int64_t ignore_index,
    // 减少损失的方式，可选参数
    const NLLLossFuncOptions::reduction_t& reduction) {
  // 如果输入张量维度小于2，则报错
  if (input.dim() < 2) {
    TORCH_CHECK(false, "Expected 2 or more dimensions (got ", input.dim(), ")");
  }

  // 如果输入张量和目标张量的第一维度大小不一致，则报错
  if (input.sizes()[0] != target.sizes()[0]) {


继续下面的代码注释……
    # 使用 TORCH_CHECK 宏来检查条件，如果条件为 false，则输出错误信息
    TORCH_CHECK(
        false,
        "Expected input batch_size (",
        input.sizes()[0],
        ") to match target batch_size (",
        target.sizes()[0],
        ").");
  }

  # 返回使用 torch::nll_loss_nd 计算的负对数似然损失值
  return torch::nll_loss_nd(
      input,                     # 输入张量，通常为模型输出的预测结果
      target,                    # 目标张量，通常为真实的目标标签
      weight,                    # 权重张量，用于加权损失的计算，可选
      enumtype::reduction_get_enum(reduction),  # 损失的降维方式，如求平均或求和
      ignore_index               # 忽略的类别索引，计算损失时会跳过这些类别
  );
/// ifndef 指令，用于检查是否已定义宏 DOXYGEN_SHOULD_SKIP_THIS，以避免 Doxygen 文档生成工具处理此部分内容
#ifndef DOXYGEN_SHOULD_SKIP_THIS

/// detail 命名空间，包含了内部细节函数的实现
namespace detail {

/// 计算负对数似然损失（NLLLoss）的函数，根据输入、目标和可选选项
inline Tensor nll_loss(
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight,
    int64_t ignore_index,
    enumtype::reduction_t reduction) {
  return torch::nll_loss(
      input,
      target,
      weight,
      ignore_index,
      enumtype::reduction_get_enum(reduction));
}

/// 计算交叉熵损失（Cross Entropy Loss）的函数，根据输入、目标和可选选项
inline Tensor cross_entropy(
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight,
    int64_t ignore_index,
    enumtype::reduction_t reduction,
    double label_smoothing) {
  return torch::cross_entropy_loss(
      input,
      target,
      weight,
      enumtype::reduction_get_enum(reduction),
      ignore_index,
      label_smoothing);
}

/// 计算带 logits 的二元交叉熵损失的函数，根据输入、目标、权重、减少方式和正权重
inline Tensor binary_cross_entropy_with_logits(
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight,
    enumtype::reduction_t reduction,
    const Tensor& pos_weight) {
  /// 检查目标张量与输入张量的尺寸是否相同，确保匹配
  TORCH_CHECK(
      target.sizes() == input.sizes(),
      "Target size (",
      target.sizes(),
      ") must be the same as input size (",
      input.sizes(),
      ")");

  return torch::binary_cross_entropy_with_logits(
      input,
      target,
      weight,
      pos_weight,
      enumtype::reduction_get_enum(reduction));
}

} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// 查看关于 torch::nn::functional::nll_loss 函数的详细行为说明
///
/// 查看 `torch::nn::functional::NLLLossFuncOptions` 类的文档，了解该函数支持的可选参数
///
/// 示例：
/// ```
/// namespace F = torch::nn::functional;
/// F::nll_loss(input, target,
/// F::NLLLossFuncOptions().ignore_index(-100).reduction(torch::kMean));
/// ```
inline Tensor nll_loss(
    const Tensor& input,
    const Tensor& target,
    const NLLLossFuncOptions& options = {}) {
  /// 调用 detail 命名空间中的 nll_loss 函数，传递输入、目标和选项中的参数
  return detail::nll_loss(
      input,
      target,
      options.weight(),
      options.ignore_index(),
      options.reduction());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
/// 计算交叉熵损失的具体实现函数，根据输入、目标、权重、忽略索引、减少方式和标签平滑度
inline Tensor cross_entropy(
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight,
    int64_t ignore_index,
    CrossEntropyFuncOptions::reduction_t reduction,
    double label_smoothing) {
  return torch::cross_entropy_loss(
      input,
      target,
      weight,
      enumtype::reduction_get_enum(reduction),
      ignore_index,
      label_smoothing);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// 查看关于 torch::nn::functional::cross_entropy 函数的详细行为说明
///
/// 查看 `torch::nn::functional::CrossEntropyFuncOptions` 类的文档，了解该函数支持的可选参数
///
/// 示例：
/// ```
/// namespace F = torch::nn::functional;
/// F::cross_entropy(input, target,
/// F::CrossEntropyFuncOptions().ignore_index(-100).reduction(torch::kMean));
/// ```
inline Tensor cross_entropy(
    const Tensor& input,
    const Tensor& target,
    const CrossEntropyFuncOptions& options = {}) {
  /// 调用 detail 命名空间中的 cross_entropy 函数，传递输入、目标和选项中的参数
  return detail::cross_entropy(
      input,
      target,
      options.weight(),
      options.ignore_index(),
      options.reduction(),
      options.label_smoothing());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
/// 计算带 logits 的二元交叉熵损失的具体实现函数，根据输入、目标、权重、减少方式和正权重
inline Tensor binary_cross_entropy_with_logits(
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight,
    BinaryCrossEntropyWithLogitsFuncOptions::reduction_t reduction,
    const Tensor& pos_weight) {
  TORCH_CHECK(
      target.sizes() == input.sizes(),
      "Target size (",
      target.sizes(),
      ") must be the same as input size (",
      input.sizes(),
      ")");

  return torch::binary_cross_entropy_with_logits(
      input,
      target,
      weight,
      pos_weight,
      enumtype::reduction_get_enum(reduction));
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// 查看关于 torch::nn::functional::binary_cross_entropy_with_logits 函数的详细行为说明
///
/// 示例：未完，被裁剪
/// 定义了一个内联函数，计算带有 logits 的二元交叉熵损失函数。
/// 该函数可以根据提供的选项进行配置，例如权重、缩减方式等。
///
/// 参数说明：
/// input: 输入张量，通常是模型的输出 logits。
/// target: 目标张量，通常是真实的标签。
/// options: 一个 BinaryCrossEntropyWithLogitsFuncOptions 对象，用于配置函数的可选参数。
///
/// 返回值：
/// 返回计算得到的二元交叉熵损失的张量。
///
/// 示例用法：
/// ```
/// namespace F = torch::nn::functional;
/// F::binary_cross_entropy_with_logits(input, target,
///     F::BinaryCrossEntropyWithLogitsFuncOptions()
///         .pos_weight(pos_weight)
///         .reduction(torch::kSum));
/// ```
inline Tensor binary_cross_entropy_with_logits(
    const Tensor& input,
    const Tensor& target,
    const BinaryCrossEntropyWithLogitsFuncOptions& options = {}) {
  // 调用详细实现函数，传递输入、目标以及选项中的权重、缩减方式和正样本权重
  return detail::binary_cross_entropy_with_logits(
      input,
      target,
      options.weight(),
      options.reduction(),
      options.pos_weight());
}

} // namespace functional
} // namespace nn
} // namespace torch
```