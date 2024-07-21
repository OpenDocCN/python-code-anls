# `.\pytorch\torch\csrc\api\include\torch\nn\options\loss.h`

```
#pragma once

#include <torch/arg.h>
#include <torch/csrc/Export.h>
#include <torch/enum.h>
#include <torch/types.h>

namespace torch {
namespace nn {

/// Options for the `L1Loss` module.
///
/// Example:
/// ```
/// L1Loss model(L1LossOptions(torch::kNone));
/// ```
struct TORCH_API L1LossOptions {
  typedef std::variant<enumtype::kNone, enumtype::kMean, enumtype::kSum>
      reduction_t;

  TORCH_OPTIONS_CTOR_VARIANT_ARG3(L1LossOptions, reduction, kNone, kMean, kSum)

  /// Specifies the reduction to apply to the output.
  TORCH_ARG(reduction_t, reduction) = torch::kMean;
};

namespace functional {
/// Options for `torch::nn::functional::l1_loss`.
///
/// See the documentation for `torch::nn::L1LossOptions` class to learn what
/// arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::l1_loss(input, target, F::L1LossFuncOptions(torch::kNone));
/// ```
using L1LossFuncOptions = L1LossOptions;
} // namespace functional

// ============================================================================

/// Options for the `KLDivLoss` module.
///
/// Example:
/// ```
/// KLDivLoss
/// model(KLDivLossOptions().reduction(torch::kNone).log_target(false));
/// ```
struct TORCH_API KLDivLossOptions {
  typedef std::variant<
      enumtype::kNone,
      enumtype::kBatchMean,
      enumtype::kSum,
      enumtype::kMean>
      reduction_t;

  TORCH_OPTIONS_CTOR_VARIANT_ARG4(
      KLDivLossOptions,
      reduction,
      kNone,
      kBatchMean,
      kSum,
      kMean)

  /// Specifies the reduction to apply to the output.
  /// ``'none'`` | ``'batchmean'`` | ``'sum'`` | ``'mean'``. Default: ``'mean'``
  TORCH_ARG(reduction_t, reduction) = torch::kMean;

  /// Specifies whether `target` is accepted in the log space. Default: False
  TORCH_ARG(bool, log_target) = false;
};

namespace functional {
/// Options for `torch::nn::functional::kl_div`.
///
/// See the documentation for `torch::nn::KLDivLossOptions` class to learn what
/// arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::kl_div(input, target,
/// F::KLDivFuncOptions().reduction(torch::kNone).log_target(false));
/// ```
using KLDivFuncOptions = KLDivLossOptions;
} // namespace functional

// ============================================================================

/// Options for the `MSELoss` module.
///
/// Example:
/// ```
/// MSELoss model(MSELossOptions(torch::kNone));
/// ```
struct TORCH_API MSELossOptions {
  typedef std::variant<enumtype::kNone, enumtype::kMean, enumtype::kSum>
      reduction_t;

  TORCH_OPTIONS_CTOR_VARIANT_ARG3(MSELossOptions, reduction, kNone, kMean, kSum)

  /// Specifies the reduction to apply to the output.
  /// ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'mean'``
  TORCH_ARG(reduction_t, reduction) = torch::kMean;
};

namespace functional {
/// Options for `torch::nn::functional::mse_loss`.
///
/// See the documentation for `torch::nn::MSELossOptions` class to learn what
/// arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::mse_loss(input, target, F::MSELossFuncOptions(torch::kNone));
/// ```
using MSELossFuncOptions = MSELossOptions;
} // namespace functional
/// Options for the `MSELoss` module.
///
/// Example:
/// ```
/// MSELoss model(MSELossOptions().reduction(torch::kNone).weight(weight));
/// ```
struct TORCH_API MSELossOptions {
  typedef std::variant<enumtype::kNone, enumtype::kMean, enumtype::kSum>
      reduction_t;

  /// A manual rescaling weight given to the loss of each batch element.
  TORCH_ARG(Tensor, weight) = {};
  /// Specifies the reduction to apply to the output.
  /// ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'mean'``
  TORCH_ARG(reduction_t, reduction) = torch::kMean;
};

namespace functional {
/// Options for `torch::nn::functional::mse_loss`.
///
/// See the documentation for `torch::nn::MSELossOptions` class to learn what
/// arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::mse_loss(input, target, F::MSELossFuncOptions(torch::kNone));
/// ```
using MSELossFuncOptions = MSELossOptions;
} // namespace functional
// 定义了 `MultiMarginLossOptions` 结构体，用于配置 `MultiMarginLoss` 损失函数的选项
struct TORCH_API MultiMarginLossOptions {
  // 定义了一个变量 `reduction_t`，是一个枚举类型的变体，可以是 `kNone`、`kMean` 或 `kSum`
  typedef std::variant<enumtype::kNone, enumtype::kMean, enumtype::kSum> reduction_t;

  /// Has a default value of :math:`1`. :math:`1` and :math:`2`
  /// are the only supported values.
  // 损失函数参数 `p` 的默认值是 `1`，支持的取值是 `1` 和 `2`
  TORCH_ARG(int64_t, p) = 1;
  /// Has a default value of :math:`1`.
  // 损失函数参数 `margin` 的默认值是 `1.0`
  TORCH_ARG(double, margin) = 1.0;
  /// A manual rescaling weight given to each
  /// class. If given, it has to be a Tensor of size `C`. Otherwise, it is
  /// treated as if having all ones.
  // 为每个类别提供的手动缩放权重 `weight`，如果提供，应为大小为 `C` 的张量；否则，默认为全部为 `1`。
  TORCH_ARG(Tensor, weight) = Tensor();
  /// Specifies the reduction to apply to the output:
  /// ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be
  /// applied,
  /// ``'mean'``: the sum of the output will be divided by the number of
  /// elements in the output, ``'sum'``: the output will be summed. Default:
  /// ``'mean'``
  // 指定应用于输出的减少方式：`'none'` 没有减少，`'mean'` 输出之和除以输出元素数量，`'sum'` 输出求和。默认是 `'mean'`。
  TORCH_ARG(reduction_t, reduction) = torch::kMean;
};

namespace functional {
/// Options for `torch::nn::functional::multi_margin_loss`.
///
/// See the documentation for `torch::nn::MultiMarginLossOptions` class to learn
/// what arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::multi_margin_loss(input, target,
/// F::MultiMarginLossFuncOptions().margin(2).weight(weight));
/// ```
// `torch::nn::functional` 命名空间中的 `MultiMarginLossFuncOptions` 结构体，用于配置 `multi_margin_loss` 函数的选项
using MultiMarginLossFuncOptions = MultiMarginLossOptions;
} // namespace functional

// ============================================================================

/// Options for the `CosineEmbeddingLoss` module.
///
/// Example:
/// ```
/// CosineEmbeddingLoss model(CosineEmbeddingLossOptions().margin(0.5));
/// ```
// `CosineEmbeddingLoss` 模块的选项结构体 `CosineEmbeddingLossOptions`
struct TORCH_API CosineEmbeddingLossOptions {
  // 定义了一个变量 `reduction_t`，是一个枚举类型的变体，可以是 `kNone`、`kMean` 或 `kSum`
  typedef std::variant<enumtype::kNone, enumtype::kMean, enumtype::kSum> reduction_t;

  /// Specifies the threshold for which the distance of a negative sample must
  /// reach in order to incur zero loss. Should be a number from -1 to 1, 0
  /// to 0.5 is suggested. Default: 0.0
  // 指定负样本距离达到零损失的阈值 `margin`。建议范围是 `-1` 到 `1`，推荐 `0` 到 `0.5`。默认值是 `0.0`
  TORCH_ARG(double, margin) = 0.0;
  /// Specifies the reduction to apply to the output. Default: Mean
  // 指定应用于输出的减少方式。默认是 `Mean`
  TORCH_ARG(reduction_t, reduction) = torch::kMean;
};

namespace functional {
/// Options for `torch::nn::functional::cosine_embedding_loss`.
///
/// See the documentation for `torch::nn::CosineEmbeddingLossOptions` class to
/// learn what arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::cosine_embedding_loss(input1, input2, target,
/// F::CosineEmbeddingLossFuncOptions().margin(0.5));
/// ```
// `torch::nn::functional` 命名空间中的 `CosineEmbeddingLossFuncOptions` 结构体，用于配置 `cosine_embedding_loss` 函数的选项
using CosineEmbeddingLossFuncOptions = CosineEmbeddingLossOptions;
} // namespace functional

// ============================================================================

/// Options for the `MultiLabelMarginLoss` module.
///
/// Example:
/// ```
/// MultiLabelMarginLoss model(MultiLabelMarginLossOptions(torch::kNone));
/// ```
// `MultiLabelMarginLoss` 模块的选项结构体 `MultiLabelMarginLossOptions`
// 定义了一个结构体 MultiLabelMarginLossOptions，用于配置 MultiLabelMarginLoss 损失函数的选项
struct TORCH_API MultiLabelMarginLossOptions {
  // 定义了一个枚举类型 reduction_t，可以是 kNone、kMean、kSum 中的一种
  typedef std::variant<enumtype::kNone, enumtype::kMean, enumtype::kSum> reduction_t;

  // 定义了 MultiLabelMarginLossOptions 的构造函数，接受 reduction 参数，并支持 kNone、kMean、kSum 三个选项
  TORCH_OPTIONS_CTOR_VARIANT_ARG3(
      MultiLabelMarginLossOptions,
      reduction,
      kNone,
      kMean,
      kSum)

  /// 指定要应用于输出的缩减方式: 'none' | 'mean' | 'sum'.
  /// 'none': 不应用缩减，'mean': 输出的总和将除以输出中的元素数，'sum': 对输出求和。默认值: 'mean'
  TORCH_ARG(reduction_t, reduction) = torch::kMean;
};

namespace functional {
/// `torch::nn::functional::multilabel_margin_loss` 的选项。
///
/// 查看 `torch::nn::MultiLabelMarginLossOptions` 类的文档以了解支持的参数。
///
/// 示例:
/// ```
/// namespace F = torch::nn::functional;
/// F::multilabel_margin_loss(input, target,
/// F::MultilabelMarginLossFuncOptions(torch::kNone));
/// ```
using MultilabelMarginLossFuncOptions = MultiLabelMarginLossOptions;
} // namespace functional

// ============================================================================

/// `SoftMarginLoss` 模块的选项。
///
/// 示例:
/// ```
/// SoftMarginLoss model(SoftMarginLossOptions(torch::kNone));
/// ```
struct TORCH_API SoftMarginLossOptions {
  // 定义了一个枚举类型 reduction_t，可以是 kNone、kMean、kSum 中的一种
  typedef std::variant<enumtype::kNone, enumtype::kMean, enumtype::kSum> reduction_t;

  // 定义了 SoftMarginLossOptions 的构造函数，接受 reduction 参数，并支持 kNone、kMean、kSum 三个选项
  TORCH_OPTIONS_CTOR_VARIANT_ARG3(
      SoftMarginLossOptions,
      reduction,
      kNone,
      kMean,
      kSum)

  /// 指定要应用于输出的缩减方式: 'none' | 'mean' | 'sum'.
  /// 'none': 不应用缩减，'mean': 输出的总和将除以输出中的元素数，'sum': 对输出求和。默认值: 'mean'
  TORCH_ARG(reduction_t, reduction) = torch::kMean;
};

namespace functional {
/// `torch::nn::functional::soft_margin_loss` 的选项。
///
/// 查看 `torch::nn::SoftMarginLossOptions` 类的文档以了解支持的参数。
///
/// 示例:
/// ```
/// namespace F = torch::nn::functional;
/// F::soft_margin_loss(input, target,
/// F::SoftMarginLossFuncOptions(torch::kNone));
/// ```
using SoftMarginLossFuncOptions = SoftMarginLossOptions;
} // namespace functional

// ============================================================================

/// `MultiLabelSoftMarginLoss` 模块的选项。
///
/// 示例:
/// ```
/// MultiLabelSoftMarginLoss
/// model(MultiLabelSoftMarginLossOptions().reduction(torch::kNone).weight(weight));
/// ```
/// Options for the `MultiLabelSoftMarginLoss` module.
///
/// Defines the configuration options available for the `torch::nn::MultiLabelSoftMarginLoss` module.
struct TORCH_API MultiLabelSoftMarginLossOptions {
  // 定义 reduction_t 类型为 std::variant，可以是 enumtype::kNone、enumtype::kMean、enumtype::kSum 之一
  typedef std::variant<enumtype::kNone, enumtype::kMean, enumtype::kSum> reduction_t;

  /// A manual rescaling weight given to each
  /// class. If given, it has to be a Tensor of size `C`. Otherwise, it is
  /// treated as if having all ones.
  TORCH_ARG(Tensor, weight) = Tensor();  // 权重张量，默认为空张量

  /// Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
  /// 'none': no reduction will be applied, 'mean': the sum of the output will
  /// be divided by the number of elements in the output, 'sum': the output will
  /// be summed. Default: 'mean'
  TORCH_ARG(reduction_t, reduction) = torch::kMean;  // 指定输出的减少方式，默认为均值
};

namespace functional {
/// Options for `torch::nn::functional::multilabel_soft_margin_loss`.
///
/// See the documentation for `torch::nn::MultiLabelSoftMarginLossOptions` class
/// to learn what arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::multilabel_soft_margin_loss(input, target,
/// F::MultilabelSoftMarginLossFuncOptions().reduction(torch::kNone).weight(weight));
/// ```
using MultilabelSoftMarginLossFuncOptions = MultiLabelSoftMarginLossOptions;
} // namespace functional

// ============================================================================

/// Options for the `TripletMarginLoss` module.
///
/// Example:
/// ```
/// TripletMarginLoss
/// model(TripletMarginLossOptions().margin(3).p(2).eps(1e-06).swap(false));
/// ```
struct TORCH_API TripletMarginLossOptions {
  // 定义 reduction_t 类型为 std::variant，可以是 enumtype::kNone、enumtype::kMean、enumtype::kSum 之一
  typedef std::variant<enumtype::kNone, enumtype::kMean, enumtype::kSum> reduction_t;

  /// Specifies the threshold for which the distance of a negative sample must
  /// reach in order to incur zero loss. Default: 1
  TORCH_ARG(double, margin) = 1.0;  // 负样本距离达到此阈值才会导致零损失，默认为 1.0

  /// Specifies the norm degree for pairwise distance. Default: 2
  TORCH_ARG(double, p) = 2.0;  // 用于成对距离的范数度量，默认为 2.0

  TORCH_ARG(double, eps) = 1e-6;  // 默认为 1e-6

  /// The distance swap is described in detail in the paper Learning shallow
  /// convolutional feature descriptors with triplet losses by V. Balntas,
  /// E. Riba et al. Default: False
  TORCH_ARG(bool, swap) = false;  // 距离交换，默认为 false

  /// Specifies the reduction to apply to the output. Default: Mean
  TORCH_ARG(reduction_t, reduction) = torch::kMean;  // 指定输出的减少方式，默认为均值
};

namespace functional {
/// Options for `torch::nn::functional::triplet_margin_loss`.
///
/// See the documentation for `torch::nn::TripletMarginLossOptions` class to
/// learn what arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::triplet_margin_loss(anchor, positive, negative,
/// F::TripletMarginLossFuncOptions().margin(1.0));
/// ```
using TripletMarginLossFuncOptions = TripletMarginLossOptions;
} // namespace functional

// ============================================================================

/// Options for the `TripletMarginWithDistanceLoss` module.
///
/// Example:
/// ```
/// TripletMarginWithDistanceLoss
/// model(TripletMarginWithDistanceLossOptions().margin(3).swap(false));
/// ```
/// ```
/// 结构体 `TripletMarginWithDistanceLossOptions`，用于定义三元组损失函数的选项。
struct TORCH_API TripletMarginWithDistanceLossOptions {
  /// 用于表示归约（reduction）类型的枚举 `reduction_t`，可以是 kNone、kMean 或 kSum。
  typedef std::variant<enumtype::kNone, enumtype::kMean, enumtype::kSum> reduction_t;
  /// 表示用于计算两个张量之间距离的函数类型 `distance_function_t`。
  typedef std::function<Tensor(const Tensor&, const Tensor&)> distance_function_t;

  /// 指定一个非负的实值函数，用于量化两个张量的相似程度。
  /// 如果未指定，将使用 `F::pairwise_distance`。默认值为 `nullopt`。
  TORCH_ARG(std::optional<distance_function_t>, distance_function) = c10::nullopt;

  /// 指定一个非负的间隔（margin），表示正负样本之间距离的最小差异，使得损失为0。
  /// 较大的间隔会惩罚负样本与锚点的距离不足正样本的情况。默认值为 1。
  TORCH_ARG(double, margin) = 1.0;

  /// 是否使用文献 Learning shallow convolutional feature descriptors with triplet losses
  /// by V. Balntas, E. Riba et al. 中描述的距离交换机制。
  /// 如果为 true，并且正样本与负样本的距离比锚点与负样本的距离更近，则在损失计算中交换正样本和锚点。
  /// 默认值为 false。
  TORCH_ARG(bool, swap) = false;

  /// 指定应用于输出的归约类型。默认值为 Mean。
  TORCH_ARG(reduction_t, reduction) = torch::kMean;
};

/// 命名空间 `functional` 中的别名，用于 `torch::nn::functional::triplet_margin_with_distance_loss` 函数的选项。
///
/// 参见 `torch::nn::TripletMarginWithDistanceLossOptions` 类的文档以了解支持的参数。
///
/// 示例：
/// ```
/// namespace F = torch::nn::functional;
/// F::triplet_margin_with_distance_loss(anchor, positive, negative,
///   F::TripletMarginWithDistanceLossFuncOptions().margin(1.0));
/// ```
using TripletMarginWithDistanceLossFuncOptions = TripletMarginWithDistanceLossOptions;
} // namespace functional

// ============================================================================

/// `CTCLoss` 模块的选项。
///
/// 示例：
/// ```
/// CTCLoss
/// model(CTCLossOptions().blank(42).zero_infinity(false).reduction(torch::kSum));
/// ```
struct TORCH_API CTCLossOptions {
  /// 表示归约（reduction）类型的枚举 `reduction_t`，可以是 kNone、kMean 或 kSum。
  typedef std::variant<enumtype::kNone, enumtype::kMean, enumtype::kSum> reduction_t;

  /// 空白标签。默认值为 `0`。
  TORCH_ARG(int64_t, blank) = 0;

  /// 指定应用于输出的归约类型。默认值为 Mean。
  TORCH_ARG(reduction_t, reduction) = torch::kMean;

  /// 是否将无限损失及其相关梯度置零。
  /// 默认为 `false`。当输入过短以至于无法对齐目标时，会出现无限损失。
  TORCH_ARG(bool, zero_infinity) = false;
};

namespace functional {
/// `torch::nn::functional::ctc_loss` 函数的选项。
///
/// 参见 `torch::nn::CTCLossOptions` 类的文档以了解支持的参数。
///
/// 示例：
/// ```
/// namespace F = torch::nn::functional;
/// ```
/// Options for the `SmoothL1Loss` module.
///
/// Example:
/// ```
/// SmoothL1Loss model(SmoothL1LossOptions().reduction(torch::kNone).beta(0.5));
/// ```
struct TORCH_API SmoothL1LossOptions {
  typedef std::variant<enumtype::kNone, enumtype::kMean, enumtype::kSum>
      reduction_t;

  // 定义构造函数，支持三种不同的减少方式 'none' | 'mean' | 'sum'
  TORCH_OPTIONS_CTOR_VARIANT_ARG3(
      SmoothL1LossOptions,
      reduction,
      kNone,
      kMean,
      kSum)

  /// Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
  /// 'none': no reduction will be applied, 'mean': the sum of the output will
  /// be divided by the number of elements in the output, 'sum': the output will
  /// be summed. Default: 'mean'
  TORCH_ARG(reduction_t, reduction) = torch::kMean;
  /// Specifies the threshold at which to change between L1 and L2 loss.
  /// If beta is not specified, a value of 1.0 will be used.
  /// Default: nullopt
  TORCH_ARG(std::optional<double>, beta) = c10::nullopt;
};

namespace functional {
/// Options for `torch::nn::functional::smooth_l1_loss`.
///
/// See the documentation for `torch::nn::SmoothL1LossOptions` class to learn
/// what arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::smooth_l1_loss(input, target, F::SmoothL1LossFuncOptions(torch::kNone));
/// ```
using SmoothL1LossFuncOptions = SmoothL1LossOptions;
} // namespace functional

// ============================================================================

/// Options for the `HuberLoss` module.
///
/// Example:
/// ```
/// HuberLoss model(HuberLossOptions().reduction(torch::kNone).delta(0.5));
/// ```
struct TORCH_API HuberLossOptions {
  typedef std::variant<enumtype::kNone, enumtype::kMean, enumtype::kSum>
      reduction_t;

  // 定义构造函数，支持三种不同的减少方式 'none' | 'mean' | 'sum'
  TORCH_OPTIONS_CTOR_VARIANT_ARG3(
      HuberLossOptions,
      reduction,
      kNone,
      kMean,
      kSum)

  /// Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
  /// 'none': no reduction will be applied, 'mean': the sum of the output will
  /// be divided by the number of elements in the output, 'sum': the output will
  /// be summed. Default: 'mean'
  TORCH_ARG(reduction_t, reduction) = torch::kMean;
  /// Specifies the threshold at which to change between L1 and L2 loss.
  /// Default: 1.0
  TORCH_ARG(double, delta) = 1.0;
};

namespace functional {
/// Options for `torch::nn::functional::huber_loss`.
///
/// See the documentation for `torch::nn::HuberLossOptions` class to learn what
/// arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::huber_loss(input, target, F::HuberLossFuncOptions(torch::kNone));
/// ```
using HuberLossFuncOptions = HuberLossOptions;
} // namespace functional
// ============================================================================

/// Options for the `PoissonNLLLoss` module.
///
/// Example:
/// ```
/// PoissonNLLLoss
/// model(PoissonNLLLossOptions().log_input(false).full(true).eps(0.42).reduction(torch::kSum));
/// ```
struct TORCH_API PoissonNLLLossOptions {
  typedef std::variant<enumtype::kNone, enumtype::kMean, enumtype::kSum>
      reduction_t;

  /// Determines whether to compute the loss as `exp(input) - target * input` (true) or `input - target * log(input + eps)` (false).
  TORCH_ARG(bool, log_input) = true;
  /// Indicates whether to include the Stirling approximation term `target * log(target) - target + 0.5 * log(2 * pi * target)` in the loss computation.
  TORCH_ARG(bool, full) = false;
  /// A small value added to `input` to prevent evaluation of `log(0)` when `log_input` is false. Default: 1e-8
  TORCH_ARG(double, eps) = 1e-8;
  /// Specifies the reduction to apply to the output. Default: Mean
  TORCH_ARG(reduction_t, reduction) = torch::kMean;
};

namespace functional {
/// Options for `torch::nn::functional::poisson_nll_loss`.
///
/// See the documentation for `torch::nn::PoissonNLLLossOptions` class to learn
/// what arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::poisson_nll_loss(input, target,
/// F::PoissonNLLLossFuncOptions().reduction(torch::kNone));
/// ```
using PoissonNLLLossFuncOptions = PoissonNLLLossOptions;
} // namespace functional

// ============================================================================

/// Options for the `MarginRankingLoss` module.
///
/// Example:
/// ```
/// MarginRankingLoss
/// model(MarginRankingLossOptions().margin(0.5).reduction(torch::kSum));
/// ```
struct TORCH_API MarginRankingLossOptions {
  typedef std::variant<enumtype::kNone, enumtype::kMean, enumtype::kSum>
      reduction_t;

  /// Default margin value for `MarginRankingLoss`. Default: 0
  TORCH_ARG(double, margin) = 0;
  /// Specifies the reduction to apply to the output. Default: Mean
  TORCH_ARG(reduction_t, reduction) = torch::kMean;
};

namespace functional {
/// Options for `torch::nn::functional::margin_ranking_loss`.
///
/// See the documentation for `torch::nn::MarginRankingLossOptions` class to
/// learn what arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::margin_ranking_loss(input1, input2, target,
/// F::MarginRankingLossFuncOptions().margin(0.5).reduction(torch::kSum));
/// ```
using MarginRankingLossFuncOptions = MarginRankingLossOptions;
} // namespace functional

// ============================================================================

/// Options for the `NLLLoss` module.
///
/// Example:
/// ```
/// NLLLoss model(NLLLossOptions().ignore_index(-100).reduction(torch::kMean));
/// ```
/// Structure defining options for the NLLLoss module.
struct TORCH_API NLLLossOptions {
  /// Typedef for the type of reduction strategy.
  typedef std::variant<enumtype::kNone, enumtype::kMean, enumtype::kSum>
      reduction_t;

  /// Optional weight tensor for manual rescaling of class weights.
  /// Default: empty tensor (treated as all ones).
  TORCH_ARG(Tensor, weight) = {};

  /// Specifies the target value that should be ignored during gradient computation.
  /// Default: -100.
  TORCH_ARG(int64_t, ignore_index) = -100;

  /// Specifies the reduction strategy to apply to the output.
  /// Default: Mean.
  TORCH_ARG(reduction_t, reduction) = torch::kMean;
};

namespace functional {
/// Alias for NLLLossOptions, used for torch::nn::functional::nll_loss.
///
/// See torch::nn::NLLLossOptions documentation for supported arguments.
///
/// Example usage:
/// ```
/// namespace F = torch::nn::functional;
/// F::nll_loss(input, target, F::NLLLossFuncOptions().ignore_index(-100).reduction(torch::kMean));
/// ```
using NLLLossFuncOptions = NLLLossOptions;
} // namespace functional

// ============================================================================

/// Structure defining options for the CrossEntropyLoss module.
struct TORCH_API CrossEntropyLossOptions {
  /// Typedef for the type of reduction strategy.
  typedef std::variant<enumtype::kNone, enumtype::kMean, enumtype::kSum>
      reduction_t;

  /// Optional weight tensor for manual rescaling of class weights.
  /// Default: empty tensor (treated as all ones).
  TORCH_ARG(Tensor, weight) = {};

  /// Specifies the target value that should be ignored during gradient computation.
  /// Default: -100.
  TORCH_ARG(int64_t, ignore_index) = -100;

  /// Specifies the reduction strategy to apply to the output.
  /// Default: Mean.
  TORCH_ARG(reduction_t, reduction) = torch::kMean;

  /// Specifies the amount of smoothing to apply when computing the loss.
  /// Default: 0.0 (no smoothing).
  TORCH_ARG(double, label_smoothing) = 0.0;
};

namespace functional {
/// Alias for CrossEntropyLossOptions, used for torch::nn::functional::cross_entropy.
///
/// See torch::nn::CrossEntropyLossOptions documentation for supported arguments.
///
/// Example usage:
/// ```
/// namespace F = torch::nn::functional;
/// F::cross_entropy(input, target, F::CrossEntropyFuncOptions().ignore_index(-100).reduction(torch::kMean));
/// ```
using CrossEntropyFuncOptions = CrossEntropyLossOptions;
} // namespace functional

// ============================================================================

/// Structure defining options for the BCEWithLogitsLoss module.
///
/// Example:
/// ```
/// BCEWithLogitsLoss
/// model(BCEWithLogitsLossOptions().reduction(torch::kNone).weight(weight));
/// ```
/// 结构体 `BCEWithLogitsLossOptions` 的定义，用于存储二元交叉熵损失函数的选项。
struct TORCH_API BCEWithLogitsLossOptions {
  /// 定义了一个类型别名 `reduction_t`，可以是 `enumtype::kNone`、`enumtype::kMean` 或 `enumtype::kSum` 中的一个变量。
  typedef std::variant<enumtype::kNone, enumtype::kMean, enumtype::kSum> reduction_t;
  /// 手动设置的每个批次元素损失的重新缩放权重。
  /// 如果提供，必须是大小为 `nbatch` 的 Tensor。
  TORCH_ARG(Tensor, weight) = {};
  /// 指定要应用于输出的减少方式。默认为平均 (`torch::kMean`)。
  TORCH_ARG(reduction_t, reduction) = torch::kMean;
  /// 正样本的权重。
  /// 必须是长度等于类数的向量。
  TORCH_ARG(Tensor, pos_weight) = {};
};

/// 在 `torch::nn::functional` 命名空间下，提供给 `torch::nn::functional::binary_cross_entropy_with_logits` 使用的选项。
///
/// 查看 `torch::nn::BCEWithLogitsLossOptions` 类的文档，了解支持的参数。
///
/// 示例:
/// ```
/// namespace F = torch::nn::functional;
/// F::binary_cross_entropy_with_logits(input, target,
/// F::BinaryCrossEntropyWithLogitsFuncOptions().pos_weight(pos_weight).reduction(torch::kSum));
/// ```
namespace functional {
using BinaryCrossEntropyWithLogitsFuncOptions = BCEWithLogitsLossOptions;
} // namespace functional

} // namespace nn
} // namespace torch
```