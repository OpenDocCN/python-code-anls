# `.\pytorch\torch\csrc\api\include\torch\nn\modules\loss.h`

```py
#pragma once
// 预处理指令，确保头文件只被包含一次

#include <torch/expanding_array.h>
#include <torch/nn/cloneable.h>
#include <torch/nn/functional/loss.h>
#include <torch/nn/options/loss.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>

#include <torch/csrc/Export.h>

#include <cstddef>
#include <vector>

namespace torch {
namespace nn {

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ L1Loss ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Creates a criterion that measures the mean absolute error (MAE) between each
/// element in the input : math :`x` and target : `y`.
/// See https://pytorch.org/docs/main/nn.html#torch.nn.L1Loss to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::L1LossOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// L1Loss model(L1LossOptions(torch::kNone));
/// ```py
struct TORCH_API L1LossImpl : Cloneable<L1LossImpl> {
  explicit L1LossImpl(L1LossOptions options_ = {});

  void reset() override;

  /// Pretty prints the `L1Loss` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// Computes the forward pass of the `L1Loss` module.
  /// Calculates the mean absolute error (MAE) between `input` and `target`.
  Tensor forward(const Tensor& input, const Tensor& target);

  /// The options with which this `Module` was constructed.
  L1LossOptions options;
};

/// A `ModuleHolder` subclass for `L1LossImpl`.
/// See the documentation for `L1LossImpl` class to learn what methods it
/// provides, and examples of how to use `L1Loss` with
/// `torch::nn::L1LossOptions`. See the documentation for `ModuleHolder` to
/// learn about PyTorch's module storage semantics.
TORCH_MODULE(L1Loss);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ KLDivLoss ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// The Kullback-Leibler divergence loss measure
/// See https://pytorch.org/docs/main/nn.html#torch.nn.KLDivLoss to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::KLDivLossOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// KLDivLoss model(KLDivLossOptions().reduction(torch::kNone));
/// ```py
struct TORCH_API KLDivLossImpl : Cloneable<KLDivLossImpl> {
  explicit KLDivLossImpl(KLDivLossOptions options_ = {});

  void reset() override;

  /// Pretty prints the `KLDivLoss` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// Computes the forward pass of the `KLDivLoss` module.
  /// Calculates the Kullback-Leibler divergence between `input` and `target`.
  Tensor forward(const Tensor& input, const Tensor& target);

  /// The options with which this `Module` was constructed.
  KLDivLossOptions options;
};

/// A `ModuleHolder` subclass for `KLDivLossImpl`.
/// See the documentation for `KLDivLossImpl` class to learn what methods it
/// provides, and examples of how to use `KLDivLoss` with
/// `torch::nn::KLDivLossOptions`. See the documentation for `ModuleHolder` to
/// learn about PyTorch's module storage semantics.
TORCH_MODULE(KLDivLoss);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MSELoss ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/// Creates a criterion that measures the mean squared error (squared L2 norm)
/// between each element in the input `x` and target `y`.
/// See https://pytorch.org/docs/main/nn.html#torch.nn.MSELoss to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::MSELossOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// MSELoss model(MSELossOptions(torch::kNone));
/// ```py
struct TORCH_API MSELossImpl : Cloneable<MSELossImpl> {
  explicit MSELossImpl(MSELossOptions options_ = {});

  void reset() override;

  /// Pretty prints the `MSELoss` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// Computes the forward pass of the `MSELoss` module.
  /// Calculates mean squared error between `input` and `target`.
  Tensor forward(const Tensor& input, const Tensor& target);

  /// The options with which this `Module` was constructed.
  MSELossOptions options;
};

/// A `ModuleHolder` subclass for `MSELossImpl`.
/// See the documentation for `MSELossImpl` class to learn what methods it
/// provides, and examples of how to use `MSELoss` with
/// `torch::nn::MSELossOptions`. See the documentation for `ModuleHolder` to
/// learn about PyTorch's module storage semantics.
TORCH_MODULE(MSELoss);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ BCELoss ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Creates a criterion that measures the Binary Cross Entropy
/// between the target and the output.
/// See https://pytorch.org/docs/main/nn.html#torch.nn.BCELoss to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::BCELossOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// BCELoss model(BCELossOptions().reduction(torch::kNone).weight(weight));
/// ```py
struct TORCH_API BCELossImpl : Cloneable<BCELossImpl> {
  explicit BCELossImpl(BCELossOptions options_ = {});

  void reset() override;

  /// Pretty prints the `BCELoss` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// Computes the forward pass of the `BCELoss` module.
  /// Calculates binary cross entropy loss between `input` and `target`.
  Tensor forward(const Tensor& input, const Tensor& target);

  /// The options with which this `Module` was constructed.
  BCELossOptions options;
};

/// A `ModuleHolder` subclass for `BCELossImpl`.
/// See the documentation for `BCELossImpl` class to learn what methods it
/// provides, and examples of how to use `BCELoss` with
/// `torch::nn::BCELossOptions`. See the documentation for `ModuleHolder` to
/// learn about PyTorch's module storage semantics.
TORCH_MODULE(BCELoss);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ HingeEmbeddingLoss
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Creates a criterion that measures the loss given an input tensor `x`
/// and a labels tensor `y` (containing 1 or -1).
/// See https://pytorch.org/docs/main/nn.html#torch.nn.HingeEmbeddingLoss to
/// learn about the exact behavior of this module.
///
/// See the documentation for `torch::nn::HingeEmbeddingLossOptions` class to
/// learn what constructor arguments are supported for this module.
/// learn what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// HingeEmbeddingLoss
/// model(HingeEmbeddingLossOptions().margin(4).reduction(torch::kNone));
/// ```py
struct TORCH_API HingeEmbeddingLossImpl : Cloneable<HingeEmbeddingLossImpl> {
  explicit HingeEmbeddingLossImpl(HingeEmbeddingLossOptions options_ = {});

  // 重置模块状态的方法
  void reset() override;

  /// Pretty prints the `HingeEmbeddingLoss` module into the given `stream`.
  // 将 `HingeEmbeddingLoss` 模块以美观的方式打印到指定流中的方法
  void pretty_print(std::ostream& stream) const override;

  // 计算正向传播的方法，接受输入张量 `input` 和目标张量 `target`
  Tensor forward(const Tensor& input, const Tensor& target);

  /// The options with which this `Module` was constructed.
  // 使用此 `Module` 构造的选项
  HingeEmbeddingLossOptions options;
};

/// A `ModuleHolder` subclass for `HingeEmbeddingLossImpl`.
/// See the documentation for `HingeEmbeddingLossImpl` class to learn what
/// methods it provides, and examples of how to use `HingeEmbeddingLoss` with
/// `torch::nn::HingeEmbeddingLossOptions`. See the documentation for
/// `ModuleHolder` to learn about PyTorch's module storage semantics.
TORCH_MODULE(HingeEmbeddingLoss);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MultiMarginLoss
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Creates a criterion that optimizes a multi-class classification hinge
/// loss (margin-based loss) between input :math:`x` (a 2D mini-batch `Tensor`)
/// and output :math:`y` (which is a 1D tensor of target class indices, :math:`0
/// \leq y \leq \text{x.size}(1)-1`). See
/// https://pytorch.org/docs/main/nn.html#torch.nn.MultiMarginLoss to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::MultiMarginLossOptions` class to learn
/// what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// MultiMarginLoss model(MultiMarginLossOptions().margin(2).weight(weight));
/// ```py
struct TORCH_API MultiMarginLossImpl : public Cloneable<MultiMarginLossImpl> {
  explicit MultiMarginLossImpl(MultiMarginLossOptions options_ = {});

  // 重置模块状态的方法
  void reset() override;

  /// Pretty prints the `MultiMarginLoss` module into the given `stream`.
  // 将 `MultiMarginLoss` 模块以美观的方式打印到指定流中的方法
  void pretty_print(std::ostream& stream) const override;

  // 计算正向传播的方法，接受输入张量 `input` 和目标张量 `target`
  Tensor forward(const Tensor& input, const Tensor& target);

  /// The options with which this `Module` was constructed.
  // 使用此 `Module` 构造的选项
  MultiMarginLossOptions options;
};

/// A `ModuleHolder` subclass for `MultiMarginLossImpl`.
/// See the documentation for `MultiMarginLossImpl` class to learn what methods
/// it provides, and examples of how to use `MultiMarginLoss` with
/// `torch::nn::MultiMarginLossOptions`. See the documentation for
/// `ModuleHolder` to learn about PyTorch's module storage semantics.
TORCH_MODULE(MultiMarginLoss);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CosineEmbeddingLoss
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Creates a criterion that measures the loss given input tensors
/// `input1`, `input2`, and a `Tensor` label `target` with values 1 or
/// -1. This is used for measuring whether two inputs are similar or
/// dissimilar, using the cosine distance, and is typically used for learning
/// nonlinear embeddings or semi-supervised learning.
/// See https://pytorch.org/docs/main/nn.html#torch.nn.CosineEmbeddingLoss to
/// learn about the exact behavior of this module.
///
/// See the documentation for `torch::nn::CosineEmbeddingLossOptions` class to
/// learn what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// CosineEmbeddingLoss model(CosineEmbeddingLossOptions().margin(0.5));
/// ```py
struct TORCH_API CosineEmbeddingLossImpl
    : public Cloneable<CosineEmbeddingLossImpl> {
  explicit CosineEmbeddingLossImpl(CosineEmbeddingLossOptions options_ = {});

  /// Resets the state of the `CosineEmbeddingLoss` module.
  void reset() override;

  /// Pretty prints the `CosineEmbeddingLoss` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// Computes the forward pass of the `CosineEmbeddingLoss` module.
  /// Calculates the loss based on the cosine distance between `input1` and `input2`,
  /// given the `target`.
  Tensor forward(
      const Tensor& input1,
      const Tensor& input2,
      const Tensor& target);

  /// The options with which this `Module` was constructed.
  CosineEmbeddingLossOptions options;
};

/// A `ModuleHolder` subclass for `CosineEmbeddingLossImpl`.
/// See the documentation for `CosineEmbeddingLossImpl` class to learn what
/// methods it provides, and examples of how to use `CosineEmbeddingLoss` with
/// `torch::nn::CosineEmbeddingLossOptions`. See the documentation for
/// `ModuleHolder` to learn about PyTorch's module storage semantics.
TORCH_MODULE(CosineEmbeddingLoss);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SmoothL1Loss
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Creates a criterion that uses a squared term if the absolute
/// element-wise error falls below beta and an L1 term otherwise.
/// It is less sensitive to outliers than the `MSELoss` and in some cases
/// prevents exploding gradients (e.g. see the paper `Fast R-CNN` by Ross
/// Girshick). See https://pytorch.org/docs/main/nn.html#torch.nn.SmoothL1Loss
/// to learn about the exact behavior of this module.
///
/// See the documentation for `torch::nn::SmoothL1LossOptions` class to learn
/// what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// SmoothL1Loss model(SmoothL1LossOptions().reduction(torch::kNone).beta(0.5));
/// ```py
struct TORCH_API SmoothL1LossImpl : public Cloneable<SmoothL1LossImpl> {
  explicit SmoothL1LossImpl(SmoothL1LossOptions options = {});

  /// Resets the state of the `SmoothL1Loss` module.
  void reset() override;

  /// Pretty prints the `SmoothL1Loss` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// Computes the forward pass of the `SmoothL1Loss` module.
  /// Calculates the loss based on the smooth L1 distance between `input` and `target`.
  Tensor forward(const Tensor& input, const Tensor& target);

  /// The options with which this `Module` was constructed.
  SmoothL1LossOptions options;
};

/// A `ModuleHolder` subclass for `SmoothL1LossImpl`.
/// See the documentation for `SmoothL1LossImpl` class to learn what methods it
/// provides, and examples of how to use `SmoothL1Loss` with
/// `torch::nn::SmoothL1LossOptions`. See the documentation for `ModuleHolder`
/// to learn about PyTorch's module storage semantics.
// 定义一个名为 SmoothL1Loss 的 Torch 模块

TORCH_MODULE(SmoothL1Loss);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ HuberLoss
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// 创建一个标准，如果绝对误差在 delta 以下，则使用平方项，否则使用 delta 缩放的 L1 项。
/// 详细了解此模块的行为，请参阅 https://pytorch.org/docs/main/nn.html#torch.nn.HuberLoss。
///
/// 查看 `torch::nn::HuberLossOptions` 类的文档，了解此模块支持哪些构造参数。
///
/// 示例:
/// ```
/// HuberLoss model(HuberLossOptions().reduction(torch::kNone).delta(0.5));
/// ```py
struct TORCH_API HuberLossImpl : public Cloneable<HuberLossImpl> {
  explicit HuberLossImpl(HuberLossOptions options_ = {});

  void reset() override;

  /// 将 `HuberLoss` 模块漂亮地打印到给定的 `stream` 中。
  void pretty_print(std::ostream& stream) const override;

  /// 对输入 `input` 和目标 `target` 执行前向传播。
  Tensor forward(const Tensor& input, const Tensor& target);

  /// 构造此 `Module` 的选项。
  HuberLossOptions options;
};

/// `HuberLossImpl` 的 `ModuleHolder` 子类。
/// 查看 `HuberLossImpl` 类的文档，了解它提供了哪些方法，以及如何使用 `torch::nn::HuberLossOptions` 使用 `HuberLoss`。
/// 查看 `ModuleHolder` 的文档，了解 PyTorch 的模块存储语义。
TORCH_MODULE(HuberLoss);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MultiLabelMarginLoss
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// 创建一个标准，优化输入 `x`（一个二维小批量 `Tensor`）和输出 `y`（目标类别索引的二维 `Tensor`）之间的多类别多标签边界损失（基于边界的损失）。
/// 详细了解此模块的行为，请参阅 https://pytorch.org/docs/main/nn.html#torch.nn.MultiLabelMarginLoss。
///
/// 查看 `torch::nn::MultiLabelMarginLossOptions` 类的文档，了解此模块支持哪些构造参数。
///
/// 示例:
/// ```
/// MultiLabelMarginLoss model(MultiLabelMarginLossOptions(torch::kNone));
/// ```py
struct TORCH_API MultiLabelMarginLossImpl
    : public Cloneable<MultiLabelMarginLossImpl> {
  explicit MultiLabelMarginLossImpl(MultiLabelMarginLossOptions options_ = {});

  void reset() override;

  /// 将 `L1Loss` 模块漂亮地打印到给定的 `stream` 中。
  void pretty_print(std::ostream& stream) const override;

  /// 对输入 `input` 和目标 `target` 执行前向传播。
  Tensor forward(const Tensor& input, const Tensor& target);

  /// 构造此 `Module` 的选项。
  MultiLabelMarginLossOptions options;
};

/// `MultiLabelMarginLossImpl` 的 `ModuleHolder` 子类。
/// 查看 `MultiLabelMarginLossImpl` 类的文档，了解它提供了哪些方法，以及如何使用 `torch::nn::MultiLabelMarginLossOptions` 使用 `MultiLabelMarginLoss`。
/// 查看 `ModuleHolder` 的文档，了解 PyTorch 的模块存储语义。
// 定义一个 TORCH_MODULE 宏，用于创建 MultiLabelMarginLoss 类型的 Torch 模块
TORCH_MODULE(MultiLabelMarginLoss);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SoftMarginLoss
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// 创建一个 criterion，用于优化输入张量 x 和目标张量 y 之间的二分类 logistic 损失
/// y 包含值为 1 或 -1。详见 https://pytorch.org/docs/main/nn.html#torch.nn.SoftMarginLoss
///
/// 查看 `torch::nn::SoftMarginLossOptions` 类的文档，了解该模块支持的构造参数。
///
/// 示例：
/// ```
/// SoftMarginLoss model(SoftMarginLossOptions(torch::kNone));
/// ```py
struct TORCH_API SoftMarginLossImpl : public Cloneable<SoftMarginLossImpl> {
  explicit SoftMarginLossImpl(SoftMarginLossOptions options_ = {});

  /// 将 `SoftMarginLoss` 模块美化打印到给定的 `stream` 中。
  void pretty_print(std::ostream& stream) const override;

  void reset() override;

  /// 前向传播函数，计算输入张量和目标张量之间的损失
  Tensor forward(const Tensor& input, const Tensor& target);

  /// 构造该 `Module` 的选项。
  SoftMarginLossOptions options;
};

/// 用于 `SoftMarginLossImpl` 的 `ModuleHolder` 子类。
/// 查看 `SoftMarginLossImpl` 类的文档，了解其提供的方法，以及如何使用
/// `torch::nn::SoftMarginLossOptions` 创建 `SoftMarginLoss` 模块。
/// 查看 `ModuleHolder` 的文档，了解 PyTorch 模块存储语义。
TORCH_MODULE(SoftMarginLoss);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MultiLabelSoftMarginLoss
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// 创建一个 criterion，用于优化输入 x 和目标 y 之间的多标签一对所有损失，基于最大熵
/// x 和 y 的大小为 (N, C)。详见 https://pytorch.org/docs/main/nn.html#torch.nn.MultiLabelSoftMarginLoss
///
/// 查看 `torch::nn::MultiLabelSoftMarginLossOptions` 类的文档，了解该模块支持的构造参数。
///
/// 示例：
/// ```
/// MultiLabelSoftMarginLoss model(MultiLabelSoftMarginLossOptions().reduction(torch::kNone).weight(weight));
/// ```py
struct TORCH_API MultiLabelSoftMarginLossImpl
    : public Cloneable<MultiLabelSoftMarginLossImpl> {
  explicit MultiLabelSoftMarginLossImpl(
      MultiLabelSoftMarginLossOptions options_ = {});

  /// 将 `MultiLabelSoftMarginLoss` 模块美化打印到给定的 `stream` 中。
  void pretty_print(std::ostream& stream) const override;

  void reset() override;

  /// 前向传播函数，计算输入张量和目标张量之间的损失
  Tensor forward(const Tensor& input, const Tensor& target);

  /// 构造该 `Module` 的选项。
  MultiLabelSoftMarginLossOptions options;
};

/// 用于 `MultiLabelSoftMarginLossImpl` 的 `ModuleHolder` 子类。
/// 查看 `MultiLabelSoftMarginLossImpl` 类的文档，了解其提供的方法，以及如何使用
/// `torch::nn::MultiLabelSoftMarginLossOptions` 创建 `MultiLabelSoftMarginLoss` 模块。
/// Define a TORCH_MODULE for MultiLabelSoftMarginLoss, allowing it to be used as a Torch module.
/// See `ModuleHolder` documentation for details on PyTorch's module storage semantics.
TORCH_MODULE(MultiLabelSoftMarginLoss);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ TripletMarginLoss
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Define a criterion for computing triplet loss between input tensors `x1`, `x2`, `x3` with a specified margin.
/// This is used to measure relative similarity between samples (`a`, `p`, `n` representing anchor, positive examples, and negative examples).
/// Input tensors should have shapes (N, D).
/// See https://pytorch.org/docs/main/nn.html#torch.nn.TripletMarginLoss for detailed behavior.
///
/// Refer to `torch::nn::TripletMarginLossOptions` for supported constructor arguments.
///
/// Example:
/// ```
/// TripletMarginLoss
/// model(TripletMarginLossOptions().margin(3).p(2).eps(1e-06).swap(false));
/// ```py
struct TORCH_API TripletMarginLossImpl : public Cloneable<TripletMarginLossImpl> {
  explicit TripletMarginLossImpl(TripletMarginLossOptions options_ = {});

  void reset() override;

  /// Pretty prints the `TripletMarginLoss` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// Computes the forward pass of the `TripletMarginLoss` module.
  Tensor forward(const Tensor& anchor, const Tensor& positive, const Tensor& negative);

  /// The options with which this `Module` was constructed.
  TripletMarginLossOptions options;
};

/// A `ModuleHolder` subclass for `TripletMarginLossImpl`.
/// Provides access to methods of `TripletMarginLossImpl` and usage examples with `torch::nn::TripletMarginLossOptions`.
/// See `ModuleHolder` documentation for PyTorch's module storage semantics.
TORCH_MODULE(TripletMarginLoss);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ TripletMarginWithDistanceLoss
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Define a criterion for computing triplet loss based on input tensors `a`, `p`, and `n`
/// (representing anchor, positive, and negative examples, respectively).
/// Uses a nonnegative real-valued distance function to compute distances.
/// See https://pytorch.org/docs/main/nn.html#torch.nn.TripletMarginWithDistanceLoss for detailed behavior.
///
/// Refer to `torch::nn::TripletMarginWithDistanceLossOptions` for supported constructor arguments.
///
/// Example:
/// ```
/// TripletMarginWithDistanceLoss
/// model(TripletMarginWithDistanceLossOptions().margin(3).swap(false));
/// ```py
/// ```
/// Negative log likelihood loss with Poisson distribution of target.
/// See https://pytorch.org/docs/main/nn.html#torch.nn.PoissonNLLLoss to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::PoissonNLLLossOptions` class to learn
/// what constructor arguments are supported for this module.
///
/// Example:
/// ```py
/// PoissonNLLLoss
/// model(PoissonNLLLossOptions().log_input(true).full(true).eps(1e-6));
/// ```
struct TORCH_API PoissonNLLLossImpl : public Cloneable<PoissonNLLLossImpl> {
  explicit PoissonNLLLossImpl(PoissonNLLLossOptions options_ = {});

  void reset() override;

  /// Pretty prints the `PoissonNLLLoss` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  Tensor forward(
      const Tensor& input,
      const Tensor& target);

  /// The options with which this `Module` was constructed.
  PoissonNLLLossOptions options;
};

/// A `ModuleHolder` subclass for `PoissonNLLLossImpl`.
/// See the documentation for `PoissonNLLLossImpl` class to learn what methods it
/// provides, and examples of how to use `PoissonNLLLoss` with
/// `torch::nn::PoissonNLLLossOptions`. See the documentation for `ModuleHolder` to
/// learn about PyTorch's module storage semantics.
TORCH_MODULE(PoissonNLLLoss);
/// 创建一个`PoissonNLLLoss`模块，配置选项包括禁用输入的对数转换（log_input(false)）、完全模式（full(true)）、epsilon值为0.42（eps(0.42)）、损失计算方式为总和（reduction(torch::kSum)）。
/// ```py
struct TORCH_API PoissonNLLLossImpl : public Cloneable<PoissonNLLLossImpl> {
  explicit PoissonNLLLossImpl(PoissonNLLLossOptions options_ = {});

  void reset() override;

  /// 将`PoissonNLLLoss`模块以美观的形式打印到给定的`stream`中。
  void pretty_print(std::ostream& stream) const override;

  /// 对给定的`log_input`和`targets`张量进行前向传播计算。
  Tensor forward(const Tensor& log_input, const Tensor& targets);

  /// 构造此`Module`时使用的选项。
  PoissonNLLLossOptions options;
};

/// `PoissonNLLLossImpl`的`ModuleHolder`子类。
/// 参考`PoissonNLLLossImpl`类的文档了解其提供的方法，以及如何使用`torch::nn::PoissonNLLLossOptions`配置`PoissonNLLLoss`。
/// 参考`ModuleHolder`的文档了解PyTorch模块的存储语义。
TORCH_MODULE(PoissonNLLLoss);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MarginRankingLoss ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// 创建一个准则，用于计算给定两个一维小批量张量`x1`和`x2`，以及一个标签一维小批量张量`y`（包含1或-1）的损失。
/// 参考 https://pytorch.org/docs/main/nn.html#torch.nn.MarginRankingLoss 了解此模块的详细行为。
///
/// 参考`torch::nn::MarginRankingLossOptions`类的文档了解此模块支持的构造参数。
///
/// 示例:
/// ```
/// MarginRankingLoss
/// model(MarginRankingLossOptions().margin(0.5).reduction(torch::kSum));
/// ```py
struct TORCH_API MarginRankingLossImpl
    : public Cloneable<MarginRankingLossImpl> {
  explicit MarginRankingLossImpl(MarginRankingLossOptions options_ = {});

  void reset() override;

  /// 将`MarginRankingLoss`模块以美观的形式打印到给定的`stream`中。
  void pretty_print(std::ostream& stream) const override;

  /// 对给定的`input1`、`input2`和`targets`张量进行前向传播计算。
  Tensor forward(
      const Tensor& input1,
      const Tensor& input2,
      const Tensor& targets);

  /// 构造此`Module`时使用的选项。
  MarginRankingLossOptions options;
};

/// `MarginRankingLossImpl`的`ModuleHolder`子类。
/// 参考`MarginRankingLossImpl`类的文档了解其提供的方法，以及如何使用`torch::nn::MarginRankingLossOptions`配置`MarginRankingLoss`。
/// 参考`ModuleHolder`的文档了解PyTorch模块的存储语义。
TORCH_MODULE(MarginRankingLoss);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ NLLLoss ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// 负对数似然损失。用于训练一个包含`C`类别的分类问题。
/// 参考 https://pytorch.org/docs/main/nn.html#torch.nn.NLLLoss 了解此模块的详细行为。
///
/// 参考`torch::nn::NLLLossOptions`类的文档了解此模块支持的构造参数。
///
/// 示例:
/// ```
/// 创建一个 NLLLoss 模型，使用给定的选项初始化，其中 ignore_index 设置为 -100，reduction 设置为 torch::kMean。
/// ```py
struct TORCH_API NLLLossImpl : public Cloneable<NLLLossImpl> {
  explicit NLLLossImpl(NLLLossOptions options_ = {});

  /// 将 `NLLLoss` 模块美化打印到给定的流中。
  void pretty_print(std::ostream& stream) const override;

  void reset() override;

  /// 对输入和目标计算前向传播，返回计算出的 Tensor。
  Tensor forward(const Tensor& input, const Tensor& target);

  /// 构造此 `Module` 的选项。
  NLLLossOptions options;

  /// 给每个类别手动设置的重新缩放权重。
  Tensor weight;
};

/// `NLLLossImpl` 的 `ModuleHolder` 子类。
/// 查看 `NLLLossImpl` 类的文档以了解其提供的方法，以及如何使用 `torch::nn::NLLLossOptions` 使用 `NLLLoss` 的示例。
/// 查看 `ModuleHolder` 的文档以了解 PyTorch 模块存储语义。
TORCH_MODULE(NLLLoss);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CrossEntropyLoss
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// 创建一个计算输入和目标之间交叉熵损失的标准。查看 https://pytorch.org/docs/main/nn.html#torch.nn.CrossEntropyLoss 以了解此模块的确切行为。
///
/// 查看 `torch::nn::CrossEntropyLossOptions` 类的文档，了解支持此模块的构造函数参数。
///
/// 示例:
/// ```
/// CrossEntropyLoss
/// model(CrossEntropyLossOptions().ignore_index(-100).reduction(torch::kMean));
/// ```py
struct TORCH_API CrossEntropyLossImpl : public Cloneable<CrossEntropyLossImpl> {
  explicit CrossEntropyLossImpl(CrossEntropyLossOptions options_ = {});

  void reset() override;

  /// 将 `CrossEntropyLoss` 模块美化打印到给定的流中。
  void pretty_print(std::ostream& stream) const override;

  /// 对输入和目标计算前向传播，返回计算出的 Tensor。
  Tensor forward(const Tensor& input, const Tensor& target);

  /// 构造此 `Module` 的选项。
  CrossEntropyLossOptions options;

  /// 给每个类别手动设置的重新缩放权重。
  Tensor weight;
};

/// `CrossEntropyLossImpl` 的 `ModuleHolder` 子类。
/// 查看 `CrossEntropyLossImpl` 类的文档以了解其提供的方法，以及如何使用 `torch::nn::CrossEntropyLossOptions` 使用 `CrossEntropyLoss` 的示例。
/// 查看 `ModuleHolder` 的文档以了解 PyTorch 模块存储语义。
TORCH_MODULE(CrossEntropyLoss);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ BCEWithLogitsLoss
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// 此损失将 `Sigmoid` 层和 `BCELoss` 结合为单个类。此版本比分别使用 `Sigmoid` 和 `BCELoss` 更稳定，
/// 因为通过将操作组合成一个层，我们利用了数值稳定性的对数-求和-指数技巧（log-sum-exp trick）。
/// 查看 https://pytorch.org/docs/main/nn.html#torch.nn.BCEWithLogitsLoss 以了解此模块的确切行为。
///
/// 
/// See the documentation for `torch::nn::BCEWithLogitsLossOptions` class to
/// learn what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// BCEWithLogitsLoss
/// model(BCEWithLogitsLossOptions().reduction(torch::kNone).weight(weight));
/// ```py
struct TORCH_API BCEWithLogitsLossImpl
    : public Cloneable<BCEWithLogitsLossImpl> {
  
  /// Constructor for initializing BCEWithLogitsLossImpl with given options.
  explicit BCEWithLogitsLossImpl(BCEWithLogitsLossOptions options_ = {});

  /// Resets the state of the module.
  void reset() override;

  /// Pretty prints the `BCEWithLogitsLoss` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// Computes the forward pass of the `BCEWithLogitsLoss` module.
  ///
  /// Computes the loss given the input tensor and the target tensor.
  /// Returns the computed loss tensor.
  Tensor forward(const Tensor& input, const Tensor& target);

  /// The options with which this `Module` was constructed.
  BCEWithLogitsLossOptions options;

  /// A manual rescaling weight given to the loss of each batch element.
  Tensor weight;

  /// A weight of positive examples.
  Tensor pos_weight;
};

/// A `ModuleHolder` subclass for `BCEWithLogitsLossImpl`.
///
/// This provides storage and initialization semantics for `BCEWithLogitsLossImpl`
/// within PyTorch's module ecosystem.
/// See the documentation for `BCEWithLogitsLossImpl` class to learn what
/// methods it provides, and examples of how to use `BCEWithLogitsLoss` with
/// `torch::nn::BCEWithLogitsLossOptions`.
/// See the documentation for `ModuleHolder` to learn about PyTorch's module storage semantics.
TORCH_MODULE(BCEWithLogitsLoss);

} // namespace nn
} // namespace torch
```