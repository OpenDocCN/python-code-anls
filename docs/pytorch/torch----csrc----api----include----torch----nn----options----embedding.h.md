# `.\pytorch\torch\csrc\api\include\torch\nn\options\embedding.h`

```py
#pragma once


// 指令：指示编译器只包含该头文件一次，防止多次包含导致的重定义错误

#include <torch/arg.h>
#include <torch/csrc/Export.h>
#include <torch/enum.h>
#include <torch/types.h>


// 包含 Torch C++ 库的头文件，提供了 Torch 库中的各种类型和函数声明

namespace torch {
namespace nn {


// 命名空间：torch 下的 nn 命名空间，包含了 Torch 深度学习框架中的神经网络相关的类和函数

/// Options for the `Embedding` module.
///
/// Example:
/// ```
/// Embedding model(EmbeddingOptions(10,
/// 2).padding_idx(3).max_norm(2).norm_type(2.5).scale_grad_by_freq(true).sparse(true));
/// ```py
struct TORCH_API EmbeddingOptions {


// 结构体声明：EmbeddingOptions 是用于 Embedding 模块的选项结构体

  EmbeddingOptions(int64_t num_embeddings, int64_t embedding_dim);


// 构造函数声明：初始化 EmbeddingOptions 结构体对象，设置词典大小和嵌入维度

  /// The size of the dictionary of embeddings.
  TORCH_ARG(int64_t, num_embeddings);
  /// The size of each embedding vector.
  TORCH_ARG(int64_t, embedding_dim);
  /// If specified, the entries at `padding_idx` do not contribute to the
  /// gradient; therefore, the embedding vector at `padding_idx` is not updated
  /// during training, i.e. it remains as a fixed "pad". For a newly constructed
  /// Embedding, the embedding vector at `padding_idx` will default to all
  /// zeros, but can be updated to another value to be used as the padding
  /// vector.
  TORCH_ARG(std::optional<int64_t>, padding_idx) = c10::nullopt;
  /// If given, each embedding vector with norm larger than `max_norm` is
  /// renormalized to have norm `max_norm`.
  TORCH_ARG(std::optional<double>, max_norm) = c10::nullopt;
  /// The p of the p-norm to compute for the `max_norm` option. Default ``2``.
  TORCH_ARG(double, norm_type) = 2.;
  /// If given, this will scale gradients by the inverse of frequency of the
  /// words in the mini-batch. Default ``false``.
  TORCH_ARG(bool, scale_grad_by_freq) = false;
  /// If ``true``, gradient w.r.t. `weight` matrix will be a sparse tensor.
  TORCH_ARG(bool, sparse) = false;
  /// The learnable weights of the module of shape (num_embeddings,
  /// embedding_dim)
  TORCH_ARG(torch::Tensor, _weight) = Tensor();


// 成员变量声明：EmbeddingOptions 结构体的成员变量，包括词典大小、嵌入维度、填充索引、最大范数等选项

};

// ============================================================================

/// Options for the `Embedding::from_pretrained` function.


// 注释：定义了 Embedding::from_pretrained 函数的选项结构
/// 结构体定义，用于表示从预训练模型加载的嵌入层选项
struct TORCH_API EmbeddingFromPretrainedOptions {
  /// 如果为 ``true``，则在学习过程中张量不会被更新。
  /// 等同于 ``embedding.weight.requires_grad_(false)``。默认为 ``true``
  TORCH_ARG(bool, freeze) = true;
  /// 如果指定，`padding_idx` 处的条目不会对梯度产生贡献；
  /// 因此，在训练过程中，`padding_idx` 处的嵌入向量不会被更新，即保持固定的“填充”状态。
  TORCH_ARG(std::optional<int64_t>, padding_idx) = c10::nullopt;
  /// 如果给定，每个嵌入向量的范数大于 `max_norm` 将被重新归一化为 `max_norm`。
  TORCH_ARG(std::optional<double>, max_norm) = c10::nullopt;
  /// 用于 `max_norm` 选项计算的 p-norm。默认为 ``2``。
  TORCH_ARG(double, norm_type) = 2.;
  /// 如果给定，将根据小批量中单词频率的倒数来缩放梯度。默认为 ``false``。
  TORCH_ARG(bool, scale_grad_by_freq) = false;
  /// 如果为 ``true``，则 `weight` 矩阵的梯度将是稀疏张量。
  TORCH_ARG(bool, sparse) = false;
};

// ============================================================================

namespace functional {

/// `torch::nn::functional::embedding` 的选项。
///
/// 示例:
/// ```
/// namespace F = torch::nn::functional;
/// F::embedding(input, weight,
/// F::EmbeddingFuncOptions().norm_type(2.5).scale_grad_by_freq(true).sparse(true));
/// ```py
struct TORCH_API EmbeddingFuncOptions {
  /// 如果指定，`padding_idx` 处的条目不会对梯度产生贡献；
  /// 因此，在训练过程中，`padding_idx` 处的嵌入向量不会被更新，即保持固定的“填充”状态。
  TORCH_ARG(std::optional<int64_t>, padding_idx) = c10::nullopt;
  /// 如果给定，每个嵌入向量的范数大于 `max_norm` 将被重新归一化为 `max_norm`。
  TORCH_ARG(std::optional<double>, max_norm) = c10::nullopt;
  /// 用于 `max_norm` 选项计算的 p-norm。默认为 ``2``。
  TORCH_ARG(double, norm_type) = 2.;
  /// 如果给定，将根据小批量中单词频率的倒数来缩放梯度。默认为 ``false``。
  TORCH_ARG(bool, scale_grad_by_freq) = false;
  /// 如果为 ``true``，则 `weight` 矩阵的梯度将是稀疏张量。
  TORCH_ARG(bool, sparse) = false;
};

} // namespace functional

// ============================================================================

/// `EmbeddingBag` 模块的选项。
///
/// 示例:
/// ```
/// EmbeddingBag model(EmbeddingBagOptions(10,
/// 2).max_norm(2).norm_type(2.5).scale_grad_by_freq(true).sparse(true).mode(torch::kSum));
/// ```py
typedef std::variant<enumtype::kSum, enumtype::kMean, enumtype::kMax>
    EmbeddingBagMode;
/// 定义了一个结构体 `EmbeddingBagOptions`，用于存储嵌入袋的选项参数。
struct TORCH_API EmbeddingBagOptions {
  /// 构造函数，初始化嵌入袋的选项参数。
  EmbeddingBagOptions(int64_t num_embeddings, int64_t embedding_dim);

  /// 嵌入字典的大小。
  TORCH_ARG(int64_t, num_embeddings);
  /// 每个嵌入向量的维度大小。
  TORCH_ARG(int64_t, embedding_dim);
  /// 如果指定，则对于大于 `max_norm` 的每个嵌入向量进行重新归一化，使其范数为 `max_norm`。
  TORCH_ARG(std::optional<double>, max_norm) = c10::nullopt;
  /// 计算 `max_norm` 选项时使用的 p 范数。默认为 ``2``。
  TORCH_ARG(double, norm_type) = 2.;
  /// 如果指定，将按照单词在小批量中的频率的倒数来缩放梯度。默认为 ``false``。
  /// 注意：当 ``mode="kMax"`` 时，不支持此选项。
  TORCH_ARG(bool, scale_grad_by_freq) = false;
  /// 规定如何减少嵌入袋的方式。``"kSum"``, ``"kMean"``, 或 ``"kMax"``。``"kSum"`` 对加权和进行计算，考虑 `per_sample_weights`。``"kMean"`` 对袋中值的平均值进行计算，``"kMax"`` 对每个袋中的最大值进行计算。
  TORCH_ARG(EmbeddingBagMode, mode) = torch::kMean;
  /// 如果 ``true``，`weight` 矩阵的梯度将是稀疏张量。注意：当 ``mode="kMax"`` 时，不支持此选项。
  TORCH_ARG(bool, sparse) = false;
  /// 模块的可学习权重，形状为 (num_embeddings, embedding_dim)。
  TORCH_ARG(torch::Tensor, _weight) = Tensor();
  /// 如果 ``true``，`offsets` 会有一个额外的元素，最后一个元素等于 `indices` 的大小。这与 CSR 格式相匹配。
  TORCH_ARG(bool, include_last_offset) = false;
  /// 如果指定，`padding_idx` 处的条目不会对梯度做出贡献；因此，训练期间不会更新 `padding_idx` 处的嵌入向量，即它保持为固定的“填充”。对于新构造的 EmbeddingBag，`padding_idx` 处的嵌入向量默认为全零，但可以更新为另一个值以用作填充向量。注意，`padding_idx` 处的嵌入向量被排除在减少操作之外。
  TORCH_ARG(std::optional<int64_t>, padding_idx) = c10::nullopt;
};

// ============================================================================

/// `EmbeddingBag::from_pretrained` 函数的选项。
// 定义了一个结构体 EmbeddingBagFromPretrainedOptions，用于配置预训练嵌入包的选项

struct TORCH_API EmbeddingBagFromPretrainedOptions {
  /// 如果为 true，则在学习过程中不更新张量。相当于 `embeddingbag.weight.requires_grad_(false)`。默认为 true
  TORCH_ARG(bool, freeze) = true;
  /// 如果提供，每个嵌入向量的范数大于 `max_norm` 将被重新归一化为 `max_norm`
  TORCH_ARG(std::optional<double>, max_norm) = c10::nullopt;
  /// 计算 `max_norm` 选项时使用的 p 范数。默认为 `2`
  TORCH_ARG(double, norm_type) = 2.;
  /// 如果提供，将按照单词在小批量中的频率的倒数来缩放梯度。默认为 false。注意：当 `mode="kMax"` 时不支持此选项。
  TORCH_ARG(bool, scale_grad_by_freq) = false;
  /// `"kSum"`, `"kMean"` 或 `"kMax"`。指定减少嵌入包的方式。`"kSum"` 计算加权和，考虑 `per_sample_weights`。`"kMean"` 计算包中值的平均值，`"kMax"` 计算每个包中的最大值。
  TORCH_ARG(EmbeddingBagMode, mode) = torch::kMean;
  /// 如果为 true，则 `weight` 矩阵的梯度将是一个稀疏张量。注意：当 `mode="kMax"` 时不支持此选项。
  TORCH_ARG(bool, sparse) = false;
  /// 如果为 true，则 `offsets` 有一个额外的元素，最后一个元素等同于 `indices` 的大小。这符合 CSR 格式。注意：当前仅在 `mode="sum"` 时支持此选项。
  TORCH_ARG(bool, include_last_offset) = false;
  /// 如果指定，`padding_idx` 处的条目不会贡献梯度；因此，在训练期间不会更新 `padding_idx` 处的嵌入向量，即它保持固定的“填充”。注意，`padding_idx` 处的嵌入向量被排除在减少之外。
  TORCH_ARG(std::optional<int64_t>, padding_idx) = c10::nullopt;
};

// ============================================================================

namespace functional {

/// 用于 `torch::nn::functional::embedding_bag` 的选项。
///
/// 示例:
/// ```
/// namespace F = torch::nn::functional;
/// F::embedding_bag(input, weight,
/// F::EmbeddingBagFuncOptions().mode(torch::kSum).offsets(offsets));
/// ```py
/// 定义了 EmbeddingBagFuncOptions 结构体，用于存储 EmbeddingBag 函数的参数选项。
struct TORCH_API EmbeddingBagFuncOptions {
  /// 当 `input` 是一维时使用。`offsets` 确定了 `input` 中每个 bag（序列）的起始索引位置。
  TORCH_ARG(torch::Tensor, offsets) = Tensor();
  
  /// 如果给定，每个嵌入向量的范数大于 `max_norm` 的将被重新归一化为具有 `max_norm` 的范数。
  TORCH_ARG(std::optional<double>, max_norm) = c10::nullopt;
  
  /// 计算 `max_norm` 选项时使用的 p 范数。默认为 ``2``。
  TORCH_ARG(double, norm_type) = 2.;
  
  /// 如果给定，将按照单词在小批量中的频率的倒数来缩放梯度。默认 ``false``。
  /// 注意：当 ``mode="kMax"`` 时不支持此选项。
  TORCH_ARG(bool, scale_grad_by_freq) = false;
  
  /// 指定了减少 bag 的方式。``"kSum"`` 计算加权和，考虑 `per_sample_weights`。
  /// ``"kMean"`` 计算 bag 中值的平均值，``"kMax"`` 计算每个 bag 的最大值。
  TORCH_ARG(EmbeddingBagMode, mode) = torch::kMean;
  
  /// 如果 ``true``，权重矩阵 `weight` 的梯度将是稀疏张量。
  /// 注意：当 ``mode="kMax"`` 时不支持此选项。
  TORCH_ARG(bool, sparse) = false;
  
  /// 一个 float / double 权重张量，或者 None 表示所有权重都应为 1。如果指定，
  /// `per_sample_weights` 必须与 input 具有相同的形状，并且如果 `offsets` 不是 None，则被视为具有相同的 `offsets`。
  TORCH_ARG(torch::Tensor, per_sample_weights) = Tensor();
  
  /// 如果 ``true``，`offsets` 有一个额外的元素，最后一个元素等同于 `indices` 的大小。这符合 CSR 格式。
  /// 注意：当前仅在 ``mode="sum"`` 时支持此选项。
  TORCH_ARG(bool, include_last_offset) = false;
  
  /// 如果指定，`padding_idx` 处的条目不会贡献梯度；因此，在训练过程中，padding_idx 处的嵌入向量不会更新，即它保持固定的 "pad"。
  /// 注意，`padding_idx` 处的嵌入向量在减少中被排除。
  TORCH_ARG(std::optional<int64_t>, padding_idx) = c10::nullopt;
};
```