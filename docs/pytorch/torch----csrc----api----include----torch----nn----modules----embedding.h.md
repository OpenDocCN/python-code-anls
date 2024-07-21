# `.\pytorch\torch\csrc\api\include\torch\nn\modules\embedding.h`

```py
// 预处理指令，表示本文件只被编译一次
#pragma once

// 包含 Torch 的相关头文件
#include <torch/nn/cloneable.h>
#include <torch/nn/functional/embedding.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/options/embedding.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>

// 包含标准库头文件
#include <cstddef>

// Torch 命名空间
namespace torch {
namespace nn {

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Embedding
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// 在固定大小的嵌入表中执行查找。
/// 参考 https://pytorch.org/docs/main/nn.html#torch.nn.Embedding 以了解本模块的确切行为。
///
/// 查看 `torch::nn::EmbeddingOptions` 类的文档，了解此模块支持的构造参数。
///
/// 示例:
/// ```
/// Embedding model(EmbeddingOptions(10, 2).padding_idx(3).max_norm(2).norm_type(2.5).scale_grad_by_freq(true).sparse(true));
/// ```py
class TORCH_API EmbeddingImpl : public torch::nn::Cloneable<EmbeddingImpl> {
 public:
  // 构造函数，根据给定的参数创建嵌入层
  EmbeddingImpl(int64_t num_embeddings, int64_t embedding_dim)
      : EmbeddingImpl(EmbeddingOptions(num_embeddings, embedding_dim)) {}
  explicit EmbeddingImpl(EmbeddingOptions options_);

  // 重置模型参数
  void reset() override;

  // 重置嵌入层的参数
  void reset_parameters();

  /// 将 `Embedding` 模块漂亮地打印到给定的 `stream` 中。
  void pretty_print(std::ostream& stream) const override;

  /// 使用给定的 `indices` 在嵌入表中进行查找，并返回结果。
  Tensor forward(const Tensor& indices);

  /// 用于配置此 `Embedding` 模块的选项。
  /// 构造后对 `EmbeddingOptions` 的更改不会生效。
  EmbeddingOptions options;

  /// 嵌入表。
  Tensor weight;
};

/// `EmbeddingImpl` 的 `ModuleHolder` 子类。
/// 查看 `EmbeddingImpl` 类的文档，了解其提供的方法，以及如何使用 `torch::nn::EmbeddingOptions` 使用 `Embedding` 的示例。
/// 查看 `ModuleHolder` 的文档，了解 PyTorch 的模块存储语义。
class Embedding : public torch::nn::ModuleHolder<EmbeddingImpl> {
 public:
  using torch::nn::ModuleHolder<EmbeddingImpl>::ModuleHolder;

  /// 查看 `torch::nn::EmbeddingFromPretrainedOptions` 类的文档，了解此函数支持的可选参数。
  static Embedding from_pretrained(
      const torch::Tensor& embeddings,
      const EmbeddingFromPretrainedOptions& options = {}) {
    TORCH_CHECK(
        embeddings.dim() == 2,
        "Embeddings parameter is expected to be 2-dimensional");

    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    int64_t rows, cols;
    // 获取张量的维度大小
    rows = embeddings.size(0);
    cols = embeddings.size(1);
    # 使用给定的行数和列数创建嵌入层对象，并配置其参数
    Embedding embedding(EmbeddingOptions(rows, cols)
                            ._weight(embeddings)  # 设置嵌入层的权重为给定的嵌入矩阵
                            .padding_idx(options.padding_idx())  # 设置填充索引，用于填充序列的位置
                            .max_norm(options.max_norm())  # 设置权重的最大范数限制
                            .norm_type(options.norm_type())  # 设置权重的归一化类型
                            .scale_grad_by_freq(options.scale_grad_by_freq())  # 根据词频缩放梯度
                            .sparse(options.sparse()));  # 指定是否使用稀疏张量表示嵌入层
    # 根据选项决定是否冻结权重，设置是否需要计算梯度
    embedding->weight.set_requires_grad(!options.freeze());
    # 返回创建的嵌入层对象
    return embedding;
}
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ EmbeddingBag
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// 计算嵌入包的总和或平均值，而无需实例化中间嵌入。
/// 请参阅 https://pytorch.org/docs/main/nn.html#torch.nn.EmbeddingBag 了解此模块的确切行为。
///
/// 查看 `torch::nn::EmbeddingBagOptions` 类的文档以了解此模块支持的构造函数参数。
///
/// 示例:
/// ```
/// EmbeddingBag model(EmbeddingBagOptions(10,
/// 2).max_norm(2).norm_type(2.5).scale_grad_by_freq(true).sparse(true).mode(torch::kSum).padding_idx(1));
/// ```py
class TORCH_API EmbeddingBagImpl
    : public torch::nn::Cloneable<EmbeddingBagImpl> {
 public:
  EmbeddingBagImpl(int64_t num_embeddings, int64_t embedding_dim)
      : EmbeddingBagImpl(EmbeddingBagOptions(num_embeddings, embedding_dim)) {}
  explicit EmbeddingBagImpl(EmbeddingBagOptions options_);

  void reset() override;

  void reset_parameters();

  /// 将 `EmbeddingBag` 模块漂亮地打印到给定的 `stream` 中。
  void pretty_print(std::ostream& stream) const override;

  /// 用于配置此 `EmbeddingBag` 模块的选项。
  EmbeddingBagOptions options;
  /// 嵌入表。
  Tensor weight;

  Tensor forward(
      const Tensor& input,
      const Tensor& offsets = {},
      const Tensor& per_sample_weights = {});

 protected:
  FORWARD_HAS_DEFAULT_ARGS({1, AnyValue(Tensor())}, {2, AnyValue(Tensor())})
};

/// `EmbeddingBagImpl` 的 `ModuleHolder` 子类。
/// 查看 `EmbeddingBagImpl` 类的文档，了解其提供的方法，以及如何使用 `torch::nn::EmbeddingBagOptions` 使用 `EmbeddingBag` 的示例。
/// 查看 `ModuleHolder` 的文档，了解 PyTorch 的模块存储语义。
class EmbeddingBag : public torch::nn::ModuleHolder<EmbeddingBagImpl> {
 public:
  using torch::nn::ModuleHolder<EmbeddingBagImpl>::ModuleHolder;

  /// 查看 `torch::nn::EmbeddingBagFromPretrainedOptions` 类的文档，了解此函数支持的可选参数。
  static EmbeddingBag from_pretrained(
      const torch::Tensor& embeddings,
      const EmbeddingBagFromPretrainedOptions& options = {}) {
    TORCH_CHECK(
        embeddings.dim() == 2,
        "Embeddings parameter is expected to be 2-dimensional");

    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    int64_t rows, cols;
    rows = embeddings.size(0);
    cols = embeddings.size(1);

    EmbeddingBag embeddingbag(
        EmbeddingBagOptions(rows, cols)
            ._weight(embeddings)
            .max_norm(options.max_norm())
            .norm_type(options.norm_type())
            .scale_grad_by_freq(options.scale_grad_by_freq())
            .mode(options.mode())
            .sparse(options.sparse())
            .padding_idx(options.padding_idx()));
    embeddingbag->weight.set_requires_grad(!options.freeze());
    return embeddingbag;
  }
};
} // 结束 nn 命名空间
} // 结束 torch 命名空间
```