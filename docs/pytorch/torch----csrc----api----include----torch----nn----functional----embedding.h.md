# `.\pytorch\torch\csrc\api\include\torch\nn\functional\embedding.h`

```
#pragma once

#include <torch/nn/options/embedding.h> // 包含嵌入操作的选项定义

namespace torch {
namespace nn {
namespace functional {

inline Tensor one_hot(const Tensor& tensor, int64_t num_classes = -1) {
  return torch::one_hot(tensor, num_classes); // 调用torch库中的one_hot函数进行独热编码
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline void _no_grad_embedding_renorm_(
    Tensor weight,
    const Tensor& input,
    float max_norm,
    float norm_type) {
  torch::NoGradGuard no_grad; // 创建一个禁止梯度计算的上下文管理器
  torch::embedding_renorm_(weight, input, max_norm, norm_type); // 调用torch库中的embedding_renorm_函数进行嵌入张量的归一化
}

inline Tensor embedding(
    const Tensor& input,
    const Tensor& weight,
    std::optional<int64_t> padding_idx,
    std::optional<double> max_norm,
    double norm_type,
    bool scale_grad_by_freq,
    bool sparse) {
  auto input_ = input; // 复制输入张量

  if (padding_idx != c10::nullopt) { // 检查是否提供了填充索引
    if (*padding_idx > 0) {
      TORCH_CHECK(
          *padding_idx < weight.size(0),
          "Padding_idx must be within num_embeddings"); // 检查填充索引是否在有效范围内
    } else if (*padding_idx < 0) {
      TORCH_CHECK(
          *padding_idx >= -weight.size(0),
          "Padding_idx must be within num_embedding");
      padding_idx = weight.size(0) + *padding_idx; // 将负数填充索引转换为有效的索引值
    }
  } else {
    padding_idx = -1; // 如果未提供填充索引，则设置为默认值-1
  }

  if (max_norm != c10::nullopt) { // 检查是否提供了最大范数
    input_ = input_.contiguous(); // 强制输入张量为连续的
    // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
    _no_grad_embedding_renorm_(weight, input_, *max_norm, norm_type); // 调用内部函数对嵌入权重进行归一化操作
  }
  return torch::embedding(
      weight, input_, *padding_idx, scale_grad_by_freq, sparse); // 调用torch库中的embedding函数进行嵌入操作
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.embedding
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::EmbeddingFuncOptions`
/// class to learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::embedding(input, weight,
/// F::EmbeddingFuncOptions().norm_type(2.5).scale_grad_by_freq(true).sparse(true));
/// ```
inline Tensor embedding(
    const Tensor& input,
    const Tensor& weight,
    const EmbeddingFuncOptions& options = {}) {
  return detail::embedding(
      input,
      weight,
      options.padding_idx(),
      options.max_norm(),
      options.norm_type(),
      options.scale_grad_by_freq(),
      options.sparse()); // 调用内部detail命名空间中的embedding函数，并根据选项进行配置
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor embedding_bag(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& offsets,
    std::optional<double> max_norm,
    double norm_type,
    bool scale_grad_by_freq,
    EmbeddingBagMode mode,
    bool sparse,
    const Tensor& per_sample_weights,
    bool include_last_offset,
    `
        std::optional<int64_t> padding_idx) {  // 函数定义，接收一个可选的 padding_idx 参数
      auto input_ = input;  // 将 input 赋值给 input_ 变量
      auto offsets_ = offsets;  // 将 offsets 赋值给 offsets_ 变量
      auto per_sample_weights_ = per_sample_weights;  // 将 per_sample_weights 赋值给 per_sample_weights_ 变量
    
      // 检查 per_sample_weights 是否定义，且其形状是否与 input 相同
      TORCH_CHECK(
          !per_sample_weights_.defined() ||
              input_.sizes() == per_sample_weights_.sizes(),
          "embedding_bag: If per_sample_weights (",
          per_sample_weights_.sizes(),
          ") is not null, then it must have the same shape as the input (",
          input_.sizes(),
          ")");
    
      // 如果 input 的维度为 2
      if (input_.dim() == 2) {
        // 检查 offsets 是否定义，若 input 是二维，则 offsets 必须为 null
        TORCH_CHECK(
            !offsets_.defined(),
            "If input is 2D, then offsets has to be null, as input is treated is a mini-batch of fixed length sequences. However, found offsets of type Tensor");
        // 生成从 0 到 input.numel()，步长为 input.size(1) 的序列
        offsets_ = torch::arange(
            0,
            input_.numel(),
            input_.size(1),
            torch::TensorOptions().dtype(torch::kLong).device(input_.device()));
        // 将 input 重塑为一维向量
        input_ = input_.reshape(-1);
        // 如果 per_sample_weights 定义，将其重塑为一维向量
        if (per_sample_weights_.defined()) {
          per_sample_weights_ = per_sample_weights_.reshape(-1);
        }
      } else if (input_.dim() == 1) {  // 如果 input 的维度为 1
        // 检查 offsets 是否定义，并且必须是 1D Tensor
        TORCH_CHECK(
            offsets_.defined(), "offsets has to be a 1D Tensor but got null");
        TORCH_CHECK(offsets_.dim() == 1, "offsets has to be a 1D Tensor");
      } else {  // 如果 input 的维度不是 1 或 2
        // 抛出异常，提示 input 的维度必须是 1 或 2
        TORCH_CHECK(
            false,
            "input has to be 1D or 2D Tensor, but got Tensor of dimension ",
            input_.dim());
      }
    
      // 定义 mode_enum 变量
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      int mode_enum;
      // 判断 mode 的类型，并赋值给 mode_enum
      if (std::holds_alternative<enumtype::kSum>(mode)) {
        mode_enum = 0;
      } else if (std::holds_alternative<enumtype::kMean>(mode)) {
        mode_enum = 1;
      } else if (std::holds_alternative<enumtype::kMax>(mode)) {
        mode_enum = 2;
        // 检查 max 模式是否支持 scaling the gradient by the frequency
        TORCH_CHECK(
            !scale_grad_by_freq,
            "max mode does not support scaling the gradient by the frequency");
        // 检查 max 模式是否支持 sparse weights
        TORCH_CHECK(!sparse, "max mode does not support sparse weights");
      } else {  // 如果 mode 不在 sum, mean, max 三种模式中
        TORCH_CHECK(false, "mode has to be one of sum, mean or max");
      }
    
      // 如果 max_norm 不为 null，执行 _no_grad_embedding_renorm_ 函数
      if (max_norm != c10::nullopt) {
        // 调用 _no_grad_embedding_renorm_ 函数，进行归一化处理
        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
        _no_grad_embedding_renorm_(weight, input_, *max_norm, norm_type);
      }
    
      // 检查 per_sample_weights 是否定义，并且 mode 是否为 sum
      TORCH_CHECK(
          !per_sample_weights_.defined() || std::get_if<enumtype::kSum>(&mode),
          "embedding_bag: per_sample_weights was not null. ",
          "per_sample_weights is only supported for mode='kSum' (got mode='",
          torch::enumtype::get_enum_name(mode),
          "').Please open a feature request on GitHub.");
    
      // 调用 torch::embedding_bag 函数，返回结果的第一个元素
      return std::get<0>(torch::embedding_bag(
          weight,
          input_,
          offsets_,
          scale_grad_by_freq,
          mode_enum,
          sparse,
          per_sample_weights_,
          include_last_offset,
          padding_idx));
/// 结束 detail 命名空间
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// 查看关于这个功能的确切行为的详细说明，请访问：
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.embedding_bag
///
/// 查看 `torch::nn::functional::EmbeddingBagFuncOptions` 类的文档，了解这个功能支持的可选参数。
///
/// 示例：
/// ```
/// namespace F = torch::nn::functional;
/// F::embedding_bag(input, weight,
///   F::EmbeddingBagFuncOptions().mode(torch::kSum).offsets(offsets));
/// ```
inline Tensor embedding_bag(
    const Tensor& input,
    const Tensor& weight,
    const EmbeddingBagFuncOptions& options = {}) {
  // 调用 detail 命名空间中的 embedding_bag 函数，传递所有选项参数
  return detail::embedding_bag(
      input,
      weight,
      options.offsets(),
      options.max_norm(),
      options.norm_type(),
      options.scale_grad_by_freq(),
      options.mode(),
      options.sparse(),
      options.per_sample_weights(),
      options.include_last_offset(),
      options.padding_idx());
}

// 结束 functional 命名空间
} // namespace functional

// 结束 nn 命名空间
} // namespace nn

// 结束 torch 命名空间
} // namespace torch
```