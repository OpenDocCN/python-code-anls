# `.\pytorch\torch\csrc\api\include\torch\nn\functional\distance.h`

```
#pragma once

#include <torch/nn/options/distance.h>

namespace torch {
namespace nn {
namespace functional {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
// 定义了计算余弦相似度的内部函数
inline Tensor cosine_similarity(
    const Tensor& x1,
    const Tensor& x2,
    int64_t dim,
    double eps) {
  return torch::cosine_similarity(x1, x2, dim, eps);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// 查看此功能的详细行为，请参考官方文档：
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.cosine_similarity
///
/// 查看 `torch::nn::functional::CosineSimilarityFuncOptions` 类的文档，
/// 以了解此功能支持的可选参数。
///
/// 示例:
/// ```
/// namespace F = torch::nn::functional;
/// F::cosine_similarity(input1, input2,
///   F::CosineSimilarityFuncOptions().dim(1));
/// ```
// 定义了计算余弦相似度的函数，支持多种可选参数配置
inline Tensor cosine_similarity(
    const Tensor& x1,
    const Tensor& x2,
    const CosineSimilarityFuncOptions& options = {}) {
  return detail::cosine_similarity(x1, x2, options.dim(), options.eps());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
// 定义了计算成对距离的内部函数
inline Tensor pairwise_distance(
    const Tensor& x1,
    const Tensor& x2,
    double p,
    double eps,
    bool keepdim) {
  return torch::pairwise_distance(x1, x2, p, eps, keepdim);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// 查看此功能的详细行为，请参考官方文档：
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.pairwise_distance
///
/// 查看 `torch::nn::functional::PairwiseDistanceFuncOptions` 类的文档，
/// 以了解此功能支持的可选参数。
///
/// 示例:
/// ```
/// namespace F = torch::nn::functional;
/// F::pairwise_distance(input1, input2, F::PairwiseDistanceFuncOptions().p(1));
/// ```
// 定义了计算成对距离的函数，支持多种可选参数配置
inline Tensor pairwise_distance(
    const Tensor& x1,
    const Tensor& x2,
    const PairwiseDistanceFuncOptions& options = {}) {
  return detail::pairwise_distance(
      x1, x2, options.p(), options.eps(), options.keepdim());
}

// ============================================================================

/// 计算输入中每对行向量之间的 p 范数距离。
/// 如果行是连续的，则此函数将更快。
// 定义了计算 p 范数距离的函数
inline Tensor pdist(const Tensor& input, double p = 2.0) {
  return torch::pdist(input, p);
}

} // namespace functional
} // namespace nn
} // namespace torch
```