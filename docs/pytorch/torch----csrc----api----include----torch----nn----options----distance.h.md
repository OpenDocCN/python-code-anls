# `.\pytorch\torch\csrc\api\include\torch\nn\options\distance.h`

```py
#pragma once

#include <torch/arg.h>  // 包含 Torch 库中的参数定义头文件
#include <torch/csrc/Export.h>  // 包含 Torch 库中的导出定义头文件
#include <torch/types.h>  // 包含 Torch 库中的类型定义头文件

namespace torch {
namespace nn {

/// Options for the `CosineSimilarity` module.
///
/// Example:
/// ```
/// CosineSimilarity model(CosineSimilarityOptions().dim(0).eps(0.5));
/// ```py
struct TORCH_API CosineSimilarityOptions {
  /// Dimension where cosine similarity is computed. Default: 1
  TORCH_ARG(int64_t, dim) = 1;  // 计算余弦相似度的维度，默认为1
  /// Small value to avoid division by zero. Default: 1e-8
  TORCH_ARG(double, eps) = 1e-8;  // 避免除零的小值，默认为1e-8
};

namespace functional {
/// Options for `torch::nn::functional::cosine_similarity`.
///
/// See the documentation for `torch::nn::CosineSimilarityOptions` class to
/// learn what arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::cosine_similarity(input1, input2,
/// F::CosineSimilarityFuncOptions().dim(1));
/// ```py
using CosineSimilarityFuncOptions = CosineSimilarityOptions;  // 使用 CosineSimilarityOptions 的别名
} // namespace functional

// ============================================================================

/// Options for the `PairwiseDistance` module.
///
/// Example:
/// ```
/// PairwiseDistance
/// model(PairwiseDistanceOptions().p(3).eps(0.5).keepdim(true));
/// ```py
struct TORCH_API PairwiseDistanceOptions {
  /// The norm degree. Default: 2
  TORCH_ARG(double, p) = 2.0;  // 范数的度数，默认为2
  /// Small value to avoid division by zero. Default: 1e-6
  TORCH_ARG(double, eps) = 1e-6;  // 避免除零的小值，默认为1e-6
  /// Determines whether or not to keep the vector dimension. Default: false
  TORCH_ARG(bool, keepdim) = false;  // 决定是否保持向量的维度，默认为false
};

namespace functional {
/// Options for `torch::nn::functional::pairwise_distance`.
///
/// See the documentation for `torch::nn::PairwiseDistanceOptions` class to
/// learn what arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::pairwise_distance(input1, input2, F::PairwiseDistanceFuncOptions().p(1));
/// ```py
using PairwiseDistanceFuncOptions = PairwiseDistanceOptions;  // 使用 PairwiseDistanceOptions 的别名
} // namespace functional

} // namespace nn
} // namespace torch
```