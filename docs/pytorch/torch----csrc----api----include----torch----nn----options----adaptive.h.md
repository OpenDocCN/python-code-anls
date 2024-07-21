# `.\pytorch\torch\csrc\api\include\torch\nn\options\adaptive.h`

```py
#pragma once

#include <torch/arg.h>
#include <torch/csrc/Export.h>
#include <torch/types.h>

namespace torch {
namespace nn {

/// Options for the `AdaptiveLogSoftmaxWithLoss` module.
///
/// Example:
/// ```
/// AdaptiveLogSoftmaxWithLoss model(AdaptiveLogSoftmaxWithLossOptions(8, 10,
/// {4, 8}).div_value(2.).head_bias(true));
/// ```py
struct TORCH_API AdaptiveLogSoftmaxWithLossOptions {
  /* implicit */ AdaptiveLogSoftmaxWithLossOptions(
      int64_t in_features,            // 输入张量中的特征数
      int64_t n_classes,              // 数据集中的类数
      std::vector<int64_t> cutoffs);  // 用于将目标分配到其桶中的截断值

  /// Number of features in the input tensor
  TORCH_ARG(int64_t, in_features);   // 输入张量中的特征数

  /// Number of classes in the dataset
  TORCH_ARG(int64_t, n_classes);     // 数据集中的类数

  /// Cutoffs used to assign targets to their buckets
  TORCH_ARG(std::vector<int64_t>, cutoffs);  // 用于将目标分配到其桶中的截断值

  /// value used as an exponent to compute sizes of the clusters. Default: 4.0
  TORCH_ARG(double, div_value) = 4.;  // 用作指数来计算聚类的大小。默认值为 4.0

  /// If ``true``, adds a bias term to the 'head' of
  /// the adaptive softmax. Default: false
  TORCH_ARG(bool, head_bias) = false; // 如果为 true，则在自适应 softmax 的 'head' 中添加偏置项。默认值为 false
};

} // namespace nn
} // namespace torch
```