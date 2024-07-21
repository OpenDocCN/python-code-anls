# `.\pytorch\torch\csrc\api\src\nn\options\adaptive.cpp`

```
#include <torch/nn/options/adaptive.h>


// 包含 Torch 库中的 adaptive.h 头文件

namespace torch {
namespace nn {

AdaptiveLogSoftmaxWithLossOptions::AdaptiveLogSoftmaxWithLossOptions(
    int64_t in_features,
    int64_t n_classes,
    std::vector<int64_t> cutoffs)
    : in_features_(in_features),
      n_classes_(n_classes),
      cutoffs_(std::move(cutoffs)) {}

// 定义 AdaptiveLogSoftmaxWithLossOptions 构造函数，初始化成员变量 in_features_, n_classes_, cutoffs_

} // namespace nn
} // namespace torch


// 定义命名空间 nn 下的 torch 命名空间，以及其中的 AdaptiveLogSoftmaxWithLossOptions 类和构造函数
```