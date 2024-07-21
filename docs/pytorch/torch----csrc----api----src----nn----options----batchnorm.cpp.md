# `.\pytorch\torch\csrc\api\src\nn\options\batchnorm.cpp`

```
#include <torch/nn/options/batchnorm.h>

# 包含 Torch 的 BatchNorm 头文件


namespace torch {
namespace nn {

# 定义命名空间 torch::nn


BatchNormOptions::BatchNormOptions(int64_t num_features)
    : num_features_(num_features) {}

# BatchNormOptions 类的构造函数实现，接收 num_features 参数并初始化成员变量 num_features_


} // namespace nn
} // namespace torch

# 命名空间闭合，结束 torch::nn 命名空间定义
```