# `.\pytorch\torch\csrc\api\src\nn\options\normalization.cpp`

```py
// 包含头文件 <torch/nn/options/normalization.h>

// 定义 torch 命名空间
namespace torch {
// 定义 nn 命名空间
namespace nn {

// 实现 LayerNormOptions 构造函数，接受 normalized_shape 参数
LayerNormOptions::LayerNormOptions(std::vector<int64_t> normalized_shape)
    : normalized_shape_(std::move(normalized_shape)) {}

// 实现 CrossMapLRN2dOptions 构造函数，接受 size 参数
CrossMapLRN2dOptions::CrossMapLRN2dOptions(int64_t size) : size_(size) {}

// 实现 GroupNormOptions 构造函数，接受 num_groups 和 num_channels 参数
GroupNormOptions::GroupNormOptions(int64_t num_groups, int64_t num_channels)
    : num_groups_(num_groups), num_channels_(num_channels) {}

// 进入 functional 命名空间
namespace functional {

// 实现 LayerNormFuncOptions 构造函数，接受 normalized_shape 参数
LayerNormFuncOptions::LayerNormFuncOptions(
    std::vector<int64_t> normalized_shape)
    : normalized_shape_(std::move(normalized_shape)) {}

// 实现 GroupNormFuncOptions 构造函数，接受 num_groups 参数
GroupNormFuncOptions::GroupNormFuncOptions(int64_t num_groups)
    : num_groups_(num_groups) {}

} // namespace functional

} // namespace nn
} // namespace torch
```