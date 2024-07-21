# `.\pytorch\torch\csrc\api\src\nn\options\pooling.cpp`

```
// 包含了 Torch 深度学习库中的池化操作的选项头文件
#include <torch/nn/options/pooling.h>

// 定义了 Torch 深度学习库的命名空间 nn
namespace torch {
namespace nn {

// 实例化了 AvgPoolOptions 模板结构体，针对 1 维、2 维和 3 维情况
template struct AvgPoolOptions<1>;
template struct AvgPoolOptions<2>;
template struct AvgPoolOptions<3>;

// 实例化了 MaxPoolOptions 模板结构体，针对 1 维、2 维和 3 维情况
template struct MaxPoolOptions<1>;
template struct MaxPoolOptions<2>;
template struct MaxPoolOptions<3>;

// 实例化了 AdaptiveMaxPoolOptions 模板结构体，针对不同维度的可变尺寸情况
template struct AdaptiveMaxPoolOptions<ExpandingArray<1>>;
template struct AdaptiveMaxPoolOptions<ExpandingArrayWithOptionalElem<2>>;
template struct AdaptiveMaxPoolOptions<ExpandingArrayWithOptionalElem<3>>;

// 实例化了 AdaptiveAvgPoolOptions 模板结构体，针对不同维度的可变尺寸情况
template struct AdaptiveAvgPoolOptions<ExpandingArray<1>>;
template struct AdaptiveAvgPoolOptions<ExpandingArrayWithOptionalElem<2>>;
template struct AdaptiveAvgPoolOptions<ExpandingArrayWithOptionalElem<3>>;

// 实例化了 MaxUnpoolOptions 模板结构体，针对 1 维、2 维和 3 维情况
template struct MaxUnpoolOptions<1>;
template struct MaxUnpoolOptions<2>;
template struct MaxUnpoolOptions<3>;

// 实例化了 LPPoolOptions 模板结构体，针对 1 维、2 维和 3 维情况
template struct LPPoolOptions<1>;
template struct LPPoolOptions<2>;
template struct LPPoolOptions<3>;

} // namespace nn
} // namespace torch
```