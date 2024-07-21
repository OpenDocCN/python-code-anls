# `.\pytorch\torch\csrc\api\src\nn\options\instancenorm.cpp`

```py
#include <torch/nn/options/instancenorm.h>
// 引入实例归一化选项的头文件

namespace torch {
namespace nn {
// 定义命名空间 torch 和 nn

InstanceNormOptions::InstanceNormOptions(int64_t num_features)
    : num_features_(num_features) {}
// 实现 InstanceNormOptions 类的构造函数，初始化 num_features_ 成员变量为给定的 num_features 值

} // namespace nn
} // namespace torch
// 结束命名空间 torch 和 nn
```