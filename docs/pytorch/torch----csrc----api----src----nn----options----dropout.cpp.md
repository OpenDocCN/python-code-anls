# `.\pytorch\torch\csrc\api\src\nn\options\dropout.cpp`

```
#include <torch/nn/options/dropout.h>


// 包含头文件 <torch/nn/options/dropout.h>，用于引入 Dropout 相关选项

namespace torch {
namespace nn {

// 进入 torch 命名空间下的 nn 命名空间

DropoutOptions::DropoutOptions(double p) : p_(p) {}

// DropoutOptions 类的构造函数定义，接受一个 double 类型参数 p，初始化成员变量 p_

} // namespace nn
} // namespace torch

// 退出 nn 命名空间和 torch 命名空间
```