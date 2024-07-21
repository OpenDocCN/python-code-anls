# `.\pytorch\torch\csrc\jit\tensorexpr\operators\softmax.h`

```
#pragma once
// 防止头文件被重复包含

#include <torch/csrc/jit/tensorexpr/kernel.h>
// 包含头文件 kernel.h

namespace torch {
namespace jit {
namespace tensorexpr {
// 命名空间 torch::jit::tensorexpr

Tensor computeSoftmax(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    bool log_softmax);
// 声明函数 computeSoftmax，接受输入参数、输出形状、输出步长和是否进行 log_softmax

} // namespace tensorexpr
} // namespace jit
} // namespace torch
// 命名空间结束
```