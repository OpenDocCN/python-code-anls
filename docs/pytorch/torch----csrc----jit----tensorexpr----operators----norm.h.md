# `.\pytorch\torch\csrc\jit\tensorexpr\operators\norm.h`

```py
#pragma once

# 预处理指令：指示编译器只包含当前头文件一次


#include <torch/csrc/jit/tensorexpr/kernel.h>

# 包含头文件：引入Torch库中Tensor Expression的内核功能


namespace torch {
namespace jit {
namespace tensorexpr {

# 命名空间定义：进入Torch库、JIT编译器和Tensor Expression模块的命名空间


Tensor computeBatchNorm(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device);

# 函数声明：声明一个名为`computeBatchNorm`的函数，接受输入参数、输出形状、输出步长、输出类型和设备信息，并返回一个Tensor对象


} // namespace tensorexpr
} // namespace jit
} // namespace torch

# 命名空间结束：结束命名空间定义，分别为Tensor Expression、JIT编译器和Torch库
```