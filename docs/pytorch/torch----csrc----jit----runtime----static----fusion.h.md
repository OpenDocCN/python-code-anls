# `.\pytorch\torch\csrc\jit\runtime\static\fusion.h`

```py
#pragma once
// 使用预处理命令#pragma once，确保此头文件只被编译一次

#include <torch/csrc/jit/ir/ir.h>
// 包含 Torch 的 IR 头文件，用于处理图形表示和优化

namespace torch::jit {

TORCH_API void fuseStaticSubgraphs(
    std::shared_ptr<Graph> graph,
    size_t min_size);
// 定义了一个 Torch API 函数fuseStaticSubgraphs，用于融合静态子图

TORCH_API void performTensorExprFusion(
    std::shared_ptr<Graph> graph,
    std::vector<IValue> sample_inputs);
// 定义了一个 Torch API 函数performTensorExprFusion，用于执行张量表达式的融合

} // namespace torch::jit
// 结束了 torch::jit 命名空间
```