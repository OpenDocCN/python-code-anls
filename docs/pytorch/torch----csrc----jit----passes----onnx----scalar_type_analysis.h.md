# `.\pytorch\torch\csrc\jit\passes\onnx\scalar_type_analysis.h`

```
#pragma once

# 预处理指令，确保本头文件只被编译一次，用于防止多重包含。


#include <torch/csrc/jit/ir/ir.h>

# 包含 Torch 库中的 IR 相关头文件，用于处理计算图的中间表示。


namespace torch {
namespace jit {

# 进入 Torch 的 jit 命名空间，用于包裹定义的函数和类，避免命名冲突。


TORCH_API void ScalarTypeAnalysisForONNX(
    const std::shared_ptr<Graph>& graph,
    bool lowprecision_cast,
    int opset_version);

# 声明名为 `ScalarTypeAnalysisForONNX` 的函数，用于执行对 ONNX 格式的图进行标量类型分析。接受一个图的共享指针 `graph`，一个指示是否进行低精度转换的布尔值 `lowprecision_cast`，以及操作集版本号 `opset_version`。


void ScalarTypeAnalysisNodeForONNX(Node* n);

# 声明名为 `ScalarTypeAnalysisNodeForONNX` 的函数，用于执行对 ONNX 节点进行标量类型分析，接受一个节点指针 `n`。


} // namespace jit
} // namespace torch

# 结束 Torch 的 jit 命名空间。
```