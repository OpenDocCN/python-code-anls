# `.\pytorch\torch\csrc\jit\passes\frozen_graph_optimizations.h`

```py
#pragma once
// 使用预处理命令#pragma once，确保头文件只被编译一次

#include <torch/csrc/jit/ir/ir.h>
// 包含 Torch 的 JIT 模块中的 IR 头文件

/** \brief 运行一组优化，用于优化冻结图
 *
 * 当前这组优化包括：
 * - FoldFrozenConvBatchnorm （折叠冻结的卷积批归一化）
 * - FoldFrozenConvAddOrSub （折叠冻结的卷积加法或减法）
 * - FoldFrozenConvMulOrDiv （折叠冻结的卷积乘法或除法）
 * - FoldFrozenLinearBatchnorm （折叠冻结的线性层批归一化）
 */

namespace torch {
namespace jit {

// 使用 TORCH_API 定义一个公共函数 OptimizeFrozenGraph，该函数会修改传入的图对象
TORCH_API void OptimizeFrozenGraph(
    std::shared_ptr<Graph>& graph,  // 传入的参数为一个图的共享指针，将会被修改
    bool optimize_numerics = true);  // 是否优化数值计算，默认为真

} // namespace jit
} // namespace torch
```