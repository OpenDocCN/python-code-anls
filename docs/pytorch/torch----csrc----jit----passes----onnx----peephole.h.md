# `.\pytorch\torch\csrc\jit\passes\onnx\peephole.h`

```py
#pragma once
// 预处理指令，表示只包含一次此头文件

#include <torch/csrc/jit/ir/ir.h>
// 包含 Torch 库中的 IR 相关头文件

namespace torch {
namespace jit {

void PeepholeOptimizeONNX(
    std::shared_ptr<Graph>& graph,
    int opset_version,
    bool fixed_batch_size);
// 声明 PeepholeOptimizeONNX 函数，接受一个图的共享指针，操作集版本号和固定批处理大小作为参数

} // namespace jit
} // namespace torch
// 命名空间声明，将所有 Torch JIT 相关的内容放入 torch::jit 命名空间
```