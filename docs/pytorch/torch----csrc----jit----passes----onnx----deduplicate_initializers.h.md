# `.\pytorch\torch\csrc\jit\passes\onnx\deduplicate_initializers.h`

```
#pragma once
// 使用指令指示编译器只包含该文件一次，避免多重包含的问题

#include <memory>
// 包含内存管理相关的头文件

#include <torch/csrc/jit/ir/ir.h>
// 包含 Torch 的 JIT（即时编译）模块中的 IR（中间表示）相关头文件

namespace torch {
namespace jit {

void DeduplicateInitializers(
    std::shared_ptr<Graph>& g,
    // 使用智能指针管理 Graph 对象的共享所有权
    std::map<std::string, IValue>& paramsDict,
    // 一个映射，将字符串键映射到 Torch 的 IValue 对象
    bool is_train);
    // 布尔参数，指示当前是否处于训练模式

} // namespace jit
} // namespace torch
// 命名空间声明，将 DeduplicateInitializers 函数置于 torch::jit 命名空间中
```