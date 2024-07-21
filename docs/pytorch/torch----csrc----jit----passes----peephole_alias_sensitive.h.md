# `.\pytorch\torch\csrc\jit\passes\peephole_alias_sensitive.h`

```py
#pragma once
// 使用 `#pragma once` 预处理指令，确保头文件只被编译一次，防止多重包含的问题

#include <torch/csrc/jit/ir/ir.h>
// 包含 Torch 库中的 IR 头文件，用于处理 JIT 编译器的中间表示

namespace torch {
namespace jit {

// 基于别名敏感的Peephole优化
// 当前作为PeepholeOptimize的一部分调用
// 如果shape_peepholes为true，则在TensorType上进行优化
// 返回true表示图被修改了
// 如果shape_peepholes为true，则在TensorType上进行优化
TORCH_API bool PeepholeOptimizeAliasSensitive(
    const std::shared_ptr<Graph>& graph,
    bool shape_peepholes);
// 声明了一个函数PeepholeOptimizeAliasSensitive，接受一个共享指针指向Graph对象和一个布尔值作为参数
// 这个函数用于进行基于别名敏感的Peephole优化，返回一个布尔值表示是否修改了图

} // namespace jit
} // namespace torch
// 命名空间声明结束
```