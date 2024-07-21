# `.\pytorch\torch\csrc\jit\passes\peephole_non_tensor.h`

```py
#pragma once


// 使用 #pragma once 预处理指令，确保头文件只被编译一次，提升编译效率


#include <torch/csrc/jit/ir/ir.h>


// 包含 torch 库中的头文件 ir.h，用于引入与 JIT 图 IR 相关的定义和功能


namespace torch {
namespace jit {


// 定义命名空间 torch::jit，用于封装与 PyTorch JIT 相关的功能和类


// return true if graph is modified
// Optimizing General Graph Patterns that
// are not covered in peephole.cpp and peephole_list_idioms


// 函数声明：如果图被修改则返回 true
// 优化一般图模式，这些模式不在 peephole.cpp 和 peephole_list_idioms 中涵盖


TORCH_API bool PeepholeOptimizeNonTensor(const std::shared_ptr<Graph>& graph);


// 函数声明：PeepholeOptimizeNonTensor 函数原型，接受一个指向 Graph 的 shared_ptr 参数，
// 使用 TORCH_API 宏指定函数的可见性和导出规则，返回一个布尔值，表示图是否被修改


} // namespace jit
} // namespace torch


// 结束命名空间定义，确保所有的 torch::jit 相关功能都在 torch 命名空间下
```