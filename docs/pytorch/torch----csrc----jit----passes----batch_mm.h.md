# `.\pytorch\torch\csrc\jit\passes\batch_mm.h`

```
#pragma once
// 使用#pragma once指令，确保头文件只被编译一次，防止多重包含的问题

#include <torch/csrc/jit/ir/ir.h>
// 包含torch的IR模块，用于处理图形表达和优化

namespace torch {
namespace jit {

TORCH_API void BatchMM(std::shared_ptr<Graph>& graph);
// 声明一个名为BatchMM的函数，接受一个Graph类型的智能指针作为参数，并且使用TORCH_API进行声明，以便在动态链接库中正确导出和导入

} // namespace jit
} // namespace torch
// 命名空间声明，将函数BatchMM放置在torch::jit命名空间下，确保函数名不与其他命名空间下的函数冲突
```