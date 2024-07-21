# `.\pytorch\torch\csrc\jit\passes\add_if_then_else.h`

```
#pragma once


// 使用#pragma once指令确保头文件只被编译一次，防止多重包含问题


#include <torch/csrc/jit/ir/ir.h>


// 包含torch库中的ir.h头文件，该文件定义了IR图的相关结构和操作


namespace torch {
namespace jit {


// 命名空间torch::jit，用于封装torch库中的即时编译（JIT）相关功能


TORCH_API bool AddIfThenElseOp(std::shared_ptr<Graph>& graph);


// 声明一个函数AddIfThenElseOp，该函数接受一个Graph对象的共享指针作为参数，并返回一个bool值
// TORCH_API是一个宏，用于声明这个函数是在torch库中被导出的API接口


} // namespace jit
} // namespace torch


// 结束命名空间jit和torch的定义
```