# `.\pytorch\torch\csrc\jit\passes\create_functional_graphs.h`

```py
#pragma once
// 使用预处理指令#pragma once，确保头文件只被编译一次

#include <torch/csrc/Export.h>
// 包含torch库中的Export.h头文件

#include <torch/csrc/jit/ir/ir.h>
// 包含torch库中jit命名空间下的ir.h头文件，用于IR表示的相关功能

namespace torch {
namespace jit {

TORCH_API void CreateFunctionalGraphs(const std::shared_ptr<Graph>& graph);
// 声明了一个函数CreateFunctionalGraphs，接受一个指向Graph对象的shared_ptr参数
// 这个函数用于创建函数式图形

TORCH_API void InlineFunctionalGraphs(const std::shared_ptr<Graph>& graph);
// 声明了一个函数InlineFunctionalGraphs，接受一个指向Graph对象的shared_ptr参数
// 这个函数用于内联函数式图形

} // namespace jit
} // namespace torch
// torch命名空间下的jit子命名空间中声明了两个函数，用于创建和内联函数式图形
```