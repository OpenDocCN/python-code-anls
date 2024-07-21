# `.\pytorch\torch\csrc\jit\passes\remove_redundant_profiles.h`

```py
#pragma once
// 使用#pragma once指令确保头文件只被编译一次

#include <torch/csrc/jit/ir/ir.h>
// 包含torch库中的IR相关头文件

namespace torch {
namespace jit {

TORCH_API void RemoveRedundantProfiles(std::shared_ptr<Graph>& graph);
// 定义函数RemoveRedundantProfiles，接受一个Graph对象的智能指针作为参数，并且使用了TORCH_API宏进行标记

TORCH_API void RemoveRedundantProfiles(Block* block, AliasDb& db);
// 定义函数RemoveRedundantProfiles，接受一个Block指针和AliasDb引用作为参数，并且使用了TORCH_API宏进行标记

} // namespace jit
} // namespace torch
// 命名空间jit和torch包含了与JIT相关的函数和类声明
```