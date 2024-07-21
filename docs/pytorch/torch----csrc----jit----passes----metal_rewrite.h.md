# `.\pytorch\torch\csrc\jit\passes\metal_rewrite.h`

```
#pragma once
#include <torch/csrc/jit/api/module.h>
// 引入包含了 Torch JIT API 的模块头文件

#include <torch/csrc/jit/ir/ir.h>
// 引入了 Torch JIT IR 相关的头文件

#include <string>
// 引入了处理字符串的标准库

#include <vector>
// 引入了处理向量的标准库

namespace torch {
namespace jit {
// Torch JIT 命名空间开始

TORCH_API void metalInsertPrePackedOps(std::shared_ptr<Graph>& graph);
// 定义了一个函数原型，用于在图中插入预打包操作

TORCH_API void metalInsertPrePackedOps(script::Module& module);
// 定义了一个函数原型，用于在模块中插入预打包操作

TORCH_API void metalFusePrePackedConvWithClamp(script::Module& module);
// 定义了一个函数原型，用于融合预打包卷积与截断操作

TORCH_API void metalFoldPrePackingOps(script::Module& module);
// 定义了一个函数原型，用于折叠预打包操作

TORCH_API script::Module metalOptimizeForMobile(
    const script::Module& module,
    const std::vector<std::string>& preserved_methods);
// 定义了一个函数原型，用于为移动设备优化模块，保留指定方法

} // namespace jit
} // namespace torch
// Torch JIT 命名空间结束
```