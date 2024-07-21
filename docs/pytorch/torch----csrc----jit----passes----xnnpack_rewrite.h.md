# `.\pytorch\torch\csrc\jit\passes\xnnpack_rewrite.h`

```py
#pragma once
// 预处理指令，确保头文件只被编译一次

#include <torch/csrc/jit/api/module.h>
// 包含 Torch 的模块 API 头文件

#include <torch/csrc/jit/ir/ir.h>
// 包含 Torch 的 IR（Intermediate Representation，中间表示）头文件

#include <torch/csrc/jit/passes/mobile_optimizer_type.h>
// 包含 Torch 移动优化器类型定义的头文件

namespace torch {
namespace jit {

TORCH_API void transformConv1dToConv2d(std::shared_ptr<Graph>& graph);
// 将 Conv1d 转换为 Conv2d 的函数声明，操作基于图形对象的共享指针

TORCH_API void transformConv1dToConv2d(script::Module& module);
// 将 Conv1d 转换为 Conv2d 的函数声明，操作基于脚本模块对象

TORCH_API void insertPrePackedOps(std::shared_ptr<Graph>& graph);
// 在图形对象中插入预打包操作的函数声明

TORCH_API void insertPrePackedOps(script::Module& module);
// 在脚本模块对象中插入预打包操作的函数声明

TORCH_API void fusePrePackedLinearConvWithClamp(script::Module& module);
// 融合预打包的线性和卷积操作并包含 Clamp 的函数声明

TORCH_API void FoldPrePackingOps(script::Module& module);
// 折叠预打包操作的函数声明，操作基于脚本模块对象

TORCH_API script::Module optimizeForMobile(
    const script::Module& module,
    const std::set<MobileOptimizerType>& optimization_blocklist = {},
    const std::vector<std::string>& preserved_methods = {});
// 为移动设备优化模块的函数声明，可指定优化器类型黑名单和保留方法列表

} // namespace jit
} // namespace torch
```