# `.\pytorch\torch\csrc\jit\passes\vulkan_rewrite.h`

```
#pragma once


// 指令，告诉预处理器如果这个文件已经被包含就不要再次包含
#include <torch/csrc/jit/api/module.h>
// 包含 Torch C++ API 模块头文件
#include <torch/csrc/jit/ir/ir.h>
// 包含 Torch JIT 中间表示的头文件
#include <torch/csrc/jit/passes/mobile_optimizer_type.h>
// 包含 Torch 移动优化器类型的头文件

namespace torch {
namespace jit {
// Torch JIT 命名空间开始

// 声明一个函数，用于在图中插入 Vulkan 专用的预打包操作
TORCH_API void vulkanInsertPrePackedOps(std::shared_ptr<Graph>& graph);
// 声明一个函数，用于在模块中插入 Vulkan 专用的预打包操作
TORCH_API void vulkanInsertPrePackedOps(script::Module& module);
// 声明一个函数，用于融合模块中的 Vulkan 专用的预打包 Conv 和 Clamp 操作
TORCH_API void vulkanFusePrePackedConvWithClamp(script::Module& module);
// 声明一个函数，用于折叠模块中的预打包操作
TORCH_API void vulkanFoldPrePackingOps(script::Module& module);
// 声明一个函数，用于优化模块以适应移动设备的需求
TORCH_API script::Module vulkanOptimizeForMobile(
    const script::Module& module,
    const std::set<MobileOptimizerType>& optimization_blocklist,
    const std::vector<std::string>& preserved_methods);
// Torch JIT 命名空间结束
} // namespace jit
} // namespace torch
```