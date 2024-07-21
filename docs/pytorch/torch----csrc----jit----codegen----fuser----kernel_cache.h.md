# `.\pytorch\torch\csrc\jit\codegen\fuser\kernel_cache.h`

```py
#pragma once
// 预处理指令，确保头文件只被编译一次

#include <c10/util/Optional.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/jit/codegen/fuser/kernel_spec.h>
#include <torch/csrc/jit/ir/ir.h>

#include <cstdint>
#include <functional>

namespace torch {
namespace jit {
namespace fuser {

// 声明命名空间 torch::jit::fuser

// 线程安全的缓存接口。

// 规范化图形，通过规范化和擦除形状信息
TORCH_API std::shared_ptr<Graph> normalizeGraphForCache(
    const std::shared_ptr<Graph>& graph);

// 存储给定的图形，返回用于访问它的键
TORCH_API int64_t store(std::shared_ptr<Graph> graph);

// 给定图形，找到基于它的 KernelSpec
TORCH_API at::optional<KernelSpec*> lookupGraph(std::shared_ptr<Graph> graph);

// 返回与给定键对应的图形（如果存在）
TORCH_API at::optional<KernelSpec*> retrieve(const int64_t key);

// 返回融合键 -> KernelSpec 缓存的大小。
// 仅用于测试。
TORCH_API int64_t debugNumCachedKernelSpecs();

} // namespace fuser
} // namespace jit
} // namespace torch
```