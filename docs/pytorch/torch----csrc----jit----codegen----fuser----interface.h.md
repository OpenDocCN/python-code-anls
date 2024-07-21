# `.\pytorch\torch\csrc\jit\codegen\fuser\interface.h`

```py
// 防止头文件重复包含，只在第一次引入时有效
#pragma once

// 包含 ATen 库的主头文件，提供张量操作支持
#include <ATen/ATen.h>

// 包含 ATen 库的核心堆栈支持
#include <ATen/core/stack.h>

// 包含 Torch 的导出宏定义
#include <torch/csrc/Export.h>

// 包含 Torch JIT 的中间表示（IR）的头文件
#include <torch/csrc/jit/ir/ir.h>

// 包含 C++ 标准整数类型支持
#include <cstdint>

// 包含 C++ 标准内存管理支持
#include <memory>

// 包含 C++ 标准向量容器支持
#include <vector>

// Torch 命名空间，包含 JIT 子命名空间
namespace torch {
namespace jit {

// 定义常量，表示 CPU 设备编号
constexpr int kCPUDevice = -1;

// 注册融合（fusion）组节点，返回可用于后续运行融合的键值
TORCH_API int64_t registerFusion(const Node* fusion_group);

// 运行具有给定键值的融合操作，将输入和输出处理在堆栈上
TORCH_API void runFusion(const int64_t key, Stack& stack);

// 返回是否可以在 CPU 上进行融合操作
TORCH_API bool canFuseOnCPU();

// 返回是否可以在 GPU 上进行融合操作
TORCH_API bool canFuseOnGPU();

// 设置是否允许在 CPU 上进行融合操作（默认情况下由于不稳定而禁用）
TORCH_API void overrideCanFuseOnCPU(bool value);

// 设置在 CPU 上进行融合操作时必须使用 LLVM Codegen 而不是 SimplieIREval
TORCH_API void overrideMustUseLLVMOnCPU(bool value);

// 设置是否允许在 GPU 上进行融合操作（默认情况下启用）
TORCH_API void overrideCanFuseOnGPU(bool value);

// 将给定图形视为融合组并在指定设备上启动它，使用给定的输入，返回输出张量数组
TORCH_API std::vector<at::Tensor> debugLaunchGraph(
    Graph& graph,
    at::ArrayRef<at::Tensor> inputs);

// 将给定图形视为融合组并返回生成的内核代码
TORCH_API std::string debugGetFusedKernelCode(
    Graph& graph,
    at::ArrayRef<at::Tensor> inputs);

// 返回已编译内核数量
TORCH_API size_t nCompiledKernels();

} // namespace jit
} // namespace torch
```