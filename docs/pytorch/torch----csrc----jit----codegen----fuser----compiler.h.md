# `.\pytorch\torch\csrc\jit\codegen\fuser\compiler.h`

```py
#pragma once

#include <ATen/core/stack.h>  // 引入 ATen 库中的 stack 头文件
#include <torch/csrc/Export.h>  // 引入 torch 的导出定义头文件
#include <torch/csrc/jit/codegen/fuser/arg_spec.h>  // 引入融合器参数规范头文件
#include <torch/csrc/jit/codegen/fuser/fused_kernel.h>  // 引入融合内核头文件
#include <torch/csrc/jit/codegen/fuser/interface.h>  // 引入融合接口头文件
#include <torch/csrc/jit/codegen/fuser/kernel_spec.h>  // 引入融合内核规范头文件
#include <torch/csrc/jit/ir/ir.h>  // 引入 JIT IR 头文件

#include <cstdint>  // 引入标准整数类型定义头文件
#include <vector>  // 引入向量容器头文件

namespace torch {
namespace jit {
namespace fuser {

// 执行与设备无关的“预先”编译给定的 fusion_group，如果尚未注册
// 返回一个可以稍后运行融合操作的键
TORCH_API int64_t registerFusion(const Node* fusion_group);

// 使用 ArgSpec 中指定的运行时参数，执行设备特定的“运行时”编译给定的内核
// 输出使用指定设备上的 map_size 分配
TORCH_API std::shared_ptr<FusedKernel> compileKernel(
    const KernelSpec& spec,
    const ArgSpec& arg_spec,
    const std::vector<int64_t>& map_size,
    const at::Device device);

// 返回已编译的内核数量
TORCH_API size_t nCompiledKernels();

// 调试融合器
TORCH_API int debugFuser();

// 定义一个函数类型 FusedKernelConstructor，用于构造 FusedKernel 的共享指针
using FusedKernelConstructor = std::function<std::shared_ptr<FusedKernel>(
    int16_t device,
    std::string name,
    std::string code,
    std::vector<TensorDesc> input_desc,
    std::vector<TensorDesc> output_desc,
    std::vector<PartitionDesc> chunk_desc,
    std::vector<PartitionDesc> concat_desc,
    bool has_random)>;

// 注册一个融合后端，使用指定的 backend_type 和构造器 ctor
TORCH_API void registerFusionBackend(
    at::Device::Type backend_type,
    FusedKernelConstructor ctor);

// 检查是否存在指定的融合后端
TORCH_API bool hasFusionBackend(at::Device::Type backend_type);

// 用于注册融合后端的结构体，构造时自动调用 registerFusionBackend
struct TORCH_API RegisterFusionBackend {
  RegisterFusionBackend(
      at::Device::Type backend_type,
      FusedKernelConstructor ctor) {
    registerFusionBackend(backend_type, std::move(ctor));
  }
};

} // namespace fuser
} // namespace jit
} // namespace torch
```