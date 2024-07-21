# `.\pytorch\torch\csrc\jit\codegen\fuser\cpu\fused_kernel.h`

```
// 防止头文件被多次包含
#pragma once

// 包含 ATen 库的头文件
#include <ATen/ATen.h>
// 包含 Torch 的导出头文件
#include <torch/csrc/Export.h>
// 包含融合内核的头文件
#include <torch/csrc/jit/codegen/fuser/fused_kernel.h>

// 包含 C++ 标准库头文件
#include <cstdint>
#include <memory>
#include <string>

// 前向声明 DynamicLibrary 结构体，位于 at 命名空间下
namespace at {
struct DynamicLibrary;
}

// Torch 命名空间
namespace torch {
// JIT 命名空间
namespace jit {
// 融合命名空间
namespace fuser {
// CPU 融合命名空间
namespace cpu {

// 表示编译后的 CPU 内核及运行所需的元数据
struct TORCH_API FusedKernelCPU : public FusedKernel {
  // 构造函数，初始化 CPU 融合内核对象
  FusedKernelCPU(
      std::string name,
      std::string code,
      std::vector<TensorDesc> input_desc,
      std::vector<TensorDesc> output_desc,
      std::vector<PartitionDesc> chunk_desc,
      std::vector<PartitionDesc> concat_desc,
      bool has_random);

  // 返回内核所属的后端类型为 CPU
  at::Backend backend() const override {
    return at::Backend::CPU;
  }

  // 启动原始内核函数，传递元素数量和参数列表
  void launch_raw(const uint32_t numel, std::vector<void*>& arguments)
      const override {
    kernel(numel, arguments.data());
  }

 private:
  // 动态库对象的独占指针
  std::unique_ptr<at::DynamicLibrary> so_lib;
  // 指向内核函数的函数指针，初始化为 nullptr
  void (*kernel)(uint32_t, void**) = nullptr;
};

} // namespace cpu
} // namespace fuser
} // namespace jit
} // namespace torch
```