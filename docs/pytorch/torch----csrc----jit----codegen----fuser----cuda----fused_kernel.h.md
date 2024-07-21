# `.\pytorch\torch\csrc\jit\codegen\fuser\cuda\fused_kernel.h`

```
#pragma once
// 预处理命令，确保头文件只被包含一次

#include <ATen/ATen.h>
// 包含 ATen 库，用于张量操作

#include <torch/csrc/Export.h>
// 包含 Torch 导出相关的宏定义

#include <torch/csrc/jit/codegen/fuser/fused_kernel.h>
// 包含 Torch JIT 中融合内核相关的头文件

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
// 包含 CUDA 和 NVRTC 相关的头文件

#include <cstdint>
// 包含 C++ 标准中的整数类型

#include <string>
#include <vector>
// 包含 C++ 标准中的字符串和向量容器

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

// 查询代码生成的目标架构和目标设备
TORCH_CUDA_CU_API void codegenOutputQuery(
    const cudaDeviceProp* const prop,
    int& major,
    int& minor,
    bool& compile_to_sass);
// 声明一个函数，用于查询代码生成的输出目标架构和目标设备信息

// 表示一个实际 CUDA 函数的元数据的类
// 注意：CUDA 函数是按设备定义的
struct TORCH_CUDA_CU_API FusedKernelCUDA
    : public ::torch::jit::fuser::FusedKernel {
  FusedKernelCUDA(
      at::DeviceIndex device,
      std::string name,
      std::string code,
      std::vector<TensorDesc> input_desc,
      std::vector<TensorDesc> output_desc,
      std::vector<PartitionDesc> chunk_desc,
      std::vector<PartitionDesc> concat_desc,
      bool has_random);
  // 构造函数，用于初始化 FusedKernelCUDA 对象

  ~FusedKernelCUDA() override;
  // 虚析构函数，释放资源

  void launch_raw(const uint32_t numel, std::vector<void*>& arguments)
      const override;
  // 虚函数，用于执行 CUDA 内核的启动

  at::Backend backend() const override {
    return at::Backend::CUDA;
  }
  // 返回后端类型为 CUDA

 private:
  static constexpr auto kBlockSize = 128;
  // 静态成员常量，定义 CUDA 线程块大小

  // 注意：为每个设备存储设备属性和计算启动启发式信息
  //  在启动时获取这些值会导致性能下降
  at::DeviceIndex device_;  // 设备索引
  int maxBlocks_;           // 最大线程块数
  cudaDeviceProp* prop_;    // CUDA 设备属性指针
  std::vector<char> ptx_;   // PTX 代码存储向量
  CUmodule module_;         // CUDA 模块对象
  CUfunction function_;     // CUDA 函数对象
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
```