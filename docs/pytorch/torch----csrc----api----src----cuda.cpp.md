# `.\pytorch\torch\csrc\api\src\cuda.cpp`

```py
/// 包含 CUDA 相关头文件

#include <torch/cuda.h>

/// 包含 ATen 的上下文相关头文件
#include <ATen/Context.h>

/// 包含设备保护工具
#include <c10/core/DeviceGuard.h>

/// 包含范围工具，用于迭代
#include <c10/util/irange.h>

/// 包含标准库的定义，例如 size_t
#include <cstddef>

/// 定义了 torch 命名空间
namespace torch {
/// 定义了 cuda 命名空间，用于 CUDA 相关操作
namespace cuda {

/// 返回当前系统中的 CUDA 设备数量
size_t device_count() {
  return at::detail::getCUDAHooks().getNumGPUs();
}

/// 检查当前系统是否支持 CUDA 并且至少有一个 GPU 可用
bool is_available() {
  // 注意：此处的语义与 at::globalContext().hasCUDA() 不同；
  // ATen 的函数告诉你是否有可用的驱动程序和 CUDA 构建，
  // 而这个函数还告诉你是否实际上有任何 GPU。
  // 此函数与 at::cuda::is_available() 的语义匹配
  return cuda::device_count() > 0;
}

/// 检查当前系统是否支持 CuDNN 并且至少有一个 GPU 可用
bool cudnn_is_available() {
  return is_available() && at::detail::getCUDAHooks().hasCuDNN();
}

/// 设置当前 GPU 的随机种子
void manual_seed(uint64_t seed) {
  if (is_available()) {
    auto index = at::detail::getCUDAHooks().current_device();
    auto gen = at::detail::getCUDAHooks().getDefaultCUDAGenerator(index);
    {
      // 查看注释 [使用随机生成器时获取锁]
      std::lock_guard<std::mutex> lock(gen.mutex());
      gen.set_current_seed(seed);
    }
  }
}

/// 设置所有可用 GPU 的随机种子
void manual_seed_all(uint64_t seed) {
  auto num_gpu = device_count();
  for (const auto i : c10::irange(num_gpu)) {
    auto gen = at::detail::getCUDAHooks().getDefaultCUDAGenerator(i);
    {
      // 查看注释 [使用随机生成器时获取锁]
      std::lock_guard<std::mutex> lock(gen.mutex());
      gen.set_current_seed(seed);
    }
  }
}

/// 同步指定的 CUDA 设备
void synchronize(int64_t device_index) {
  // 检查是否有可用的 CUDA GPU
  TORCH_CHECK(is_available(), "No CUDA GPUs are available");

  // 获取系统中的 CUDA 设备数量
  int64_t num_gpus = cuda::device_count();

  // 检查设备索引是否合法
  TORCH_CHECK(
      device_index == -1 || device_index < num_gpus,
      "Device index out of range: ",
      device_index);

  // 执行 CUDA 设备同步操作
  at::detail::getCUDAHooks().deviceSynchronize(device_index);
}

} // namespace cuda
} // namespace torch
```