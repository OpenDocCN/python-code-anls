# `.\pytorch\torch\csrc\api\include\torch\cuda.h`

```
#pragma once

// 使用 `#pragma once` 预处理指令，确保头文件只被编译一次，避免重复包含


#include <torch/csrc/Export.h>

// 包含 Torch 库的导出头文件 `Export.h`


#include <cstddef>
#include <cstdint>

// 包含标准库头文件 `<cstddef>` 和 `<cstdint>`，分别提供对 `std::size_t` 和 `std::uint64_t` 类型的定义


namespace torch {
namespace cuda {

// 进入命名空间 `torch`，再进入命名空间 `cuda`


/// Returns the number of CUDA devices available.
size_t TORCH_API device_count();

// 声明函数 `device_count()`，返回 CUDA 设备的数量，函数声明带有 `TORCH_API` 标记


/// Returns true if at least one CUDA device is available.
bool TORCH_API is_available();

// 声明函数 `is_available()`，返回是否至少有一个 CUDA 设备可用，函数声明带有 `TORCH_API` 标记


/// Returns true if CUDA is available, and CuDNN is available.
bool TORCH_API cudnn_is_available();

// 声明函数 `cudnn_is_available()`，返回 CUDA 和 CuDNN 是否都可用，函数声明带有 `TORCH_API` 标记


/// Sets the seed for the current GPU.
void TORCH_API manual_seed(uint64_t seed);

// 声明函数 `manual_seed(uint64_t seed)`，为当前 GPU 设置随机种子，函数声明带有 `TORCH_API` 标记


/// Sets the seed for all available GPUs.
void TORCH_API manual_seed_all(uint64_t seed);

// 声明函数 `manual_seed_all(uint64_t seed)`，为所有可用的 GPU 设置随机种子，函数声明带有 `TORCH_API` 标记


/// Waits for all kernels in all streams on a CUDA device to complete.
void TORCH_API synchronize(int64_t device_index = -1);

// 声明函数 `synchronize(int64_t device_index = -1)`，等待指定 CUDA 设备上所有流中的所有内核完成，函数声明带有 `TORCH_API` 标记


} // namespace cuda
} // namespace torch

// 结束命名空间 `cuda` 和命名空间 `torch`
```