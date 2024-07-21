# `.\pytorch\torch\csrc\api\include\torch\xpu.h`

```py
#pragma once
// 预处理指令，确保头文件只被包含一次

#include <torch/csrc/Export.h>
// 包含 Torch 的导出宏头文件

#include <cstddef>
#include <cstdint>
// 包含标准 C++ 头文件，用于定义大小和整数类型

namespace torch::xpu {

/// Returns the number of XPU devices available.
// 返回可用 XPU 设备的数量
size_t TORCH_API device_count();

/// Returns true if at least one XPU device is available.
// 如果至少有一个 XPU 设备可用，则返回 true
bool TORCH_API is_available();

/// Sets the seed for the current GPU.
// 设置当前 GPU 的随机种子
void TORCH_API manual_seed(uint64_t seed);

/// Sets the seed for all available GPUs.
// 设置所有可用 GPU 的随机种子
void TORCH_API manual_seed_all(uint64_t seed);

/// Waits for all kernels in all streams on a XPU device to complete.
// 等待 XPU 设备上所有流中的所有内核完成
void TORCH_API synchronize(int64_t device_index);

} // namespace torch::xpu
// 结束 torch::xpu 命名空间
```