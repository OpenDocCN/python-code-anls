# `.\pytorch\torch\csrc\inductor\aoti_runtime\device_utils.h`

```
#pragma once

// WARNING: Be careful when adding new includes here. This header will be used
// in model.so, and should not refer to any aten/c10 headers except the stable
// C ABI defined in torch/csrc/inductor/aoti_torch/c/shim.h. The same rule
// applies to other files under torch/csrc/inductor/aoti_runtime/.

#ifdef USE_CUDA

// FIXME: Currently, CPU and CUDA backend are mutually exclusive.
// This is a temporary workaround. We need a better way to support
// multi devices.

// 包含 CUDA 相关的头文件，用于 GPU 加速计算
#include <cuda.h>
#include <cuda_runtime_api.h>

// 定义一个宏 AOTI_RUNTIME_DEVICE_CHECK，用于检查 CUDA 运行时错误并抛出异常
#define AOTI_RUNTIME_DEVICE_CHECK(EXPR)                    \
  do {                                                     \
    const cudaError_t code = EXPR;                         \
    const char* msg = cudaGetErrorString(code);            \
    if (code != cudaSuccess) {                             \
      throw std::runtime_error(                            \
          std::string("CUDA error: ") + std::string(msg)); \
    }                                                      \
  } while (0)

namespace torch {
namespace aot_inductor {

// 使用 CUDA 流作为设备流类型
using DeviceStreamType = cudaStream_t;

} // namespace aot_inductor
} // namespace torch

#else // !USE_CUDA

// 如果没有定义 USE_CUDA，则定义一个宏 AOTI_RUNTIME_DEVICE_CHECK，用于检查 CPU 运行时错误并抛出异常
#define AOTI_RUNTIME_DEVICE_CHECK(EXPR)            \
  bool ok = EXPR;                                  \
  if (!ok) {                                       \
    throw std::runtime_error("CPU runtime error"); \
  }

namespace torch {
namespace aot_inductor {

// 如果不使用 CUDA，则设备流类型为空指针
using DeviceStreamType = void*;

} // namespace aot_inductor
} // namespace torch

#endif // USE_CUDA
```