# `.\pytorch\c10\cuda\CUDAException.h`

```py
#pragma once

#include <c10/cuda/CUDADeviceAssertionHost.h>
#include <c10/cuda/CUDAMacros.h>
#include <c10/cuda/CUDAMiscFunctions.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>
#include <cuda.h>

// Note [CHECK macro]
// ~~~~~~~~~~~~~~~~~~
// This is a macro so that AT_ERROR can get accurate __LINE__
// and __FILE__ information.  We could split this into a short
// macro and a function implementation if we pass along __LINE__
// and __FILE__, but no one has found this worth doing.

// Used to denote errors from CUDA framework.
// This needs to be declared here instead util/Exception.h for proper conversion
// during hipify.
namespace c10 {
// Define a custom exception class CUDAError inheriting from c10::Error
class C10_CUDA_API CUDAError : public c10::Error {
  using Error::Error; // Inherit constructors from c10::Error
};
} // namespace c10

// Macro for checking CUDA expressions for errors
#define C10_CUDA_CHECK(EXPR)                                        \
  do {                                                              \
    const cudaError_t __err = EXPR;                                 \
    // Call a function to handle CUDA error checking and reporting    \
    c10::cuda::c10_cuda_check_implementation(                       \
        static_cast<int32_t>(__err),                                \
        __FILE__,                                                   \
        __func__, /* Line number data type not well-defined between \
                      compilers, so we perform an explicit cast */  \
        static_cast<uint32_t>(__LINE__),                            \
        true);                                                      \
  } while (0)

// Macro for warning about CUDA errors
#define C10_CUDA_CHECK_WARN(EXPR)                              \
  do {                                                         \
    const cudaError_t __err = EXPR;                            \
    // Check if the CUDA expression resulted in an error        \
    if (C10_UNLIKELY(__err != cudaSuccess)) {                  \
      auto error_unused C10_UNUSED = cudaGetLastError();       \
      (void)error_unused;                                      \
      // Issue a warning about the CUDA error                   \
      TORCH_WARN("CUDA warning: ", cudaGetErrorString(__err)); \
    }                                                          \
  } while (0)

// Macro indicating a CUDA error is handled in a non-standard way
#define C10_CUDA_ERROR_HANDLED(EXPR) EXPR

// Macro to intentionally ignore a CUDA error
#define C10_CUDA_IGNORE_ERROR(EXPR)                             \
  do {                                                          \
    const cudaError_t __err = EXPR;                             \
    // Check if the CUDA expression resulted in an error        \
    if (C10_UNLIKELY(__err != cudaSuccess)) {                   \
      // Clear the last CUDA error to ignore it                \
      cudaError_t error_unused C10_UNUSED = cudaGetLastError(); \
      (void)error_unused;                                       \
    }                                                           \
  } while (0)

// Macro to clear the last CUDA error
#define C10_CUDA_CLEAR_ERROR()                                \
  do {                                                        \
    // Clear the last CUDA error                              \
    cudaError_t error_unused C10_UNUSED = cudaGetLastError(); \
    (void)error_unused;                                       \
  } while (0)
/// 确保每次 CUDA 核心启动后立即使用，以确保启动正确，并在出现问题时提供接近源代码的早期诊断。
#define C10_CUDA_KERNEL_LAUNCH_CHECK() C10_CUDA_CHECK(cudaGetLastError())

/// 启动 CUDA 核心并附加处理设备端断言失败所需的所有信息。检查启动是否成功。
#define TORCH_DSA_KERNEL_LAUNCH(                                      \
    kernel, blocks, threads, shared_mem, stream, ...)                 \
  do {                                                                \
    auto& launch_registry =                                           \
        c10::cuda::CUDAKernelLaunchRegistry::get_singleton_ref();     \
    kernel<<<blocks, threads, shared_mem, stream>>>(                  \
        __VA_ARGS__,                                                  \
        launch_registry.get_uvm_assertions_ptr_for_current_device(),  \
        launch_registry.insert(                                       \
            __FILE__, __FUNCTION__, __LINE__, #kernel, stream.id())); \
    C10_CUDA_KERNEL_LAUNCH_CHECK();                                   \
  } while (0)

namespace c10::cuda {

/// 在 CUDA 失败时，格式化关于该失败的详细错误消息，并检查设备端断言是否失败。
C10_CUDA_API void c10_cuda_check_implementation(
    const int32_t err,
    const char* filename,
    const char* function_name,
    const int line_number,
    const bool include_device_assertions);

} // namespace c10::cuda
```