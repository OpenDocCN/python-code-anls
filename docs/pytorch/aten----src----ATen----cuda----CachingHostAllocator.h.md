# `.\pytorch\aten\src\ATen\cuda\CachingHostAllocator.h`

```py
#pragma once
// 预处理指令：确保此头文件仅被包含一次

#include <ATen/core/CachingHostAllocator.h>
// 引入 ATen 库中的 CachingHostAllocator 头文件
#include <c10/core/Allocator.h>
// 引入 c10 核心库中的 Allocator 头文件
#include <c10/cuda/CUDAStream.h>
// 引入 c10 CUDA 库中的 CUDAStream 头文件

namespace at::cuda {

//
// A caching allocator for CUDA host allocations (pinned memory).
// 用于 CUDA 主机内存分配（固定内存）的缓存分配器

// This provides a drop-in replacement for THCudaHostAllocator, which re-uses
// freed pinned (page-locked) memory allocations. This avoids device
// synchronizations due to cudaFreeHost calls.
// 此类提供了对 THCudaHostAllocator 的替代，可以重复使用已释放的固定内存分配，
// 避免由于 cudaFreeHost 调用而导致的设备同步。

// To ensure correct behavior, THCCachingHostAllocator_recordEvent must be
// called anytime a pointer from this allocator is used in a cudaMemcpyAsync
// call between host and device, and passed the corresponding context from the
// allocation. This is currently invoked by at::native::copy_kernel_cuda.
// 为确保正确的行为，必须在此分配器中的指针在主机与设备之间的 cudaMemcpyAsync 调用中使用时，
// 调用 THCCachingHostAllocator_recordEvent，并传递相应的上下文。目前由 at::native::copy_kernel_cuda 调用。

TORCH_CUDA_CPP_API c10::Allocator* getCachingHostAllocator();
// 获取缓存主机分配器的实例的函数声明

// Records an event in the specified stream. The allocation corresponding to the
// input `ptr`/`ctx` will not be re-used until the event has occurred.
// 在指定的流中记录事件。与输入 `ptr`/`ctx` 相对应的分配将在事件发生之前不被重新使用。
TORCH_CUDA_CPP_API bool CachingHostAllocator_recordEvent(
    void* ptr,
    void* ctx,
    c10::cuda::CUDAStream stream);
// 记录事件的函数声明，指定分配的指针和上下文，以及相关的 CUDA 流

// Releases cached pinned memory allocations via cudaHostFree
// 通过 cudaHostFree 释放缓存的固定内存分配
TORCH_CUDA_CPP_API void CachingHostAllocator_emptyCache();
// 清空缓存的函数声明

inline TORCH_CUDA_CPP_API at::DataPtr HostAlloc(size_t size) {
  return getCachingHostAllocator()->allocate(size);
}
// 在主机上分配内存的函数声明，使用缓存主机分配器实例来分配指定大小的内存

} // namespace at::cuda
// 命名空间结束声明，at::cuda


这段代码是关于在 CUDA 环境下使用缓存分配器（用于固定内存）的声明和说明。
```