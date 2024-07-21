# `.\pytorch\aten\src\ATen\xpu\CachingHostAllocator.h`

```py
#pragma once
// 预处理指令，确保头文件只被编译一次

#include <ATen/core/CachingHostAllocator.h>
// 包含 CachingHostAllocator 头文件，提供了缓存主机分配器的实现

#include <ATen/xpu/XPUEvent.h>
#include <c10/core/Allocator.h>
#include <c10/xpu/XPUStream.h>

namespace at::xpu {
// 进入 at::xpu 命名空间

TORCH_XPU_API c10::Allocator* getCachingHostAllocator();
// 声明函数 getCachingHostAllocator()，返回 c10::Allocator* 类型的指针，
// 是获取缓存主机分配器的函数，TORCH_XPU_API 表示这是一个 Torch XPU API 函数

TORCH_XPU_API bool CachingHostAllocator_recordEvent(
    void* ptr,
    void* ctx,
    c10::xpu::XPUStream stream);
// 声明函数 CachingHostAllocator_recordEvent()，返回 bool 类型，
// 用于记录事件到缓存主机分配器，接受指针 ptr、ctx 和 XPUStream 作为参数

TORCH_XPU_API void CachingHostAllocator_emptyCache();
// 声明函数 CachingHostAllocator_emptyCache()，返回 void，
// 用于清空缓存主机分配器的缓存

inline TORCH_XPU_API at::DataPtr HostAlloc(size_t size) {
  return getCachingHostAllocator()->allocate(size);
}
// 定义了 inline 的 HostAlloc(size_t size) 函数，返回 at::DataPtr 类型，
// 使用 getCachingHostAllocator() 获取分配器，并分配 size 大小的内存

} // namespace at::xpu
// 结束 at::xpu 命名空间
```