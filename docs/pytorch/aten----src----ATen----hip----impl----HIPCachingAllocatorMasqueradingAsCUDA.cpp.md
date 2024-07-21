# `.\pytorch\aten\src\ATen\hip\impl\HIPCachingAllocatorMasqueradingAsCUDA.cpp`

```
// 包含 C10 核心的内存分配器头文件
#include <c10/core/Allocator.h>
// 包含用于 HIP 的 HIPCachingAllocatorMasqueradingAsCUDA 的具体实现头文件
#include <ATen/hip/impl/HIPCachingAllocatorMasqueradingAsCUDA.h>

// 定义命名空间 c10::hip::HIPCachingAllocatorMasqueradingAsCUDA
namespace c10 { namespace hip {
namespace HIPCachingAllocatorMasqueradingAsCUDA {

// 获取静态分配器实例的函数，返回一个指向静态 HIPAllocatorMasqueradingAsCUDA 对象的指针
Allocator* get() {
  // 静态 HIPAllocatorMasqueradingAsCUDA 对象 allocator 初始化，其基于 HIPCachingAllocator 实例
  static HIPAllocatorMasqueradingAsCUDA allocator(HIPCachingAllocator::get());
  // 返回指向 allocator 对象的指针
  return &allocator;
}

// 记录流的函数，将 HIPStreamMasqueradingAsCUDA 转换为 HIP 流并记录到 HIPCachingAllocator
void recordStreamMasqueradingAsCUDA(const DataPtr& ptr, HIPStreamMasqueradingAsCUDA stream) {
  // 调用 HIPCachingAllocator 的 recordStream 方法记录 ptr 指向的数据和 HIP 流
  HIPCachingAllocator::recordStream(ptr, stream.hip_stream());
}

} // namespace HIPCachingAllocatorMasqueradingAsCUDA
}} // namespace c10::hip
```