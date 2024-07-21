# `.\pytorch\aten\src\ATen\cuda\ThrustAllocator.h`

```py
#pragma once

// 预处理指令：#pragma once 用于确保头文件只被编译一次，防止多重包含。


#include <cstddef>
#include <c10/cuda/CUDACachingAllocator.h>

// 包含标准库头文件 <cstddef> 和 CUDA Caching 分配器头文件 <c10/cuda/CUDACachingAllocator.h>。


namespace at::cuda {

// 命名空间声明：定义命名空间 at::cuda。


/// Allocator for Thrust to re-route its internal device allocations
/// to the THC allocator
class ThrustAllocator {
public:
  typedef char value_type;
  
  char* allocate(std::ptrdiff_t size) {
    return static_cast<char*>(c10::cuda::CUDACachingAllocator::raw_alloc(size));
  }
  
  void deallocate(char* p, size_t size) {
    c10::cuda::CUDACachingAllocator::raw_delete(p);
  }
};

// 类 ThrustAllocator 的定义：用于将 Thrust 内部的设备分配重定向到 THC 分配器。
// - 类型定义：value_type 被定义为 char。
// - 成员函数 allocate(size)：分配 size 大小的内存，并使用 CUDACachingAllocator 进行原始内存分配。
// - 成员函数 deallocate(p, size)：释放指针 p 所指向的内存，使用 CUDACachingAllocator 进行原始内存释放。


} // namespace at::cuda

// 命名空间结束：at::cuda 命名空间的结束标记。
```