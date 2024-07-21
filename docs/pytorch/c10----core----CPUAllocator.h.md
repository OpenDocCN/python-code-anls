# `.\pytorch\c10\core\CPUAllocator.h`

```
#pragma once
// 预处理指令，确保头文件只被包含一次

#include <cstdint>
// 包含 C++ 标准库中的整数类型头文件

#include <cstring>
// 包含 C 语言风格字符串操作函数的头文件

#include <mutex>
// 包含互斥锁相关的头文件，用于多线程同步

#include <unordered_map>
// 包含无序映射容器相关的头文件，用于存储键值对的哈希表

#include <c10/core/Allocator.h>
// 包含 C10 核心模块的 Allocator 类的头文件

#include <c10/macros/Export.h>
// 包含 C10 导出相关的宏定义的头文件

#include <c10/util/Flags.h>
// 包含 C10 中处理命令行参数的 Flags 相关的头文件

// TODO: rename to c10
// 提示信息，建议将项目中的标识符从 caffe2 修改为 c10

namespace c10 {

using MemoryDeleter = void (*)(void*);
// 定义 MemoryDeleter 类型别名，表示一个指向 void 返回类型，参数为 void* 的函数指针

// A helper function that is basically doing nothing.
// 一个辅助函数，实际上什么也不做
C10_API void NoDelete(void*);

// A simple struct that is used to report C10's memory allocation,
// deallocation status and out-of-memory events to the profiler
// 一个简单的结构体，用于向分析器报告 C10 的内存分配、释放状态和内存耗尽事件
class C10_API ProfileedCPUMemoryReporter {
 public:
  ProfiledCPUMemoryReporter() = default;
  // 默认构造函数

  void New(void* ptr, size_t nbytes);
  // 记录新分配的内存块的指针和字节数

  void OutOfMemory(size_t nbytes);
  // 记录内存耗尽事件和所需的字节数

  void Delete(void* ptr);
  // 记录释放的内存块的指针

 private:
  std::mutex mutex_;
  // 互斥锁，用于保护 size_table_ 的并发访问

  std::unordered_map<void*, size_t> size_table_;
  // 无序映射，存储指针和其对应的字节数

  size_t allocated_ = 0;
  // 已分配的总字节数

  size_t log_cnt_ = 0;
  // 记录的事件数目
};

C10_API ProfileedCPUMemoryReporter& profiledCPUMemoryReporter();
// 返回全局唯一的 ProfileedCPUMemoryReporter 对象的引用

// Get the CPU Allocator.
// 获取 CPU 分配器的函数声明
C10_API at::Allocator* GetCPUAllocator();

// Sets the CPU allocator to the given allocator: the caller gives away the
// ownership of the pointer.
// 将 CPU 分配器设置为给定的分配器，并释放调用者对指针的所有权
C10_API void SetCPUAllocator(at::Allocator* alloc, uint8_t priority = 0);

// Get the Default CPU Allocator
// 获取默认的 CPU 分配器
C10_API at::Allocator* GetDefaultCPUAllocator();

// Get the Default Mobile CPU Allocator
// 获取默认的移动 CPU 分配器
C10_API at::Allocator* GetDefaultMobileCPUAllocator();

// The CPUCachingAllocator is experimental and might disappear in the future.
// The only place that uses it is in StaticRuntime.
// Set the CPU Caching Allocator
// 设置 CPU 缓存分配器
C10_API void SetCPUCachingAllocator(Allocator* alloc, uint8_t priority = 0);

// Get the CPU Caching Allocator
// 获取 CPU 缓存分配器
C10_API Allocator* GetCPUCachingAllocator();

} // namespace c10
// 命名空间 c10 的结束标记
```