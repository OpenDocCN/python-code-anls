# `.\pytorch\c10\mobile\CPUCachingAllocator.cpp`

```
// 包含 C10 库中的头文件，用于 CPU 内存分配和异常处理
#include <c10/core/impl/alloc_cpu.h>
#include <c10/mobile/CPUCachingAllocator.h>
#include <c10/util/Exception.h>

// 定义命名空间 c10
namespace c10 {

// 匿名命名空间，用于保存线程局部变量 caching_allocator_ptr
namespace {
thread_local CPUCachingAllocator* caching_allocator_ptr{nullptr};
} // namespace

// 静态成员变量定义和初始化
std::mutex CPUCachingAllocator::mutex_;
ska::flat_hash_map<void*, size_t> CPUCachingAllocator::allocation_map_;

// 分配内存并缓存的内联函数定义
inline void* CPUCachingAllocator::allocate_and_cache(const size_t bytes) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  void* ptr;
  try {
    // 调用 c10::alloc_cpu 分配内存
    ptr = c10::alloc_cpu(bytes);
  } catch (c10::Error&) {
    // 如果分配失败，尝试释放所有缓存的可用块
    free_cached();
    // 再次尝试分配
    ptr = c10::alloc_cpu(bytes);
  }
  // 记录分配的指针和大小到 allocation_map_
  allocation_map_[ptr] = bytes;
  return ptr;
}

// 分配内存的函数定义，加锁以确保线程安全
void* CPUCachingAllocator::allocate(const size_t bytes) {
  std::lock_guard<std::mutex> guard(mutex_);
  // 查找指定大小的内存块是否可用
  const auto& it = available_map_.find(bytes);
  if (it == available_map_.end() || it->second.empty()) {
    // 如果未找到可用块，则调用 allocate_and_cache 进行分配
    return allocate_and_cache(bytes);
  }
  // 从可用块中取出最后一个，并返回其指针
  return it->second.pop_back_val();
}

// 释放内存的函数定义，加锁以确保线程安全
void CPUCachingAllocator::free(void* ptr) {
  // 注意：实际上并未释放内存，只是将其添加到可用块中
  std::lock_guard<std::mutex> guard(mutex_);
  // 如果 allocation_map_ 中存在指定指针，则表示该内存是由该分配器分配的
  const auto& it = allocation_map_.find(ptr);
  if (it == allocation_map_.end()) {
    // 如果未找到，直接调用 c10::free_cpu 释放内存
    c10::free_cpu(ptr);
    return;
  }
  // 获取该内存块的大小，并将其添加到可用块中
  const size_t alloc_size = it->second;
  available_map_[alloc_size].push_back(ptr);
}

// 记录释放内存的函数定义，加锁以确保线程安全
void CPUCachingAllocator::record_free(void* ptr) {
  // 此函数捕获分配器之外释放内存的情况
  std::lock_guard<std::mutex> guard(mutex_);
  // 如果 allocation_map_ 中存在指定指针，则从中移除
  const auto& it = allocation_map_.find(ptr);
  if (it != allocation_map_.end()) {
    allocation_map_.erase(it);
  }
}

// 释放所有缓存内存的函数定义
void CPUCachingAllocator::free_cached() {
  // 遍历可用块，并释放其所有缓存的内存
  for (const auto& it : available_map_) {
    for (const auto ptr : it.second) {
      // 对于迭代器 `it` 所指向的第二个元素（假设其为一个指针数组），遍历其中的每个指针 `ptr`
      c10::free_cpu(ptr);
      // 使用 c10 库函数释放 CPU 上的内存 `ptr`
      // 当缓存的内存返回给操作系统时，必须将其从 allocation_map 中移除
      allocation_map_.erase(ptr);
      // 在 allocation_map 中移除指针 `ptr`
    }
  }
  // 清空 available_map_ 容器，可能是清除所有可用内存块的映射关系
  available_map_.clear();
}

# CPUCachingAllocator 类的析构函数，用于释放缓存的资源
CPUCachingAllocator::~CPUCachingAllocator() {
    调用 free_cached() 方法释放已缓存的资源
    free_cached();
}

# 获取线程本地的 CPUCachingAllocator 实例指针
CPUCachingAllocator* GetThreadLocalCachingAllocator() {
    直接返回全局变量 caching_allocator_ptr，即当前线程的缓存分配器实例
    return caching_allocator_ptr;
}

# WithCPUCachingAllocatorGuard 类的构造函数，用于创建 CPUCachingAllocator 的临时保护
WithCPUCachingAllocatorGuard::WithCPUCachingAllocatorGuard(
    CPUCachingAllocator* allocator)
    : prev_caching_allocator_ptr_(GetThreadLocalCachingAllocator()) {
    将当前线程的 caching_allocator_ptr 保存到 prev_caching_allocator_ptr_，以便析构时恢复
    caching_allocator_ptr = allocator;
}

# WithCPUCachingAllocatorGuard 类的析构函数，用于恢复之前的 CPUCachingAllocator 实例指针
WithCPUCachingAllocatorGuard::~WithCPUCachingAllocatorGuard() {
    恢复之前保存的线程本地的 caching_allocator_ptr，以保证不影响原有的线程状态
    caching_allocator_ptr = prev_caching_allocator_ptr_;
}

} // namespace c10
```