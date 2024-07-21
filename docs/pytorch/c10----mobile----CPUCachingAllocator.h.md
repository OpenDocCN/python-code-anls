# `.\pytorch\c10\mobile\CPUCachingAllocator.h`

```py
#pragma once
// 预处理指令，确保本文件只被编译一次

#include <cstddef>
// 包含头文件，提供 size_t 类型定义

#include <mutex>
// 包含头文件，提供互斥锁功能

#include <c10/macros/Export.h>
// 包含头文件，导出符号定义

#include <c10/util/SmallVector.h>
// 包含头文件，提供 SmallVector 容器工具类

#include <c10/util/flat_hash_map.h>
// 包含头文件，提供 flat_hash_map 哈希表实现

/*
 * CPUCachingAllocator:
 * DISCLAIMER:
 *    This is subject to change (beta) and only supported on mobile builds.
 *    If code snippet such as in 'Usage pattern' is used outside of mobile
 *    build you will not observe the intended behavior.
 *    See below for more information.
 * Why?
 *    It has been observed that some mobile platforms, such as pixel 3, return
 *    memory aggressively to the system. This results in page faults in some
 *    cases and ends up hurting performance. This caching allocator aims to address
 *    that. Furthermore it also allows users to specify their own allocator by
 *    implementing allocate/free virtual interfaces. What are the cons? There are
 *    some cons that were observed where use of caching allocator led to worse
 *    performance on some platforms. Reason being that the caching mechanism used
 *    by this allocator left us worse off compared to the corresponding platform's
 *    tuned memory allocator. In that case it seemed better to not use this
 *    allocator. Note there are some ideas to fix this in the works.
 *
 * Usage:
 * Usage pattern:
 * Instantiate and own the caching allocator.
 * std::unique_ptr<c10::CPUCachingAllocator> caching_allocator =
 *   std::make_unique<c10::CPUCachingAllocator>();
 * Use caching allocator with a scoped guard at inference time.
 * {
 * WithCPUCachingAllocatorGuard(caching_allocator.get());
 * ... model.forward(...);
 * }
 */

namespace c10 {
// 命名空间 c10 的开始
class C10_API CPUCachingAllocator {
  /*
   * What it does:
   * Caches all the allocations carried out by this allocator.
   * Cache key is the size of the allocation.
   * If requested size is found in the cache returns the cached pointer.
   * What it does not do:
   * No speculative allocation for any future allocations.
   */
 private:
  // 内联函数，分配内存并将其缓存，返回分配的指针
  inline void* allocate_and_cache(const size_t bytes);
  // 释放缓存的内存
  void free_cached();

 protected:
  // 不变性条件:
  // 1. 如果内存通过此分配器分配，则指针将存在于 allocation_map_ 中，
  //    除非分配器通过 free_cached 将内存返回给操作系统。
  //   1.1. 因此，即使通过此分配器"释放"内存（因此缓存），它仍将保留在
  //        allocation_map_ 中。此外，它还将存在于 available_map_ 中。
  //        因此，分配的内存指针可以同时存在于 allocation_map_ 和
  //        available_map_ 中。
  // 2. 当通过此分配器分配但在其作用域之外释放时，可以从 allocation_map_
  //    中移除内存指针。
  // 3. available_map_ 仅包含由此分配器分配且随后由此分配器释放的内存。
  // 因为上述不变性条件，分配的内存指针不能仅存在于 available_map_ 中，
  // 除非它也存在于 allocation_map_ 中。
  ska::flat_hash_map<size_t, c10::SmallVector<void*, 16>> available_map_;
  static ska::flat_hash_map<void*, size_t> allocation_map_;
  // 由于 allocation_map 是全局实例，通过所有公共 API 进行修改/读取，因此需要全局互斥锁。
  static std::mutex mutex_;

 public:
  static void record_free(void* ptr);
  virtual ~CPUCachingAllocator();
  // 检查缓存以查看是否可以找到大小为 bytes 的分配。
  // 如果找到，则返回缓存的内存；否则分配内存，记录并缓存，然后返回。
  virtual void* allocate(const size_t bytes);
  // 检查要释放的内存是否被早期调用 allocate 标记为分配。如果是，则缓存分配；否则释放。
  virtual void free(void* ptr);
};

CPUCachingAllocator* GetDefaultCPUCachingAllocator();

bool ThreadLocalCachingAllocatorEnabled();
CPUCachingAllocator* GetThreadLocalCachingAllocator();

class C10_API WithCPUCachingAllocatorGuard {
 public:
  // 构造函数，设置 CPUCachingAllocator 的保护。
  WithCPUCachingAllocatorGuard(CPUCachingAllocator* allocator);
  // 析构函数，清理 CPUCachingAllocator 的保护。
  ~WithCPUCachingAllocatorGuard();

 private:
  CPUCachingAllocator* prev_caching_allocator_ptr_{nullptr};
};

} // namespace c10
```