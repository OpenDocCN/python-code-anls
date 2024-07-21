# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\qnnpack\AlignedAllocator.h`

```py
/*
 * 版权所有（c）Facebook, Inc. 及其关联公司。
 * 保留所有权利。
 *
 * 此源代码根据根目录中的 LICENSE 文件中的 BSD 样式许可证进行许可。
 */

#pragma once

#include <cstddef>
#include <limits>

#include <stdlib.h>

// 定义模板类 AlignedAllocator，用于按指定对齐方式分配内存
template <typename T, size_t Alignment>
class AlignedAllocator;

// 针对 void 类型的特化模板 AlignedAllocator
template <size_t Alignment>
class AlignedAllocator<void, Alignment> {
 public:
  typedef void* pointer;
  typedef const void* const_pointer;
  typedef void value_type;

  // 重新绑定模板，用于其他类型的内存分配
  template <class U>
  struct rebind {
    typedef AlignedAllocator<U, Alignment> other;
  };
};

// 模板类 AlignedAllocator 的主定义
template <typename T, size_t Alignment>
class AlignedAllocator {
 public:
  typedef T value_type;
  typedef T* pointer;
  typedef const T* const_pointer;
  typedef T& reference;
  typedef const T& const_reference;
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;

  // C++14 以上版本支持的容器移动分配
#if __cplusplus >= 201402L
  typedef std::true_type propagate_on_container_move_assignment;
#endif

  // 重新绑定模板，用于其他类型的内存分配
  template <class U>
  struct rebind {
    typedef AlignedAllocator<U, Alignment> other;
  };

 public:
  // 默认构造函数，无异常抛出
  inline AlignedAllocator() noexcept = default;

  // 复制构造函数，无异常抛出
  template <class U>
  inline AlignedAllocator(
      const AlignedAllocator<U, Alignment>& other) noexcept {}

  // 返回最大分配的对象数
  inline size_type max_size() const noexcept {
    return (std::numeric_limits<size_type>::max() - size_type(Alignment)) /
        sizeof(T);
  }

  // 返回对象的地址
  inline pointer address(reference x) const noexcept {
    return std::addressof(x);
  }

  // 返回 const 对象的地址
  inline const_pointer address(const_reference x) const noexcept {
    return std::addressof(x);
  }

  // 分配 n 个对象的内存，按 Alignment 对齐
  inline pointer allocate(
      size_type n,
      typename AlignedAllocator<void, Alignment>::const_pointer hint = 0) {
#if defined(__ANDROID__)
    // 在 Android 平台上使用 memalign 进行内存分配
    void* memory = memalign(Alignment, n * sizeof(T));
    if (memory == 0) {
#if !defined(__GNUC__) || defined(__EXCEPTIONS)
      throw std::bad_alloc(); // 内存分配失败抛出异常
#endif
    }
#else
    // 在其他平台上使用 posix_memalign 进行内存分配
    void* memory = nullptr;
    if (posix_memalign(&memory, Alignment, n * sizeof(T)) != 0) {
#if !defined(__GNUC__) || defined(__EXCEPTIONS)
      throw std::bad_alloc(); // 内存分配失败抛出异常
#endif
    }
#endif
    return static_cast<pointer>(memory); // 返回分配的内存地址
  }

  // 释放由 allocate 分配的内存
  inline void deallocate(pointer p, size_type n) noexcept {
    free(static_cast<void*>(p)); // 调用 free 函数释放内存
  }

  // 构造对象，使用 args 参数列表进行初始化
  template <class U, class... Args>
  inline void construct(U* p, Args&&... args) {
    ::new (static_cast<void*>(p)) U(std::forward<Args>(args)...); // 使用 placement new 进行对象构造
  }

  // 销毁对象
  template <class U>
  inline void destroy(U* p) {
    p->~U(); // 调用对象的析构函数
  }
};
```