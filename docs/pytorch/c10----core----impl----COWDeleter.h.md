# `.\pytorch\c10\core\impl\COWDeleter.h`

```
#pragma once
// 预处理指令，确保此头文件只被包含一次

#include <c10/macros/Export.h>
// 包含 c10 库的导出宏定义

#include <c10/util/UniqueVoidPtr.h>
// 包含 c10 库中的 UniqueVoidPtr 类定义

#include <atomic>
// 包含原子操作相关的头文件

#include <cstdint>
// 包含标准整数类型定义的头文件

#include <memory>
// 包含智能指针及相关操作的头文件

#include <shared_mutex>
// 包含共享互斥锁相关的头文件

#include <variant>
// 包含 variant 类模板及相关操作的头文件

namespace c10::impl::cow {

// 定义一个 COWDeleterContext 类，用作 DataPtr 的 ctx 参数以实现写时复制 (COW) DataPtr。
class C10_API COWDeleterContext {
 public:
  // 创建一个实例，持有数据和原始删除器的指针。
  //
  // 注意，如果在没有实例化的情况下最后一个引用消失，则我们的析构函数将调用删除器。
  explicit COWDeleterContext(std::unique_ptr<void, DeleterFnPtr> data);

  // 增加当前引用计数。
  void increment_refcount();

  // 请参阅此目录中的 README.md 以了解锁定策略。
  
  // 表示对上下文的引用。
  //
  // 这是由 decrement_refcount 返回的，允许调用者在共享锁下复制数据。
  using NotLastReference = std::shared_lock<std::shared_mutex>;

  // 表示对上下文的最后引用。
  //
  // 当剩余的引用计数为最后一个并且所有挂起的复制操作完成后，此指针将被 decrement_refcount 返回。
  using LastReference = std::unique_ptr<void, DeleterFnPtr>;

  // 减少引用计数，返回一个指示如何处理引用的句柄。
  std::variant<NotLastReference, LastReference> decrement_refcount();

 private:
  // 析构函数是私有的，这应该只在 UniqueVoidPtr 中使用，使用 cow::delete_context 作为删除器。
  ~COWDeleterContext();

  std::shared_mutex mutex_;  // 使用共享互斥锁保护数据访问
  std::unique_ptr<void, DeleterFnPtr> data_;  // 持有数据和删除器的唯一指针
  std::atomic<std::int64_t> refcount_ = 1;  // 原子操作的引用计数
};

// cow_deleter 用作 DataPtr 的 ctx_deleter 以实现 COW DataPtr。
//
// 警告：仅在指向用 new 在堆上分配的 COWDeleterContext 的指针上调用此函数，
// 因为当引用计数达到 0 时，上下文将使用 delete 删除。
C10_API void cow_deleter(void* ctx);

} // namespace c10::impl::cow
```