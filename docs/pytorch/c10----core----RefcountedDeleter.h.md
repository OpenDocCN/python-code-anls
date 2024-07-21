# `.\pytorch\c10\core\RefcountedDeleter.h`

```py
#pragma once
// 只允许该头文件被编译一次

#include <c10/core/Storage.h>
// 引入 c10 核心库中的 Storage 头文件
#include <c10/macros/Export.h>
// 引入 c10 导出宏定义头文件
#include <c10/util/UniqueVoidPtr.h>
// 引入 c10 工具中的 UniqueVoidPtr 头文件

#include <atomic>
// 引入原子操作库
#include <memory>
// 引入内存管理库

namespace c10 {
// 进入 c10 命名空间

// RefcountedDeleterContext 对象用作 DataPtr 的 ctx 参数，用于实现共享的 DataPtr。
// 通常情况下，DataPtr 是唯一的，但我们使用此自定义上下文和下面的 `refcounted_deleter` 函数，
// 使 DataPtr 的行为像一个非唯一的 DataPtr。该上下文对象保存了一个内部上下文和删除函数，
// 当引用计数达到 0 时，处理实际的数据删除。

// 此共享 DataPtr 特性仅在多个 Python 解释器中的 MultiPy 中使用时才使用。
// 在 Storage 具有 PyObject 保留之前，解释器可以共享同一个 StorageImpl 实例。
// 但是现在，为了正确管理僵尸 PyObject，一个 StorageImpl 只能与一个解释器关联。
// 因此，我们通过为每个解释器创建不同的 StorageImpl 实例，但它们都指向相同的数据，
// 来跨 Python 解释器共享存储。

struct C10_API RefcountedDeleterContext {
  // 构造函数，初始化 RefcountedDeleterContext 对象
  RefcountedDeleterContext(void* other_ctx, c10::DeleterFnPtr other_deleter)
      : other_ctx(other_ctx, other_deleter), refcount(1) {}

  // 持有内部上下文和处理数据删除的删除函数的 unique_ptr
  std::unique_ptr<void, c10::DeleterFnPtr> other_ctx;
  // 引用计数，原子整数类型
  std::atomic_int refcount;
};

// `refcounted_deleter` 用作 DataPtr 的 `ctx_deleter`，以实现共享的 DataPtr。
// 警告：只能在使用 `new` 在堆上分配的 RefcountedDeleterContext 指针上调用此函数，
// 因为当引用计数达到 0 时，上下文将使用 `delete` 进行删除。
C10_API void refcounted_deleter(void* ctx_);

// 如果 storage 的 DataPtr 不使用 `refcounted_deleter`，则用一个使用它的 DataPtr 替换它，
// 这样它可以在多个 StorageImpl 之间共享
C10_API void maybeApplyRefcountedDeleter(const c10::Storage& storage);

// 创建一个新的 StorageImpl，指向相同的数据。
// 如果原始 StorageImpl 的 DataPtr 不使用 `refcounted_deleter`，将用一个使用它的 DataPtr 替换它
C10_API c10::Storage newStorageImplFromRefcountedDataPtr(
    const c10::Storage& storage);

} // namespace c10
// 结束 c10 命名空间
```