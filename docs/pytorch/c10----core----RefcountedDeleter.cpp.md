# `.\pytorch\c10\core\RefcountedDeleter.cpp`

```py
#include <c10/core/RefcountedDeleter.h>

#include <mutex>

namespace c10 {

// 自定义函数，用于减少引用计数并在必要时删除相关的上下文
void refcounted_deleter(void* ctx_) {
  // 将上下文转换为正确的类型
  RefcountedDeleterContext& ctx =
      *reinterpret_cast<RefcountedDeleterContext*>(ctx_);
  // 减少引用计数
  ctx.refcount--;
  // 如果引用计数减少到零，则释放相关的上下文
  if (ctx.refcount == 0) {
    ctx.other_ctx = nullptr;
    delete &ctx;
  }
}

// 用于替换数据指针时的互斥量
std::mutex replace_data_ptr_mutex;

// 根据存储对象是否已共享，可能应用引用计数删除器
void maybeApplyRefcountedDeleter(const c10::Storage& storage) {
  // 锁定互斥量，确保线程安全
  std::lock_guard<std::mutex> guard(replace_data_ptr_mutex);
  // 获取可变的数据指针
  c10::DataPtr& data_ptr = storage.mutable_data_ptr();

  // 如果数据指针的删除器已经是引用计数删除器，则无需操作
  if ((void*)data_ptr.get_deleter() == (void*)&c10::refcounted_deleter) {
    // 数据指针已经共享
    return;
  }

  // 获取数据指针的当前值
  void* data = data_ptr.get();
  void* other_ctx = data_ptr.get_context();
  c10::DeleterFnPtr other_deleter = data_ptr.get_deleter();
  c10::Device device = data_ptr.device();

  // 释放原始 DataPtr 的上下文，以防在替换原始 DataPtr 时删除数据
  data_ptr.release_context();

  // 创建新的引用计数删除器上下文
  c10::RefcountedDeleterContext* refcount_ctx =
      new c10::RefcountedDeleterContext(other_ctx, other_deleter);

  // 创建新的 DataPtr，使用引用计数删除器
  c10::DataPtr new_data_ptr(
      data,
      reinterpret_cast<void*>(refcount_ctx),
      &c10::refcounted_deleter,
      device);
  
  // 设置存储对象的新 DataPtr
  storage.set_data_ptr(std::move(new_data_ptr));
}

// 根据引用计数数据指针创建新的存储对象实现
c10::Storage newStorageImplFromRefcountedDataPtr(const c10::Storage& storage) {
  // 可能应用引用计数删除器到存储对象
  c10::maybeApplyRefcountedDeleter(storage);

  // 获取存储对象的实现指针
  c10::StorageImpl* storage_impl = storage.unsafeGetStorageImpl();

  // 获取存储对象的可变数据指针
  c10::DataPtr& data_ptr = storage.mutable_data_ptr();

  // 复制当前数据指针，以防止修改原始数据
  c10::DataPtr new_data_ptr(
      data_ptr.get(),
      data_ptr.get_context(),
      data_ptr.get_deleter(),
      data_ptr.device());

  // 增加引用计数，此行代码应立即在创建 new_data_ptr 后执行
  reinterpret_cast<c10::RefcountedDeleterContext*>(data_ptr.get_context())
      ->refcount++;

  // 创建新的存储对象并返回
  c10::Storage new_storage = c10::make_intrusive<c10::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      storage_impl->nbytes(),
      std::move(new_data_ptr),
      storage_impl->allocator(),
      /*resizable=*/storage_impl->resizable());
  return new_storage;
}

} // namespace c10
```