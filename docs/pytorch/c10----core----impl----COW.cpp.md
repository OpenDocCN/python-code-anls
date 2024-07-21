# `.\pytorch\c10\core\impl\COW.cpp`

```py
// 包含 COW（Copy-On-Write）实现的头文件
#include <c10/core/impl/COW.h>

// 包含 Allocator 相关的头文件
#include <c10/core/Allocator.h>
#include <c10/core/StorageImpl.h>
#include <c10/core/alignment.h>

// 包含 COWDeleter 相关的头文件
#include <c10/core/impl/COWDeleter.h>

// 包含异常处理相关的头文件
#include <c10/util/Exception.h>

// 包含并行保护相关的头文件
#include <c10/util/ParallelGuard.h>

// 包含 UniqueVoidPtr 相关的头文件
#include <c10/util/UniqueVoidPtr.h>

// 包含标准库的内存管理头文件
#include <memory>

// 包含 std::optional 头文件，用于可选类型
#include <optional>

namespace c10::impl::cow {

namespace {

// 使用 COWDeleterContext 创建一个包装了 copy-on-write DataPtr 的 DataPtr
at::DataPtr make_data_ptr(
    at::DataPtr const& data_ptr,
    cow::COWDeleterContext& ctx) {
  return at::DataPtr(data_ptr.get(), &ctx, cow::cow_deleter, data_ptr.device());
}

// 复制一个 copy-on-write DataPtr
at::DataPtr copy_data_ptr(at::DataPtr const& data_ptr) {
  auto* ctx = data_ptr.cast_context<cow::COWDeleterContext>(cow::cow_deleter);
  TORCH_INTERNAL_ASSERT(ctx != nullptr);
  ctx->increment_refcount();
  return make_data_ptr(data_ptr, *ctx);
}

} // namespace

// 检查一个 StorageImpl 是否具有简单的 data pointer
bool has_simple_data_ptr(const c10::StorageImpl& storage) {
  const c10::DataPtr& data_ptr = storage.data_ptr();
  const void* ctx = data_ptr.get_context();
  const void* data = data_ptr.get();
  const c10::Allocator* allocator = storage.allocator();
  if (allocator != nullptr) {
    return allocator->is_simple_data_ptr(data_ptr);
  } else {
    return ctx == data;
  }
}

// 检查一个 DataPtr 是否为 copy-on-write DataPtr
bool is_cow_data_ptr(const c10::DataPtr& data_ptr) {
  return (void*)data_ptr.get_deleter() == (void*)&cow::cow_deleter;
}

// 对一个 StorageImpl 执行延迟克隆，返回一个新的 intrusive_ptr<StorageImpl>
c10::intrusive_ptr<StorageImpl> lazy_clone_storage(StorageImpl& storage) {
  const at::DataPtr& data_ptr = storage.data_ptr();

  // 存在三种可能的情况：
  //
  // 1) StorageImpl 具有普通的 data pointer，没有异常的上下文。在这种情况下，
  //    我们知道没有盲目的别名引用到 StorageImpl：它们都将是公共别名，
  //    用户需要手动同步。
  //
  //    在这种情况下不需要锁定。
  //
  // 2) StorageImpl 已经具有 copy on write 上下文。存在与盲目别名的潜在竞态条件
  //    （即用户不需要与之同步的别名）。因为我们的输入 storage 绑定到数据的
  //    活跃引用，我们知道它不会消失。一个盲目别名可能正在从中复制，
  //    但是我们将获取上下文的互斥锁以保护我们。
  //
  //    在这种情况下也不需要锁定，因为我们只是包装一个我们知道不会消失的上下文。
  //
  // 3) StorageImpl 具有非 copy on write 上下文。这种情况不受支持，因此我们只返回空值。
  //
  //    在这种情况下也不需要锁定。

  std::optional<DataPtr> new_data_ptr; // 必须在下面设置

  if (has_simple_data_ptr(storage)) {
    // 情况 1）我们有一个简单的 data pointer：包装它。
    std::unique_ptr<void, DeleterFnPtr> original_ctx =
        storage._mutable_data_ptr_no_checks().move_context();

    // 将此保存为结果。
    // 创建新的数据指针，使用 make_data_ptr 函数，将原始数据指针 data_ptr 和一个移动语义的 cow::COWDeleterContext 对象传入
    new_data_ptr = make_data_ptr(
        data_ptr, *new cow::COWDeleterContext(std::move(original_ctx)));

    // 将 storage 更新为新的 copy-on-write 上下文中的数据指针
    storage.set_data_ptr_noswap(copy_data_ptr(*new_data_ptr));
  } else if (is_cow_data_ptr(data_ptr)) {
    // 情况2): 数据指针已经存在 copy-on-write 上下文。只需返回一个新的 storage 实现。
    new_data_ptr = copy_data_ptr(data_ptr);
  } else {
    // 情况3): 存在上下文，但不是 copy-on-write。在这种情况下无法处理，返回空指针。
    return nullptr;
  }

  // 断言新的数据指针 new_data_ptr 有值
  TORCH_INTERNAL_ASSERT(new_data_ptr.has_value());

  // 使用 make_storage_impl 函数创建并返回存储实现对象
  return make_storage_impl(
      StorageImpl::use_byte_size_t(),
      storage.sym_nbytes(),
      *std::move(new_data_ptr),
      storage.allocator(),
      storage.resizable(),
      storage.device_type());
}

C10_API void materialize_cow_storage(StorageImpl& storage) {
  // 断言并行保护未启用，不允许在 at::parallel_for 的循环函数中实例化存储
  TORCH_INTERNAL_ASSERT(
      !c10::ParallelGuard::is_enabled(),
      "Materializing a storage in the loop function of at::parallel_for is forbidden");
  // 获取存储的数据指针
  const at::DataPtr& data_ptr = storage.data_ptr();

  // 将数据指针转换为 cow::COWDeleterContext 上下文
  auto* ctx = data_ptr.cast_context<cow::COWDeleterContext>(cow::cow_deleter);
  // 断言上下文不为空
  TORCH_INTERNAL_ASSERT(ctx != nullptr);

  // 减少引用计数并获取结果
  auto result = ctx->decrement_refcount();

  // 下面的分支必须设置这个变量
  std::optional<DataPtr> new_data_ptr;

  if (std::holds_alternative<cow::COWDeleterContext::LastReference>(result)) {
    // 这是数据的唯一引用。如果有竞争写入，上下文保证在给出结果之前已经完成了它们。
    std::unique_ptr<void, DeleterFnPtr> data =
        std::get<cow::COWDeleterContext::LastReference>(std::move(result));
    TORCH_INTERNAL_ASSERT(data.get() == data_ptr.get());
    // 创建新的 DataPtr，将所有权转移给 new_data_ptr
    new_data_ptr = DataPtr(
        data.release(), data_ptr.get(), data.get_deleter(), data_ptr.device());
  } else {
    TORCH_INTERNAL_ASSERT(
        std::holds_alternative<cow::COWDeleterContext::NotLastReference>(
            result));
    // 我们不需要消耗结果，它只是一个共享锁，确保在复制数据时数据仍然存在。
    // 使用分配器克隆数据指针，返回新的 DataPtr
    new_data_ptr = storage.allocator()->clone(data_ptr.get(), storage.nbytes());
  }

  // 断言新的数据指针有值
  TORCH_INTERNAL_ASSERT(new_data_ptr.has_value());
  // 设置新的数据指针并返回旧的数据指针
  DataPtr old_data_ptr =
      storage.set_data_ptr_no_materialize_cow(*std::move(new_data_ptr));
  // 上面已经减少了上下文的引用计数，释放对上下文的引用，以防再次减少引用计数
  old_data_ptr.release_context();
}

} // namespace c10::impl::cow
```