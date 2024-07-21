# `.\pytorch\c10\core\Storage.h`

```py
#pragma once
// 预处理指令，确保头文件只被包含一次

#include <c10/core/Allocator.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/core/StorageImpl.h>
#include <c10/core/SymInt.h>
#include <c10/macros/Export.h>
#include <c10/util/Exception.h>
#include <c10/util/ExclusivelyOwned.h>
#include <c10/util/MaybeOwned.h>
#include <c10/util/UniqueVoidPtr.h>
#include <c10/util/intrusive_ptr.h>
#include <cstddef>
#include <utility>
// 包含所需的头文件

namespace c10 {

struct Storage;
// 声明 Storage 结构体

C10_API bool isSharedStorageAlias(
    const Storage& storage0,
    const Storage& storage1);
// 声明用于检查存储是否共享别名的函数

struct C10_API Storage {
  public:
  struct use_byte_size_t {};
  struct unsafe_borrow_t {
    explicit unsafe_borrow_t() = default;
  };

  Storage() = default;
  // 默认构造函数，使用默认参数

  Storage(c10::intrusive_ptr<StorageImpl> ptr)
      : storage_impl_(std::move(ptr)) {}
  // 构造函数，使用给定的 StorageImpl 指针初始化存储对象

  // 使用给定的分配器分配内存缓冲区并创建存储对象
  Storage(
      use_byte_size_t /*use_byte_size*/,
      const SymInt& size_bytes,
      Allocator* allocator = nullptr,
      bool resizable = false)
      : storage_impl_(c10::make_intrusive<StorageImpl>(
            StorageImpl::use_byte_size_t(),
            size_bytes,
            allocator,
            resizable)) {}

  // 使用预分配的内存缓冲区创建存储对象，分配器用于可能的未来重新分配
  Storage(
      use_byte_size_t /*use_byte_size*/,
      size_t size_bytes,
      at::DataPtr data_ptr,
      at::Allocator* allocator = nullptr,
      bool resizable = false)
      : storage_impl_(c10::make_intrusive<StorageImpl>(
            StorageImpl::use_byte_size_t(),
            size_bytes,
            std::move(data_ptr),
            allocator,
            resizable)) {}

  protected:
  explicit Storage(unsafe_borrow_t, const Storage& rhs)
      : storage_impl_(c10::intrusive_ptr<c10::StorageImpl>::reclaim(
            rhs.storage_impl_.get())) {}
  // 受保护的构造函数，用于通过借用创建存储对象，直接从另一个存储对象中获取 StorageImpl 指针

  friend MaybeOwnedTraits<Storage>;

  public:
  // 用于临时使用 Caffe2 API 创建部分初始化（dtype 或内存）存储对象的传统构造函数
  static Storage create_legacy(at::Device device) {
    auto allocator = GetAllocator(device.type());
    return Storage(c10::make_intrusive<StorageImpl>(
        StorageImpl::use_byte_size_t(),
        0,
        allocator->allocate(0), // materialize a non-default Device.
        allocator,
        true));
  }

  // 重置存储对象为 legacy 状态，不需要新创建的 StorageImpl
  void reset_legacy() {
    TORCH_CHECK(resizable() && allocator());
    set_nbytes(0);
    set_data_ptr_noswap(allocator()->allocate(0));
  }

  // 设置存储对象的字节数，用于未来移除
  // 设置存储对象的字节数，支持 SymInt 类型的参数
  void set_nbytes(size_t size_bytes) const {
    storage_impl_->set_nbytes(size_bytes);
  }

  void set_nbytes(c10::SymInt size_bytes) const {
    storage_impl_->set_nbytes(std::move(size_bytes));
  }

  // 检查存储对象是否可调整大小
  bool resizable() const {
  // 返回存储实现是否支持调整大小的布尔值
  return storage_impl_->resizable();
}

size_t nbytes() const {
  // 返回存储实现的总字节数
  return storage_impl_->nbytes();
}

SymInt sym_nbytes() const {
  // 返回符号化的存储实现的总字节数
  return storage_impl_->sym_nbytes();
}
// get() use here is to get const-correctness

const void* data() const {
  // 返回存储实现的常量数据指针
  return storage_impl_->data();
}

void* mutable_data() const {
  // 返回存储实现的可变数据指针
  return storage_impl_->mutable_data();
}

at::DataPtr& mutable_data_ptr() const {
  // 返回存储实现的可变数据指针的引用
  return storage_impl_->mutable_data_ptr();
}

const at::DataPtr& data_ptr() const {
  // 返回存储实现的数据指针的常量引用
  return storage_impl_->data_ptr();
}

// 返回先前的数据指针
at::DataPtr set_data_ptr(at::DataPtr&& data_ptr) const {
  return storage_impl_->set_data_ptr(std::move(data_ptr));
}

void set_data_ptr_noswap(at::DataPtr&& data_ptr) const {
  // 设置存储实现的数据指针，不交换指针
  return storage_impl_->set_data_ptr_noswap(std::move(data_ptr));
}

DeviceType device_type() const {
  // 返回存储实现的设备类型
  return storage_impl_->device_type();
}

at::Allocator* allocator() const {
  // 返回存储实现的分配器指针
  return storage_impl_->allocator();
}

at::Device device() const {
  // 返回存储实现的设备对象
  return storage_impl_->device();
}

StorageImpl* unsafeReleaseStorageImpl() {
  // 释放并返回不安全的存储实现指针
  return storage_impl_.release();
}

StorageImpl* unsafeGetStorageImpl() const noexcept {
  // 返回不安全的存储实现指针，不抛出异常
  return storage_impl_.get();
}

c10::weak_intrusive_ptr<StorageImpl> getWeakStorageImpl() const {
  // 返回存储实现的弱引用指针
  return c10::weak_intrusive_ptr<StorageImpl>(storage_impl_);
}

operator bool() const {
  // 检查存储实现是否有效
  return storage_impl_;
}

size_t use_count() const {
  // 返回存储实现的引用计数
  return storage_impl_.use_count();
}

inline bool unique() const {
  // 检查存储实现是否是唯一的（独占）
  return storage_impl_.unique();
}

bool is_alias_of(const Storage& other) const {
  // 检查当前存储是否与另一个存储共享数据
  return (
      storage_impl_ == other.storage_impl_ ||
      isSharedStorageAlias(*this, other));
}

void UniqueStorageShareExternalPointer(
    void* src,
    size_t capacity,
    DeleterFnPtr d = nullptr) {
  // 在满足特定条件时，共享外部指针到存储实现
  if (!storage_impl_.unique()) {
    TORCH_CHECK(
        false,
        "UniqueStorageShareExternalPointer can only be called when use_count == 1");
  }
  storage_impl_->UniqueStorageShareExternalPointer(src, capacity, d);
}

void UniqueStorageShareExternalPointer(
    at::DataPtr&& data_ptr,
    size_t capacity) {
  // 在满足特定条件时，共享移动语义的外部指针到存储实现
  if (!storage_impl_.unique()) {
    TORCH_CHECK(
        false,
        "UniqueStorageShareExternalPointer can only be called when use_count == 1");
  }
  storage_impl_->UniqueStorageShareExternalPointer(
      std::move(data_ptr), capacity);
}

protected:
c10::intrusive_ptr<StorageImpl> storage_impl_;
# 定义模板特化，处理类型为 c10::Storage 的 MaybeOwnedTraits 结构
template <>
struct MaybeOwnedTraits<c10::Storage> {
  // 指定所有权类型为 c10::Storage
  using owned_type = c10::Storage;
  // 指定借用类型为 c10::Storage
  using borrow_type = c10::Storage;

  // 创建借用类型对象，从给定的 owned_type 对象中
  static borrow_type createBorrow(const owned_type& from) {
    return borrow_type(borrow_type::unsafe_borrow_t{}, from);
  }

  // 将 rhs 的值赋给 lhs 的借用类型对象，并释放 lhs 原有的存储实现
  static void assignBorrow(borrow_type& lhs, const borrow_type& rhs) {
    lhs.unsafeReleaseStorageImpl();
    lhs = borrow_type(borrow_type::unsafe_borrow_t{}, rhs);
  }

  // 销毁借用类型对象，释放其存储实现（假设它已经是 +0 的状态）
  static void destroyBorrow(borrow_type& toDestroy) {
    toDestroy.unsafeReleaseStorageImpl(); // "leak" it, but it was already +0.
  }

  // 从借用类型对象 borrow 中获取其引用作为 owned_type 类型的常量引用
  static const owned_type& referenceFromBorrow(const borrow_type& borrow) {
    return borrow;
  }

  // 从借用类型对象 borrow 中获取其指针作为 owned_type 类型的常量指针
  static const owned_type* pointerFromBorrow(const borrow_type& borrow) {
    return &borrow;
  }

  // 调试用：判断借用类型对象 borrow 是否有效（始终返回 true）
  static bool debugBorrowIsValid(const borrow_type& /*borrow*/) {
    return true;
  }
};

# 定义模板特化，处理类型为 c10::Storage 的 ExclusivelyOwnedTraits 结构
template <>
struct ExclusivelyOwnedTraits<c10::Storage> {
  // 指定表示类型为 c10::Storage
  using repr_type = c10::Storage;
  // 指定指针类型为 c10::Storage*
  using pointer_type = c10::Storage*;
  // 指定常量指针类型为 const c10::Storage*
  using const_pointer_type = const c10::Storage*;

  // 返回一个表示空状态的 c10::Storage 对象
  static repr_type nullRepr() {
    return c10::Storage();
  }

  // 在指定位置就地创建一个 c10::Storage 对象
  template <class... Args>
  static repr_type createInPlace(Args&&... args) {
    return c10::Storage(std::forward<Args>(args)...);
  }

  // 将传入的 c10::Storage 对象 x 移动到表示类型对象中并返回
  static repr_type moveToRepr(c10::Storage&& x) {
    return std::move(x);
  }

  // 获取并返回传入的 c10::Storage 对象 x 的所有权，并将其设置为空状态
  static c10::Storage take(c10::Storage& x) {
    return std::move(x);
  }

  // 获取表示类型对象 x 的指针
  static pointer_type getImpl(repr_type& x) {
    return &x;
  }

  // 获取表示类型对象 x 的常量指针
  static const_pointer_type getImpl(const repr_type& x) {
    return &x;
  }
};

// 结束命名空间 c10
} // namespace c10
```