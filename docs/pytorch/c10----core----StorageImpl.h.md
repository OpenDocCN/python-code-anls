# `.\pytorch\c10\core\StorageImpl.h`

```py
#pragma once

#include <c10/core/Allocator.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/core/SymInt.h>
#include <c10/core/impl/COW.h>
#include <c10/core/impl/COWDeleter.h>
#include <c10/core/impl/PyObjectSlot.h>
#include <c10/macros/Export.h>
#include <c10/util/Exception.h>
#include <c10/util/UniqueVoidPtr.h>
#include <c10/util/intrusive_ptr.h>
#include <cstddef>
#include <utility>

namespace c10 {

// 声明了两个异常处理函数，用于在特定情况下报错或警告
C10_API void throwNullDataPtrError();
C10_API void warnDeprecatedDataPtr();

// 一个 StorageImpl 结构体表示张量的底层数据缓冲区。
// 这个概念继承自 Torch7 的原始代码库；虽然我们希望摆脱这个概念，
// 但实际上是一项艰巨的工作，目前还没有人去做。
//
// 注意：storage 应当唯一拥有数据指针；例如，
// 两个非空数据指针仅当它们来自同一个 storage 时才是别名。
// 技术上可以违反此不变量（例如，可以使用 at::from_blob 创建非拥有 StorageImpl），
// 但这会导致很多问题，包括：
//
// - 如果有两个 storage 指向同一数据，则普通的释放器会出错，
//   因为普通的释放器假设唯一拥有权，但如果有两个 storage 指向同一数据，
//   那么意味着存在某种共享所有权。因此，你的释放器实际上必须在内部执行某种引用计数操作。
// - Python 中的深拷贝依赖于 storage 的相等性而不是数据指针的相等性；
//   因此，如果有两个独立的 storage 指向相同的数据，那么在这种情况下数据将被复制（在此之前一个数据指针，之后两个数据指针）。
// - 版本计数不会正常工作，因为我们在 storage 层面上进行所有 VC 跟踪
//   （除非你明确地使用 detach 断开 VC 连接）；因为数据指针相同，所以突变是完全不被跟踪的。
struct C10_API StorageImpl : public c10::intrusive_ptr_target {
 public:
  struct use_byte_size_t {};

  // 构造函数，初始化 StorageImpl 实例
  StorageImpl(
      use_byte_size_t /*use_byte_size*/,
      SymInt size_bytes,           // 用于表示字节大小的符号整数
      at::DataPtr data_ptr,        // 数据指针，用于持有底层数据
      at::Allocator* allocator,    // 分配器，用于分配和释放内存
      bool resizable)              // 是否可调整大小的标志位
      : data_ptr_(std::move(data_ptr)),                // 初始化数据指针
        size_bytes_(std::move(size_bytes)),            // 初始化字节大小
        size_bytes_is_heap_allocated_(size_bytes_.is_heap_allocated()),  // 检查字节大小是否在堆上分配
        resizable_(resizable),                        // 初始化可调整大小标志
        received_cuda_(false),                        // 接收到 CUDA 标志，默认为 false
        allocator_(allocator) {                       // 初始化分配器
    if (resizable) {
      TORCH_INTERNAL_ASSERT(
          allocator_, "For resizable storage, allocator must be provided");  // 如果可调整大小，则分配器不能为空
    }
  // 调用 refresh_has_data_ptr_check 函数，用于更新数据指针的检查状态
  refresh_has_data_ptr_check();
}

// StorageImpl 类的构造函数，初始化一个存储实例
StorageImpl(
    use_byte_size_t /*use_byte_size*/,
    const SymInt& size_bytes,
    at::Allocator* allocator,
    bool resizable)
    : StorageImpl(  // 委托给另一个构造函数进行初始化
          use_byte_size_t(),  // 使用默认的 use_byte_size_t 参数进行初始化
          size_bytes,
          size_bytes.is_heap_allocated()  // 根据 size_bytes 是否在堆上分配决定分配策略
              ? allocator->allocate(0)  // 如果是堆上分配，分配空间大小为 0
              : allocator->allocate(size_bytes.as_int_unchecked()),  // 否则分配 size_bytes 所示大小的空间
          allocator,
          resizable) {}

// 禁用移动赋值运算符
StorageImpl& operator=(StorageImpl&& other) = delete;
// 禁用复制赋值运算符
StorageImpl& operator=(const StorageImpl&) = delete;
// 禁用默认构造函数
StorageImpl() = delete;
// 禁用移动构造函数
StorageImpl(StorageImpl&& other) = delete;
// 禁用复制构造函数
StorageImpl(const StorageImpl&) = delete;
// 默认析构函数
~StorageImpl() override = default;

// 重置存储实例的数据
void reset() {
  data_ptr_.clear();  // 清空数据指针
  size_bytes_ = 0;  // 将大小设置为 0
  size_bytes_is_heap_allocated_ = false;  // 标记大小未在堆上分配
}

// 释放存储实例的资源，不调用 release_resources 是不必要的
void release_resources() override {
  data_ptr_.clear();  // 清空数据指针
}

// 返回存储实例的字节数
size_t nbytes() const {
  // 检查 size_bytes 是否在堆上分配，若是则报错
  TORCH_CHECK(!size_bytes_is_heap_allocated_);
  return size_bytes_.as_int_unchecked();  // 返回存储实例的大小
}

// 返回符号化字节数
SymInt sym_nbytes() const {
  return size_bytes_;  // 返回存储实例的符号化大小
}

// 设置存储实例的字节数，以字节为单位
void set_nbytes(size_t size_bytes) {
  size_bytes_ = static_cast<int64_t>(size_bytes);  // 将传入的字节数转换为 int64_t 并设置
  size_bytes_is_heap_allocated_ = false;  // 标记大小未在堆上分配
}

// 设置存储实例的字节数，以符号化整数为单位
void set_nbytes(c10::SymInt size_bytes) {
  size_bytes_ = std::move(size_bytes);  // 使用移动语义设置符号化整数大小
}

// 返回存储实例是否可调整大小
bool resizable() const {
  return resizable_;  // 返回存储实例的可调整大小属性
}

// 返回存储实例的数据指针（只读）
const at::DataPtr& data_ptr() const {
  return data_ptr_;  // 返回存储实例的数据指针
}

// 返回存储实例的数据指针（可读写）
at::DataPtr& mutable_data_ptr() {
  if (C10_UNLIKELY(has_data_ptr_check_)) {
    if (throw_on_mutable_data_ptr_) {
      throwNullDataPtrError();  // 如果数据指针为空则抛出异常
    }
    if (warn_deprecated_on_mutable_data_ptr_) {
      warnDeprecatedDataPtr();  // 发出已弃用数据指针警告
    }
    maybe_materialize_cow();  // 可能需要创建写时复制的数据指针
  }
  return data_ptr_;  // 返回可读写的数据指针
}

// 返回存储实例的数据指针，绕过所有检查
at::DataPtr& _mutable_data_ptr_no_checks() {
  return data_ptr_;  // 返回不经检查的数据指针
}

// 设置存储实例的数据指针，并返回之前的数据指针
at::DataPtr set_data_ptr(at::DataPtr&& data_ptr) {
  // 需要先确保旧的 COW 数据指针已实现，因为它被返回为可变的
  maybe_materialize_cow();
  return set_data_ptr_no_materialize_cow(std::move(data_ptr));  // 设置新的数据指针并返回旧的数据指针
}

// 设置存储实例的数据指针，不交换
void set_data_ptr_noswap(at::DataPtr&& data_ptr) {
  data_ptr_ = std::move(data_ptr);  // 设置新的数据指针
  refresh_has_data_ptr_check();  // 刷新数据指针检查状态
}

// 返回存储实例的数据指针（只读）
const void* data() const {
  return data_ptr_.get();  // 返回数据指针的指针
}

// 返回存储实例的数据指针（可读写）
void* mutable_data() {
  if (C10_UNLIKELY(has_data_ptr_check_)) {
    if (throw_on_mutable_data_ptr_) {
      throwNullDataPtrError();  // 如果数据指针为空则抛出异常
    }
    if (warn_deprecated_on_mutable_data_ptr_) {
      warnDeprecatedDataPtr();  // 发出已弃用数据指针警告
    }
    maybe_materialize_cow();  // 可能需要创建写时复制的数据指针
  }
  return data_ptr_.mutable_get();  // 返回可读写的数据指针的指针
}

// 返回存储实例所在的设备类型
at::DeviceType device_type() const {
  return data_ptr_.device().type();  // 返回数据指针所在的设备类型
}

// 返回存储实例的分配器
at::Allocator* allocator() {
  return allocator_;
}

// 返回当前张量使用的分配器
const at::Allocator* allocator() const {
  return allocator_;
}

// 通常不建议使用此方法，但在已分配张量后覆盖重新分配方法时偶尔有用
void set_allocator(at::Allocator* allocator) {
  allocator_ = allocator;
}

// 返回数据指针所在设备
Device device() const {
  return data_ptr_.device();
}

// 设置张量是否可调整大小
void set_resizable(bool resizable) {
  if (resizable) {
    // 只有当张量可调整大小时需要分配器
    AT_ASSERT(allocator_);
  }
  resizable_ = resizable;
}

/**
 * 仅在 use_count 为 1 时可调用
 */
void UniqueStorageShareExternalPointer(
    void* src,
    size_t size_bytes,
    DeleterFnPtr d = nullptr) {
  UniqueStorageShareExternalPointer(
      at::DataPtr(src, src, d, data_ptr_.device()), size_bytes);
}

/**
 * 仅在 use_count 为 1 时可调用
 */
void UniqueStorageShareExternalPointer(
    at::DataPtr&& data_ptr,
    size_t size_bytes) {
  data_ptr_ = std::move(data_ptr);
  size_bytes_ = static_cast<int64_t>(size_bytes);
  size_bytes_is_heap_allocated_ = false;
  allocator_ = nullptr;
  resizable_ = false;
}

// 此方法仅能在存储构造后调用，并且不能用于修改存储状态
void set_received_cuda(bool received_cuda) {
  received_cuda_ = received_cuda;
}

// 返回是否收到 CUDA 数据
bool received_cuda() {
  return received_cuda_;
}

// 返回 Python 对象槽的指针
impl::PyObjectSlot* pyobj_slot() {
  return &pyobj_slot_;
}

// 返回 Python 对象槽的指针（常量版本）
const impl::PyObjectSlot* pyobj_slot() const {
  return &pyobj_slot_;
}

// 设置在可变数据指针时抛出异常
void set_throw_on_mutable_data_ptr() {
  throw_on_mutable_data_ptr_ = true;
  refresh_has_data_ptr_check();
}

// 设置在可变数据指针时警告已弃用
void set_warn_deprecated_on_mutable_data_ptr() {
  warn_deprecated_on_mutable_data_ptr_ = true;
  refresh_has_data_ptr_check();
}

protected:
// materialize_cow_storage 需要调用 set_data_ptr_no_materlize_cow
friend void c10::impl::cow::materialize_cow_storage(StorageImpl& storage);

// 返回先前的 data_ptr。如果旧的 data_ptr 是 COW，这避免了实现它
at::DataPtr set_data_ptr_no_materialize_cow(at::DataPtr&& data_ptr) {
  at::DataPtr old_data_ptr(std::move(data_ptr_));
  data_ptr_ = std::move(data_ptr);
  refresh_has_data_ptr_check();
  return old_data_ptr;
}

private:
// 刷新 has_data_ptr_check，用于 COW 或在可变数据指针时抛出异常或警告已弃用时
void refresh_has_data_ptr_check() {
  has_data_ptr_check_ = is_cow() || throw_on_mutable_data_ptr_ ||
      warn_deprecated_on_mutable_data_ptr_;
}

// 判断是否为 COW 张量
inline bool is_cow() const {
  return c10::impl::cow::is_cow_data_ptr(data_ptr_);
}

// 如果是写时复制张量，则触发复制
void maybe_materialize_cow() {
  if (is_cow()) {
    impl::cow::materialize_cow_storage(*this);
  }
}
    }
  }  // 结束匿名命名空间

  DataPtr data_ptr_;  // 数据指针，可能指向不同类型的数据
  SymInt size_bytes_;  // 大小（字节数），使用符号整数表示
  bool size_bytes_is_heap_allocated_;  // 大小是否在堆上分配
  bool resizable_;  // 是否可调整大小

  // 标识该存储对象是否来自另一个进程，没有本地的 CUDA 内存分配
  bool received_cuda_;

  // 所有对 data/data_ptr 调用的特殊检查都通过这个布尔值进行保护。
  // 这是为了性能考虑：.data/.data_ptr 调用通常在热路径上。
  bool has_data_ptr_check_ = false;

  // 当调用 mutable_data_ptr() 或 mutable_data() 时是否应该抛出异常。
  bool throw_on_mutable_data_ptr_ = false;

  // 当调用 mutable_data_ptr() 或 mutable_data() 时是否应该发出警告。
  bool warn_deprecated_on_mutable_data_ptr_ = false;

  Allocator* allocator_;  // 分配器对象指针，用于内存分配
  impl::PyObjectSlot pyobj_slot_;  // Python 对象槽
// 声明一个名为 StorageImplCreateHelper 的类型别名，该别名表示一个函数指针类型，
// 该函数接受 StorageImpl::use_byte_size_t、SymInt 类型的参数 size_bytes、DataPtr 类型的参数 data_ptr、
// Allocator 指针类型的参数 allocator、以及一个布尔值 resizable，并返回一个 intrusive_ptr<StorageImpl>。
using StorageImplCreateHelper = intrusive_ptr<StorageImpl> (*)(
    StorageImpl::use_byte_size_t,
    SymInt size_bytes,
    DataPtr data_ptr,
    Allocator* allocator,
    bool resizable);

// 设置 StorageImplCreateHelper 类型的创建函数指针，与指定的设备类型相关联。
C10_API void SetStorageImplCreate(DeviceType t, StorageImplCreateHelper fptr);

// 获取与指定设备类型相关联的 StorageImplCreateHelper 类型的创建函数指针。
C10_API StorageImplCreateHelper GetStorageImplCreate(DeviceType t);

// 定义在 c10 命名空间内的 make_storage_impl 函数，用于创建 StorageImpl 对象。
C10_API c10::intrusive_ptr<c10::StorageImpl> make_storage_impl(
    c10::StorageImpl::use_byte_size_t use_byte_size,
    c10::SymInt size_bytes,
    c10::DataPtr data_ptr,
    c10::Allocator* allocator,
    bool resizable,
    std::optional<at::Device> device_opt);

// 结束 c10 命名空间的定义
} // namespace c10
```