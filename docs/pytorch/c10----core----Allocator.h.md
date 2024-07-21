# `.\pytorch\c10\core\Allocator.h`

```py
// 防止头文件重复包含，只在第一次包含时有效
#pragma once

// 包含标准库头文件
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>

// 包含C10库相关头文件
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/macros/Export.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <c10/util/ThreadLocalDebugInfo.h>
#include <c10/util/UniqueVoidPtr.h>

// 声明C10命名空间
namespace c10 {

// DataPtr是一个唯一指针（带有附加的删除器和一些删除器的上下文信息），
// 用于管理内存，并记录数据所在的设备类型。
class C10_API DataPtr {
 private:
  // 使用UniqueVoidPtr管理内存
  c10::detail::UniqueVoidPtr ptr_;
  // 记录数据所在的设备类型
  Device device_;

 public:
  // 默认构造函数，数据指针为空，设备类型为CPU
  DataPtr() : ptr_(), device_(DeviceType::CPU) {}
  
  // 指定数据指针和设备类型的构造函数
  DataPtr(void* data, Device device) : ptr_(data), device_(device) {}
  
  // 指定数据指针、上下文指针、删除器及设备类型的构造函数
  DataPtr(void* data, void* ctx, DeleterFnPtr ctx_deleter, Device device)
      : ptr_(data, ctx, ctx_deleter), device_(device) {}
  
  // 获取数据指针
  void* operator->() const {
    return ptr_.get();
  }
  
  // 清空数据指针
  void clear() {
    ptr_.clear();
  }
  
  // 获取数据指针
  void* get() const {
    return ptr_.get();
  }
  
  // 获取可变的数据指针
  void* mutable_get() {
    return ptr_.get();
  }
  
  // 获取上下文指针
  void* get_context() const {
    return ptr_.get_context();
  }
  
  // 释放上下文指针
  void* release_context() {
    return ptr_.release_context();
  }
  
  // 移动上下文指针的所有权
  std::unique_ptr<void, DeleterFnPtr>&& move_context() {
    return ptr_.move_context();
  }
  
  // 将数据指针转换为指定类型的指针
  template <typename T>
  T* cast_context(DeleterFnPtr expected_deleter) const {
    return ptr_.cast_context<T>(expected_deleter);
  }
  
  // 获取删除器函数指针
  DeleterFnPtr get_deleter() const {
    return ptr_.get_deleter();
  }
  
  // 隐式转换为布尔值，判断DataPtr是否有效
  operator bool() const {
    return static_cast<bool>(ptr_);
  }
};

} // namespace c10
    // 返回指向 ptr_ 的 deleter 函数的指针
    return ptr_.get_deleter();
  }
  /**
   * 比较 DataPtr 中的 deleter 与 expected_deleter 是否匹配。
   * 如果匹配，则将 deleter 替换为 new_deleter 并返回 true；
   * 否则不做任何操作并返回 false。
   *
   * 一般来说，不安全地无条件设置 DataPtr 的 deleter 是不安全的，
   * 因为你不知道 deleter 的具体内容，因此很难在不存储原始 deleter 的情况下
   * 正确处理 deleter 的释放（这很难做到，因为 DeleterFnPtr 不是闭包，
   * 并且 DataPtr 上下文只有一个字，通常没有足够的空间来同时存储原始
   * deleter 和其上下文）。然而，在某些情况下，你确切地知道 deleter 的内容，
   * 并且有一个新的 deleter 手动包装旧的 deleter。在这种情况下，
   * 可以安全地在断言 deleter 匹配后交换 deleter。
   *
   * 对于 new_deleter，有什么要求？它必须仍然正确地处理传入的 void* 指针，
   * 其中 void* 是原始 deleter 的上下文。因此，通常期望新的 deleter 看起来像这样：
   *
   *      [](void* ptr) {
   *        some_new_stuff(ptr);
   *        get_orig_allocator()->raw_deleter(ptr);
   *      }
   *
   * 注意，关闭原始 allocator 是行不通的；你没有足够的空间来这样做！
   * 同样，假设传入的指针是内存指针是不安全的；确保阅读 Allocator 的源代码
   * 以确认这一点。
   */
  C10_NODISCARD bool compare_exchange_deleter(
      DeleterFnPtr expected_deleter,
      DeleterFnPtr new_deleter) {
    // 调用 ptr_ 的 compare_exchange_deleter 方法比较并替换 deleter
    return ptr_.compare_exchange_deleter(expected_deleter, new_deleter);
  }
  Device device() const {
    // 返回当前 DataPtr 的设备信息
    return device_;
  }
  // 不安全地修改 DataPtr 上的设备信息。在正常使用中，通常不需要调用此函数。
  // 我们在实现详细的 Note [Masquerading as CUDA] 中需要使用此函数。
  void unsafe_set_device(Device device) {
    // 设置 DataPtr 的设备信息为指定的 device
    device_ = device;
  }
};

// NB: Device is NOT tested for here; a CUDA nullptr is as much a nullptr as a
// CPU nullptr
// 注意：此处不测试设备；CUDA 的 nullptr 和 CPU 的 nullptr 是一样的

inline bool operator==(const DataPtr& dp, std::nullptr_t) noexcept {
  return !dp;
}
// 比较操作符重载：判断 DataPtr 是否等于 nullptr

inline bool operator==(std::nullptr_t, const DataPtr& dp) noexcept {
  return !dp;
}
// 比较操作符重载：判断 nullptr 是否等于 DataPtr

inline bool operator!=(const DataPtr& dp, std::nullptr_t) noexcept {
  return dp;
}
// 比较操作符重载：判断 DataPtr 是否不等于 nullptr

inline bool operator!=(std::nullptr_t, const DataPtr& dp) noexcept {
  return dp;
}
// 比较操作符重载：判断 nullptr 是否不等于 DataPtr

// Note [raw_allocate/raw_deallocate and Thrust]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Thrust 的自定义分配器支持要求我们编写类似以下的代码：
//
//  class ThrustAllocator {
//    char* allocate(size_t);
//    void deallocate(char*, size_t);
//  };
//
// 这对基于 unique_ptr 的分配器接口并不友好，因为在释放时无法获取上下文。
//
// 然而，在某些情况下，上下文恰好与数据指针相同。这种情况下，我们可以支持“raw”
// 分配和释放接口。这就是 raw_deleter 的用途。默认情况下，它返回 nullptr，这意味着
// 未实现 raw 接口。务必在可能时实现它，否则会错误地报告 raw 接口不支持，实际上是支持的。

struct C10_API Allocator {
  virtual ~Allocator() = default;

  virtual DataPtr allocate(size_t n) = 0;
  // 分配 n 字节大小的内存，返回 DataPtr 对象

  // Clones an allocation that came from this allocator.
  //
  // To perform the copy, this function calls `copy_data`, which
  // must be implemented by derived classes.
  //
  // Note that this explicitly ignores any context that may have been
  // attached to the input data.
  //
  // Requires: input data was allocated by the same allocator.
  // 克隆由此分配器分配的内存。
  //
  // 调用 `copy_data` 来执行复制，这个函数必须由派生类实现。
  //
  // 注意：这明确忽略了附加到输入数据的任何上下文。
  //
  // 要求：输入数据必须由同一分配器分配。

  DataPtr clone(const void* data, std::size_t n);

  // Checks if DataPtr has a simple context, not wrapped with any out of the
  // ordinary contexts.
  // 检查 DataPtr 是否具有简单的上下文，没有任何非常规的包装。

  virtual bool is_simple_data_ptr(const DataPtr& data_ptr) const;

  // If this returns a non nullptr, it means that allocate()
  // is guaranteed to return a unique_ptr with this deleter attached;
  // it means the rawAllocate and rawDeallocate APIs are safe to use.
  // This function MUST always return the same BoundDeleter.
  //
  // 如果此函数返回非 nullptr，则表示 allocate() 调用会返回一个附带此 deleter 的 unique_ptr；
  // 这意味着 rawAllocate 和 rawDeallocate 接口可以安全使用。
  // 此函数必须始终返回相同的 BoundDeleter。

  virtual DeleterFnPtr raw_deleter() const {
    return nullptr;
  }

  // Allocate raw memory without any context.
  // 分配没有任何上下文的原始内存。

  void* raw_allocate(size_t n) {
    auto dptr = allocate(n);
    AT_ASSERT(dptr.get() == dptr.get_context());
    return dptr.release_context();
  }

  // Deallocate raw memory using the raw_deleter().
  // 使用 raw_deleter() 释放原始内存。

  void raw_deallocate(void* ptr) {
    auto d = raw_deleter();
    AT_ASSERT(d);
    // 断言确保 d 不为空指针
    // Note: Assertion assumes raw_deleter() always returns a valid deleter.
  }
};
    d(ptr);
  }



// 调用析构函数 `d` 来释放指针 `ptr` 指向的对象。
// 这里假设 `d` 是一个合适的析构函数或者类似的释放函数。
// 注意：没有提供 `d` 函数的具体实现，所以假设其功能是正确的。

// Copies data from one allocation to another.
// Pure virtual, so derived classes must define behavior.
// Derived class implementation can simply call `default_copy_data`
// to use `std::memcpy`.
//
// Requires: src and dest were allocated by this allocator
// Requires: src and dest both have length >= count
virtual void copy_data(void* dest, const void* src, std::size_t count) const = 0;



protected:
// Uses `std::memcpy` to copy data.
// Child classes can use this as `copy_data` when an alternative copy
// API is not needed.
void default_copy_data(void* dest, const void* src, std::size_t count) const;



// 使用 `std::memcpy` 来复制数据。
// 当派生类不需要使用替代的复制 API 时，可以使用这个作为 `copy_data` 的实现。
// 这个函数实现了基本的数据复制，适用于大多数情况。
// 结构体定义：InefficientStdFunctionContext，用于生成具有任意 std::function 删除器的 DataPtr。
// 在某些面向用户的函数中，提供一个接口，用于从外部数据构建张量，并接受任意 std::function 删除器。
// 在代码中搜索 InefficientStdFunctionContext 可找到这些出现的位置。
struct C10_API InefficientStdFunctionContext {
  void* ptr_;  // 指向数据的指针
  std::function<void(void*)> deleter_;  // 删除器函数对象，用于释放指针指向的资源

  // 构造函数，初始化指针和删除器函数对象
  InefficientStdFunctionContext(void* ptr, std::function<void(void*)> deleter)
      : ptr_(ptr), deleter_(std::move(deleter)) {}

  // 析构函数，释放指针指向的资源（如果存在删除器函数对象）
  ~InefficientStdFunctionContext() {
    if (deleter_) {
      deleter_(ptr_);
    }
  }

  // 静态成员函数，用于创建一个带有指定删除器的 DataPtr
  static DataPtr makeDataPtr(
      void* ptr,
      std::function<void(void*)> deleter,
      Device device);
};

/** 
 * 设置设备类型 `t` 的分配器。传入的分配器指针应具有静态生存期；此函数不接管原始指针的所有权。
 * 这样做的原因是防止调用 SetAllocator 时使得特定设备的分配器的现有指针无效。
 *
 * 注意，此函数不是线程安全的，并且假定该函数仅在初始化期间调用。
 *
 * 'priority' 标志用于在希望覆盖默认分配器时引入，因为分配器是静态设置的。默认优先级为 0，表示最低。
 * 只有具有更高或相等优先级的分配器才能覆盖现有分配器。
 */
C10_API void SetAllocator(DeviceType t, Allocator* alloc, uint8_t priority = 0);

/**
 * 获取设备类型 `t` 的分配器。
 */
C10_API Allocator* GetAllocator(const DeviceType& t);

// 模板结构体：AllocatorRegisterer<t>
template <DeviceType t>
struct AllocatorRegisterer {
  // 构造函数，注册特定设备类型 `t` 的分配器
  explicit AllocatorRegisterer(Allocator* alloc) {
    SetAllocator(t, alloc);
  }
};

// 宏定义：REGISTER_ALLOCATOR(t, f)
#define REGISTER_ALLOCATOR(t, f)                       \
  namespace {                                          \
  static c10::AllocatorRegisterer<t> g_allocator_d(f); \
  }

// 接口结构体：MemoryReportingInfoBase，用于报告每个设备的线程局部内存使用情况
struct C10_API MemoryReportingInfoBase : public c10::DebugInfoBase {
  MemoryReportingInfoBase();
  ~MemoryReportingInfoBase() override = default;

  /**
   * 报告内存使用情况。
   *
   * ptr：指向分配的内存块的指针。
   * alloc_size：指针分配的大小。
   * total_allocated：总分配的内存大小。
   * total_reserved：内存池的总大小，包括已使用和未使用的部分（如果适用）。
   * device：设备类型。
   */
  virtual void reportMemoryUsage(
      void* ptr,
      int64_t alloc_size,
      size_t total_allocated,
      size_t total_reserved,
      Device device) = 0;

  /**
   * 报告内存不足情况。
   *
   * alloc_size：分配请求的大小。
   * total_allocated：总分配的内存大小。
   * total_reserved：内存池的总大小，包括已使用和未使用的部分（如果适用）。
   * device：设备类型。
   */
  virtual void reportOutOfMemory(
      int64_t alloc_size,
      size_t total_allocated,
      size_t total_reserved,
      Device device);

  /**
   * 返回内存分析是否启用。
   */
  virtual bool memoryProfilingEnabled() const = 0;
};

// 返回内存分析是否启用。
C10_API bool memoryProfilingEnabled();
// 报告内存使用情况给性能分析器，记录指针、分配大小、总分配量、总保留量、设备信息
C10_API void reportMemoryUsageToProfiler(
    void* ptr,
    int64_t alloc_size,
    size_t total_allocated,
    size_t total_reserved,
    Device device);

// 报告内存耗尽情况给性能分析器，记录分配大小、总分配量、总保留量、设备信息
C10_API void reportOutOfMemoryToProfiler(
    int64_t alloc_size,
    size_t total_allocated,
    size_t total_reserved,
    Device device);

// 用于在分配器中收集追溯信息的结构
struct GatheredContext {
  virtual ~GatheredContext() = default;
};

// c10 命名空间的结束位置
} // namespace c10
```