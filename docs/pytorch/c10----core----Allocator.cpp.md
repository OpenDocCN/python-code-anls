# `.\pytorch\c10\core\Allocator.cpp`

```py
// 包含 C10 核心模块的 Allocator 头文件
#include <c10/core/Allocator.h>

// 包含 C10 工具中的 ThreadLocalDebugInfo 头文件
#include <c10/util/ThreadLocalDebugInfo.h>

// 进入 C10 命名空间
namespace c10 {

// 实现 Allocator 类中的 clone 方法，用于复制数据
DataPtr Allocator::clone(const void* data, std::size_t n) {
  // 分配新的数据内存空间
  DataPtr new_data = allocate(n);
  // 将原始数据复制到新分配的内存中
  copy_data(new_data.mutable_get(), data, n);
  // 返回包含新数据的 DataPtr 对象
  return new_data;
}

// 实现 Allocator 类中的默认数据复制方法
void Allocator::default_copy_data(
    void* dest,
    const void* src,
    std::size_t count) const {
  // 使用 memcpy 函数将 src 指向的数据复制到 dest 指向的内存空间
  std::memcpy(dest, src, count);
}

// 判断给定的 DataPtr 是否为简单数据指针
bool Allocator::is_simple_data_ptr(const DataPtr& data_ptr) const {
  // 判断 DataPtr 对象的指针和上下文是否相同
  return data_ptr.get() == data_ptr.get_context();
}

// 静态函数，用于删除非高效的 std::function 上下文对象
static void deleteInefficientStdFunctionContext(void* ptr) {
  // 将指针转换为 InefficientStdFunctionContext 类型，并释放其内存
  delete static_cast<InefficientStdFunctionContext*>(ptr);
}

// 在 InefficientStdFunctionContext 类中创建 DataPtr 对象
at::DataPtr InefficientStdFunctionContext::makeDataPtr(
    void* ptr,
    std::function<void(void*)> deleter,
    Device device) {
  // 返回一个 DataPtr 对象，包含指针 ptr 和对应的删除器
  return {
      ptr,
      new InefficientStdFunctionContext(ptr, std::move(deleter)),
      &deleteInefficientStdFunctionContext,
      device};
}

// 定义全局的 Allocator 指针数组，用于不同设备类型的内存分配器
C10_API at::Allocator* allocator_array[at::COMPILE_TIME_MAX_DEVICE_TYPES];

// 定义全局的优先级数组，用于不同设备类型的内存分配器优先级
C10_API uint8_t allocator_priority[at::COMPILE_TIME_MAX_DEVICE_TYPES] = {0};

// 设置特定设备类型的内存分配器和优先级
void SetAllocator(at::DeviceType t, at::Allocator* alloc, uint8_t priority) {
  // 如果给定优先级高于当前记录的优先级，则更新对应设备类型的分配器和优先级
  if (priority >= allocator_priority[static_cast<int>(t)]) {
    allocator_array[static_cast<int>(t)] = alloc;
    allocator_priority[static_cast<int>(t)] = priority;
  }
}

// 获取特定设备类型的内存分配器
at::Allocator* GetAllocator(const at::DeviceType& t) {
  // 获取特定设备类型的分配器，并确保其不为空
  auto* alloc = allocator_array[static_cast<int>(t)];
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(alloc, "Allocator for ", t, " is not set.");
  return alloc;
}

// 检查内存分析功能是否启用
bool memoryProfilingEnabled() {
  // 获取当前线程的内存报告信息，并检查是否启用了内存分析
  auto* reporter_ptr = static_cast<MemoryReportingInfoBase*>(
      ThreadLocalDebugInfo::get(DebugInfoKind::PROFILER_STATE));
  return reporter_ptr && reporter_ptr->memoryProfilingEnabled();
}

// 将内存使用情况报告给性能分析器
void reportMemoryUsageToProfiler(
    void* ptr,
    int64_t alloc_size,
    size_t total_allocated,
    size_t total_reserved,
    Device device) {
  // 获取当前线程的内存报告信息，并将内存使用情况报告给性能分析器
  auto* reporter_ptr = static_cast<MemoryReportingInfoBase*>(
      ThreadLocalDebugInfo::get(DebugInfoKind::PROFILER_STATE));
  if (reporter_ptr) {
    reporter_ptr->reportMemoryUsage(
        ptr, alloc_size, total_allocated, total_reserved, device);
  }
}

// 向性能分析器报告内存不足情况
void reportOutOfMemoryToProfiler(
    int64_t alloc_size,
    size_t total_allocated,
    size_t total_reserved,
    Device device) {
  // 获取当前线程的内存报告信息，并向性能分析器报告内存不足情况
  auto* reporter_ptr = static_cast<MemoryReportingInfoBase*>(
      ThreadLocalDebugInfo::get(DebugInfoKind::PROFILER_STATE));
  if (reporter_ptr) {
    reporter_ptr->reportOutOfMemory(
        alloc_size, total_allocated, total_reserved, device);
  }
}

// MemoryReportingInfoBase 类的默认构造函数实现
MemoryReportingInfoBase::MemoryReportingInfoBase() = default;

// 向性能分析器报告内存不足情况
void MemoryReportingInfoBase::reportOutOfMemory(
    int64_t /*alloc_size*/,
    size_t /*total_allocated*/,
    size_t /*total_reserved*/,
    Device /*device*/) {
  // 此方法暂无实际实现，留空
}
    size_t /*total_reserved*/,
    Device /*device*/) {}



// 定义函数，但未提供函数名和参数列表
    size_t /*total_reserved*/,
    // size_t 类型的参数 total_reserved，表示总共保留的大小
    Device /*device*/) {}
    // Device 类型的参数 device，表示设备信息，函数体为空
} // 结束 c10 命名空间
```