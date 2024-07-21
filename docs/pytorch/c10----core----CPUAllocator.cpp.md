# `.\pytorch\c10\core\CPUAllocator.cpp`

```
#include <c10/core/Allocator.h>
#include <c10/core/CPUAllocator.h>
#include <c10/core/DeviceType.h>
#include <c10/core/alignment.h>
#include <c10/core/impl/alloc_cpu.h>
#include <c10/mobile/CPUCachingAllocator.h>
#include <c10/mobile/CPUProfilingAllocator.h>
#include <c10/util/Logging.h>

// TODO: rename flag to C10
// 定义一个名为 caffe2_report_cpu_memory_usage 的布尔类型标志，控制是否打印详细的内存使用情况
C10_DEFINE_bool(
    caffe2_report_cpu_memory_usage,
    false,
    "If set, print out detailed memory usage");

namespace c10 {

// 定义 DefaultCPUAllocator 结构体，实现 at::Allocator 接口，用于 CPU 内存分配
struct C10_API DefaultCPUAllocator final : at::Allocator {
  DefaultCPUAllocator() = default;

  // 实现 allocate 方法，用于分配指定字节数的内存
  at::DataPtr allocate(size_t nbytes) override {
    void* data = nullptr;
    try {
      // 调用 c10::alloc_cpu 分配内存
      data = c10::alloc_cpu(nbytes);
    } catch (c10::Error& e) {
      // 如果分配失败，通过 profiledCPUMemoryReporter 报告内存不足
      profiledCPUMemoryReporter().OutOfMemory(nbytes);
      throw e;
    }
    // 调用 profiledCPUMemoryReporter 报告新分配的内存
    profiledCPUMemoryReporter().New(data, nbytes);
    // 返回 DataPtr，其中包括数据指针、原始数据指针和自定义的删除函数
    return {data, data, &ReportAndDelete, at::Device(at::DeviceType::CPU)};
  }

  // 静态方法 ReportAndDelete，用于报告并删除内存块
  static void ReportAndDelete(void* ptr) {
    if (!ptr) {
      return;
    }
    // 调用 profiledCPUMemoryReporter 报告内存删除
    profiledCPUMemoryReporter().Delete(ptr);
    // 调用 free_cpu 释放内存
    free_cpu(ptr);
  }

  // 实现 raw_deleter 方法，返回删除函数指针
  at::DeleterFnPtr raw_deleter() const override {
    return &ReportAndDelete;
  }

  // 实现 copy_data 方法，用于复制数据
  void copy_data(void* dest, const void* src, std::size_t count) const final {
    default_copy_data(dest, src, count);
  }
};

// 返回静态的 ProfiledCPUMemoryReporter 对象引用
ProfiledCPUMemoryReporter& profiledCPUMemoryReporter() {
  static ProfiledCPUMemoryReporter reporter_;
  return reporter_;
}

// 定义模板类 GuardedAllocator，用于 QNNPACK 和 XNNPACK 的安全内存分配
// 这个分配器在移动平台上是默认的，以确保访问超出边界的安全性
//
// PreGuardBytes: 分配前保护字节数
// PostGuardBytes: 分配后保护字节数
template <uint32_t PreGuardBytes, uint32_t PostGuardBytes>
class DefaultMobileCPUAllocator final : public at::Allocator {
 public:
  // 默认构造函数
  DefaultMobileCPUAllocator() = default;
  // 默认析构函数
  ~DefaultMobileCPUAllocator() override = default;

  // 静态方法：用于删除指针，支持内存分配器缓存和性能分析
  static void deleter(void* const pointer) {
    // 如果指针为空，直接返回
    if (C10_UNLIKELY(!pointer)) {
      return;
    }
    // TODO: 在移动设备上通过更好的线程局部存储支持启用
    // profiledCPUMemoryReporter().Delete(pointer);

    // 获取线程局部缓存分配器和性能分析分配器的指针
    auto allocator_ptr = GetThreadLocalCachingAllocator();
    auto profiling_allocator_ptr = GetThreadLocalProfilingAllocator();

    // 根据可用的分配器释放内存
    if (allocator_ptr != nullptr) {
      allocator_ptr->free(pointer);
    } else if (profiling_allocator_ptr != nullptr) {
      profiling_allocator_ptr->free(pointer);
    } else {
      // 默认情况下使用标准的 CPU 内存释放函数
      c10::free_cpu(pointer);
      // 当未启用缓存分配器时，释放内存会增加额外的成本
      // NOLINTNEXTLINE(clang-analyzer-unix.Malloc)
      CPUCachingAllocator::record_free(pointer);

      // 获取线程局部内存分配器，并记录释放的内存
      auto allocation_planner = GetThreadLocalAllocationPlanner();
      if (allocation_planner != nullptr) {
        allocation_planner->record_free(pointer);
      }
    }
  }

  // 分配指定大小的内存并返回 DataPtr 对象
  DataPtr allocate(const size_t nbytes) override {
    // 如果请求的字节数为零，返回一个空指针的 DataPtr 对象
    if (C10_UNLIKELY(0u == nbytes)) {
      return {
          nullptr,
          nullptr,
          &deleter,
          at::Device(DeviceType::CPU),
      };
    }

    // 计算实际分配的内存大小（包括前后的保护字节）
    auto alloc_size = PreGuardBytes + nbytes + PostGuardBytes;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    void* data;
    // 获取线程局部缓存分配器和性能分析分配器的指针
    auto allocator_ptr = GetThreadLocalCachingAllocator();
    auto profiling_allocator_ptr = GetThreadLocalProfilingAllocator();

    // 根据可用的分配器分配内存
    if (allocator_ptr != nullptr) {
      data = allocator_ptr->allocate(alloc_size);
    } else if (profiling_allocator_ptr != nullptr) {
      data = profiling_allocator_ptr->allocate(alloc_size);
    } else {
      // 使用标准的 CPU 内存分配函数
      try {
        data = c10::alloc_cpu(alloc_size);
      } catch (c10::Error& e) {
        // 内存分配失败时进行异常处理，并通知内存性能分析
        profiledCPUMemoryReporter().OutOfMemory(alloc_size);
        throw e;
      }
      // 获取线程局部内存分配器，并记录分配的内存
      auto allocation_planner = GetThreadLocalAllocationPlanner();
      if (allocation_planner != nullptr) {
        allocation_planner->record_allocation(alloc_size, data);
      }
    }
    // 通知内存性能分析分配了新的内存
    profiledCPUMemoryReporter().New(data, alloc_size);
    // 返回 DataPtr 对象，指向实际数据的起始位置（跳过前面的保护字节）
    return {
        reinterpret_cast<uint8_t*>(data) + PreGuardBytes,
        data,
        &deleter,
        at::Device(DeviceType::CPU),
    };
  }

  // 返回当前分配器的删除函数指针
  DeleterFnPtr raw_deleter() const override {
    return deleter;
  }

  // 检查指定的 DataPtr 是否指向简单数据块
  bool is_simple_data_ptr(const c10::DataPtr& data_ptr) const final {
    return reinterpret_cast<const uint8_t*>(data_ptr.get()) ==
        reinterpret_cast<const uint8_t*>(data_ptr.get_context()) +
        PreGuardBytes;
  }

  // 复制数据的函数，调用默认的数据复制函数
  void copy_data(void* dest, const void* src, std::size_t count) const final {
    default_copy_data(dest, src, count);
  }
};

// 返回 CPU 分配器的全局实例
void NoDelete(void*) {}

at::Allocator* GetCPUAllocator() {
  return GetAllocator(DeviceType::CPU);
}
// 设置 CPU 分配器，将指定的分配器和优先级传递给设备类型为 CPU 的分配器设置函数
void SetCPUAllocator(at::Allocator* alloc, uint8_t priority) {
  SetAllocator(DeviceType::CPU, alloc, priority);
}

// 移动 CPU 分配器必须始终存在，即使在非移动构建中也是如此，因为 QNNPACK 和 XNNPACK 不是移动特定的。
//
// 预防保护：8 字节用于 QNNPACK，但设置为 gAlignment 以确保 SIMD 对齐，
//            不是在分配的内存上，而是在返回给用户的内存位置上。
// 后防保护：16 字节用于 XNNPACK。
// NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,cppcoreguidelines-avoid-non-const-global-variables)
static DefaultMobileCPUAllocator<gAlignment, 16u> g_mobile_cpu_allocator;

// 获取默认的移动 CPU 分配器
at::Allocator* GetDefaultMobileCPUAllocator() {
  return &g_mobile_cpu_allocator;
}

#ifdef C10_MOBILE

// 获取默认的 CPU 分配器，用于移动设备
at::Allocator* GetDefaultCPUAllocator() {
  return GetDefaultMobileCPUAllocator();
}

// 将 g_mobile_cpu_allocator 注册为 CPU 分配器
REGISTER_ALLOCATOR(DeviceType::CPU, &g_mobile_cpu_allocator);

#else

// 全局默认的 CPU 分配器
static DefaultCPUAllocator g_cpu_alloc;

// 获取默认的 CPU 分配器
at::Allocator* GetDefaultCPUAllocator() {
  return &g_cpu_alloc;
}

// 将 g_cpu_alloc 注册为 CPU 分配器
REGISTER_ALLOCATOR(DeviceType::CPU, &g_cpu_alloc);

#endif /* C10_Mobile */

// CPU 内存分析报告器的新内存分配事件处理
void ProfiledCPUMemoryReporter::New(void* ptr, size_t nbytes) {
  if (nbytes == 0) {
    return;
  }
  auto profile_memory = memoryProfilingEnabled();
  size_t allocated = 0;
  if (FLAGS_caffe2_report_cpu_memory_usage || profile_memory) {
    // 使用互斥锁保护共享数据结构的操作
    std::lock_guard<std::mutex> guard(mutex_);
    // 将指针及其分配的字节数记录到 size_table_
    size_table_[ptr] = nbytes;
    // 更新当前已分配的总字节数
    allocated_ += nbytes;
    allocated = allocated_;
  }
  if (FLAGS_caffe2_report_cpu_memory_usage) {
    // 如果设置了内存使用报告标志，记录分配的内存信息
    LOG(INFO) << "C10 alloc " << nbytes << " bytes, total alloc " << allocated
              << " bytes.";
  }
  if (profile_memory) {
    // 如果启用了内存分析功能，向分析器报告内存使用情况
    reportMemoryUsageToProfiler(
        ptr,
        static_cast<int64_t>(nbytes),
        allocated,
        0,
        c10::Device(c10::DeviceType::CPU));
  }
}

// CPU 内存分析报告器的内存释放事件处理
void ProfiledCPUMemoryReporter::Delete(void* ptr) {
  size_t nbytes = 0;
  auto profile_memory = memoryProfilingEnabled();
  size_t allocated = 0;
  if (FLAGS_caffe2_report_cpu_memory_usage || profile_memory) {
    // 使用互斥锁保护共享数据结构的操作
    std::lock_guard<std::mutex> guard(mutex_);
    // 查找并移除指针对应的字节数记录
    auto it = size_table_.find(ptr);
    if (it != size_table_.end()) {
      allocated_ -= it->second;
      allocated = allocated_;
      nbytes = it->second;
      size_table_.erase(it);
    } else {
      // C10_LOG_EVERY_MS 可能在某些构建中每次记录，使用简单计数器以避免冗余日志
      if (log_cnt_++ % 1000 == 0) {
        LOG(WARNING) << "Memory block of unknown size was allocated before "
                     << "the profiling started, profiler results will not "
                     << "include the deallocation event";
      }
    }
  }
  if (nbytes == 0) {
    return;
  }
  if (FLAGS_caffe2_report_cpu_memory_usage) {
    // 如果设置了内存使用报告标志，记录释放的内存信息
    LOG(INFO) << "C10 deleted " << nbytes << " bytes, total alloc " << allocated
              << " bytes.";
  }
  if (profile_memory) {
    // 如果启用了内存分析功能，向分析器报告内存释放情况
    reportMemoryUsageToProfiler(
        ptr,
        static_cast<int64_t>(-static_cast<int64_t>(nbytes)),
        allocated,
        0,
        c10::Device(c10::DeviceType::CPU));
  }
}
    // 调用函数 reportMemoryUsageToProfiler，用于向性能分析器报告内存使用情况
    reportMemoryUsageToProfiler(
        // 参数1: 指针 ptr，用于标识内存位置
        ptr,
        // 参数2: 负的 nbytes，表示释放的内存大小（以字节为单位）
        -static_cast<int64_t>(nbytes),
        // 参数3: allocated，表示当前内存是否已分配
        allocated,
        // 参数4: 额外标志位，此处为0，通常表示默认情况
        0,
        // 参数5: c10::Device，指定内存所在设备，这里是CPU设备
        c10::Device(c10::DeviceType::CPU));
    }
}

// 定义 ProfiledCPUMemoryReporter 类的 OutOfMemory 方法，处理内存耗尽情况
void ProfiledCPUMemoryReporter::OutOfMemory(size_t nbytes) {
  // 检查是否启用了内存分析
  auto profile_memory = memoryProfilingEnabled();
  size_t allocated = 0;
  // 如果需要报告 CPU 内存使用或者启用了内存分析，则获取当前已分配的内存量
  if (FLAGS_caffe2_report_cpu_memory_usage || profile_memory) {
    // 使用互斥锁保护临界区
    std::lock_guard<std::mutex> guard(mutex_);
    allocated = allocated_;
  }
  // 如果请求分配的字节数为 0，则直接返回
  if (nbytes == 0) {
    return;
  }
  // 如果设置了 FLAGS_caffe2_report_cpu_memory_usage 标志，则记录内存不足信息
  if (FLAGS_caffe2_report_cpu_memory_usage) {
    LOG(INFO) << "C10 Out of Memory. Trying to allocate " << nbytes
              << " bytes, total alloc " << allocated << " bytes.";
  }
  // 如果启用了内存分析，则向分析器报告内存耗尽情况
  if (profile_memory) {
    reportOutOfMemoryToProfiler(
        static_cast<int64_t>(nbytes),
        allocated,
        0,
        c10::Device(c10::DeviceType::CPU));
  }
}

// 设置全局 CPU 缓存分配器及其优先级
C10_API at::Allocator* cpu_caching_alloc = nullptr;
C10_API uint8_t cpu_caching_alloc_priority = 0;

// 设置 CPU 缓存分配器的函数
void SetCPUCachingAllocator(Allocator* alloc, uint8_t priority) {
  // 如果传入的优先级大于等于当前优先级，则更新 CPU 缓存分配器及其优先级
  if (priority >= cpu_caching_alloc_priority) {
    cpu_caching_alloc = alloc;
    cpu_caching_alloc_priority = priority;
  }
}

// 获取当前设置的 CPU 缓存分配器
Allocator* GetCPUCachingAllocator() {
  // 如果 CPU 缓存分配器未设置，则记录警告信息，并返回默认分配器
  if (cpu_caching_alloc == nullptr) {
    VLOG(1)
        << "There is not caching allocator registered for CPU, use the default allocator instead.";
    return GetAllocator(DeviceType::CPU);
  }
  // 返回当前设置的 CPU 缓存分配器
  return cpu_caching_alloc;
}

// 结束 c10 命名空间的定义
} // namespace c10
```