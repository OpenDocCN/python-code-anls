# `.\pytorch\c10\cuda\CUDACachingAllocator.h`

```
#pragma once

#include <c10/core/Allocator.h>  // 引入内存分配器的核心功能
#include <c10/cuda/CUDAGraphsC10Utils.h>  // 引入与CUDA图形相关的C10实用工具
#include <c10/cuda/CUDAMacros.h>  // 引入CUDA宏定义
#include <c10/cuda/CUDAStream.h>  // 引入CUDA流对象
#include <c10/util/ApproximateClock.h>  // 引入近似时钟实用工具
#include <c10/util/Exception.h>  // 引入异常处理工具
#include <c10/util/Registry.h>  // 引入注册表功能

#include <array>  // 引入数组容器
#include <atomic>  // 引入原子操作支持
#include <cstddef>  // 引入标准尺寸类型支持
#include <cstdint>  // 引入标准整数类型支持
#include <functional>  // 引入函数对象功能
#include <memory>  // 引入内存管理工具
#include <string>  // 引入字符串处理功能
#include <unordered_set>  // 引入无序集合功能
#include <utility>  // 引入实用工具

namespace c10 {

// Caching allocator will execute every registered callback if it unable to find
// block inside of already allocated area.
// 缓存分配器将在无法在已分配区域内找到块时执行每个注册的回调函数。
class C10_CUDA_API FreeMemoryCallback {
 public:
  virtual ~FreeMemoryCallback() = default;  // 虚析构函数
  virtual bool Execute() = 0;  // 纯虚函数，用于执行回调操作
};

C10_DECLARE_REGISTRY(FreeCudaMemoryCallbacksRegistry, FreeMemoryCallback);  // 声明回调注册表

// 宏定义，用于注册释放内存回调函数
#define REGISTER_FREE_MEMORY_CALLBACK(name, ...) \
  C10_REGISTER_CLASS(FreeCudaMemoryCallbacksRegistry, name, __VA_ARGS__);

} // namespace c10

//
// TODO: Turn this into an honest to goodness class. I briefly attempted to do
// this, but it was a bit irritating to figure out how to also correctly
// apply pimpl pattern so I didn't have to leak any internal implementation
// details in the header (CUDACachingAllocator could be made a pimpl, but
// you also need to appropriately define a class which is a subclass
// of Allocator. Not impossible, but required a bit more surgery than
// I wanted to do at the time.)
//
// Why is this using a namespace rather than old-style THCCachingAllocator_
// prefix?  Mostly because it made the HIPify rules easier to write; _ is
// not counted as a word boundary, so you would otherwise have to list each
// of these functions.
//
// 将此部分转换为一个真正的类。我曾试图这样做，但要正确应用pimpl模式，以避免在头文件中泄露任何内部实现细节，这有点麻烦
// （CUDACachingAllocator可以被设为pimpl，但您还需要适当定义一个Allocator的子类。这不是不可能，但比我当时想做的要复杂一些）。
//
// 为什么要使用命名空间而不是旧式的THCCachingAllocator_前缀？主要是因为这样写HIPify规则更容易；
// 下划线不被视为单词边界，因此您否则必须列出每个函数。

namespace c10::cuda::CUDACachingAllocator {

extern const size_t kLargeBuffer;  // 声明一个常量 kLargeBuffer

struct Stat {
  int64_t current = 0;  // 当前使用量
  int64_t peak = 0;  // 峰值使用量
  int64_t allocated = 0;  // 已分配总量
  int64_t freed = 0;  // 已释放总量
};

enum struct StatType : uint64_t {
  AGGREGATE = 0,  // 总计类型的统计信息
  SMALL_POOL = 1,  // 小型池的统计信息
  LARGE_POOL = 2,  // 大型池的统计信息
  NUM_TYPES = 3 // 记得在添加新的统计类型时更新这个值
};

typedef std::array<Stat, static_cast<size_t>(StatType::NUM_TYPES)> StatArray;  // 统计信息数组类型定义

// Struct containing memory allocator summary statistics for a device.
// 结构体，包含设备内存分配器的总结统计信息。
// 结构体，用于存储设备的统计信息
struct DeviceStats {
  // COUNT: 客户端代码请求的分配次数统计
  StatArray allocation;
  // COUNT: 通过cudaMalloc()分配的段数统计
  StatArray segment;
  // COUNT: 活动内存块的数量（被分配或被流使用）
  StatArray active;
  // COUNT: 不活动的、分裂的内存块数量（未分配但无法通过cudaFree释放）
  StatArray inactive_split;

  // SUM: 由此内存分配器分配的字节数统计
  StatArray allocated_bytes;
  // SUM: 此内存分配器保留的字节数（包括空闲和已使用）
  StatArray reserved_bytes;
  // SUM: 活动内存块中的字节数统计
  StatArray active_bytes;
  // SUM: 不活动的、分裂的内存块中的字节数统计
  StatArray inactive_split_bytes;
  // SUM: 客户端代码请求的字节数统计
  StatArray requested_bytes;

  // COUNT: CUDA malloc调用失败导致缓存刷新的总次数统计
  int64_t num_alloc_retries = 0;

  // COUNT: OOM（内存耗尽）总次数统计（即在缓存刷新后的CUDA调用失败次数）
  int64_t num_ooms = 0;

  // COUNT: 从池中分配的超大块的总数统计
  Stat oversize_allocations;

  // COUNT: 需要malloc的超大块的总数统计
  Stat oversize_segments;

  // COUNT: synchronize_and_free_events()调用总次数统计
  int64_t num_sync_all_streams = 0;

  // COUNT: CUDA分配调用总次数统计，包括cuMemMap和cudaMalloc
  int64_t num_device_alloc = 0;

  // COUNT: CUDA释放调用总次数统计，包括cuMemUnmap和cudaFree
  int64_t num_device_free = 0;

  // SIZE: 允许分裂的最大块大小
  int64_t max_split_size = 0;
};

// CreateContextFn类型定义为指向创建GatheredContext共享指针的函数
typedef std::shared_ptr<GatheredContext> (*CreateContextFn)();

// 表示分配块信息的结构体（即cudaMalloc的分数部分）
struct BlockInfo {
  size_t size = 0; // 块的大小
  size_t requested_size = 0; // 请求的块大小
  int32_t gc_counter = 0; // 垃圾回收计数器
  bool allocated = false; // 是否已分配
  bool active = false; // 是否活跃
  std::shared_ptr<GatheredContext>
      context_when_allocated; // 分配时的上下文（每个观察者的上下文）
};

// 表示内存段信息的结构体（即一个连续的cudaMalloc）
struct SegmentInfo {
  c10::DeviceIndex device = 0; // 设备索引
  size_t address = 0; // 地址
  size_t total_size = 0; // 总大小
  size_t requested_size = 0; // 请求的大小（未取整，实际请求的大小）
  size_t allocated_size = 0; // 已分配的大小
  size_t active_size = 0; // 活跃的大小
  cudaStream_t stream = nullptr; // CUDA流
  bool is_large = false; // 是否为大块
  bool is_expandable = false; // 是否可扩展
  MempoolId_t owner_private_pool_id = {0, 0}; // 拥有者私有池ID
  std::vector<BlockInfo> blocks; // 块信息向量
  std::shared_ptr<GatheredContext> context_when_allocated; // 分配时的上下文
};

// 表示分配器状态的结构体
struct AllocatorState {
  virtual ~AllocatorState() = default; // 虚析构函数
};

// 用于时间跟踪的联合体
union trace_time_ {
  time_t t_; // 时间戳
  approx_time_t approx_t_; // 近似时间
};

// 跟踪条目的结构体
struct TraceEntry {
  enum Action {
    ALLOC, // 向缓存分配器请求新内存的API调用
    FREE_REQUESTED, // 向缓存分配器请求释放内存的API调用
    // ...
    FREE_COMPLETED, // 分配器可能需要延迟释放，因为内存仍然通过record_stream在另一个流中使用。这个事件在实际完成释放时生成。
    SEGMENT_ALLOC, // 调用cudaMalloc从操作系统获取更多内存
    SEGMENT_FREE, // 调用cudaFree将内存返回给操作系统（例如用于碎片整理或清空缓存）
    SEGMENT_MAP, // 调用cuMemMap（与可扩展段一起使用）
    SEGMENT_UNMAP, // 取消映射段的一部分（与可扩展段一起使用）
    SNAPSHOT, // 调用snapshot，用于将内存快照与跟踪事件关联
    OOM, // 分配器抛出OutOfMemoryError（addr_表示cuda报告的空闲字节数）
    USER_DEFINED // 来自用户定义API的调用，如record_function
  };
  TraceEntry(
      Action action, // 操作类型
      c10::DeviceIndex device, // 设备索引
      size_t addr, // 地址（对于OOM，这是cuda报告的空闲字节数）
      size_t size, // 大小
      cudaStream_t stream, // CUDA流
      approx_time_t time, // 时间戳
      std::shared_ptr<GatheredContext> context = nullptr) // 收集的上下文（可选）
      : action_(action),
        device_(device),
        addr_(addr),
        context_(std::move(context)),
        stream_(stream),
        size_(size) {
    time_.approx_t_ = time; // 设置时间戳
  }
  Action action_; // 跟踪事件的操作类型
  c10::DeviceIndex device_; // 设备索引
  size_t addr_; // 对于OOM，这是cuda报告的空闲字节数
  std::shared_ptr<GatheredContext> context_; // 收集的上下文
  cudaStream_t stream_{}; // CUDA流
  size_t size_; // 大小
  trace_time_ time_{}; // 时间戳
};

// 定义 AllocatorConfigInfo 结构体，包含内存管理配置信息
struct AllocatorConfigInfo {
  double garbage_collection_threshold;  // 垃圾回收阈值
  size_t max_split_size;                // 最大分割大小
  size_t pinned_num_register_threads;   // 固定注册线程数
  bool expandable_segments;             // 是否可扩展段
  bool release_lock_on_malloc;          // 在 malloc 时释放锁
  bool pinned_use_host_register;        // 固定使用主机寄存器
  std::string last_allocator_settings;  // 上次分配器设置
  std::vector<size_t> roundup_power2_divisions;  // 2 的幂次方舍入分割
};

// 定义 SnapshotInfo 结构体，包含快照信息
struct SnapshotInfo {
  std::vector<SegmentInfo> segments;                  // 段信息向量
  std::vector<std::vector<TraceEntry>> device_traces; // 设备追踪信息向量的向量
  AllocatorConfigInfo config_metadata;                // 内存分配器配置元数据
};

// 定义 CheckpointDelta 结构体，描述检查点增量
// 返回释放在池中的指针和分配的数据指针
struct CheckpointDelta {
  std::vector<void*> ptrs_freed;     // 被释放的指针
  std::vector<at::DataPtr> dataptrs_allocd;  // 已分配的数据指针
};

// 枚举类型 RecordContext，表示记录上下文
enum struct RecordContext {
  NEVER = 0,    // 从不记录
  STATE = 1,    // 仅保留活跃分配的堆栈
  ALLOC = 2,    // 另外记录跟踪历史中的分配的堆栈
  ALL = 3,      // 另外记录释放时的堆栈
};

// 定义 format_size 函数，用于将大小格式化为字符串
std::string format_size(uint64_t size);

// 定义 OutOfMemoryObserver 类型，是一个函数对象，处理内存不足的观察器
using OutOfMemoryObserver = std::function<void(
    int64_t device,
    size_t allocated,
    size_t device_total,
    size_t device_free)>;

// 定义 AllocatorTraceTracker 类型，是一个函数对象，用于跟踪内存分配器的追踪
using AllocatorTraceTracker = std::function<void(const TraceEntry&)>;

// 定义 CUDAAllocator 类，继承自 Allocator 类
class CUDAAllocator : public Allocator {
 public:
  // 纯虚函数，分配指定字节数的原始内存
  virtual void* raw_alloc(size_t nbytes) = 0;
  // 纯虚函数，分配指定字节数的原始内存，并指定 CUDA 流
  virtual void* raw_alloc_with_stream(size_t nbytes, cudaStream_t stream) = 0;
  // 纯虚函数，释放指定的原始内存
  virtual void raw_delete(void* ptr) = 0;
  // 纯虚函数，初始化函数
  virtual void init(int device_count) = 0;
  // 纯虚函数，是否已初始化
  virtual bool initialized() = 0;
  // 纯虚函数，设置内存分配的分数比例
  virtual void setMemoryFraction(double fraction, c10::DeviceIndex device) = 0;
  // 纯虚函数，清空缓存
  virtual void emptyCache() = 0;
  // 纯虚函数，获取缓存信息
  virtual void cacheInfo(c10::DeviceIndex device, size_t* largestBlock) = 0;
  // 纯虚函数，获取基础分配的指针和大小
  virtual void* getBaseAllocation(void* ptr, size_t* size) = 0;
  // 纯虚函数，记录流的操作
  virtual void recordStream(const DataPtr&, CUDAStream stream) = 0;
  // 纯虚函数，获取设备统计信息
  virtual DeviceStats getDeviceStats(c10::DeviceIndex device) = 0;
  // 纯虚函数，重置累积的统计信息
  virtual void resetAccumulatedStats(c10::DeviceIndex device) = 0;
  // 纯虚函数，重置峰值统计信息
  virtual void resetPeakStats(c10::DeviceIndex device) = 0;
  // 纯虚函数，生成快照信息
  virtual SnapshotInfo snapshot() = 0;
  // 纯虚函数，开始将分配到池中
  virtual void beginAllocateToPool(
      c10::DeviceIndex device,
      MempoolId_t mempool_id,
      std::function<bool(cudaStream_t)> filter) = 0;
  // 纯虚函数，结束将分配到池中
  virtual void endAllocateToPool(
      c10::DeviceIndex device,
      MempoolId_t mempool_id) = 0;
  // 纯虚函数，释放池中的内存
  virtual void releasePool(c10::DeviceIndex device, MempoolId_t mempool_id) = 0;
  // 虚函数，默认实现为抛出错误
  // 检查池中的活跃分配是否与预期的活跃分配相等
  virtual bool checkPoolLiveAllocations(
      c10::DeviceIndex device,
      MempoolId_t mempool_id,
      const std::unordered_set<void*>& expected_live_allocations) {
    TORCH_CHECK(
        false,
        name(),
        " does not yet support checkPoolLiveAllocations. "
        "If you need it, please file an issue describing your use case.");
  }
  // 纯虚函数，获取 IPC 设备指针的共享指针
  virtual std::shared_ptr<void> getIpcDevPtr(std::string handle) = 0;
  // 虚函数，默认实现返回是否启用历史记录
  virtual bool isHistoryEnabled() {
    // 如果需要支持检查池中活跃分配，请提交问题以便进一步讨论
    TORCH_CHECK(false,
                name(),
                " does not yet support checkPoolLiveAllocations. "
                "If you need it, please file an issue describing your use case.");
  }
};
    // 使用 TORCH_CHECK 函数来验证条件，如果条件为 false，则抛出错误信息，包含模块名称，提示不支持记录历史操作，建议提交问题描述以解决。
    TORCH_CHECK(
        false,
        name(),
        " does not yet support recordHistory. "
        "If you need it, please file an issue describing your use case.");
  }

  // 虚函数，用于启用或禁用记录历史操作功能，可以指定上下文记录器、分配跟踪最大条目数和记录时机
  virtual void recordHistory(
      bool enabled,
      CreateContextFn context_recorder,
      size_t alloc_trace_max_entries,
      RecordContext when) = 0;

  // 虚函数，用于记录注解，接受一个共享指针作为参数
  virtual void recordAnnotation(const std::shared_ptr<GatheredContext>& name){};

  // 虚函数，用于附加内存不足观察器，接受一个观察器对象作为参数
  virtual void attachOutOfMemoryObserver(OutOfMemoryObserver observer) = 0;

  // 虚函数，用于附加分配器跟踪器回调，回调函数在设备分配器锁定时调用
  // 回调中不能再获取额外的锁，尤其是不能获取 Python GIL，否则可能导致死锁
  virtual void attachAllocatorTraceTracker(AllocatorTraceTracker tracker) = 0;

  // 虚函数，用于启用设备间对等访问，指定两个设备索引
  virtual void enablePeerAccess(
      c10::DeviceIndex dev,
      c10::DeviceIndex dev_to_access) = 0;

  // 虚函数，用于异步内存拷贝，根据是否启用对等访问选择适当的内存拷贝方式
  virtual cudaError_t memcpyAsync(
      void* dst,
      int dstDevice,
      const void* src,
      int srcDevice,
      size_t count,
      cudaStream_t stream,
      bool p2p_enabled) = 0;

  // 虚函数，获取指定设备和内存池 ID 的检查点状态
  virtual std::shared_ptr<AllocatorState> getCheckpointState(
      c10::DeviceIndex device,
      MempoolId_t id) = 0;

  // 虚函数，设置指定设备的内存池状态检查点
  virtual CheckpointDelta setCheckpointPoolState(
      c10::DeviceIndex device,
      std::shared_ptr<AllocatorState> pps) = 0;

  // 虚函数，返回当前对象的名称字符串
  virtual std::string name() = 0;
// Allocator object, statically initialized
// See BackendInitializer in CUDACachingAllocator.cpp.
// Atomic loads on x86 are just normal loads,
// (atomic stores are different), so reading this value
// is no different than loading a pointer.
// 分配器对象，静态初始化
// 参见 CUDACachingAllocator.cpp 中的 BackendInitializer。
// 在 x86 上，原子加载只是普通加载，
// （原子存储是不同的），因此读取这个值
// 和加载指针没有区别。
C10_CUDA_API extern std::atomic<CUDAAllocator*> allocator;

// 获取当前的 CUDAAllocator 指针
inline CUDAAllocator* get() {
  return allocator.load();
}

// 直接被客户端调用。
// 分配指定大小的内存空间
inline void* raw_alloc(size_t nbytes) {
  return get()->raw_alloc(nbytes);
}

// 分配指定大小的内存空间，并指定 CUDA 流
inline void* raw_alloc_with_stream(size_t nbytes, cudaStream_t stream) {
  return get()->raw_alloc_with_stream(nbytes, stream);
}

// 释放由 raw_alloc 分配的内存空间
inline void raw_delete(void* ptr) {
  return get()->raw_delete(ptr);
}

// 初始化 CUDAAllocator，设置设备数量
inline void init(int device_count) {
  return get()->init(device_count);
}

// 设置指定设备上的内存使用比例
inline void setMemoryFraction(double fraction, c10::DeviceIndex device) {
  return get()->setMemoryFraction(fraction, device);
}

// 清空缓存
inline void emptyCache() {
  return get()->emptyCache();
}

// 获取指定设备上的缓存信息
inline void cacheInfo(c10::DeviceIndex device, size_t* largestBlock) {
  return get()->cacheInfo(device, largestBlock);
}

// 获取指针所指内存的基本分配信息
inline void* getBaseAllocation(void* ptr, size_t* size) {
  return get()->getBaseAllocation(ptr, size);
}

// 记录数据指针在 CUDA 流上的操作
inline void recordStream(const DataPtr& dataPtr, CUDAStream stream) {
  return get()->recordStream(dataPtr, stream);
}

// 获取指定设备的设备统计信息
inline DeviceStats getDeviceStats(c10::DeviceIndex device) {
  return get()->getDeviceStats(device);
}

// 重置指定设备上已累积的统计信息
inline void resetAccumulatedStats(c10::DeviceIndex device) {
  return get()->resetAccumulatedStats(device);
}

// 重置指定设备上的峰值统计信息
inline void resetPeakStats(c10::DeviceIndex device) {
  return get()->resetPeakStats(device);
}

// 创建当前状态的快照信息
inline SnapshotInfo snapshot() {
  return get()->snapshot();
}

// 获取指定设备和内存池 ID 的检查点状态
inline std::shared_ptr<AllocatorState> getCheckpointState(
    c10::DeviceIndex device,
    MempoolId_t id) {
  return get()->getCheckpointState(device, id);
}

// 设置指定设备和内存池 ID 的检查点池状态
inline CheckpointDelta setCheckpointPoolState(
    c10::DeviceIndex device,
    std::shared_ptr<AllocatorState> pps) {
  return get()->setCheckpointPoolState(device, std::move(pps));
}

// 开始分配到内存池的操作，指定设备、内存池 ID 和过滤器函数
inline void beginAllocateToPool(
    c10::DeviceIndex device,
    MempoolId_t mempool_id,
    std::function<bool(cudaStream_t)> filter) {
  get()->beginAllocateToPool(device, mempool_id, std::move(filter));
}

// 结束指定设备和内存池 ID 的分配操作
inline void endAllocateToPool(c10::DeviceIndex device, MempoolId_t mempool_id) {
  get()->endAllocateToPool(device, mempool_id);
}

// 记录历史分配信息，包括是否启用、上下文记录函数、最大跟踪条目数和记录时机
inline void recordHistory(
    bool enabled,
    CreateContextFn context_recorder,
    size_t alloc_trace_max_entries,
    RecordContext when) {
  return get()->recordHistory(
      enabled, context_recorder, alloc_trace_max_entries, when);
}

// 记录注解信息
inline void recordAnnotation(const std::shared_ptr<GatheredContext>& name) {
  return get()->recordAnnotation(name);
}

// 检查历史记录是否启用
inline bool isHistoryEnabled() {
  return get()->isHistoryEnabled();
}

// 检查指定设备和内存池 ID 的活动分配情况
inline bool checkPoolLiveAllocations(
    c10::DeviceIndex device,
    MempoolId_t mempool_id,
    // 调用当前对象的 get 方法，获取指向资源的指针，并调用其 checkPoolLiveAllocations 方法
    return get()->checkPoolLiveAllocations(
        // 将设备对象传递给 checkPoolLiveAllocations 方法，用于检查内存池中的分配情况
        device, 
        // 将内存池 ID 传递给 checkPoolLiveAllocations 方法，指定要检查的内存池
        mempool_id, 
        // 将预期的活动分配情况传递给 checkPoolLiveAllocations 方法，作为检查的依据
        expected_live_allocations);
}

// 定义一个内联函数，用于将内存耗尽观察器附加到当前 CUDA 分配器实例上
inline void attachOutOfMemoryObserver(OutOfMemoryObserver observer) {
  return get()->attachOutOfMemoryObserver(std::move(observer));
}

// 定义一个内联函数，用于将内存分配器跟踪器附加到当前 CUDA 分配器实例上
inline void attachAllocatorTraceTracker(AllocatorTraceTracker tracker) {
  return get()->attachAllocatorTraceTracker(std::move(tracker));
}

// 定义一个内联函数，用于释放指定设备上指定内存池的内存
inline void releasePool(c10::DeviceIndex device, MempoolId_t mempool_id) {
  return get()->releasePool(device, mempool_id);
}

// 如果不属于 CUDA_ALLOCATOR_BACKEND_INTERFACE，则定义一个内联函数，用于获取 IPC 设备指针
inline std::shared_ptr<void> getIpcDevPtr(std::string handle) {
  return get()->getIpcDevPtr(std::move(handle));
}

// 定义一个内联函数，用于获取当前 CUDA 分配器实例的名称
inline std::string name() {
  return get()->name();
}

// 定义一个内联函数，用于在 CUDA 设备之间异步复制内存数据
inline cudaError_t memcpyAsync(
    void* dst,
    int dstDevice,
    const void* src,
    int srcDevice,
    size_t count,
    cudaStream_t stream,
    bool p2p_enabled) {
  return get()->memcpyAsync(
      dst, dstDevice, src, srcDevice, count, stream, p2p_enabled);
}

// 定义一个内联函数，用于启用 CUDA 设备之间的对等访问
inline void enablePeerAccess(
    c10::DeviceIndex dev,
    c10::DeviceIndex dev_to_access) {
  return get()->enablePeerAccess(dev, dev_to_access);
}

// 结束命名空间 c10::cuda::CUDACachingAllocator
} // namespace c10::cuda::CUDACachingAllocator
```