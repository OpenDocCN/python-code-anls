# `.\pytorch\c10\cuda\CUDAMallocAsyncAllocator.cpp`

```py
// 引入所需的头文件

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/UniqueVoidPtr.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/irange.h>

#include <unordered_set>
#include <vector>

// 定义命名空间 c10::cuda::CUDACachingAllocator::CudaMallocAsync

namespace c10::cuda::CUDACachingAllocator::CudaMallocAsync {

#if CUDA_VERSION >= 11040
// 如果 CUDA 版本大于或等于 11.4.0，则执行以下代码

// CUDA device allocator that uses cudaMallocAsync to implement
// the same interface as CUDACachingAllocator.cpp.
// 使用 cudaMallocAsync 实现与 CUDACachingAllocator.cpp 相同接口的 CUDA 设备分配器

// Designed to be safe for CUDA graph capture.
// Interactions with CUDA graph capture are mediated by
// notifyCaptureBegin
// notifyCaptureAboutToEnd
// notifyCaptureEnded
// notifyCaptureDestroy
// 设计用于 CUDA 图捕获的安全性。
// 与 CUDA 图捕获的交互由 notifyCaptureBegin、notifyCaptureAboutToEnd、notifyCaptureEnded 和 notifyCaptureDestroy 调解

// Implementation details, not declared in CUDACachingAllocator.h
// 实现细节，未在 CUDACachingAllocator.h 中声明

namespace {

// General helpers
// 一般辅助函数

// Struct to represent a usage stream associated with a CUDA device index
// 结构体，表示与 CUDA 设备索引相关联的使用流
struct UsageStream {
  cudaStream_t stream;    // CUDA 流对象
  c10::DeviceIndex device;  // 设备索引

  UsageStream() = default;
  UsageStream(cudaStream_t s, c10::DeviceIndex d) : stream(s), device(d) {}
  UsageStream(const UsageStream& us) = default;
  UsageStream(const UsageStream&& us) noexcept
      : stream(us.stream), device(us.device) {}
  UsageStream& operator=(UsageStream other) {
    stream = other.stream;
    device = other.device;
    return *this;
  }
};

// Equality operator for UsageStream
// UsageStream 的相等性操作符
bool operator==(const UsageStream& lhs, const UsageStream& rhs) {
  return (lhs.stream == rhs.stream) && (lhs.device == rhs.device);
}

// Hash function for UsageStream, used in unordered containers
// UsageStream 的哈希函数，在无序容器中使用
struct UsageStreamHash {
  size_t operator()(const UsageStream& us) const noexcept {
    return std::hash<void*>{}(us.stream) + size_t(us.device);
  }
};

// Struct to represent pointer usage information
// 结构体，表示指针的使用信息
struct PtrUsage {
  ska::flat_hash_set<UsageStream, UsageStreamHash> recorded_streams;  // 记录的使用流集合
  UsageStream creation_stream{};  // 创建流
  uint64_t size;  // 大小
  bool captured;  // 是否被捕获
  PtrUsage(uint64_t s, bool c) : size(s), captured(c) {}
};

// Device count initialization
// 设备计数初始化
int device_count = 0;

// Flags to track device initialization status
// 用于跟踪设备初始化状态的标志
std::vector<bool> devs_initialized_flags;

// Dummy streams for unifying free operations
// 用于统一释放操作的虚拟流
std::vector<UsageStream> dummy_unifying_free_streams;

// General mutex for thread synchronization
// 一般互斥锁，用于线程同步
std::mutex general_mutex;
/**
 * Note [Avoid freeing uncaptured ptrs during CUDA graph capture]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * During CUDA graph capture, it's illegal to call cudaFreeAsync
 * on a pointer that came from a non-captured cudaMallocAsync.
 * Unfortunately, Python being what it is, it's impossible to be
 * sure no uncaptured tensor will ever have its destructor called
 * in a capturing region.
 * We avoid errors by
 *  1. remembering if allocated pointers were captured or uncaptured
 *  2. during capture, if we detect an attempt to free an uncaptured
 *     allocation on a capturing stream, don't free it immediately,
 *     just remember it and defer its cudaFreeAsync call to after
 *     the end of capture (specifically, to notifyCaptureEnded).
 */

// 使用 ska::flat_hash_map 保存指针和其使用情况的映射关系
using PtrInfo = ska::flat_hash_map<void*, PtrUsage>;
// 存储指针及其使用情况的信息
PtrInfo ptr_info;
// 存储在捕获期间需要推迟释放的未图形化指针列表
std::vector<void*> ungraphed_ptrs_defer_free_until_no_capture;

// These two help setMemoryFraction limit the amount of memory
// used by PyTorch in particular (as opposed to other libraries
// in the same process that might be sharing the same cudaMemPool_t).
// 以下两个向量用于设置 setMemoryFraction 限制 PyTorch 使用的内存量
std::vector<size_t> pytorch_used_bytes;
std::vector<size_t> pytorch_memory_limits;

// Graph-specific helpers

/**
 * Note [Avoid dangling free streams during CUDA graph capture]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * During capture, all stream dependencies must branch out from
 * the stream on which capture began and rejoin this initial stream
 * before capture ends.
 * The user rigs desired forking and joining with event waits.
 * But it's hard to be sure when tensor destructors get called relative
 * to the final joins.
 * For example, suppose a user
 *   forks work stream B from initial capture stream A
 *   creates a tensor T in B
 *   joins by syncing A with B
 *   ends capture.
 * All well and good, right? Maybe not: maybe T went out of scope
 * and its destructor got called AFTER the rejoin, leaving the graph with
 * "unjoined work": a dangling cudaFreeAsync node in stream B.
 * Ensuring that all tensor destructors for all side stream tensors
 * are called before side streams rejoin the main stream is
 * difficult. The user might have to add a bunch of explicit
 * "del"s at the right spots in code that was fine for ordinary
 * eager execution.
 * Fortunately, we can spare the user this burden:
 * during capture, we remember _all_ free streams,
 * and manually rejoin them with the capture stream during
 * notifyCaptureAboutToEnd.
 * This approach is heavy-handed, but hopefully capture only needs to
 * happen once, so we don't mind being heavy-handed.
 *
 * TODO: If, someday, we augment the graph bindings to support recapture
 * https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#whole-graph-update
 * (eg, as a way to accommodate dynamic params) we should think more
 * carefully about the CPU overhead of remembering and rejoining
 * all free streams during capture. Maybe it's not a big deal.
 */
// 存储已释放流的无序集合
std::unordered_set<UsageStream, UsageStreamHash> capture_free_streams;
// 指示是否正在进行捕获的标志
bool capture_underway = false;

// 实现函数

// 假设调用者持有 general_mutex
inline void lazy_init_device(c10::DeviceIndex device) {
  if (!devs_initialized_flags[device]) {
    CUDAGuard g(device);

    // 参见此处的 "Retaining memory in the pool":
    // https://developer.nvidia.com/blog/using-cuda-stream-ordered-memory-allocator-part-1/
    cudaMemPool_t mempool = nullptr;
    C10_CUDA_CHECK(cudaDeviceGetDefaultMemPool(&mempool, device));
    uint64_t threshold = UINT64_MAX;
    C10_CUDA_CHECK(cudaMemPoolSetAttribute(
        mempool, cudaMemPoolAttrReleaseThreshold, &threshold));

    // 我认为这些选项默认都是启用的，但我希望显式启用它们以确保意识到它们的存在。
    int enable = 1;
    C10_CUDA_CHECK(cudaMemPoolSetAttribute(
        mempool, cudaMemPoolReuseFollowEventDependencies, &enable));
    C10_CUDA_CHECK(cudaMemPoolSetAttribute(
        mempool, cudaMemPoolReuseAllowOpportunistic, &enable));
    C10_CUDA_CHECK(cudaMemPoolSetAttribute(
        mempool, cudaMemPoolReuseAllowInternalDependencies, &enable));

    // 从当前设备获取一个流用作多流使用的 "统一" 自由流
    const auto dufs = getStreamFromPool();
    dummy_unifying_free_streams[device] =
        UsageStream(dufs.stream(), dufs.device_index());

    pytorch_used_bytes[device] = 0;
    pytorch_memory_limits[device] = UINT64_MAX;

    devs_initialized_flags[device] = true;
  }
}

inline void sync_raw(cudaStream_t dependency, cudaStream_t dependent) {
  // CUDACachingAllocator.cpp 使用原始的 cuda 事件，我们也是如此。
  cudaEvent_t event = nullptr;
  C10_CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
  C10_CUDA_CHECK(cudaEventRecord(event, dependency));
  C10_CUDA_CHECK(cudaStreamWaitEvent(dependent, event));
  C10_CUDA_CHECK(cudaEventDestroy(event));
}

// 假设调用者持有 general_mutex
inline void free_impl(PtrInfo::iterator& it) {
  // 可能的微优化：如果我们在这里进行值复制，我们可以将 ptr_info.erase(it) 提前到这里并立即释放锁。
  const auto& recorded_streams = it->second.recorded_streams;
  const auto& creation_stream = it->second.creation_stream;

  // 如果使用流是空（默认）流，
  // cudaFreeAsync 会从当前上下文推断设备，
  // 因此我们需要设置正确的上下文环境。
  CUDAGuard g(creation_stream.device);

  if (recorded_streams.empty()) {
    // ptr 仅在一个流上使用过，这个流必须是原始分配流。
    // 在原始分配流上释放 ptr。
    C10_CUDA_CHECK(cudaFreeAsync(it->first, creation_stream.stream));

    if (C10_UNLIKELY(capture_underway)) {
      // 参见 "Note [Avoid dangling free streams during CUDA graph capture]"
      capture_free_streams.insert(creation_stream);
    }
  } else {
    // ptr was used on many streams. We don't know which was the most recent.
    // There could even have been multiple most recent usage streams acting
    // on different regions of the memory.
    // But cudaFreeAsync only accepts a single most recent usage stream.
    // We can still safely free ptr with a trick:
    // Use a dummy "unifying stream", sync the unifying stream with all of
    // ptr's usage streams, and pass the dummy stream to cudaFreeAsync.

    // 从原始分配设备上获取虚拟的“统一释放流”
    auto dummy_unifying_free_stream =
        dummy_unifying_free_streams[creation_stream.device];
    // 断言虚拟释放流与原始分配流设备相同
    TORCH_INTERNAL_ASSERT(
        dummy_unifying_free_stream.device == creation_stream.device);

    // 不需要再次保护，因为已经在 creation_stream.device 上
    // 同步 creation_stream.stream 和 dummy_unifying_free_stream.stream
    sync_raw(creation_stream.stream, dummy_unifying_free_stream.stream);

    // 使用流的数量通常很少（一般为低位数）
    for (const auto& recorded_stream : recorded_streams) {
      // 此处的逻辑适应于某些使用流可能在其他设备上的情况，
      // 如果某些使用内核通过 p2p 访问内存，则可能发生这种情况。

      // cudaEventRecord 要求输入事件和流在同一设备上。
      CUDAGuard g_usage(recorded_stream.device);

      // 同步 recorded_stream.stream 和 dummy_unifying_free_stream.stream
      sync_raw(recorded_stream.stream, dummy_unifying_free_stream.stream);
    }

    // 在虚拟的“统一释放流”中异步释放 ptr
    C10_CUDA_CHECK(cudaFreeAsync(it->first, dummy_unifying_free_stream.stream));
    // 此时，除非 dummy_unifying_free_stream 恰好别名为某些未来的用户流，
    // 否则分配只能用于“机会性”重用，即如果 CPU 看到 dummy_unifying_free_stream
    // 已经达到所有使用流上记录事件的解析点。理论上，我们可以通过例如
    // 替换 cudaStreamWaitEvent(dummy_unifying_free_stream.stream, event);
    // 为 cudaStreamWaitEvent(creation_stream.stream, event);
    // 然后直接在 creation_stream.stream 中进行 cudaFreeAsync，
    // 但这会强制创建潜在的 creation_stream.stream 对所有 recorded_streams 的错误依赖。

    if (C10_UNLIKELY(capture_underway)) {
      // 参见 Note [Avoid dangling free streams during CUDA graph capture]
      capture_free_streams.emplace(
          dummy_unifying_free_stream.stream, dummy_unifying_free_stream.device);
    }
  }

  // 更新 pytorch_used_bytes 的设备上已使用字节数
  pytorch_used_bytes[creation_stream.device] -= it->second.size;

  // 删除 ptr_info 中的迭代器 it
  ptr_info.erase(it);
}

// 释放异步分配的内存块
void freeAsync(void* ptr) {
  // 使用通用互斥锁保护临界区
  std::lock_guard<std::mutex> lk(general_mutex);

  // 检查CUDA操作中的错误状态
  auto err = cudaGetLastError();
  C10_CUDA_CHECK(err);

  // 在ptr_info中查找ptr的迭代器
  auto it = ptr_info.find(ptr);
  // 断言确保ptr在ptr_info中存在，否则报错
  TORCH_INTERNAL_ASSERT(it != ptr_info.end(), "ptr not found in ptr_info");

  // 如果CUDA图捕获正在进行
  if (C10_UNLIKELY(capture_underway)) {
    // 如果ptr未被捕获
    if (!it->second.captured) {
      // 发出警告，说明在图捕获期间对未捕获的分配空间调用了freeAsync()
      TORCH_WARN_ONCE(
          "freeAsync() was called on an uncaptured allocation during graph capture "
          "(address = ",
          ptr,
          "). This may be benign, for example, a Python tensor in the capture "
          "might happen to shadow (use the same name as) an unrelated temporary "
          "tensor from somewhere before capture, pushing the earlier tensor "
          "out of scope. "
          "However, if the tensor we're freeing here IS used by the capture, "
          "freeing it is an error, and may cause illegal memory accesses or "
          "memory corruption during graph replay.");

      // 查看Note [Avoid freeing uncaptured ptrs during CUDA graph capture]
      // 将未捕获的ptr推入延迟释放列表，以避免迭代器失效
      ungraphed_ptrs_defer_free_until_no_capture.push_back(ptr);
      return;
    }
  }
  // 如果CUDA图未捕获，并且ptr已捕获
  else if (C10_UNLIKELY(it->second.captured)) {
    // 发出警告，说明尝试释放已捕获分配的未捕获的空间
    TORCH_WARN(
        "Attempting uncaptured free of a captured allocation with address ",
        ptr,
        "\nThis is technically allowed, but may indicate you are losing "
        "the last user-visible tensor through which the allocation can "
        "be accessed, so you'll have no way to view the data after "
        "future replays of the owning graph.");
  }

  // 调用内部函数实现free操作
  free_impl(it);
}

// 对称于NativeCachingAllocator::malloc的异步内存分配函数
void mallocAsync(
    void** devPtr,
    c10::DeviceIndex device,
    size_t size,
    cudaStream_t stream) {
  // 断言确保设备索引在有效范围内
  TORCH_INTERNAL_ASSERT(
      0 <= device && device < device_count,
      "Invalid device index ",
      device,
      ": did you call init?");

  // 如果流是空（默认）流，则cudaMallocAsync会从环境上下文推断设备
  // 因此需要设置正确的环境上下文
  CUDAGuard g(device);

  // 使用通用互斥锁保护临界区
  std::lock_guard<std::mutex> lk(general_mutex);

  // 如果不是CUDA图捕获过程，并且存在待释放的未捕获指针
  if (!capture_underway &&
      !ungraphed_ptrs_defer_free_until_no_capture.empty()) {
    // 查看Note [Avoid freeing uncaptured ptrs during CUDA graph capture]
    // 遍历延迟释放列表中的所有指针，并在ptr_info中释放它们
    for (const auto ptr : ungraphed_ptrs_defer_free_until_no_capture) {
      auto it = ptr_info.find(ptr);
      TORCH_INTERNAL_ASSERT(it != ptr_info.end(), "ptr not found in ptr_info");
      free_impl(it);
    }
    // 清空在无捕获情况下延迟释放的指针定义集合
    ungraphed_ptrs_defer_free_until_no_capture.clear();
  }

  // 惰性初始化设备
  lazy_init_device(device);

  // 防御性地检查是否存在先前的 CUDA 错误状态。
  auto err = cudaGetLastError();
  C10_CUDA_CHECK(err);

  // TODO: 是否可以在不持有 general_mutex 的情况下避免调用 cudaMallocAsync，
  // 也许可以通过让 lazy_init_device 使用单独的 once_flags 或者内部静态初始化器来实现？
  if (pytorch_used_bytes[device] + size > pytorch_memory_limits[device]) {
    // 如果当前设备已使用的字节数加上请求分配的内存大小超过了设备内存限制，
    // 则将错误状态设置为内存分配错误。
    err = cudaErrorMemoryAllocation;
  } else {
    // 否则，调用 cudaMallocAsync 分配内存到设备上指定地址 devPtr，大小为 size，使用 stream 进行异步操作。
    err = cudaMallocAsync(devPtr, size, stream);
  }

  if (err == cudaErrorMemoryAllocation) {
    // 清除 CUDA 的内部错误状态，以便用户可以捕获内存不足异常，在脚本侧释放资源并重试分配操作。
    (void)cudaGetLastError(); // 清除 CUDA 错误
    size_t device_free = 0;
    size_t device_total = 0;
    C10_CUDA_CHECK(cudaMemGetInfo(&device_free, &device_total));
    // 抛出带有详细信息的 OutOfMemoryError 异常，说明内存分配超过了允许的限制。
    TORCH_CHECK_WITH(
        OutOfMemoryError,
        false,
        "Allocation on device ",
        device,
        " would exceed allowed memory. (out of memory)",
        "\nCurrently allocated     : ",
        format_size(pytorch_used_bytes[device]),
        "\nRequested               : ",
        format_size(size),
        "\nDevice limit            : ",
        format_size(device_total),
        "\nFree (according to CUDA): ",
        format_size(device_free),
        "\nPyTorch limit (set by user-supplied memory fraction)"
        "\n                        : ",
        format_size(pytorch_memory_limits[device]));
  } else {
    // 否则，检查 CUDA 操作是否成功，若成功则继续执行。
    C10_CUDA_CHECK(err);
  }

  // 将分配的指针信息插入 ptr_info 的映射中，记录指针的使用情况和大小。
  auto inserted = ptr_info.emplace(*devPtr, PtrUsage(size, capture_underway));
  // 在不允许重复插入的前提下插入指针信息，如果已存在相同地址的记录则触发内部断言错误。
  TORCH_INTERNAL_ASSERT(
      inserted.second,
      "address returned by cudaMallocAsync already exists "
      "in ptr_info");

  // 记录指针的创建流和所属设备。
  inserted.first->second.creation_stream = {stream, device};

  // 更新当前设备已使用的字节数。
  pytorch_used_bytes[device] += size;
}

} // anonymous namespace

// 定义了一个本地的内存释放函数声明
void local_raw_delete(void* ptr);

// 与 CUDACachingAllocator.cpp 相同的模式。
// 继承自 CUDAAllocator 的结构体 CudaMallocAsyncAllocator
struct CudaMallocAsyncAllocator : public CUDAAllocator {
  // 重写 allocate 方法，用于分配内存
  DataPtr allocate(size_t size) override {
    // 定义常量 1 EB（艾字节）的大小
    constexpr size_t one_exa_bytes = 1152921504606846976ULL;
    // 检查是否超过了内存限制
    TORCH_CHECK_WITH(
        OutOfMemoryError,
        size < one_exa_bytes,
        "CUDA out of memory. Tried to allocate more than 1EB memory.");
    
    // 获取当前设备索引
    c10::DeviceIndex device = 0;
    C10_CUDA_CHECK(c10::cuda::GetDevice(&device));
    
    // 初始化指针 r 为 nullptr
    void* r = nullptr;
    // 如果 size 不为 0，则异步分配内存
    if (size != 0) {
      mallocAsync(&r, device, size, cuda::getCurrentCUDAStream(device));
    }
    
    // 返回 DataPtr 结构体，包含分配的内存指针、原始指针、本地释放函数、设备信息
    return {r, r, &local_raw_delete, Device(DeviceType::CUDA, device)};
  }
  
  // 返回本地的内存释放函数指针
  DeleterFnPtr raw_deleter() const override {
    return &local_raw_delete;
  }

  // 初始化函数，不应该执行任何创建上下文的调用，
  // 只为后续的 init 函数调用设置每个设备池的基础
  void init(int dev_count) override {
    // 定义静态的 called 变量，用于初始化设备数量和相关数据结构
    static bool called = [](int dev_count) {
      // 初始化设备数量
      device_count = dev_count;
      // 初始化设备初始化标志数组
      devs_initialized_flags.resize(dev_count, false);
      // 初始化虚拟的释放流数组
      dummy_unifying_free_streams.resize(dev_count);
      // 初始化 PyTorch 使用的字节数数组
      pytorch_used_bytes.resize(dev_count);
      // 初始化 PyTorch 内存限制数组
      pytorch_memory_limits.resize(dev_count);
      return true;
    }(dev_count);
    (void)called;
  }

  // 检查是否已初始化
  bool initialized() override {
    return !devs_initialized_flags.empty();
  }

  // 静态内联函数，用于断言设备索引的有效性
  static inline void assertValidDevice(c10::DeviceIndex device) {
    TORCH_CHECK(
        0 <= device && device < device_count, "Invalid device argument.");
  }

  // 设置内存分配的比例和设备索引
  void setMemoryFraction(double fraction, c10::DeviceIndex device) override {
    // 断言分配比例在有效范围内
    TORCH_INTERNAL_ASSERT(
        0 <= fraction && fraction <= 1,
        "invalid fraction:",
        fraction,
        ". Please set within (0, 1).");

    // 加锁以保证线程安全性
    std::lock_guard<std::mutex> lk(general_mutex);
    // 断言设备索引的有效性
    assertValidDevice(device);
    // 切换当前设备为指定设备
    CUDAGuard g(device);
    // 执行延迟初始化设备的操作
    lazy_init_device(device);

    // 查询设备的空闲和总内存信息
    size_t device_free = 0;
    size_t device_total = 0;
    C10_CUDA_CHECK(cudaMemGetInfo(&device_free, &device_total));
    
    // 设置 PyTorch 内存限制为指定设备的总内存乘以分配比例
    pytorch_memory_limits[device] =
        static_cast<uint64_t>(fraction * static_cast<double>(device_total));

    // 可选：使用 cudaMemPoolSetAttribute 设置软内存限制
    // cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold);
    // 这是一个软提示：驱动程序允许池的保留内存在高 cudaMallocAsync 需求的区域中超过阈值，
    // 但会及时将保留内存调整回阈值
  // 当内存使用量低于阈值时执行清空缓存操作。作者可能认为这种策略引入了性能的不确定性。
  // 清空缓存的操作，使用互斥锁确保线程安全性
  void emptyCache() override {
    std::lock_guard<std::mutex> lk(general_mutex);

    // 遍历所有设备，如果设备已初始化，则执行以下操作
    for (int dev = 0; dev < device_count; dev++) {
      if (devs_initialized_flags[dev]) {
        // 切换当前 CUDA 设备到指定设备
        CUDAGuard g(static_cast<c10::DeviceIndex>(dev));

        // 获取指定设备的默认内存池并进行同步
        cudaMemPool_t mempool = nullptr;
        cudaDeviceGetDefaultMemPool(&mempool, dev);
        cudaDeviceSynchronize();

        // 将指定内存池中的内存调整到指定大小（这里是调整到0，即清空内存池）
        cudaMemPoolTrimTo(mempool, 0);
      }
    }
  }

  // 获取设备缓存信息的方法，用于为后续 cudnnFind 调用提供合理的最大工作空间大小
  void cacheInfo(c10::DeviceIndex device, size_t* maxWorkspaceGuess) override {
    // cacheInfo 方法的唯一消费者是 Conv_v7.cpp 中的 getMaxWorkspaceSize。
    // cacheInfo 的作用是为 getMaxWorkspaceSize 提供一个合理的最大工作空间大小，
    // 以便于即将进行的 cudnnFind 调用。
    //
    // 本地分配器的 cacheInfo 方法选择返回其最大未使用块的大小，
    // 即本地分配器可以立即和异步地服务的最大分配（无需 cudaMalloc）。
    //
    // 这里采用不同的启发式方法：通过一些经验试错确定最大可用的工作空间大小。
    // 尽管在性能上可能不够高效，因为 cacheInfo 是 cudnnFind 的前奏。
    //
    // 算法缓存随后存储使用工作空间大小 <= maxWorkspaceGuess 的最佳执行算法。
    // 对于相同的参数集合，后续调用会命中缓存并尝试分配相同的工作空间。
    // 如果在将来的某个调用中，工作空间分配失败（例如因为环境内存不足），
    // 则绑定会重新运行 cudnnFind，并在此之前再次调用 cacheInfo 以估计一个新的（更小的）可用工作空间。
    // 在几次这样的调用后，缓存应该稳定到一个算法，其工作空间大小足够小，以便每次都成功（对于该参数集）。
    //
    // 因此，这里的策略是返回一个粗略但较大的猜测值，并让绑定在随时间需要时进行修剪。
    //
    // 唯一的注意事项是，即使现在和将来的调用中分配工作空间时没有内存不足错误，
    // 很难确定那些后来无错误的 cudaMallocAsync 是否快速且直接来自池（即 cudaMallocAsync 是否需要从系统中保留更多内存）。
    // 希望在重复的工作空间请求后，池的保留内存也稳定到一个所有请求都直接来自池的点。
    std::lock_guard<std::mutex> lk(general_mutex);
    assertValidDevice(device);
    // 将当前 CUDA 设备切换到指定设备
    CUDAGuard g(device);
    // 如果设备尚未初始化，则延迟初始化设备
    lazy_init_device(device);

    // 获取设备上的内存信息，包括自由上限和总量
    size_t free_upper_bound = 0;
    size_t device_total = 0;
    C10_CUDA_CHECK(cudaMemGetInfo(&free_upper_bound, &device_total));
    // 断言：自由上限加上已使用字节不应超过设备总内存
    TORCH_INTERNAL_ASSERT(
        free_upper_bound + pytorch_used_bytes[device] <= device_total);
    // 计算预估的内存分配大小，取较小值：free_upper_bound 或 (设备的内存限制 - 当前已使用的字节数)
    size_t guess = std::min(
        free_upper_bound,
        pytorch_memory_limits[device] - pytorch_used_bytes[device]);

    // 获取当前 CUDA 流
    auto stream = c10::cuda::getCurrentCUDAStream();

    // 创建一个空指针，用于后续的内存分配操作
    void* dummy = nullptr;

    // 防御性地检查是否存在之前的 CUDA 错误状态
    auto err = cudaGetLastError();
    C10_CUDA_CHECK(err);

    // 循环尝试内存分配，直到成功分配为止
    while (true) {
      // 检查是否预估的内存分配大小超过了设备的内存限制
      if (pytorch_used_bytes[device] + guess > pytorch_memory_limits[device]) {
        // 如果超出限制，设置错误状态为内存分配错误
        err = cudaErrorMemoryAllocation;
      } else {
        // 否则，尝试异步分配内存
        err = cudaMallocAsync(&dummy, guess, stream);
      }

      // 如果成功分配内存，则释放该内存并返回预估的大小
      if (err == cudaSuccess) {
        cudaFreeAsync(dummy, stream);
        *maxWorkspaceGuess = guess;
        return;
      } else if (err == cudaErrorMemoryAllocation) {
        // 如果出现内存分配错误，清除 CUDA 错误状态，并尝试减小预估大小的一半
        (void)cudaGetLastError(); // 清除 CUDA 错误状态
        guess >>= 1; // 快速并粗略地减小一半大小，用于下一轮迭代
      } else {
        // 如果出现其他 CUDA 错误，抛出异常
        C10_CUDA_CHECK(err);
      }
    }
  }

  // 获取给定指针的基础分配和大小
  void* getBaseAllocation(void* ptr, size_t* size) override {
    std::lock_guard<std::mutex> lk(general_mutex);

    // 在 ptr_info 中查找指针，确保其存在
    auto it = ptr_info.find(ptr);
    TORCH_INTERNAL_ASSERT(it != ptr_info.end(), "ptr not found in ptr_info");

    // 如果请求了大小，则返回其大小
    if (size) {
      *size = it->second.size;
    }

    // 返回指针本身
    return ptr;
  }

  // 记录给定数据指针的 CUDA 流信息
  void recordStream(const DataPtr& ptr, cuda::CUDAStream stream) override {
    std::lock_guard<std::mutex> lk(general_mutex);

    auto ptr_val = ptr.get();

    // 如果数据指针为空，则无需记录流信息
    if (!ptr_val) {
      return;
    }

    // 确保 ptr_info 中存在该指针的信息
    auto it = ptr_info.find(ptr_val);
    TORCH_INTERNAL_ASSERT(it != ptr_info.end(), "ptr not found in ptr_info");

    // 准备要记录的流信息，包括流的 ID 和设备索引
    UsageStream to_record{stream.stream(), stream.device_index()};

    // 如果要记录的流与数据的创建流相同，发出警告，因为记录流是多余的
    if (to_record == it->second.creation_stream) {
      TORCH_WARN_ONCE(
          "Called record_stream on tensor whose original creation stream "
          "matches the recorded stream. This is unnecessary and has no effect.");
    } else {
      // 否则，将要记录的流信息插入到已记录流的集合中
      it->second.recorded_streams.insert(to_record);
    }
  }

  // 获取 IPC 设备指针的共享指针，目前不支持该功能
  std::shared_ptr<void> getIpcDevPtr(std::string handle) override {
    TORCH_CHECK(
        false,
        "cudaMallocAsync does not yet support getIpcDevPtr. "
        "If you need it, please file an issue describing your use case.");
  }

  // 记录历史操作，目前不支持该功能
  void recordHistory(
      bool enabled,
      CreateContextFn context_recorder,
      size_t alloc_trace_max_entries,
      RecordContext when) override {
    TORCH_CHECK(
        false,
        "cudaMallocAsync does not yet support recordHistory. "
        "If you need it, please file an issue describing your use case.");
  }

  // 添加内存耗尽观察器，目前不支持该功能
  void attachOutOfMemoryObserver(OutOfMemoryObserver observer) override {
    // 确保 cudaMallocAsync 尚不支持 attachOutOfMemoryObserver 功能，抛出错误信息
    TORCH_CHECK(
        false,
        "cudaMallocAsync does not yet support attachOutOfMemoryObserver. "
        "If you need it, please file an issue describing your use case.");
    }
    
    // 确保 cudaMallocAsync 尚不支持 attachAllocatorTraceTracker 功能，抛出错误信息
    void attachAllocatorTraceTracker(AllocatorTraceTracker tracker) override {
      TORCH_CHECK(
          false,
          "cudaMallocAsync does not yet support attachAllocatorTraceTracker. "
          "If you need it, please file an issue describing your use case.");
    }
    
    // 确保 cudaMallocAsync 尚不支持 getCheckpointState 功能，抛出错误信息
    std::shared_ptr<AllocatorState> getCheckpointState(
        c10::DeviceIndex device,
        MempoolId_t id) override {
      TORCH_CHECK(
          false,
          "cudaMallocAsync does not yet support getCheckpointState. "
          "If you need it, please file an issue describing your use case.");
    }
    
    // 确保 cudaMallocAsync 尚不支持 setCheckpointPoolState 功能，抛出错误信息
    CheckpointDelta setCheckpointPoolState(
        c10::DeviceIndex device,
        std::shared_ptr<AllocatorState> pps) override {
      TORCH_CHECK(
          false,
          "cudaMallocAsync does not yet support setCheckpointPoolState. "
          "If you need it, please file an issue describing your use case.");
    }
    
    // 获取设备的统计信息
    // 如果设备尚未使用，则返回全为 0 的统计信息，而不会创建上下文
    DeviceStats getDeviceStats(c10::DeviceIndex device) override {
      // 断言设备索引的有效性
      assertValidDevice(device);
    
      // 当前内存池保留的内存
      uint64_t reserved_mem_current = 0;
      // 自上次重置以来内存池保留的内存的高水位
      uint64_t reserved_mem_peak = 0;
      // 当前内存池使用的内存
      uint64_t used_mem_current = 0;
      // 内存使用的高水位
      uint64_t used_mem_peak = 0;
    
      // 使用互斥锁保护共享资源
      std::lock_guard<std::mutex> lk(general_mutex);
    
      // 如果设备已初始化
      if (devs_initialized_flags[device]) {
        // 切换到指定设备
        CUDAGuard g(device);
    
        // 获取设备的默认内存池
        cudaMemPool_t mempool = nullptr;
        C10_CUDA_CHECK(cudaDeviceGetDefaultMemPool(&mempool, device));
    
        // 获取内存池当前保留的内存量
        C10_CUDA_CHECK(cudaMemPoolGetAttribute(
            mempool, cudaMemPoolAttrReservedMemCurrent, &reserved_mem_current));
    
        // 获取内存池自上次重置以来的内存高水位
        C10_CUDA_CHECK(cudaMemPoolGetAttribute(
            mempool, cudaMemPoolAttrReservedMemHigh, &reserved_mem_peak));
    
        // 获取内存池当前使用的内存量
        C10_CUDA_CHECK(cudaMemPoolGetAttribute(
            mempool, cudaMemPoolAttrUsedMemCurrent, &used_mem_current));
    
        // 获取内存池使用的内存高水位
        C10_CUDA_CHECK(cudaMemPoolGetAttribute(
            mempool, cudaMemPoolAttrUsedMemHigh, &used_mem_peak));
      }
    
      // 创建设备统计信息对象
      DeviceStats stats;
    
      // 对于本地分配器，许多统计类型是特定的，我们不做处理，它们的结构体 Stat 将包含零值。
      // allocated_bytes 表示尚未释放的 malloc() 分配的内存总字节数。
      // active_bytes 表示尚未释放到自由池的 malloc() 分配的内存总字节数，包括所有 allocated_bytes，
      // 以及“悬空状态”块的字节数，这些块已经被 free() 但由于未完成的流使用而尚未 free_block() 回到池中。
      //
    // 在 cudaMallocAsync 分配器中：
    // 我们仅仅询问驱动程序关于活跃内存的意见。
    // 我们不需要区分 allocated_bytes 和 active_bytes。
    stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].current =
        static_cast<int64_t>(used_mem_current);
    stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].peak =
        static_cast<int64_t>(used_mem_peak);
    stats.active_bytes[static_cast<size_t>(StatType::AGGREGATE)].current =
        static_cast<int64_t>(used_mem_current);
    stats.active_bytes[static_cast<size_t>(StatType::AGGREGATE)].peak =
        static_cast<int64_t>(used_mem_peak);
    stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current =
        static_cast<int64_t>(reserved_mem_current);
    stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].peak =
        static_cast<int64_t>(reserved_mem_peak);

    return stats;
  }

  // 重置累积统计数据，针对指定设备
  void resetAccumulatedStats(c10::DeviceIndex device) override {
    assertValidDevice(device);
    TORCH_WARN_ONCE(
        "For backend:cudaMallocAsync, resetAccumulatedStats has no effect.");
  }

  // 重置峰值统计数据，针对指定设备
  void resetPeakStats(c10::DeviceIndex device) override {
    assertValidDevice(device);

    // 在 CUDA 中设置默认内存池，获取默认的内存池对象
    CUDAGuard g(device);
    cudaMemPool_t mempool = nullptr;
    C10_CUDA_CHECK(cudaDeviceGetDefaultMemPool(&mempool, device));
    // 使用零作为重置值是 CUDA 驱动团队推荐的方法
    // Vivek Kini 表示：
    //   "将其重置为零（这是设置 ReservedMemHigh 时唯一有效的值）会将其重置为驱动程序内部的 ReservedMemCurrent
    //    （UsedMemHigh/UsedMemCurrent 也是如此）"
    uint64_t zero = 0;
    C10_CUDA_CHECK(cudaMemPoolSetAttribute(
        mempool, cudaMemPoolAttrReservedMemHigh, &zero));
    C10_CUDA_CHECK(
        cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrUsedMemHigh, &zero));
  }

  // 获取快照信息（此处返回空对象，因为 cudaMallocAsync 不跟踪个别内存块）
  SnapshotInfo snapshot() override {
    TORCH_CHECK(
        false,
        "Calling snapshot with backend:cudaMallocAsync is not meaningful. "
        "(For backend:native, snapshot returns a detailed summary of all "
        "blocks tracked by the allocator, but the cudaMallocAsync backend "
        "does not track individual blocks.)");
    // 替代方案：TORCH_WARN
    return {};
  }

  // CUDAGraph 交互操作：开始将分配操作分配给内存池
  void beginAllocateToPool(
      c10::DeviceIndex device,
      MempoolId_t mempool_id,
      std::function<bool(cudaStream_t)>) override {
    std::lock_guard<std::mutex> lk(general_mutex);

    TORCH_INTERNAL_ASSERT(capture_free_streams.empty());
    TORCH_CHECK(
        !capture_underway,
        "Only one capture at a time is allowed in a process.")
    capture_underway = true;
  }

  // 结束将分配操作分配给内存池
  void endAllocateToPool(c10::DeviceIndex device, MempoolId_t mempool_id)
      override {
    assertValidDevice(device);

    std::lock_guard<std::mutex> lk(general_mutex);
    // 检查是否正在进行内存捕获，如果未进行捕获，则抛出错误信息
    TORCH_CHECK(
        capture_underway,
        "CudaMallocAsync::notifyCaptureAboutToEnd called, "
        "but CudaMallocAsync::capture_underway is false.");

    // 获取当前 CUDA 设备的流
    auto capture_stream = cuda::getCurrentCUDAStream(device);

    // 遍历捕获释放的流列表
    // 注释：遍历每一个释放流，确保事件记录和流在同一个设备上
    for (const auto& free_stream : capture_free_streams) {
      // 切换到对应设备上的 CUDA 环境
      CUDAGuard g(free_stream.device);

      // 创建一个禁用计时的 CUDA 事件
      cudaEvent_t event = nullptr;
      C10_CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
      // 记录事件，并等待流完成事件
      C10_CUDA_CHECK(cudaEventRecord(event, free_stream.stream));
      C10_CUDA_CHECK(cudaStreamWaitEvent(capture_stream.stream(), event));
      // 销毁事件
      C10_CUDA_CHECK(cudaEventDestroy(event));
    }

    // 清空捕获释放流列表
    capture_free_streams.clear();

    // 再次检查是否正在进行内存捕获，如果未进行捕获，则抛出错误信息
    TORCH_CHECK(
        capture_underway,
        "CudaMallocAsync::notifyCaptureEnded called, "
        "but CudaMallocAsync::capture_underway is false.");
    // 将内存捕获状态设置为未进行中
    capture_underway = false;
  }

  // 释放内存池中特定设备和内存池 ID 的内存
  void releasePool(c10::DeviceIndex device, MempoolId_t mempool_id) override {
    // Q: 这里是否需要进行特殊处理，例如清理在原始捕获期间创建的长期指针？
    // A: 我认为不需要。
    //    这些分配能够在捕获过程中幸存下来，是因为用户持有对它们的显式引用，
    //    当用户完成对它们的使用后，这些张量的析构函数将调用每个指针的 freeAsync()。
    //    freeAsync() 可能会导致 TORCH_WARN("Attempting uncaptured free of a captured allocation..."，
    //    但是过时的指针不会永久泄漏到 ptr_info 中。
  }

  // 分配指定大小的内存，并返回指针
  void* raw_alloc(size_t nbytes) override {
    if (nbytes == 0) {
      return nullptr;
    }
    // 获取当前 CUDA 设备索引
    c10::DeviceIndex device = 0;
    C10_CUDA_CHECK(c10::cuda::GetDevice(&device));
    // 分配异步内存，并返回分配的指针
    void* r = nullptr;
    mallocAsync(&r, device, nbytes, cuda::getCurrentCUDAStream(device));
    return r;
  }

  // 使用指定流分配指定大小的内存，并返回指针
  void* raw_alloc_with_stream(size_t nbytes, cudaStream_t stream) override {
    if (nbytes == 0) {
      return nullptr;
    }
    // 获取当前 CUDA 设备索引
    c10::DeviceIndex device = 0;
    C10_CUDA_CHECK(c10::cuda::GetDevice(&device));
    // 分配异步内存，并返回分配的指针
    void* r = nullptr;
    mallocAsync(&r, device, nbytes, stream);
    return r;
  }

  // 释放分配的内存
  void raw_delete(void* ptr) override {
    // 异步释放内存指针
    freeAsync(ptr);
  }

  // 允许设备间的对等访问
  void enablePeerAccess(c10::DeviceIndex dev, c10::DeviceIndex dev_to_access)
      override {
    // 双重检查分配器后端未更改，否则会出现错误。
    // cudaMallocAsync 池不受 cudaDeviceEnablePeerAccess 影响。我们需要特定于池的启用。
    // 参考：https://developer.nvidia.com/blog/using-cuda-stream-ordered-memory-allocator-part-2/
    // 切换到指定设备的 CUDA 环境
    c10::cuda::CUDAGuard device_guard(dev);
    // 创建一个 CUDA 内存池对象，并初始化为 nullptr
    cudaMemPool_t mempool = nullptr;
    // 获取默认设备的内存池，并将其赋值给 mempool
    C10_CUDA_CHECK(cudaDeviceGetDefaultMemPool(&mempool, dev_to_access));
    // 创建一个 CUDA 内存访问描述符，并初始化为空
    cudaMemAccessDesc desc = {};
    // 设置描述符中的内存位置类型为设备类型
    desc.location.type = cudaMemLocationTypeDevice;
    // NOLINTNEXTLINE(bugprone-signed-char-misuse)
    // 设置描述符中的设备 ID，用于访问特定设备的内存
    desc.location.id = dev;
    // 设置描述符中的访问标志为读写保护读写
    desc.flags = cudaMemAccessFlagsProtReadWrite;
    // 将描述符应用于内存池，以便进行后续的内存访问管理
    C10_CUDA_CHECK(cudaMemPoolSetAccess(mempool, &desc, 1 /* numDescs */));
  }
  // 实现 override 的 memcpyAsync 函数，用于异步内存复制
  cudaError_t memcpyAsync(
      void* dst,
      int dstDevice,
      const void* src,
      int srcDevice,
      size_t count,
      cudaStream_t stream,
      bool p2p_enabled) override {
    // 如果启用对等内存复制(p2p_enabled 为真)或目标设备与源设备相同，则使用设备到设备的内存复制
    if (p2p_enabled || dstDevice == srcDevice) {
      return cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToDevice, stream);
    } else {
      // 否则，使用对等设备之间的内存复制
      return cudaMemcpyPeerAsync(dst, dstDevice, src, srcDevice, count, stream);
    }
  }
  // 实现 override 的 name 函数，返回当前类的名称字符串
  std::string name() override {
    return "cudaMallocAsync";
  }
  // 实现 final 的 copy_data 函数，执行设备到设备的内存复制操作
  void copy_data(void* dest, const void* src, std::size_t count) const final {
    // 使用 CUDA API 将源内存复制到目标内存，复制大小为 count 字节
    C10_CUDA_CHECK(
        cudaMemcpy(dest, src, count, cudaMemcpyKind::cudaMemcpyDeviceToDevice));
  }
};

CudaMallocAsyncAllocator device_allocator;


// 定义一个名为 device_allocator 的全局变量，类型为 CudaMallocAsyncAllocator，用于异步 CUDA 内存分配
CudaMallocAsyncAllocator device_allocator;



void local_raw_delete(void* ptr) {
  freeAsync(ptr);
}


// 定义一个名为 local_raw_delete 的函数，用于释放传入指针所指向的内存块
void local_raw_delete(void* ptr) {
  freeAsync(ptr);
}



CUDAAllocator* allocator() {
  return &device_allocator;
}


// 定义一个名为 allocator 的函数，返回一个指向 device_allocator 的指针，类型为 CUDAAllocator*
CUDAAllocator* allocator() {
  return &device_allocator;
}



#else
CUDAAllocator* allocator() {
  TORCH_CHECK(false, "Cannot use cudaMallocAsyncAllocator with cuda < 11.4.");
  return nullptr;
}


// 如果没有满足前置条件，则定义一个名为 allocator 的函数，检查条件是否为 false，如果是，则抛出错误信息
// 在 CUDA 小于版本 11.4 的情况下无法使用 cudaMallocAsyncAllocator。
CUDAAllocator* allocator() {
  TORCH_CHECK(false, "Cannot use cudaMallocAsyncAllocator with cuda < 11.4.");
  return nullptr;
}



#endif


// 结束 #ifdef 条件编译指令
#endif



} // namespace c10::cuda::CUDACachingAllocator::CudaMallocAsync


// 结束命名空间 namespace c10::cuda::CUDACachingAllocator::CudaMallocAsync
} // namespace c10::cuda::CUDACachingAllocator::CudaMallocAsync
```