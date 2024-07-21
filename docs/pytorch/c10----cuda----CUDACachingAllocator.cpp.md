# `.\pytorch\c10\cuda\CUDACachingAllocator.cpp`

```py
// 包含 CUDA 缓存分配器的头文件
#include <c10/cuda/CUDACachingAllocator.h>

// 包含 GPU 跟踪实现的头文件
#include <c10/core/impl/GPUTrace.h>
// 包含 CUDA 分配器配置的头文件
#include <c10/cuda/CUDAAllocatorConfig.h>
// 包含 CUDA 异常处理的头文件
#include <c10/cuda/CUDAException.h>
// 包含 CUDA 功能的头文件
#include <c10/cuda/CUDAFunctions.h>
// 包含 CUDA 设备守卫的头文件
#include <c10/cuda/CUDAGuard.h>
// 包含一次性调用的实用功能的头文件
#include <c10/util/CallOnce.h>
// 包含作用域退出的实用功能的头文件
#include <c10/util/ScopeExit.h>
// 包含唯一空指针的实用功能的头文件
#include <c10/util/UniqueVoidPtr.h>
// 包含平面哈希映射的实用功能的头文件
#include <c10/util/flat_hash_map.h>
// 包含哈希算法的实用功能的头文件
#include <c10/util/hash.h>
// 包含整数范围的实用功能的头文件
#include <c10/util/irange.h>
// 包含 LLVM 数学额外功能的实用功能的头文件
#include <c10/util/llvmMathExtras.h>
// 包含静态跟踪点的实用功能的头文件
#include <c10/util/static_tracepoint.h>

// 如果不使用 ROCm 并且支持 PyTorch C10 驱动 API，则包含以下头文件
#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)
#include <c10/cuda/driver_api.h>
#include <sys/types.h>
#include <unistd.h>
#endif

// 包含异常处理的实用功能的头文件
#include <c10/util/Exception.h>
// 包含 CUDA 运行时 API 的头文件
#include <cuda_runtime_api.h>

#include <algorithm>  // 包含算法库的头文件
#include <cstddef>    // 包含标准定义的头文件
#include <cstdint>    // 包含标准整数定义的头文件
#include <deque>      // 包含双端队列的头文件
#include <iostream>   // 包含标准输入输出流的头文件
#include <memory>     // 包含内存管理的头文件
#include <mutex>      // 包含互斥锁的头文件
#include <regex>      // 包含正则表达式的头文件
#include <set>        // 包含集合的头文件
#include <utility>    // 包含实用程序的头文件
#include <vector>     // 包含向量的头文件

// 定义用于内存分配追踪的信号量
TORCH_SDT_DEFINE_SEMAPHORE(malloc)
TORCH_SDT_DEFINE_SEMAPHORE(free)

// 定义用于注册自由 CUDA 内存回调的注册表
namespace c10 {
    C10_DEFINE_REGISTRY(FreeCudaMemoryCallbacksRegistry, FreeMemoryCallback);

    // CUDA 缓存分配器的命名空间
    namespace cuda::CUDACachingAllocator {

        // "large" 分配可能会在 20 MiB 块中进行打包
        const size_t kLargeBuffer = 20971520;

        // CUDA 设备本地的实现命名空间
        namespace Native {

            //
            // 另一个 CUDA 设备分配的缓存分配器。
            //
            // - 分配与流相关联。一旦释放，块可以在同一流上重新分配，但不能在任何其他流上重新分配。
            // - 分配器尝试找到适合请求大小的最小缓存块。如果块大于请求的大小，则可以拆分。
            // - 如果没有找到块，则分配器将委托给 cudaMalloc。
            // - 如果 cudaMalloc 失败，则分配器将尝试释放一个足够大小的未拆分的缓存块，并重试分配。
            //   如果这也失败，则分配器将尝试释放所有未拆分的缓存块，并重试分配。
            // - 大（>1MB）和小分配存储在不同的池中。小请求打包到 2MB 缓冲区中。大请求将使用最小的可用空闲块或使用 cudaMalloc 分配新块。
            // - 为了减少碎片化，介于 1MB 和 10MB 之间的请求将分配并拆分一个 20MB 块，如果没有足够大小的空闲块。
            // - 为了进一步减少碎片化，不允许分裂大于 max_split_size 的块。这些超大缓存块仍将满足超过其大小 1MB 的请求。
            //
            // 使用该分配器时，分配和释放应逻辑上被视为与流相关联的内存段的 "使用"，就像内核启动一样。
            // 如果从多个流使用内存段，则程序员必须插入正确的同步。
            //
            // 库提供了一个 recordStream() 函数，以帮助插入正确的同步
            //
/**
 * synchronization when allocations are used on multiple streams. This will
 * ensure that the block is not reused before each recorded stream completes
 * work.
 */

/**
 * Note [Interaction with CUDA graph capture]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Graph capture performs a dry run of a region of execution, freezing all CUDA
 * work (and virtual addresses used during that work) into a "graph." The graph
 * may be "replayed" like a single giant kernel, with greatly reduced CPU
 * overhead as well as modestly improved GPU performance.
 *
 * Because capture bakes in memory addresses, the memory used during capture
 * must be available for the graph to use during replay. DeviceCachingAllocator
 * assigns and frees memory eagerly and dynamically, so if we're not careful
 * about managing graphs' memory, at replay time those memory addresses could be
 * used by other tensors.
 *
 * To guarantee a graph's baked in addresses are safe to reuse in replay,
 * DeviceAllocator satisfies allocations from a graph-private memory pool during
 * capture, and doesn't begin cudaFreeing those addresses until the graph is
 * destroyed.
 *
 * Within the private pool, allocations are freed and reassigned as usual during
 * capture. Memory regions will be used in a consistent order during replay. So
 * a private pool doesn't use memory more wastefully than the default pools
 * during capture, but it does reserve its high-water mark of used memory away
 * from the default pools as long as the capture(s) it served survive
 * (regardless whether those captures are idle or replaying).
 *
 * CUDAGraph's requests for private pools are mediated by
 * DeviceAllocator::notifyCaptureBegin,
 *                  notifyCaptureAboutToEnd,
 *                  notifyCaptureEnded,
 *                  notifyCaptureDestroy.
 */

// Minimum block size for allocations, rounded to 512 bytes
constexpr size_t kMinBlockSize = 512;

// Largest size considered "small" allocation is 1 MiB (1048576 bytes)
constexpr size_t kSmallSize = 1048576;

// "Small" allocations are packed in 2 MiB (2097152 bytes) blocks
constexpr size_t kSmallBuffer = 2097152;

// Minimum size for large allocations, between 1 MiB and 10 MiB (10485760 bytes)
constexpr size_t kMinLargeAlloc = 10485760;

// Round up large allocations to 2 MiB (2097152 bytes)
constexpr size_t kRoundLarge = 2097152;

// Anonymous namespace for local definitions
namespace {

// Alias for a set of CUDA streams using ska::flat_hash_set
using stream_set = ska::flat_hash_set<cuda::CUDAStream>;

// Alias for an array indicating status types
using StatTypes = std::array<bool, static_cast<size_t>(StatType::NUM_TYPES)>;

// Increase the statistics by a specified amount
void increase_stat(Stat& stat, size_t amount) {
  stat.current += static_cast<int64_t>(amount);
  stat.peak = std::max(stat.current, stat.peak);
  stat.allocated += static_cast<int64_t>(amount);
}

// Decrease the statistics by a specified amount
void decrease_stat(Stat& stat, size_t amount) {
  stat.current -= static_cast<int64_t>(amount);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      stat.current >= 0,
      "Negative tracked stat in CUDA allocator (likely logic error).");
  stat.freed += static_cast<int64_t>(amount);
}

// Reset the accumulated statistics
void reset_accumulated_stat(Stat& stat) {
  stat.allocated = 0;
  stat.freed = 0;
}

} // namespace
// 重置给定 Stat 对象的 peak 属性为当前值
void reset_peak_stat(Stat& stat) {
    stat.peak = stat.current;
}

// 遍历 StatTypes 中选中的每一种统计类型，并对每种类型执行函数 f
template <typename Func>
void for_each_selected_stat_type(const StatTypes& stat_types, Func f) {
    // 使用范围遍历 stat_types 的大小
    for (const auto stat_type : c10::irange(stat_types.size())) {
        // 如果 stat_types 中对应位置为真，则调用函数 f 处理该类型
        if (stat_types[stat_type]) {
            f(stat_type);
        }
    }
}

// 减少 StatArray 中每种选定类型的统计数据的值
void decrease_stat_array(
    StatArray& stat_array,
    size_t amount,
    const StatTypes& stat_types) {
  // 对 stat_types 中选中的每一种类型，在 stat_array 中减少对应类型的统计数据
  for_each_selected_stat_type(
      stat_types, [&stat_array, amount](size_t stat_type) {
        decrease_stat(stat_array[stat_type], amount);
      });
}

struct Block;
struct PrivatePool;
typedef bool (*Comparison)(const Block*, const Block*);
static bool BlockComparatorSize(const Block* a, const Block* b);
static bool BlockComparatorAddress(const Block* a, const Block* b);

// BlockPool 结构体，管理块的池子
struct BlockPool {
  // 构造函数，初始化 blocks 和 unmapped 两个集合，以及 is_small 和 owner_PrivatePool 属性
  BlockPool(bool small, PrivatePool* private_pool = nullptr)
      : blocks(BlockComparatorSize),  // 使用 BlockComparatorSize 进行 blocks 的排序
        unmapped(BlockComparatorAddress),  // 使用 BlockComparatorAddress 进行 unmapped 的排序
        is_small(small),  // 设置 is_small 属性
        owner_PrivatePool(private_pool) {}  // 设置 owner_PrivatePool 属性

  // 不要直接向 blocks 集合中插入 Block，使用 insert_into_blocks() 方法
  std::set<Block*, Comparison> blocks;
  std::set<Block*, Comparison> unmapped;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const bool is_small;  // 是否为小块
  PrivatePool* owner_PrivatePool;  // 拥有者 PrivatePool
  int64_t get_free_blocks_call_count{0};  // 获取空闲块的调用计数

  // 向 blocks 集合中插入一个 Block，并更新 gc 计数器
  std::pair<std::set<Block*, Comparison>::iterator, bool> insert_into_blocks(
      Block* block);
};

struct ExpandableSegment;
/*
Note [Expandable Segments]

Rationale

For large (>2MB) allocations, the allocator calls cudaMalloc to get allocations
*/

struct Block {
  c10::DeviceIndex device; // GPU设备索引
  cudaStream_t stream; // 分配流
  stream_set stream_uses; // 使用了该块的流集合
  size_t size; // 块的大小（字节）
  size_t requested_size; // 最初请求的内存大小
  BlockPool* pool{nullptr}; // 拥有该内存池的指针
  void* ptr{nullptr}; // 内存地址
  bool allocated{false}; // 使用标志
  bool mapped{true}; // 虚拟地址范围是否映射到物理页。当 expandable_segment_ 为空时，始终为 true。
                     // 当为 false 时，该块将按其 expandable_segment_ 的段大小对齐。
  Block* prev{nullptr}; // 如果从较大的分配中分割出来，则是前一个块
  Block* next{nullptr}; // 如果从较大的分配中分割出来，则是后一个块
  int event_count{0}; // 未完成的 CUDA 事件数
  int64_t gc_count_base{0}; // 插入 Block 时的 get_free_blocks_call_count 值
  std::shared_ptr<GatheredContext> context_when_allocated;
  // 仅对段中的第一个块（prev == null 时）设置
  // 记录调用 cudaMalloc 时的帧信息
  // 而 context_when_allocated 记录我们最后一次从缓存中分配该内存的时间点。
  std::shared_ptr<GatheredContext> context_when_segment_allocated;

  ExpandableSegment* expandable_segment_{nullptr};

  Block(
      c10::DeviceIndex device,
      cudaStream_t stream,
      size_t size,
      BlockPool* pool,
      void* ptr)
      : device(device),
        stream(stream),
        stream_uses(),
        size(size),
        requested_size(0),
        pool(pool),
        ptr(ptr) {}

  // 用于搜索键的构造函数
  Block(c10::DeviceIndex device, cudaStream_t stream, size_t size)
      : device(device),
        stream(stream),
        stream_uses(),
        size(size),
        requested_size(0) {}

  // 获取 gc_count
  size_t gc_count() {
    TORCH_INTERNAL_ASSERT(pool);
    return static_cast<int>(pool->get_free_blocks_call_count - gc_count_base);
  }

  // 是否已分割
  bool is_split() const {
    return (prev != nullptr) || (next != nullptr);
  }

  // 插入块到链表中
  void splice(Block* before, Block* after) {
    if (before) {
      TORCH_INTERNAL_ASSERT(before->next == after);
      before->next = this;
    }
    prev = before;
    if (after) {
      TORCH_INTERNAL_ASSERT(after->prev == before);
      after->prev = this;
    }
    next = after;
  }
};

// 将块插入到块集合中
std::pair<std::set<Block*, Comparison>::iterator, bool> BlockPool::
    insert_into_blocks(Block* block) {
  block->gc_count_base = get_free_blocks_call_count;
  return blocks.insert(block);
}

struct SegmentRange {
  char* ptr; // 段范围的指针
  size_t size; // 段的大小
  SegmentRange(void* p, size_t s) : ptr(static_cast<char*>(p)), size(s) {}
};

#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)
# 创建可扩展的内存段，允许在需要更多内存时动态扩展段的大小。
# 这种方法不是为每次分配创建一个新段，而是在需要时尝试为每个流创建一个可以动态增长的段。
# 当程序以 N + 1 的批大小运行时，分配将优雅地填充到一个大段中，直到填满为止。
# 然后会请求更多内存，并将其附加到段的末尾。这个过程不会创建太多无法使用的内存碎片，因此更有可能成功找到这些内存。

# 使用 expandable_segments:True 选项来启用/禁用这种行为。
# 我们使用 CUDA 的低级内存 API，类似于 mmap，来扩展内存段。
# 这些 API 将物理内存的分配 (cuMemCreate) 与虚拟地址空间的分配 (cuMemAddressReserve)
# 以及它们之间的关联 (cuMemMap/cuMemSetAccess) 分开。

# 当我们分配新段时，会分配足够的地址空间来映射 GPU 的整个物理内存（有 256TiB 的地址空间），
# 但我们只映射足够当前程序所需的物理内存量。随着需要更多内存，我们会向段添加更多物理内存。
# 这可以在当前的 GPU 页面（每个页面大小为 2MiB）的粒度上工作。

# 如果内存不足，我们可以取消映射段中所有与空物理页面对应的内存，并将其返回给 CUDA，
# 以便在其他地方使用。
/*
struct ExpandableSegment {
  // 构造函数，初始化可扩展段对象
  ExpandableSegment(
      c10::DeviceIndex device,  // 设备索引
      cudaStream_t stream,      // CUDA 流
      size_t size,              // 段的大小
      std::vector<c10::DeviceIndex> peers)  // 相关设备的索引向量
      : device_(device),        // 初始化设备索引
        stream_(stream),
        // 指定小池为2MB，大池为20MB
        segment_size_(size),
        peers_(std::move(peers)) {  // 初始化设备索引向量

    cudaDeviceProp prop{};
    // 获取指定设备的属性信息
    C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, device_));
    // 为了应对需要在段的早期解除映射页面并放置在末尾的情况，分配足够的地址空间
    // 这允许某些情况下我们必须在段的早期解除映射页面并放置在末尾。
    max_handles_ = numSegments(prop.totalGlobalMem + prop.totalGlobalMem / 8);
  C10_CUDA_DRIVER_CHECK(DriverAPI::get()->cuMemAddressReserve_(
      &ptr_, segment_size_ * max_handles_, 0ULL, 0, 0ULL));
}
// begin must be aligned to segment_size_.
// returns the actual range mapped, which may be
// greater than requested if size is not aligned to segment_size_.
// return size of 0 indicates OOM
SegmentRange map(SegmentRange range) {
  auto begin = segmentLeft(range.ptr);
  auto end = segmentRight(range.ptr + range.size);
  TORCH_INTERNAL_ASSERT(ptr() + begin * segment_size_ == range.ptr);
  if (begin == end) {
    return rangeFromHandles(begin, end);
  }
  while (end > handles_.size()) {
    handles_.emplace_back(c10::nullopt);
  }
  for (auto i : c10::irange(begin, end)) {
    TORCH_INTERNAL_ASSERT(!handles_.at(i));
    CUmemGenericAllocationHandle handle = 0;
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    // NOLINTNEXTLINE(bugprone-signed-char-misuse)
    prop.location.id = static_cast<int>(device_);
    auto status =
        DriverAPI::get()->cuMemCreate_(&handle, segment_size_, &prop, 0);
    if (status == CUDA_ERROR_OUT_OF_MEMORY) {
      for (auto j : c10::irange(begin, i)) {
        auto h = handles_.at(j).value();
        handles_.at(j) = c10::nullopt;
        C10_CUDA_DRIVER_CHECK(DriverAPI::get()->cuMemRelease_(h));
      }
      trimHandles();
      return rangeFromHandles(begin, begin);
    }
    C10_CUDA_DRIVER_CHECK(status);
    handles_.at(i) = handle;
  }
  for (auto i : c10::irange(begin, end)) {
    C10_CUDA_DRIVER_CHECK(DriverAPI::get()->cuMemMap_(
        ptr_ + i * segment_size_,
        segment_size_,
        0,
        handles_.at(i).value(),
        0ULL));
  }

  setAccess(device_, begin, end);
  for (auto p : peers_) {
    setAccess(p, begin, end);
  }
  return rangeFromHandles(begin, end);
}

// unmaps all the completely empty segment_size_ segments between
// [begin, begin + size), returns the offset where the range begin,
// and the actual size unmapped (multiple of segment_size_)
SegmentRange unmap(SegmentRange range) {
  auto begin = segmentRight(range.ptr);
  auto end = segmentLeft(range.ptr + range.size);
  if (begin >= end) {
    return SegmentRange{range.ptr, 0};
  }
  unmapHandles(begin, end);
  return rangeFromHandles(begin, end);
}

char* ptr() const {
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  return reinterpret_cast<char*>(ptr_);
}
size_t size() const {
  return max_handles_ * segment_size_;
}

void addPeer(c10::DeviceIndex device) {
  peers_.push_back(device);
  forEachAllocatedRange(
      [&](size_t begin, size_t end) { setAccess(device, begin, end); });
}

~ExpandableSegment() {
  forEachAllocatedRange(
      [&](size_t begin, size_t end) { unmapHandles(begin, end); });
  // 调用 C10_CUDA_DRIVER_CHECK 宏来检查 CUDA 驱动 API 调用的结果，并释放显存地址
  C10_CUDA_DRIVER_CHECK(DriverAPI::get()->cuMemAddressFree_(
      ptr_, segment_size_ * max_handles_));
}

private:
  // 设置访问权限，指定设备、起始和结束位置
  void setAccess(c10::DeviceIndex device, size_t begin, size_t end) {
    CUmemAccessDesc desc;
    desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    // NOLINTNEXTLINE(bugprone-signed-char-misuse)
    desc.location.id = static_cast<int>(device);
    desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    // 调用 CUDA 驱动 API 来设置内存访问权限
    C10_CUDA_DRIVER_CHECK(DriverAPI::get()->cuMemSetAccess_(
        ptr_ + begin * segment_size_, (end - begin) * segment_size_, &desc, 1));
  }

  // 取消映射指定范围内的内存句柄
  void unmapHandles(size_t begin, size_t end) {
    // 注意：与 cudaFree 不同，MemUnmap 和 MemRelease 在所有情况下都不会同步，因此我们必须等待流完成，以确保内存真正释放。

    // 不能调用 c10::cuda::stream_synchronize，因为它可能会获取全局解释器锁（GIL），导致死锁
    // 锁定顺序必须是 GIL -> 分配器锁
    C10_CUDA_CHECK(cudaStreamSynchronize(stream_));
    // 遍历指定范围的内存句柄，取消映射并释放内存
    for (auto i : c10::irange(begin, end)) {
      CUmemGenericAllocationHandle h = handles_.at(i).value();
      handles_.at(i) = c10::nullopt;
      // 调用 CUDA 驱动 API 取消映射指定内存地址
      C10_CUDA_DRIVER_CHECK(DriverAPI::get()->cuMemUnmap_(
          ptr_ + segment_size_ * i, segment_size_));
      // 调用 CUDA 驱动 API 释放指定内存句柄
      C10_CUDA_DRIVER_CHECK(DriverAPI::get()->cuMemRelease_(h));
    }
    // 整理内存句柄数组，删除空的句柄
    trimHandles();
  }

  // 清理 handles_ 数组末尾的空句柄
  void trimHandles() {
    while (!handles_.empty() && !handles_.back()) {
      handles_.pop_back();
    }
  }

  // 对于每个分配的内存范围，执行指定的函数
  void forEachAllocatedRange(const std::function<void(size_t, size_t)>& fn) {
    size_t start = 0;
    // 遍历 handles_ 数组中的句柄
    for (auto i : c10::irange(handles_.size())) {
      // 如果当前位置有句柄，并且前一个位置没有句柄，则更新起始位置
      if (handles_.at(i) && (i == 0 || !handles_.at(i - 1))) {
        start = i;
      }
      // 如果当前位置有句柄，并且下一个位置没有句柄，则调用传入的函数处理范围 [start, i+1)
      if (handles_.at(i) && (i + 1 == handles_.size() || !handles_.at(i + 1))) {
        fn(start, i + 1);
      }
    }
  }

  // 根据内存大小计算分段数量
  size_t numSegments(size_t size) {
    return (size + segment_size_ - 1) / segment_size_;
  }

  // 计算指针 p 所在的左侧段索引
  size_t segmentLeft(char* p) {
    auto size = p - ptr();
    return size / segment_size_;
  }

  // 计算指针 p 所在的右侧段索引
  size_t segmentRight(char* p) {
    auto size = p - ptr();
    return numSegments(size);
  }

  // 根据给定的起始和结束索引，创建 SegmentRange 对象
  SegmentRange rangeFromHandles(size_t begin, size_t end) {
    return SegmentRange(
        ptr() + segment_size_ * begin, segment_size_ * (end - begin));
  }

  c10::DeviceIndex device_;
  cudaStream_t stream_;
  CUdeviceptr ptr_{};
  size_t max_handles_{0};
  size_t segment_size_;
  std::vector<std::optional<CUmemGenericAllocationHandle>> handles_;
  // 存储额外映射此内存的设备索引（除了 device_ 外的其他设备）
  std::vector<c10::DeviceIndex> peers_;
#else
// 定义了一个名为 ExpandableSegment 的结构体，用于表示可扩展的段
struct ExpandableSegment {
  // 构造函数，接收设备索引、CUDA 流、大小和对等设备列表
  ExpandableSegment(
      c10::DeviceIndex device,
      cudaStream_t stream,
      size_t size,
      const std::vector<c10::DeviceIndex>& peers) {
    // 断言，如果执行到此处应该为 false，并输出错误信息
    TORCH_INTERNAL_ASSERT(false, "expandable segment not supported");
  }
  // 将段范围映射的方法，返回空的段范围
  SegmentRange map(SegmentRange range) {
    return SegmentRange(nullptr, 0);
  }
  // 将段范围取消映射的方法，同样返回空的段范围
  SegmentRange unmap(SegmentRange range) {
    return SegmentRange(nullptr, 0);
  }
  // 返回空指针的方法，用于获取指向字符的指针
  char* ptr() const {
    return nullptr;
  }
  // 返回 0 的方法，用于获取大小
  size_t size() const {
    return 0;
  }
  // 添加对等设备的方法，仅声明不做任何操作
  void addPeer(c10::DeviceIndex device) {}
};
#endif

// BlockState, BlockPoolState, and PrivatePoolState 包含了重建私有池到先前状态所需的信息。
// 详见 [Checkpointing PrivatePoolState] 注释
struct BlockState {
  // 设备索引，默认为 0
  c10::DeviceIndex device = 0;
  // CUDA 流，默认为 nullptr
  cudaStream_t stream = nullptr;
  // 使用流集合，默认为空集合
  stream_set stream_uses = {};
  // 大小，默认为 0
  size_t size = 0;
  // 指针，默认为 nullptr
  void* ptr = nullptr;
  // 是否已分配，默认为 false
  bool allocated = false;
  // 垃圾回收计数基础，默认为 0
  int64_t gc_count_base = 0;
  // 保持事件计数等于 0 的不变性；
  // 历史将在检查点中保持不变
  // 注释声明，不执行任何实际操作
  BlockState(Block* block);
};

struct SegmentState {
  // 块状态向量
  std::vector<BlockState> blocks;
  // 是否是小段，默认为 false
  bool is_small = false;

  // 构造函数，接收块头指针作为参数
  SegmentState(Block* head);
};

struct PrivatePoolState : AllocatorState {
  // 省略 use_count 和 cudaMalloc_count，因为它们保持不变
  // 拥有者 ID，默认为 {0, 0}
  MempoolId_t owner_id = {0, 0};
  
  // 段状态向量
  std::vector<SegmentState> segments;

  // 构造函数，接收池 ID 和私有池头块向量作为参数
  PrivatePoolState(
      MempoolId_t pool_id,
      const std::vector<Block*>& private_pool_head_blocks);
};

struct RestoreResult {
  // 释放的分配空间指针向量
  std::vector<void*> allocations_freed;
  // 创建的分配块向量
  std::vector<Block*> allocations_created;
};

// 静态函数，用于比较块大小的排序，返回较小的块
static bool BlockComparatorSize(const Block* a, const Block* b) {
  // 如果流不同，则按流地址排序
  if (a->stream != b->stream) {
    return (uintptr_t)a->stream < (uintptr_t)b->stream;
  }
  // 否则按块大小排序
  if (a->size != b->size) {
    return a->size < b->size;
  }
  // 最后按块指针地址排序
  return (uintptr_t)a->ptr < (uintptr_t)b->ptr;
}

// 静态函数，用于比较块地址的排序，返回较小的块
static bool BlockComparatorAddress(const Block* a, const Block* b) {
  // 如果流不同，则按流地址排序
  if (a->stream != b->stream) {
    return (uintptr_t)a->stream < (uintptr_t)b->stream;
  }
  // 否则按块指针地址排序
  return (uintptr_t)a->ptr < (uintptr_t)b->ptr;
}

// 分配参数结构体，包含设备索引、大小、CUDA 流、块池、分配大小和设备统计信息的引用
struct AllocParams {
  // 构造函数，初始化搜索关键字、块池和分配大小
  AllocParams(
      c10::DeviceIndex device,
      size_t size,
      cudaStream_t stream,
      BlockPool* pool,
      size_t alloc_size,
      DeviceStats& stats)
      : search_key(device, stream, size), pool(pool), alloc_size(alloc_size) {}

  // 返回设备索引的方法
  c10::DeviceIndex device() const {
    return search_key.device;
  }
  // 返回 CUDA 流的方法
  cudaStream_t stream() const {
    return search_key.stream;
  }
  // 返回大小的方法
  size_t size() const {
    return search_key.size;
  }

  // 搜索关键字块
  Block search_key;
  // 块池指针
  BlockPool* pool;
  // 分配大小
  size_t alloc_size;
  // 块指针，默认为 nullptr
  Block* block{nullptr};
  // 统计类型，默认为 false
  StatTypes stat_types = {false};
  // CUDA 错误，默认为 cudaSuccess
  cudaError_t err{cudaSuccess};
};

// 注意：当同时从多个线程并发调用 cudaEventCreate 时可能非常昂贵（至少在某些设备/驱动组合上是如此）。
// 因此，我们 a) 在每个设备级别上串行化事件创建，b) 对事件进行池化
// 避免频繁调用 cudaEventCreate 和 cudaEventDestroy。这会显著提升在高分配速率的多线程工作负载下的性能。
class EventPool {
 public:
  using Event = std::unique_ptr<cudaEvent_t, std::function<void(cudaEvent_t*)>>;
  // TODO: 显式设备数量
  EventPool() : pools_(at::cuda::device_count()) {}

  // 获取指定设备的事件对象
  Event get(c10::DeviceIndex device) {
    TORCH_INTERNAL_ASSERT(0 <= device);
    TORCH_INTERNAL_ASSERT(device < static_cast<int>(pools_.size()));
    auto& pool = pools_[device];
    auto destructor = [&pool](cudaEvent_t* event) {
      std::lock_guard<std::mutex> g(pool.mutex_);
      pool.event_pool_.push_back(std::unique_ptr<cudaEvent_t>(event));
    };

    // 尝试从每个设备的池中获取事件
    {
      std::lock_guard<std::mutex> g(pool.mutex_);
      if (!pool.event_pool_.empty()) {
        auto* event = pool.event_pool_.back().release();
        pool.event_pool_.pop_back();
        return Event(event, destructor);
      }
    }
    // 否则，分配一个新的事件，并在销毁时返回到池中
    auto new_ptr = std::make_unique<cudaEvent_t>();
    C10_CUDA_CHECK(
        cudaEventCreateWithFlags(new_ptr.get(), cudaEventDisableTiming));

    return Event(new_ptr.release(), destructor);
  }

  // 清空所有池中的缓存
  void empty_cache() {
    for (auto& pool : pools_) {
      std::lock_guard<std::mutex> g(pool.mutex_);
      pool.event_pool_.clear();
    }
  }

 private:
  // 每个设备的事件池结构体
  struct PerDevicePool {
    alignas(64) std::mutex mutex_;
    std::vector<std::unique_ptr<cudaEvent_t>> event_pool_;
  };
  std::vector<PerDevicePool> pools_;
};

// CUDA 图表辅助结构体
struct PrivatePool {
  PrivatePool()
      : large_blocks(/*small=*/false, this),
        small_blocks(/*small=*/true, this) {}
  PrivatePool(const PrivatePool&) = delete;
  PrivatePool(PrivatePool&&) = delete;
  PrivatePool& operator=(const PrivatePool&) = delete;
  // 使用此池的活跃图表数目
  int use_count{1};
  // 为此池分配但尚未释放的 cudaMalloc 数量。当 use_count 和 cudaMalloc_count
  // 都为零时，可以从 graph_pools 中删除此 PrivatePool。
  int cudaMalloc_count{0};
  // 不在此处维护私有 BlockPools，而是将所有块（私有或公共）放入顶级的
  // large_blocks 和 small_blocks 中，并通过在 BlockComparator 中的流检查上方
  // 添加 "pool id" 检查来区分私有块。然而，BlockComparator 是性能关键的部分，
  // 我不愿意给它增加更多逻辑。
  BlockPool large_blocks;
  BlockPool small_blocks;
};

// BlockState 类的构造函数
BlockState::BlockState(Block* block)
    : device(block->device),
      stream(block->stream),
      stream_uses(block->stream_uses),
      size(block->size),
      ptr(block->ptr),
      allocated(block->allocated),
      gc_count_base(block->gc_count_base) {
  TORCH_CHECK(
      block->event_count == 0,
      "Events should have synchronized when checkpointing block");
};
// 构造函数，初始化 SegmentState 对象，从给定的链表头开始
SegmentState::SegmentState(Block* head) {
  // 检查链表头的前一个节点必须为空，并且池子不能为空
  TORCH_INTERNAL_ASSERT(head->prev == nullptr && head->pool != nullptr);
  // 设置 is_small 标志为头部池子的 is_small 属性
  is_small = head->pool->is_small;

  // 遍历链表中所有块，将它们添加到 blocks 容器中
  for (Block* curr = head; curr != nullptr; curr = curr->next) {
    blocks.emplace_back(curr);
  }
}

// 构造函数，初始化 PrivatePoolState 对象
PrivatePoolState::PrivatePoolState(
    MempoolId_t pool_id,
    const std::vector<Block*>& private_pool_head_blocks)
    : owner_id(std::move(pool_id)) {
  // 遍历私有池头部块的列表，将它们添加到 segments 容器中
  for (Block* head : private_pool_head_blocks) {
    segments.emplace_back(head);
  }
}

// MempoolId_t 的哈希函数对象
struct MempoolIdHash {
  std::size_t operator()(const MempoolId_t& mempool_id) const noexcept {
    // 如果 mempool_id 的第一个成员不为 0，返回第一个成员，否则返回第二个成员
    return mempool_id.first != 0 ? mempool_id.first : mempool_id.second;
  }
};

// 可能捕获 cudaMalloc 的函数，根据当前 CUDA 流的捕获状态来决定是否捕获
cudaError_t cudaMallocMaybeCapturing(void** p, size_t size) {
  if (at::cuda::currentStreamCaptureStatusMayInitCtx() ==
      at::cuda::CaptureStatus::None) {
    // 如果当前 CUDA 流不是捕获状态，直接调用 cudaMalloc
    return C10_CUDA_ERROR_HANDLED(cudaMalloc(p, size));
  } else {
    // 如果当前 CUDA 流是捕获状态，设置为放松的 CUDA 流捕获模式
    // 并且调用 cudaMalloc
    at::cuda::CUDAStreamCaptureModeGuard g{cudaStreamCaptureModeRelaxed};
    return C10_CUDA_ERROR_HANDLED(cudaMalloc(p, size));
  }
}

// 静态函数，返回关于进程内存信息的报告字符串
static std::string reportProcessMemoryInfo(c10::DeviceIndex device) {
#ifdef PYTORCH_C10_DRIVER_API_SUPPORTED
  // 获取 NVML 句柄
  void* nvml_handle = DriverAPI::get_nvml_handle();
  if (!nvml_handle) {
    return "";  // 如果没有 NVML 句柄，返回空字符串
  }
  // 初始化 NVML
  static c10::once_flag nvml_init;
  c10::call_once(nvml_init, [] {
    TORCH_INTERNAL_ASSERT(NVML_SUCCESS == DriverAPI::get()->nvmlInit_v2_());
  });

  // 获取 CUDA 设备属性
  cudaDeviceProp prop{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  // 获取 CUDA 设备的 PCI 总线 ID
  char pci_id[80];
  snprintf(
      pci_id,
      sizeof(pci_id),
      NVML_DEVICE_PCI_BUS_ID_FMT,
      prop.pciDomainID,
      prop.pciBusID,
      prop.pciDeviceID);

  // 获取通过 PCI 总线 ID 获取的 NVML 设备句柄
  nvmlDevice_t nvml_device = nullptr;
  TORCH_INTERNAL_ASSERT(
      NVML_SUCCESS ==
      DriverAPI::get()->nvmlDeviceGetHandleByPciBusId_v2_(
          pci_id, &nvml_device));

  // 获取 CUDA 设备上运行的进程信息
  std::vector<nvmlProcessInfo_v1_t> procs(8);
  unsigned int size = procs.size();
  nvmlReturn_t r{};
  while ((r = DriverAPI::get()->nvmlDeviceGetComputeRunningProcesses_(
              nvml_device, &size, procs.data())) ==
         NVML_ERROR_INSUFFICIENT_SIZE) {
    procs.resize(size);
  }
  unsigned int self_pid = getpid();
  std::stringstream ss;
  TORCH_INTERNAL_ASSERT(NVML_SUCCESS == r);
  ss << "";  // 初始化字符串流

  // 遍历进程列表，生成进程内存信息报告
  for (auto i : c10::irange(size)) {
    auto& proc = procs[i];
    if (self_pid == proc.pid) {
      ss << "Including non-PyTorch memory, this process";
    } else {
      ss << "Process " << proc.pid;
    }
    ss << " has " << format_size(proc.usedGpuMemory) << " memory in use. ";
  }
  return ss.str();  // 返回生成的报告字符串
#else
  return "";  // 如果未定义 PYTORCH_C10_DRIVER_API_SUPPORTED，返回空字符串
#endif
}
  // 锁，用于保护所有操作的互斥访问
  mutable std::recursive_mutex mutex;

  // 设备统计信息
  DeviceStats stats;

  // 大于1MB的未分配缓存块
  BlockPool large_blocks;

  // 小于等于1MB的未分配缓存块
  BlockPool small_blocks;

  // 活跃的分配块集合，包括从graph_pools或上述BlockPools中获取的所有活跃分配
  ska::flat_hash_set<Block*> active_blocks;

  // captures_underway 跟踪是否正在将某些分配重定向到特定池
  // 大部分时间为空，此时malloc可以避免在热路径中调用cudaStreamGetCaptureInfo
  std::vector<std::pair<MempoolId_t, std::function<bool(cudaStream_t)>>>
      captures_underway;

  // needs_events_deferred_until_no_capture 用于跟踪需要延迟事件的块，直到不再捕获
  std::vector<Block*> needs_events_deferred_until_no_capture;

  // 未完成的cuda事件
  ska::flat_hash_map<
      cuda::CUDAStream,
      std::deque<std::pair<EventPool::Event, Block*>>>
      cuda_events;

  // 记录已使用内存
  size_t total_allocated_memory = 0;

  // 允许的最大内存
  size_t allowed_memory_maximum = 0;

  // 所有活跃的可扩展段
  std::vector<ExpandableSegment*> expandable_segments_;

  // 具有对等访问权限的设备索引
  std::vector<c10::DeviceIndex> devices_with_peer_access_;

  // 设置分数的标志
  bool set_fraction = false;

  // 记录历史的标志
  bool record_history = false;

  // 记录上下文创建函数的原子变量
  std::atomic<CreateContextFn> context_recorder_;

  // 下一个分配跟踪索引
  size_t alloc_trace_next = 0;

  // 记录上下文的类型
  RecordContext record_context_ = RecordContext::NEVER;

  // 分配跟踪的最大条目数
  size_t alloc_trace_max_entries_ = 1;

  // 分配跟踪的指针，必须故意泄漏，因为它可能持有对已销毁的Python状态的引用
  std::vector<TraceEntry>* alloc_trace;

  // CUDA图特定的成员

  // 私有的CUDA图池
  ska::flat_hash_map<MempoolId_t, std::unique_ptr<PrivatePool>, MempoolIdHash>
      graph_pools;

  // 不再由任何图引用的池，它们的BlockPools可以释放
  ska::flat_hash_map<MempoolId_t, PrivatePool*, MempoolIdHash>
      graph_pools_freeable;

  // 内存不足观察器列表
  std::vector<OutOfMemoryObserver> oom_observers_;

  // 跟踪分配器的追踪器列表
  std::vector<AllocatorTraceTracker> trace_trackers_;

  // 将块映射到流集，包含块在捕获CUDA图期间使用的流
  std::unordered_map<Block*, stream_set> block_to_cudagraph_stream_uses;
  
public:
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  // 构造函数，初始化大块和小块BlockPools以及分配跟踪
  DeviceCachingAllocator()
      : large_blocks(/*small=*/false),
        small_blocks(/*small=*/true),
        alloc_trace(new std::vector<TraceEntry>()) {
  // 设置最大分割大小为 CUDA 分配器配置的最大分割大小
  stats.max_split_size =
      static_cast<int64_t>(CUDAAllocatorConfig::max_split_size());
  // 将上下文记录器设为 nullptr，即不记录上下文
  context_recorder_.store(nullptr);
}

// 记录历史操作的方法
void recordHistory(
    bool enabled,                          // 是否启用记录历史
    CreateContextFn context_recorder,       // 上下文记录器的创建函数
    size_t alloc_trace_max_entries,         // 分配跟踪最大条目数
    RecordContext when) {                   // 记录上下文的时机
  std::unique_lock<std::recursive_mutex> lock(mutex);  // 获取递归互斥锁
  TORCH_CHECK(when == RecordContext::NEVER || context_recorder);  // 断言检查：当记录上下文时机不是 NEVER 时，上下文记录器不能为空
  record_history = enabled;                 // 设置记录历史标志
  context_recorder_.store(record_history ? context_recorder : nullptr);  // 根据是否启用记录历史，设置上下文记录器
  alloc_trace_max_entries_ = std::max(size_t(1), alloc_trace_max_entries);  // 设置分配跟踪最大条目数，至少为1
  record_context_ = enabled ? when : RecordContext::NEVER;  // 根据是否启用记录历史，设置记录上下文时机
  if (!enabled) {                           // 如果未启用记录历史
    alloc_trace_next = 0;                   // 重置分配跟踪的下一个位置
    alloc_trace->clear();                   // 清空分配跟踪数据
  }
}

// 记录用户定义注解的方法
void recordAnnotation(const std::shared_ptr<GatheredContext>& name) {
  record_trace(TraceEntry::USER_DEFINED, 0, 0, nullptr, 0, name);  // 调用记录追踪方法，记录用户定义注解
}

// 返回历史记录是否启用的方法
bool isHistoryEnabled() {
  return record_history;                    // 返回记录历史的状态
}

// 检查池中活跃分配的方法
bool checkPoolLiveAllocations(
    MempoolId_t mempool_id,                 // 内存池 ID
    const std::unordered_set<void*>& expected_live_allocations) {  // 预期的活跃分配集合
  std::unique_lock<std::recursive_mutex> lock(mutex);  // 获取递归互斥锁

  PrivatePool* pool = nullptr;              // 私有池指针
  auto pool_it = graph_pools.find(mempool_id);  // 查找指定 ID 的图池
  TORCH_CHECK(pool_it != graph_pools.end(), "Could not find pool of id");  // 断言检查：确保找到了指定 ID 的图池
  pool = pool_it->second.get();             // 获取找到的图池指针

  TORCH_INTERNAL_ASSERT(pool != nullptr);   // 内部断言：确保池指针不为空

  size_t allocated_pool_blocks = 0;         // 分配给池的块数

  for (Block* b : active_blocks) {          // 遍历活跃块列表
    TORCH_INTERNAL_ASSERT(b != nullptr);    // 内部断言：确保块指针不为空
    TORCH_INTERNAL_ASSERT(b->pool != nullptr);  // 内部断言：确保块所属池不为空
    if (b->allocated && b->pool->owner_PrivatePool == pool) {  // 如果块已分配且其所属池是指定的私有池
      if (!expected_live_allocations.count(b->ptr)) {  // 如果预期的活跃分配集合中不包含该块的指针
        return false;                        // 返回 false，表示检查未通过
      }

      allocated_pool_blocks += 1;            // 增加分配给池的块数计数
    }
  }

  return allocated_pool_blocks == expected_live_allocations.size();  // 返回是否分配给池的块数与预期的活跃分配集合大小相等
}

// 添加内存不足观察器的方法
void attachOutOfMemoryObserver(OutOfMemoryObserver observer) {
  oom_observers_.emplace_back(std::move(observer));  // 将内存不足观察器移入观察器列表
}

// 添加分配器跟踪追踪器的方法
void attachAllocatorTraceTracker(AllocatorTraceTracker tracker) {
  std::unique_lock<std::recursive_mutex> lock(mutex);  // 获取递归互斥锁
  trace_trackers_.emplace_back(std::move(tracker));  // 将分配器跟踪追踪器移入追踪器列表
}

// 可能收集上下文的方法，必须在不使用互斥锁的情况下调用，以避免与 Python 的死锁
std::shared_ptr<GatheredContext> maybeGatherContext(RecordContext level) {
  if (record_context_ < level) {
    return nullptr;                        // 如果记录上下文的时机小于指定级别，则返回空指针
  }
  return context_recorder_.load()();       // 否则加载并返回上下文记录器生成的上下文
}

// 所有公共方法（除了上面的方法）都会获取分配器互斥锁
// 因此，不要从另一个公共方法调用公共方法。
// 分配内存的方法
Block* malloc(
    c10::DeviceIndex device,                // 设备索引
    size_t orig_size,                       // 原始大小
    cudaStream_t stream) {                  // CUDA 流
  // 在锁之外完成，因为我们不知道记录器需要哪些锁...
  auto context = maybeGatherContext(RecordContext::STATE);  // 可能收集状态上下文

  std::unique_lock<std::recursive_mutex> lock(mutex);  // 获取递归互斥锁
    // 如果当前没有进行中的捕获操作
    if (C10_LIKELY(captures_underway.empty())) {
      // 处理未完成的分配的生命周期事件，这些分配在多个流中使用
      // （检查它们在GPU端的使用是否完成，如果完成则回收它们的内存）
      //
      // Q. 为什么在可能进行捕获时跳过 process_events？
      // A. process_events 包括 cudaEventQueries，在 CUDA 图捕获期间是非法的。
      //    简单愚蠢的解决方案：推迟回收这些分配，直到捕获完成后。跨流内存使用并不常见，
      //    因此推迟在捕获期间对内存使用的影响应该很小。
      process_events(context);
    }
    // 将原始大小舍入为合适的大小
    size_t size = round_size(orig_size);
    // 获取适合该大小和流的内存池
    auto& pool = get_pool(size, stream);
    // 获取分配大小
    const size_t alloc_size = get_allocation_size(size);
    // 构造分配参数
    AllocParams params(device, size, stream, &pool, alloc_size, stats);
    // 获取适合该池的统计类型
    params.stat_types = get_stat_types_for_pool(pool);

    // 首先尝试从现有池中获取一个块
    bool block_found =
        // 在池中查找空闲块
        get_free_block(params)
        // 触发回调并重试查找
        || (trigger_free_memory_callbacks(params) && get_free_block(params));

    // 无法重用现有块；尝试获取一个新块
    if (!block_found) {
      // 如果设置了分数并且垃圾回收阈值大于0
      if (C10_UNLIKELY(
              set_fraction &&
              CUDAAllocatorConfig::garbage_collection_threshold() > 0.0)) {
        // 执行缓存块的垃圾回收
        garbage_collect_cached_blocks(context);
      }
      // 尝试分配块
      // 警告：在调用 cudaMalloc 时，alloc_block 可能会释放分配器锁。
      // 到目前为止，此函数尚未修改分配器状态，但请记住，alloc_block 的每次调用
      // 可能会跨调用释放锁，因此观察到的任何分配器状态可能会在调用之间发生更改。
      block_found = alloc_block(params, false, context, lock)
          // 释放足够的可用缓存块以满足分配并重试分配
          || (release_available_cached_blocks(params, context) &&
              alloc_block(params, false, context, lock))
          // 释放所有非拆分的缓存块并重试分配
          || (C10_LIKELY(captures_underway.empty()) &&
              release_cached_blocks(context) &&
              alloc_block(params, true, context, lock));
    }

    }

    // 检查是否应该拆分剩余块
    bool split_remainder = should_split(params.block, params.size());
    // 返回找到的分配块
    return alloc_found_block(
        params, orig_size, std::move(context), split_remainder);
  }

  // 分配找到的块
  Block* alloc_found_block(
      const AllocParams& params,
      size_t orig_size,
      std::shared_ptr<GatheredContext> context,
      bool split_remainder) {
    auto size = params.size();
    auto device = params.device();
    auto pool = params.pool;
    auto stream = params.stream();

    // 内部断言：参数错误为空或块指针为空
    TORCH_INTERNAL_ASSERT(
        params.err == cudaSuccess && params.block != nullptr &&
        params.block->ptr != nullptr);
    // 获取分配块
    Block* block = params.block;
    // 声明一个指向 Block 类型的指针 remaining，并初始化为 nullptr
    Block* remaining = nullptr;
    
    // 检查当前 block 是否已经被分割过
    const bool already_split = block->is_split();
    
    // 如果需要分割剩余部分
    if (split_remainder) {
      // 将当前 block 赋值给 remaining
      remaining = block;
    
      // 创建一个新的 Block 对象，并初始化其属性
      block = new Block(device, stream, size, pool, block->ptr);
    
      // 继承 remaining 的 expandable_segment_ 属性
      block->expandable_segment_ = remaining->expandable_segment_;
    
      // 继承 remaining 的 prev 属性
      block->prev = remaining->prev;
      if (block->prev) {
        block->prev->next = block;
      }
    
      // 设置 block 的 next 属性为 remaining
      block->next = remaining;
    
      // 设置 remaining 的 prev 属性为 block
      remaining->prev = block;
    
      // 更新 remaining 的 ptr 指针，使其指向剩余部分的新位置
      remaining->ptr = static_cast<char*>(remaining->ptr) + size;
    
      // 更新 remaining 的 size 属性，减去已分配的大小
      remaining->size -= size;
    
      // 将 remaining 插入到内存池 pool 中，并返回插入状态
      bool inserted = pool->insert_into_blocks(remaining).second;
    
      // 断言插入操作成功
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(inserted);
    
      // 如果 block 之前已经被分割过，并且不是可扩展段
      if (already_split && !block->expandable_segment_) {
        // 减少统计数据中 inactive_split_bytes 的统计值
        decrease_stat_array(
            stats.inactive_split_bytes, block->size, params.stat_types);
      } else if (!block->expandable_segment_) {
        // 创建一个新的分割的 inactive block，增加相应的统计数据
        for_each_selected_stat_type(params.stat_types, [&](size_t stat_type) {
          increase_stat(stats.inactive_split_bytes[stat_type], remaining->size);
          increase_stat(stats.inactive_split[stat_type], 1);
        });
      }
    
    // 如果 block 已经被分割过，并且不是可扩展段
    } else if (already_split && !block->expandable_segment_) {
      // 将 block 变为 active 状态，减少相应的统计数据
      for_each_selected_stat_type(params.stat_types, [&](size_t stat_type) {
        decrease_stat(stats.inactive_split_bytes[stat_type], block->size);
        decrease_stat(stats.inactive_split[stat_type], 1);
      });
    }
    
    // 将 block 标记为已分配状态
    block->allocated = true;
    
    // 记录 block 的请求大小
    block->requested_size = orig_size;
    
    // 将分配时的上下文信息移动给 block
    block->context_when_allocated = std::move(context);
    
    // 记录内存分配的跟踪信息
    record_trace(
        TraceEntry::ALLOC,
        int64_t(block->ptr),
        orig_size,
        block->stream,
        block->device,
        block->context_when_allocated);
    
    // 将 block 插入到活跃块集合中，并返回插入状态
    bool inserted = active_blocks.insert(block).second;
    
    // 断言插入操作成功
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(inserted);
    
    // 更新相应的统计数据：增加 allocation、allocated_bytes、active 和 active_bytes
    for_each_selected_stat_type(params.stat_types, [&](size_t stat_type) {
      increase_stat(stats.allocation[stat_type], 1);
      increase_stat(stats.allocated_bytes[stat_type], block->size);
      increase_stat(stats.active[stat_type], 1);
      increase_stat(stats.active_bytes[stat_type], block->size);
      increase_stat(stats.requested_bytes[stat_type], block->requested_size);
    });
    
    // 如果 block 的大小超过 CUDAAllocatorConfig 的最大分割大小，则增加 oversize_allocations 统计值
    if (block->size >= CUDAAllocatorConfig::max_split_size())
      increase_stat(stats.oversize_allocations, 1);
    
    // 向性能分析器报告内存使用情况
    c10::reportMemoryUsageToProfiler(
        block->ptr,
        static_cast<int64_t>(block->size),
        stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
        stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
        c10::Device(c10::DeviceType::CUDA, device));
    return block;
  }

  // 释放给定的内存块
  void free(Block* block) {
    // 获取可能的上下文信息，并锁定递归互斥量
    std::shared_ptr<GatheredContext> context =
        maybeGatherContext(RecordContext::ALL);
    std::lock_guard<std::recursive_mutex> lock(mutex);

    // 标记该块为未分配状态
    block->allocated = false;

    // 存储原始块指针和大小，用于报告之前可能会修改底层块的逻辑
    auto orig_block_ptr = block->ptr;
    auto orig_block_size = block->size;

    // 获取适用于池的统计类型，并逐个减少相应的统计数据
    StatTypes stat_types = get_stat_types_for_pool(*block->pool);
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      decrease_stat(stats.allocation[stat_type], 1);
      decrease_stat(stats.allocated_bytes[stat_type], block->size);
    });

    // 记录释放操作的追踪信息，包括指针、请求大小、流、设备以及分配时的上下文
    record_trace(
        TraceEntry::FREE_REQUESTED,
        int64_t(block->ptr),
        block->requested_size,
        block->stream,
        block->device,
        context ? context : block->context_when_allocated);

    // 如果块的大小超过配置的最大分割大小，则增加超大分配统计
    if (block->size >= CUDAAllocatorConfig::max_split_size())
      decrease_stat(stats.oversize_allocations, 1);

    // 如果块使用了流，则根据情况推迟事件记录或直接插入事件
    if (!block->stream_uses.empty()) {
      if (C10_UNLIKELY(!captures_underway.empty())) {
        // 在 CUDA 图捕获期间禁止对记录的事件进行 cudaEventQuery。
        // 保守地推迟记录生命周期事件，直到下次调用 process_events()。
        needs_events_deferred_until_no_capture.push_back(block);
      } else {
        insert_events(block);
      }
    } else {
      // 否则直接释放该块
      free_block(block, context);
    }

    // 向分析器报告内存使用情况的变化，包括释放的大小以及当前的分配和保留字节
    c10::reportMemoryUsageToProfiler(
        orig_block_ptr,
        -static_cast<int64_t>(orig_block_size),
        stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
        stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
        c10::Device(c10::DeviceType::CUDA, block->device));
  }

  // 获取给定内存块的基本分配指针，并返回其总大小（如果提供了 outSize 指针）
  void* getBaseAllocation(Block* block, size_t* outSize) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    // 检查是否为可扩展段分配的张量，如果是，抛出错误
    TORCH_CHECK(
        !block->expandable_segment_,
        "Tensors allocated with expandable_segments:True cannot be shared between processes. Consider using expandable_segments:False in data loading workers via torch.cuda.memory._set_allocator_settings('expandable_segments:False')");

    // 移动到块链表的头部，找到基本分配指针
    while (block->prev) {
      block = block->prev;
    }
    void* basePtr = block->ptr;
    // 如果提供了 outSize 指针，则计算总大小并存储到 outSize
    if (outSize) {
      size_t size = 0;
      while (block) {
        size += block->size;
        block = block->next;
      }
      *outSize = size;
    }
    return basePtr;
  }

  // 记录给定内存块使用的流
  void recordStream(Block* block, cuda::CUDAStream stream) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    // 如果流与分配流相同，则忽略此次记录
    if (stream.stream() == block->stream) {
      // ignore uses on the allocation stream, since those don't require any
      // special synchronization
      return;
    }
    // 否则将该流添加到块的流使用集合中
    block->stream_uses.insert(stream);
    if (C10_UNLIKELY(!captures_underway.empty())) {
      // 如果 captures_underway 不为空，则将当前 block 对应的 stream 插入 block_to_cudagraph_stream_uses 中
      block_to_cudagraph_stream_uses[block].insert(stream);
    }
  }

  /** set memory fraction to limit maximum allocated memory **/
  void setMemoryFraction(double fraction) {
    // 获取当前设备的空闲和总内存信息
    size_t device_free = 0;
    size_t device_total = 0;
    C10_CUDA_CHECK(cudaMemGetInfo(&device_free, &device_total));
    // 根据给定的分数 fraction 设置允许的最大内存限制
    allowed_memory_maximum =
        static_cast<size_t>(fraction * static_cast<double>(device_total));
    set_fraction = true;
  }

  /** returns cached blocks to the system allocator **/
  void emptyCache() {
    // 获取所有上下文记录，可能汇总所有上下文
    auto context = maybeGatherContext(RecordContext::ALL);
    // 锁定递归互斥锁，确保线程安全
    std::lock_guard<std::recursive_mutex> lock(mutex);
    // 释放缓存中的所有块
    release_cached_blocks(context);
  }

  /** Retrieves size of largest unused block held by the memory cache **/
  void cacheInfo(size_t* largest) {
    // 锁定递归互斥锁，确保线程安全
    std::lock_guard<std::recursive_mutex> lock(mutex);
    // 如果传入的 largest 是 0，则初始化一个乐观的初始猜测
    if (*largest == 0) {
      size_t tmp_bytes = 0;
      // 使用空闲内存作为 *largest 的乐观初始猜测
      C10_CUDA_CHECK(cudaMemGetInfo(
          largest,
          &tmp_bytes));
    }
    // 获取各个缓存池中的最大未使用块的大小信息
    cache_info_aux(large_blocks, largest);
    cache_info_aux(small_blocks, largest);
    // 遍历每个图形池，获取其大块和小块的最大未使用块的大小信息
    for (const auto& gp : graph_pools) {
      cache_info_aux(gp.second->large_blocks, largest);
      cache_info_aux(gp.second->small_blocks, largest);
    }
  }

  /** Returns a copy of the memory allocator stats **/
  DeviceStats getStats() {
    // 锁定递归互斥锁，确保线程安全
    std::lock_guard<std::recursive_mutex> lock(mutex);
    // 返回内存分配器统计信息的副本
    return stats;
  }

  /** Resets the historical accumulation stats for the device **/
  void resetAccumulatedStats() {
    // 锁定递归互斥锁，确保线程安全
    std::lock_guard<std::recursive_mutex> lock(mutex);

    // 重置设备的历史累积统计信息
    for (const auto statType :
         c10::irange(static_cast<size_t>(StatType::NUM_TYPES))) {
      reset_accumulated_stat(stats.allocation[statType]);
      reset_accumulated_stat(stats.segment[statType]);
      reset_accumulated_stat(stats.active[statType]);
      reset_accumulated_stat(stats.inactive_split[statType]);
      reset_accumulated_stat(stats.allocated_bytes[statType]);
      reset_accumulated_stat(stats.reserved_bytes[statType]);
      reset_accumulated_stat(stats.active_bytes[statType]);
      reset_accumulated_stat(stats.inactive_split_bytes[statType]);
      reset_accumulated_stat(stats.requested_bytes[statType]);
    }

    stats.num_alloc_retries = 0;
    stats.num_ooms = 0;
    stats.num_sync_all_streams = 0;
    stats.num_device_alloc = 0;
    stats.num_device_free = 0;
    reset_accumulated_stat(stats.oversize_allocations);
    reset_accumulated_stat(stats.oversize_segments);
  }

  /** Resets the historical peak stats for the device **/
  void resetPeakStats() {
    // 锁定递归互斥锁，确保线程安全
    std::lock_guard<std::recursive_mutex> lock(mutex);
    // 重置设备的历史峰值统计信息
    // （此处还有未完整的代码，需要补充完整）
    // 遍历所有统计类型
    for (const auto statType :
         c10::irange(static_cast<size_t>(StatType::NUM_TYPES))) {
      // 重置各类统计数据的峰值
      reset_peak_stat(stats.allocation[statType]);
      reset_peak_stat(stats.segment[statType]);
      reset_peak_stat(stats.active[statType]);
      reset_peak_stat(stats.inactive_split[statType]);
      reset_peak_stat(stats.allocated_bytes[statType]);
      reset_peak_stat(stats.reserved_bytes[statType]);
      reset_peak_stat(stats.active_bytes[statType]);
      reset_peak_stat(stats.inactive_split_bytes[statType]);
      reset_peak_stat(stats.requested_bytes[statType]);
    }
    // 重置超大分配和超大段的统计峰值
    reset_peak_stat(stats.oversize_allocations);
    reset_peak_stat(stats.oversize_segments);
  }

  /* Checkpoint the state of a private pool necessary to return it to its
   * current state */
  // 获取私有池的检查点状态，以便恢复到当前状态
  std::unique_ptr<PrivatePoolState> getCheckpointState(MempoolId_t id) {
    auto context = maybeGatherContext(RecordContext::ALL);
    std::lock_guard<std::recursive_mutex> lock(mutex);
    insert_events_deferred_until_no_capture(context);

    auto pool = graph_pools.find(id);
    if (pool != graph_pools.end()) {
      auto private_pool_head_blocks =
          get_private_pool_head_blocks(pool->second.get());
      return std::make_unique<PrivatePoolState>(id, private_pool_head_blocks);
    } else if (graph_pools_freeable.count(id)) {
      TORCH_CHECK(false, "Not expected to checkpoint freeable graph");
    } else {
      TORCH_CHECK(false, "Could not find pool of id");
    }
  }

  // 释放分配给私有池的块
  void freeBlocksAllocatedToPool(PrivatePool* private_pool, RestoreResult& rr) {
    auto pool_blocks = get_private_pool_head_blocks(private_pool);

    std::vector<Block*> head_blocks;
    // 找到私有池的头块
    for (Block* block : pool_blocks) {
      if (block->prev == nullptr) {
        head_blocks.push_back(block);
      }
    }

    // 逐个释放头块链表上的块
    for (Block* block : head_blocks) {
      Block* curr = block;

      while (curr) {
        // 释放块时，指针不应改变，仅邻接块会改变
        if (curr->allocated) {
          TORCH_CHECK(
              curr->event_count == 0,
              "Events should have synchronized when setting checkpointed block");
          rr.allocations_freed.push_back(curr->ptr);
          free(curr);
          TORCH_CHECK(!curr->allocated)
        }
        curr = curr->next;
      }
    }

    // 确保所有私有池的块都被正确释放
    for (Block* b : get_private_pool_head_blocks(private_pool)) {
      Block* curr = b;
      while (curr) {
        TORCH_CHECK(!curr->allocated);
        curr = curr->next;
      }
    }
  }

  // 将分段状态设置为检查点状态，用于可能分为多个块的分配
  void setSegmentStateToCheckpoint(
      Block* block,
      SegmentState& segment,
      const std::shared_ptr<GatheredContext>& context,
      RestoreResult& rr) {
    Block* curr_block = block;
    Block* last_block = block;

    TORCH_INTERNAL_ASSERT(block->pool);
    BlockPool& pool = *block->pool;
    const auto segment_len = segment.blocks.size();
    // 分配段中的所有块
    for (size_t i = 0; i < segment_len; ++i) {
      // 获取当前块状态的引用
      auto& block_state = segment.blocks.at(i);
      // 准备分配参数，包括设备、块大小、流、内存池、块大小、统计信息
      AllocParams params(
          block_state.device,
          block_state.size,
          block_state.stream,
          &pool,
          block_state.size,
          stats);
      // 从内存池中删除当前块
      pool.blocks.erase(curr_block);
      // 设置参数中的块
      params.block = curr_block;
      // 获取适用于内存池的统计类型
      params.stat_types = get_stat_types_for_pool(pool);

      // 根据 `max_split_size` 的值决定是否要分割块，
      // 由于在生成检查点后可能会改变，因此确保重现检查点时的行为
      bool split = (i + 1) < segment.blocks.size();

      // 调用分配函数，获取新块的指针，如果发生分割，`curr_block` 将指向下一个块
      curr_block = alloc_found_block(params, block_state.size, context, split);

      // 断言当前块的指针与状态与块状态相匹配
      TORCH_CHECK(curr_block->ptr == block_state.ptr);
      TORCH_CHECK(curr_block->size == block_state.size);

      // 将上一个处理的块设置为当前块，并移动到下一个块
      last_block = curr_block;
      curr_block = curr_block->next;

      // 检查当前块的下一个指针是否为空，应当与 `i + 1 < segment_len` 相符
      TORCH_CHECK((curr_block != nullptr) == ((i + 1) < (segment_len)));
    }

    // 将 `last_block` 指针移到段中的第一个块
    while (last_block->prev) {
      last_block = last_block->prev;
    }

    // 将 `curr_block` 设置为 `last_block`
    curr_block = last_block;

    // 释放在检查点中未分配的块
    for (size_t i = 0; i < segment_len; ++i, curr_block = curr_block->next) {
      // 获取当前块状态的引用
      auto& block_state = segment.blocks.at(i);
      // 断言当前块指针不为空
      TORCH_INTERNAL_ASSERT(curr_block != nullptr);

      // 如果块已分配，则将其添加到分配创建列表中，继续下一个循环
      if (block_state.allocated) {
        rr.allocations_created.push_back(curr_block);
        continue;
      }

      // 否则，释放当前块的内存
      free(curr_block);

      // 断言当前块的指针、分配状态和大小与块状态匹配
      TORCH_CHECK(curr_block->ptr == block_state.ptr);
      TORCH_CHECK(curr_block->allocated == block_state.allocated);
      TORCH_CHECK(curr_block->size == block_state.size);
    /**
     * Note [Checkpointing PrivatePoolState]
     *
     * Refer above to Note [Interaction with CUDA graph capture]. Allocations made
     * during graph capture are made from a separate private pool. During graph
     * capture allocations behave as usual. During graph replay the allocator
     * state does not change even as new tensors are created. The private pool
     * will not free its blocks to the main caching allocator until cuda graph use
     * is finished to prevent an allocation from eager clobbering the memory from
     * a live but unaccounted for tensor that was created during replay.
     *
     * `make_graphed_callables`, a series of separate callables chained in
     * successive cuda graphs, can share a memory pool because after a cuda graph
     * recording the allocations in the shared private pool exactly reflect the
     * tensors that are allocated.
     *
     * We would like to extend callable chaining to support a graphed callable
     * tree. In this scenario, we have a tree of callable chains which will be
     * captured with cuda graphs. In the diagram below, we have a tree with four
     * callables, A, B, C, and D. Suppose we have captured, and subsequently
     * replayed, A, B, and C. Then on a new invocation, we replay A and B, but
     * would now like to record D. At this point the private pool will not reflect
     * any of the live tensors created during graph replay. Allocations made
     * during a new recording with the pool could overwrite those live tensors.
     *
     * In order to record a new graph capture after replaying prior callables in
     * the tree, we need the allocator to reflect the state of the live tensors.
     * We checkpoint the state of the private pool after each recording, and then
     * reapply it when we are starting a new recording chain. Additionally, we
     * must free the allocations for any tensors that died between the end of our
     * previous graph replaying and our new recording. All of the allocated
     * segments that existed in the checkpointed state must still exist in the
     * pool. There may also exist new allocated blocks.
     * (TODO : link note [live tensors between iterations] when it exists). For
     * every block that is currently allocated but no allocated in the snapshot,
     * we will return a pointer to their block.
     *
     *
     *  ---------------> A ---------------> B ---------------> C
     *                                      |
     *                                      |
     *                                      |
     *                                      |
     *                                      ╰ ---------------> D
     */
    RestoreResult setCheckpointPoolState(PrivatePoolState& pps) {
        // To reset the caching allocator state we will
        // - Free all the blocks currently allocated to the pool (see [live tensors
        // between iterations])
        // - Allocate all the blocks in a checkpointed segment, whether they are
        // live or not
        // 标记方法用于设置私有池状态的检查点，以便在CUDA图捕获期间恢复状态
    
        // 为了重置缓存分配器状态，我们将执行以下操作：
        // - 释放当前分配给池的所有块（参见[live tensors between iterations]）
        // - 分配在检查点段中的所有块，无论它们是否活跃
        // - 返回状态恢复结果对象
    
        // 以上是设置私有池状态的详细描述，用于支持CUDA图中的图形化可调用对象树，
        // 在此过程中，我们需要在录制新图之前恢复和重新应用池的状态。
        // 这确保了分配给CUDA图捕获期间的所有张量的状态都得到正确地反映和处理。
        // 此外，还要处理在前次图形重放结束后和新录制之前释放的张量分配。
    
        // 下面的示例图形化可调用对象树演示了四个可调用对象A、B、C和D之间的依赖关系，
        // 以及如何在新调用中重新记录未捕获的D。
    
        //  ---------------> A ---------------> B ---------------> C
        //                                      |
        //                                      |
        //                                      |
        //                                      |
        //                                      ╰ ---------------> D
    
        // 以上是用于解释和记录图形化可调用对象树的重要说明。
        // 它描述了在多次调用和重放过程中如何管理和恢复私有池的状态。
    // 释放检查点段中未使用的块
    // 这段代码可能可以优化，但目前很好地重用了现有的 API，并且不在热路径上。

    // 在锁之外完成，因为我们不知道记录器需要获取哪些锁...

    // 获取状态为 RecordContext::STATE 的上下文信息，如果可能的话
    std::shared_ptr<GatheredContext> context =
        maybeGatherContext(RecordContext::STATE);

    // 使用递归互斥锁对 mutex 进行加锁
    std::lock_guard<std::recursive_mutex> lock(mutex);

    // 恢复结果对象
    RestoreResult rr;

    // 检查是否不应将可释放的图形状数据进行检查点
    TORCH_CHECK(
        !graph_pools_freeable.count(pps.owner_id),
        "Not expected to checkpoint freeable graph");

    // 查找私有池的 ID 对应的池对象
    auto pool = graph_pools.find(pps.owner_id);
    TORCH_CHECK(pool != graph_pools.end(), "Could not find private pool id");

    // 获取私有池对象
    PrivatePool* private_pool = pool->second.get();

    // 释放给定私有池中分配的块
    freeBlocksAllocatedToPool(private_pool, rr);

    // 创建指针到块的映射，用于快速查找
    std::unordered_map<void*, Block*> ptrs_to_blocks;
    // 此时，所有的块应该都是空闲的，因此它们都应该在块集合中
    for (Block* block : private_pool->small_blocks.blocks) {
      ptrs_to_blocks[block->ptr] = block;
    }
    for (Block* block : private_pool->large_blocks.blocks) {
      ptrs_to_blocks[block->ptr] = block;
    }

    // 遍历 pps.segments 中的段
    for (auto& segment : pps.segments) {
      auto ptr = segment.blocks.at(0).ptr;
      TORCH_CHECK(ptrs_to_blocks.count(ptr), " could not find ", ptr)
      auto block = ptrs_to_blocks[ptr];

      // 将段设置为检查点状态
      setSegmentStateToCheckpoint(block, segment, context, rr);
    }

    // 返回恢复结果对象
    return rr;
  }

  /** Dump a complete snapshot of the memory held by the allocator. Potentially
   * VERY expensive. **/
  // 获取分配器持有的内存的完整快照。可能非常昂贵。
  std::vector<SegmentInfo> snapshot() {
    // 使用递归互斥锁对 mutex 进行加锁
    std::lock_guard<std::recursive_mutex> lock(mutex);

    // 创建私有池到内存池 ID 的映射
    std::unordered_map<PrivatePool*, MempoolId_t> pool_to_id;
    pool_to_id.reserve(graph_pools.size() + graph_pools_freeable.size());
    for (const auto& pair : graph_pools) {
      pool_to_id[pair.second.get()] = pair.first;
    }
    for (const auto& pair : graph_pools_freeable) {
      pool_to_id[pair.second] = pair.first;
    }

    // 活跃块的总数
    size_t total_active = 0;
    // 结果向量，用于存储段信息
    std::vector<SegmentInfo> result;
    // 获取所有块的快照
    const auto all_blocks = get_all_blocks();
    for (const Block* const head_block : all_blocks) {
      // 对于可扩展的段，我们为每个连续映射的内存范围报告一个段
      // 检查前一个块是否存在且已映射，如果是，则跳过当前块处理
      if (head_block->prev && head_block->prev->mapped) {
        continue;
      }
      // 向结果向量中添加一个新的空段信息
      result.emplace_back();
      // 获取刚刚添加的段信息的引用
      SegmentInfo& segment_info = result.back();
      // 将设备信息赋值给段信息
      segment_info.device = head_block->device;
      // 将指针地址转换为大小_t类型并赋值给段信息的地址
      segment_info.address = reinterpret_cast<size_t>(head_block->ptr);
      // 将流信息赋值给段信息
      segment_info.stream = head_block->stream;
      // 设置是否为大段（如果不是小段则为大段）
      segment_info.is_large = (!head_block->pool->is_small);
      // 设置是否为可扩展段
      segment_info.is_expandable = head_block->expandable_segment_;
      // 设置段分配时的上下文
      segment_info.context_when_allocated =
          head_block->context_when_segment_allocated;
      // 查找块所属的内存池在pool_to_id中的ID
      auto mempool_id = pool_to_id.find(head_block->pool->owner_PrivatePool);
      // 如果找到对应的ID，则将其赋值给段信息的私有池ID
      if (mempool_id != pool_to_id.end()) {
        segment_info.owner_private_pool_id = mempool_id->second;
      }

      // 初始化一个指向当前块的指针
      const Block* block = head_block;
      // 遍历当前段中的所有块，直到遇到未映射的块为止
      while (block != nullptr && block->mapped) {
        // 向当前段信息的块向量中添加一个新的空块信息
        segment_info.blocks.emplace_back();
        // 获取刚刚添加的块信息的引用
        BlockInfo& block_info = segment_info.blocks.back();

        // 将块的大小赋值给块信息
        block_info.size = block->size;
        // 将请求的块大小赋值给块信息
        block_info.requested_size = block->requested_size;
        // 将块的分配状态赋值给块信息
        block_info.allocated = block->allocated;
        // 判断块是否活跃，并将结果赋值给块信息的活跃状态
        block_info.active = block->allocated || (block->event_count > 0) ||
            !block->stream_uses.empty();

        // 更新当前段信息的总大小
        segment_info.total_size += block_info.size;
        // 如果块已分配，则更新当前段信息的已分配大小
        if (block_info.allocated) {
          segment_info.allocated_size += block_info.size;
        }
        // 如果块活跃，则更新当前段信息的活跃大小、请求大小
        if (block_info.active) {
          segment_info.active_size += block_info.size;
          segment_info.requested_size += block_info.requested_size;
        }
        // 将块分配时的上下文信息赋值给块信息
        block_info.context_when_allocated = block->context_when_allocated;
        // 移动到下一个块
        block = block->next;
      }
      // 更新总活跃内存大小
      total_active += segment_info.active_size;
    }

    // 对结果向量中的段信息按地址进行排序
    std::sort(
        result.begin(),
        result.end(),
        [](const SegmentInfo& a, const SegmentInfo& b) {
          return a.address < b.address;
        });

    // 记录追踪快照条目，传入总活跃内存大小
    record_trace(TraceEntry::SNAPSHOT, 0, total_active, nullptr, 0, nullptr);
    // 返回结果向量
    return result;
  }

  // 从分配跟踪中生成追踪信息
  std::vector<TraceEntry> trace(
      const std::function<time_t(approx_time_t)>& tsc_to_us) {
    // 使用递归互斥锁保护访问
    std::lock_guard<std::recursive_mutex> lock(mutex);
    // 初始化结果向量
    std::vector<TraceEntry> result;
    // 预留足够的空间以容纳分配跟踪的所有条目
    result.reserve(alloc_trace->size());
    // 将分配跟踪的部分条目插入结果向量
    result.insert(
        result.end(),
        alloc_trace->begin() +
            static_cast<std::vector<TraceEntry>::difference_type>(
                alloc_trace_next),
        alloc_trace->end());
    // 将分配跟踪的剩余条目插入结果向量
    result.insert(
        result.end(),
        alloc_trace->begin(),
        alloc_trace->begin() +
            static_cast<std::vector<TraceEntry>::difference_type>(
                alloc_trace_next));

    // 将结果向量中的所有时间戳从TSC转换为微秒级的Epoch时间
    for (auto& te : result) {
      te.time_.t_ = tsc_to_us(te.time_.approx_t_);
    }

    // 返回包含所有转换后的追踪条目的结果向量
    return result;
  }
    return result;
  }



  // This function takes the size and number of divisions argument and rounds
  // up the size argument to the nearest power-of-2 division.
  // For example, if we need to round-up 1200 and the number of divisions is 4,
  // the size 1200 lies between 1024 and 2048. Dividing this range into 4 equal
  // parts results in values 1024, 1280, 1536, and 1792. So the function will
  // return 1280 as the nearest ceiling of power-of-2 division.
  static size_t roundup_power2_next_division(size_t size, size_t divisions) {
    if (llvm::isPowerOf2_64(size)) {
      return size;
    }

    TORCH_CHECK(divisions >= 2, "Only 2 or more divisions are supported");

    // Calculate the nearest power-of-2 division for the given size.
    size_t power2_floor = llvm::PowerOf2Floor(size);
    size_t power2_division =
        power2_floor >> (63 - llvm::countLeadingZeros(divisions));
    if (C10_UNLIKELY(power2_division == 0)) {
      // If the calculated division is zero, return the next power-of-2 ceiling.
      return (power2_floor << 1);
    }
    size_t round_size_floor = size & (~(power2_division - 1));
    return (round_size_floor == size) ? size
                                      : round_size_floor + power2_division;
  }

  static size_t round_size(size_t size) {
    if (size < kMinBlockSize) {
      return kMinBlockSize;
    } else {
      auto divisions = CUDAAllocatorConfig::roundup_power2_divisions(size);
      if (divisions > 1 && size > (kMinBlockSize * divisions)) {
        // If size is greater than the minimum block size times divisions,
        // round up to the nearest power-of-2 division.
        return roundup_power2_next_division(size, divisions);
      } else {
        // Otherwise, round up to the nearest multiple of the minimum block size.
        return kMinBlockSize * ((size + kMinBlockSize - 1) / kMinBlockSize);
      }
    }
  }

  // See Note [Interaction with CUDA graph capture]

  // Called by CUDAGraph::capture_begin
  void beginAllocateToPool(
      MempoolId_t mempool_id,
      std::function<bool(cudaStream_t)> filter) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    auto it = graph_pools.find(mempool_id);
    if (it == graph_pools.end()) {
      // If mempool_id does not reference an existing pool, create a new pool
      // for this capture.
      graph_pools.emplace(mempool_id, std::make_unique<PrivatePool>());
    } else {
      // If mempool_id references an existing pool, increment its use_count to
      // share it with the current capture.
      TORCH_INTERNAL_ASSERT(it->second->use_count > 0);
      it->second->use_count++;
    }
    for (auto it2 = captures_underway.begin(); it2 != captures_underway.end();
         ++it2) {
      TORCH_CHECK(
          it2->first != mempool_id,
          "beginAllocateToPool: already recording to mempool_id");
    }
    captures_underway.emplace_back(mempool_id, std::move(filter));
  }

  // Called by CUDAGraph::capture_end
  void endAllocateToPool(MempoolId_t mempool_id) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    // 遍历正在进行的捕获操作列表
    for (auto it = captures_underway.begin(); it != captures_underway.end();
         ++it) {
      // 如果找到与给定 mempool_id 相关的捕获操作，则移除该操作并返回
      if (it->first == mempool_id) {
        captures_underway.erase(it);
        return;
      }
    }
    // 如果没有找到与 mempool_id 相关的捕获操作，则抛出异常
    TORCH_CHECK(
        false, "endAllocatePool: not currently recording to mempool_id");
  }

  // CUDAGraph::reset 调用该方法
  void releasePool(MempoolId_t mempool_id) {
    // 使用递归互斥锁保护下面的代码块
    std::lock_guard<std::recursive_mutex> lock(mutex);
    // 查找具有给定 mempool_id 的图形池
    auto it = graph_pools.find(mempool_id);
    TORCH_INTERNAL_ASSERT(it != graph_pools.end());
    // 减少该 mempool 的使用计数，并确保使用计数大于等于 0
    auto uc = --(it->second->use_count);
    TORCH_INTERNAL_ASSERT(uc >= 0);
    // 当使用计数为 0 时，表示没有图形再使用该 mempool
    if (uc == 0) {
      // 允许 free_cached_blocks 开始释放该池的内存，并确保该池尚未被释放
      bool inserted =
          graph_pools_freeable.insert({mempool_id, it->second.get()}).second;
      TORCH_INTERNAL_ASSERT(inserted);
    }
  }

  // 添加具有对等访问权限的设备
  void addPeerAccess(c10::DeviceIndex dev_to_access) {
    // 如果设备已经具有对等访问权限，则直接返回
    if (std::find(
            devices_with_peer_access_.begin(),
            devices_with_peer_access_.end(),
            dev_to_access) != devices_with_peer_access_.end()) {
      return;
    }
    // 否则将设备添加到具有对等访问权限的设备列表中
    devices_with_peer_access_.push_back(dev_to_access);
    // 为所有可扩展段添加对等设备
    for (auto& es : expandable_segments_) {
      es->addPeer(dev_to_access);
    }
  }

  // 检查是否已分配可扩展段
  bool hasAllocatedExpandableSegments() const {
    return !expandable_segments_.empty();
  }

 private:
  // 所有私有方法不获取分配器互斥锁

  // 获取所有块的指针
  std::vector<const Block*> get_all_blocks() const {
    std::vector<const Block*> blocks;
    // 添加所有小块到 blocks
    blocks.insert(
        blocks.end(), small_blocks.blocks.begin(), small_blocks.blocks.end());
    // 添加所有大块到 blocks
    blocks.insert(
        blocks.end(), large_blocks.blocks.begin(), large_blocks.blocks.end());
    // 添加所有图形池的小块到 blocks
    for (const auto& gp : graph_pools) {
      blocks.insert(
          blocks.end(),
          gp.second->small_blocks.blocks.begin(),
          gp.second->small_blocks.blocks.end());
      // 添加所有图形池的大块到 blocks
      blocks.insert(
          blocks.end(),
          gp.second->large_blocks.blocks.begin(),
          gp.second->large_blocks.blocks.end());
    }
    // 添加所有活动块到 blocks
    blocks.insert(blocks.end(), active_blocks.begin(), active_blocks.end());
    return blocks;
  }

  // 获取私有池头块的指针
  std::vector<Block*> get_private_pool_head_blocks(PrivatePool* pool) const {
    std::vector<Block*> blocks;
    // 返回空块列表

        return blocks;
    }
    // 遍历活跃块列表中的每个块
    for (Block* b : active_blocks) {
      // 检查块是否属于小块池或大块池，并且是链表中的第一个块
      if ((b->pool == &pool->small_blocks || b->pool == &pool->large_blocks) &&
          b->prev == nullptr) {
        // 如果满足条件，则将块添加到 blocks 列表中
        blocks.push_back(b);
      }
    }

    // 遍历小块池中的块列表
    for (Block* b : pool->small_blocks.blocks) {
      // 检查块是否是链表中的第一个块
      if (b->prev == nullptr) {
        // 如果满足条件，则将块添加到 blocks 列表中
        blocks.push_back(b);
      }
    }

    // 遍历大块池中的块列表
    for (Block* b : pool->large_blocks.blocks) {
      // 检查块是否是链表中的第一个块
      if (b->prev == nullptr) {
        // 如果满足条件，则将块添加到 blocks 列表中
        blocks.push_back(b);
      }
    }

    // 返回收集到的 blocks 列表
    return blocks;
  }

  // 返回能够容纳指定大小的空闲地址空间的最小可能地址的块
  // 可能由自由和未映射的段组成
  Block* find_expandable_block(
      c10::DeviceIndex device,
      cudaStream_t stream,
      BlockPool* pool,
      size_t size) {
    // 创建一个用于查找的关键块
    Block key(device, stream, 0);

    // Lambda 函数，用于检查块是否可分配
    auto allocatable = [](Block* b) {
      return b && !b->allocated && b->event_count == 0 &&
          b->stream_uses.empty();
    };

    // Lambda 函数，用于检查块是否具有足够的可用地址空间
    auto has_available_address_space = [&](Block* b) {
      size_t bytes = 0;
      while (bytes < size && allocatable(b)) {
        bytes += b->size;
        b = b->next;
      }
      return bytes >= size;
    };

    // 在未映射的块池中查找第一个符合条件的块
    for (auto it = pool->unmapped.lower_bound(&key);
         it != pool->unmapped.end() && (*it)->stream == stream;
         ++it) {
      Block* c = *it;
      // 找到未映射段的最低地址
      // 但可能有一个相邻的空闲段也可以使用
      if (allocatable(c->prev)) {
        c = c->prev;
      }
      // 如果找到具有足够地址空间的块，则返回该块
      if (has_available_address_space(c)) {
        return c;
      }
    }

    // 如果未找到合适的块，则需要创建一个可扩展段
    auto segment_size = pool->is_small ? kSmallBuffer : kLargeBuffer;
    expandable_segments_.emplace_back(new ExpandableSegment(
        device, stream, segment_size, devices_with_peer_access_));

    ExpandableSegment* es = expandable_segments_.back();
    Block* candidate = new Block(device, stream, es->size(), pool, es->ptr());
    candidate->mapped = false;
    candidate->expandable_segment_ = es;
    pool->unmapped.insert(candidate);
    // 返回新创建的候选块
    return candidate;
  }

  // 将指定大小的块映射到给定上下文中
  bool map_block(
      Block* to_map,
      size_t size,
      const std::shared_ptr<GatheredContext>& ctx) {
    // 断言：未映射的块不应保留历史上下文
    TORCH_INTERNAL_ASSERT(!to_map->mapped && size <= to_map->size);
    TORCH_INTERNAL_ASSERT(
        !to_map->context_when_allocated); // unmapped blocks should not keep
                                          // history

    // 尝试将块映射到可扩展段
    auto mapped_range =
        to_map->expandable_segment_->map(SegmentRange{to_map->ptr, size});
    
    // 如果映射失败，则返回 false
    if (mapped_range.size == 0) {
      return false;
    }

    // 断言：成功映射的内存段应满足大小和地址的要求
    TORCH_INTERNAL_ASSERT(
        mapped_range.ptr == to_map->ptr && mapped_range.size >= size);

    // 更新块池中的未映射块列表，将块标记为已映射状态
    BlockPool& pool = *to_map->pool;
    pool.unmapped.erase(to_map);
    to_map->mapped = true;
    // 映射成功
    return true;
  }
    if (mapped_range.size < to_map->size) {
      // 如果映射范围小于要映射块的大小，则执行以下操作：
      // 创建一个新的未映射块，大小为原始块大小减去映射范围大小，
      // 使用给定的设备、流、内存大小、内存池和偏移位置初始化
      Block* remaining = new Block(
          to_map->device,
          to_map->stream,
          to_map->size - mapped_range.size,
          &pool,
          static_cast<char*>(to_map->ptr) + mapped_range.size);
      remaining->mapped = false;
      remaining->expandable_segment_ = to_map->expandable_segment_;
      // 将新创建的未映射块插入未映射块集合中
      remaining->splice(to_map, to_map->next);
      pool.unmapped.insert(remaining);
      // 更新原始块的大小为映射范围大小
      to_map->size = mapped_range.size;
    }

    // 尝试合并当前块与其前后相邻的块
    try_merge_blocks(to_map, to_map->prev, pool);
    try_merge_blocks(to_map, to_map->next, pool);

    // 将当前块插入到块池中
    pool.insert_into_blocks(to_map);

    // 更新统计信息
    total_allocated_memory += mapped_range.size;
    // 获取与块池相关的统计类型
    StatTypes stat_types = get_stat_types_for_pool(*to_map->pool);
    // 对选定的每种统计类型执行增加统计函数
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      increase_stat(stats.reserved_bytes[stat_type], mapped_range.size);
    });

    // 增加设备分配数量统计
    stats.num_device_alloc++;
    // 记录跟踪信息：段映射，指针地址，映射大小，流，设备，上下文
    record_trace(
        TraceEntry::SEGMENT_MAP,
        int64_t(mapped_range.ptr),
        mapped_range.size,
        to_map->stream,
        to_map->device,
        ctx);
    // 如果当前块没有前驱且未设置段分配时的上下文，则设置上下文为当前上下文
    if (!to_map->prev && !to_map->context_when_segment_allocated) {
      to_map->context_when_segment_allocated = ctx;
    }

    // 返回映射成功标志
    return true;
  }

  // 尝试分配可扩展块
  Block* try_allocate_expandable_block(
      c10::DeviceIndex device,
      cudaStream_t stream,
      BlockPool* pool,
      size_t size,
      const std::shared_ptr<GatheredContext>& ctx) {
    // 查找一个可扩展的块作为候选块
    Block* candidate = find_expandable_block(device, stream, pool, size);
    // 如果候选块未映射，并且无法映射块到候选块，则返回空指针
    if (!candidate->mapped &&
        !map_block(candidate, std::min(candidate->size, size), ctx)) {
      return nullptr;
    }
    // 确保候选块已映射
    TORCH_INTERNAL_ASSERT(candidate->mapped);

    // 当候选块大小小于请求大小时，循环执行以下操作
    while (candidate->size < size) {
      // 映射剩余块，并尝试将其与自由块合并
      auto remaining = size - candidate->size;
      auto new_candidate = candidate->next;
      if (!map_block(
              new_candidate, std::min(remaining, candidate->next->size), ctx)) {
        return nullptr;
      }
      candidate = new_candidate;
    }
    // 从块池中删除候选块
    pool->blocks.erase(candidate);
    // 返回成功映射的候选块
    return candidate;
  }

  // 移动块到缓存的自由块池中
  /** moves a block into a pool of cached free blocks */
  void free_block(
      Block* block,
      const std::shared_ptr<GatheredContext>& context) {
    // 确保块未分配，事件计数为零，且流使用列表为空
    TORCH_INTERNAL_ASSERT(
        !block->allocated && block->event_count == 0 &&
        block->stream_uses.empty());

    // 记录跟踪信息：释放完成，指针地址，请求大小，流，设备，上下文
    record_trace(
        TraceEntry::FREE_COMPLETED,
        int64_t(block->ptr),
        block->requested_size,
        block->stream,
        block->device,
        context ? context : block->context_when_allocated);

    // 重置块的分配上下文为空
    block->context_when_allocated = nullptr;
    // 记录原始块大小
    size_t original_block_size = block->size;
    // 获取请求的内存块大小
    size_t requested_size = block->requested_size;

    // 引用当前内存块所属的池
    auto& pool = *block->pool;
    // 初始化非活跃分裂块和大小的净变化
    int64_t net_change_inactive_split_blocks = 0;
    int64_t net_change_inactive_split_size = 0;

    // 准备合并候选项，即当前块的前一个和后一个块
    const std::array<Block*, 2> merge_candidates = {block->prev, block->next};
    for (Block* merge_candidate : merge_candidates) {
      // 尝试合并当前块与候选块，获取合并的大小
      const auto subsumed_size = try_merge_blocks(block, merge_candidate, pool);
      // 如果成功合并，更新非活跃分裂块和大小的净变化
      if (subsumed_size > 0) {
        net_change_inactive_split_blocks -= 1;
        net_change_inactive_split_size -= static_cast<int64_t>(subsumed_size);
      }
    }

    // 从活跃块列表中移除当前块
    active_blocks.erase(block);
    // 确保将块插入到相应的池中，以便释放
    // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
    bool inserted = pool.insert_into_blocks(block).second;
    TORCH_INTERNAL_ASSERT(inserted);

    // 如果当前块已经分裂，则更新非活跃分裂块和大小的净变化
    if (block->is_split()) {
      net_change_inactive_split_blocks += 1;
      net_change_inactive_split_size += static_cast<int64_t>(block->size);
    }

    // 获取池的统计类型
    StatTypes stat_types = get_stat_types_for_pool(pool);

    // 对于每种选择的统计类型，执行以下操作
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      // 如果块不是可扩展段，则处理非活跃分裂统计数据
      if (!block->expandable_segment_) {
        // 根据非活跃分裂块的变化更新相应的统计数据
        if (net_change_inactive_split_blocks > 0) {
          increase_stat(
              stats.inactive_split[stat_type],
              static_cast<size_t>(net_change_inactive_split_blocks));
        } else if (net_change_inactive_split_blocks < 0) {
          decrease_stat(
              stats.inactive_split[stat_type],
              static_cast<size_t>(-net_change_inactive_split_blocks));
        }
        if (net_change_inactive_split_size > 0) {
          increase_stat(
              stats.inactive_split_bytes[stat_type],
              static_cast<size_t>(net_change_inactive_split_size));
        } else if (net_change_inactive_split_size < 0) {
          decrease_stat(
              stats.inactive_split_bytes[stat_type],
              static_cast<size_t>(-net_change_inactive_split_size));
        }
      }
      // 减少活跃块相关的统计数据
      decrease_stat(stats.active[stat_type], 1);
      decrease_stat(stats.active_bytes[stat_type], original_block_size);
      decrease_stat(stats.requested_bytes[stat_type], requested_size);
    });
  }

  /** combine previously split blocks. returns the size of the subsumed block,
   * or 0 on failure. */
  // 尝试合并之前分裂的块，返回合并块的大小，失败返回0
  size_t try_merge_blocks(Block* dst, Block* src, BlockPool& pool) {
    // 检查合并条件：src存在、未分配、无事件计数、无流使用，并且与dst映射状态相同
    if (!src || src->allocated || src->event_count > 0 ||
        !src->stream_uses.empty() || dst->mapped != src->mapped) {
      return 0;
    }

    // 断言dst和src均为分裂状态
    AT_ASSERT(dst->is_split() && src->is_split());

    // 获取请求的内存块大小
    size_t requested_size = block->requested_size;

    // 引用当前内存块所属的池
    auto& pool = *block->pool;
    // 初始化非活跃分裂块和大小的净变化
    int64_t net_change_inactive_split_blocks = 0;
    int64_t net_change_inactive_split_size = 0;

    // 准备合并候选项，即当前块的前一个和后一个块
    const std::array<Block*, 2> merge_candidates = {block->prev, block->next};
    for (Block* merge_candidate : merge_candidates) {
      // 尝试合并当前块与候选块，获取合并的大小
      const auto subsumed_size = try_merge_blocks(block, merge_candidate, pool);
      // 如果成功合并，更新非活跃分裂块和大小的净变化
      if (subsumed_size > 0) {
        net_change_inactive_split_blocks -= 1;
        net_change_inactive_split_size -= static_cast<int64_t>(subsumed_size);
      }
    }

    // 从活跃块列表中移除当前块
    active_blocks.erase(block);
    // 确保将块插入到相应的池中，以便释放
    // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
    bool inserted = pool.insert_into_blocks(block).second;
    TORCH_INTERNAL_ASSERT(inserted);

    // 如果当前块已经分裂，则更新非活跃分裂块和大小的净变化
    if (block->is_split()) {
      net_change_inactive_split_blocks += 1;
      net_change_inactive_split_size += static_cast<int64_t>(block->size);
    }

    // 获取池的统计类型
    StatTypes stat_types = get_stat_types_for_pool(pool);

    // 对于每种选择的统计类型，执行以下操作
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      // 如果块不是可扩展段，则处理非活跃分裂统计数据
      if (!block->expandable_segment_) {
        // 根据非活跃分裂块的变化更新相应的统计数据
        if (net_change_inactive_split_blocks > 0) {
          increase_stat(
              stats.inactive_split[stat_type],
              static_cast<size_t>(net_change_inactive_split_blocks));
        } else if (net_change_inactive_split_blocks < 0) {
          decrease_stat(
              stats.inactive_split[stat_type],
              static_cast<size_t>(-net_change_inactive_split_blocks));
        }
        if (net_change_inactive_split_size > 0) {
          increase_stat(
              stats.inactive_split_bytes[stat_type],
              static_cast<size_t>(net_change_inactive_split_size));
        } else if (net_change_inactive_split_size < 0) {
          decrease_stat(
              stats.inactive_split_bytes[stat_type],
              static_cast<size_t>(-net_change_inactive_split_size));
        }
      }
      // 减少活跃块相关的统计数据
      decrease_stat(stats.active[stat_type], 1);
      decrease_stat(stats.active_bytes[stat_type], original_block_size);
      decrease_stat(stats.requested_bytes[stat_type], requested_size);
    });
  }

  /** combine previously split blocks. returns the size of the subsumed block,
   * or 0 on failure. */
  // 尝试合并之前分裂的块，返回合并块的大小，失败返回0
  size_t try_merge_blocks(Block* dst, Block* src, BlockPool& pool) {
    // 检查合并条件：src存在、未分配、无事件计数、无流使用，并且与dst映射状态相同
    if (!src || src->allocated || src->event_count > 0 ||
        !src->stream_uses.empty() || dst->mapped != src->mapped) {
      return 0;
    }

    // 断言dst和src均为分裂状态
    AT_ASSERT(dst->is_split() && src->is_split());
    // 检查是否 dst 的前一个节点是 src
    if (dst->prev == src) { // [src dst]
      // 将 dst 的指针设置为 src 的指针
      dst->ptr = src->ptr;
      // 将 dst 的前一个节点设置为 src 的前一个节点
      dst->prev = src->prev;
      // 如果 dst 的前一个节点存在，则更新其后继节点为 dst
      if (dst->prev) {
        dst->prev->next = dst;
      }
      // 将 src 的分配时上下文移动给 dst
      dst->context_when_segment_allocated =
          std::move(src->context_when_segment_allocated);
    } else { // [dest src]
      // 将 dst 的后继节点设置为 src 的后继节点
      dst->next = src->next;
      // 如果 dst 的后继节点存在，则更新其前一个节点为 dst
      if (dst->next) {
        dst->next->prev = dst;
      }
    }
    // 计算被消耗的大小并保存在 subsumed_size 中
    const size_t subsumed_size = src->size;
    // 增加 dst 的大小，以包含消耗的大小
    dst->size += subsumed_size;
    // 标记为已擦除
    // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
    auto erased =
        src->mapped ? pool.blocks.erase(src) : pool.unmapped.erase(src);
    // 断言确保成功擦除了 src
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(erased == 1);
    // 删除 src 指向的块对象
    delete src;

    // 返回被消耗的大小
    return subsumed_size;
  }

  BlockPool& get_pool(size_t size, cudaStream_t stream) {
    // captures_underway 是对当前流是否可能被捕获的一种保守估计。
    // 如果 captures_underway 不为空，说明可能有线程开始但尚未结束捕获操作，
    // 可以提前返回对应的 BlockPool，避免调用 cudaStreamCaptureStatus。
    if (C10_UNLIKELY(!captures_underway.empty())) {
      for (auto& entry : captures_underway) {
        if (entry.second(stream)) {
          auto it1 = graph_pools.find(entry.first);
          TORCH_INTERNAL_ASSERT(it1 != graph_pools.end());
          // 如果请求的 size 小于等于 kSmallSize，则返回对应图形池的 small_blocks
          // 否则返回 large_blocks
          if (size <= kSmallSize) {
            return it1->second->small_blocks;
          } else {
            return it1->second->large_blocks;
          }
        }
      }
    }
    // 如果没有捕获操作进行中，根据请求的 size 返回对应的 BlockPool
    if (size <= kSmallSize) {
      return small_blocks;
    } else {
      return large_blocks;
    }
  }

  StatTypes get_stat_types_for_pool(const BlockPool& pool) {
    // 创建一个默认为 false 的 StatTypes 对象
    StatTypes stat_types = {false};
    // 标记 AGGREGATE 类型为 true
    stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
    // 根据 pool 的大小类型，标记 SMALL_POOL 或 LARGE_POOL 类型为 true
    stat_types[static_cast<size_t>(
        pool.is_small ? StatType::SMALL_POOL : StatType::LARGE_POOL)] = true;
    // 返回标记后的 StatTypes 对象
    return stat_types;
  }

  bool should_split(const Block* block, size_t size) {
    // 计算 block 中剩余的大小
    size_t remaining = block->size - size;
    // 如果 block 所属的 pool 是 small 或者 CUDAAllocatorConfig 配置允许可扩展的段
    if (block->pool->is_small || CUDAAllocatorConfig::expandable_segments()) {
      // 判断剩余大小是否大于等于 kMinBlockSize
      return remaining >= kMinBlockSize;
    } else {
      // 否则，判断 size 是否小于 CUDAAllocatorConfig 允许的最大分割大小
      // 并且剩余大小是否大于 kSmallSize
      return (size < CUDAAllocatorConfig::max_split_size()) &&
          (remaining > kSmallSize);
    }
  }

  static size_t get_allocation_size(size_t size) {
    // 如果请求的 size 小于等于 kSmallSize，返回 kSmallBuffer
    if (size <= kSmallSize) {
      return kSmallBuffer;
    } else if (size < kMinLargeAlloc) {
      // 如果请求的 size 小于 kMinLargeAlloc，返回 kLargeBuffer
      return kLargeBuffer;
    } else {
      // 否则，返回大于等于 size 的 kRoundLarge 的最小整数倍
      return kRoundLarge * ((size + kRoundLarge - 1) / kRoundLarge);
    }
  }

  bool get_free_block(AllocParams& p) {
    // 获取参数 p 所属的 BlockPool
    BlockPool& pool = *p.pool;

    // 如果设置了 set_fraction，并且垃圾回收阈值大于 0.0
    if (C10_UNLIKELY(
            set_fraction &&
            CUDAAllocatorConfig::garbage_collection_threshold() > 0.0)) {
      // 增加 pool 的 get_free_blocks_call_count 计数
      ++pool.get_free_blocks_call_count;
    }
    // 查找大于等于 p.search_key 的第一个块的迭代器
    auto it = pool.blocks.lower_bound(&p.search_key);
    // 如果迭代器指向的块为空或者其 stream 不等于 p 请求的 stream，则返回 false
    if (it == pool.blocks.end() || (*it)->stream != p.stream())
      return false;
    // 检查当前迭代器指向的块是否为可扩展段
    if ((*it)->expandable_segment_) {
      // 如果 CUDAAllocatorConfig::expandable_segments() 返回 true
      if (CUDAAllocatorConfig::expandable_segments()) {
        // 如果我们分配给可扩展部分的块，为了“最佳适配”，考虑其大小是它可以扩展到的大小，而不是当前的大小。
        // 这意味着有时我们必须在选择此段之前搜索更大“大小”的块。
        
        // 定义一个 lambda 函数 expandable_size，用于计算块的可扩展大小
        auto expandable_size = [](Block* b) {
          return b->size + (b->next && !b->next->mapped ? b->next->size : 0);
        };
        
        // 获取下一个迭代器，并且只要当前块仍为可扩展段且下一个块与当前块的流相同，
        // 并且下一个块的可扩展大小小于当前块的可扩展大小，就继续向后迭代
        auto next = it;
        next++;
        while ((*it)->expandable_segment_ && next != pool.blocks.end() &&
               (*next)->stream == p.stream() &&
               expandable_size(*next) < expandable_size(*it)) {
          it = next++;
        }
      } else {
        // 在已分配一些块为可扩展后，稀有情况下关闭了可扩展段。
        // 例如，由于我们无法通过 IPC 共享可扩展内存，有人可能会暂时禁用它。
        // 在这种情况下，我们只需要找到非可扩展块。
        
        // 继续向后迭代直到找到非可扩展块或者迭代到达块列表末尾
        do {
          it++;
        } while (it != pool.blocks.end() && (*it)->expandable_segment_ &&
                 (*it)->stream == p.stream());
        
        // 如果迭代器已经到达块列表末尾或者当前块的流与要求的流不同，返回 false
        if (it == pool.blocks.end() || (*it)->stream != p.stream()) {
          return false;
        }
      }
    }

    // 如果请求的块大小小于最大拆分大小，并且当前块的大小大于等于最大拆分大小，则返回 false
    if ((p.size() < CUDAAllocatorConfig::max_split_size()) &&
        ((*it)->size >= CUDAAllocatorConfig::max_split_size()))
      return false;
    
    // 如果请求的块大小大于等于最大拆分大小，并且当前块的大小大于等于请求的大小加上 kLargeBuffer，则返回 false
    if ((p.size() >= CUDAAllocatorConfig::max_split_size()) &&
        ((*it)->size >= p.size() + kLargeBuffer))
      return false;
    
    // 将当前块指针设置为参数 p 的块，并从块池中移除该块
    p.block = *it;
    pool.blocks.erase(it);
    
    // 返回 true 表示成功找到符合条件的块并处理
    return true;
  }
    // 初始化可释放块计数为0
    int freeable_block_count = 0;
    // 遍历大块对象集合中的每一个块
    for (auto& b : large_blocks.blocks) {
      // 如果块未分裂
      if (!b->is_split()) {
        // 增加总年龄，累加块的垃圾回收计数
        total_age += b->gc_count();
        // 增加可释放块计数
        ++freeable_block_count;
      }
    }
    // 如果没有可释放的块，则直接返回
    // No free-able blocks?
    if (freeable_block_count == 0) {
      return;
    }

    // 反复执行垃圾回收，直到回收量超过目标大小
    // Repeat GC until we reach reclaim > target size.
    bool block_freed = true;
    while (gc_reclaimed < target_size && block_freed == true &&
           freeable_block_count > 0) {
      // 计算块的平均年龄阈值
      double age_threshold =
          static_cast<double>(total_age) / freeable_block_count;
      // 假设此次循环内没有块被释放
      // Stop iteration if we can no longer free a block.
      block_freed = false;

      // 逐个遍历大块对象集合中的块
      auto it = large_blocks.blocks.begin();
      while (it != large_blocks.blocks.end()) {
        Block* block = *it;
        ++it;
        // 如果块未分裂且其垃圾回收计数大于等于年龄阈值
        if (!block->is_split() &&
            static_cast<double>(block->gc_count()) >= age_threshold) {
          // 标记至少有一个块被释放
          block_freed = true;
          // 增加已回收的内存大小
          gc_reclaimed += block->size;
          // 减少总年龄，减少可释放块计数
          total_age -= block->gc_count(); // Decrement the age
          freeable_block_count--; // One less block that can be freed
          // 释放当前块
          release_block(block, context);
        }
      }
    }
  }

  // 此函数假设在调用时已经获取了全局锁。
  // 在此函数中进行cudaMalloc同步调用，而持有锁的情况下可能会昂贵。
  // 因此，我们将锁传入函数中，以便在cudaMalloc调用前临时释放锁，并在调用后重新获取锁，
  // 以避免阻塞其他线程。
  bool alloc_block(
      AllocParams& p,
      bool isRetry,
      const std::shared_ptr<GatheredContext>& ctx,
      std::unique_lock<std::recursive_mutex>& lock) {
    // 防御性地检查预先存在的CUDA错误状态
    // Defensively checks for preexisting CUDA error state.
    C10_CUDA_CHECK(cudaGetLastError());

    size_t size = p.alloc_size;
    void* ptr = nullptr;

    // 如果是重试分配，则增加重试计数
    if (isRetry) {
      stats.num_alloc_retries += 1;
    }

    // 如果设置了分配比例，并且总分配内存加上当前尝试分配的大小超过了允许的最大内存大小，
    // 则返回内存分配错误
    if (set_fraction &&
        total_allocated_memory + size > allowed_memory_maximum) {
      p.err = cudaErrorMemoryAllocation;
      return false;
    } else if (
        CUDAAllocatorConfig::expandable_segments() &&
        // 我们的私有池的检查点逻辑尚不支持可扩展段结构
        // our checkpointing logic for private pools doesn't support
        // the expandable_segments_ structure yet
        !p.pool->owner_PrivatePool) {
      // 尝试分配可扩展块
      p.block = try_allocate_expandable_block(
          p.device(), p.stream(), p.pool, p.size(), ctx);
      if (p.block) {
        p.err = cudaSuccess;
      } else {
        p.err = cudaErrorMemoryAllocation;
      }
      return bool(p.block);
    }
  } else {
    // 如果CUDAAllocatorConfig::release_lock_on_cudamalloc()返回true，
    // 在作用域退出时重新获取锁。这样可以防止在cudaMallocMaybeCapturing函数中发生任何潜在异常。
    auto sg = c10::make_scope_exit([&]() { lock.lock(); });
    // 解锁以允许cudaMallocMaybeCapturing函数执行
    lock.unlock();
    // 调用cudaMallocMaybeCapturing函数，将分配结果保存在p.err中
    p.err = cudaMallocMaybeCapturing(&ptr, size);
  }
  // 如果CUDAAllocatorConfig::release_lock_on_cudamalloc()返回true，则检查锁是否已经获取
  if (CUDAAllocatorConfig::release_lock_on_cudamalloc()) {
    TORCH_CHECK(
        lock.owns_lock(), "Failed to acquire lock after cudaMalloc");
  }

  // 如果cudaMallocMaybeCapturing失败，处理错误情况
  if (p.err != cudaSuccess) {
    if (p.err == cudaErrorMemoryAllocation) {
      // 如果这是第一次尝试（!isRetry），我们可以原谅并清除CUDA的内部错误状态。
      //
      // 如果这是第二次尝试（isRetry），malloc的TORCH_CHECK_WITH将抛出一个有用的异常。
      // 用户可以选择捕获异常，在脚本中释放一些资源，并再次尝试分配。在这种情况下，
      // 我们也可以原谅并清除CUDA的内部错误状态。
      (void)cudaGetLastError();
    } else {
      // 如果错误与内存分配无关，应立即抛出异常。
      C10_CUDA_CHECK(p.err);
    }
    return false;
  }

  // 如果p.pool->owner_PrivatePool存在，增加其cudaMalloc_count计数
  if (p.pool->owner_PrivatePool) {
    // 该块用于CUDA图的PrivatePool。
    p.pool->owner_PrivatePool->cudaMalloc_count++;
  }

  // 增加总分配内存计数
  total_allocated_memory += size;

  // 创建新的Block对象，并将其分配给p.block
  p.block = new Block(p.device(), p.stream(), size, p.pool, (char*)ptr);

  // 针对每个选择的统计类型，增加相应的统计信息
  for_each_selected_stat_type(p.stat_types, [&](size_t stat_type) {
    increase_stat(stats.segment[stat_type], 1);
    increase_stat(stats.reserved_bytes[stat_type], size);
  });

  // 如果分配的大小大于等于CUDAAllocatorConfig::max_split_size()，增加超大块计数
  if (size >= CUDAAllocatorConfig::max_split_size())
    increase_stat(stats.oversize_segments, 1);

  // 检查p.block来自new操作，而不是cudaMalloc。此处不应为nullptr。
  TORCH_INTERNAL_ASSERT(p.block != nullptr && p.block->ptr != nullptr);

  // 增加设备分配数量计数
  stats.num_device_alloc++;

  // 记录跟踪信息，标记为SEGMENT_ALLOC，记录相应的参数信息
  record_trace(
      TraceEntry::SEGMENT_ALLOC,
      int64_t(p.block->ptr),
      p.block->size,
      p.stream(),
      p.device(),
      ctx);

  // 记录分配时的上下文信息到p.block
  p.block->context_when_segment_allocated = ctx;

  // 分配成功，返回true
  return true;
}

/** Free one or more oversize blocks to the system allocator.  But only enough
 * **/
/** to satisfy the target size **/
bool release_available_cached_blocks(
    const AllocParams& p,
    const std::shared_ptr<GatheredContext>& context) {
  // 如果CUDAAllocatorConfig::max_split_size()等于std::numeric_limits<size_t>::max()，返回false
  if (CUDAAllocatorConfig::max_split_size() ==
      std::numeric_limits<size_t>::max())
    return false;

  // 将p.pool解引用为BlockPool对象的引用
  BlockPool& pool = *p.pool;

  // 由于std::unique_ptr的存在，block无法直接复制，使用构造函数创建搜索键
  Block key(p.search_key.device, p.search_key.stream, p.search_key.size);
    // 如果 key.size 小于最大可拆分大小，则将 key.size 设置为最大可拆分大小
    key.size = (key.size < CUDAAllocatorConfig::max_split_size())
        ? CUDAAllocatorConfig::max_split_size()
        : key.size;

    // 在内存池中查找第一个不小于 key 的块的迭代器
    auto it = pool.blocks.lower_bound(&key);

    // 如果找不到合适的块或者找到的块的流不同于 p.stream()
    if (it == pool.blocks.end() || (*it)->stream != p.stream()) {
      // 如果 it 是指向 begin() 的迭代器，表示没有足够大的单个块；释放多个超大块，从最大的开始
      if (it == pool.blocks.begin())
        return false;

      size_t totalReleased = 0;
      --it; // 向前移动一个项目，现在指向正确流的最大块

      // 当释放的总大小小于 key.size，并且块的大小大于等于最大可拆分大小，并且块的流与 p.stream() 相同时
      while ((totalReleased < key.size) &&
             ((*it)->size >= CUDAAllocatorConfig::max_split_size()) &&
             ((*it)->stream == p.stream())) {
        auto cur = it;
        totalReleased += (*it)->size;
        if (it != pool.blocks.begin()) {
          --it;
          // 释放当前块到系统中
          release_block(*cur, context);
        } else {
          release_block(*cur, context);
          break;
        }
      }

      // 如果释放的总大小仍然小于 key.size，则返回 false
      if (totalReleased < key.size)
        return false;
    } else {
      // 否则，释放找到的块到系统中
      release_block(*it, context);
    }

    // 操作成功，返回 true
    return true;
  }

  // 释放所有缓存的块到系统中
  bool release_cached_blocks(const std::shared_ptr<GatheredContext>& context) {
    // 首先确保由于未完成的事件而无法分配的所有块都返回到池中
    synchronize_and_free_events(context);

    // 释放所有非拆分的缓存块到系统中
    release_blocks(large_blocks, context);
    release_blocks(small_blocks, context);

    // 遍历可释放的图池
    for (auto it = graph_pools_freeable.begin();
         it != graph_pools_freeable.end();) {
      // 根据策略，确保 it 所指向的图池没有被使用
      TORCH_INTERNAL_ASSERT(it->second->use_count == 0);

      // 释放该图池中的所有小块和大块到系统中
      release_blocks(it->second->small_blocks, context);
      release_blocks(it->second->large_blocks, context);

      // 如果 cudaMalloc_count 为 0，则从图池中删除该图
      if (it->second->cudaMalloc_count == 0) {
        auto erase_count = graph_pools.erase(it->first);
        TORCH_INTERNAL_ASSERT(erase_count == 1);
        it = graph_pools_freeable.erase(it);
      } else {
        ++it;
      }
    }

    // 操作成功，返回 true
    return true;
  }

  // 释放可扩展段落的块
  void release_expandable_segment(Block* block) {
    // 确保块的大小与其可扩展段的大小一致
    TORCH_INTERNAL_ASSERT(
        block->size == block->expandable_segment_->size(),
        "block disagrees with segment");

    // 确保块未映射
    TORCH_INTERNAL_ASSERT(!block->mapped);

    // 在可扩展段落列表中查找并删除块的可扩展段
    auto it = std::find(
        expandable_segments_.begin(),
        expandable_segments_.end(),
        block->expandable_segment_);
    TORCH_INTERNAL_ASSERT(it != expandable_segments_.end());
    expandable_segments_.erase(it);

    // 从块的池中删除未映射的块
    block->pool->unmapped.erase(block);

    // 删除块的可扩展段和块本身
    delete block->expandable_segment_;
    delete block;
  }

  // 释放块到系统中
  void release_block(
      Block* block,
      const std::shared_ptr<GatheredContext>& context) {
    // 确保块没有可扩展段
    TORCH_INTERNAL_ASSERT(!block->expandable_segment_);

    // 增加设备释放的统计数
    stats.num_device_free++;
    // 记录释放的内存块的跟踪信息，包括段类型、内存块地址、大小、流、设备，以及分配段时的上下文信息（如果有）
    record_trace(
        TraceEntry::SEGMENT_FREE,
        int64_t(block->ptr),
        block->size,
        block->stream,
        block->device,
        context ? context : block->context_when_segment_allocated);

    // 调用 CUDA 的内存释放函数，释放当前内存块
    C10_CUDA_CHECK(cudaFree((void*)block->ptr));
    // 减去已分配内存总量中当前块的大小
    total_allocated_memory -= block->size;

    // 获取当前内存块所属的内存池
    auto* pool = block->pool;
    if (pool->owner_PrivatePool) {
      // 如果内存池属于 CUDA 图的私有池，确保其 cudaMalloc_count 大于零
      TORCH_INTERNAL_ASSERT(pool->owner_PrivatePool->cudaMalloc_count > 0);
      pool->owner_PrivatePool->cudaMalloc_count--;
    }

    // 获取适用于当前内存池的统计类型
    StatTypes stat_types = get_stat_types_for_pool(*pool);
    // 针对每种选择的统计类型，递减相应的统计数据：segment 和 reserved_bytes
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      decrease_stat(stats.segment[stat_type], 1);
      decrease_stat(stats.reserved_bytes[stat_type], block->size);
    });

    // 如果当前块的大小超过了 CUDAAllocatorConfig 的最大分割大小，则减少超大段的统计计数
    if (block->size >= CUDAAllocatorConfig::max_split_size())
      decrease_stat(stats.oversize_segments, 1);
    
    // 从当前内存池的块列表中删除当前块
    pool->blocks.erase(block);
    // 删除当前块对象
    delete block;
  }

  // 解除映射一个内存块
  void unmap_block(
      Block* block,
      const std::shared_ptr<GatheredContext>& context) {
    // 使用扩展段的 unmap 方法解除映射当前块
    auto unmapped = block->expandable_segment_->unmap(
        SegmentRange{block->ptr, block->size});
    // 如果未映射大小为零，则直接返回
    if (unmapped.size == 0) {
      return;
    }
    // 从当前内存池的块列表中删除当前块
    block->pool->blocks.erase(block);

    // 计算前面未映射的空间大小
    ptrdiff_t before_size =
        static_cast<char*>(unmapped.ptr) - static_cast<char*>(block->ptr);
    if (before_size > 0) {
      // 如果前面的空间大于零，则创建前面的空闲块，并将其插入到块列表中
      Block* before_free = new Block(
          block->device, block->stream, before_size, block->pool, block->ptr);
      before_free->expandable_segment_ = block->expandable_segment_;
      before_free->splice(block->prev, block);
      block->pool->insert_into_blocks(before_free);
    }

    // 计算后面未映射的空间大小
    auto after_size = block->size - (before_size + unmapped.size);
    if (after_size > 0) {
      // 如果后面的空间大于零，则创建后面的空闲块，并将其插入到块列表中
      Block* after_free = new Block(
          block->device,
          block->stream,
          after_size,
          block->pool,
          static_cast<char*>(unmapped.ptr) + unmapped.size);
      after_free->expandable_segment_ = block->expandable_segment_;
      after_free->splice(block, block->next);
      block->pool->insert_into_blocks(after_free);
    }

    // 更新当前块的指针和大小，标记为未映射
    block->ptr = unmapped.ptr;
    block->size = unmapped.size;
    block->mapped = false;

    // 尝试合并当前块与其前后块
    try_merge_blocks(block, block->prev, *block->pool);
    try_merge_blocks(block, block->next, *block->pool);
    // 将当前块标记为未映射
    block->pool->unmapped.insert(block);

    // 更新统计数据：总分配内存减少未映射大小
    total_allocated_memory -= unmapped.size;
    // 获取当前块所属内存池的统计类型
    StatTypes stat_types = get_stat_types_for_pool(*block->pool);
    // 针对每种选择的统计类型，递减相应的 reserved_bytes 统计数据
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      decrease_stat(stats.reserved_bytes[stat_type], unmapped.size);
    });

    // 增加设备上空闲块的计数
    stats.num_device_free++;
  // 记录跟踪信息，标记为段取消映射，记录取消映射的指针、大小、流、设备以及上下文
  record_trace(
      TraceEntry::SEGMENT_UNMAP,
      int64_t(unmapped.ptr),
      unmapped.size,
      block->stream,
      block->device,
      context ? context : block->context_when_segment_allocated);
}

// 释放所有非分裂块
void release_blocks(
    BlockPool& pool,
    const std::shared_ptr<GatheredContext>& context) {
  std::vector<Block*> to_unmap;
  // 迭代访问块池中的每一个块
  auto it = pool.blocks.begin();
  while (it != pool.blocks.end()) {
    Block* block = *it;
    ++it;
    // 如果块具有可扩展段，将其添加到待释放列表中
    if (block->expandable_segment_) {
      // 由于解映射会改变自由池，因此仅收集需要释放的块，避免使迭代器失效
      to_unmap.push_back(block);
    } else if (!block->prev && !block->next) {
      // 如果块没有前驱和后继，则释放该块
      release_block(block, context);
    }
  }
  // 对待释放的块进行解映射和释放可扩展段
  for (Block* block : to_unmap) {
    unmap_block(block, context);
    if (!block->prev && !block->next) {
      release_expandable_segment(block);
    }
  }
}

// 创建内部事件，并返回该事件
EventPool::Event create_event_internal(c10::DeviceIndex idx) {
  // 静态事件池指针，用于避免关闭问题
  static auto* event_pool = new EventPool();
  return event_pool->get(idx);
}

// 同步所有未处理事件，并释放相关的块
void synchronize_and_free_events(
    const std::shared_ptr<GatheredContext>& context) {
  // 增加同步所有流的统计计数
  stats.num_sync_all_streams++;

  // 确保没有进行捕获，插入延迟处理的结束生命周期事件
  TORCH_INTERNAL_ASSERT(captures_underway.empty());
  insert_events_deferred_until_no_capture(context);

  // 对CUDA事件进行同步和释放相关块
  for (auto& st : cuda_events) {
    for (auto& e : st.second) {
      EventPool::Event event = std::move(e.first);
      Block* block = e.second;

      // 同步 CUDA 事件
      C10_CUDA_CHECK(cudaEventSynchronize(*event));

      // 减少块关联事件计数，如果计数为零，则释放块
      block->event_count--;
      if (block->event_count == 0) {
        free_block(block, context);
      }
    }
  }

  // 清空 CUDA 事件容器
  cuda_events.clear();
}

// 移除在 cudagraph 捕获期间添加的流使用
void remove_cudagraph_stream_uses(Block* block) {
  if (C10_UNLIKELY(
          block_to_cudagraph_stream_uses.find(block) !=
          block_to_cudagraph_stream_uses.end())) {
    // 移动块的流使用到临时流集合，并断言块的流使用为空
    stream_set streams(std::move(block->stream_uses));
    AT_ASSERT(block->stream_uses.empty());
    // 遍历流集合，将未包含在 cudagraph 流使用中的流重新加入块的流使用中
    for (auto& stream : streams) {
      if (block_to_cudagraph_stream_uses[block].find(stream) ==
          block_to_cudagraph_stream_uses[block].end()) {
        block->stream_uses.insert(stream);
      }
    }
    // 移除块到 cudagraph 流使用映射
    block_to_cudagraph_stream_uses.erase(block);
  }
}

// 插入块的事件处理
void insert_events(Block* block) {
  c10::DeviceIndex prev_device = 0;
  C10_CUDA_CHECK(c10::cuda::GetDevice(&prev_device));

  // 移动块的流使用到临时流集合，并断言块的流使用为空
  stream_set streams(std::move(block->stream_uses));
  AT_ASSERT(block->stream_uses.empty());
    // 遍历流列表中的每一个流对象
    for (auto& stream : streams) {
      // 设置当前 CUDA 设备为流对象指定的设备
      C10_CUDA_CHECK(c10::cuda::SetDevice(stream.device_index()));

      // 创建当前流对象对应的 CUDA 事件
      EventPool::Event event = create_event_internal(stream.device_index());
      // 记录 CUDA 事件在流上的发生
      C10_CUDA_CHECK(cudaEventRecord(*event, stream.stream()));

      // 块的事件计数加一
      block->event_count++;
      // 将创建的事件与块关联并加入到 CUDA 事件映射中
      cuda_events[stream].emplace_back(std::move(event), block);
    }

    // 恢复先前的 CUDA 设备
    C10_CUDA_CHECK(c10::cuda::MaybeSetDevice(prev_device));
  }

  // 插入事件直到没有捕获的需要
  void insert_events_deferred_until_no_capture(
      const std::shared_ptr<GatheredContext>& context) {
    // 如果存在需要延迟直到没有捕获的事件块
    if (C10_UNLIKELY(!needs_events_deferred_until_no_capture.empty())) {
      // 遍历需要延迟直到没有捕获的事件块列表
      for (auto* block : needs_events_deferred_until_no_capture) {
        // 断言块的流使用列表不为空
        TORCH_INTERNAL_ASSERT(!block->stream_uses.empty());
        // 只有在 cudagraph 之前记录的流才会被用来插入事件
        // 因为我们知道所有在 cudagraph 期间记录的流必须已经完成
        // 参考 CUDA 编程指南第 3.2.8.7.3.1 节 跨流依赖和事件
        remove_cudagraph_stream_uses(block);
        // 插入事件到块中
        insert_events(block);
        // 如果事件计数为零，释放块资源
        if (block->event_count == 0) {
          free_block(block, context);
        }
      }
      // 清空需要延迟直到没有捕获的事件块列表
      needs_events_deferred_until_no_capture.clear();
    }
  }

  // 处理事件
  void process_events(const std::shared_ptr<GatheredContext>& context) {
    // 插入延迟到没有捕获的事件
    insert_events_deferred_until_no_capture(context);

    // 处理未完成的 cudaEvents。完成的事件将从队列中移除，并且对应分配的 'event_count' 减一。
    // 我们为每个流维护一个独立的事件列表，以避免如果某些流有长时间运行的操作，则出现头部阻塞延迟。

    // 迭代处理不同的流
    for (auto it = cuda_events.begin(); it != cuda_events.end();) {
      // 遍历当前流的 (event, block) 对
      while (!it->second.empty()) {
        auto& e = it->second.front();
        EventPool::Event event = std::move(e.first);
        Block* block = e.second;

        // 查询 CUDA 事件的状态
        cudaError_t err = C10_CUDA_ERROR_HANDLED(cudaEventQuery(*event));
        if (err == cudaErrorNotReady) {
          // 如果事件尚未就绪，则忽略并清除错误
          (void)cudaGetLastError();
          // 保留事件的所有权（unique ptr 形式）
          e.first = std::move(event);
          break;
        } else if (err != cudaSuccess) {
          // 否则，检查 CUDA 错误
          C10_CUDA_CHECK(err);
        }

        // 减少块的事件计数
        block->event_count--;
        // 如果块的事件计数为零，释放块资源
        if (block->event_count == 0) {
          free_block(block, context);
        }
        // 移除处理完成的事件
        it->second.pop_front();
      }

      // 如果当前流的事件列表为空，则从 cuda_events 映射中移除
      if (it->second.empty()) {
        it = cuda_events.erase(it);
      } else {
        it++;
      }
    }
  }

  // 遍历给定池中给定设备的所有内存块的大小
  void cache_info_aux(const BlockPool& pool, size_t* largest) {
    // 遍历内存块池中的所有块
    for (const auto& block : pool.blocks) {
      // 获取当前块的大小
      const auto blocksize = block->size;
      // 如果当前块的大小大于已知的最大值，则更新最大值
      if (blocksize > *largest) {
        *largest = blocksize;
      }
      // 省略了其他处理
    }
  }
  }

  // 记录跟踪信息的函数，记录操作、地址、大小、流、设备、上下文等信息
  void record_trace(
      TraceEntry::Action action,                 // 操作类型
      size_t addr,                               // 地址
      size_t size,                               // 大小
      cudaStream_t stream,                       // CUDA 流
      c10::DeviceIndex device,                   // 设备索引
      std::shared_ptr<GatheredContext> context) {// 共享的上下文指针
    // 如果不记录历史且跟踪器为空，则直接返回
    if (!record_history && trace_trackers_.empty())
      return;

    // 创建跟踪条目对象
    auto te = TraceEntry(
        action,                                 // 操作类型
        device,                                 // 设备索引
        addr,                                   // 地址
        size,                                   // 大小
        stream,                                 // CUDA 流
        getApproximateTime(),                   // 获取近似时间
        // 如果记录上下文选项大于等于ALLOC，则移动上下文指针，否则置为nullptr
        record_context_ >= RecordContext::ALLOC ? std::move(context) : nullptr);

    // 遍历所有跟踪器，将当前跟踪条目传递给回调函数
    for (const auto& cb : trace_trackers_) {
      cb(te);
    }

    // 如果需要记录历史
    if (record_history) {
      // 如果分配跟踪列表的大小小于最大条目数
      if (alloc_trace->size() < alloc_trace_max_entries_) {
        // 将当前跟踪条目追加到分配跟踪列表末尾
        alloc_trace->emplace_back(te);
      } else {
        // 否则，替换分配跟踪列表中的指定索引处的条目
        (*alloc_trace)[alloc_trace_next++] = te;
        // 如果索引已经达到列表最大大小，则重置为0
        if (alloc_trace_next == alloc_trace_max_entries_) {
          alloc_trace_next = 0;
        }
      }
    }
  }
};

// 返回是否强制所有分配跳过缓存分配器直接使用 cudaMalloc。在调试 GPU 内存错误时很有用，
// 因为缓存分配器会干扰 cuda-memcheck 的工作。
bool forceUncachedAllocator() {
  // 静态变量，检查环境变量 "PYTORCH_NO_CUDA_MEMORY_CACHING" 是否存在
  static bool force_uncached =
      getenv("PYTORCH_NO_CUDA_MEMORY_CACHING") != nullptr;
  return force_uncached;
}

// 用于释放未缓存的内存块的函数
static void uncached_delete(void* ptr) {
  // 如果使用了 SDT，记录释放事件
  if (TORCH_SDT_IS_ENABLED(free)) {
    TORCH_SDT_WITH_SEMAPHORE(free, ptr);
  }

  // 获取当前 GPU 跟踪器，用于追踪 GPU 内存的释放
  const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
  if (C10_UNLIKELY(interp)) {
    (*interp)->trace_gpu_memory_deallocation(
        c10::kCUDA, reinterpret_cast<uintptr_t>(ptr));
  }
  // 使用 CUDA API 释放 GPU 内存
  C10_CUDA_CHECK(cudaFree(ptr));
}

// 用于释放本地内存的函数声明
void local_raw_delete(void* ptr);

// NativeCachingAllocator 类继承自 CUDAAllocator，用于管理 GPU 内存的缓存分配
class NativeCachingAllocator : public CUDAAllocator {
 private:
  // 分片分配区域，每个区域有独立的互斥锁以减少竞争
  static constexpr size_t kNumMutexShard = 67;

  // 对齐的互斥锁结构体，用于互斥访问分片的分配块映射
  struct alignas(64) AlignedMutex {
    std::mutex m;
  };

  // 互斥锁数组，用于分片映射的互斥访问
  std::array<AlignedMutex, kNumMutexShard> mutex;

  // 按设备指针分配的块映射表数组
  std::array<ska::flat_hash_map<void*, Block*>, kNumMutexShard>
      allocated_blocks;

  // 根据指针获取互斥分片 ID
  static size_t get_mutex_shard_id(void* ptr) {
    return twang_mix64((size_t)ptr) % kNumMutexShard;
  }

  // 添加已分配块到映射表
  void add_allocated_block(Block* block) {
    const auto mutex_shard_id = get_mutex_shard_id(block->ptr);
    std::lock_guard<std::mutex> lock(mutex[mutex_shard_id].m);
    allocated_blocks[mutex_shard_id][block->ptr] = block;
  }

  // 时间转换器，用于 GPU 内存追踪
  c10::ApproximateClockToUnixTimeConverter clock_converter;

 public:
  // 设备缓存分配器的智能指针数组
  std::vector<std::unique_ptr<DeviceCachingAllocator>> device_allocator;

  // 根据指针获取已分配块，可选择是否移除
  Block* get_allocated_block(void* ptr, bool remove = false) {
    const auto mutex_shard_id = get_mutex_shard_id(ptr);
    std::lock_guard<std::mutex> lock(mutex[mutex_shard_id].m);
    auto it = allocated_blocks[mutex_shard_id].find(ptr);
    if (it == allocated_blocks[mutex_shard_id].end()) {
      return nullptr;
    }
    Block* block = it->second;
    if (remove) {
      allocated_blocks[mutex_shard_id].erase(it);
    }
    return block;
  }

  // 初始化方法，根据设备数调整设备缓存分配器数组
  void init(int device_count) override {
    const auto size = static_cast<int64_t>(device_allocator.size());
    if (size < device_count) {
      device_allocator.resize(device_count);
      for (const auto i : c10::irange(size, device_count)) {
        device_allocator[i] = std::make_unique<DeviceCachingAllocator>();
      }
    }
  }

  // 检查是否已初始化
  bool initialized() override {
    return !device_allocator.empty();
  }

  /** 
   * 分配一个块，该块可安全从提供的流使用
   * @param devPtr 用于存储设备指针的指针
   * @param device 设备索引
   * @param size 分配的大小
   * @param stream 分配块的 CUDA 流
   */
  void malloc(
      void** devPtr,
      c10::DeviceIndex device,
      size_t size,
      cudaStream_t stream) {
  // 对设备索引进行边界检查，确保设备索引在有效范围内
  TORCH_INTERNAL_ASSERT(
      0 <= device && static_cast<size_t>(device) < device_allocator.size(),
      "Allocator not initialized for device ",
      device,
      ": did you call init?");
  // 在指定设备上分配内存块，并将分配的块添加到已分配块列表中
  Block* block = device_allocator[device]->malloc(device, size, stream);
  // 将分配的块添加到已分配块的管理列表中
  add_allocated_block(block);
  // 将设备指针指向分配块的内存地址
  *devPtr = (void*)block->ptr;
  // 获取 GPU 跟踪器的实例，用于记录 GPU 内存分配操作
  const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
  // 如果存在 GPU 跟踪器的实例，则调用其记录 GPU 内存分配操作的方法
  if (C10_UNLIKELY(interp)) {
    (*interp)->trace_gpu_memory_allocation(
        c10::kCUDA, reinterpret_cast<uintptr_t>(*devPtr));
  }
}

// 释放给定的设备指针对应的内存块
void free(void* ptr) {
  // 如果指针为空，则直接返回
  if (!ptr) {
    return;
  }
  // 根据指针获取对应的已分配块的实例
  Block* block = get_allocated_block(ptr, true /* remove */);
  // 如果未找到对应的块，则抛出异常
  if (!block) {
    TORCH_CHECK(false, "invalid device pointer: ", ptr);
  }
  // 获取 GPU 跟踪器的实例，用于记录 GPU 内存释放操作
  const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
  // 如果存在 GPU 跟踪器的实例，则调用其记录 GPU 内存释放操作的方法
  if (C10_UNLIKELY(interp)) {
    (*interp)->trace_gpu_memory_deallocation(
        c10::kCUDA, reinterpret_cast<uintptr_t>(block->ptr));
  }
  // 释放块所在设备上的内存块
  device_allocator[block->device]->free(block);
}

// 设置指定设备的内存分配分数，确保分数在有效范围内，并调用 CUDA 设置设备
void setMemoryFraction(double fraction, c10::DeviceIndex device) override {
  // 对设备索引进行边界检查，确保设备索引在有效范围内
  TORCH_INTERNAL_ASSERT(
      0 <= device && static_cast<size_t>(device) < device_allocator.size(),
      "Allocator not initialized for device ",
      device,
      ": did you call init?");
  // 对分数进行边界检查，确保分数在 (0, 1) 范围内
  TORCH_INTERNAL_ASSERT(
      0 <= fraction && fraction <= 1,
      "invalid fraction:",
      fraction,
      ". Please set within (0, 1).");
  // 设置 CUDA 设备为指定设备
  C10_CUDA_CHECK(c10::cuda::SetDevice(device));
  // 设置设备分配器在指定设备上的内存分配分数
  device_allocator[device]->setMemoryFraction(fraction);
}

// 记录历史内存分配操作的上下文信息到所有设备分配器中
void recordHistory(
    bool enabled,
    CreateContextFn context_recorder,
    size_t alloc_trace_max_entries,
    RecordContext when) override {
  // 遍历所有设备分配器，将历史记录功能设置为启用或禁用，并设置相关参数
  for (auto& allocator : device_allocator) {
    allocator->recordHistory(
        enabled, context_recorder, alloc_trace_max_entries, when);
  }
}

// 记录给定名称的注释到所有设备分配器中
void recordAnnotation(const std::shared_ptr<GatheredContext>& name) override {
  // 遍历所有设备分配器，将给定的注释名称记录到各设备分配器中
  for (auto& allocator : device_allocator) {
    allocator->recordAnnotation(name);
  }
}

// 检查历史记录功能是否已启用并返回结果
bool isHistoryEnabled() override {
  // 获取当前 CUDA 设备索引
  c10::DeviceIndex device = 0;
  C10_CUDA_CHECK(c10::cuda::GetDevice(&device));
  // 返回当前 CUDA 设备上历史记录功能的启用状态
  return device_allocator[device]->isHistoryEnabled();
}

// 检查指定设备上的内存池中预期的活动分配是否与实际一致
bool checkPoolLiveAllocations(
    c10::DeviceIndex device,
    MempoolId_t mempool_id,
    const std::unordered_set<void*>& expected_live_allocations) override {
  // 调用指定设备分配器的方法，检查内存池中的实际活动分配情况与预期是否一致
  return device_allocator[device]->checkPoolLiveAllocations(
      mempool_id, expected_live_allocations);
}

// 注册内存分配异常观察器到所有设备分配器中
void attachOutOfMemoryObserver(OutOfMemoryObserver observer) override {
  // 遍历所有设备分配器，注册内存分配异常观察器
  for (auto& allocator : device_allocator) {
    allocator->attachOutOfMemoryObserver(observer);
  }
}

// 注册分配器跟踪器到所有设备分配器中
void attachAllocatorTraceTracker(AllocatorTraceTracker tracker) override {
  // 遍历所有设备分配器，注册分配器跟踪器
  for (auto& allocator : device_allocator) {
    allocator->attachAllocatorTraceTracker(tracker);
  }
}

// 清空所有设备上的缓存
void emptyCache() override {
    // 遍历 device_allocator 中的每个设备分配器，调用其 emptyCache 方法
    for (auto& da : device_allocator)
      da->emptyCache();
  }

  // 获取指针所在块的基本分配信息，并返回其指针和大小
  void* getBaseAllocation(void* ptr, size_t* outSize) override {
    // 根据指针获取其所在的 Block 对象
    Block* block = get_allocated_block(ptr);
    // 如果未找到对应的 Block，抛出错误并显示无效设备指针
    if (!block) {
      TORCH_CHECK(false, "invalid device pointer: ", ptr);
    }
    // 调用对应设备的 getBaseAllocation 方法获取基本分配信息
    return device_allocator[block->device]->getBaseAllocation(block, outSize);
  }

  // 记录指定数据指针与 CUDA 流的关联关系
  void recordStream(const DataPtr& ptr, cuda::CUDAStream stream) override {
    // 空张量的 storage().data() 可能为空指针，因为这些张量没有关联的块，因此在此不做任何操作
    if (!ptr.get()) {
      return;
    }

    // 如果张量不是由当前实例分配的，则跳过记录
    // 这通常发生在跨进程共享 CUDA 张量时，我们实现了基于引用计数的共享机制，
    // 以确保张量在一个进程释放时不会在另一个进程中仍在使用时被意外释放
    if (ptr.get_deleter() != &local_raw_delete)
      return;

    // 获取指针所在的 Block 对象，确保其不为空
    Block* block = get_allocated_block(ptr.get());
    TORCH_INTERNAL_ASSERT(block != nullptr, "No allocated block can be found");
    // 调用对应设备分配器的 recordStream 方法记录流信息
    device_allocator[block->device]->recordStream(block, stream);
  }

  // 生成当前状态的快照信息
  SnapshotInfo snapshot() override {
    // 设置转换器，用于将时间戳从 TSC 转换为微秒
    auto tsc_to_ns = clock_converter.makeConverter();
    auto tsc_to_us = [=](approx_time_t t_approx) {
      return tsc_to_ns(t_approx) / 1000;
    };

    // 初始化快照信息
    SnapshotInfo result;
    // 遍历每个设备分配器，记录设备轨迹和快照信息
    for (auto& da : device_allocator) {
      result.device_traces.emplace_back(da->trace(tsc_to_us));
      auto snap = da->snapshot();
      result.segments.insert(result.segments.end(), snap.begin(), snap.end());
    }

    // 配置元数据信息
    auto& md = result.config_metadata;
    md.garbage_collection_threshold =
        CUDAAllocatorConfig::garbage_collection_threshold();
    md.max_split_size = CUDAAllocatorConfig::max_split_size();
    md.pinned_num_register_threads =
        CUDAAllocatorConfig::pinned_num_register_threads();
    md.expandable_segments = CUDAAllocatorConfig::expandable_segments();
    md.release_lock_on_malloc =
        CUDAAllocatorConfig::release_lock_on_cudamalloc();
    md.pinned_use_host_register =
        CUDAAllocatorConfig::pinned_use_cuda_host_register();
    md.last_allocator_settings = CUDAAllocatorConfig::last_allocator_settings();
    md.roundup_power2_divisions =
        CUDAAllocatorConfig::roundup_power2_divisions();

    // 返回生成的快照信息
    return result;
  }

  // 获取指定设备和内存池 ID 的检查点状态
  std::shared_ptr<AllocatorState> getCheckpointState(
      c10::DeviceIndex device,
      MempoolId_t id) override {
    return device_allocator[device]->getCheckpointState(id);
  }

  /**
   * @brief 根据给定的设备和分配器状态将私有池状态恢复到其之前的状态
   *
   * @param device - 要操作的池的设备
   * @param as - 分配器状态
   * @param stale_live_storages - 当前分配但在设置检查点后不再分配的张量存储。
   * 对于这些存储，我们将移除它们的删除函数。
   * @return CheckpointDelta - 释放指针和包含新检查点状态中所有分配块的数据指针的 deleter 函数。
   */
  CheckpointDelta setCheckpointPoolState(
      c10::DeviceIndex device,
      std::shared_ptr<AllocatorState> as) override {
    // 将 as 转换为 PrivatePoolState 共享指针
    std::shared_ptr<PrivatePoolState> pps =
        std::dynamic_pointer_cast<PrivatePoolState>(as);

    // 断言 pps 不为空，期望其为 PrivatePoolState
    TORCH_CHECK(pps, "Expected PrivatePoolState");

    // 调用设备分配器的 setCheckpointPoolState 方法，获取返回结果 rr
    auto rr = device_allocator[device]->setCheckpointPoolState(*pps);

    // 初始化 CheckpointDelta 对象 cpd
    CheckpointDelta cpd;

    // 遍历 rr 中的 allocations_freed，释放每个指针并将其添加到 cpd.ptrs_freed 中
    for (void* ptr : rr.allocations_freed) {
      get_allocated_block(ptr, /*remove*/ true);
      cpd.ptrs_freed.push_back(ptr);
    }

    // 遍历 rr 中的 allocations_created，添加每个块并将其数据指针信息添加到 cpd.dataptrs_allocd 中
    for (Block* block : rr.allocations_created) {
      add_allocated_block(block);
      cpd.dataptrs_allocd.emplace_back(
          block->ptr,
          block->ptr,
          &local_raw_delete,
          Device(DeviceType::CUDA, device));
    }

    // 返回 cpd 对象
    return cpd;
  }

  /**
   * @brief 分配给定大小的内存块
   *
   * @param size - 要分配的内存块大小
   * @return DataPtr - 包含分配的设备指针、删除函数及其设备类型信息的数据指针
   */
  DataPtr allocate(size_t size) override {
    // 定义 1 EB（exabyte）的字节数
    constexpr size_t one_exa_bytes = 1152921504606846976ULL;

    // 断言不超过 1 EB 大小的内存分配
    TORCH_CHECK_WITH(
        OutOfMemoryError,
        size < one_exa_bytes,
        "CUDA out of memory. Tried to allocate more than 1EB memory.");

    // 获取当前设备索引
    c10::DeviceIndex device = 0;
    C10_CUDA_CHECK(c10::cuda::GetDevice(&device));

    // 初始化设备指针和删除函数
    void* devPtr = nullptr;
    void (*deleteFunc)(void*) = &local_raw_delete;

    // 获取当前 CUDA 流
    CUDAStream stream = cuda::getCurrentCUDAStream(device);

    // 如果强制使用非缓存分配器
    if (forceUncachedAllocator()) {
      deleteFunc = &uncached_delete;

      // 分配内存并在可能时跟踪 GPU 内存分配
      C10_CUDA_CHECK(cudaMalloc(&devPtr, size));
      const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
      if (C10_UNLIKELY(interp)) {
        (*interp)->trace_gpu_memory_allocation(
            c10::kCUDA, reinterpret_cast<uintptr_t>(devPtr));
      }
    } else {
      // 如果大小不为 0，则使用缓存分配器分配内存
      if (size != 0) {
        this->malloc(&devPtr, device, size, stream);
      }
    }

    // 如果大小不为 0 且开启了 malloc 的 SDT（Software Development Tools）追踪，记录追踪信息
    if (size && TORCH_SDT_IS_ENABLED(malloc)) {
      TORCH_SDT_WITH_SEMAPHORE(malloc, devPtr, device, size, stream.id());
    }

    // 返回 DataPtr 对象，包含分配的设备指针、删除函数及其设备类型信息
    return {devPtr, devPtr, deleteFunc, Device(DeviceType::CUDA, device)};
  }

  /**
   * @brief 返回原始删除函数指针
   *
   * @return DeleterFnPtr - 指向适当删除函数的指针，根据是否强制使用非缓存分配器而变化
   */
  DeleterFnPtr raw_deleter() const override {
    // 如果强制使用非缓存分配器，则返回 uncached_delete 函数指针，否则返回 local_raw_delete 函数指针
    if (forceUncachedAllocator()) {
      return &uncached_delete;
    } else {
      return &local_raw_delete;
    }
  }

  void cacheInfo(c10::DeviceIndex device, size_t* largestBlock) override {
  // 调用 device_allocator 中对应设备的 cacheInfo 方法，传入 largestBlock 参数
  device_allocator[device]->cacheInfo(largestBlock);
}

// 确保设备索引有效，若无效则抛出异常
void assertValidDevice(c10::DeviceIndex device) {
  // 获取设备分配器的数量
  const auto device_num = device_allocator.size();
  // 检查设备索引是否在有效范围内，否则抛出错误信息
  TORCH_CHECK(
      0 <= device && device < static_cast<int64_t>(device_num),
      "Invalid device argument ",
      device,
      ": did you call init?");
}

// 获取指定设备的设备统计信息
DeviceStats getDeviceStats(c10::DeviceIndex device) override {
  // 确保设备索引有效
  assertValidDevice(device);
  // 返回设备分配器中对应设备的统计信息
  return device_allocator[device]->getStats();
}

// 重置指定设备的累积统计信息
void resetAccumulatedStats(c10::DeviceIndex device) override {
  // 确保设备索引有效
  assertValidDevice(device);
  // 调用设备分配器中对应设备的 resetAccumulatedStats 方法
  device_allocator[device]->resetAccumulatedStats();
}

// 重置指定设备的峰值统计信息
void resetPeakStats(c10::DeviceIndex device) override {
  // 确保设备索引有效
  assertValidDevice(device);
  // 调用设备分配器中对应设备的 resetPeakStats 方法
  device_allocator[device]->resetPeakStats();
}

// 开始为 CUDA 图形分配内存池中的空间
void beginAllocateToPool(
    c10::DeviceIndex device,
    MempoolId_t mempool_id,
    std::function<bool(cudaStream_t)> filter) override {
  // 确保设备索引有效
  assertValidDevice(device);
  // 调用设备分配器中对应设备的 beginAllocateToPool 方法
  device_allocator[device]->beginAllocateToPool(
      std::move(mempool_id), std::move(filter));
}

// 结束为 CUDA 图形分配内存池中的空间
void endAllocateToPool(c10::DeviceIndex device, MempoolId_t mempool_id)
    override {
  // 确保设备索引有效
  assertValidDevice(device);
  // 调用设备分配器中对应设备的 endAllocateToPool 方法
  device_allocator[device]->endAllocateToPool(mempool_id);
}

// 释放指定设备上指定内存池的内存空间
void releasePool(c10::DeviceIndex device, MempoolId_t mempool_id) override {
  // 确保设备索引有效
  assertValidDevice(device);
  // 调用设备分配器中对应设备的 releasePool 方法
  device_allocator[device]->releasePool(std::move(mempool_id));
}

// 分配指定大小的内存，并返回其指针
void* raw_alloc(size_t nbytes) override {
  if (nbytes == 0) {
    return nullptr;
  }
  c10::DeviceIndex device = 0;
  // 获取当前 CUDA 设备索引
  C10_CUDA_CHECK(c10::cuda::GetDevice(&device));
  void* r = nullptr;
  // 调用 malloc 方法分配内存，并返回分配的内存指针
  malloc(&r, device, nbytes, cuda::getCurrentCUDAStream(device));
  return r;
}

// 在指定流上分配指定大小的内存，并返回其指针
void* raw_alloc_with_stream(size_t nbytes, cudaStream_t stream) override {
  if (nbytes == 0) {
    return nullptr;
  }
  c10::DeviceIndex device = 0;
  // 获取当前 CUDA 设备索引
  C10_CUDA_CHECK(c10::cuda::GetDevice(&device));
  void* r = nullptr;
  // 调用 malloc 方法分配内存，并返回分配的内存指针
  malloc(&r, device, nbytes, stream);
  return r;
}

// 启用两个设备之间的对等访问
void enablePeerAccess(c10::DeviceIndex dev, c10::DeviceIndex dev_to_access)
    override {
  // 使用 CUDAGuard 设置当前设备
  c10::cuda::CUDAGuard device_guard(dev);
  // 尝试启用设备之间的对等访问
  cudaError_t err = cudaDeviceEnablePeerAccess(dev_to_access, 0);
  if (err == cudaErrorPeerAccessAlreadyEnabled) {
    // 如果对等访问已经启用，忽略错误并清除错误状态
    (void)cudaGetLastError();
  } else {
    // 否则，检查并抛出 CUDA 错误
    C10_CUDA_CHECK(err);
  }
  // 向设备分配器中的 dev_to_access 设备添加对等访问
  device_allocator[dev_to_access]->addPeerAccess(dev);
}

// 在指定流上异步执行内存拷贝操作
cudaError_t memcpyAsync(
    void* dst,
    int dstDevice,
    const void* src,
    int srcDevice,
    size_t count,
    cudaStream_t stream,
    bool p2p_enabled) override {
    // 如果启用了点对点传输（p2p_enabled），或者内存在同一设备上映射，则可以使用 memcpy
    // 如果源设备等于目标设备，也可以使用 memcpy
    // 当目标设备和源设备都来自 cudaMalloc 时，也可以使用 memcpy
    if (p2p_enabled ||
        srcDevice == dstDevice ||
        (!device_allocator[dstDevice]->hasAllocatedExpandableSegments() &&
         !device_allocator[srcDevice]->hasAllocatedExpandableSegments())) {
      // 在当前流上异步执行 cudaMemcpyDeviceToDevice 操作
      return cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToDevice, stream);
    }
    // 当未启用 p2p 时，只有 cudaMemcpyPeerAsync 可以正确处理未通过 cudaMalloc 分配的内存
    // 返回 cudaMemcpyPeerAsync 操作的结果
    return cudaMemcpyPeerAsync(dst, dstDevice, src, srcDevice, count, stream);
  }

  // 覆盖基类的 raw_delete 函数
  void raw_delete(void* ptr) override {
    // 调用当前类的 free 函数释放指针 ptr
    this->free(ptr);
  }

  // 在 CUDA IPC 中，发送进程将张量发送给接收进程，getIpcDevPtr 被接收进程调用
  // 用于将发送进程的 CUDA 内存映射到自己的地址空间中。
  //
  // CUDA IPC 只允许共享与 cudaIpcMemHandle_t 关联的大内存块，
  // 每个进程每个上下文只能打开一次。同一 IPC 内存块中可以有多种类型的存储，
  // 因此我们必须缓存设备指针以在需要时构造类型化存储。
  //
  // ipcMemHandle_to_devptr 将 cudaIpcMemHandle_t 映射到在接收进程中可以用于访问
  // 发送进程内存块的设备指针。它只保存设备指针的弱引用在映射中，
  // 共享指针将用于在此 CudaMalloc 分配中重建所有存储，并且在引用计数为 0 时
  // 将在 cudaIpcCloseMemHandle 中删除。
  //
  std::mutex IpcMutex; // 定义一个互斥量以保护 IPC 操作
  ska::flat_hash_map<std::string, std::weak_ptr<void>> ipcMemHandle_to_devptr; // 使用 flat_hash_map 存储 IPC 句柄到设备指针的映射
  std::shared_ptr<void> getIpcDevPtr(std::string handle) override {
    std::lock_guard<std::mutex> lock(IpcMutex); // 使用互斥量进行加锁

    auto iter = ipcMemHandle_to_devptr.find(handle); // 查找 IPC 句柄对应的设备指针
    if (iter != ipcMemHandle_to_devptr.end()) {
      auto devptr = iter->second.lock(); // 尝试获取设备指针的强引用
      if (devptr)
        return devptr; // 如果设备指针有效，则返回共享指针
    }
    // 如果 ipcMemHandle 尚未打开，或者已经过期，则打开它以启用 IPC 访问该内存块。
    void* dev = nullptr;
    auto ipc_handle =
        reinterpret_cast<const cudaIpcMemHandle_t*>(handle.c_str()); // 将 IPC 句柄转换为 cudaIpcMemHandle_t
    C10_CUDA_CHECK(cudaIpcOpenMemHandle(
        &dev, *ipc_handle, cudaIpcMemLazyEnablePeerAccess)); // 打开 IPC 句柄以启用对应内存块的 Peer 访问
    // devPtr 必须在创建时在相同设备上删除。
    c10::DeviceIndex curr_device = 0; // 定义当前设备索引
    C10_CUDA_CHECK(c10::cuda::GetDevice(&curr_device)); // 获取当前 CUDA 设备
    auto sp =
        std::shared_ptr<void>(dev, [handle, curr_device, this](void* ptr) {
          cuda::CUDAGuard device_guard(curr_device); // 切换到当前设备的 CUDA Guard
          std::lock_guard<std::mutex> deleter_lock(IpcMutex); // 使用互斥量保护删除操作
          C10_CUDA_CHECK(cudaIpcCloseMemHandle(ptr)); // 关闭 IPC 句柄对应的内存块
          ipcMemHandle_to_devptr.erase(handle); // 从映射中删除 IPC 句柄
        });
    std::weak_ptr<void> wp = sp; // 创建指向共享指针的弱引用，用于存储在映射中
    // 为了避免额外的搜索，可以使用 insert() 来插入映射。
    // 插入键值对到 ipcMemHandle_to_devptr 映射中，如果键已存在则不覆盖（ptr expired 指针已过期）。
    // 由于在 sp 的删除器中已经删除了条目，所以现在进行这样的操作应该是安全的。
    ipcMemHandle_to_devptr.insert(iter, {handle, wp});

    // 返回字符串 "native"，覆盖基类中的虚函数 name()
    return sp;
  }
  // 返回字符串 "native"，实现基类中的纯虚函数 name()
  std::string name() override {
    return "native";
  }
  // 使用 CUDA 函数 cudaMemcpy 在设备之间拷贝数据
  void copy_data(void* dest, const void* src, std::size_t count) const final {
    // 进行 CUDA 错误检查，拷贝数据从源地址 src 到目标地址 dest，拷贝 count 字节
    C10_CUDA_CHECK(
        cudaMemcpy(dest, src, count, cudaMemcpyKind::cudaMemcpyDeviceToDevice));
  }
};

// 定义了一个名为 allocator 的 NativeCachingAllocator 对象
NativeCachingAllocator allocator;

// 本地内存释放函数，用于释放指针 ptr 所指向的内存
void local_raw_delete(void* ptr) {
  // 如果 TORCH_SDT_IS_ENABLED 宏启用了 free 事件跟踪
  if (TORCH_SDT_IS_ENABLED(free)) {
    // 在释放指针 ptr 前执行 free 事件跟踪
    TORCH_SDT_WITH_SEMAPHORE(free, ptr);
  }

  // 调用 allocator 对象的 free 方法释放 ptr 指向的内存
  allocator.free(ptr);
}

} // namespace Native

// 用于将大小 size 格式化为人类可读的字符串表示
std::string format_size(uint64_t size) {
  std::ostringstream os;
  os.precision(2);
  os << std::fixed;
  // 根据大小 size 的不同范围选择不同的单位进行格式化
  if (size <= 1024) {
    os << size << " bytes";
  } else if (size <= 1048576) {
    os << (static_cast<double>(size) / 1024.0);
    os << " KiB";
  } else if (size <= 1073741824ULL) {
    os << static_cast<double>(size) / 1048576.0;
    os << " MiB";
  } else {
    os << static_cast<double>(size) / 1073741824.0;
    os << " GiB";
  }
  // 返回格式化后的字符串
  return os.str();
}

namespace CudaMallocAsync {
// 如果将其放在单独的头文件中，它会在 HIPify 过程中被错误重命名
CUDAAllocator* allocator();

} // namespace CudaMallocAsync

// BackendStaticInitializer 结构体，用于在加载时解析后端环境变量，并根据配置选择 CUDAAllocator
struct BackendStaticInitializer {
  // 解析环境变量 PYTORCH_CUDA_ALLOC_CONF，以确定后端配置
  CUDAAllocator* parseEnvForBackend() {
    const char* val = getenv("PYTORCH_CUDA_ALLOC_CONF");
    if (val != nullptr) {
      const std::string config(val);

      // 使用正则表达式分割配置字符串
      std::regex exp("[\\s,]+");
      std::sregex_token_iterator it(config.begin(), config.end(), exp, -1);
      std::sregex_token_iterator end;
      std::vector<std::string> options(it, end);

      // 遍历配置项，根据 "backend" 配置选择合适的分配器
      for (auto option : options) {
        std::regex exp2("[:]+");
        std::sregex_token_iterator it2(option.begin(), option.end(), exp2, -1);
        std::sregex_token_iterator end2;
        std::vector<std::string> kv(it2, end2);
        if (kv.size() >= 2) {
          if (kv[0] == "backend") {
            if (kv[1] == "cudaMallocAsync")
              return CudaMallocAsync::allocator();
            if (kv[1] == "native")
              return &Native::allocator;
          }
        }
      }
    }
    // 默认返回 Native::allocator
    return &Native::allocator;
  }

  // 构造函数，在初始化时解析并设置 allocator 对象
  BackendStaticInitializer() {
    auto r = parseEnvForBackend();
    allocator.store(r);
  }
};

// atomic 类型的 allocator 变量，用于存储 CUDAAllocator 指针
std::atomic<CUDAAllocator*> allocator;
// 创建 BackendStaticInitializer 的全局对象，用于后端初始化
BackendStaticInitializer backend_static_initializer;

} // namespace cuda::CUDACachingAllocator

} // namespace c10
```