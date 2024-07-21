# `.\pytorch\aten\src\ATen\mps\MPSAllocator.h`

```
// 版权声明，该部分代码版权归 Apple Inc. 所有
#pragma once

// 引入 MPSAllocatorInterface.h 头文件，该文件定义了 Metal Heaps 分配器接口
#include <ATen/mps/MPSAllocatorInterface.h>
// 引入 MPSEvent.h 头文件，用于 Metal Performance Shaders (MPS) 事件处理
#include <ATen/mps/MPSEvent.h>
// 引入 MPSStream.h 头文件，用于 Metal Performance Shaders (MPS) 流处理
#include <ATen/mps/MPSStream.h>

#include <cstdio>                      // C 标准输入输出
#include <mutex>                       // 互斥锁
#include <set>                         // STL 集合容器
#include <unordered_set>               // STL 无序集合容器
#include <mach/vm_page_size.h>         // macOS 内存页大小
#include <c10/util/flat_hash_map.h>    // 使用了 FlatHashMap 实现的哈希映射

// 此实现基于 CUDACachingAllocator。
// 使用 Metal Heaps 来提高缓冲区分配的性能。
// 不要直接包含此头文件，应该使用 MPSAllocatorInterface.h。
// TODO: 统一与 CUDACachingAllocator 的逻辑，并移除冗余代码。

namespace at::mps::HeapAllocator {

// 定义常量，指定不同类型的内存块大小阈值
static const size_t kMaxSmallAlloc = MB(1);    // 最大的 "小" 分配为 1 MiB
static const size_t kMinLargeAlloc = MB(10);   // 分配大小介于 1 到 10 MiB 之间可以使用 kLargeHeap
static const size_t kRoundLarge    = MB(2);    // 大的分配大小向上舍入为 2 MiB
static const size_t kSmallHeap     = MB(8);    // "小" 分配封装在 8 MiB 的堆中
static const size_t kLargeHeap     = MB(32);   // "大" 分配封装在 32 MiB 的堆中
static const size_t kXLargeHeapD   = MB(128);  // "超大" 分配（离散设备）封装在 128 MiB 的堆中
static const size_t kXLargeHeapU   = MB(1024); // "超大" 分配（统一内存设备）封装在 1 GiB 的堆中
static const size_t kMaxScalarAlloc = (sizeof(int64_t)); // 最大的 "标量" 分配大小为 int64_t 的大小

// 缓冲池可以根据不同的使用标志进行定制化
enum UsageFlags : uint32_t {
  PRIVATE = 0,            // 私有池，非共享
  SMALL   = (1 << 0),     // "小" 堆的标志，大小为 kSmallHeap；"大" 堆的标志为 kLargeHeap
  SHARED  = (1 << 1),     // 在具有统一内存的设备上共享池；否则在主机和设备之间私有
  MANAGED = (1 << 2),     // 管理存储模式
  HAZARD  = (1 << 3),     // 启用资源的自动危险跟踪
  SCALAR  = (1 << 4),     // 用于将 CPU 标量值导入 GPU 并在 MPS 流中使用
};

// 调试详细级别标志
enum DebugVerbosity : uint32_t {
  SILENT      = 0,        // 静默模式，不输出调试信息
  PROFILING   = (1 << 0), // 输出系统内存使用的一般性分析数据
  ALLOCATIONS = (1 << 1), // 输出缓冲区分配信息
  RECYCLES    = (1 << 2), // 输出缓冲区回收信息
  RELEASES    = (1 << 3), // 输出缓冲区释放信息
  LARGE_ONLY  = (1 << 4), // 仅记录大缓冲池的事务信息
};

struct HeapBlock;
  // 定义一个结构体 BufferBlock，用于管理 Metal 缓冲区
struct BufferBlock {
  id<MTLBuffer> buffer; // Metal 缓冲区对象
  void* cpu_ptr = nullptr; // 指向共享 Metal 缓冲区在 CPU 上映射的指针
  size_t size; // 对齐后的缓冲区大小
  size_t requested_size; // 请求的缓冲区大小（未对齐）
  std::vector<int64_t> shape; // 缓冲区形状，用于在缓存的图中获取视图的基础
  bool in_use = false; // 缓冲区是否正在使用
  HeapBlock* heap; // 指向所属堆块的指针
  id_t buf_id; // 缓冲区块的唯一标识符
  uint32_t gc_count = 0; // 用于垃圾回收的候选最近最少使用的计数器
  uint32_t use_count = 0; // 缓冲区块使用计数器
  static uint64_t buffer_counter; // 缓冲区块唯一标识符计数器
  MPSEventPtr event; // 用于同步 GPU/CPU 操作的 Metal 事件

  // 构造函数，初始化缓冲区块对象
  BufferBlock(size_t Size, size_t RequestedSize = 0, const id<MTLBuffer> Buffer = nullptr,
              HeapBlock* Heap = nullptr) :
              buffer(Buffer), size(Size), requested_size(RequestedSize),
              heap(Heap), buf_id(Buffer ? ++buffer_counter : 0) { }

  // 静态方法，用于比较两个缓冲区块大小及其缓冲区对象的地址
  static bool Comparator(const BufferBlock* a, const BufferBlock* b) {
    return (a->size != b->size) ? a->size < b->size : (uintptr_t)a->buffer < (uintptr_t)b->buffer;
  }

  // 静态方法，将指定大小按指定对齐方式向上对齐
  static size_t alignUp(size_t Size, size_t Alignment) {
    assert(((Alignment - 1) & Alignment) == 0); // 断言，确保对齐值是 2 的幂次方
    return ((Size + Alignment - 1) & ~(Alignment - 1));
  }

  // 返回 Metal 缓冲区的保留计数
  uint32_t retainCount() const { return [buffer retainCount]; }
};

// BufferBlock 指针比较函数类型
typedef bool (*BufferComparison)(const BufferBlock*, const BufferBlock*);

// 分配参数结构体，用于管理缓冲池的分配操作
struct AllocParams {
  AllocParams(size_t Alloc_Size, size_t Requested_Size, BufferPool* Pool) :
              search_key(Alloc_Size), pool(Pool), requested_size(Requested_Size) { }
  
  // 返回搜索键的大小
  size_t size() const { return search_key.size; }

  BufferBlock search_key; // 搜索的缓冲区块
  BufferPool* pool; // 指向所属缓冲池的指针
  BufferBlock* buffer_block = nullptr; // 分配的缓冲区块
  size_t requested_size; // 请求的缓冲区大小
  bool has_memory_pressure = false; // 是否超出低水位限制，需要应用策略进行缓解压力
  bool has_unified_memory = true; // 是否在统一内存设备上分配
};

// 堆块结构体，用于管理 Metal 堆
struct HeapBlock {
  id<MTLHeap> heap; // Metal 堆对象
  struct { size_t total, available; } size; // 堆的总大小和可用大小
  BufferPool* pool; // 指向所属缓冲池的指针
  unsigned int n_buffers = 0; // 堆内缓冲区块数量
  id_t heap_id; // 堆块的唯一标识符
  bool is_split; // 指示是否将此堆拆分以子分配多个缓冲区块，否则为单一缓冲区块
  static uint64_t heap_counter; // 堆块唯一标识符计数器

  // 构造函数，初始化堆块对象
  HeapBlock(size_t Size, const id<MTLHeap> Heap = nullptr, BufferPool *Pool = nullptr) :
            heap(Heap), size({.total = Size, .available = Size}), pool(Pool),
            heap_id(Heap ? ++heap_counter : 0), is_split(true) { }

  // 静态方法，获取 Metal 资源选项
  static MTLResourceOptions getOptions(uint32_t usage) {
    // TODO: check the caching performance of write-combined mode
    MTLResourceOptions options = MTLResourceCPUCacheModeDefaultCache;

    if (usage & UsageFlags::MANAGED)
      options |= MTLResourceStorageModeManaged;

    // 返回 Metal 资源选项
    return options;
  }
};
    // 如果资源用途包含共享标志，则设置选项为共享存储模式
    else if (usage & UsageFlags::SHARED)
      options |= MTLResourceStorageModeShared;
    // 否则设置选项为私有存储模式
    else
      options |= MTLResourceStorageModePrivate;

    // 根据使用标志中的危险标志设置选项的危险跟踪模式
    options |= (usage & UsageFlags::HAZARD) ? MTLResourceHazardTrackingModeTracked : MTLResourceHazardTrackingModeUntracked;

    // 返回设置好的选项
    return options;
  }

  // 根据参数创建一个堆块
  static HeapBlock* createHeapBlock(AllocParams& params, id<MTLDevice> device, uint32_t usage) {
    HeapBlock *heapBlock = nullptr;
    bool is_split = true;
    const size_t size = params.size();
    MTLHeapDescriptor *d = [MTLHeapDescriptor new];
    if (d) {
      const size_t kXLargeHeap = params.has_unified_memory ? kXLargeHeapU : kXLargeHeapD;
      // 根据大小设置堆块描述符的大小
      if (size <= kMaxSmallAlloc) {
        d.size = kSmallHeap;
      } else if (size < kMinLargeAlloc) {
        d.size = kLargeHeap;
      } else if (size < kXLargeHeap / 2 && !params.has_memory_pressure) {
        d.size = kXLargeHeap;
      } else {
        d.size = kRoundLarge * ((size + kRoundLarge - 1) / kRoundLarge);
        is_split = false;
      }
      // 根据使用标志设置存储模式
      d.storageMode = (usage & UsageFlags::SHARED) ? MTLStorageModeShared : MTLStorageModePrivate;
      d.cpuCacheMode = MTLCPUCacheModeDefaultCache;
      // 根据使用标志设置危险跟踪模式
      // 这会自动处理 Metal 缓冲区访问的同步，性能稍微降低。
      d.hazardTrackingMode = (usage & UsageFlags::HAZARD) ? MTLHazardTrackingModeTracked : MTLHazardTrackingModeUntracked;
      // 根据使用标志获取资源选项
      d.resourceOptions = getOptions(usage);
      d.type = MTLHeapTypeAutomatic;
      // 使用设备创建一个堆
      id<MTLHeap> heap = [device newHeapWithDescriptor: d];
      if (heap) {
        // 设置堆的可清除状态为非易失性
        [heap setPurgeableState:MTLPurgeableStateNonVolatile];
        // 获取堆的可用大小
        const size_t heap_size = heapAvailableSize(heap);
        // 创建堆块对象
        heapBlock = new HeapBlock(heap_size, heap, params.pool);
        if (heapBlock) {
          // 设置堆块是否分割
          heapBlock->is_split = is_split;
        }
      }
      [d release]; // 释放堆描述符对象
    }
    return heapBlock; // 返回创建的堆块对象或者 nullptr
  }

  // 比较函数，用于堆块对象的大小排序
  static bool Comparator(const HeapBlock* a, const HeapBlock* b) {
    return (a->size.available != b->size.available) ? a->size.available < b->size.available :
                                                      (uintptr_t)a->heap < (uintptr_t)b->heap;
  }

  // 获取堆的可用大小，可以指定对齐方式，默认为页面大小对齐
  static NSUInteger heapAvailableSize(id<MTLHeap> heap, size_t Alignment = vm_page_size) {
    return [heap maxAvailableSizeWithAlignment:Alignment];
  }

  // 返回堆块的大小
  NSUInteger Size() {
    return [heap size];
  }

  // 创建一个新的 Metal 缓冲区对象
  id<MTLBuffer> newMTLBuffer(size_t length, uint32_t usage) {
    // 使用堆创建一个指定长度和选项的 Metal 缓冲区
    id<MTLBuffer> buf = [heap newBufferWithLength:length options:getOptions(usage)];
    if (buf) {
      // 更新可用大小
      updateAvailableSize();
      // 增加缓冲区计数
      n_buffers++;
    }
    return buf; // 返回创建的 Metal 缓冲区对象或者 nil
  }

  // 释放 Metal 缓冲区对象，返回释放前的保留计数
  uint32_t releaseMTLBuffer(id<MTLBuffer>& buffer) {
    const uint32_t retainCount = [buffer retainCount];
    [buffer release]; // 释放缓冲区
    buffer = nil; // 置空缓冲区引用
    // 更新可用大小
    updateAvailableSize();
    // 减少缓冲区计数
    n_buffers--;
    return retainCount; // 返回释放前的保留计数
  }

  // 释放堆对象，返回释放前的保留计数
  uint32_t releaseMTLHeap() {
    // 获取堆对象的保留计数
    const uint32_t retainCount = [heap retainCount];
    // 断言堆对象不包含任何缓冲区，如果不是空的则抛出错误
    TORCH_INTERNAL_ASSERT(!n_buffers);
    // 将堆对象设置为可释放状态为空
    [heap setPurgeableState:MTLPurgeableStateEmpty];
    // 释放堆对象的所有权
    [heap release];
    // 将堆对象引用置为nil，指向空对象
    heap = nil;
    // 将可用大小的成员变量设置为0
    size.available = 0;
    // 返回最初获取的堆对象的保留计数
    return retainCount;
  }
  // 返回堆对象的当前保留计数
  uint32_t retainCount() const { return [heap retainCount]; }
  // 更新可用大小的成员变量，通过查询堆对象的可用大小函数
  void updateAvailableSize() { size.available = heapAvailableSize(heap); }
};
typedef bool (*HeapComparison)(const HeapBlock*, const HeapBlock*);

// 结构体定义：BufferPool，用于管理不同类型的缓冲池
struct BufferPool {
  // 枚举类型 Kind，表示不同类型的缓冲池
  enum class Kind {
    PRIVATE_SMALL,
    PRIVATE_LARGE,
    SHARED_SMALL,
    SHARED_LARGE,
    SCALAR,
  };

  // 构造函数，初始化 BufferPool 对象
  BufferPool(const id<MTLDevice> Device, uint32_t Usage) :
             device(Device), usage(Usage),
             // 初始化成员变量 heaps 和 available_buffers，使用各自的比较函数
             heaps(HeapBlock::Comparator), available_buffers(BufferBlock::Comparator) { }

  // 成员变量：Metal 设备对象
  const id<MTLDevice> device;
  // 成员变量：用于自定义池的使用标志（参见 UsageFlags 枚举）
  const uint32_t usage;
  // 成员变量：池中的缓冲区总数
  uint32_t n_buffers = 0;
  // 成员变量：在此池中分配的总大小
  size_t allocated_size = 0;
  // 成员变量：池中可用的总内存大小
  size_t available_size = 0;
  // 成员变量：按可用内存大小排序的堆块列表
  std::set<HeapBlock*, HeapComparison> heaps;
  // 成员变量：仅包含可用缓冲区的列表（即未在使用中的缓冲区）
  std::set<BufferBlock*, BufferComparison> available_buffers;
  // 成员变量：处于“悬空”状态的缓冲区列表，这些缓冲区已从 PyTorch 端释放，
  // 但由于仍被 retainCount > 1 的命令缓冲区使用，无法返回到池中。
  // 这些缓冲区将在命令缓冲区的 completionHandler 回调调用后返回到池中。
  std::unordered_set<BufferBlock*> buffers_pending_free;
  // 成员变量：待更新大小的堆块列表
  std::unordered_set<HeapBlock*> heaps_pending_update;
};

// 类定义：MPSHeapAllocatorImpl，实现 MPS 堆分配器
class MPSHeapAllocatorImpl {
public:
  // 构造函数，初始化 MPSHeapAllocatorImpl 对象
  explicit MPSHeapAllocatorImpl() :
    // 获取 MPS 设备并初始化成员变量
    m_device(at::mps::MPSDevice::getInstance()->device()),
    // 获取设备的最大缓冲区长度并初始化成员变量
    m_max_buffer_size([m_device maxBufferLength]),
    // 获取默认的 MPS 流并初始化成员变量
    m_stream(getDefaultMPSStream()),
    // 获取 MPS 事件池并初始化成员变量
    m_event_pool(getMPSEventPool()) {
    // 初始化分配器
    init_allocator();
  }

  // 析构函数，用于清理 MPSHeapAllocatorImpl 对象
  ~MPSHeapAllocatorImpl() {
    // 遍历 MPSAllocatorCallbacksRegistry 中的键，并执行 MPS 分配器回调
    for (const auto& name : MPSAllocatorCallbacksRegistry()->Keys()) {
      MPSAllocatorCallbacksRegistry()->Create(name)->executeMPSAllocatorCallback(buffer_block ? buffer_block->buffer : nullptr, event);
    }
    // 返回析构函数的执行结果
    return true;
  }
};

// 命名空间结束标记：at::mps::HeapAllocator
} // namespace at::mps::HeapAllocator
```