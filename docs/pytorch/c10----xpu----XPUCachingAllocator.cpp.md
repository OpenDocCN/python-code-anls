# `.\pytorch\c10\xpu\XPUCachingAllocator.cpp`

```
// 包含 C10 库中的相关头文件
#include <c10/util/flat_hash_map.h>
#include <c10/util/irange.h>
#include <c10/xpu/XPUCachingAllocator.h>

#include <deque>
#include <mutex>
#include <set>
#include <vector>

// 定义命名空间 c10::xpu::XPUCachingAllocator
namespace c10::xpu::XPUCachingAllocator {

// 新分配的内存以 512 字节对齐
constexpr size_t kDeviceAlignment = 512;
// 所有大小至少都要舍入到 512 字节
constexpr size_t kMinBlockSize = 512;
// 最大的 "小" 分配是 1 MiB
constexpr size_t kSmallSize = 1048576;
// "小" 分配在 2 MiB 的块中打包
constexpr size_t kSmallBuffer = 2097152;
// "大" 分配可以打包在 20 MiB 的块中
constexpr size_t kLargeBuffer = 20971520;
// 大小在 1 MiB 到 10 MiB 之间的分配可以使用 kLargeBuffer
constexpr size_t kMinLargeAlloc = 10485760;
// 将大分配舍入到 2 MiB
constexpr size_t kRoundLarge = 2097152;

// 匿名命名空间，定义了一个别名 stream_set 使用 flat_hash_set<xpu::XPUStream> 类型
namespace {
using stream_set = ska::flat_hash_set<xpu::XPUStream>;

// 声明 Block 结构体，用于描述内存块信息
struct Block;
typedef bool (*Comparison)(const Block*, const Block*);
bool BlockComparatorSize(const Block* a, const Block* b);

// BlockPool 结构体，用于管理内存块的池子，可以是 "small" 或 "large" 类型
struct BlockPool {
  // 构造函数，根据 small 参数选择比较器，初始化 blocks 为按大小比较的集合
  BlockPool(bool small) : blocks(BlockComparatorSize), is_small(small) {}
  std::set<Block*, Comparison> blocks;  // 使用比较器 Comparison 的块集合
  const bool is_small;  // 标志是否为小块池
};

// Block 结构体，描述一个内存块的详细信息
struct Block {
  DeviceIndex device;  // 设备索引
  sycl::queue* queue{nullptr};  // 分配流的底层队列
  stream_set stream_uses;  // 使用该块的流集合
  size_t size;  // 块的大小（字节）
  size_t requested_size;  // 最初请求的内存大小
  BlockPool* pool{nullptr};  // 拥有该内存池的池
  void* ptr{nullptr};  // 内存地址
  bool allocated{false};  // 使用标志
  Block* prev{nullptr};  // 如果从较大的分配中分割出来，则为前一个块
  Block* next{nullptr};  // 如果从较大的分配中分割出来，则为后一个块
  int event_count{0};  // 未完成的 XPU 事件数

  // 构造函数，初始化块的各个属性
  Block(DeviceIndex device,
        sycl::queue* queue,
        size_t size,
        BlockPool* pool,
        void* ptr)
      : device(device),
        queue(queue),
        stream_uses(),
        size(size),
        requested_size(0),
        pool(pool),
        ptr(ptr) {}

  // 搜索关键字的构造函数
  Block(DeviceIndex device, sycl::queue* queue, size_t size)
      : device(device),
        queue(queue),
        stream_uses(),
        size(size),
        requested_size(0) {}

  // 检查块是否被分割
  bool is_split() const {
    return (prev != nullptr) || (next != nullptr);
  }
};

// 按大小比较 Block 结构体的函数
bool BlockComparatorSize(const Block* a, const Block* b) {
  if (a->queue != b->queue) {
    return reinterpret_cast<uintptr_t>(a->queue) <
        reinterpret_cast<uintptr_t>(b->queue);
  }
  if (a->size != b->size) {
    return a->size < b->size;
  }
  return reinterpret_cast<uintptr_t>(a->ptr) <
      reinterpret_cast<uintptr_t>(b->ptr);
}
// 定义了一个结构体 AllocParams，用于表示分配参数
struct AllocParams {
  // 构造函数，初始化所有成员变量，其中包括搜索键 search_key、内存池 pool、分配大小 alloc_size
  AllocParams(
      DeviceIndex device,
      size_t size,
      sycl::queue* queue,
      BlockPool* pool,
      size_t alloc_size)
      : search_key(device, queue, size),  // 初始化搜索键
        pool(pool),                      // 初始化内存池指针
        alloc_size(alloc_size),          // 初始化分配大小
        block(nullptr) {}                // 初始化块指针为 nullptr

  // 返回设备索引
  DeviceIndex device() const {
    return search_key.device;
  }

  // 返回 SYCL 队列指针
  sycl::queue* queue() const {
    return search_key.queue;
  }

  // 返回分配的大小
  size_t size() const {
    return search_key.size;
  }

  // 搜索键，用于查找块
  Block search_key;
  // 内存池指针，用于分配块
  BlockPool* pool;
  // 分配的大小
  size_t alloc_size;
  // 指向块的指针
  Block* block;
};

} // 匿名命名空间结束

// 设备缓存分配器类
class DeviceCachingAllocator {
 private:
  // 递归互斥锁，用于保护共享资源
  mutable std::recursive_mutex mutex;
  // 大块内存池，用于缓存大于1 MB的未分配块
  BlockPool large_blocks;
  // 小块内存池，用于缓存1 MB或更小的未分配块
  BlockPool small_blocks;
  // 活跃块集合，包含已分配或正在使用的块
  ska::flat_hash_set<Block*> active_blocks;
  // XPU 流事件映射表，将每个流映射到一系列事件及相关块的队列
  ska::flat_hash_map<xpu::XPUStream, std::deque<std::pair<sycl::event, Block*>>>
      xpu_events;
  // 设备索引
  DeviceIndex device_index;

  // 尝试合并块，将 src 块合并到 dst 块中，并更新块池
  size_t try_merge_blocks(Block* dst, Block* src, BlockPool& pool) {
    // 如果 src 为 nullptr，或者已分配，或者有事件在进行中，或者有流在使用，则返回 0
    if (!src || src->allocated || src->event_count > 0 ||
        !src->stream_uses.empty()) {
      return 0;
    }

    // 断言 dst 和 src 均已分割
    TORCH_INTERNAL_ASSERT(dst->is_split() && src->is_split());
    // 如果 src 在 dst 前面，则更新 dst 的指针和前驱指针
    if (dst->prev == src) { // [src dst]
      dst->ptr = src->ptr;
      dst->prev = src->prev;
      if (dst->prev) {
        dst->prev->next = dst;
      }
    } else { // 如果 dst 在 src 前面，则更新 dst 的后继指针和 src 的后继指针的前驱指针
      dst->next = src->next;
      if (dst->next) {
        dst->next->prev = dst;
      }
    }
    // 计算被合并的大小
    const size_t subsumed_size = src->size;
    dst->size += subsumed_size;
    // 从块池中擦除 src 块
    auto erased = pool.blocks.erase(src);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(erased == 1);
    delete src;

    return subsumed_size;
  }

  // 释放块，将不再使用的块从活跃块集合中移除，并尝试合并相邻的块
  void free_block(Block* block) {
    // 断言块未分配、事件计数为 0、无流在使用
    TORCH_INTERNAL_ASSERT(
        !block->allocated && block->event_count == 0 &&
        block->stream_uses.empty());

    // 获取块所属的块池
    auto& pool = *block->pool;
    // 合并块的候选数组，包括块的前驱和后继块
    const std::array<Block*, 2> merge_candidates = {block->prev, block->next};
    for (Block* merge_candidate : merge_candidates) {
      // 尝试合并块
      try_merge_blocks(block, merge_candidate, pool);
    }

    // 从活跃块集合中移除块
    active_blocks.erase(block);
    // 将块插入到块池中
    bool inserted = pool.blocks.insert(block).second;
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(inserted);
  }

  // 处理 XPU 流事件，检查事件是否完成，若完成则减少相关块的事件计数，并释放不再使用的块
  void process_events() {
    using namespace sycl::info;
    for (auto it = xpu_events.begin(); it != xpu_events.end();) {
      while (!it->second.empty()) {
        auto& e = it->second.front();
        auto event = e.first;
        auto* block = e.second;
        // 如果事件未完成，则跳出循环
        if (event.get_info<event::command_execution_status>() !=
            event_command_status::complete) {
          break;
        }
        // 减少块的事件计数
        block->event_count--;
        // 如果事件计数为 0，则释放块
        if (block->event_count == 0) {
          free_block(block);
        }
        // 弹出处理完成的事件
        it->second.pop_front();
      }

      // 如果事件队列为空，则从映射表中移除该条目
      if (it->second.empty()) {
        it = xpu_events.erase(it);
      } else {
        it++;
      }
    }
  }

  // 静态方法，将大小调整为最接近的 2 的幂次方
  static size_t round_size(size_t size) {
    if (size < kMinBlockSize) {
      return kMinBlockSize;
    } else {
      return kMinBlockSize * ((size + kMinBlockSize - 1) / kMinBlockSize);
    }
  }

  static size_t get_allocation_size(size_t size) {
    if (size <= kSmallSize) {
      return kSmallBuffer;
    } else if (size < kMinLargeAlloc) {
      return kLargeBuffer;
    } else {
      return kRoundLarge * ((size + kRoundLarge - 1) / kRoundLarge);
    }
  }

  BlockPool& get_pool(size_t size) {
    if (size < kSmallSize) {
      return small_blocks;
    } else {
      return large_blocks;
    }
  }

  bool get_free_block(AllocParams& p) {
    BlockPool& pool = *p.pool;
    auto it = pool.blocks.lower_bound(&p.search_key);
    if (it == pool.blocks.end() || (*it)->queue != p.queue()) {
      return false;
    }
    p.block = *it;
    pool.blocks.erase(it);
    return true;
  }

  bool alloc_block(AllocParams& p) {
    auto size = p.alloc_size;
    auto device = p.device();
    void* ptr = sycl::aligned_alloc_device(
        kDeviceAlignment,
        size,
        xpu::get_raw_device(device),
        xpu::get_device_context());
    if (!ptr) {
      return false;
    }
    p.block = new Block(device, p.queue(), size, p.pool, ptr);
    return true;
  }

  void synchronize_and_free_events() {
    for (auto& xe : xpu_events) {
      for (auto& e : xe.second) {
        auto event = e.first;
        auto* block = e.second;
        event.wait();
        block->event_count--;
        if (block->event_count == 0) {
          free_block(block);
        }
      }
    }
    xpu_events.clear();
  }

  void release_block(Block* block) {
    /*
     * Note [Safe to Free Blocks on BlockPool]
     *
     * Callers must ensure that all accesses to the block, whose raw pointer is
     * allocated by SYCL APIs, have been completed before invoking sycl::free.
     *
     * We have to do a device-level synchronization before free these blocks to
     * guarantee that all kernels can access to the blocks have finished.
     */
    sycl::free(block->ptr, xpu::get_device_context());
    auto* pool = block->pool;
    pool->blocks.erase(block);
    delete block;
  }

  void release_blocks(BlockPool& pool) {
    auto it = pool.blocks.begin();
    while (it != pool.blocks.end()) {
      Block* block = *it;
      ++it;
      if (!block->prev && !block->next) {
        release_block(block);
      }
    }
  }

  bool release_cached_blocks() {
    synchronize_and_free_events();
    // See Note [Safe to Free Blocks on BlockPool]
    c10::xpu::syncStreamsOnDevice(device_index);

    release_blocks(large_blocks);
    release_blocks(small_blocks);
    return true;
  }

  bool should_split(const Block* block, size_t size) {
    size_t remaining = block->size - size;
    if (block->pool->is_small) {
      return remaining >= kMinBlockSize;
    } else {
      return remaining > kSmallSize;
    }
  }

  Block* alloc_found_block(
      AllocParams params,
      size_t orig_size,
      bool split_remainder) {


注释：


    if (size < kMinBlockSize) {
      // 如果请求的大小小于最小块大小，返回最小块大小
      return kMinBlockSize;
    } else {
      // 否则，返回大于等于请求大小的最小的 kMinBlockSize 的倍数
      return kMinBlockSize * ((size + kMinBlockSize - 1) / kMinBlockSize);
    }
  }

  static size_t get_allocation_size(size_t size) {
    if (size <= kSmallSize) {
      // 如果请求的大小小于等于 kSmallSize，返回 kSmallBuffer
      return kSmallBuffer;
    } else if (size < kMinLargeAlloc) {
      // 如果请求的大小小于 kMinLargeAlloc，返回 kLargeBuffer
      return kLargeBuffer;
    } else {
      // 否则，返回大于等于请求大小的最小的 kRoundLarge 的倍数
      return kRoundLarge * ((size + kRoundLarge - 1) / kRoundLarge);
    }
  }

  BlockPool& get_pool(size_t size) {
    if (size < kSmallSize) {
      // 如果请求的大小小于 kSmallSize，返回 small_blocks 对象的引用
      return small_blocks;
    } else {
      // 否则，返回 large_blocks 对象的引用
      return large_blocks;
    }
  }

  bool get_free_block(AllocParams& p) {
    BlockPool& pool = *p.pool;
    // 在 pool 中寻找大于等于 p.search_key 的块
    auto it = pool.blocks.lower_bound(&p.search_key);
    // 如果找不到或者找到的块的队列与 p.queue() 不匹配，返回 false
    if (it == pool.blocks.end() || (*it)->queue != p.queue()) {
      return false;
    }
    // 将找到的块赋值给 p.block，从 pool 中删除这个块，并返回 true
    p.block = *it;
    pool.blocks.erase(it);
    return true;
  }

  bool alloc_block(AllocParams& p) {
    auto size = p.alloc_size;
    auto device = p.device();
    // 在指定设备上用 SYCL API 分配大小为 size 的内存，返回指针 ptr
    void* ptr = sycl::aligned_alloc_device(
        kDeviceAlignment,
        size,
        xpu::get_raw_device(device),
        xpu::get_device_context());
    // 如果分配失败，返回 false
    if (!ptr) {
      return false;
    }
    // 用分配的内存创建一个新的 Block 对象，并赋值给 p.block
    p.block = new Block(device, p.queue(), size, p.pool, ptr);
    return true;
  }

  void synchronize_and_free_events() {
    // 同步并释放所有 xpu_events 中的事件和关联的块
    for (auto& xe : xpu_events) {
      for (auto& e : xe.second) {
        auto event = e.first;
        auto* block = e.second;
        event.wait();  // 等待事件完成
        block->event_count--;  // 减少关联块的事件计数
        if (block->event_count == 0) {
          free_block(block);  // 如果事件计数为 0，释放块
        }
      }
    }
    xpu_events.clear();  // 清空 xpu_events 容器
  }

  void release_block(Block* block) {
    /*
     * Note [Safe to Free Blocks on BlockPool]
     *
     * 调用者必须确保对块的所有访问（其原始指针由 SYCL API 分配）在调用 sycl::free 之前已完成。
     *
     * 在释放这些块之前，我们必须进行设备级别的同步，以确保所有可以访问到这些块的内核都已经完成。
     */
    sycl::free(block->ptr, xpu::get_device_context());  // 释放块的内存
    auto* pool = block->pool;
    pool->blocks.erase(block);  // 从 pool 中移除块
    delete block;  // 删除块对象
  }

  void release_blocks(BlockPool& pool) {
    auto it = pool.blocks.begin();
    while (it != pool.blocks.end()) {
      Block* block = *it;
      ++it;
      // 如果块没有前驱和后继（即不在链表中），释放这个块
      if (!block->prev && !block->next) {
        release_block(block);
      }
    }
  }

  bool release_cached_blocks() {
    synchronize_and_free_events();  // 同步并释放所有事件
    // See Note [Safe to Free Blocks on BlockPool]
    c10::xpu::syncStreamsOnDevice(device_index);  // 在设备上同步流

    release_blocks(large_blocks);  // 释放大块链表中的所有块
    release_blocks(small_blocks);  // 释放小块链表中的所有块
    return true;
  }

  bool should_split(const Block* block, size_t size) {
    size_t remaining = block->size - size;
    if (block->pool->is_small) {
      // 如果块来自小块池，并且剩余大小大于等于 kMinBlockSize，返回 true
      return remaining >= kMinBlockSize;
    } else {
      // 如果块来自大块池，并且剩余大小大于 kSmall
    // 获取参数 params 的大小
    auto size = params.size();
    // 获取参数 params 的设备
    auto device = params.device();
    // 获取参数 params 的内存块池
    BlockPool* pool = params.pool;
    // 获取参数 params 的队列
    sycl::queue* queue = params.queue();

    // 断言参数 params 的内存块不为空且指针不为空
    TORCH_INTERNAL_ASSERT(
        params.block != nullptr && params.block->ptr != nullptr);
    // 获取参数 params 的内存块
    Block* block = params.block;
    // 初始化 remaining 为空指针
    Block* remaining = nullptr;

    // 如果需要拆分剩余部分
    if (split_remainder) {
      // 将 remaining 指向当前内存块
      remaining = block;

      // 创建一个新的内存块 block，连接到原内存块之前
      block = new Block(device, queue, size, pool, block->ptr);
      block->prev = remaining->prev;
      if (block->prev) {
        block->prev->next = block;
      }
      block->next = remaining;

      // 更新 remaining 的 prev 指针，更新 remaining 的指针位置和大小
      remaining->prev = block;
      remaining->ptr = static_cast<char*>(remaining->ptr) + size;
      remaining->size -= size;
      // 将 remaining 插入到内存块池中
      bool inserted = pool->blocks.insert(remaining).second;
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(inserted);
    }

    // 设置内存块的 allocated 标志为 true，设置请求的大小为原始大小
    block->allocated = true;
    block->requested_size = orig_size;
    // 将内存块插入到活动内存块集合中
    bool inserted = active_blocks.insert(block).second;
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(inserted)

    // 返回内存块
    return block;
  }

  // 插入事件到内存块中
  void insert_events(Block* block) {
    // 移动内存块的 stream 使用到 streams 中
    stream_set streams(std::move(block->stream_uses));
    TORCH_INTERNAL_ASSERT(block->stream_uses.empty());
    // 遍历 streams 中的 stream，增加事件计数并插入到 xpu_events 中
    for (auto& stream : streams) {
      block->event_count++;
      xpu_events[stream].emplace_back(
          stream.queue().ext_oneapi_submit_barrier(), block);
    }
  }

 public:
  // 构造函数，初始化大内存块和小内存块的标志，设备索引
  DeviceCachingAllocator(DeviceIndex device_index)
      : large_blocks(/* small */ false),
        small_blocks(/* small */ true),
        device_index(device_index) {}

  // 分配内存块
  Block* malloc(DeviceIndex device, size_t orig_size, sycl::queue& queue) {
    // 加锁
    std::scoped_lock<std::recursive_mutex> lock(mutex);
    // 处理事件
    process_events();
    // 对原始大小进行取整
    size_t size = round_size(orig_size);
    // 获取对应大小的内存块池
    auto& pool = get_pool(size);
    // 获取分配大小
    const size_t alloc_size = get_allocation_size(size);
    // 初始化参数
    AllocParams params(device, size, &queue, &pool, alloc_size);

    // 首先尝试从现有池中获取内存块
    bool block_found = get_free_block(params);
    // 无法重用现有内存块，尝试获取新的内存块
    if (!block_found) {
      block_found = alloc_block(params) ||
          (release_cached_blocks() && alloc_block(params));
    }
    // 检查是否找到内存块，否则抛出异常
    TORCH_CHECK(
        block_found,
        "XPU out of memory, please use `empty_cache` to release all unoccupied cached memory.");
    // 检查是否需要拆分内存块
    bool split_remainder = should_split(params.block, params.size());
    // 返回找到的内存块
    return alloc_found_block(std::move(params), orig_size, split_remainder);
  }

  // 释放内存块
  void free(Block* block) {
    // 加锁
    std::scoped_lock<std::recursive_mutex> lock(mutex);
    // 将内存块的 allocated 标志设置为 false
    block->allocated = false;

    // 如果内存块的 stream 使用不为空，则插入事件
    if (!block->stream_uses.empty()) {
      insert_events(block);
    } else {
      // 否则释放内存块
      free_block(block);
    }
  }

  // 记录流使用到内存块中
  void recordStream(Block* block, xpu::XPUStream stream) {
    // 加锁
    std::scoped_lock<std::recursive_mutex> lock(mutex);
    // 如果流队列与内存块队列相同，则返回
    if (stream.queue() == *block->queue) {
      return;
    }
    // 将流使用插入到内存块中
    block->stream_uses.insert(stream);
  }

  // 清空缓存
  void emptyCache() {
    // 加锁
    std::scoped_lock<std::recursive_mutex> lock(mutex);
    // 释放缓存的内存块
    release_cached_blocks();
  }
};

void local_raw_delete(void* ptr);

class XPUAllocator : public Allocator {
 private:
  // 互斥锁，用于保护分配的块映射表
  std::mutex mutex;
  // 分配的块映射表，将指针映射到块对象
  ska::flat_hash_map<void*, Block*> allocated_blocks;

  // 添加分配的块到映射表中
  void add_allocated_block(Block* block) {
    std::lock_guard<std::mutex> lock(mutex);
    allocated_blocks[block->ptr] = block;
  }

  // 根据指针获取分配的块对象，并可选择是否从映射表中移除
  Block* get_allocated_block(void* ptr, bool remove = false) {
    std::scoped_lock<std::mutex> lock(mutex);
    auto it = allocated_blocks.find(ptr);
    if (it == allocated_blocks.end()) {
      return nullptr;
    }
    Block* block = it->second;
    if (remove) {
      allocated_blocks.erase(it);
    }
    return block;
  }

 public:
  // 设备缓存分配器的列表
  std::vector<std::unique_ptr<DeviceCachingAllocator>> device_allocators;

  // 初始化函数，根据设备数量调整设备缓存分配器列表
  void init(DeviceIndex device_count) {
    const auto size = static_cast<DeviceIndex>(device_allocators.size());
    if (size < device_count) {
      device_allocators.resize(device_count);
      for (const auto i : c10::irange(size, device_count)) {
        device_allocators[i] = std::make_unique<DeviceCachingAllocator>(i);
      }
    }
  }

  // 分配内存函数，将分配的块添加到映射表中，并记录 GPU 内存分配的追踪信息
  void malloc(
      void** devPtr,
      DeviceIndex device,
      size_t size,
      sycl::queue& queue) {
    TORCH_INTERNAL_ASSERT(
        0 <= device && static_cast<size_t>(device) < device_allocators.size(),
        "Allocator not initialized for device ",
        static_cast<int16_t>(device),
        ": did you call init?");
    Block* block = device_allocators[device]->malloc(device, size, queue);
    add_allocated_block(block);
    *devPtr = block->ptr;
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_memory_allocation(
          c10::kXPU, reinterpret_cast<uintptr_t>(*devPtr));
    }
  }

  // 释放内存函数，释放分配的块，并记录 GPU 内存释放的追踪信息
  void free(void* ptr) {
    if (!ptr) {
      return;
    }
    Block* block = get_allocated_block(ptr, /* remove */ true);
    TORCH_CHECK(block, "invalid device pointer: ", ptr);
    device_allocators[block->device]->free(block);
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_memory_deallocation(
          c10::kXPU, reinterpret_cast<uintptr_t>(block->ptr));
    }
  }

  // 清空缓存函数，遍历所有设备缓存分配器并清空缓存
  void emptyCache() {
    for (auto& da : device_allocators) {
      da->emptyCache();
    }
  }

  // 记录流函数，根据指针查找分配的块，并记录流信息
  void recordStream(const DataPtr& ptr, XPUStream stream) {
    if (!ptr.get()) {
      return;
    }
    if (ptr.get_deleter() != &local_raw_delete) {
      return;
    }

    Block* block = get_allocated_block(ptr.get());
    TORCH_CHECK(block, "No allocated block can be found.");
    device_allocators[block->device]->recordStream(block, stream);
  }

  // 分配内存的接口实现，根据当前设备分配内存
  DataPtr allocate(size_t size) override {
    auto device = c10::xpu::current_device();
    void* r = nullptr;
    if (size != 0) {
      this->malloc(&r, device, size, xpu::getCurrentXPUStream(device));
    }
  // 返回一个包含指定设备类型和设备对象的 Device 对象，并注册 local_raw_delete 作为其释放函数
  return {r, r, &local_raw_delete, Device(DeviceType::XPU, device)};
}

// 返回 local_raw_delete 函数的指针作为 raw_deleter 的实现
DeleterFnPtr raw_deleter() const override {
  return &local_raw_delete;
}

// 使用当前 XPU 设备和流分配指定大小的内存，并返回指向分配内存的指针 r
void* raw_alloc(size_t size) {
  if (size == 0) {
    return nullptr;
  }
  auto device = c10::xpu::current_device();
  void* r = nullptr;
  malloc(&r, device, size, xpu::getCurrentXPUStream(device));
  return r;
}

// 使用指定的 XPU 流和当前设备分配指定大小的内存，并返回指向分配内存的指针 r
void* raw_alloc_with_stream(size_t size, XPUStream stream) {
  if (size == 0) {
    return nullptr;
  }
  auto device = c10::xpu::current_device();
  void* r = nullptr;
  malloc(&r, device, size, stream);
  return r;
}

// 调用自身的 free 方法释放给定的指针 ptr
void raw_delete(void* ptr) {
  this->free(ptr);
}

// 使用当前 XPU 流复制从 src 指针指向的数据到 dest 指针指向的位置，复制的字节数为 count
void copy_data(void* dest, const void* src, std::size_t count) const final {
  xpu::getCurrentXPUStream().queue().memcpy(dest, src, count);
}
};

static XPUAllocator allocator;

// 定义一个静态的XPUAllocator对象，用于内存分配

void local_raw_delete(void* ptr) {
  // 调用allocator对象的free方法释放ptr指向的内存
  allocator.free(ptr);
}

// 返回指向allocator对象的指针
Allocator* get() {
  return &allocator;
}

// 调用allocator对象的init方法，初始化内存分配器，参数为设备数量
void init(DeviceIndex device_count) {
  return allocator.init(device_count);
}

// 调用allocator对象的emptyCache方法，清空内存缓存
void emptyCache() {
  return allocator.emptyCache();
}

// 调用allocator对象的raw_alloc方法，分配指定大小的内存并返回其指针
void* raw_alloc(size_t size) {
  return allocator.raw_alloc(size);
}

// 调用allocator对象的raw_delete方法，释放ptr指向的内存
void raw_delete(void* ptr) {
  return allocator.raw_delete(ptr);
}

// 调用allocator对象的recordStream方法，记录数据指针dataPtr在流stream上的操作
void recordStream(const DataPtr& dataPtr, XPUStream stream) {
  return allocator.recordStream(dataPtr, stream);
}

// 注册XPUAllocator对象为XPUCachingAllocator命名空间下的分配器
REGISTER_ALLOCATOR(kXPU, &allocator)

} // namespace c10::xpu::XPUCachingAllocator
```