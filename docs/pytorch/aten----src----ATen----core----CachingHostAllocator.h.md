# `.\pytorch\aten\src\ATen\core\CachingHostAllocator.h`

```py
// 包含C10核心库中的Allocator.h，提供内存分配相关功能
#include <c10/core/Allocator.h>
// 包含C10工具库中的Optional.h，提供可选值类型的支持
#include <c10/util/Optional.h>
// 包含C10工具库中的flat_hash_map.h，提供快速哈希映射的实现
#include <c10/util/flat_hash_map.h>
// 包含C10工具库中的llvmMathExtras.h，提供数学计算相关的额外函数
#include <c10/util/llvmMathExtras.h>

// 包含STL库中的deque头文件，提供双端队列容器
#include <deque>
// 包含STL库中的mutex头文件，提供互斥锁支持
#include <mutex>
// 包含STL库中的set头文件，提供集合容器
#include <set>

// 定义AT命名空间，包含了本文件中的所有类型和函数
namespace at {

/**
 * HostBlock通常是用于固定内存中的基础内存块。它可能与设备运行时的Event和Stream有关。
 * 可能是一个基础结构或接口，每个后端都可以继承和扩展它。
 */
template <typename S>
struct HostBlock {
  // 构造函数，用于按大小搜索键
  HostBlock(size_t size) : size_(size) {}

  // 带指针参数的构造函数
  HostBlock(size_t size, void* ptr) : size_(size), ptr_(ptr) {}

  std::mutex mutex_; // 互斥锁，用于保护内部数据访问
  size_t size_{0}; // 内存块的大小，单位字节
  void* ptr_{nullptr}; // 内存块的地址
  bool allocated_{false}; // 是否正在使用的标志
  size_t event_count_{0}; // 相关事件的数量
  ska::flat_hash_set<S> streams_; // 使用该内存块的流集合
};

/**
 * ComparatorSize用于按块大小在主机内存块集合中进行查找支持。
 */
template <typename B>
struct ComparatorSize {
  bool operator()(const B* a, const B* b) const {
    // 按大小比较两个HostBlock对象，返回比较结果
    if (a->size_ != b->size_) {
      return a->size_ < b->size_;
    }
    // 如果大小相同，则按地址比较两个HostBlock对象
    return (uintptr_t)a->ptr_ < (uintptr_t)b->ptr_;
  }
};

/**
 * CachingHostAllocatorImpl是主机内存分配器的缓存实现。
 * S表示流的类型，E表示事件的类型，B表示内存块的类型，C表示比较器类型。
 */
template <
    typename S,
    typename E,
    typename B = HostBlock<S>,
    typename C = ComparatorSize<B>>
struct CachingHostAllocatorImpl {
  virtual ~CachingHostAllocatorImpl() = default;

 public:
  // 分配函数，返回数据指针和内存块对
  virtual std::pair<void*, void*> allocate(size_t size) {
    // 如果请求大小为0，返回空指针对
    if (size == 0) {
      return {nullptr, nullptr};
    }

    process_events(); // 处理事件

    // 首先，尝试从空闲列表中分配
    auto* block = get_free_block(size);
    if (block) {
      return {block->ptr_, reinterpret_cast<void*>(block)};
    }

    // 将分配大小向上舍入到最接近的2的幂，以提高重用性
    size_t roundSize = c10::llvm::PowerOf2Ceil(size);
    void* ptr = nullptr;
    allocate_host_memory(roundSize, &ptr);

    // 创建一个新的内存块
    block = new B(roundSize, ptr);
    block->allocated_ = true;

    add_allocated_block(block); // 添加到已分配块集合中
    return {block->ptr_, reinterpret_cast<void*>(block)};
  }

  // 释放函数，接收上下文指针参数
  virtual void free(void* ctx) {
    if (!ctx) {
      return;
    }

    // 注意：我们可以假设free正确地与alloc配对，因此不需要在blocks_中查找ctx。
    auto* block = reinterpret_cast<B*>(ctx);

    std::optional<std::vector<E>> events;
    {
      std::lock_guard<std::mutex> g(block->mutex_); // 使用互斥锁保护区域
      block->allocated_ = false; // 标记为未使用
      if (block->streams_.empty()) {
        TORCH_INTERNAL_ASSERT(block->event_count_ == 0); // 断言流集合为空时事件计数为0
      } else {
        events = std::vector<E>(); // 创建事件向量
        events->reserve(block->streams_.size()); // 预留空间
        for (auto stream : block->streams_) {
          record_stream(events, stream); // 记录每个流的事件
        }
        block->event_count_ += events->size(); // 更新事件计数
        block->streams_.clear(); // 清空流集合
      }
    }
    if (!events) {
      // 如果事件列表为空，则将块插入到空闲列表中
      std::lock_guard<std::mutex> g(free_list_mutex_);
      free_list_.insert(block);
    } else {
      // 否则，将已记录的事件恢复到使用的流中
      std::lock_guard<std::mutex> g(events_mutex_);
      for (auto&& event : *events) {
        events_.emplace_front(std::move(event), block);
      }
    }
  }

  virtual bool record_event(void* ptr, void* ctx, S stream) {
    auto* block = reinterpret_cast<B*>(ctx);

    // 注意：我们需要检查传入的 `ctx` 是否有效。这是因为 `record_event`（通过 `CachingHostAllocator_recordEvent`）可能会在任意张量上调用，
    // 并且不能保证对应的是固定内存分配。因此，在继续之前，我们需要检查 `ctx` 是否有效。
    {
      std::lock_guard<std::mutex> g(blocks_mutex_);
      if (blocks_.find(block) != blocks_.end()) {
        // 现在我们知道可以安全访问这个对象了。
        std::lock_guard<std::mutex> gb(block->mutex_);
        TORCH_INTERNAL_ASSERT(block->allocated_);
        block->streams_.insert(stream);
        return true;
      }
      auto it = ptr_to_block_.find(ptr);
      if (it != ptr_to_block_.end()) {
        block = it->second;
        std::lock_guard<std::mutex> g(block->mutex_);
        TORCH_INTERNAL_ASSERT(block->allocated_);
        block->streams_.insert(stream);
        return true;
      }
    }

    return false;
  }

  virtual void empty_cache() {
    // 将任何可用的块刷新到空闲列表中。
    process_events();

    // 从空闲列表中移除所有元素，从块列表中移除它们，并释放相关的固定内存分配。
    // 这需要同时持有空闲列表互斥锁和块互斥锁，并且是唯一一个同时持有多个互斥锁的函数。
    std::lock(free_list_mutex_, blocks_mutex_);
    std::lock_guard<std::mutex> gf(free_list_mutex_, std::adopt_lock);
    std::lock_guard<std::mutex> gb(blocks_mutex_, std::adopt_lock);

    std::vector<B*> blocks_to_remove(free_list_.begin(), free_list_.end());
    free_list_.clear();
    for (auto* block : blocks_to_remove) {
      blocks_.erase(block);
      ptr_to_block_.erase(block->ptr_);
      free_block(block);
      delete block;
    }
  }

  virtual void copy_data(void* dest, const void* src, std::size_t count) const {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "Not implemented for copy_data");
  }

 private:
  virtual void add_allocated_block(B* block) {
    // 添加已分配的块到块列表中
    std::lock_guard<std::mutex> g(blocks_mutex_);
    blocks_.insert(block);
    ptr_to_block_.insert({block->ptr_, block});
  }

  virtual B* get_free_block(size_t size) {
    // 获取指定大小的空闲块
    std::lock_guard<std::mutex> g(free_list_mutex_);
    B key(size);
    auto it = free_list_.lower_bound(&key);
    if (it != free_list_.end()) {
      B* block = *it;
      block->allocated_ = true;
      free_list_.erase(it);
      return block;
    }
    return nullptr;
  }

  virtual void process_events() {

    while (true) {
      // 避免在持有互斥锁时调用 cudaEventDestroy，因此将中间事件移出锁，放入此对象中处理最后的事件

      // 处理最后一个事件
      std::optional<std::pair<E, B*>> processed;
      {
        std::lock_guard<std::mutex> g(events_mutex_);
        if (!events_.empty()) {
          processed = std::move(events_.back());
          events_.pop_back();
        }
      }

      if (!processed) {
        // 如果没有处理的事件了，退出函数
        return;
      }

      // 否则，查询事件
      {
        // 现在，看看是否可以处理这个事件
        auto& event = processed->first;
        if (!query_event(event)) {
          // 如果事件还没准备好，将事件推回队列的末尾
          {
            std::lock_guard<std::mutex> g(events_mutex_);
            events_.push_back(std::move(*processed));
          }
          return;
        }
      }

      // 处理事件
      TORCH_INTERNAL_ASSERT(processed);
      auto* block = processed->second;
      bool available = false;
      {
        std::lock_guard<std::mutex> g(block->mutex_);
        TORCH_INTERNAL_ASSERT(!block->allocated_)
        block->event_count_--;
        if (block->event_count_ == 0) {
          available = true;
        }
      }

      if (available) {
        // 如果事件可用，将块插入到空闲列表中
        std::lock_guard<std::mutex> g(free_list_mutex_);
        free_list_.insert(block);
      }
    }
  }

  /* These following functions are runtime-related. */

  // 在主机上分配页锁定内存。
  virtual void allocate_host_memory(size_t size, void** ptr) {
    TORCH_CHECK_NOT_IMPLEMENTED(
        false, "Not implemented for allocate_host_memory");
  }

  // 释放块并释放块中包含的指针。
  virtual void free_block(B* block) {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "Not implemented for free_block");
  }

  // 记录流上的事件，并将事件存储到事件列表中。
  virtual void record_stream(std::optional<std::vector<E>>& events, S stream) {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "Not implemented for record_stream");
  }

  // 查询事件是否已完成。
  virtual bool query_event(E& event) {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "Not implemented for query_event");
  }

  alignas(64) std::mutex blocks_mutex_;
  ska::flat_hash_set<B*> blocks_; // 块列表
  ska::flat_hash_map<void*, B*> ptr_to_block_;

  // 注意：在高度多线程场景中，对此互斥锁进行分片可能会带来收益。
  alignas(64) std::mutex free_list_mutex_;
  // 注意：在微基准测试中，替代数据结构可能会带来显著优势。
  std::set<B*, C> free_list_; // 空闲列表

  alignas(64) std::mutex events_mutex_;
  std::deque<std::pair<E, B*>> events_; // 事件队列，与块配对
};

// 结构体模板 CachingHostAllocatorInterface 继承自 at::Allocator，用于缓存主机内存分配器接口
template <typename T>
struct CachingHostAllocatorInterface : public at::Allocator {
  // 默认构造函数，创建一个 T 类型的唯一指针实现
  CachingHostAllocatorInterface() :impl_(std::make_unique<T>()) {}

  // 覆盖基类方法 allocate，抛出错误提示信息
  at::DataPtr allocate(size_t size) override {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "Not implemented for allocate");
  }

  // 自定义方法 free，调用 impl_ 的 free 方法释放内存
  void free(void* ctx) {
    impl_->free(ctx);
  }

  // 模板方法 record_event，调用 impl_ 的 record_event 方法记录事件
  template <typename S>
  bool record_event(void* ptr, void* ctx, S stream) {
    return impl_->record_event(ptr, ctx, stream);
  }

  // 方法 empty_cache，调用 impl_ 的 empty_cache 方法清空缓存
  void empty_cache() {
    impl_->empty_cache();
  }

  // 覆盖基类方法 copy_data，调用 impl_ 的 copy_data 方法复制数据
  void copy_data(void* dest, const void* src, std::size_t count)
      const override {
    impl_->copy_data(dest, src, count);
  }

  // 唯一指针实现的成员变量 impl_
  std::unique_ptr<T> impl_;
};

// 命名空间 at 结束标记
} // namespace at
```