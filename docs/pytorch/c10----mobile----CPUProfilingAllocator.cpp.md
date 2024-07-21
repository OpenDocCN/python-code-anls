# `.\pytorch\c10\mobile\CPUProfilingAllocator.cpp`

```
namespace c10 {

namespace {
// 线程本地变量，用于分配规划器和 CPU 分析分配器的指针
thread_local AllocationPlanner* allocation_planner{nullptr};
thread_local CPUProfilingAllocator* profiling_allocator{nullptr};

// 内存块结构体，表示内存块的起始和结束偏移量
struct MemBlock {
  uint64_t start_offset, end_offset;
  MemBlock(uint64_t s, uint64_t e) : start_offset(s), end_offset(e) {}
  // 比较运算符重载，用于比较内存块的起始偏移量
  bool operator<(const MemBlock& other) const {
    return start_offset < other.start_offset;
  }
};

// 内存事件类型枚举，表示分配、释放或无效事件
enum class EventType { Allocate = 0, Free, Invalid };

// 内存事件结构体，记录时间戳、分配 ID、大小和事件类型
struct MemEvent {
  uint64_t time;
  uint64_t allocation_id;
  uint64_t size;
  EventType type{EventType::Invalid};
  MemEvent(uint64_t t, uint64_t id, uint64_t s, EventType e)
      : time(t), allocation_id(id), size(s), type(e) {}
};

// 判断两个内存块是否重叠的函数
bool overlaps(const MemBlock& a, const MemBlock& b) {
  // 两个块不重叠条件为：
  // |---a--------|--------------b--------|
  // start_a     end_a <= start_b       end_b
  return !(
      (a.end_offset <= b.start_offset) || (b.end_offset <= a.start_offset));
}

// 验证分配计划的函数
bool validate_allocation_plan(
    const std::vector<MemEvent>& alloc_events,
    const std::vector<uint64_t>& allocation_offsets) {
  // 使用 set 来保存当前分配的内存块
  std::set<MemBlock> allocations;
  for (const auto& event : alloc_events) {
    auto alloc_id = event.allocation_id;
    // 跳过非 AllocationPlan 管理的分配
    if (allocation_offsets[alloc_id] == std::numeric_limits<uint64_t>::max()) {
      continue;
    }
    auto start_offset = allocation_offsets[alloc_id];
    auto end_offset = allocation_offsets[alloc_id] + event.size;
    MemBlock mem_block(start_offset, end_offset);
    if (event.type == EventType::Allocate) {
      // 查找并检查新分配的内存块是否与已有的内存块重叠
      auto it = allocations.lower_bound(mem_block);
      if (it != allocations.end()) {
        auto next_block = *it;
        if (overlaps(next_block, mem_block)) {
          return false;
        }
      }
      if (it != allocations.begin()) {
        auto prev_block = *(--it);
        if (overlaps(prev_block, mem_block)) {
          return false;
        }
      }
      // 将新分配的内存块加入集合中
      allocations.emplace(mem_block);
    } else if (event.type == EventType::Free) {
      // 查找并删除要释放的内存块
      auto it = allocations.find(mem_block);
      TORCH_CHECK(
          (*it).end_offset == end_offset,
          "Enf offset of allocation being freed must match the one recorded.");
      TORCH_CHECK(
          it != allocations.end(),
          "ProfilingAllocator: Allocate event "
          "must have preceded deallocate event.");
      allocations.erase(it);
    } else {
      // 对于无效事件类型，抛出异常
      TORCH_CHECK(false, "ProfilingAllocator: Invalid event type.");
    }
  }
  return true;
}

// 创建并排序内存事件的函数
std::vector<MemEvent> create_and_sort_mem_events(
    const std::vector<uint64_t>& allocation_sizes,
    const std::vector<uint64_t>& allocation_lifetimes) {
  std::vector<MemEvent> events;
  for (uint64_t i = 0; i < allocation_sizes.size(); ++i) {
    // 如果观察到的分配在生命周期之外被释放
    // 如果allocation_lifetimes[i]的值为uint64_t的最大值，表示这次分配不受AllocationPlan管理，跳过处理
    if (allocation_lifetimes[i] == std::numeric_limits<uint64_t>::max()) {
      continue;
    }
    // 将内存事件加入事件列表，表示分配操作
    events.emplace_back(i, i, allocation_sizes[i], EventType::Allocate);
    // 将内存事件加入事件列表，表示释放操作，使用allocation_lifetimes[i]作为释放时间
    events.emplace_back(
        allocation_lifetimes[i], i, allocation_sizes[i], EventType::Free);
  }
  // 对事件列表按时间排序，以便后续按时间顺序处理内存操作
  std::sort(
      events.begin(),
      events.end(),
      [](const MemEvent& a, const MemEvent& b) -> bool {
        return a.time < b.time;
      });
  // 返回排序后的事件列表，包含所有分配和释放操作
  return events;
  // Step 1. Construct all allocation/free events.
  //         Sort these events by timestamp.
  std::vector<uint64_t> formulate_greedy_allocation_plan(
      const std::vector<uint64_t>& allocation_sizes,
      const std::vector<uint64_t>& allocation_lifetimes) {
    // Step 2. Iterate through all events.
    // Initialize data structures for managing free memory chunks.

    // lower_bound on this map will get all candidates of
    // the right size for allocation.
    std::map<uint64_t, uint64_t> free_size_to_offset;

    // This provides fast lookup when we want to insert freed block
    // back, especially when we want to merge blocks.
    ska::flat_hash_map<uint64_t, std::map<uint64_t, uint64_t>::iterator>
        free_start_offset_to_size_iter;
    ska::flat_hash_map<uint64_t, std::map<uint64_t, uint64_t>::iterator>
        free_end_offset_to_size_iter;

    // Upon free end_ptr = offset + size
    // If end_ptr exists merge freed allocation
    // Also find corresponding offset in size_to_offset
    // Remove that entry and update with new size and offset
    // If end_ptr does not exist then just insert offset,size
    // in map and correspondingly size, offset in the other map.
    // Merging should always be done recursively until no more chunks
    // that can be found.
    // After last free we should have only one entry left in these maps.

    // Allocate memory for storing the offsets of allocations.
    std::vector<uint64_t> allocation_offsets(
        allocation_sizes.size(), std::numeric_limits<uint64_t>::max());

    // Create and sort memory events based on allocation sizes and lifetimes.
    auto mem_events =
        create_and_sort_mem_events(allocation_sizes, allocation_lifetimes);

    // Track the maximum offset encountered during allocations.
    uint64_t max_offset{0};

    // Iterate through sorted memory events.
    for (const auto& mem_event : mem_events) {
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      uint64_t alloc_offset;
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      uint64_t new_offset, new_size;
    // 如果内存事件的类型是分配事件
    if (mem_event.type == EventType::Allocate) {
      // 查找第一个大小不小于所需大小的空闲块
      auto it = free_size_to_offset.lower_bound(mem_event.size);
      // 如果没有找到足够大的连续空闲块
      if (it == free_size_to_offset.end()) {
        // 分配一个新的空间，起始位置为当前最大偏移量
        alloc_offset = max_offset;
        // 更新最大偏移量为分配后的偏移量
        max_offset += mem_event.size;
      } else {
        // 如果找到了符合大小要求的空闲块
        // 1. 通过从中分配来修改该块
        //    1.1 删除整个空闲块
        //    1.2 删除反向映射条目
        // 2. 如果块还有剩余空间，将剩余部分重新插入到映射中
        //    同时重新插入反向映射条目
        alloc_offset = it->second;
        new_offset = alloc_offset + mem_event.size;
        new_size = it->first - mem_event.size;
        free_size_to_offset.erase(it);
        free_start_offset_to_size_iter.erase(alloc_offset);
        free_end_offset_to_size_iter.erase(alloc_offset + it->first);
        if (new_size > 0) {
          // 将剩余空间插入空闲块映射中
          auto ref_it = free_size_to_offset.emplace(new_size, new_offset).first;
          free_start_offset_to_size_iter.emplace(new_offset, ref_it);
          free_end_offset_to_size_iter.emplace(new_offset + new_size, ref_it);
        }
      }
      // 记录分配的偏移量到分配 ID 的映射
      allocation_offsets[mem_event.allocation_id] = alloc_offset;
    } else {
      // 1. 检查释放的块是否与其结束边界处的现有空闲块相邻。这通过检查
      //    free_end_offset_to_size_iter 来完成。
      //    如果找到这样的块，则移除它并调整被释放块的大小。
      // 2. 类似地，检查释放的块是否与其开始边界处的现有空闲块相邻。这通过检查
      //    free_start_offset_to_size_iter 来完成。
      //    如果找到这样的块，则移除它并调整被释放块的大小。
      // 3. 将释放的块插入映射中。
      auto freed_offset = allocation_offsets[mem_event.allocation_id];
      auto freed_size = mem_event.size;
      auto end_offset = freed_offset + freed_size;

      // 当另一个空闲块存在于此块的末尾时进行合并
      auto end_it = free_start_offset_to_size_iter.find(end_offset);
      if (end_it != free_start_offset_to_size_iter.end()) {
        auto merge_block_iter = end_it->second;
        auto merge_block_size = merge_block_iter->first;
        freed_size += merge_block_size;
        free_size_to_offset.erase(merge_block_iter);
        free_start_offset_to_size_iter.erase(end_it);
        // 如果块正在合并，则还要从 free_end_offset_to_size_iter 中移除
        free_end_offset_to_size_iter.erase(end_offset + merge_block_size);
      }

      // 当释放块存在于另一个空闲块的末尾时进行合并
      auto start_it = free_end_offset_to_size_iter.find(freed_offset);
      if (start_it != free_end_offset_to_size_iter.end()) {
        auto merge_block_iter = start_it->second;
        auto merge_block_size = merge_block_iter->first;
        freed_size += merge_block_size;
        freed_offset -= merge_block_size;
        free_size_to_offset.erase(merge_block_iter);
        free_end_offset_to_size_iter.erase(start_it);
        // 如果块正在合并，则还要从 free_start_offset_to_size_iter 中移除
        free_start_offset_to_size_iter.erase(freed_offset);
      }

      // 将释放的块插入到 free_size_to_offset 映射中
      auto freed_block_it =
          free_size_to_offset.emplace(freed_size, freed_offset).first;
      free_start_offset_to_size_iter.emplace(freed_offset, freed_block_it);
      free_end_offset_to_size_iter.emplace(
          freed_offset + freed_size, freed_block_it);
    }
  }
  // 检查分配计划是否有效，若无效则抛出异常
  TORCH_CHECK(
      validate_allocation_plan(mem_events, allocation_offsets),
      "ProfilingAllocator: Allocation plan invalid.");
  // 返回更新后的分配偏移量映射
  return allocation_offsets;
} // namespace



void AllocationPlan::clear() {
  // 清空分配计划中的所有数据结构
  allocation_sizes.clear();
  allocation_lifetimes.clear();
  allocation_offsets.clear();
}

void AllocationPlanner::record_allocation(
    const uint64_t size,
    const void* ptr) {
  // 如果处于验证模式，则验证分配操作
  if (validation_mode_) {
    // 更新验证成功状态，并验证分配操作
    validation_success = validation_success && validate_allocation(size, ptr);
    return;
  }
  // 将分配的大小记录到分配计划中
  allocation_plan_->allocation_sizes.push_back(size);
  // 设置分配的生命周期为最大值
  allocation_plan_->allocation_lifetimes.push_back(
      std::numeric_limits<uint64_t>::max());
  // 记录分配指针与其对应的 ID
  allocation_ptr_to_id_[ptr] = allocation_id_;
  // 增加分配的 ID
  allocation_id_++;
}

void AllocationPlanner::record_free(const void* ptr) {
  // 如果处于验证模式，则验证释放操作
  if (validation_mode_) {
    // 更新验证成功状态，并验证释放操作
    validation_success = validation_success && validate_free(ptr);
    return;
  }
  // 查找要释放的指针在分配计划中的 ID
  auto it = allocation_ptr_to_id_.find(ptr);
  // 如果未找到，说明要释放的对象在记录之外，直接返回
  if (it == allocation_ptr_to_id_.end()) {
    // 在 WithProfileAllocationGuard 之外分配的对象释放记录
    return;
  }
  // 获取分配 ID
  auto id = it->second;
  // 检查分配 ID 是否有效
  TORCH_CHECK(
      id < allocation_plan_->allocation_lifetimes.size(),
      "Allocation must have been recorded during record_allocation.");
  // 更新释放对象的生命周期 ID
  allocation_plan_->allocation_lifetimes[id] = allocation_id_;
}

bool AllocationPlanner::validate_allocation(
    const uint64_t size,
    const void* ptr) {
  // 如果分配 ID 超出大小或者分配大小不匹配，则发出警告
  if (allocation_id_ >= allocation_plan_->allocation_sizes.size() ||
      allocation_plan_->allocation_sizes[allocation_id_] != size) {
    TORCH_WARN(
        "Allocation request does not match plan:",
        "Allocation id:",
        allocation_id_,
        ", Number of recorded allocations:",
        allocation_plan_->allocation_sizes.size(),
        ", Recorded size of the requested allocation:",
        allocation_plan_->allocation_sizes[allocation_id_],
        ", but got:",
        size);

    return false;
  }
  // 记录分配指针与其对应的 ID
  allocation_ptr_to_id_[ptr] = allocation_id_;
  // 增加分配 ID
  allocation_id_++;
  return true;
}

bool AllocationPlanner::validate_free(const void* ptr) {
  // 查找要释放的指针在分配计划中的 ID
  auto it = allocation_ptr_to_id_.find(ptr);
  // 如果未找到，说明要释放的对象在记录之外，认为验证通过
  if (it == allocation_ptr_to_id_.end()) {
    // 在验证范围之外分配的对象释放验证通过
    return true;
  }
  // 获取分配 ID
  auto id = (*it).second;
  // 检查分配 ID 是否有效
  TORCH_CHECK(
      id < allocation_plan_->allocation_lifetimes.size(),
      "Allocation must have been recorded during validate_allocation.");
  // 获取对象生命周期 ID
  auto lifetime_id = allocation_plan_->allocation_lifetimes[id];
  // 返回释放对象的生命周期 ID 是否等于当前分配 ID
  return (lifetime_id == allocation_id_);
}

void AllocationPlanner::formulate_plan() {
  // 使用贪婪算法制定分配计划的偏移量
  allocation_plan_->allocation_offsets = formulate_greedy_allocation_plan(
      allocation_plan_->allocation_sizes,
      allocation_plan_->allocation_lifetimes);
  // 初始化总大小为0
  allocation_plan_->total_size = 0;
  // 遍历所有分配大小
  for (const auto i : c10::irange(allocation_plan_->allocation_sizes.size())) {
    // 如果分配的生命周期为最大值，则跳过
    if (allocation_plan_->allocation_lifetimes[i] ==
        std::numeric_limits<uint64_t>::max()) {
      continue;
    }
    // 计算分配的限制
    auto limit = allocation_plan_->allocation_offsets[i] +
        allocation_plan_->allocation_sizes[i];


这些注释涵盖了每行代码的功能和目的，确保代码的每个部分都能清晰地理解其作用。
    # 将allocation_plan_的total_size更新为当前总大小和限制中的较大值
    allocation_plan_->total_size =
        std::max(allocation_plan_->total_size, limit);
}

void AllocationPlanner::clear() {
  // 清空分配计划
  allocation_plan_->clear();
  // 清空指针到ID的映射
  allocation_ptr_to_id_.clear();
}

void CPUProfilingAllocator::set_plan(const AllocationPlan* plan) {
  // 检查分配计划不能为空指针
  TORCH_CHECK(plan != nullptr, "Allocation plan is nullptr.");
  // 设置新的分配计划
  plan_ = plan;
  // 重置分配ID
  allocation_id_ = 0;
  // 清空指针到ID的映射
  allocation_ptr_to_id_.clear();
  // 如果当前内存大小小于新计划的总大小，则释放现有内存并重新分配更大的内存空间
  if (current_size_ < plan->total_size) {
    // 释放现有的 CPU 内存
    c10::free_cpu(blob_);
    // 重新分配 CPU 内存为新计划的总大小
    blob_ = c10::alloc_cpu(plan->total_size);
    // 更新当前内存大小
    current_size_ = plan->total_size;
  }
}

void CPUProfilingAllocator::unset_plan() {
  // 重置分配ID
  allocation_id_ = 0;
  // 清空指针到ID的映射
  allocation_ptr_to_id_.clear();
  // 取消当前的分配计划
  plan_ = nullptr;
}

void* CPUProfilingAllocator::allocate(const size_t bytes) {
  // 检查请求的字节数是否与计划中的分配大小匹配
  TORCH_CHECK(
      bytes == plan_->allocation_sizes[allocation_id_],
      "Got allocation request that does not match with the plan.");
  // 如果分配的生命周期是无限大，则直接分配新的内存
  if (plan_->allocation_lifetimes[allocation_id_] ==
      std::numeric_limits<uint64_t>::max()) {
    // 增加分配ID
    allocation_id_++;
    // 分配 CPU 内存
    return c10::alloc_cpu(bytes);
  }
  // 计算分配的指针位置
  void* ptr = reinterpret_cast<uint8_t*>(blob_) +
      plan_->allocation_offsets[allocation_id_];
  // 将指针映射到分配ID
  allocation_ptr_to_id_[ptr] = allocation_id_;
  // 增加分配ID
  allocation_id_++;
  // 返回分配的指针
  return ptr;
}

void CPUProfilingAllocator::free(void* const ptr) {
  // 查找指针在映射中的位置
  auto it = allocation_ptr_to_id_.find(ptr);
  // 如果未找到，说明是外部或未受管理的内存释放请求
  if (it == allocation_ptr_to_id_.end()) {
    // 或者，释放未受管理的内存或外部分配的内存
    c10::free_cpu(ptr);
    return;
  }
  // 获取分配ID
  auto id = it->second;
  // 检查分配ID是否在计划范围内
  TORCH_CHECK(
      id < plan_->allocation_lifetimes.size(),
      "Freeing allocation that is not accordingly to the plan.");
  // 获取分配的生命周期ID
  auto lifetime_id = plan_->allocation_lifetimes[id];
  // 检查分配的生命周期ID是否与当前分配ID匹配
  TORCH_CHECK(
      lifetime_id == allocation_id_,
      "Lifetime of allocations do not match: allocation_id ",
      id,
      ", expected:",
      lifetime_id,
      ", got:",
      allocation_id_);
}

CPUProfilingAllocator::~CPUProfilingAllocator() {
  // 释放 CPU 内存
  c10::free_cpu(blob_);
}

WithProfileAllocationsGuard::WithProfileAllocationsGuard(AllocationPlan* plan) {
  // 检查是否已经有分配器存在，不支持嵌套的分配器
  TORCH_CHECK(
      allocation_planner == nullptr,
      "Nesting profiling allocations is not supported.");
  // 创建新的分配计划器并清空
  planner_ = std::make_unique<AllocationPlanner>(plan);
  planner_->clear();
  // 设置全局分配计划器为当前创建的计划器
  allocation_planner = planner_.get();
}

WithProfileAllocationsGuard::~WithProfileAllocationsGuard() {
  // 生成分配计划
  planner_->formulate_plan();
  // 释放全局分配计划器
  allocation_planner = nullptr;
}

WithValidateAllocationPlanGuard::WithValidateAllocationPlanGuard(
    AllocationPlan* plan,
    // 检查是否已经存在分配规划器，不支持嵌套分配规划。
    TORCH_CHECK(
        allocation_planner == nullptr,
        "Nesting profiling allocations is not supported.");
    // 创建一个唯一指针，指向一个新的AllocationPlanner对象，使用给定的计划。
    planner_ = std::make_unique<AllocationPlanner>(plan, true);
    // 将成功标志的指针指向函数参数中传递的成功标志变量。
    success_ = success;
    // 将全局变量allocation_planner指向刚创建的分配规划器对象。
    allocation_planner = planner_.get();
}

// WithValidateAllocationPlanGuard 析构函数的实现
WithValidateAllocationPlanGuard::~WithValidateAllocationPlanGuard() {
  // 将 validation_success 的值赋给 success_ 指向的变量
  *success_ = planner_->validation_success;
  // 将 allocation_planner 置为 nullptr
  allocation_planner = nullptr;
}

// 获取线程本地的 AllocationPlanner 对象
AllocationPlanner* GetThreadLocalAllocationPlanner() {
  return allocation_planner;
}

// WithProfilingAllocatorGuard 构造函数的实现
WithProfilingAllocatorGuard::WithProfilingAllocatorGuard(
    CPUProfilingAllocator* allocator,
    const AllocationPlan* plan) {
  // 检查是否已经存在其他的 profiling_allocator，不支持嵌套使用
  TORCH_CHECK(
      profiling_allocator == nullptr,
      "Nesting profiling allocators is not supported.");
  // 将传入的 allocator 赋值给 profiling_allocator
  profiling_allocator = allocator;
  // 设置 profiling_allocator 的分配计划为传入的 plan
  profiling_allocator->set_plan(plan);
}

// WithProfilingAllocatorGuard 析构函数的实现
WithProfilingAllocatorGuard::~WithProfilingAllocatorGuard() {
  // 取消 profiling_allocator 的当前分配计划
  profiling_allocator->unset_plan();
  // 将 profiling_allocator 置为 nullptr
  profiling_allocator = nullptr;
}

// 获取线程本地的 CPUProfilingAllocator 对象
CPUProfilingAllocator* GetThreadLocalProfilingAllocator() {
  return profiling_allocator;
}

} // namespace c10
```