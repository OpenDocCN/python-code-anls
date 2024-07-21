# `.\pytorch\c10\mobile\CPUProfilingAllocator.h`

```py
#pragma once

#include <c10/macros/Export.h> // 包含导出宏定义
#include <c10/util/flat_hash_map.h> // 包含使用平面哈希映射的实用工具
#include <cstddef> // 包含标准尺寸定义
#include <cstdint> // 包含标准整数类型定义
#include <memory> // 包含内存管理相关的标准库头文件
#include <vector> // 包含向量容器的标准库头文件

namespace c10 {

/*
 * Given a sequence of allocations in a thread, AllocationPlan records
 * 1. size of each allocation
 * 2. Lifetime of each allocation.
 * 3. allocation offsets: Memory offset for each allocation in a single blob of
 * memory
 * 4. Total size of a blob of memory required to satisfy all the allocations.
 */
class C10_API AllocationPlan {
 private:
  std::vector<uint64_t> allocation_sizes; // 记录每个分配的大小
  std::vector<uint64_t> allocation_lifetimes; // 记录每个分配的生命周期
  std::vector<uint64_t> allocation_offsets; // 将每个分配映射到内存块中的偏移量
  uint64_t total_size{0}; // 所有分配所需内存块的总大小
  void clear(); // 清空分配计划
  friend class AllocationPlanner; // 声明 AllocationPlanner 为友元类
  friend class CPUProfilingAllocator; // 声明 CPUProfilingAllocator 为友元类
};

/*
 * Map of memory ptr to allocation id. This is auxiliary information only
 * used to establish lifetime of allocations.
 */
class C10_API AllocationPlanner {
 private:
  AllocationPlan* allocation_plan_{nullptr}; // 指向 AllocationPlan 对象的指针
  ska::flat_hash_map<const void*, uint64_t> allocation_ptr_to_id_; // 将内存指针映射到分配ID的哈希映射
  uint64_t allocation_id_{0}; // 当前分配的ID
  bool validation_mode_{false}; // 是否处于验证模式

  bool validate_allocation(const uint64_t size, const void* ptr); // 验证分配的有效性
  bool validate_free(const void* ptr); // 验证释放操作的有效性

 public:
  bool validation_success{true}; // 验证成功标志，默认为 true

  AllocationPlanner() = delete; // 禁用默认构造函数
  AllocationPlanner(AllocationPlan* plan, bool validate = false) // 构造函数，接受 AllocationPlan 对象和验证模式
      : allocation_plan_(plan), validation_mode_(validate) {} // 初始化成员变量

  void record_allocation(const uint64_t size, const void* ptr); // 记录分配操作
  void record_free(const void* ptr); // 记录释放操作
  void formulate_plan(); // 制定分配计划
  void clear(); // 清空分配器
};

// NOT THREAD SAFE profiling allocator.
class C10_API CPUProfilingAllocator {
 private:
  const AllocationPlan* plan_{nullptr}; // 指向 AllocationPlan 对象的常量指针
  uint64_t allocation_id_{0}; // 当前分配的ID
  uint64_t current_size_{0}; // 当前已分配的内存大小
  void* blob_{nullptr}; // 指向内存块的指针
  ska::flat_hash_map<const void*, uint64_t> allocation_ptr_to_id_; // 将内存指针映射到分配ID的哈希映射

 public:
  ~CPUProfilingAllocator(); // 析构函数

  void set_plan(const AllocationPlan* plan); // 设置分配计划
  void unset_plan(); // 取消设置的分配计划
  void* allocate(const size_t bytes); // 分配指定大小的内存
  void free(void* const ptr); // 释放内存
};

} // namespace c10
/*
 * Usage: Profile allocations made by one run of the model.
 * AllocationPlan plan;
 * {
 *   WithProfileAllocationsGuard profile_guard(&plan);
 *   module.forward(...);
 * }
 * plan now contains allocation plan.
 */
class C10_API WithProfileAllocationsGuard {
 public:
  // 构造函数，接受一个指向 AllocationPlan 的指针，并初始化内部的 AllocationPlanner 对象
  WithProfileAllocationsGuard(AllocationPlan* plan);

  // 析构函数，用于释放内部的 AllocationPlanner 对象
  ~WithProfileAllocationsGuard();

 private:
  // 持有一个唯一指针指向 AllocationPlanner 对象，用于分配内存并记录分配信息
  std::unique_ptr<AllocationPlanner> planner_;
};

/*
 * Usage: Validate allocation plan made with WithProfileAllocationGuard
 * bool plan_validation_success, success = true;
 * for (some number of representative inputs)
 * {
 *   WithValidateAllocationPlanGuard(&plan, &plan_validation_success);
 *   module.forward(...);
 *   success = success && plan_validation_success;
 * }
 * success == true means allocations are according to plan
 * else for some inputs allocation pattern changed.
 */
class C10_API WithValidateAllocationPlanGuard {
 public:
  // 构造函数，接受一个指向 AllocationPlan 的指针和一个 bool 类型的指针，用于验证分配计划
  WithValidateAllocationPlanGuard(AllocationPlan* plan, bool* success);

  // 析构函数，用于释放内部的 AllocationPlanner 对象
  ~WithValidateAllocationPlanGuard();

 private:
  // 持有一个唯一指针指向 AllocationPlanner 对象，用于验证分配计划
  std::unique_ptr<AllocationPlanner> planner_;
  // 指向 bool 类型变量的指针，用于存储验证结果
  bool* success_;
};

// 获取当前线程的 AllocationPlanner 对象的指针
AllocationPlanner* GetThreadLocalAllocationPlanner();

/*
 * Usage: Allocate tensors accordingly to allocation plan
 * First make allocation plan.
 *  See WithProfileAllocationsGuard usage.
 * Second validate allocation plan.
 *  See WithValidateAllocationPlanGuard usage.
 * CPUProfilingAllocator profiling_allocator;
 * {
 *   WithProfilingAllocatorGuard allocator_guard(&profiling_allocator, &plan);
 *   module.forward(...);
 * }
 */
class C10_API WithProfilingAllocatorGuard {
 public:
  // 构造函数，接受一个 CPUProfilingAllocator 对象指针和一个 AllocationPlan 对象指针，用于分配内存并进行性能分析
  WithProfilingAllocatorGuard(
      CPUProfilingAllocator* allocator,
      const AllocationPlan* plan);

  // 析构函数，用于释放内部的 AllocationPlanner 对象
  ~WithProfilingAllocatorGuard();
};

// 获取当前线程的 CPUProfilingAllocator 对象的指针
CPUProfilingAllocator* GetThreadLocalProfilingAllocator();

} // namespace c10
```