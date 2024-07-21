# `.\pytorch\torch\csrc\CudaIPCTypes.h`

```py
#pragma once
#ifdef USE_CUDA
#include <c10/core/Allocator.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/Logging.h>
#include <cuda_runtime_api.h>
#include <torch/csrc/Export.h>
#include <cstddef>
namespace torch {

// CUDA IPC function to collect IPC resources
TORCH_CUDA_CU_API bool CudaIPCCollect();

// Structure to hold data received via CUDA IPC
struct CudaIPCReceivedData final {
  CudaIPCReceivedData() = default;
  explicit CudaIPCReceivedData(std::shared_ptr<void> shared_ptr)
      : shared_ptr_(std::move(shared_ptr)) {}
  std::shared_ptr<void> shared_ptr_;
};

// Structure to hold data sent via CUDA IPC
struct CudaIPCSentData final {
  std::string handle_;              // Unique identifier for the IPC data
  uint64_t offset_;                 // Offset in the shared memory block
  uint64_t* counter_ptr_;           // Reference counter shared memory block
  at::DataPtr original_ptr_;        // Original memory allocation
  cudaEvent_t event_;               // Synchronization event for CUDA
  bool event_sync_required_;        // Flag indicating if event synchronization is required
  at::Device device_;               // CUDA device associated with the data

  CudaIPCSentData(
      std::string handle,
      uint64_t offset,
      uint64_t* counter_ptr,
      at::Device device);
  ~CudaIPCSentData();

  // Retrieve the current value of the reference counter
  uint64_t counter_value();

  // Accessor for the IPC data handle
  std::string handle() {
    return handle_;
  }

  // Accessor for the offset in the shared memory
  uint64_t offset() {
    return offset_;
  }

  // Set the original data pointer
  void set_original_ptr(at::DataPtr data_ptr) {
    original_ptr_ = std::move(data_ptr);
  }
};

// CUDA IPC function to obtain a new reference-counted data pointer
TORCH_CUDA_CU_API at::DataPtr GetNewRefCountedSentData(
    void* data,
    at::Device device);

// Anonymous namespace for constants related to CUDA IPC
namespace {

// Size of the reference counter file for CUDA IPC
inline constexpr int64_t CUDA_IPC_REF_COUNTER_FILE_SIZE = 10000;

// Threshold for warning after exceeding a certain number of blocks in limbo
inline constexpr int64_t CUDA_IPC_WARN_AFTER_X_BLOCKS_IN_LIMBO = 1000;

// Maximum number of CUDA events used for IPC
// Empirically determined for CUDA versions up to 10.1
inline constexpr int64_t CUDA_IPC_MAXIMUM_EVENTS_TO_USE = 1000;

// Structure to manage CUDA IPC data blocks awaiting deletion
struct CudaIPCSentDataLimbo final {
  ~CudaIPCSentDataLimbo();

  // Collect unused IPC data blocks
  bool collect();

  // Add a new IPC data block to the limbo
  void add(std::unique_ptr<CudaIPCSentData> shared_block);

  // Retrieve the current number of blocks in limbo
  uint64_t size();

 private:
  // Vector to store IPC data blocks
  std::vector<std::unique_ptr<CudaIPCSentData>> shared_blocks_;

  // Mutex for thread-safe access to the limbo
  std::mutex limbo_mutex_;
};

// Structure representing the reference counters file for CUDA IPC
struct CudaIPCRefCountersFile final {
  CudaIPCRefCountersFile(
      std::string handle,
      uint64_t size,
      at::DataPtr data_ptr)
      : size_(size),
        handle_(std::move(handle)),
        refcounted_shared_mem_(std::move(data_ptr)) {}

  // Retrieve the current reference counter pointer
  uint64_t* counter_ptr() {
    return static_cast<uint64_t*>(refcounted_shared_mem_.get()) + next_offset_;
  }

  // Set the value of the reference counter
  void set_counter(uint64_t value) {
    *counter_ptr() = value;
  }

  // Check if there are available offsets in the reference counters file
  bool have_offsets() {
    return next_offset_ < size_;
  }

  // Check if there are used slots in the reference counters file
  bool offsets_in_use() {
    return used_slots_;
  }

  // Retrieve the next available offset
  uint64_t get_offset() {
    return next_offset_;
  }

  // Move to the next offset
  void rotate_offset() {
    next_offset_++;
    used_slots_++;
  }

  // Return a used offset to the pool
  void return_offset(uint64_t offset /* unused */) {
    used_slots_--;
  }

  // Retrieve the handle associated with the reference counters file
  std::string handle() {
    // 返回 handle_ 变量的值
    return handle_;
  }

 private:
    // 下一个偏移量，初始化为 0
  uint64_t next_offset_{0};
    // 数据结构的大小
  uint64_t size_;
    // 已使用的插槽数量，初始化为 0
  uint64_t used_slots_{0};
    // 存储句柄的字符串
  std::string handle_;
    // 引用计数的共享内存指针
  at::DataPtr refcounted_shared_mem_;
};

} // namespace
} // namespace torch

namespace c10 {
namespace {

// 定义一个类 CudaIPCCollectCallback，继承自 FreeMemoryCallback 类
class CudaIPCCollectCallback : public FreeMemoryCallback {
 public:
  // 实现 Execute 方法，执行 CUDA IPC 收集操作
  bool Execute() override {
    return torch::CudaIPCCollect();
  }
};

} // namespace

} // namespace c10

#endif


这段代码是C++代码，主要内容包括定义一个类 `CudaIPCCollectCallback`，该类继承自 `FreeMemoryCallback` 类，并实现了 `Execute` 方法，用于执行 CUDA IPC 收集操作。
```