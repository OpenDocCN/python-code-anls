# `.\pytorch\torch\csrc\distributed\c10d\SymmetricMemory.hpp`

```py
#pragma once
// 使用预处理指令#pragma once确保头文件只被编译一次

#include <ATen/ATen.h>
// 包含ATen库，提供Tensor操作和计算功能
#include <torch/csrc/distributed/c10d/Store.hpp>
// 包含torch分布式库中的Store.hpp头文件

namespace c10d {
namespace symmetric_memory {

// SymmetricMemory represents symmetric allocations across a group of devices.
// The allocations represented by a SymmetricMemory object are accessible by
// all devices in the group. The class can be used for op-level custom
// communication patterns (via the get_buffer APIs and the synchronization
// primitives), as well as custom communication kernels (via the buffer and
// signal_pad device pointers).
//
// To acquire a SymmetricMemory object, each rank first allocates
// identical-sized memory via SymmetricMemoryAllocator::alloc(), then invokes
// SymmetricMemoryAllocator::rendezvous() on the memory to establish the
// association across peer buffers. The rendezvous is a one-time process, and
// the mapping between a local memory memory and the associated SymmetricMemory
// object is unique.
//
// NOTE [symmetric memory signal pad]
// Signal pads are P2P-accessible memory regions designated for
// synchronization. SymmetricMemory offers built-in synchronization primitives
// such as barriers, put_signal, and wait_signal, which are all based on signal
// pads. Users may utilize signal pads for their own synchronization logic,
// provided that the signal pads remain zero-filled following successful
// synchronization.
//
// NOTE [symmetric memory synchronization channel]
// Synchronization channels allow users to use a single SymmetricMemory object
// to perform isolated synchronizations on different streams. For example,
// consider the case in which two barriers are issued on two streams for
// different purposes. Without the concept of channels, we cannot guarantee the
// correctness of the barriers since signals issued from barrier on stream A
// can be received by the barrier on stream B. By specifying different channels
// for these two barriers, they can operate correctly in parallel.
class TORCH_API SymmetricMemory : public c10::intrusive_ptr_target {
 public:
  virtual ~SymmetricMemory() {}
  // 虚析构函数，确保正确释放派生类的资源

  virtual std::vector<void*> get_buffer_ptrs() = 0;
  // 纯虚函数，获取缓冲区指针的向量

  virtual std::vector<void*> get_signal_pad_ptrs() = 0;
  // 纯虚函数，获取信号填充区指针的向量

  // get_buffer_ptrs_dev() and get_signal_pad_ptrs_dev() each return a pointer
  // to a device array of size world_size, containing buffer pointers and
  // signal pad pointers, respectively.
  virtual void** get_buffer_ptrs_dev() = 0;
  // 纯虚函数，返回大小为world_size的设备数组指针，包含缓冲区指针

  virtual void** get_signal_pad_ptrs_dev() = 0;
  // 纯虚函数，返回大小为world_size的设备数组指针，包含信号填充区指针

  virtual size_t get_buffer_size() = 0;
  // 纯虚函数，获取缓冲区大小

  virtual size_t get_signal_pad_size() = 0;
  // 纯虚函数，获取信号填充区大小

  virtual at::Tensor get_buffer(
      int rank,
      c10::IntArrayRef sizes,
      c10::ScalarType dtype,
      int64_t storage_offset) = 0;
  // 纯虚函数，获取指定排名的缓冲区Tensor对象

  virtual void barrier(int channel) = 0;
  // 纯虚函数，根据通道进行障碍同步

  virtual void put_signal(int dst_rank, int channel) = 0;
  // 纯虚函数，向目标排名和通道放置信号

  virtual void wait_signal(int src_rank, int channel) = 0;
  // 纯虚函数，等待源排名和通道的信号

  virtual int get_rank() = 0;
  // 纯虚函数，获取本地进程的排名

  virtual int get_world_size() = 0;
  // 纯虚函数，获取世界中进程的总数
};

} // namespace symmetric_memory
} // namespace c10d
class SymmetricMemoryAllocator : public c10::intrusive_ptr_target {
 public:
  virtual ~SymmetricMemoryAllocator(){};
  // 虚析构函数，用于子类对象的正确析构

  virtual void* alloc(
      size_t size,
      int device_idx,
      const std::string& group_name) = 0;
  // 纯虚函数，分配对称内存，由子类实现具体分配逻辑

  virtual void free(void* ptr) = 0;
  // 纯虚函数，释放分配的内存块，由子类实现具体释放逻辑

  virtual size_t get_alloc_size(void* ptr) = 0;
  // 纯虚函数，获取指定内存块的分配大小，由子类实现具体获取逻辑

  virtual c10::intrusive_ptr<SymmetricMemory> rendezvous(void* ptr) = 0;
  // 纯虚函数，建立与指定内存块相关的对称内存访问，由子类实现具体逻辑

  virtual bool is_rendezvous_completed(void* ptr) = 0;
  // 纯虚函数，检查与指定内存块的对称内存访问是否已完成，由子类实现具体逻辑
};

C10_EXPORT bool is_finalizing();
// 检查是否正在进行最终化操作

C10_EXPORT void register_allocator(
    c10::DeviceType device_type,
    c10::intrusive_ptr<SymmetricMemoryAllocator> allocator);
// 注册对称内存分配器到指定设备类型

C10_EXPORT c10::intrusive_ptr<SymmetricMemoryAllocator> get_allocator(
    c10::DeviceType device_type);
// 获取指定设备类型的对称内存分配器

// 设置用于在由 `group_name` 标识的设备组上进行对称分配的存储器。组的概念是逻辑的；
// 用户可以使用预定义的组（例如由 ProcessGroup 标识的设备组）或创建自定义组。
// 注意，对称内存分配器后端可能会为实际的对称操作使用更高效的通信通道，仅在启动时使用存储器。
TORCH_API void set_group_info(
    const std::string& group_name,
    int rank,
    int world_size,
    c10::intrusive_ptr<Store> store);

struct GroupInfo {
  int rank;
  int world_size;
  c10::intrusive_ptr<c10d::Store> store;
};

C10_EXPORT const GroupInfo& get_group_info(const std::string& group_name);
// 获取用于指定组名的组信息

// 与 empty_strided 相同，但允许通过 SymmetricMemory::rendezvous() 建立对分配的张量的对称内存访问。
// 该函数本身不是集合操作。它在底层调用 SymmetricMemoryAllocator::alloc() 来请求设备上的内存分配。
//
// NOTE [symmetric memory persistent allocation]
// 如果提供了 `alloc_id`，empty_strided_p2p 将执行持久化分配。这使得函数可以缓存已分配的内存，
// 确保使用相同 `alloc_id` 的调用接收由相同内存地址支持的张量。为了安全起见，如果仍然存在先前的持久化分配
// （即返回的张量的存储仍然存在），相同 `alloc_id` 的持久化分配将失败。
// 这种确定性与通信缓冲区的内存规划（例如由 Inductor 执行）允许通信算法可靠地重用先前建立的远程内存访问。
TORCH_API at::Tensor empty_strided_p2p(
    c10::IntArrayRef size,
    c10::IntArrayRef stride,
    c10::ScalarType dtype,
    c10::Device device,
    const std::string& group_name,
    std::optional<uint64_t> alloc_id);

// 在通过 empty_strided_p2p() 和 empty_strided_p2p_persistent() 分配的张量上建立对称内存访问。
// rendezvous() 是一次性过程，本地内存区域与关联的 SymmetricMemory 对象之间的映射是唯一的。
// 后续对
// 定义了 TORCH_API 宏，用于声明公开可用的 Torch C++ API 函数
TORCH_API c10::intrusive_ptr<SymmetricMemory> rendezvous(
    const at::Tensor& tensor);

// 返回与给定张量关联的 SymmetricMemory 对象。
// 调用此函数之前必须先调用 rendezvous()，但不需要同时调用。
TORCH_API c10::intrusive_ptr<SymmetricMemory> get_symmetric_memory(
    const at::Tensor& tensor);

// 声明 symmetric_memory 命名空间，用于包含与对称内存相关的功能
namespace symmetric_memory {
// 声明结束 symmetric_memory 命名空间
} // namespace symmetric_memory

// 声明结束 c10d 命名空间
} // namespace c10d
```