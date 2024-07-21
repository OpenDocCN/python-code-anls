# `.\pytorch\torch\csrc\distributed\c10d\CUDASymmetricMemory.hpp`

```
#pragma once
// 包含 ATen 库的头文件
#include <ATen/ATen.h>
// 包含 Torch 分布式 c10d 库的 Store.hpp 头文件
#include <torch/csrc/distributed/c10d/Store.hpp>
// 包含 Torch 分布式 c10d 库的 SymmetricMemory.hpp 头文件
#include <torch/csrc/distributed/c10d/SymmetricMemory.hpp>

// 进入 c10d 命名空间，其中包含 symmetric_memory 子命名空间
namespace c10d {
namespace symmetric_memory {

// 根据不同条件定义 HandleType 类型
#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)
using HandleType = CUmemGenericAllocationHandle;
#else
using HandleType = void*;
#endif

// CUDASymmetricMemory 类，继承自 SymmetricMemory
class CUDASymmetricMemory : public SymmetricMemory {
 public:
  // 构造函数，接受多个参数来初始化成员变量
  CUDASymmetricMemory(
      std::vector<HandleType> handles,    // 分配的句柄数组
      size_t block_size,                  // 块大小
      std::vector<void*> buffers,         // 缓冲区指针数组
      std::vector<void*> signal_pads,     // 信号填充指针数组
      size_t buffer_size,                 // 缓冲区大小
      int local_device_idx,               // 本地设备索引
      int rank,                           // 进程的排名
      int world_size);                    // 进程总数

  // 析构函数，用于清理资源
  ~CUDASymmetricMemory() override;

  // 返回缓冲区指针数组
  std::vector<void*> get_buffer_ptrs() override;

  // 返回信号填充指针数组
  std::vector<void*> get_signal_pad_ptrs() override;

  // 返回缓冲区指针数组的设备版本
  void** get_buffer_ptrs_dev() override;

  // 返回信号填充指针数组的设备版本
  void** get_signal_pad_ptrs_dev() override;

  // 返回缓冲区大小
  size_t get_buffer_size() override;

  // 返回信号填充大小
  size_t get_signal_pad_size() override;

  // 获取指定排名的缓冲区作为 Tensor
  at::Tensor get_buffer(
      int rank,                          // 目标排名
      c10::IntArrayRef sizes,            // Tensor 尺寸
      c10::ScalarType dtype,             // Tensor 数据类型
      int64_t storage_offset) override;  // 存储偏移量

  // 执行通信屏障操作
  void barrier(int channel) override;

  // 发送信号给目标排名的指定通道
  void put_signal(int dst_rank, int channel) override;

  // 等待来自指定排名的信号的指定通道
  void wait_signal(int src_rank, int channel) override;

  // 获取本地进程的排名
  int get_rank() override;

  // 获取总进程数
  int get_world_size() override;

 private:
  std::vector<HandleType> handles_;      // 句柄数组
  size_t block_size_;                    // 块大小
  std::vector<void*> buffers_;           // 缓冲区指针数组
  std::vector<void*> signal_pads_;       // 信号填充指针数组
  size_t buffer_size_;                   // 缓冲区大小
  int local_device_idx_;                 // 本地设备索引
  int rank_;                             // 进程的排名
  int world_size_;                       // 进程总数
  void** buffers_dev_;                   // 缓冲区指针数组的设备版本
  void** signal_pads_dev_;               // 信号填充指针数组的设备版本
  std::optional<std::function<void(void)>> finalizer_;  // 可选的清理函数
};

// Block 结构体，继承自 c10::intrusive_ptr_target
struct Block : public c10::intrusive_ptr_target {
  HandleType handle;                    // 句柄
  int device_idx;                       // 设备索引
  size_t block_size;                    // 块大小
  size_t buffer_size;                   // 缓冲区大小
  size_t signal_pad_offset;             // 信号填充偏移量
  std::string group_name;               // 组名
  c10::intrusive_ptr<CUDASymmetricMemory> symm_mem = nullptr;  // 对称内存指针

  // 构造函数，接受多个参数来初始化成员变量
  Block(
      HandleType handle,
      int device_idx,
      size_t block_size,
      size_t buffer_size,
      size_t signal_pad_offset,
      const std::string& group_name)
      : handle(handle),
        device_idx(device_idx),
        block_size(block_size),
        buffer_size(buffer_size),
        signal_pad_offset(signal_pad_offset),
        group_name(group_name),
        symm_mem(nullptr) {}
};

// CUDASymmetricMemoryAllocator 类，继承自 SymmetricMemoryAllocator
class CUDASymmetricMemoryAllocator : public SymmetricMemoryAllocator {
 public:
  // 分配指定大小的内存块，返回指针
  void* alloc(size_t size, int device_idx, const std::string& group_name) override;

  // 释放指定指针对应的内存块
  void free(void* ptr) override;

  // 获取指定指针对应的内存块大小
  size_t get_alloc_size(void* ptr) override;

  // 在分布式环境中处理指定指针的同步内存对象
  c10::intrusive_ptr<SymmetricMemory> rendezvous(void* ptr) override;

  // 检查指定指针对应的同步操作是否已完成
  bool is_rendezvous_completed(void* ptr) override;

 private:
  // 根据指针查找对应的 Block 对象
  c10::intrusive_ptr<Block> find_block(void* ptr);

  std::shared_mutex mutex_;                               // 互斥锁
  std::unordered_map<void*, c10::intrusive_ptr<Block>> ptr_to_block_;  // 指针到 Block 对象的映射
};

} // namespace symmetric_memory
} // namespace c10d
```