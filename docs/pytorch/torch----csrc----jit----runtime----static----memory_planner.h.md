# `.\pytorch\torch\csrc\jit\runtime\static\memory_planner.h`

```py
#pragma once

#include <torch/csrc/jit/runtime/static/impl.h>

namespace torch::jit {

// A StorageGroup represents a collection of tensors that share backing storage.
class StorageGroup {
 public:
  // Every storage group must contain at least one tensor.
  explicit StorageGroup(at::Tensor* tensor) : group_{tensor} {}

  // 添加一个新的张量到存储组中
  void addTensor(at::Tensor* tensor) {
    group_.push_back(tensor);
  }

  // 返回存储组中的所有张量
  const std::vector<at::Tensor*>& group() const {
    return group_;
  }

  // 返回存储组中最大张量的大小
  size_t maxTensorSize() const {
    return max_tensor_size_;
  }

  // 设置存储组中最大张量的大小
  void setMaxTensorSize(size_t new_size) {
    max_tensor_size_ = new_size;
  }

  // 返回存储组管理的张量数量
  size_t numManagedTensors() const {
    return group_.size();
  }

 private:
  // The size attribute represents the amount of memory that will be
  // allocated for all tensors in this storage group. Initially it
  // is zero, eventually it gets updated by the MemoryPlanner.
  size_t max_tensor_size_ = 0;
  std::vector<at::Tensor*> group_{};
};

// A contiguous buffer of `StorageImpl`s
class ManagedStorages {
 public:
  // 构造函数
  ManagedStorages();

  // 析构函数
  ~ManagedStorages();

  // 分配存储空间
  void allocate(size_t capacity);

  // 释放存储空间
  void deallocate();

  // 返回是否已分配存储空间的状态
  bool is_allocated() const {
    return storages_ != nullptr;
  }

  // 将新的 StorageImpl 追加到缓冲区中
  void append(at::StorageImpl& storageImpl);

  // 返回给定索引处的 StorageImpl 引用
  at::StorageImpl& operator[](size_t idx) {
    TORCH_INTERNAL_ASSERT(storages_ != nullptr);
    return storages_[idx];
  }

  // 返回给定索引处的 const StorageImpl 引用
  const at::StorageImpl& operator[](size_t idx) const {
    TORCH_INTERNAL_ASSERT(storages_ != nullptr);
    return storages_[idx];
  }

  // 返回当前存储的数量
  size_t size() const {
    return size_;
  }

  // 返回缓冲区是否为空
  bool empty() const {
    return size_ == 0;
  }

  // 返回缓冲区的总容量
  size_t capacity() const {
    return capacity_;
  }

 private:
  // We will use placement-new to add new storages to this buffer
  // 用于放置新的存储的缓冲区
  at::StorageImpl* storages_;

  // 当前放置到存储缓冲区中的存储数量
  size_t size_;

  // 存储缓冲区的总分配容量
  size_t capacity_;
};

// Assigns storage to managed tensors based on input parameters
// 将存储分配给管理的张量，基于输入参数
TORCH_API std::vector<StorageGroup> assignStorageToManagedTensors(
    graph_node_list nodes,
    const ManagedTensorRanges& ranges,
    const c10::FastMap<const Value*, at::Tensor*>& tensor_value_to_tensor);

// There are three types of ops in a processed graph in Static Runtime:
//   1. op with _out variant
//   2. view-producing op
//   3. tensor-producing op (could be replaced with type 1 by adding the _out
//      variant to Static Runtime)
// In Static Runtime, type 2 ops are replaced with their corresponding copy
// versions when enable_out_variant is enabled and become type 1 ops.The memory
// planner only manages tensors that are outputs of type 1 ops. For type 3, the
// output tensors are allocated inside the operator and can't be directly
// managed by memory planner.
//
// Memory planner tries to minimize the number of memory allocations by
// 内存规划器试图通过...
// tracking the output tensors of ops with _out variants with unique DataPtr
// (part of StorageImpl). It tries to do this in several steps:
//   1. record the max memory usage for each Tensor with unique DataPtr at the
//      end of each iteration
//   2. in the next iteration, allocate the buffer for the max total usage and
//      compute the offset of each allocation with regard to the single memory
//      buffer, optionally reusing memory. In the first iteration, we rely on
//      the default allocator for memory allocation.
//   3. free the buffer at the end of each iteration
// Steps 1 and 3 are handled by `deallocate()`, and step 2 by `allocate()`.
// Only models with simple output types are supported, i.e. None, Tensor or
// List/Tuple/Dict of Tensors. Complex output types such as List of Lists are
// not supported.
//
// Additional Optimizations:
//
// [Borrowed IValue Outputs]
// A few native ops (notably, `static_runtime::dict_unpack` and
// `static_runtime::VarTupleUnpack`) simply unpack IValues to a bunch of
// outputs without modification. For example, `dict_unpack` does the following:
// for each key in inputs:
//     output[i] = dict_input[key]
// To avoid refcount bumps, the outputs of these ops are non-owning references.
// This requires special logic in the memory planner - when adding an op that
// borrows outputs, be sure that the memory planner is updated accordingly!
//
// [Managed Output Tensors]
// The memory planner is able to manage output tensors if the appropriate
// `StaticModuleOptions` are set. However, the memory planner handles output
// tensors separately from regular intermediate tensors:
// 1. They don't participate in memory reuse.
// 2. The memory planner cannot reclaim their backing storage until they have
//    been explicitly freed by the client.

class MemoryPlanner {
 public:
  // Constructor initializes the MemoryPlanner instance with necessary parameters.
  MemoryPlanner(
      BlockRunner* block_runner,         // Pointer to BlockRunner instance
      const BlockInfo& block_info,       // Reference to BlockInfo object
      bool enable_out_variant,           // Flag indicating if _out variants are enabled
      bool manage_output_tensors);       // Flag indicating if output tensors should be managed

  // Disable copying and moving operations for MemoryPlanner
  MemoryPlanner(const MemoryPlanner&) = delete;
  MemoryPlanner& operator=(const MemoryPlanner&) = delete;
  MemoryPlanner(MemoryPlanner&&) = delete;
  MemoryPlanner& operator=(MemoryPlanner&&) = delete;
  virtual ~MemoryPlanner() = default;

  // Allocate memory based on current planning
  void allocate();

  // Deallocate memory at the end of each iteration
  void deallocate();

  // Deallocate output tensors managed by the planner
  void deallocateOutputTensors();

  // Get total number of managed tensors
  size_t total_num_managed_tensors() const {
    return num_managed_tensors_;
  }

  // Get total number of reused tensors
  size_t total_reused_tensors() const {
    return reused_tensors_;
  }

  // Get total number of managed output tensors
  size_t total_num_managed_output_tensors() const {
    return managed_output_tensors_.size();
  }

  // Get total number of unmanaged tensors (non-scalars)
  C10_NODISCARD size_t total_num_unmanaged() const {
    return num_unmanaged_non_scalars() + num_unmanaged_scalars();
  }

  // Get number of unmanaged non-scalar tensors
  C10_NODISCARD size_t num_unmanaged_non_scalars() const {
    return unmanaged_ivalues_.size() + unmanaged_borrowed_ivalues_.size();
  }

  // Get number of unmanaged scalar tensors
  C10_NODISCARD size_t num_unmanaged_scalars() const {
    // Implementation continues in the omitted part...
  // 返回未管理的标量值的数量
  return num_unmanaged_scalar_ivalues_;
}

// 返回总的受管理内存大小
size_t total_managed() const {
  return managed_bytes_;
}

// 返回输出缓冲区的字节数
size_t numOutputBufferBytes() const {
  return output_buffer_bytes_;
}

// 检查 `ivalue` 是否作为受管理的张量存在。仅用于 DCHECK() 断言检查。
bool isManagedOutputTensor(const IValue& ivalue) const {
  // 如果输出缓冲区未分配，返回 false
  if (!output_buffer_ || // output buffer got already deallocated.
      output_buffer_bytes_ == 0 || // memory planning is not yet initialized.
      !ivalue.isTensor() // a non-tensor is never managed
  ) {
    return false;
  }
  // 获取 ivalue 对应的张量
  const auto& tensor = ivalue.toTensor();
  // 如果张量没有有效的存储或数据指针，返回 false
  if (!tensor.has_storage() || !tensor.storage().data_ptr()) {
    return false;
  }
  // TODO: Improve this once D31357486 is landed.
  // 获取张量数据指针，并进行边界检查
  uint8_t* tensor_ptr =
      static_cast<uint8_t*>(tensor.storage().data_ptr().get());
  uint8_t* buffer_start = static_cast<uint8_t*>(output_buffer_.get());
  uint8_t* buffer_end = buffer_start + output_buffer_bytes_;
  // 检查张量数据指针是否在输出缓冲区范围内
  return buffer_start <= tensor_ptr && tensor_ptr < buffer_end;
}

// 检查给定的 StorageImpl 是否属于已管理的存储中的一个
bool isManagedStorageImpl(const at::StorageImpl* impl) const {
  // 如果没有已管理的存储，返回 false
  if (storages_.empty()) {
    return false;
  }
  // 将 StorageImpl 指针转换为整数类型进行比较，检查其是否在已管理存储的范围内
  const auto impl_p = reinterpret_cast<uintptr_t>(impl);
  const auto start = reinterpret_cast<uintptr_t>(&storages_[0]);
  const auto end =
      reinterpret_cast<uintptr_t>(&storages_[0] + storages_.size());
  return impl_p >= start && impl_p < end;
}

// 检查给定的数据指针是否与内部缓冲区发生重叠
bool overlapWithInternalBuffer(void* data_ptr) {
    return buffer_start_ <= data_ptr && data_ptr < buffer_end_;
  }



# 返回一个布尔值，指示给定的数据指针是否位于预分配缓冲区的有效范围内
  }

 protected:
  # 分配指定大小的缓冲区，并返回其起始地址
  uint8_t* allocateBuffer(size_t num_bytes);

  size_t managed_bytes_{0};
  size_t reused_tensors_{0};

  // We allocate StorageImpls ourselves so that 1) we don't have to do
  // an extra two loads per Tensor (which will likely miss in the CPU
  // data cache) first reading the Storage (i.e., StorageImpl pointer)
  // from the TensorImpl object and then second dereferencing it and
  // 2) our memory access pattern during allocate() has high locality.
  // We don't have any guarantee that the model doesn't change the
  // Storage for managed tensors out from under us during execution,
  // so we have to check the StorageImpls each time we deallocate.
  ManagedStorages storages_;

  // Contains the size (in bytes) of the data to be allocated for each storage
  std::vector<size_t> storages_nbytes_;

 private:
  // ivalues created in one run but not managed by MemoryPlanner
  std::vector<IValue*> unmanaged_ivalues_;

  // Special class of unmanaged values: some native ops create IValues
  // in a "borrowed" state that can and must be cleaned up without a
  // reference count decrement.
  std::vector<IValue*> unmanaged_borrowed_ivalues_;

  // Even more special class of unmanaged values: if select_tensor
  // outputs are outputs of the graph, then they need to be restored
  // to an ordinary "strong reference" state.
  std::vector<IValue*> borrowed_ivalues_needing_incref_;

  std::vector<std::pair<size_t, at::Tensor*>> managed_output_tensors_{};
  at::DataPtr buffer_; // allocated each time we call Run()
  uint8_t* buffer_start_{nullptr};
  uint8_t* buffer_end_{nullptr};
  size_t num_managed_tensors_{0};
  size_t num_unmanaged_scalar_ivalues_{0};

  at::DataPtr output_buffer_;
  size_t output_buffer_bytes_{0};

  virtual void allocateManagedTensors() = 0;
  virtual void deallocateManagedTensors() = 0;

  void allocateOutputTensors();
};

// 定义一个名为 StandardMemoryPlanner 的类，继承自 MemoryPlanner
class StandardMemoryPlanner : public MemoryPlanner {
 public:
  // 构造函数，接受 BlockRunner 指针、BlockInfo 对象引用以及若干布尔类型参数作为参数
  StandardMemoryPlanner(
      BlockRunner* block_runner,
      const BlockInfo& block_info,
      bool enable_out_variant,
      bool manage_output_tensors,
      bool optimize_memory);

 protected:
  // 重写基类虚函数 allocateManagedTensors，用于分配托管的张量
  void allocateManagedTensors() override;
  // 重写基类虚函数 deallocateManagedTensors，用于释放托管的张量
  void deallocateManagedTensors() override;

  // 存储托管张量的向量
  std::vector<StorageGroup> managed_tensors_{};
};

// 结束命名空间 torch::jit
} // namespace torch::jit
```