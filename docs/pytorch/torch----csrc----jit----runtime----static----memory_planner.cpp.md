# `.\pytorch\torch\csrc\jit\runtime\static\memory_planner.cpp`

```py
// 包含 C++ 头文件：c10 核心库中的对齐相关功能
#include <c10/core/alignment.h>
// 包含 Torch JIT 静态内存规划器的头文件
#include <torch/csrc/jit/runtime/static/memory_planner.h>

// 包含 ATen 张量的头文件
#include <ATen/Tensor.h>
// 包含 Torch JIT 别名分析的头文件
#include <torch/csrc/jit/ir/alias_analysis.h>
// 包含 Torch JIT 运行时静态实现的日志头文件
#include <torch/csrc/jit/jit_log.h>
// 包含 Torch JIT 静态运行时实现的头文件
#include <torch/csrc/jit/runtime/static/impl.h>
// 包含迭代器标准库头文件
#include <iterator>

// Torch JIT 命名空间
namespace torch::jit {

// 匿名命名空间，用于实现私有函数和变量
namespace {

// 检查是否为特殊未管理情况的函数
bool isUnmanagedSpecialCase(const ProcessedNode& pnode, size_t output_idx) {
  DCHECK(output_idx < pnode.outputs().size());
  // 定义静态符号表中的 to_maybe_copy_out 符号
  static const auto to_maybe_copy_out_symbol =
      c10::Symbol::fromQualString("static_runtime::to_maybe_copy_out");
  // 启发式和特殊情况：
  // 如果 to_maybe_copy_out 在第一次迭代中实际上没有做任何事情，
  // 则假定它在之后的迭代中也不会做任何事情，并避免管理其输出。
  return pnode.node()->kind() == to_maybe_copy_out_symbol &&
      pnode.Output(output_idx).isNone();
}

// 将处理节点映射为张量的函数
c10::FastMap<const Value*, at::Tensor*> tensorValueToTensor(
    const std::vector<ProcessedNode>& nodes,
    const c10::FastSet<const Value*>& managed_tensor_values) {
  c10::FastMap<const Value*, at::Tensor*> tensor_value_to_tensor;
  for (auto& pnode : nodes) {
    auto* node = pnode.node();
    for (const auto output_idx : c10::irange(node->outputs().size())) {
      auto* output = node->output(output_idx);

      // 如果输出张量不在受管理张量值集合中，则继续下一次迭代
      if (managed_tensor_values.find(output) == managed_tensor_values.end()) {
        continue;
      }

      auto& ival = pnode.Output(output_idx);

      // ival 在特殊情况下允许为 None，例如 to_maybe_copy_out
      DCHECK(
          ival.isTensor() ||
          (ival.isNone() && isUnmanagedSpecialCase(pnode, output_idx)));

      // 如果 ival 是张量，则将其添加到映射中
      if (ival.isTensor()) {
        tensor_value_to_tensor.emplace(
            output,
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
            const_cast<at::Tensor*>(&ival.toTensor()));
      }
    }
  }
  return tensor_value_to_tensor;
}

// 计算对齐的张量大小函数：如果已经对齐，则不改变大小，否则增加大小以实现对齐
size_t compute_aligned_tensor_size(size_t nbytes) {
  // 注意：以下所有内容都是基于 size_t 类型的
  return (nbytes + c10::gAlignment - 1) & (~(c10::gAlignment - 1));
}

// 分配缓冲区的函数
at::DataPtr allocate_buffer(size_t size) {
  // 获取 CPU 缓存分配器
  at::Allocator* allocator = c10::GetCPUCachingAllocator();
  // 使用分配器分配指定大小的缓冲区
  return allocator->allocate(size);
}

} // namespace

// 将存储分配给受管理张量的函数
std::vector<StorageGroup> assignStorageToManagedTensors(
    graph_node_list nodes,
    const ManagedTensorRanges& ranges,
    const c10::FastMap<const Value*, at::Tensor*>& tensor_value_to_tensor) {
  // 存储管理张量组的向量
  std::vector<StorageGroup> managed_tensor_groups;
  // 映射每个 Value* 到其分配的存储组
  c10::FastMap<const Value*, size_t> storage_group_mapping;
  // 在每次迭代中，此向量存储可重用的存储组集合
  std::vector<size_t> free_storage_groups;

  // 创建新存储组的 Lambda 函数
  auto makeNewStorageGroup = [&](const Value* value) {
    // 获取当前存储组的编号
    const auto storage_group = managed_tensor_groups.size();
    // 将 value 映射到其存储组
    storage_group_mapping.emplace(value, storage_group);


这里只展示了部分代码的注释，因为完整的代码超过了最大长度限制。
    // 获取指向 value 的 tensor 指针，从 tensor_value_to_tensor 映射中查找
    auto* tensor_ptr = tensor_value_to_tensor.at(value);
    // 将 tensor_ptr 添加到 managed_tensor_groups 中作为新的管理组
    managed_tensor_groups.emplace_back(tensor_ptr);
  };

  auto assignToAvailableStorageGroup = [&](const Value* value) {
    // 确保 free_storage_groups 不为空
    DCHECK(!free_storage_groups.empty());
    // 获取最后一个空闲存储组的索引
    const auto storage_group = free_storage_groups.back();
    // 断言 storage_group 索引小于 managed_tensor_groups 的大小
    TORCH_DCHECK_LT(storage_group, managed_tensor_groups.size());
    // 将 value 映射到 storage_group
    storage_group_mapping.emplace(value, storage_group);
    // 获取指向 value 的 tensor 指针
    auto* tensor_ptr = tensor_value_to_tensor.at(value);
    // 将 tensor_ptr 添加到对应的 managed_tensor_groups[storage_group] 中
    managed_tensor_groups[storage_group].addTensor(tensor_ptr);
    // 弹出最后一个空闲存储组
    free_storage_groups.pop_back();
  };

  auto isManagedTensor = [&](const Value* value) {
    // 检查 value 是否在 tensor_value_to_tensor 中管理
    return tensor_value_to_tensor.find(value) != tensor_value_to_tensor.end();
  };

  for (auto* node : nodes) {
    // 为输出分配存储组
    for (const auto output_idx : c10::irange(node->outputs().size())) {
      Value* output = node->output(output_idx);
      // 如果输出不在管理范围内，则跳过
      if (!isManagedTensor(output)) {
        continue;
      }
      // 如果空闲存储组为空，则创建新的存储组
      if (free_storage_groups.empty()) {
        makeNewStorageGroup(output);
        continue;
      }
      // 将输出分配到可用的存储组中
      assignToAvailableStorageGroup(output);
    }

    // 这个节点可能是某些受管理 tensor 的最后使用点。如果是，我们可以标记相应的存储组为自由状态。
    if (ranges.nodeFreesManagedTensors(node)) {
      // 获取节点后可用的新 tensor 值集合
      const auto& new_free_tensors =
          ranges.availableTensorValuesAfterNode(node);
      for (auto* tensor_value : new_free_tensors) {
        // 在处理 to_maybe_copy_out 等特殊情况时，需要在此处检查 tensor_value 是否受管理
        if (!isManagedTensor(tensor_value)) {
          continue;
        }
        // 获取 tensor_value 对应的存储组索引
        const auto storage_group = storage_group_mapping.at(tensor_value);
        // 将存储组索引推回 free_storage_groups 中
        free_storage_groups.push_back(storage_group);
      }
    }
  }
  // 返回管理的 tensor 组列表
  return managed_tensor_groups;
}

ManagedStorages::ManagedStorages()
    : storages_(nullptr), size_(0), capacity_(0) {}

// 析构函数，释放所有已分配的存储空间
ManagedStorages::~ManagedStorages() {
  // 调用 deallocate() 函数释放存储空间
  deallocate();
}

// 分配存储空间
void ManagedStorages::allocate(size_t capacity) {
  // 检查是否已经分配存储空间
  TORCH_CHECK(!is_allocated(), "Must deallocate before allocating again");
  // 再次确认 size_ 是否为 0，以确保未分配
  TORCH_INTERNAL_ASSERT(size_ == 0);
  capacity_ = capacity;
  // 使用 reinterpret_cast 将内存块视为 at::StorageImpl 数组
  storages_ = reinterpret_cast<at::StorageImpl*>(
      new unsigned char[capacity_ * sizeof(at::StorageImpl)]);
}

// 释放存储空间
void ManagedStorages::deallocate() {
  if (is_allocated()) {
    // 逐个调用存储空间的析构函数
    for (const size_t idx : c10::irange(size_)) {
      storages_[idx].~StorageImpl();
    }
    // 释放内存块
    delete[] reinterpret_cast<unsigned char*>(storages_);
    capacity_ = 0;
    size_ = 0;
    storages_ = nullptr;
  }
}

// 向 ManagedStorages 中添加新的 at::StorageImpl 对象
void ManagedStorages::append(at::StorageImpl& storageImpl) {
  // 检查是否有足够的空间来添加新对象
  TORCH_INTERNAL_ASSERT(size_ < capacity_);
  // 在预分配的内存空间中构造新的 at::StorageImpl 对象
  new (&storages_[size_]) at::StorageImpl(
      at::StorageImpl::use_byte_size_t(),
      storageImpl.nbytes(),
      storageImpl.allocator(),
      storageImpl.resizable());
  size_++;
}

namespace {

// 检查 set 中是否包含特定的 Value 对象 v
bool setIncludes(const c10::FastSet<const Value*>& set, const Value* v) {
  return set.find(v) != set.end();
}

// 为输出张量分配存储空间
std::vector<std::pair<size_t, at::Tensor*>> assignStorageToOutputTensors(
    BlockRunner* block_runner,
    const c10::FastSet<const Value*>& managed_output_tensor_values) {
  std::vector<std::pair<size_t, at::Tensor*>> managed_output_tensors;
  // 遍历每个节点的输出
  for (auto& pnode : block_runner->nodes()) {
    for (const auto i : c10::irange(pnode.outputs().size())) {
      auto& ival = pnode.Output(i);
      const auto* val = pnode.node()->outputs()[i];
      // 检查是否为受管理的输出张量，并非特殊情况
      if (!setIncludes(managed_output_tensor_values, val) ||
          isUnmanagedSpecialCase(pnode, i)) {
        continue;
      }
      TORCH_CHECK(ival.isTensor());
      at::Tensor* tensor = &ival.toTensor();
      // 将管理的输出张量添加到结果向量中
      managed_output_tensors.emplace_back(0, tensor);
    }
  }
  return managed_output_tensors;
}

} // namespace

// 内存规划器构造函数
MemoryPlanner::MemoryPlanner(
    BlockRunner* block_runner,
    const BlockInfo& block_info,
    bool enable_out_variant,
    bool manage_output_tensors) {
  const auto& managed_tensor_values = block_info.managed_tensor_values();
  const auto& managed_output_tensor_values =
      block_info.managed_output_tensor_values();
  const auto& leaked_values = block_info.leaked_values();

  // 收集未管理的输出 IValue
  c10::FastSet<IValue*> unmanaged_ivalues;
  c10::FastSet<IValue*> unmanaged_borrowed_ivalues;
  // 遍历每个处理过的节点，确定是否有借用的输出
  for (ProcessedNode& pnode : block_runner->nodes()) {
    const auto borrows_outputs = borrowsOutputs(pnode.node()->kind());
  // 遍历节点的输出
  for (const auto i : c10::irange(pnode.outputs().size())) {
    // 获取当前输出节点的值
    const Value* out_v = pnode.node()->outputs()[i];
    // 检查当前输出节点是否在受管理的张量值集合中
    const bool in_managed_tensors = setIncludes(managed_tensor_values, out_v);
    // 检查当前输出节点是否是未受管理的特殊情况
    const bool is_unmanaged_special_case = isUnmanagedSpecialCase(pnode, i);

    // 如果节点在受管理的张量值集合中且不是未受管理的特殊情况，增加受管理的张量计数
    if (in_managed_tensors && !is_unmanaged_special_case) {
      ++num_managed_tensors_;
    }

    // 检查当前输出节点是否在受管理的集合中，或者如果管理输出张量已关闭则检查标志，或者在泄露值集合中
    const bool in_managed_sets = in_managed_tensors ||
        // 输出张量管理可能已关闭，因此我们需要在此处检查标志
        (manage_output_tensors &&
         setIncludes(managed_output_tensor_values, out_v)) ||
        setIncludes(leaked_values, out_v);

    // 如果节点在受管理的集合中且不是未受管理的特殊情况，则继续下一个输出节点
    if (in_managed_sets && !is_unmanaged_special_case) {
      continue;
    }

    // 如果输出节点存储在 IValue 中时不需要堆分配，则增加未受管理的标量 IValue 计数
    if (doesNotHeapAllocateWhenStoredInIValue(*out_v->type())) {
      // 标量不需要在每次迭代后释放
      num_unmanaged_scalar_ivalues_++;
    } else if (borrows_outputs) {
      // 如果当前情况允许借用输出，将输出节点添加到未受管理的借用 IValue 集合中
      IValue& out = pnode.Output(i);
      unmanaged_borrowed_ivalues.insert(&out);
    } else {
      // 否则将输出节点添加到未受管理的 IValue 集合中
      IValue& out = pnode.Output(i);
      unmanaged_ivalues.insert(&out);
    }
  }

  // 遍历块运行器的输出
  for (IValue* output : block_runner->outputs()) {
    // 在未受管理的借用 IValue 集合中查找输出
    auto it = unmanaged_borrowed_ivalues.find(output);
    if (it != unmanaged_borrowed_ivalues.end()) {
      // 如果找到，将输出添加到需要增加引用计数的借用 IValue 集合中，并从未受管理的借用 IValue 集合中删除
      borrowed_ivalues_needing_incref_.push_back(output);
      unmanaged_borrowed_ivalues.erase(it);
    } else {
      // 否则，从未受管理的 IValue 集合中删除输出
      unmanaged_ivalues.erase(output);
    }
  }

  // 将未受管理的 IValue 集合复制到 unmanaged_ivalues_
  unmanaged_ivalues_.reserve(unmanaged_ivalues.size());
  unmanaged_ivalues_.insert(
      unmanaged_ivalues_.begin(),
      unmanaged_ivalues.begin(),
      unmanaged_ivalues.end());

  // 将未受管理的借用 IValue 集合复制到 unmanaged_borrowed_ivalues_
  unmanaged_borrowed_ivalues_.reserve(unmanaged_borrowed_ivalues.size());
  unmanaged_borrowed_ivalues_.insert(
      unmanaged_borrowed_ivalues_.begin(),
      unmanaged_borrowed_ivalues.begin(),
      unmanaged_borrowed_ivalues.end());

  // 如果启用了输出变体并且管理输出张量，则为输出张量分配存储空间
  if (enable_out_variant && manage_output_tensors) {
    managed_output_tensors_ = assignStorageToOutputTensors(
        block_runner, managed_output_tensor_values);
  }
}

uint8_t* MemoryPlanner::allocateBuffer(size_t num_bytes) {
  // 调用 allocate_buffer 函数分配内存，并将其赋给成员变量 buffer_
  buffer_ = allocate_buffer(num_bytes);
  // 将 buffer_ 转换为 uint8_t* 类型，并存储在 start 变量中
  uint8_t* start = static_cast<uint8_t*>(buffer_.get());
  // 记录缓冲区的起始位置和结束位置
  buffer_start_ = start;
  buffer_end_ = start + num_bytes;
  // 返回缓冲区的起始位置
  return start;
}

void MemoryPlanner::allocateOutputTensors() {
  if (output_buffer_bytes_ == 0) {
    return;
  }
  // 检查是否已经分配了 output_buffer_
  TORCH_CHECK(
      !output_buffer_,
      "Previously allocated output_buffer_ was not deallocated properly.");
  // 分配 output_buffer_ 的内存，并将其赋给 output_buffer_
  output_buffer_ = allocate_buffer(output_buffer_bytes_);

  size_t offset = 0;
  // 将 output_buffer_ 转换为 uint8_t* 类型，并存储在 start 变量中
  uint8_t* start = static_cast<uint8_t*>(output_buffer_.get());

  // 遍历 managed_output_tensors_ 中的每个元素
  for (const auto& ms : managed_output_tensors_) {
    auto tensor_size = ms.first;
    auto* tensor = ms.second;
    if (tensor_size == 0) {
      continue;  // 如果 tensor_size 为 0，则跳过当前循环
    }
    // 检查 offset + tensor_size 是否小于等于 output_buffer_bytes_
    TORCH_DCHECK_LE(offset + tensor_size, output_buffer_bytes_);
    void* src = static_cast<void*>(start + offset);
    // 设置 tensor 的数据指针，并将 context (`src`) 设置为数据指针的一部分
    tensor->storage().set_data_ptr_noswap(
        at::DataPtr(src, /*ctx=*/src, nullptr, tensor->device()));
    tensor->storage().set_nbytes(tensor_size);
    offset += tensor_size;
  }
  // 检查 offset 是否等于 output_buffer_bytes_
  TORCH_DCHECK_EQ(offset, output_buffer_bytes_);
}

void MemoryPlanner::allocate() {
  // TODO: Improve this once D31357486 is landed.
  // 调用 allocateManagedTensors 和 allocateOutputTensors 函数
  allocateManagedTensors();
  allocateOutputTensors();
}

void MemoryPlanner::deallocate() {
  // 对 borrowed_ivalues_needing_incref_ 中的每个元素进行处理
  for (auto& iv : borrowed_ivalues_needing_incref_) {
    auto old = std::move(*iv);
    *iv = IValue(old);
    // 销毁 old 指向的对象
    c10::MaybeOwnedTraits<c10::IValue>::destroyBorrow(old);
  }
  // 对 unmanaged_ivalues_ 中的每个元素进行处理，重置 *iv 以释放对象
  for (auto& iv : unmanaged_ivalues_) {
    *iv = IValue();
  }
  // 对 unmanaged_borrowed_ivalues_ 中的每个元素进行处理

  // 重置 *iv 以释放对象，对象可能由于引用计数归还而被回收
  for (auto& iv : unmanaged_borrowed_ivalues_) {

    *iv = IValue();
  }
}
    c10::MaybeOwnedTraits<c10::IValue>::destroyBorrow(*iv);
  }
  // 调用 destroyBorrow 函数来销毁对应的 IValue 的 borrow
  // 它在管理的 StorageImpl 的所有其他拥有引用都清理完之后调用很重要。
  // 它可以重置 StorageImpl 的引用计数到（存储组中的张量数量），
  // 因此在此之后销毁任何拥有的引用将会使引用计数低于预期并触发
  // ~intrusive_ptr_target 中的调试断言。
  deallocateManagedTensors();
  // 将 buffer_ 置为空，清空其内容
  buffer_ = {};
}

// 释放所有管理的输出张量
void MemoryPlanner::deallocateOutputTensors() {
  // 初始化输出缓冲区字节数为 0
  size_t output_buffer_bytes = 0;
  // 遍历所有管理的输出张量
  for (auto& ms : managed_output_tensors_) {
    auto* tensor = ms.second;
    // 计算张量的对齐大小
    size_t current_size =
        compute_aligned_tensor_size(tensor->storage().nbytes());
    // 重置张量的存储实现
    tensor->storage().unsafeGetStorageImpl()->reset();
    // 更新管理的输出张量的大小为当前大小，如果当前大小大于原先的大小
    if (current_size > ms.first) {
      ms.first = current_size;
    }
    // 累加到输出缓冲区字节数
    output_buffer_bytes += ms.first;
  }
  // 将计算得到的输出缓冲区字节数保存到成员变量中
  output_buffer_bytes_ = output_buffer_bytes;
  // 清空输出缓冲区
  output_buffer_ = {};
}

// 标准内存规划器的构造函数
StandardMemoryPlanner::StandardMemoryPlanner(
    BlockRunner* block_runner,
    const BlockInfo& block_info,
    bool enable_out_variant,
    bool manage_output_tensors,
    bool optimize_memory)
    : MemoryPlanner(
          block_runner,
          block_info,
          enable_out_variant,
          manage_output_tensors) {
  // 获取管理的张量值的引用
  const auto& managed_tensor_values = block_info.managed_tensor_values();
  // 如果启用输出变体
  if (enable_out_variant) {
    // 将张量值映射到张量的映射表
    const auto tensor_value_to_tensor =
        tensorValueToTensor(block_runner->nodes(), managed_tensor_values);
    // 如果优化内存
    if (optimize_memory) {
      // 为管理的张量分配存储
      managed_tensors_ = assignStorageToManagedTensors(
          block_info.node_ptrs(),
          block_info.managed_tensor_ranges(),
          tensor_value_to_tensor);
    } else {
      // 否则直接将张量添加到管理的张量列表中
      for (auto& tensor : tensor_value_to_tensor) {
        managed_tensors_.emplace_back(tensor.second);
      }
    }
  }
}

// 分配管理的张量
void StandardMemoryPlanner::allocateManagedTensors() {
  // 如果管理的字节数为 0，则直接返回
  if (managed_bytes_ == 0) {
    return;
  }
  // 断言存储器的数量大于 0
  DCHECK(storages_.size() > 0);
  // 初始化偏移量为 0，分配起始地址
  size_t offset = 0;
  auto* start = allocateBuffer(managed_bytes_);

  // 重用张量数量初始化为 0，组索引初始化为 0
  reused_tensors_ = 0;
  size_t group_idx = 0;
  // 遍历存储器索引的范围
  for (const size_t storages_idx : c10::irange(storages_.size())) {
    // 获取张量大小
    auto tensor_size = storages_nbytes_[storages_idx];
    // 如果张量大小为 0，则跳过当前组
    if (tensor_size == 0) {
      group_idx++;
      continue;
    }
    // 获取存储实现指针
    at::StorageImpl* storageImpl = &storages_[storages_idx];
    // 断言偏移量加上张量大小不超过管理的字节数
    TORCH_DCHECK_LE(offset + tensor_size, managed_bytes_);
    // 获取源指针
    void* src = static_cast<void*>(start + offset);

    // 调试模式下的额外断言检查
#ifndef NDEBUG
    // 断言张量大小等于管理的张量组的最大张量大小
    TORCH_DCHECK_EQ(tensor_size, managed_tensors_[group_idx].maxTensorSize());
    // 遍历管理的张量组，断言存储实现和张量的存储实现相等
    for (auto* tensor : managed_tensors_[group_idx].group()) {
      TORCH_DCHECK_EQ(storageImpl, tensor->storage().unsafeGetStorageImpl());
    }
#endif
    // 断言管理的张量数量不为 0
    TORCH_DCHECK_NE(managed_tensors_[group_idx].numManagedTensors(), 0);
    // 更新重用张量的数量
    reused_tensors_ += managed_tensors_[group_idx].numManagedTensors() - 1;
    // 设置存储实现的数据指针为分配的源指针，并设置大小为张量大小
    storageImpl->set_data_ptr_noswap(
        at::DataPtr(src, src, nullptr, c10::Device(c10::DeviceType::CPU)));
    storageImpl->set_nbytes(tensor_size);

    // 更新偏移量
    offset += tensor_size;
    // 更新组索引
    group_idx++;
  }
  // 断言偏移量等于管理的字节数
  TORCH_DCHECK_EQ(offset, managed_bytes_);
}
  // 释放由 ops 输出的张量内存，但保留 TensorImpl 和 StorageImpl。
  managed_bytes_ = 0;

  // 在释放期间，检查 Storages 是否已经分配，确保我们每次释放时都进行检查。
  // 如果是第一次释放，需要初始化 storages_。
  unsigned group_idx = 0;
  const bool first_time = storages_.empty();
  if (C10_UNLIKELY(first_time)) {
    // 如果 storages_ 已经分配，则释放它。
    if (storages_.is_allocated()) {
      storages_.deallocate();
    }
    // 根据 managed_tensors_ 的大小重新分配 storages_。
    storages_.allocate(managed_tensors_.size());
    // 预留 storages_nbytes_ 的空间。
    storages_nbytes_.reserve(managed_tensors_.size());
  }

  // 遍历 managed_tensors_ 中的每个 ManagedTensor。
  for (auto& ms : managed_tensors_) {
    // 获取该组 ManagedTensor 中的张量列表。
    const auto& tensors = ms.group();
    // 获取该组 ManagedTensor 中张量的最大大小。
    size_t max = ms.maxTensorSize();

    // 遍历该组 ManagedTensor 中的每个张量。
    for (auto& tensor : tensors) {
      // 获取张量的 Storage。
      const auto& storage = tensor->storage();
      // 计算对齐后的张量大小。
      size_t current_size = compute_aligned_tensor_size(storage.nbytes());
      // 获取张量的 StorageImpl。
      at::StorageImpl* tensorStorageImpl = storage.unsafeGetStorageImpl();

      if (C10_UNLIKELY(first_time)) {
        // 如果是第一次释放，重置 tensorStorageImpl。

        // 确保 storages_ 的大小符合预期。
        DCHECK(
            storages_.size() == group_idx || storages_.size() == group_idx + 1);
        // 如果 storages_ 的大小与 group_idx 不匹配，则追加新的 StorageImpl。
        if (storages_.size() == group_idx) {
          storages_.append(*tensorStorageImpl);
          storages_nbytes_.emplace_back(0);
        }
        // 获取新的 StorageImpl。
        at::StorageImpl* newImpl = &storages_[storages_.size() - 1];

        // 设置 TensorImpl 的 StorageImpl，以便我们自己管理 StorageImpl 的生命周期。
        // 使用 unsafe_adapt_non_heap_allocated 可以设置 StorageImpl 的引用计数，
        // 避免 intrusive_ptr 删除它，因为 StorageImpl 不是通过 operator new 分配的。
        tensor->unsafeGetTensorImpl()->set_storage_keep_dtype(at::Storage(
            c10::intrusive_ptr<at::StorageImpl>::
                unsafe_adapt_non_heap_allocated(newImpl, tensors.size())));
      } else if (C10_UNLIKELY(tensorStorageImpl != &storages_[group_idx])) {
        // 如果 tensorStorageImpl 与 storages_[group_idx] 不匹配，则重置 tensorStorageImpl。
        
        // 将张量设置回该组的共享 StorageImpl。
        tensor->unsafeGetTensorImpl()->set_storage_keep_dtype(
            at::Storage(c10::intrusive_ptr<at::StorageImpl>::
                            unsafe_adapt_non_heap_allocated(
                                &storages_[group_idx], tensors.size())));
      }

      // 断言 tensor 的 StorageImpl 与 storages_[group_idx] 相等。
      TORCH_DCHECK_EQ(
          tensor->storage().unsafeGetStorageImpl(), &storages_[group_idx]);

      // 更新 max 值为当前张量的最大大小和当前计算大小的较大值。
      max = std::max(max, current_size);
    }
    // 将上一次运行中的张量大小记录下来，以便为下一次运行分配张量空间
    // 这是 C2 框架的传统做法，利用张量存储大小不必与实际张量大小相匹配的特性
    // 下面的逻辑用于记录下一次运行的张量存储大小
    storages_nbytes_[group_idx++] = max;
    
    // 设置最大张量大小
    ms.setMaxTensorSize(max);
    
    // 增加管理的总字节数，将当前最大张量大小加入总字节数中
    managed_bytes_ += max;
  }
  
  // 断言：存储器数组的大小应与管理的张量数组的大小相等
  TORCH_DCHECK_EQ(storages_.size(), managed_tensors_.size());
  
  // 输出管理的总字节数日志信息
  VLOG(1) << "managed_bytes: " << managed_bytes_;
}

} // namespace torch::jit
```