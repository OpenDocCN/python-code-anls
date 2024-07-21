# `.\pytorch\torch\csrc\profiler\data_flow.cpp`

```py
// 引入 Torch 的性能分析器数据流头文件
#include <torch/csrc/profiler/data_flow.h>

// 引入 C10 的实用工具中的 overloaded.h 头文件
#include <c10/util/overloaded.h>

// 引入 Torch 的性能分析器集合头文件
#include <torch/csrc/profiler/collection.h>

// Torch 性能分析器实现的命名空间
namespace torch::profiler::impl {

// 命名空间中的匿名命名空间，定义一个常量指针 NoTensorImpl，指向空指针
namespace {
static constexpr TensorImplAddress NoTensorImpl{nullptr};

// 原始张量信息结构体，包含张量实现地址、存储实现数据、设备信息、释放状态、分配 ID 引用和张量 ID 引用
struct RawTensorInfo {
  TensorImplAddress impl_;
  StorageImplData storage_;
  c10::Device device_;
  bool is_free_;

  // 用于赋值回原始结构体的引用包装器
  std::reference_wrapper<std::optional<AllocationID>> allocation_id_ref_;
  std::reference_wrapper<std::optional<TensorID>> id_ref_;
};

// 原始张量集合结构体，管理一组原始张量信息
struct RawTensors {
  // 获取原始张量信息的引用
  std::vector<RawTensorInfo>& get() {
    return tensors_;
  }

  // 重载函数调用操作符，用于处理不同类型的张量元数据
  void operator()(TensorMetadata& t) {
    tensors_.emplace_back(RawTensorInfo{
        t.impl(), t.data_, t.device_, false, t.allocation_id_, t.id_});
  }

  void operator()(std::optional<TensorMetadata>& t) {
    if (t.has_value()) {
      (*this)(*t);
    }
  }

  void operator()(ExtraFields<EventType::Allocation>& a) {
    const StorageImplData ptr{a.ptr_};
    const auto is_free = a.alloc_size_ < 0;
    tensors_.emplace_back(RawTensorInfo{
        NoTensorImpl, ptr, a.device(), is_free, a.allocation_id_, a.id_});
  }

  void operator()(std::vector<TensorMetadata>& t) {
    for (auto& ti : t) {
      (*this)(ti);
    }
  }

  // 默认模板函数，用于忽略不需要处理的类型
  template <typename T>
  void operator()(T&) {}

  // 原始张量信息的存储容器
  std::vector<RawTensorInfo> tensors_;
};
} // namespace

// 计算唯一张量 ID 的函数，接收已排序的结果集作为参数
void calculateUniqueTensorIDs(
    std::vector<std::shared_ptr<Result>>& sorted_results) {
  // 此任务等同于 https://leetcode.com/problems/number-of-islands/
  // 首先通过贪心索引分配聚类事件，然后合并重叠的组。

  // 创建一个空的原始张量信息集合
  std::vector<RawTensorInfo> tensors;

  // 将结果展平为统一的表示形式。
  // --------------------------------------------------------------------------
  {
    // 创建 RawTensors 对象以处理原始张量信息
    RawTensors raw_tensors;

    // Python 追踪器缓存值，因此仅安全使用第一种情况。

    // 使用 ska::flat_hash_set 来记录已见过的模块和优化器实例
    ska::flat_hash_set<PyModuleSelf> seen_modules;
    ska::flat_hash_set<PyOptimizerSelf> seen_optimizers;
    // 遍历已排序的结果集合中的每一个结果
    for (auto& result : sorted_results) {
      // 使用lambda表达式访问每个结果对象的不同类型
      result->visit(c10::overloaded(
          [&](ExtraFields<EventType::TorchOp>& torch_op) {
            // 对于类型为TorchOp的额外字段，遍历其输入数据
            for (auto& i : torch_op.inputs_) {
              // 访问并处理原始张量数据
              std::visit(raw_tensors, i);
            }
          },
          [&](ExtraFields<EventType::PyCall>& py_call) {
            // 对于类型为PyCall的额外字段，检查是否包含模块信息
            if (py_call.module_.has_value() &&
                seen_modules.insert(py_call.module_->self_).second) {
              // 遍历模块的参数数据
              for (auto& p : py_call.module_->parameters_) {
                // 访问并处理原始张量数据及其梯度数据
                raw_tensors(p.metadata_);
                raw_tensors(p.grad_metadata_);
              }
            }

            // 检查是否包含优化器信息
            if (py_call.optimizer_.has_value() &&
                seen_optimizers.insert(py_call.optimizer_->self_).second) {
              // 遍历优化器的参数及其状态数据
              for (auto& p : py_call.optimizer_->parameters_) {
                // 访问并处理原始张量数据及其梯度数据
                raw_tensors(p.metadata_);
                raw_tensors(p.grad_metadata_);
                for (auto& state_i : p.state_) {
                  // 访问并处理优化器状态中的原始张量数据
                  raw_tensors(state_i.second);
                }
              }
            }
          },
          [&](auto& i) { raw_tensors(i); })); // 默认情况下处理未知类型的数据
    }
    // 将raw_tensors对象的张量数据移动到tensors对象中
    tensors = std::move(raw_tensors.tensors_);
  }

  // 为Storage解决ABA问题分配ID
  // --------------------------------------------------------------------------
  {
    // 计数器，用于分配唯一的版本号
    size_t counter{1};
    // 使用自定义哈希函数的哈希映射，用于跟踪版本号
    using key_t = std::pair<StorageImplData, c10::Device>;
    ska::flat_hash_map<key_t, size_t, HashCombine> versions;
    // 遍历所有张量数据
    for (auto& t : tensors) {
      // 尝试将当前张量的存储和设备作为键插入版本映射中
      auto inserted = versions.insert({{t.storage_, t.device_}, counter});
      // 如果插入成功，则增加计数器，否则保持不变
      counter += inserted.second;
      // 将分配的版本号关联到分配ID中
      t.allocation_id_ref_.get().emplace(AllocationID(inserted.first->second));
      // 如果当前张量已释放，则从版本映射中删除该条目
      if (t.is_free_) {
        versions.erase(inserted.first);
      }
    }
  }

  // 处理任何不能证明为张量存储的分配事件
  // --------------------------------------------------------------------------
  {
    // 哈希集合，用于跟踪张量的分配ID
    ska::flat_hash_set<AllocationID> tensor_set;
    // 遍历所有张量数据
    for (const auto& t : tensors) {
      // 如果当前张量的实现类型不是NoTensorImpl
      if (t.impl_ != NoTensorImpl) {
        // 将张量的分配ID插入集合中
        // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
        tensor_set.insert(*t.allocation_id_ref_.get());
      }
    }
    // 从tensors中移除那些分配ID不在tensor_set中的张量数据
    tensors.erase(
        std::remove_if(
            tensors.begin(),
            tensors.end(),
            [&tensor_set](const auto& i) {
              // 检查当前张量的分配ID是否在tensor_set中
              auto it = tensor_set.find(*i.allocation_id_ref_.get());
              return it == tensor_set.end();
            }),
        tensors.end());
  }

  // 处理TensorImpl存储发生变化的情况
  // --------------------------------------------------------------------------
  using storage_id_pair_t = std::pair<AllocationID, AllocationID>;
  // 自定义哈希函数的哈希集合，用于跟踪相同组内的存储ID对
  ska::flat_hash_set<storage_id_pair_t, HashCombine> same_group_set;
  {
    // 哈希映射，用于跟踪TensorImpl地址与其分配ID的映射关系
    ska::flat_hash_map<TensorImplAddress, AllocationID> impl_map;
  for (const auto& t : tensors) {
    // 遍历张量数组，处理每一个张量对象
    if (!t.impl_) {
      // 如果张量对象的实现为空指针，则跳过此次循环
      continue;
    }

    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    // 获取张量的分配ID（通过引用访问），并将其存储在allocation_id中
    const auto allocation_id = *t.allocation_id_ref_.get();
    // 将（t.impl_, allocation_id）插入到impl_map中，如果已存在则不插入，返回对应的迭代器it
    const auto it = impl_map.insert({t.impl_, allocation_id}).first;

    // 为了使合并步骤正常工作，需要对pair进行排序
    it->second < allocation_id
        ? same_group_set.insert({it->second, allocation_id})
        : same_group_set.insert({allocation_id, it->second});
  }



  // Coalesce groups and assign final IDs.
  // --------------------------------------------------------------------------
  // 用于存储分配ID到最终ID的映射
  ska::flat_hash_map<AllocationID, size_t> id_map;
  {
    // 存储不重复的分组对
    std::vector<storage_id_pair_t> unique_pairs;
    for (const auto& i : same_group_set) {
      // 将same_group_set中的每个元素（即分组对）添加到unique_pairs中
      unique_pairs.push_back(i);
    }
    // 对unique_pairs进行排序
    std::sort(unique_pairs.begin(), unique_pairs.end());

    // 当前分配的最终ID
    size_t current_id{0};
    // 遍历排序后的unique_pairs
    for (const auto& i : unique_pairs) {
      // 将{i.first, current_id}插入到id_map中，并记录插入是否成功
      auto inserted = id_map.insert({i.first, current_id});
      // 根据插入操作的成功与否更新current_id
      current_id += inserted.second;
      // 将{i.second, inserted.first->second}插入到id_map中
      id_map.insert({i.second, inserted.first->second});
    }
  }



  // Write back to Tensor IDs.
  // --------------------------------------------------------------------------
  // 将最终ID写回到张量对象的ID引用中
  for (const auto& t : tensors) {
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    // 获取张量的分配ID（通过引用访问），并从id_map中获取对应的最终ID
    const auto id = id_map.at(*t.allocation_id_ref_.get());
    // 将最终ID作为TensorID对象插入到张量的ID引用中
    t.id_ref_.get().emplace(TensorID(id));
  }
}

} // namespace torch::profiler::impl
```