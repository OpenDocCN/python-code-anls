# `.\pytorch\torch\csrc\jit\codegen\fuser\kernel_spec.h`

```py
#pragma once

#include <ATen/ATen.h>
#include <ATen/core/stack.h>
#include <c10/util/Optional.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/jit/codegen/fuser/arg_spec.h>
#include <torch/csrc/jit/codegen/fuser/fused_kernel.h>
#include <torch/csrc/jit/codegen/fuser/interface.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/interpreter.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace torch {
namespace jit {
namespace fuser {

// Helper struct containing partition information: the number of tensors
// created and the dimension the partitioning is performed on.
// Note: created during upfront compilation, once the tensors are known
// at runtime the partition info is logically combined with the tensor
// descriptions to create PartitionDesc objects.
struct TORCH_API PartitionInfo {
  PartitionInfo(const int64_t _nSubTensors, const int64_t _dim)
      : nSubTensors_{_nSubTensors}, dim_{_dim} {};

  // Returns the number of sub-tensors created.
  int64_t nSubTensors() const {
    return nSubTensors_;
  }
  // Returns the dimension on which partitioning is performed.
  int64_t dim() const {
    return dim_;
  }

 private:
  int64_t nSubTensors_;  // Number of sub-tensors created
  int64_t dim_;          // Dimension for partitioning
};

// "Kernel Specification." - Contains device-independent fusion information.
// Each kernel specification contains a map of instantiated generated functions
// that implement some or most of its functionality. Multiple generated
// functions are needed by each abstract specification because of different
// devices (cpu vs gpu, different gpus) and different inputs (int vs float,
// contiguous vs discontiguous).
// Note: uses a mutex to control access to its kernel store
// Note: unordered containers do not invalidate references/pointers on
//   rehashing, which is critical for thread-safety.
// TODO: allow abstract kernels to use multiple generated kernels
// TODO: allow abstract kernels to reuse generated kernels from common pool
struct TORCH_API KernelSpec {
  // Constructor for KernelSpec, initializing with a unique key and a shared_ptr to a Graph.
  KernelSpec(const int64_t _key, const std::shared_ptr<Graph>& _graph)
      : key_{_key},
        graph_{_graph},
        code_{_graph, "<fused code>"},  // Initialize the code object for fused kernel
        nInputs_{_graph->inputs().size()},  // Number of inputs to the graph
        nTensorInputs_{},  // Number of tensor inputs (to be initialized later)
        inputBroadcastGroups_{},  // Groups for input broadcasting
        inputChunks_{},  // Input chunks (to be initialized later)
        has_random_{false},  // Flag indicating if random operation (like rand_like) is present
        kernels_{} {  // Initialize kernels map
    // Iterate over nodes in the graph to determine if there are random operations
    for (const auto n : graph_->nodes()) {
      static_assert(std::is_pointer<decltype(n)>::value, "n must be a pointer");  // Assert that n is a pointer type
      if (n->kind() == aten::rand_like) {  // Check if the node kind is rand_like
        has_random_ = true;  // Set has_random_ to true if rand_like node is found
        break;  // Exit the loop early since random operation is detected
      }
    }
    nTensorInputs_ = std::count_if(
        graph_->inputs().begin(), graph_->inputs().end(), [](const Value* v) {
          return v->type()->isSubtypeOf(*TensorType::get());
        });
  }

  // Getters

  // 返回对象的键值
  int64_t key() const {
    return key_;
  }

  // 返回对象的图
  std::shared_ptr<Graph> graph() const {
    return graph_;
  }

  // 返回对象的代码
  const Code& code() const {
    return code_;
  }

  // 返回对象的输入数量
  int64_t nInputs() const {
    return nInputs_;
  }

  // 返回对象的张量输入数量
  int64_t nTensorInputs() const {
    return nTensorInputs_;
  }

  // 返回对象的输入广播组
  std::vector<std::vector<int64_t>>& inputBroadcastGroups() {
    return inputBroadcastGroups_;
  }

  // 返回对象的输入广播组（只读）
  const std::vector<std::vector<int64_t>>& inputBroadcastGroups() const {
    return inputBroadcastGroups_;
  }

  // 返回对象的输入分块信息
  std::vector<PartitionInfo>& inputChunks() {
    return inputChunks_;
  }

  // 返回对象的输入分块信息（只读）
  const std::vector<PartitionInfo>& inputChunks() const {
    return inputChunks_;
  }

  // 返回对象是否包含随机数
  bool hasRandom() const {
    return has_random_;
  }

  // 缓存相关函数

  // 根据参数规范查找融合内核
  std::optional<std::shared_ptr<FusedKernel>> findKernel(
      const ArgSpec& arg_spec) const {
    std::lock_guard<std::mutex> guard{mutex_};
    const auto it = kernels_.find(arg_spec);
    if (it == kernels_.end())
      return c10::nullopt;
    return it->second;
  }

  // 缓存内核
  void cacheKernel(const ArgSpec& arg_spec, std::shared_ptr<FusedKernel> kernel)
      const {
    std::lock_guard<std::mutex> guard{mutex_};
    kernels_.emplace(arg_spec, kernel);
  }

 private:
  int64_t key_;
  std::shared_ptr<Graph> graph_;
  Code code_;
  uint64_t nInputs_;
  uint64_t nTensorInputs_;
  std::vector<std::vector<int64_t>> inputBroadcastGroups_;
  std::vector<PartitionInfo> inputChunks_;
  bool has_random_;
  mutable std::mutex mutex_;
  mutable std::
      unordered_map<ArgSpec, std::shared_ptr<FusedKernel>, c10::hash<ArgSpec>>
          kernels_;
};

} // namespace fuser
} // namespace jit
} // namespace torch
```