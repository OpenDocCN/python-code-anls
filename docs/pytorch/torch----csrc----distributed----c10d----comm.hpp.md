# `.\pytorch\torch\csrc\distributed\c10d\comm.hpp`

```
#pragma once

#include <ATen/ATen.h>
#include <ATen/core/ivalue.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <utility>

namespace c10d {

// Broadcast many tensors to all processes in the process group.
TORCH_API void broadcast_coalesced(
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
    at::TensorList tensors,
    size_t buffer_size,
    int rank = 0);

// This class passes bucket contents tensor to DDP communication hook.
class TORCH_API GradBucket {
 public:
  // Constructor to initialize GradBucket object.
  explicit GradBucket(
      size_t index,
      size_t bucket_count,
      at::Tensor tensor,
      std::vector<size_t> offsets,
      std::vector<size_t> lengths,
      std::vector<c10::IntArrayRef> sizes_vec,
      std::vector<at::Tensor> parameters,
      std::optional<at::Tensor> sparse_grad_indices)
      : index_(index),
        bucket_count_(bucket_count),
        buffer_(std::move(tensor)),
        offsets_(std::move(offsets)),
        lengths_(std::move(lengths)),
        sizes_vec_(std::move(sizes_vec)),
        parameters_(std::move(parameters)),
        sparse_grad_indices_(std::move(sparse_grad_indices)) {}

  // Returns the index of the bucket, which is unique across all the buckets.
  size_t getIndex() const {
    return index_;
  }

  // Returns the buffer tensor containing gradient data.
  const at::Tensor& getBuffer() const {
    return buffer_;
  }

  // Returns a mutable reference to the buffer tensor.
  at::Tensor& getBufferRef() {
    return buffer_;
  }

  // Sets the buffer tensor to a new tensor provided.
  void setBuffer(at::Tensor& buffer) {
    buffer_ = buffer;
  }

  // Returns the list of gradient tensors corresponding to parameters.
  std::vector<at::Tensor> getGradients() const;

  // Returns model parameters belonging to this bucket.
  // These are returned in the same order as gradient tensors.
  const std::vector<at::Tensor> getParameters() const {
    return parameters_;
  }

  // Checks if this bucket is the last to allreduce in an iteration.
  bool isLast() const {
    return index_ == bucket_count_ - 1;
  }

  // Returns optional sparse gradient indices for sparse tensors.
  std::optional<at::Tensor>& getSparseGradIndices() {
    return sparse_grad_indices_;
  }

 private:
  size_t index_;
  size_t bucket_count_;
  at::Tensor buffer_;

  // Offsets and lengths for variables in buffer_.
  std::vector<size_t> offsets_;
  std::vector<size_t> lengths_;
  std::vector<c10::IntArrayRef> sizes_vec_;

  // Model parameters for this bucket.
  const std::vector<at::Tensor> parameters_;

  // Optional tensor for predefined sparse indices (sparse tensors only).
  std::optional<at::Tensor> sparse_grad_indices_;
};

// Base class of both `PythonCommHook` and `CppCommHook`.
// Requires implementing 1) `runHook` method that communicates gradients
// 异步地传递输入梯度桶给注册的通信钩子。
// 一旦桶中的张量准备好，异步启动钩子，并返回一个持有通信结果的 future。
virtual c10::intrusive_ptr<c10::ivalue::Future> runHook(
    GradBucket& bucket) = 0;

// 当通信钩子的结果准备好时返回结果张量。
// 结果张量将被复制到各个参数的梯度中。
virtual at::Tensor parseHookResult(const c10::IValue& result) = 0;

// 该辅助函数由下面的 CppCommHookInterface 和 reducer 内部调用。
// 用于解析 C++ 通信钩子的结果。
TORCH_API at::Tensor parseCppCommHookResult(const c10::IValue& result);

// 这个 CppCommHook 接口只要求实现 runHook 方法，该方法可能使用一个状态。
template <typename T>
class CppCommHookInterface : public CommHookInterface {
 public:
  explicit CppCommHookInterface(T state) : state_(std::move(state)) {}

  ~CppCommHookInterface() override = default;

  // 实现基类的虚函数，将钩子结果解析为张量。
  at::Tensor parseHookResult(const c10::IValue& result) override {
    return detail::parseCppCommHookResult(result);
  }

 protected:
  T state_;
};
```