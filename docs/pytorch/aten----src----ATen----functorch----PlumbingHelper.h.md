# `.\pytorch\aten\src\ATen\functorch\PlumbingHelper.h`

```py
// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
#pragma once
#include <ATen/Tensor.h>
#include <ATen/functorch/BatchedTensorImpl.h>
#include <ATen/functorch/DynamicLayer.h>

// NOTE: [vmap plumbing]
//
// Here's how "batching rules" work.
// - we register kernels to the Batched key
// - these kernels have the same signatures as the original operators.
//   For example, at::sin(Tensor self) accepts a Tensor, and the batched kernel
//   must also accept a Tensor
// - However, it is more natural for users to write a batching rule like the
//   following: sin_batch_rule(Tensor self, optional<int> self_bdim)
// - There is some codegenerated layer (the "plumbing") that wraps the user
//   defined batching rule (e.g. sin_batch_rule) in a kernel that can be
//   registered to the Batched key.
//
// The plumbing is responsible for wrapping a batching rule into a form that may
// be registered as the kernel for the batched key.

// 匿名命名空间，定义了 at::functorch 命名空间下的一些功能函数和类
namespace at::functorch {

// 检查动态层是否逃逸，给定层和描述信息
void vmap_check_escaped(const optional<DynamicLayer> &layer, const char* what);

// 创建一个 BatchedTensor，给定一个张量、bdim（批处理维度）和层级
TORCH_API Tensor makeBatched(const Tensor& tensor, optional<int64_t> bdim, int64_t level);

// 给定一个可能是 BatchedTensor 的张量，解包它
// 如果 tensor 不是 BatchedTensor，或者是 BatchedTensor 但层级不匹配，则返回 (tensor, nullopt)
// 否则返回 (unwrap(tensor), bdim)
TORCH_API std::tuple<Tensor, std::optional<int64_t>> unwrapTensorAtLevel(const Tensor& tensor, int64_t level);

// 创建一个 BatchedTensor 向量
TORCH_API std::vector<Tensor> makeBatchedVector(const std::vector<Tensor>& tensors, optional<int64_t> bdim, int64_t level);

// 返回是否有任何张量在给定层级上进行了批处理
TORCH_API bool isBatchedAtLevel(ITensorListRef tensors, int64_t level);
TORCH_API bool isBatchedAtLevel(const c10::List<std::optional<Tensor>>& maybe_tensors, int64_t level);
TORCH_API bool isBatchedAtLevel(const Tensor& tensor, int64_t level);
TORCH_API bool isBatchedAtLevel(const std::optional<Tensor>& maybe_tensor, int64_t level);

// 方便的辅助函数，如果任何张量在给定层级上进行了批处理则返回 true
TORCH_API bool areAnyBatchedAtLevel(ArrayRef<optional<Tensor>> maybe_tensors, int64_t level);

// 对于 IValue，返回它是否参与当前层级的批处理
inline bool ivalueParticipatesInCurrentLevel(const IValue& ivalue) {
  // 如果 IValue 是张量，则获取当前动态层的层级，并检查张量是否在该层级上进行了批处理
  if (ivalue.isTensor()) {
    auto maybe_level = maybeCurrentDynamicLayer();
    TORCH_INTERNAL_ASSERT(maybe_level.has_value());
    auto current_level = maybe_level->layerId();
    return isBatchedAtLevel(ivalue.toTensor(), current_level);
  }
  // TODO: 应该真正检查这一点
  return false;
}

} // namespace at::functorch
```