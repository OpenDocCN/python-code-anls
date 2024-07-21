# `.\pytorch\aten\src\ATen\functorch\BatchRulesViews.cpp`

```
// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/functorch/BatchRulesHelper.h>
#include <utility>

#include <ATen/Operators.h>
#include <ATen/functorch/PlumbingHelper.h>
#include <ATen/functorch/BatchedFallback.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/TensorBody.h>
#include <c10/core/SymIntArrayRef.h>
#include <c10/util/SmallBuffer.h>
#include <ATen/InferSize.h>

namespace at::functorch {

// Note [Adding vmap support for an operator]
// Hey there! So you have an operator and you want to get it to work with vmap.
// For example, let's say you just invented the `sum.int` operator and want to make
// it so that the following works.
// >>> tensor = torch.randn(B, 3)
// >>> vmap(torch.sum, (0, None))(tensor, 0)` works
// There are three main ways to do so.
//
// Note [Writing batch rule for out-of-place operators]
// If your operator is out-of-place, you can write a batch rule for it.
// The batch rule defines how to perform the operator on inputs where each
// Tensor input may have an additional dimension that is being vmapped over.
// We refer to this dimension as the *batch dimension* or bdim for short.
//
// For example, let's consider writing a batch rule for
// `Tensor sum(const Tensor& self, int64_t dim)`. The signature of the
// batch rule has an additional optional<int64_t> argument after each
// Tensor argument and return. So, in this case, the batch rule has signature
//   tuple<Tensor,optional<int64_t>> sum_batch_rule(
//       const Tensor& self, optional<int64_t> self_bdim, int64_t dim);
//
// The vmap call above invokes the batch rule with `self = tensor`,
// `self_bdim = 0`, and `dim = 0`. Note that there are **no BatchedTensors**
// involved in this case; there exists some plumbing that automatically unwraps
// BatchedTensors before calling the batch rule.
//
// To write the logic of the batch rule: think about the semantics of the
// `sum` operation if `self` had an additional dimension (indicated by self_bdim):
// - If `self_bdim` is null, then we just do `result = self.sum(dim)` as usual
// - If `self_bdim` is not-null, then we need to modify `dim`. `dim` is equal
//   to whatever the user passed in (0 in this case), but we should actually
//   perform the reduction over dimension 1 and do `result = self.sum(1)`
//   because dim 0 is being vmapped over.
// Finally, we return the result as well as a new bdim
// - If `self_bdim` is null, then there's no batch dim in the result.
// - If `self_bdim` is not-null, then we return where the bdim is.
//   Since we invoked `result = self.sum(1)`, the bdim is still at dim 0.
//
// Now that we have written `sum_batch_rule`, we have to register it inside a
// TORCH_LIBRARY_IMPL block:
//   TORCH_LIBRARY_IMPL(aten, FuncTorchBatched, m) {
//     ...
}
// 实现 unsqueeze 操作的批处理规则函数
std::tuple<Tensor,optional<int64_t>> unsqueeze_batch_rule(
    const Tensor& self,                     // 输入张量
    optional<int64_t> self_bdim,            // 可选的输入批处理维度
    int64_t dim) {                          // 维度参数
  auto self_ = moveBatchDimToFront(self, self_bdim);  // 将批处理维度移动到张量维度的最前面
  auto rank = rankWithoutBatchDim(self, self_bdim);   // 计算不包含批处理维度的张量秩
  dim = maybe_wrap_dim(dim, rank + 1) + 1;             // 确保维度参数在有效范围内
  return std::make_tuple(self_.unsqueeze(dim), 0);     // 返回unsqueeze后的张量和标志位0
}

// 实现 repeat 操作的批处理规则函数
std::tuple<Tensor,optional<int64_t>> repeat_batch_rule(
    const Tensor& self,                             // 输入张量
    optional<int64_t> self_bdim,                    // 可选的输入批处理维度
    c10::SymIntArrayRef sizes) {                    // 符号整数数组引用
  SymDimVector sizes_with_bdim = { sizes.begin(), sizes.end() };  // 创建包含批处理维度的大小向量
  sizes_with_bdim.insert(sizes_with_bdim.begin(), 1);              // 在向量最前面插入维度1
  auto self_ = moveBatchDimToFront(self, self_bdim);                // 将批处理维度移动到张量维度的最前面
  while (self_.dim() < (int64_t)sizes_with_bdim.size()) {           // 确保张量维度足够大
    self_ = self_.unsqueeze(1);                                     // 如果不够大，就在维度1上unsqueeze
  }
  return std::make_tuple(self_.repeat_symint(sizes_with_bdim), 0);  // 返回repeat操作后的张量和标志位0
}

// 实现 _unsafe_view 操作的批处理规则函数
std::tuple<Tensor,optional<int64_t>> _unsafe_view_batch_rule(
    const Tensor& self,                             // 输入张量
    optional<int64_t> self_bdim,                    // 可选的输入批处理维度
    c10::SymIntArrayRef size) {                     // 符号整数数组引用
  auto self_ = moveBatchDimToFront(self, self_bdim); // 将批处理维度移动到张量维度的最前面
  SymDimVector view_size(size);                     // 创建视图大小向量
  view_size.insert(view_size.begin(), self_.sym_size(0));  // 在向量最前面插入第一个符号大小

  // 检查视图是否有效。如果无效，则复制张量以确保有效性。
  // 允许复制是因为 _unsafe_view(x) 保证不再使用 x。
  const at::SymDimVector inferred_size = at::infer_size_dv(view_size, self_.sym_numel());  // 推断出的大小向量
  const auto stride = at::detail::computeStride(self_.sym_sizes(),  // 计算步长
                                                self_.sym_strides(),
                                                inferred_size);
  if (!stride.has_value()) {                        // 如果步长无效
    self_ = self_.contiguous();                     // 则将张量变为连续的
  }
  return std::make_tuple(at::_unsafe_view_symint(self_, view_size), 0);  // 返回_unsafe_view操作后的张量和标志位0
}
std::tuple<Tensor,optional<int64_t>> flip_batch_rule(const Tensor& self, optional<int64_t> self_bdim, IntArrayRef dims) {
  // 将 batch 维度移到张量的最前面
  auto self_ = moveBatchDimToFront(self, self_bdim);
  // 存储新的维度索引
  VmapDimVector new_dims;
  // 遍历传入的维度列表，获取物理维度并存储
  for (auto i: dims) {
    new_dims.push_back(getPhysicalDim(self_, true, i));
  }
  // 返回翻转后的张量和一个虚拟的整数0
  return std::make_tuple(at::flip(self_, new_dims), 0);
}

const Tensor& resize__plumbing(
    const Tensor& self,
    IntArrayRef size,
    std::optional<MemoryFormat> optional_memory_format) {
  // 检查是否需要调整内存格式
  TORCH_CHECK(
      !optional_memory_format.has_value() ||
      optional_memory_format == c10::MemoryFormat::Contiguous,
      "resize_: batching rule only supports None or Contiguous MemoryFormat");
  // 获取当前动态层
  auto maybe_layer = maybeCurrentDynamicLayer();
  // 检查是否逃逸了当前层
  vmap_check_escaped(maybe_layer, "resize__plumbing");
  // 获取当前层级的层ID
  int64_t cur_level = maybe_layer->layerId();
  // 如果张量在当前层级没有被批处理，则直接调用原始的 resize_ 方法
  if (!isBatchedAtLevel(self, cur_level)) {
    c10::impl::ExcludeDispatchKeyGuard guard2(DispatchKey::FuncTorchBatched);
    return self.resize_(size, optional_memory_format);
  }

  // 否则，对批处理的张量执行 resize_ 操作
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  TORCH_INTERNAL_ASSERT(self_bdim.has_value());

  // TODO: 下面的算法仅适用于批处理维度为0的情况。
  // 如果要适用于其他情况，需要能够修改 BatchedTensorImpl 的 BatchDims 属性。
  TORCH_INTERNAL_ASSERT(self_bdim.value() == 0, "NYI: resize_ batch rule for batch dim != 0");

  // 移动批处理维度到张量的最前面
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  self_value = moveBatchDimToFront(self_value, self_bdim);
  // 构建新的大小向量，将当前批处理维度的大小插入到最前面
  VmapDimVector new_size(size);
  new_size.insert(new_size.begin(), self_value.size(*self_bdim));
  // 调整张量的大小
  self_value.resize_(new_size);

  // 更新包装器的大小和步长信息
  auto* batched = maybeGetBatchedImpl(self);
  TORCH_INTERNAL_ASSERT(batched);
  batched->refreshTensorMetadata();

  return self;
}

std::tuple<Tensor, optional<int64_t>> squeeze_batch_rule(const Tensor& self, optional<int64_t> bdim) {
  // 断言批处理维度的值存在
  TORCH_INTERNAL_ASSERT(bdim.has_value());
  // 对于维度为1的标量数组，复制 PyTorch 的行为
  if (self.dim() == 1) {
    return std::make_tuple(self.alias(), bdim);
  }

  // 手动计算输出形状，省略所有大小为1的维度，并跟踪批处理索引的位置变化。
  auto shape = self.sym_sizes();
  SymDimVector squeezed_sizes;
  bool before_batch_idx = true;
  int64_t new_batch_idx = 0;
  int64_t original_idx = 0;

  for (const auto& it : shape) {
    // 保留大小不为1的维度以及批处理维度（无论大小如何）
    if (it != 1 || original_idx == bdim) {
      squeezed_sizes.push_back(it);
      if (original_idx == bdim) {
        before_batch_idx = false;
      }
      // 仅对将保留在输出中的维度进行计数
      if (before_batch_idx) {
        ++new_batch_idx;
      }
    }
    // 增加原始索引值
    ++original_idx;
  }

  // 使用 self 对象的 view_symint 方法，传入 squeezed_sizes 参数，获取结果
  auto result = self.view_symint(squeezed_sizes);
  // 创建包含 result 和新批次索引值的元组，并返回
  return std::make_tuple(std::move(result), std::optional<int64_t>(new_batch_idx));
}

// 定义 squeeze_dims_batch_rule 函数，处理张量的维度挤压操作
std::tuple<Tensor, optional<int64_t>> squeeze_dims_batch_rule(
    const Tensor& self, optional<int64_t> bdim, IntArrayRef dims) {
  TORCH_INTERNAL_ASSERT(bdim.has_value());
  // 对于一维张量的特殊情况，复制 PyTorch 的行为。
  auto ndim = self.dim();
  if (ndim == 1) {
    TORCH_CHECK(
        dims.empty() || (dims.size() == 1 && dims[0] == 0),
        "Dimension is out of range (expected to be in range of [-1, 0], but got ", dims);
    return std::make_tuple(self.alias(), bdim);
  }

  // 调整高于批次维度的任何维度
  DimVector adjusted_dims(dims.begin(), dims.end());
  int64_t updated_batch_idx = *bdim;
  for (auto &d : adjusted_dims) {
    auto actual_dim = c10::maybe_wrap_dim(d, ndim - 1);
    if (actual_dim < *bdim) {
      d = actual_dim;
      if (self.sym_size(actual_dim) == 1) {
        // 在批次维度之前的列将被丢弃，因此需要相应地调整。
        --updated_batch_idx;
      }
    } else {
      // 由于要挤压的维度在批次维度之后，增加一以补偿原始批次维度。
      d = actual_dim + 1;
    }
  }
  return std::make_tuple(self.squeeze(adjusted_dims), optional<int64_t>(updated_batch_idx));
}

// 定义 squeeze_dim_batch_rule 函数，调用 squeeze_dims_batch_rule 处理单个维度挤压操作
std::tuple<Tensor, optional<int64_t>> squeeze_dim_batch_rule(
    const Tensor& self, optional<int64_t> bdim, int64_t dim) {
  return squeeze_dims_batch_rule(self, bdim, {dim});
}

// 定义 select_batching_rule 函数，处理选择操作的批处理规则
std::tuple<Tensor, optional<int64_t>> select_batching_rule(const Tensor& self, optional<int64_t> bdim, int64_t dim, c10::SymInt index) {
  if (!bdim) {
    return std::make_tuple(self.select_symint(dim, std::move(index)), nullopt);
  }

  auto _self = moveBatchDimToFront(self, bdim);
  auto dim_physical = getPhysicalDim(_self, true, dim);
  auto result = _self.select_symint(dim_physical, std::move(index));
  return std::make_tuple(std::move(result), 0);
}

// 定义 _reshape_alias_batch_rule 函数，处理带有符号整数数组的重塑别名的批处理规则
std::tuple<Tensor, optional<int64_t>> _reshape_alias_batch_rule(const Tensor& self, optional<int64_t> bdim, const c10::SymIntArrayRef shape, const c10::SymIntArrayRef strides) {
  (void) strides;
  TORCH_INTERNAL_ASSERT(bdim.has_value());

  auto self_ = moveBatchDimToFront(self, bdim);
  c10::SymDimVector new_shape(shape.size() + 1);
  new_shape[0] = self_.sym_size(0);
  std::copy(shape.begin(), shape.end(), new_shape.begin() + 1);
  return std::make_tuple(at::reshape_symint(self_, new_shape), 0);
}

// 定义 roll_batch_rule 函数，处理滚动操作的批处理规则
std::tuple<Tensor, optional<int64_t>> roll_batch_rule(const Tensor& self, optional<int64_t> bdim, SymIntArrayRef shifts, IntArrayRef dims) {
  TORCH_INTERNAL_ASSERT(bdim.has_value());

  auto self_ = moveBatchDimToFront(self, bdim);
  VmapDimVector new_dims;
  if (!dims.empty()) {
    for (auto i: dims) {
      new_dims.push_back(getPhysicalDim(self, true, i));
    }
    // 调用 at::roll_symint 函数对 self_ 张量进行轴向滚动操作，并将结果与整数 0 组成元组返回
    return std::make_tuple(at::roll_symint(self_, shifts, new_dims), 0);
  }
  // 我们将执行类似以下操作：t.reshape(a, -1).roll(1, dims=[1, ]).reshape(old_shape)
  // 获取原始张量的形状，保存在 old_shape 变量中
  auto old_shape = self_.sym_sizes();
  // 向 new_dims 向量添加一个维度索引 1
  new_dims.push_back(1);
  // 计算去除批量维度后的逻辑秩（rankWithoutBatchDim 函数的返回值）
  auto logical_rank = rankWithoutBatchDim(self, bdim);
  // 如果逻辑秩为 0，则对 self_ 张量在维度 0 上进行 unsqueeze 操作
  if (logical_rank == 0) {
    self_ = self_.unsqueeze(0);
  }

  // 将 self_ 张量按照维度 1 展平，并调用 at::roll_symint 函数进行轴向滚动操作
  auto output = at::roll_symint(self_.flatten(1), shifts, new_dims);
  // 注意：对于标量张量，不需要 unsqueeze 操作，因为使用 `old_shape` 参数的 reshape_symint 函数会处理
  output = output.reshape_symint(old_shape);
  // 返回包含 output 和整数 0 的元组
  return std::make_tuple(output, 0);
}

// 对角线操作的批处理规则，返回操作后的结果张量和可选的批处理维度
std::tuple<Tensor, optional<int64_t>> diagonal_batching_rule(
    const Tensor &self, optional<int64_t> self_bdim,
    int64_t offset, int64_t dim1, int64_t dim2)
{
  // 计算没有批处理维度时的张量秩
  auto logical_rank = rankWithoutBatchDim(self, self_bdim);
  // 将批处理维度移到张量的最前面
  auto self_ = moveBatchDimToFront(self, self_bdim);
  // 对 dim1 和 dim2 进行边界处理，使其在合法的物理维度范围内
  auto dim1_ = maybe_wrap_dim(dim1, logical_rank) + 1;
  auto dim2_ = maybe_wrap_dim(dim2, logical_rank) + 1;
  // 调用 PyTorch 的对角线操作函数
  auto result = at::diagonal(self_, offset, dim1_, dim2_);
  // 返回操作后的结果张量和一个零值的元组
  return std::make_tuple(std::move(result), 0);
}

// 对角线反向操作的批处理规则，返回操作后的结果张量和可选的批处理维度
std::tuple<Tensor, optional<int64_t>> diagonal_backward_batch_rule(
    const Tensor& grad_input, optional<int64_t> grad_input_bdim,
    c10::SymIntArrayRef input_sizes, int64_t offset, int64_t dim1, int64_t dim2) {
  // 计算没有批处理维度时的张量秩
  auto logical_rank = rankWithoutBatchDim(grad_input, grad_input_bdim);
  // 将批处理维度移到张量的最前面
  auto grad_input_ = moveBatchDimToFront(grad_input, grad_input_bdim);
  // 对 dim1 和 dim2 进行边界处理，使其在合法的物理维度范围内
  dim1 = maybe_wrap_dim(dim1, logical_rank + 1) + 1;
  dim2 = maybe_wrap_dim(dim2, logical_rank + 1) + 1;
  // 创建一个新的尺寸向量，用于反向对角线操作
  c10::SymDimVector input_sizes_(input_sizes.size() + 1);
  input_sizes_[0] = grad_input_.size(0);
  std::copy(input_sizes.begin(), input_sizes.end(), input_sizes_.begin() + 1);
  // 调用 PyTorch 的对角线反向操作函数
  auto result = at::diagonal_backward_symint(grad_input_, input_sizes_, offset, dim1, dim2);
  // 返回操作后的结果张量和一个零值的元组
  return std::make_tuple(std::move(result), 0);
}

// 切片操作的批处理规则，返回操作后的结果张量和可选的批处理维度
std::tuple<Tensor, optional<int64_t>> slice_batch_rule(
    const Tensor& self,
    optional<int64_t> self_bdim,
    int64_t dim,
    std::optional<c10::SymInt> start,
    std::optional<c10::SymInt> end,
    c10::SymInt step) {
  // 将批处理维度移到张量的最前面
  auto self_ = moveBatchDimToFront(self, self_bdim);
  // 获取物理维度上的实际维度索引
  dim = getPhysicalDim(self, self_bdim.has_value(), dim);

  // 调用 PyTorch 的符号整数切片操作函数
  auto result = self_.slice_symint(dim, std::move(start), std::move(end), std::move(step));
  // 返回操作后的结果张量和一个零值的元组
  return std::make_tuple(std::move(result), 0);
}

// 检查在标量张量上是否允许进行转置操作的特殊情况
static bool is_allowed_dim_on_scalar_tensor(int64_t dim) {
  return dim == 0 || dim == -1;
}

// 转置操作的批处理规则，返回操作后的结果张量和可选的批处理维度
std::tuple<Tensor,optional<int64_t>>
transpose_int_batch_rule(
    const Tensor& self,
    optional<int64_t> self_bdim,
    int64_t dim0,
    int64_t dim1) {
  // 如果张量是一维且转置维度在 {0, -1} 中，则直接返回原张量和批处理维度
  if (/*physical*/self.dim() == 1 && is_allowed_dim_on_scalar_tensor(dim0) &&
      is_allowed_dim_on_scalar_tensor(dim1)) {
    return std::make_tuple(self, self_bdim);
  }
  // 将批处理维度移到张量的最前面
  auto self_ = moveBatchDimToFront(self, self_bdim);
  // 获取物理维度上的实际维度索引
  dim0 = getPhysicalDim(self, self_bdim.has_value(), dim0);
  dim1 = getPhysicalDim(self, self_bdim.has_value(), dim1);
  // 调用 PyTorch 的转置操作函数
  auto result = self_.transpose(dim0, dim1);
  // 返回操作后的结果张量和一个零值的元组
  return std::make_tuple(std::move(result), 0);
}

// 排列操作的批处理规则，返回操作后的结果张量和可选的批处理维度
std::tuple<Tensor, optional<int64_t>> permute_batching_rule(
    const Tensor &self, optional<int64_t> self_bdim, IntArrayRef dims)
{
  // 如果没有指定批处理维度，直接返回原张量和批处理维度
  if (!self_bdim.has_value()) {

  // 如果没有指定批处理维度，直接返回原张量和批处理维度
  return std::make_tuple(self, self_bdim);
}
    return std::make_tuple(self.permute(dims), self_bdim);

返回一个元组，包含通过给定维度排列后的 self 对象和 self_bdim 变量。


  }

  auto self_ = moveBatchDimToFront(self, self_bdim);

将 self 对象中的批量维度（如果存在的话）移动到最前面，并将结果保存在 self_ 变量中。


  VmapDimVector dims_;
  dims_.reserve(dims.size() + 1);

声明一个名为 dims_ 的向量，用于存储维度信息，预留空间以容纳 dims.size() + 1 个元素。


  dims_.emplace_back(0);

将整数 0 添加到 dims_ 向量的末尾。


  for (auto dim : dims) {
    dims_.emplace_back(getPhysicalDim(self_, self_bdim.has_value(), dim));
  }

对于 dims 向量中的每个维度 dim，调用 getPhysicalDim 函数计算 self_ 对象中 dim 维度的物理维度，并将结果添加到 dims_ 向量的末尾。


  return std::make_tuple(self_.permute(dims_), 0);

返回一个元组，包含通过给定维度排列后的 self_ 对象和整数 0。
}

// 定义模板函数 expand_batch_rule，接受参数为 Tensor、可选的整数 self_bdim、SymIntArrayRef size、布尔值 implicit
template <typename F, F Func>
std::tuple<Tensor, optional<int64_t>> expand_batch_rule(
    const Tensor &self, optional<int64_t> self_bdim, SymIntArrayRef size, bool implicit)
{
  // 将 self 的批处理维度移至最前
  auto self_ = moveBatchDimToFront(self, self_bdim);
  // 创建一个大小为 size.size() + 1 的 SymDimVector
  SymDimVector view_size(size.size() + 1);
  // 将 self_ 的第一个维度大小复制到 view_size 的第一个位置
  view_size[0] = self_.size(0);
  // 将 size 中的尺寸复制到 view_size 的其余位置
  std::copy(size.cbegin(), size.cend(), view_size.begin() + 1);

  // 调用 at::view_copy_symint 函数，返回新的 Tensor，并将 0 作为第二个元素组成元组返回
  return std::make_tuple(at::view_copy_symint(self_, view_size), 0);
}
{
  // 获取当前张量的维度
  auto self_dim = self.dim();
  // 检查提供的尺寸数量是否足够大，以便扩展张量的维度
  TORCH_CHECK(static_cast<uint64_t>(self_dim - 1) <= size.size(),
              "expand: the number of sizes provided (", size.size(), ") ",
              "must be greater or equal to the number of dimensions in the tensor (", static_cast<uint64_t>(self_dim - 1), ")");

  // 将批量维度移动到张量维度的最前面
  auto self_ = moveBatchDimToFront(self, self_bdim);
  // 获取移动后的张量的符号尺寸
  auto self_sizes = self_.sym_sizes();
  // 获取批量维度的大小
  const auto& batch_size = self_sizes[0];

  // 创建一个新的尺寸向量，包括批量维度
  c10::SmallVector<c10::SymInt> size_(size.size() + 1);
  size_[0] = batch_size;
  std::copy(size.cbegin(), size.cend(), size_.begin() + 1);

  // 这里我们知道我们要将（逻辑上的）张量扩展到更多的维度。
  // 我们必须小心，因为由于存在批量维度，我们不能直接调用扩展函数。
  //
  // 举个例子，假设 B0 是一个批量维度，考虑 expand(Tensor[B0, 3], [2, 3])。
  // 结果应该是一个尺寸为 [B0, 2, 3] 的张量。
  // 尺寸为 [B0, 3] 的物理视图不能直接扩展为尺寸为 [B0, 2, 3]，
  // 因此这里的策略是首先将其视为尺寸为 [B0, 1, 3] 的张量，然后再扩展。
  auto extra_dims = size.size() - (self_dim - 1);
  // 创建一个视图形状的尺寸向量，初始值为 1
  c10::SmallVector<c10::SymInt> view_shape(size_.size(), /*init_value*/1);
  view_shape[0] = batch_size;
  std::copy(self_sizes.cbegin() + 1, self_sizes.cend(),
            view_shape.begin() + 1 + extra_dims);

  // 返回结果，包括批量维度的处理
  return std::make_tuple(Func(self_.view_symint(view_shape), size_, implicit), 0);
}
    // 断言确保 self_bdim 值存在，即批量维度不为空
    TORCH_INTERNAL_ASSERT(self_bdim.has_value());
    // 将 self 张量的批量维度移动到最前面，返回移动后的张量
    auto self_ = moveBatchDimToFront(self, self_bdim);
    // 计算移除批量维度后的张量的逻辑秩（维度数）
    auto logical_rank = rankWithoutBatchDim(self, self_bdim);
    // 将 dim 修正为在逻辑秩范围内的有效维度索引（加上1是为了从1开始计数）
    dim = maybe_wrap_dim(dim, logical_rank) + 1;
    // 对 self_ 张量进行不安全的按对称整数分割，返回分割后的结果
    auto result = self_.unsafe_split_symint(std::move(split_size), dim);
    // 返回包含分割结果和0的元组
    return std::make_tuple(std::move(result), 0);
}

// 定义 diag_embed_batch_rule 函数，接受一个张量和可选的批维度参数，并返回一个元组
std::tuple<Tensor, optional<int64_t>> diag_embed_batch_rule(const Tensor& self, optional<int64_t> self_bdim, int64_t offset, int64_t dim1, int64_t dim2) {
  // 计算没有批维度时的逻辑秩
  auto logical_rank = rankWithoutBatchDim(self, self_bdim);
  // 将批维度移动到张量的最前面
  auto self_ = moveBatchDimToFront(self, self_bdim);
  // 确保维度 dim1 和 dim2 在逻辑秩的基础上加一，并进行偏移
  dim1 = maybe_wrap_dim(dim1, logical_rank + 1) + 1;
  dim2 = maybe_wrap_dim(dim2, logical_rank + 1) + 1;
  // 调用 ATen 的 diag_embed 函数生成对角张量，返回结果和标志 0
  return std::make_tuple(at::diag_embed(self_, offset, dim1, dim2), 0);
}

// 定义 trace_decomp 函数，接受一个张量，检查其是否为二维矩阵并返回其对角线元素的和
Tensor trace_decomp(const Tensor& tensor) {
  // 检查张量是否为二维矩阵，否则抛出错误信息
  TORCH_CHECK(tensor.dim() == 2, "trace: expected a matrix, but got tensor with dim ", tensor.dim());
  // 返回张量对角线元素的和
  return tensor.diagonal().sum();
}

// 定义 tril_batch_rule 函数，接受一个张量和可选的批维度参数，以及一个对角线参数
std::tuple<Tensor,optional<int64_t>> tril_batch_rule(
    const Tensor& self,
    optional<int64_t> self_bdim,
    int64_t diagonal = 0) {
  // 检查张量至少有两个维度，否则抛出错误信息
  TORCH_CHECK(self.dim() >= 2, "tril: The input tensor must have at least 2 dimensions.");
  // 将批维度移动到张量的最前面
  auto self_ = moveBatchDimToFront(self, self_bdim);
  // 调用 ATen 的 tril 函数生成下三角矩阵，返回结果和标志 0
  auto result = at::tril(self_, diagonal);
  return std::make_tuple(std::move(result), 0);
}

// 定义 triu_batch_rule 函数，接受一个张量和可选的批维度参数，以及一个对角线参数
std::tuple<Tensor,optional<int64_t>> triu_batch_rule(
    const Tensor& self,
    optional<int64_t> self_bdim,
    int64_t diagonal = 0) {
  // 检查张量至少有两个维度，否则抛出错误信息
  TORCH_CHECK(self.dim() >= 2, "triu: The input tensor must have at least 2 dimensions.");
  // 将批维度移动到张量的最前面
  auto self_ = moveBatchDimToFront(self, self_bdim);
  // 调用 ATen 的 triu 函数生成上三角矩阵，返回结果和标志 0
  auto result = at::triu(self_, diagonal);
  return std::make_tuple(std::move(result), 0);
}

}
# 实现 ATen 库中 aten 命名空间下的 TORCH_LIBRARY_IMPL 宏，定义了一系列针对批处理的函数实现
TORCH_LIBRARY_IMPL(aten, FuncTorchBatched, m) {
    # 启用对 flip 函数的批处理支持，使用 flip_batch_rule 规则
    VMAP_SUPPORT(flip, flip_batch_rule);
    # 将 trace 函数映射到 trace_decomp 实现
    m.impl("trace", trace_decomp);
    # 启用对 tril 函数的批处理支持，使用 tril_batch_rule 规则
    VMAP_SUPPORT(tril, tril_batch_rule);
    # 启用对 triu 函数的批处理支持，使用 triu_batch_rule 规则
    VMAP_SUPPORT(triu, triu_batch_rule);
    # 启用对 repeat 函数的批处理支持，使用 repeat_batch_rule 规则
    VMAP_SUPPORT(repeat, repeat_batch_rule);
    # 启用对 _unsafe_view 函数的批处理支持，使用 _unsafe_view_batch_rule 规则
    VMAP_SUPPORT(_unsafe_view, _unsafe_view_batch_rule);
    # 启用对 unsqueeze 函数的批处理支持，使用 unsqueeze_batch_rule 规则
    VMAP_SUPPORT(unsqueeze, unsqueeze_batch_rule);
    # 将 resize_ 函数映射到 resize__plumbing 实现
    m.impl("resize_", resize__plumbing);
    # 启用对 select 函数的批处理支持，参数为整数，使用 select_batching_rule 规则
    VMAP_SUPPORT2(select, int, select_batching_rule);
    # 启用对 squeeze 函数的批处理支持，使用 squeeze_batch_rule 规则
    VMAP_SUPPORT(squeeze, squeeze_batch_rule);
    # 启用对 squeeze 函数的批处理支持，参数为维度，使用 squeeze_dim_batch_rule 规则
    VMAP_SUPPORT2(squeeze, dim, squeeze_dim_batch_rule);
    # 启用对 squeeze 函数的批处理支持，参数为多个维度，使用 squeeze_dims_batch_rule 规则
    VMAP_SUPPORT2(squeeze, dims, squeeze_dims_batch_rule);
    # 启用对 _reshape_alias 函数的批处理支持，使用 _reshape_alias_batch_rule 规则
    VMAP_SUPPORT(_reshape_alias, _reshape_alias_batch_rule);
    # 启用对 roll 函数的批处理支持，使用 roll_batch_rule 规则
    VMAP_SUPPORT(roll, roll_batch_rule);
    # 启用对 permute 函数的批处理支持，使用 permute_batching_rule 规则
    VMAP_SUPPORT(permute, permute_batching_rule);
    # 启用对 diagonal 函数的批处理支持，使用 diagonal_batching_rule 规则
    VMAP_SUPPORT(diagonal, diagonal_batching_rule);
    # 启用对 diagonal_backward 函数的批处理支持，使用 diagonal_backward_batch_rule 规则
    VMAP_SUPPORT(diagonal_backward, diagonal_backward_batch_rule);
    # 启用对 select_backward 函数的批处理支持，使用 select_backward_batch_rule 规则
    VMAP_SUPPORT(select_backward, select_backward_batch_rule);
    # 启用对 slice_backward 函数的批处理支持，使用 slice_backward_batch_rule 规则
    VMAP_SUPPORT(slice_backward, slice_backward_batch_rule);
    # 启用对 view 函数的批处理支持，使用 view_batching_rule 规则
    VMAP_SUPPORT(view, view_batching_rule);
    # 启用对 view_copy 函数的批处理支持，使用 view_copy_batch_rule 规则
    VMAP_SUPPORT(view_copy, view_copy_batch_rule);
    # 启用对 expand 函数的批处理支持，使用 expand_batch_rule 规则
    VMAP_SUPPORT(expand, SINGLE_ARG(expand_batch_rule<decltype(&ATEN_FN(expand)), &ATEN_FN(expand)>));
    # 启用对 expand_copy 函数的批处理支持，使用 expand_batch_rule 规则
    VMAP_SUPPORT(expand_copy, SINGLE_ARG(expand_batch_rule<decltype(&ATEN_FN(expand_copy)), &ATEN_FN(expand_copy)>));
    # 启用对 unfold 函数的批处理支持，使用 unfold_batch_rule 规则
    VMAP_SUPPORT(unfold, unfold_batch_rule);
    # 启用对 slice 函数的批处理支持，参数为张量，使用 slice_batch_rule 规则
    VMAP_SUPPORT2(slice, Tensor, slice_batch_rule);
    # 启用对 transpose 函数的批处理支持，参数为整数，使用 transpose_int_batch_rule 规则
    VMAP_SUPPORT2(transpose, int, transpose_int_batch_rule);
    # 将 t 函数映射到 native::t 实现，用于复合显式自动微分，不应放在 BatchRulesDecompositions.cpp 中
    m.impl("t", native::t);  // CompositeExplicitAutograd, should not go in BatchRulesDecompositions.cpp
    # 将 t_ 函数映射到 native::t_ 实现，用于复合显式自动微分，不应放在 BatchRulesDecompositions.cpp 中
    m.impl("t_", native::t_);  // CompositeExplicitAutograd, should not go in BatchRulesDecompositions.cpp
    # 启用对 diag_embed 函数的批处理支持，使用 diag_embed_batch_rule 规则
    VMAP_SUPPORT(diag_embed, diag_embed_batch_rule);
    # 启用对 narrow_copy 函数的批处理支持，使用 narrow_copy_batch_rule 规则
    VMAP_SUPPORT(narrow_copy, narrow_copy_batch_rule);
    # 启用对 unsafe_split 函数的批处理支持，参数为张量，使用 unsafe_split_batch_rule 规则
    VMAP_SUPPORT2(unsafe_split, Tensor, unsafe_split_batch_rule);
}
```