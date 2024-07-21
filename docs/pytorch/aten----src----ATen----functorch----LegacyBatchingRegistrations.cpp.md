# `.\pytorch\aten\src\ATen\functorch\LegacyBatchingRegistrations.cpp`

```py
// 引入 PyTorch 库的必要头文件
#include <torch/library.h>
#include <ATen/native/ResizeCommon.h>
#include <ATen/ATen.h>
#include <ATen/native/TensorShape.h>

// 引入 FunctoRCH 库的特定头文件
#include <ATen/NestedTensorImpl.h>
#include <ATen/functorch/DynamicLayer.h>
#include <ATen/functorch/TensorWrapper.h>
#include <ATen/functorch/BatchingMetaprogramming.h>
#include <ATen/functorch/LegacyVmapTransforms.h>
#include <ATen/functorch/BatchedFallback.h>
#include <ATen/functorch/BatchRulesHelper.h>

// 引入 C++ 标准库的头文件
#include <utility>

// 定义 FunctoRCH 命名空间
namespace at::functorch {

// NOTE: [What is a batching rule?]
//
// NB: the following description only applies to this file and is about
// the legacy (deprecated) batching rule API. Please see writing_batch_rules.md
// for how to write new-style batching rules.
//
// This files contains batching rules written with the legacy (now-deprecated)
// batching rule API.
// Please try to use the new-style batching rule API (see writing_batch_rules.md)
//
// A *batching rule* implements the logic of how to call an operator on inputs
// that have zero or more additional batch dimensions. When one does a vmap, the
// dimension(s) being vmap'ed over get recorded as batch dimensions.
//
// For example, vmap(torch.add)(x, y)
// 1. wraps `x` into batched_x = BatchedTensor(x, bdims=[(lvl=1, dim=0)];
// 2. wraps `y` into batched_y = BatchedTensor(y, bdims=[(lvl=1, dim=0)];
// 3. and then runs `torch.add(batched_x, batched_y)`.

// NOTE: [When should I add a batching rule?]
// When you are adding a new operator, you'll need to add a batching rule so
// that vmap can work efficiently with said operator. If you do not, we'll attempt
// to generate a slow fallback for the batching rule.

// NOTE: [How to write batching rules?]
// The signature of a batching rule should look like exactly like the C++ signature
// of its operator.
//
// First, see NOTE: [Logical vs physical args] in VmapTransforms.h for terminology.
//
// At a high level, what a batching rule does is the following:
// 1. Converts (logical) BatchedTensors to views on physical tensors.
// 2. Converts logical arguments (e.g. dimension indexes, shapes) to physical
//    arguments that correspond to the physical tensors.
// 3. Calls at:: operations on the physical tensors and arguments to produce
//    some physical results.
// 4. Converts physical results back to BatchedTensors.
//
// Steps 1, 2, and 4 differ for operators with different batching behaviors. When
// writing a new batching rule, please select a VmapTransform that matches the
// batching behavior of your operation. The VmapTransform provides helper functions
// to do steps (1), (2), and (4).
// (see NOTE: [What is an VmapTransform?] in VmapTransforms.h)

// 定义匿名命名空间，限制本文件内部可见性
// 检查给定的维度是否在标量张量上是允许的
static bool is_allowed_dim_on_scalar_tensor(int64_t dim) {
  return dim == 0 || dim == -1;
}

// 获取当前动态层级的层级 ID
static int64_t get_current_level() {
  auto maybe_level = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_level.has_value());
  return maybe_level->layerId();
}

// 检查张量是否参与当前层级
static bool participatesInCurrentLevel(const Tensor& self) {
  auto current_level = get_current_level();
  auto* maybe_batched_impl = maybeGetBatchedImpl(self);
  if (!maybe_batched_impl) {
    return false;
  }
  auto self_level = maybe_batched_impl->level();
  TORCH_INTERNAL_ASSERT(self_level <= current_level);
  return self_level == current_level;
}

// 检查张量列表是否有张量参与当前层级
static bool participatesInCurrentLevel(ITensorListRef self) {
  for (const Tensor& tensor : self) {
    if (participatesInCurrentLevel(tensor)) {
      return true;
    }
  }
  return false;
}

// 挤压张量的维度，根据批处理规则进行调整
Tensor& squeeze_dims__batching_rule(Tensor& self, IntArrayRef dims) {
  // 如果张量不参与当前层级，则使用 ExcludeDispatchKeyGuard 排除 FuncTorchBatched 分发键
  if (!participatesInCurrentLevel(self)) {
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    return self.squeeze_(dims);
  }
  auto* batched = maybeGetBatchedImpl(self);
  const auto bdim = batched->bdim();
  auto logical_dim = self.dim();

  // 如果张量维度为 0
  if (logical_dim == 0) {
    TORCH_CHECK(
        dims.empty() || (dims.size() == 1 && dims[0] == 0),
        "Dimension is out of range (expected to be in range of [-1, 0], but got ", dims);
    return self;
  }

  // 调整高于批处理维度的任何维度
  DimVector adjusted_dims(dims.begin(), dims.end());
  int64_t updated_batch_idx = bdim;
  for (auto &d : adjusted_dims) {
    auto actual_dim = c10::maybe_wrap_dim(d, logical_dim);
    if (actual_dim < bdim) {
      d = actual_dim;
      if (batched->value().sym_size(actual_dim) == 1) {
        // 在批处理维度之前的列将被删除，因此进行相应调整
        --updated_batch_idx;
      }
    } else {
      // 由于要挤压的维度在批处理维度之后，因此加一以考虑原始批处理维度
      d = actual_dim + 1;
    }
  }

  // 在批处理值上进行挤压
  batched->value().squeeze_(adjusted_dims);
  if (updated_batch_idx != bdim) {
    batched->unsafe_set_bdim(updated_batch_idx);
  }
  // 刷新张量的元数据
  batched->refreshTensorMetadata();
  return self;
}

// 挤压张量的单个维度，根据批处理规则进行调整
Tensor& squeeze_dim__batching_rule(Tensor& self, int64_t dim) {
  return squeeze_dims__batching_rule(self, {dim});
}

// 挤压张量，根据批处理规则进行调整
Tensor& squeeze__batching_rule(Tensor& self) {
  // 如果张量不参与当前层级，则使用 ExcludeDispatchKeyGuard 排除 FuncTorchBatched 分发键
  if (!participatesInCurrentLevel(self)) {
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    return self.squeeze_();
  }
  auto* batched = maybeGetBatchedImpl(self);

  // 需要确定在批处理维度之前有多少个大小为 1 的维度
  const auto bdim = batched->bdim();
  const auto physical_shape = batched->value().sizes();
  auto how_many_dims_of_size_1_before_bdim = 0;
  for (const auto i : c10::irange(0, physical_shape.size())) {
    if ((int64_t)i == bdim) {
      break;
    }
    // TO BE CONTINUED...


这里只显示了部分代码的注释，由于代码比较长，无法一次性显示完整。
    // 遍历物理形状的每个维度
    if (physical_shape[i] == 1) {
      // 如果当前维度的大小为1，增加计数器
      how_many_dims_of_size_1_before_bdim++;
    }
  }

  // 计算新的批次维度（减去所有大小为1的维度）
  int64_t new_bdim = bdim - how_many_dims_of_size_1_before_bdim;

  // 检查批次维度对应的物理形状是否为1
  if (physical_shape[bdim] != 1) {
    // 如果批次维度不为1，直接调用squeeze_()
    batched->value().squeeze_();
  } else {
    // 如果批次维度为1，则调用squeeze_()将会去除该维度
    // 通过调用unsqueeze_修正去除的维度
    batched->value().squeeze_();
    batched->value().unsqueeze(new_bdim);
  }

  // 更新批次维度的元数据
  batched->unsafe_set_bdim(new_bdim);
  // 刷新张量的元数据
  batched->refreshTensorMetadata();
  // 返回原始对象
  return self;
}

// 处理张量的 unsqueeze_ 操作的批处理规则
Tensor& unsqueeze__batching_rule(Tensor& self, int64_t dim) {
  // 如果张量不参与当前级别的批处理，排除批处理的调度键
  if (!participatesInCurrentLevel(self)) {
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    // 返回原始张量，不进行 unsqueeze_ 操作
    return self.unsqueeze_(dim);
  }

  // 获取张量的批处理实现
  auto* batched = maybeGetBatchedImpl(self);
  auto logical_dim = self.dim();
  // 将传入的维度 dim 转换为物理维度
  int64_t dim_physical = maybe_wrap_dim(dim, logical_dim + 1);

  // 如果物理维度大于等于当前批处理维度，增加批处理维度
  if (dim_physical >= batched->bdim()) {
    dim_physical = 1 + dim_physical;
  } else {
    batched->unsafe_set_bdim(batched->bdim() + 1);
  }
  // 在批处理值中对张量执行 unsqueeze_ 操作
  batched->value().unsqueeze_(dim_physical);

  // 同时需要更改一些元数据...
  batched->refreshTensorMetadata();
  // 返回修改后的原始张量
  return self;
}

// 处理张量的 transpose_ 操作的批处理规则
Tensor& transpose__batching_rule(Tensor& self, int64_t dim0, int64_t dim1) {
  // 如果张量不参与当前级别的批处理，排除批处理的调度键
  if (!participatesInCurrentLevel(self)) {
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    // 返回原始张量，不进行 transpose_ 操作
    return self.transpose_(dim0, dim1);
  }

  // 获取张量的批处理实现
  auto* batched = maybeGetBatchedImpl(self);
  auto logical_dim = self.dim();

  // 对于标量张量的特殊情况，PyTorch 允许 dim0, dim1 为 {0, -1}，返回标量张量
  if (logical_dim == 0 &&
      is_allowed_dim_on_scalar_tensor(dim0) &&
      is_allowed_dim_on_scalar_tensor(dim1)) {
    // 没有进行转置操作
    return self;
  }

  // 将 dim0 和 dim1 转换为物理维度
  dim0 = maybe_wrap_dim(dim0, logical_dim);
  dim1 = maybe_wrap_dim(dim1, logical_dim);

  // 如果物理维度大于等于当前批处理维度，增加批处理维度
  dim0 = dim0 >= batched->bdim() ? dim0 + 1 : dim0;
  dim1 = dim1 >= batched->bdim() ? dim1 + 1 : dim1;
  // 在批处理值中对张量执行 transpose_ 操作
  batched->value().transpose_(dim0, dim1);

  // 同时需要更改一些元数据...
  batched->refreshTensorMetadata();
  // 返回修改后的原始张量
  return self;
}

// 处理张量的 split 操作的批处理规则
std::vector<Tensor> split_batching_rule(const Tensor& self, int64_t split_size, int64_t dim) {
  // 如果张量不参与当前级别的批处理，排除批处理的调度键
  if (!participatesInCurrentLevel(self)) {
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    // 调用原始的 split 函数
    return at::split(self, split_size, dim);
  }

  // 将逻辑上的张量转换为物理表示
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  auto dim_physical = self_physical.getPhysicalDim(dim);
  // 在物理张量上调用 split 函数
  auto result = at::split(self_physical.tensor(), split_size, dim_physical);
  // 将物理到逻辑映射应用到结果上
  self_physical.getPhysicalToLogicalMap().applyInplace(result);
  // 返回处理后的结果张量数组
  return result;
}

// 处理张量的 split_with_sizes 操作的批处理规则
std::vector<Tensor> split_with_sizes_batching_rule(const Tensor& self, SymIntArrayRef split_sizes, int64_t dim) {
  // 如果张量不参与当前级别的批处理，排除批处理的调度键
  if (!participatesInCurrentLevel(self)) {
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    // 调用原始的 split_with_sizes_symint 函数
    return split_with_sizes_symint(self, split_sizes, dim);
  }

  // 将逻辑上的张量转换为物理表示
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  auto dim_physical = self_physical.getPhysicalDim(dim);
  // 在物理张量上调用 split_with_sizes_symint 函数
  auto result = split_with_sizes_symint(self_physical.tensor(), split_sizes, dim_physical);
  // 将物理到逻辑映射应用到结果上
  self_physical.getPhysicalToLogicalMap().applyInplace(result);
  // 返回处理后的结果张量数组
  return result;
}
// 定义函数，根据指定尺寸和维度将张量分割成多个张量，返回分割后的张量向量
std::vector<Tensor> split_with_sizes_copy_batching_rule(const Tensor& self, SymIntArrayRef split_sizes, int64_t dim) {
  // 如果张量不参与当前级别的操作，则在排除 TorchBatched 调度键的保护下执行分割操作
  if (!participatesInCurrentLevel(self)) {
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    return split_with_sizes_copy_symint(self, split_sizes, dim);
  }
  // 将逻辑上的张量转换为物理上的张量
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  // 获取物理维度
  auto dim_physical = self_physical.getPhysicalDim(dim);
  // 调用分割函数对物理张量进行分割操作
  auto result = split_with_sizes_copy_symint(self_physical.tensor(), split_sizes, dim_physical);
  // 将物理到逻辑映射应用于分割后的结果张量
  self_physical.getPhysicalToLogicalMap().applyInplace(result);
  // 返回分割后的结果张量向量
  return result;
}

// 定义函数，解绑定给定维度上的张量，返回解绑定后的张量向量
std::vector<Tensor> unbind_batching_rule(const Tensor& self, int64_t dim) {
  // 如果张量不参与当前级别的操作，则在排除 TorchBatched 调度键的保护下执行解绑定操作
  if (!participatesInCurrentLevel(self)) {
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    return at::unbind(self, dim);
  }
  // 将逻辑上的张量转换为物理上的张量
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  // 获取物理维度
  auto dim_physical = self_physical.getPhysicalDim(dim);
  // 调用解绑定函数对物理张量进行解绑定操作
  auto result = at::unbind(self_physical.tensor(), dim_physical);
  // 将物理到逻辑映射应用于解绑定后的结果张量
  self_physical.getPhysicalToLogicalMap().applyInplace(result);
  // 返回解绑定后的结果张量向量
  return result;
}

// 给定大小、步长和存储偏移，返回可以索引的最大位置，若不存在返回 nullopt（例如，零大小维度的张量）
static optional<c10::SymInt> maximum_indexable_location(
    c10::SymIntArrayRef sizes, c10::SymIntArrayRef strides, const c10::SymInt& storage_offset) {
  // 调用原生的函数计算给定大小和步长情况下的存储大小
  auto result = native::storage_size_for(sizes, strides);
  // 如果计算结果为零，返回空
  if (result == 0) {
    return nullopt;
  }
  // 返回计算结果与存储偏移的和作为可索引的最大位置
  return result + storage_offset;
}

// 检查 "物理张量" 的 "第一个切片" 在内存中可访问的范围是否合法
static void checkBasicAsStridedValidForSlice(
    const Tensor& physical_tensor,
    int64_t num_batch_dims,
    c10::SymIntArrayRef sizes,
    c10::SymIntArrayRef strides,
    const optional<c10::SymInt>& maybe_storage_offset) {
  // 获取物理张量的符号尺寸的切片和符号步长的切片
  auto slice_sizes = physical_tensor.sym_sizes().slice(num_batch_dims);
  auto slice_strides = physical_tensor.sym_strides().slice(num_batch_dims);
  // 获取基础偏移量
  auto base_offset = physical_tensor.sym_storage_offset();

  // 根据可能的存储偏移选择实际使用的存储偏移
  auto storage_offset = maybe_storage_offset.value_or(base_offset);

  // 计算作为 as_strided 操作输入的尺寸、步长和存储偏移情况下的可索引最大位置
  auto max_as_strided_loc = maximum_indexable_location(sizes, strides, storage_offset);
  // 计算切片操作下的可索引最大位置
  auto max_slice_loc = maximum_indexable_location(slice_sizes, slice_strides, base_offset);

  // 如果 as_strided 操作下的最大位置无效，直接返回
  if (!max_as_strided_loc.has_value()) {
    return;
  }
  // 如果切片操作下的最大位置无效，直接返回
  if (!max_slice_loc.has_value()) {

    return;
  }
  // 其他情况，继续执行操作
}
    # 使用 TORCH_CHECK 宏检查条件，如果为 false，则输出错误消息并终止程序
    TORCH_CHECK(false,
        "result = tensor.as_strided(", sizes, ", ",  strides, ", ", storage_offset, ") ",
        "can access memory outside of `tensor`. `tensor` has no storage but the ",
        "passed-in (size, stride, storage_offset) imply a result with some storage. ",
        "This is not supported inside of vmap, please try to rewrite the ",
        "`as_strided` call as a sequence of PyTorch view operations");
    }
    
    # 使用 TORCH_CHECK 宏检查条件，确保条件为真，否则输出错误消息并终止程序
    TORCH_CHECK(
        *max_as_strided_loc <= *max_slice_loc && base_offset <= storage_offset,
        "result = tensor.as_strided(", sizes, ", ",  strides, ", ", storage_offset, ") ",
        "can access memory outside of `tensor`. `result` can access some ",
        "memory in range [", storage_offset, ", ", *max_as_strided_loc, "], but ",
        "`tensor` can only access some memory in range [", base_offset, ", ",
        *max_slice_loc, "]. This is not supported inside of vmap, please try to ",
        "rewrite the `as_strided` call as a sequence of PyTorch view operations");
}

// as_strided 在 vmap 中的语义是什么？
// y = vmap(lambda x: x.as_strided(sizes, strides, offset))(xs)
// 这段代码返回了对 `x` 的视图 `y`，使得每个 y[i] 都具有以下特性：
// - sizes: `sizes`
// - strides: `strides`
// - storage_offset: offset + i * x.stride(batch_dim)
//
// 换句话说，这就好像我们将每个 x[i] 视为具有 storage_offset 等于 xs.offset()，
// 然后调用 as_strided(sizes, sizes, offset)。
// （这相当于对于所有 i，x[i].as_strided(
//    sizes, sizes, offset + x[i].storage_offset() - xs.offset())。）
//
// 需要注意的是，这与在 for 循环中实际运行 as_strided 是有所不同的。
// 这是因为 as_strided 接受的 `offset` 是一个*绝对*偏移量。
// 举个例子，考虑以下情况：
// >>> x = torch.tensor([0., 1., 2., 3., 4.]).as_strided([4], [1], 1)
// >>> z = [x[i].as_strided([1], [1], 1) for i in range(4)]
// 每个 z[i] 实际上都是对 x 的相同视图（z[i] == torch.tensor([1.])）！
// 然而，我们认为上面的 for 循环理解是用户的错误：
// 如果用户想要以每个样本的方式使用 as_strided，则应该写成以下形式：
// >>> z = [x[i].as_strided([1], [1], 1 + x[i].storage_offset() - 1) for i in range(4)]
Tensor as_strided_batching_rule(
    const Tensor& tensor,
    c10::SymIntArrayRef sizes,
    c10::SymIntArrayRef strides,
    optional<c10::SymInt> storage_offset) {
  if (!participatesInCurrentLevel(tensor)) {
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  // 调用 tensor 的 as_strided_symint 方法，返回结果
  return at::as_strided_symint(tensor, sizes, strides, std::move(storage_offset));
}
// 获取 tensor 的物理视图
auto physical_view = MultiBatchVmapTransform::logicalToPhysical(tensor);
// 获取物理视图的批处理维度数量
auto num_batch_dims = physical_view.numBatchDims();
// 根据物理视图获取物理尺寸
auto physical_sizes = physical_view.getPhysicalShape(sizes);
// 获取物理视图中的 tensor
const auto& physical_tensor = physical_view.tensor();

// 对 size 和 stride 进行检查，确保它们长度相同
TORCH_CHECK(sizes.size() == strides.size(),
    "Tensor.as_strided(size, stride, ...): size and stride must have the ",
    "same length! Got size ", sizes, " and stride ", strides);

// 执行基本的 as_strided 有效性检查，确保切片操作有效
checkBasicAsStridedValidForSlice(
    physical_tensor, num_batch_dims, sizes, strides, storage_offset);

// 计算物理 strides，包括批处理维度的批次 strides 和逻辑 strides
auto batch_strides = physical_tensor.strides().slice(0, num_batch_dims);
SymDimVector physical_strides;
physical_strides.reserve(num_batch_dims + strides.size());
physical_strides.insert(
    physical_strides.end(), batch_strides.begin(), batch_strides.end());
physical_strides.insert(
    physical_strides.end(), strides.begin(), strides.end());

// 如果对所有 i 都适用 zi = xs[i].as_strided(sizes, strides, offset + xs[i].offset() - xs.offset())
// 则 xs.as_strided(physical_sizes, physical_strides, offset) 总是成功，并创建一个 tensor y，
// 其中每个 y[i] 引用与 zi 相同的内存位置
auto result = physical_view.tensor().as_strided_symint(
    physical_sizes, physical_strides, std::move(storage_offset));
// 应用物理到逻辑映射，返回结果
return physical_view.getPhysicalToLogicalMap().apply(result);
// NOTE: [When will the as_strided batching rule fail?]
// If zi = xs[i].as_strided(sizes, strides, offset + xs[i].offset() - xs.offset())
// is valid for all i, then it turns out that
// xs.as_strided(physical_sizes, physical_strides, offset) always succeeds and
// creates a tensor y such that each y[i] refers to the same memory as zi.
//
// Let's say we have xs[i].as_strided(sizes, strides, offset + xs[i].offset() - xs.offset()).
// Furthermore, let's say that as a part of being "valid" this as_strided call
// does not return a result that can index memory not indexable by xs[i].

// WLOG, assume that there's only one batch dim and it is at the front of the
// `xs` tensor. Let B be the batch size and S be the stride of the batch dim.
// - If the batch dim isn't at the front of the tensor, then we can just move it
// to the front with movedim/permute. This is always valid because it just swaps
// some strides around.
// - This proof also works for tensors with multiple batch dims. We just have to
// do a little accounting:
//   - instead of [B], we'd have [B0, B1, ..., Bk].
//   - instead of [S], we'd have [S0, S1, ..., Sk].
//   - instead of i, we'd have a list of indices [I0, I1, ..., Ik]
//   - instead of S * I, we'd have \sum_{i=0}^k S_i * I_i

// [Equation 1]
// xs[i].as_strided(sizes, strides, offset + xs[i].offset() - xs.offset()) has:
// - sizes: sizes
// - strides: strides
// - offset: offset + S * i
//
// x.as_strided itself checks that:
// - (sizes, strides, offset) are in bounds for `x`'s storage.
// - strides are positive
// - offset is positive

// Claim 1: if xs[i].as_strided(sizes, strides, offset + xs[i].offset() - xs.offset())
// is valid, then
// ([B] + sizes, [S] + strides, offset + xs.offset()) are in bounds for `xs`'s storage.
//
// If we have the claim, then xs.as_strided([B] + sizes, [S] + strides, offset)
// won't error out. So all we need to check is that the memory locations are
// what we expected. See [Hand-wavy proof of Claim 1] for proof (it's not very important)

// xs.as_strided(physical_sizes, physical_strides, offset) is equivalent to
// xs.as_strided([B] + sizes, [S] + strides, offset)

// xs.as_strided([B] + sizes, [S] + strides, offset) has:
// - sizes: [B] + sizes
// - strides: [S] + strides
// - offset: offset

// xs.as_strided([B] + sizes, [S] + strides, offset)[i] has:
// - sizes: sizes
// - strides: strides
// - offset: offset + S * i
// These memory locations are exactly the same as what we got for [Equation 1],
// so the xs.as_strided([B] + sizes, [S] + strides, offset) is valid.

// [Hand-wavy proof of Claim 1]
// Part of our definition of being valid is that xs[i].as_strided(...)
// must return a tensor that only uses memory indexable by xs[i].
// This means that (sizes, strides, offset + xs[i].offset() - xs.offset()) satisfies:
//    offset + xs[i].offset() - xs.offset() + 1 + \sum_j (sizes[j] - 1) * strides[j]
// 定义一个模板函数，用于将输入张量解包并调用指定的函数 `Func` 进行处理
template <typename F, F Func, typename... ExtraArgs>
Tensor unwrap_and_call(const Tensor& input, ExtraArgs... args) {
    // 如果输入张量 `input` 不参与当前级别的处理，设置排除调度键保护
    if (!participatesInCurrentLevel(input)) {
        c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
        // 调用 `Func` 处理输入张量 `input` 和额外的参数，并返回处理后的张量
        return Func(input, args...);
    }
    
    // 防止用户传递的是批量的标量张量的情况，获取不安全的批处理实现指针
    auto* input_batched = unsafeGetBatchedImpl(input);
    // 调用 `Func` 处理批处理值的物理张量，并返回处理后的张量
    auto output_physical = Func(input_batched->value(), args...);
    // 将处理后的物理张量重新批处理，并返回结果
    return makeBatched(output_physical, input_batched->bdim(), input_batched->level());
}

// 定义一个模板函数，用于解包输入张量并调用指定的方法 `Func` 进行处理
template <typename F, F Func, typename... ExtraArgs>
Tensor unwrap_and_call_method(const Tensor& input, ExtraArgs... extra_args) {
    // 如果输入张量 `input` 不参与当前级别的处理，设置排除调度键保护
    if (!participatesInCurrentLevel(input)) {
        c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
        // 使用指定方法 `Func` 调用输入张量 `input` 的方法，并传递额外的参数，返回结果张量
        return (input.*Func)(extra_args...);
    }
    
    // 获取不安全的批处理实现指针，处理批处理值的物理张量的方法 `Func`
    auto* input_batched = unsafeGetBatchedImpl(input);
    // 使用指定方法 `Func` 调用批处理值的物理张量的方法，并传递额外的参数，返回结果张量
    auto output_physical = (input_batched->value().*Func)(extra_args...);
    // 将处理后的物理张量重新批处理，并返回结果
    return makeBatched(output_physical, input_batched->bdim(), input_batched->level());
}

// 定义一个批处理规则函数，用于在指定维度 `dim` 上连接张量列表 `tensors`
Tensor cat_batching_rule(const ITensorListRef& tensors, int64_t dim) {
    // 如果张量列表 `tensors` 不参与当前级别的处理，设置排除调度键保护
    if (!participatesInCurrentLevel(tensors)) {
        c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
        // 返回调用函数的批处理版本，处理输入张量列表 `tensors` 在指定维度 `dim` 上的连接，并返回结果张量
        return FuncTorchBatched(tensors, dim);
    }
    
    // 在批处理的情况下，获取不安全的批处理实现指针
    auto* input_batched = unsafeGetBatchedImpl(tensors);
    // 调用批处理值的物理张量的连接方法，并返回结果张量
    auto output_physical = tensors.cat(dim);
    // 将处理后的物理张量重新批处理，并返回结果
    return makeBatched(output_physical, input_batched->bdim(), input_batched->level());
}
  return at::cat(tensors, dim);
}

c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);

// NB: Probably bad for perf that we're allocating std::vectors for each level, but
// what can you do.
// 创建一个 std::vector 用于存储每个级别的 materialized tensors
auto materialized = tensors.materialize();
// 根据 legacy_cat_wrap_dim 函数调整 dim 的值，确保合适的拼接维度
dim = at::legacy_cat_wrap_dim(dim, materialized);

// Strategy:
// 我们将展开张量并将它们的批处理维度移到最前面，
// 然后放入 `tensors_to_cat` 中。对于没有批处理维度的张量，
// 将会为它们强制添加一个。
//
// 然后，我们将调用 at::cat(tensors_to_cat, ...) 进行拼接。
//
// 还有一种特殊情况，即 at::cat 会忽略逻辑形状为 [0] 的张量。如果我们遇到
// 逻辑形状为 [0]（但物理形状为 [B, 0]）的张量，我们将对张量进行切片以获得
// 形状为 [0] 的张量，然后传递给 at::cat。
// 初始化一个用于存储将要拼接的张量的 vector
std::vector<Tensor> tensors_to_cat;
// 预留足够的空间以容纳所有的 tensors
tensors_to_cat.reserve(tensors.size());
// 可能是空值的批处理维度大小，如果所有 BatchedTensor 都应该被 at::cat 的特殊情况跳过
std::optional<int64_t> bdim_size = c10::nullopt;

// 查找批处理维度的大小。如果所有 BatchedTensor 都应该被 at::cat 的特殊情况跳过，
// 则可能不存在。
for (const auto& tensor : tensors) {
  // 如果 tensor 不参与当前级别，则继续下一个循环
  if (!participatesInCurrentLevel(tensor)) {
    continue;
  }
  // 如果 tensor 应该被 at::cat 的特殊情况跳过，则继续下一个循环
  if (at::native::cat_should_skip_tensor(tensor)) {
    continue;
  }
  // 获取 BatchedImpl，并从中获取批处理维度的大小
  const auto* batched = unsafeGetBatchedImpl(tensor);
  bdim_size = batched->value().size(batched->bdim());
  break;
}

// 展开 BatchedTensor；扩展批处理维度
for (const auto& tensor : tensors) {
  // 如果 tensor 不参与当前级别
  if (!participatesInCurrentLevel(tensor)) {
    // 如果 tensor 应该被 at::cat 的特殊情况跳过或者 bdim_size 为空，则直接加入 tensors_to_cat
    if (at::native::cat_should_skip_tensor(tensor) || !bdim_size.has_value()) {
      tensors_to_cat.emplace_back(tensor);
      continue;
    }
    // 否则，为 tensor 强制添加批处理维度后加入 tensors_to_cat
    tensors_to_cat.emplace_back(ensure_has_bdim(tensor, /*has_bdim*/false, *bdim_size));
    continue;
  }
  // 如果 tensor 参与当前级别
  const auto* batched = unsafeGetBatchedImpl(tensor);
  if (at::native::cat_should_skip_tensor(tensor)) {
    // 特殊情况：对张量进行切片以获取形状为 [0] 的张量，然后加入 tensors_to_cat
    tensors_to_cat.emplace_back(batched->value().select(/*dim=*/batched->bdim(), /*index=*/0));
    continue;
  }
  // 将批处理维度移到前面，并加入 tensors_to_cat
  tensors_to_cat.emplace_back(moveBatchDimToFront(batched->value(), batched->bdim()));
}

// 计算新的拼接维度
auto new_dim = bdim_size.has_value() ? dim + 1 : dim;
// 计算新的批处理维度
std::optional<int64_t> new_bdim = bdim_size.has_value() ? c10::make_optional((int64_t)0) : nullopt;
// 调用 at::cat 进行张量拼接
auto result = at::cat(tensors_to_cat, new_dim);
// 构造一个新的 BatchedTensor，包含 result 和新的批处理维度，返回结果
return makeBatched(result, new_bdim, get_current_level());
}

# 定义一个函数，实现对输入张量列表进行批处理规则处理，返回处理后的张量
Tensor block_diag_batching_rule(TensorList tensors) {
  # 如果输入张量列表不参与当前级别的处理，则通过ExcludeDispatchKeyGuard保护，返回at::block_diag处理结果
  if (!participatesInCurrentLevel(tensors)) {
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    return at::block_diag(tensors);
  }
  # 将逻辑视图转换为物理视图
  auto physical_views = MultiBatchVmapTransform::logicalToPhysical(tensors);
  # 提取物理张量列表
  auto physical_tensors = fmap(
      physical_views, [](const VmapPhysicalView& view) -> Tensor { return view.tensor(); });
  # 断言张量列表非空，否则报错
  TORCH_INTERNAL_ASSERT(
      !tensors.empty(), "The dispatcher should not have dispatched here otherwise.");
  # 实现一个针对循环的虚拟批处理规则，这里使用一个dummy for循环，因为不确定如何更好地处理
  # 可能没有考虑到多个批次维度的情况？
  # 获取批次维度数
  auto bdim = physical_tensors[0].size(0);
  # 初始化批处理输出向量
  std::vector<Tensor> batched_outputs;
  batched_outputs.reserve(bdim);
  # 遍历批次维度
  for (const auto& i : c10::irange(bdim)) {
    # 初始化用于当前批次的输入张量向量
    std::vector<Tensor> inputs_for_batch;
    inputs_for_batch.reserve(physical_tensors.size());
    # 遍历物理张量，提取当前批次的各个张量
    for (const auto& t : physical_tensors) {
      inputs_for_batch.push_back(t[i]);
    }
    # 对当前批次的输入张量执行at::block_diag操作
    auto out_for_batch = at::block_diag(inputs_for_batch);
    # 将处理结果添加到批处理输出向量中，并在0维度上进行unsqueeze操作
    batched_outputs.push_back(out_for_batch.unsqueeze(0));
  }
  # 对批处理输出向量进行cat操作，得到最终结果
  auto result = at::cat(batched_outputs);
  # 将物理视图映射应用到结果中，并返回逻辑视图
  return physical_views[0].getPhysicalToLogicalMap().apply(result);
}

# 定义一个函数，实现对输入张量列表在指定维度上进行堆叠的批处理规则处理，返回处理后的张量
Tensor stack_batching_rule(TensorList tensors, int64_t dim) {
  # 如果输入张量列表不参与当前级别的处理，则通过ExcludeDispatchKeyGuard保护，返回at::stack处理结果
  if (!participatesInCurrentLevel(tensors)) {
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    return at::stack(tensors, dim);
  }
  # 将逻辑视图转换为物理视图
  auto physical_views = MultiBatchVmapTransform::logicalToPhysical(tensors);
  # 提取物理张量列表
  auto physical_tensors = fmap(
      physical_views, [](const VmapPhysicalView& view) -> Tensor { return view.tensor(); });
  # 断言张量列表非空，否则报错
  TORCH_INTERNAL_ASSERT(
      !tensors.empty(), "The dispatcher should not have dispatched here otherwise.");
  # 注意：stack函数将维度包装到（逻辑维度 + 1），因此我们在这里必须手动处理
  # 计算物理维度，考虑包装维度
  auto dim_physical =
      physical_views[0].numBatchDims() + maybe_wrap_dim(dim, /*logical*/tensors[0].dim() + 1);
  # 对物理张量列表在物理维度上执行at::stack操作
  auto result = at::stack(physical_tensors, dim_physical);
  # 将物理视图映射应用到结果中，并返回逻辑视图
  return physical_views[0].getPhysicalToLogicalMap().apply(result);
}

# 定义一个函数，实现创建新的空批处理规则，返回处理后的张量
Tensor new_empty_strided_batching_rule(
    const Tensor& self,
    SymIntArrayRef sym_size,
    SymIntArrayRef sym_stride,
    optional<ScalarType> dtype,
    optional<Layout> layout,
    optional<Device> device,
    optional<bool> pin_memory) {

  # 获取尺寸和步长
  auto size = C10_AS_INTARRAYREF_SLOW(sym_size);
  auto stride = C10_AS_INTARRAYREF_SLOW(sym_stride);
  # 如果自身张量不参与当前级别的处理，则通过ExcludeDispatchKeyGuard保护，返回空张量
  if (!participatesInCurrentLevel(self)) {
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    ```
  // 调用对象的方法创建一个新的空的张量视图，使用给定的参数
  return self.new_empty_strided(
      size, stride, dtype, layout, device, pin_memory);
}

// 计算逻辑视图到物理视图的转换
auto physical_view = MultiBatchVmapTransform::logicalToPhysical(self);
// 获取物理视图的物理形状
auto physical_size = physical_view.getPhysicalShape(size);

// 解释如何处理批次维度的逻辑
// 我们要在张量的内存布局前面创建批次维度，不考虑它们在原始张量中的实际位置
// 这是因为当用户通常调用 `new_empty_strided` 时，提供的 `strides` 是针对新张量的，与原始张量的步幅无关。
//
// 因此，结果的物理形状应该是 ([B0, B1, B2] + size)
// 但是物理步幅如何确定呢？
//
// 实际上，我们可以任意选择步幅:
// 例如，对于 size=[5, 3], stride=[0, 1]，我们可以决定使用
// - 物理大小：[B0, B1, B2, 5, 3]
// - 物理步幅：[9999*B1*B2, 9999*B2, 9999, 0, 1]
//
// 让我们选择一些合理的步幅，使得:
// - 批次维度在彼此之间“连续”
// - 如果 empty_strided(size, stride) 会创建一个连续的张量，那么这个新的物理张量（带有批次维度）也应该是连续的
//
// 假设 S 是使用 empty_strided(size, stride) 构造张量时的存储大小。
// 那么物理大小/步幅应该是:
// - 物理大小：[B0, B1, B2, 5, 3]
// - 物理步幅：[B1 * B2 * S, B2 * S, S, 0, 1]
auto batch_shape = IntArrayRef(
    physical_view.tensor().sizes().begin(), physical_view.numBatchDims());

// 计算默认步幅，基于给定的批次形状
auto physical_strides = at::detail::defaultStrides(batch_shape);
// 检查 size 和 stride 的维度是否匹配
TORCH_CHECK(size.size() == stride.size(),
      "new_empty_strided(sizes, strides): dimensionality of sizes (",
      size.size(), ") must match dimensionality of strides (",
      stride.size(), ")");
// 计算存储大小，基于给定的 size 和 stride
auto storage_size = native::storage_size_for(size, stride);
// 将物理步幅乘以存储大小
for (auto& physical_stride : physical_strides) {
  physical_stride *= storage_size;
}

// 将原始步幅追加到物理步幅后面
// 物理步幅 = [B1 * B2 * S, B2 * S, S] + strides
physical_strides.insert(physical_strides.end(), stride.begin(), stride.end());

// 使用物理视图的张量对象调用 new_empty_strided 方法，创建新的空张量
auto result = physical_view.tensor().new_empty_strided(
    physical_size, physical_strides, dtype, layout, device, pin_memory);
// 将物理到逻辑映射应用于结果张量，并返回映射后的结果
return physical_view.getPhysicalToLogicalMap().apply(result);
}

// 定义一个函数 `nested_cat_batching_rule`，接受一个张量列表和一个维度参数 `dim`
Tensor nested_cat_batching_rule(const ITensorListRef& tensors, int64_t dim) {
  // 检查张量列表不能为空
  TORCH_CHECK(!tensors.empty(), "cat() not supported on empty tensor list");

  // 创建一个二维向量 `unbound`，用于存储每个张量的非批处理数据
  std::vector<std::vector<Tensor>> unbound;
  // 遍历输入的张量列表
  for (const auto & tensor : tensors) {
    // 获取张量的批处理实现（如果有的话）
    auto* maybe_batched_impl = maybeGetBatchedImpl(tensor);
    // 检查是否获取到批处理实现，否则抛出错误
    TORCH_CHECK(maybe_batched_impl, "Tried to run batching rule for cat() on a non-batched tensor");
    // 获取批处理实现的值
    auto nt = maybe_batched_impl->value();
    // 检查该值是否为嵌套张量
    TORCH_CHECK(nt.is_nested(), "Tried to run batching rule for cat() on a non-nested tensor");
    // 禁用分派键 `DispatchKey::BatchedNestedTensor`，保证当前环境不是嵌套张量
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::BatchedNestedTensor);
    // 解绑嵌套张量并存储到当前张量的 `this_unbound`
    auto this_unbound = nt.unbind();
    // 如果 `unbound` 不为空，则检查前一个解绑列表与当前列表的大小是否相同
    if (!unbound.empty()) {
      TORCH_INTERNAL_ASSERT(unbound.front().size() == this_unbound.size(),
          "cat() not supported for differently-sized nested arguments");
    }
    // 将当前解绑列表 `this_unbound` 添加到 `unbound` 中
    unbound.push_back(this_unbound);
  }

  // 对每组解绑的组件执行 `cat` 操作
  const auto num_components = unbound.front().size();
  // 创建一个输出张量的向量 `outputs`
  std::vector<Tensor> outputs;
  // 遍历每个组件索引
  for (auto i : c10::irange(num_components)) {
    // 创建一个参数列表 `arg_list`
    std::vector<Tensor> arg_list;
    // 遍历 `unbound` 中的列表
    for (auto j : c10::irange(unbound.size())) {
      // 将当前组件的第 `i` 个张量添加到参数列表 `arg_list` 中
      arg_list.push_back(unbound[j][i]);
    }
    // 在指定维度 `dim` 上对 `arg_list` 中的张量执行 `cat` 操作，并将结果添加到 `outputs` 中
    outputs.push_back(at::cat(arg_list, dim));
  }

  // 注意：嵌套张量仅支持在 `dim 0` 上进行批处理
  // 从张量列表 `outputs` 中创建嵌套张量 `out_nt`
  auto out_nt = at::_nested_tensor_from_tensor_list(outputs);
  // 返回通过 `makeBatched` 函数创建的批处理张量，维度为 `0`，当前级别为 `get_current_level()`
  return makeBatched(out_nt, 0, get_current_level());
}

}

// 注册批处理库函数 `FuncTorchBatched`
TORCH_LIBRARY_IMPL(_, FuncTorchBatched, m) {
  // 使用 `batchedTensorForLoopFallback` 函数作为回退函数
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&batchedTensorForLoopFallback>());
}

// 注册 `aten` 命名空间下的批处理库函数 `FuncTorchBatched`
TORCH_LIBRARY_IMPL(aten, FuncTorchBatched, m) {
  // 以下仍为遗留，因为返回多个张量
  // 注册 `split.Tensor` 的批处理规则 `split_batching_rule`
  m.impl("split.Tensor", split_batching_rule);
  // 注册 `split_with_sizes` 的批处理规则 `split_with_sizes_batching_rule`
  m.impl("split_with_sizes", split_with_sizes_batching_rule);
  // 注册 `split_with_sizes_copy` 的批处理规则 `split_with_sizes_copy_batching_rule`
  m.impl("split_with_sizes_copy", split_with_sizes_copy_batching_rule);
  // 注册 `unbind.int` 的批处理规则 `unbind_batching_rule`
  m.impl("unbind.int", unbind_batching_rule);
  // 注册 `cat` 的批处理规则 `cat_batching_rule`
  m.impl("cat", cat_batching_rule);
  // 注册 `block_diag` 的批处理规则 `block_diag_batching_rule`
  m.impl("block_diag", block_diag_batching_rule);
  // 注册 `stack` 的批处理规则 `stack_batching_rule`

  // 以下仍为遗留，因为需要特殊的就地规则
  // 注册 `squeeze_` 的就地批处理规则 `squeeze__batching_rule`
  m.impl("squeeze_", squeeze__batching_rule);
  // 注册 `squeeze_.dim` 的就地批处理规则 `squeeze_dim__batching_rule`
  m.impl("squeeze_.dim", squeeze_dim__batching_rule);
  // 注册 `squeeze_.dims` 的就地批处理规则 `squeeze_dims__batching_rule`
  m.impl("squeeze_.dims", squeeze_dims__batching_rule);
  // 注册 `unsqueeze_` 的就地批处理规则 `unsqueeze__batching_rule`
  m.impl("unsqueeze_", unsqueeze__batching_rule);
  // 注册 `transpose_` 的就地批处理规则 `transpose__batching_rule`

  // 以下仍为遗留，因为这些非常复杂
  // 注册 `as_strided` 的批处理规则 `as_strided_batching_rule`
  m.impl("as_strided", as_strided_batching_rule);
  // 注册 `new_empty_strided` 的批处理规则 `new_empty_strided_batching_rule`
  m.impl("new_empty_strided", new_empty_strided_batching_rule);

}

// 注册批处理库函数 `BatchedNestedTensor`
TORCH_LIBRARY_IMPL(_, BatchedNestedTensor, m) {
  // 使用 `batchedNestedTensorForLoopFallback` 函数作为回退函数
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&batchedNestedTensorForLoopFallback>());
}

// TODO: 将此部分移到更合适的位置？
// 注册 `aten` 命名空间下的批处理库函数 `BatchedNestedTensor`
TORCH_LIBRARY_IMPL(aten, BatchedNestedTensor, m) {
  // 注册 `cat` 的批处理规则 `nested_cat_batching_rule`
  m.impl("cat", nested_cat_batching_rule);
}

// 结束命名空间 `at::functorch`
} // namespace at::functorch
```