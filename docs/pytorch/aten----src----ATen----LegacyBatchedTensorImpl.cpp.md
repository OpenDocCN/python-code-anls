# `.\pytorch\aten\src\ATen\LegacyBatchedTensorImpl.cpp`

```py
// 包含 ATen 库中所需的头文件，用于 LegacyBatchedTensorImpl 类的实现
#include <ATen/LegacyBatchedTensorImpl.h>

// 包含 WrapDimUtils.h，Exception.h 和 irange.h 中的相关内容，提供辅助功能和异常处理支持
#include <ATen/WrapDimUtils.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>

// 定义 at 命名空间，所有的 BatchedTensorImpl 实现都位于此命名空间中
namespace at {

// BatchedTensorImpl 构造函数的实现
BatchedTensorImpl::BatchedTensorImpl(Tensor value, BatchDims bdims)
  : TensorImpl(
      c10::DispatchKeySet(DispatchKey::Batched),  // 设置 DispatchKey 为 Batched
      value.dtype(),                             // 获取输入 Tensor 的数据类型
      value.device()                             // 获取输入 Tensor 的设备类型
    )
  , value_(std::move(value))                     // 将输入 Tensor 移动到 value_ 成员变量中
  , bdims_(std::move(bdims))                     // 将 BatchDims 移动到 bdims_ 成员变量中
{
  // 断言 value_ 已经定义，即不为 null
  TORCH_INTERNAL_ASSERT(value_.defined());

  // 设置存储访问出错时抛出异常的策略
  set_storage_access_should_throw();

  // 设置自定义大小和步幅策略为 CustomStrides
  set_custom_sizes_strides(SizesStridesPolicy::CustomStrides);

  // 检查不变量，确保对象状态正确
  checkInvariants();

  // 计算非批次维度数
  const auto public_dims = value_.dim() - bdims_.size();

  // 获取输入 Tensor 的大小和步幅信息
  const auto value_sizes = value_.sizes();
  const auto value_strides = value_.strides();

  // 调整 sizes_and_strides_ 向量的大小以容纳非批次维度的大小和步幅信息
  sizes_and_strides_.resize(public_dims);

  // 遍历非批次维度，计算其大小和步幅，并保存到 sizes_and_strides_ 向量中
  for (const auto dim : c10::irange(public_dims)) {
    // 计算实际维度索引，wrap_dim 参数为 false
    auto actual_dim = actualDim(dim, /*wrap_dim=*/false);
    sizes_and_strides_.size_at_unchecked(dim) = value_sizes.at(actual_dim);
    sizes_and_strides_.stride_at_unchecked(dim) = value_strides.at(actual_dim);
  }

  // 设置存储偏移量
  storage_offset_ = value_.storage_offset();

  // 刷新 numel 值
  refresh_numel();

  // 刷新 contiguous 属性
  refresh_contiguous();
}

// 计算实际维度索引的方法实现
int64_t BatchedTensorImpl::actualDim(int64_t dim, bool wrap_dim) const {
  // 如果 wrap_dim 为 true，则可能包装维度
  if (wrap_dim) {
    const auto ndim = sizes_and_strides_.size();
    dim = maybe_wrap_dim(dim, ndim);
  }

  // 创建 BatchDim 的位集合
  auto is_bdim = createBatchDimBitset(bdims_);

  // 计算非批次维度的数量
  int64_t non_bdim_count = 0;
  for (const auto actual_dim : c10::irange(kVmapMaxTensorDims)) {
    // 如果是批次维度，则跳过
    if (is_bdim[actual_dim]) {
      continue;
    }
    // 如果找到对应的非批次维度索引，则返回
    if (non_bdim_count == dim) {
      return actual_dim;
    }
    non_bdim_count++;
  }

  // 如果到达这里，则表示出现了断言错误，即 BatchedTensorImpl 的维度数量超过了 kVmapMaxTensorDims 的限制
  TORCH_INTERNAL_ASSERT(false);
}

// 检查不变量的方法实现
void BatchedTensorImpl::checkInvariants() const {
  int64_t prev_level = -1;
  for (const auto& bdim : bdims_) {
    // 断言批次维度的级别应该递增
    TORCH_INTERNAL_ASSERT(bdim.level() > prev_level);
    prev_level = bdim.level();
  }
}

// 返回自定义步幅的方法实现，目前与 strides_default 方法一致
IntArrayRef BatchedTensorImpl::strides_custom() const {
  return strides_default();
}

// TODO: 实现批次张量的正确连续性，然后将 sizes_strides_policy 设置回 Default
// 检查当前 BatchedTensorImpl 是否是连续的
bool BatchedTensorImpl::is_contiguous_custom(at::MemoryFormat memory_format) const {
  // 确保 memory_format 为 MemoryFormat::Contiguous，否则抛出错误信息
  TORCH_CHECK(memory_format == MemoryFormat::Contiguous,
      "NYI: querying is_contiguous inside of vmap for memory_format ",
      "other than torch.contiguous_format");
  // 返回当前对象是否连续的标志位
  return is_contiguous_;
}

// 以下是一些内部继承方法，我们不支持它们，不应该被调用
void BatchedTensorImpl::set_size(int64_t dim, int64_t new_size) {
  // 断言不支持为 BatchedTensorImpl 设置大小
  TORCH_INTERNAL_ASSERT(false, "Can't set_size for BatchedTensorImpl");
}
void BatchedTensorImpl::set_stride(int64_t dim, int64_t new_stride) {
  // 断言不支持为 BatchedTensorImpl 设置步长
  TORCH_INTERNAL_ASSERT(false, "Can't set_stride for BatchedTensorImpl");
}
void BatchedTensorImpl::set_storage_offset(int64_t storage_offset) {
  // 断言不支持为 BatchedTensorImpl 设置存储偏移量
  TORCH_INTERNAL_ASSERT(false, "Can't set_storage_offset for BatchedTensorImpl");
}

#ifdef DEBUG
// 检查是否存在存储，仅在调试模式下有效
bool BatchedTensorImpl::has_storage() const {
  // 断言在调试模式下，不应该设置 storage_ 字段
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!storage_, "BatchedTensorImpl assumes that storage_ is never set");
  // 返回 false，表示不具备存储
  return false;
}
#endif

// 返回 BatchedTensorImpl 的类型名称
const char* BatchedTensorImpl::tensorimpl_type_name() const {
  return "BatchedTensorImpl";
}

// 创建一个带批次维度的张量
Tensor makeBatched(const Tensor& tensor, BatchDims bdims) {
  // 确保输入张量不是已经带有批次维度的张量
  TORCH_INTERNAL_ASSERT(!isBatchedTensor(tensor));
  auto tensor_dim = tensor.dim();
  // 检查张量的维度是否符合 vmap 的限制
  TORCH_CHECK(
      tensor_dim <= kVmapMaxTensorDims,
      "vmap only supports tensors of dimensionality up to ", kVmapMaxTensorDims,
      "; got a tensor with dim ", tensor_dim);
  // 确保所有的批次维度级别在有效范围内
  TORCH_INTERNAL_ASSERT(
      std::all_of(bdims.begin(), bdims.end(),
          [](const BatchDim& bdim) { return bdim.level() < kVmapNumLevels; }),
      "We only support up to ", kVmapNumLevels, " nested vmaps");
  // 创建一个 BatchedTensorImpl 类型的张量并返回
  return at::detail::make_tensor<BatchedTensorImpl>(tensor, std::move(bdims));
}

// 为张量添加一个批次维度
Tensor addBatchDim(const Tensor& tensor, int64_t level, int64_t dim) {
  const auto* batched = maybeGetBatchedImpl(tensor);
  if (!batched) {
    // 如果张量不带有批次维度，则创建一个新的 BatchDims
    BatchDims bdims;
    bdims.emplace_back(level, dim);
    return at::detail::make_tensor<BatchedTensorImpl>(tensor, std::move(bdims));
  }
  // 如果张量已经带有批次维度，则在现有的 BatchDims 基础上添加新的批次维度
  BatchDims new_bdims(batched->bdims().begin(), batched->bdims().end());
  auto actual_bdim = batched->actualDim(dim, /*wrap_dim=*/true);
  new_bdims.emplace_back(level, actual_bdim);
  // 调用 makeBatched 函数创建带批次维度的张量并返回
  return makeBatched(batched->value(), std::move(new_bdims));
}

// 检查两个张量是否支持原地操作
bool inplaceIsVmapCompatible(const Tensor& self, const Tensor& other) {
  const auto* other_batched = maybeGetBatchedImpl(other);
  if (!other_batched) {
    // 如果 other 没有批次维度，则返回 true
    return true;
  }
  const auto* self_batched = maybeGetBatchedImpl(self);
  if (!self_batched) {
    // 如果 self 没有批次维度但 other 有，则返回 false
    return false;
  }
  // 获取 self 和 other 的批次维度级别，并比较它们是否兼容
  auto self_levels = createVmapLevelsBitset(self_batched->bdims());
  auto other_levels = createVmapLevelsBitset(other_batched->bdims());
  return self_levels == (self_levels | other_levels);
}

} // namespace at
```