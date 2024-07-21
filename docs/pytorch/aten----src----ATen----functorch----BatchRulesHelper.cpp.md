# `.\pytorch\aten\src\ATen\functorch\BatchRulesHelper.cpp`

```
// 包含 BatchRulesHelper.h 和 WrapDimUtils.h 头文件，用于实现批处理相关的功能
#include <ATen/functorch/BatchRulesHelper.h>
#include <ATen/WrapDimUtils.h>

// 命名空间 at::functorch 内定义函数和类
namespace at::functorch {

// 将批处理维度移到张量的最前面
Tensor moveBatchDimToFront(const Tensor& tensor, optional<int64_t> maybe_batch_dim) {
  // 如果没有提供批处理维度，直接返回原始张量
  if (!maybe_batch_dim.has_value()) {
    return tensor;
  }
  // 如果批处理维度为 0，也直接返回原始张量
  if (maybe_batch_dim.value() == 0) {
    return tensor;
  }
  // 否则，使用 movedim 函数将批处理维度移到张量的最前面
  return tensor.movedim(maybe_batch_dim.value(), 0);
}

// 返回没有批处理维度的张量秩
int64_t rankWithoutBatchDim(const Tensor& tensor, optional<int64_t> maybe_batch_dim) {
  int64_t result = tensor.dim();
  // 如果提供了批处理维度，则秩减一
  if (maybe_batch_dim.has_value()) {
    result -= 1;
  }
  return result;
}

// 返回没有批处理维度的张量元素总数
int64_t numelWithoutBatchDim(const Tensor& tensor, optional<int64_t> maybe_batch_dim) {
  // 如果没有提供批处理维度，直接返回张量的元素总数
  if (!maybe_batch_dim) {
    return tensor.numel();
  }
  // 否则，返回在批处理维度上的张量尺寸除以张量在该维度上的尺寸，得到的元素总数
  return tensor.numel() / tensor.size(*maybe_batch_dim);
}

// 如果 maybe_empty 有值，返回 new_val；否则返回空值
optional<int64_t> valIfNonempty(optional<int64_t> maybe_empty, int64_t new_val) {
  if (maybe_empty.has_value()) {
    return new_val;
  }
  return nullopt;
}

// 获取张量的物理维度，考虑是否有批处理维度和逻辑维度
int64_t getPhysicalDim(const Tensor& tensor, bool has_batch_dim, int64_t logical_dim) {
  // 假设批处理维度在张量的最前面
  optional<int64_t> bdim = has_batch_dim ? optional<int64_t>(0) : nullopt;
  // 获取没有批处理维度的张量秩
  auto rank = rankWithoutBatchDim(tensor, bdim);
  // 包装逻辑维度以适应张量的秩
  auto wrapped_dim = maybe_wrap_dim(logical_dim, rank);
  // 如果有批处理维度，返回包装后的维度加一；否则返回包装后的维度
  if (has_batch_dim) {
    return wrapped_dim + 1;
  }
  return wrapped_dim;
}

// 获取张量的物理维度向量，考虑是否有批处理维度和逻辑维度向量
VmapDimVector getPhysicalDims(const Tensor& tensor, bool has_batch_dim, IntArrayRef logical_dims) {
  // 假设批处理维度在张量的最前面
  optional<int64_t> bdim = has_batch_dim ? optional<int64_t>(0) : nullopt;
  // 获取没有批处理维度的张量秩
  auto rank = rankWithoutBatchDim(tensor, bdim);
  // 创建结果向量
  VmapDimVector result;
  result.reserve(logical_dims.size());
  // 对于每个逻辑维度，将其包装并添加到结果向量中
  for (auto d : logical_dims){
    if (has_batch_dim) {
      result.push_back(maybe_wrap_dim(d, rank)+1);
    } else {
      result.push_back(maybe_wrap_dim(d, rank));
    }
  }
  return result;
}

// 如果没有批处理维度，直接返回张量；否则将其填充到逻辑秩指定的维度
Tensor maybePadToLogicalRank(const Tensor& tensor, optional<int64_t> has_bdim, int64_t logical_rank) {
  // 如果没有提供批处理维度，直接返回原始张量
  if (!has_bdim) {
    return tensor;
  }
  // 获取张量没有批处理维度的逻辑秩
  auto tensor_logical_rank = rankWithoutBatchDim(tensor, has_bdim);
  // 如果张量的逻辑秩大于或等于指定的逻辑秩，直接返回原始张量
  if (tensor_logical_rank >= logical_rank) {
    return tensor;
  }
  // 否则，创建一个新的符号尺寸向量，将其填充到逻辑秩指定的维度，并返回视图
  VmapSymDimVector new_sizes(tensor.sym_sizes().begin(), tensor.sym_sizes().end());
  for (int64_t i = 0; i < logical_rank - tensor_logical_rank; i++) {
    new_sizes.insert(new_sizes.begin() + 1, 1);
  }
  return tensor.view_symint(SymIntArrayRef{new_sizes.begin(), new_sizes.end()});
}

// 检查随机性类型和张量是否有批处理维度
void check_randomness(RandomnessType randomness, bool any_tensor_batched) {
  TORCH_CHECK(
    randomness != RandomnessType::Error,
    "vmap: called random operation while in randomness error mode. Please either use the "
    // 这里未完全注释，请继续完善


（最后一行未完成，需要继续注释。）
    # 检查在 vmap 中是否使用了 'same' 或 'different' 的随机性标志，或者在 vmap 外执行随机性操作。
    );
    
    # 使用 TORCH_CHECK 宏来检查条件，确保 randomess 不是 RandomnessType::Same 并且任何张量是批处理的情况。
    # 如果条件满足，抛出错误消息，指出 vmap 目前不支持同一随机性与批处理张量输入的组合。
    # 建议用户提交一个 functorch 的问题报告。
}

void check_randomness(RandomnessType randomness) {
  // 调用重载函数，不传入任何张量参数，避免相同错误
  check_randomness(randomness, false); // for ops that don't take in any tensors, don't hit same error
}

Tensor reshape_dim_into(int64_t src, int64_t dst, const Tensor& x) {
  auto x_dim = x.dim();
  // 确保源维度在有效范围内
  src = maybe_wrap_dim(src, x_dim);
  // 确保目标维度在有效范围内，返回的张量维度少一维
  dst = maybe_wrap_dim(dst, x_dim - 1); // Returned Tensor has one fewer dim
  // 复制张量尺寸到新的形状向量
  VmapDimVector new_shape(x.sizes().begin(), x.sizes().end());
  // 删除指定的源维度
  new_shape.erase(new_shape.begin() + src);
  // 更新目标维度的尺寸，乘以删除维度的尺寸
  new_shape[dst] *= x.sizes()[src];
  // 重新塑形张量，并返回结果
  return at::reshape(x.movedim(src, dst), new_shape);
}

Tensor reshape_dim_outof(int64_t src, int64_t size1, const Tensor& x) {
  // 确保源维度在有效范围内
  src = maybe_wrap_dim(src, x.dim());
  // 复制张量尺寸到形状向量
  VmapDimVector shape(x.sizes().begin(), x.sizes().end());
  if (shape[src] != 0) {
    // 注意：0 % 0 会导致浮点异常
    TORCH_INTERNAL_ASSERT(shape[src] % size1 == 0);
  }
  // 将 `size1` 拆分到 `0` 大小的维度
  int64_t size2 = 0;
  if (shape[src] != 0) {
    size2 = shape[src] / size1;
  }
  shape[src] = size1;
  shape.insert(shape.begin() + src + 1, size2);
  // 重新塑形张量，并返回结果
  return at::reshape(x, shape);
}

Tensor reshape_dim_outof_symint(int64_t src, const c10::SymInt& size1, const Tensor& x) {
  // 确保源维度在有效范围内
  src = maybe_wrap_dim(src, x.dim());
  // 复制符号化尺寸到形状向量
  c10::SymDimVector shape(x.sym_sizes().begin(), x.sym_sizes().end());
  if (shape[src] != 0) {
    // 注意：0 % 0 会导致浮点异常
    TORCH_INTERNAL_ASSERT(shape[src] % size1 == 0);
  }
  c10::SymInt size2;
  // 将 `size1` 拆分到 `0` 大小的维度
  if (shape[src] == 0) {
    size2 = 0;
  } else {
    size2 = shape[src] / size1;
  }
  shape[src] = size1;
  shape.insert(shape.begin() + src + 1, size2);
  // 重新塑形符号化张量，并返回结果
  return at::reshape_symint(x, shape);
}

void vmapIncompatibleInplaceError(const char* schema_name) {
  // 报告不兼容的就地操作错误
  TORCH_CHECK(false,
    "vmap: ", schema_name, "(self, *extra_args) is not possible because ",
    "there exists a Tensor `other` in extra_args that has more elements ",
    "than `self`. This happened due to `other` being vmapped over but ",
    "`self` not being vmapped over in a vmap. ",
    "Please try to use out-of-place operators instead of ", schema_name, ". ",
    "If said operator is being called inside the PyTorch framework, ",
    "please file a bug report instead.");
}

static void handleScalarTypePromotion(Tensor& logical_scalar_tensor, Tensor& second) {
  // 获取逻辑标量张量和第二张量的结果类型
  auto result_type = at::native::result_type(logical_scalar_tensor[0], second);
  // 如果逻辑标量张量的标量类型与结果类型不同，则转换为结果类型
  if (logical_scalar_tensor.scalar_type() != result_type) {
    logical_scalar_tensor = logical_scalar_tensor.to(result_type);
  }
  // 如果第二张量的标量类型与结果类型不同，则转换为结果类型
  if (second.scalar_type() != result_type) {
    second = second.to(result_type);
  }
}

std::tuple<Tensor, Tensor> _binary_pointwise_helper(
    const Tensor& tensor, optional<int64_t> tensor_batch_dim,
    const Tensor& other, optional<int64_t> other_batch_dim,
  // 计算张量和其他张量的逻辑最大秩
  auto tensor_logical_rank = rankWithoutBatchDim(tensor, tensor_batch_dim);
  auto other_logical_rank = rankWithoutBatchDim(other, other_batch_dim);
  auto max_logical_rank = std::max(tensor_logical_rank, other_logical_rank);

  // 将批处理维度移到张量的最前面
  auto tensor_ = moveBatchDimToFront(tensor, tensor_batch_dim);
  auto other_ = moveBatchDimToFront(other, other_batch_dim);

  // 在(0维，N维)的情况下，类型提升语义是不同的 :/
  if (do_type_promotion) {
    // 检查张量是否是逻辑标量（具有批处理维度但逻辑秩为0）
    auto tensor_is_logical_scalar = (tensor_logical_rank == 0 && tensor_batch_dim.has_value());
    // 检查其他张量是否是逻辑标量
    auto other_is_logical_scalar = (other_logical_rank == 0 && other_batch_dim.has_value());
    // 如果一个张量是逻辑标量而另一个不是，则处理类型提升
    if (tensor_is_logical_scalar && !other_is_logical_scalar) {
      handleScalarTypePromotion(tensor_, other_);
    }
    if (other_is_logical_scalar && !tensor_is_logical_scalar) {
      handleScalarTypePromotion(other_, tensor_);
    }
  }

  // 如果张量的维度不对齐，需要将它们对齐
  // 示例：Tensor[B, 3] + Tensor[2, 5, 3] -> Tensor[B, 1, 1, 3] + Tensor[2, 5, 3]
  // 注意：只有具有批处理维度的张量才需要修改
  tensor_ = maybePadToLogicalRank(tensor_, tensor_batch_dim, max_logical_rank);
  other_ = maybePadToLogicalRank(other_, other_batch_dim, max_logical_rank);

  // 返回更新后的张量
  return std::make_tuple(tensor_, other_);
}

// 结束命名空间 at::functorch
} // namespace at::functorch
```