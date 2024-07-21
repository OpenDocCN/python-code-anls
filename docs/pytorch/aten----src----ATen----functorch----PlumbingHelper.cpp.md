# `.\pytorch\aten\src\ATen\functorch\PlumbingHelper.cpp`

```
// 包含所需的头文件
#include <ATen/functorch/TensorWrapper.h>
#include <ATen/functorch/DynamicLayer.h>
#include <ATen/functorch/BatchedTensorImpl.h>
#include <ATen/functorch/PlumbingHelper.h>

// 定义命名空间 at::functorch
namespace at::functorch {

// 函数用于检查 DynamicLayer 是否有值，如果无值则抛出异常
void vmap_check_escaped(const optional<DynamicLayer> &layer, const char* what) {
  // 使用 TORCH_CHECK 断言 layer 必须有值，否则抛出相应错误信息
  TORCH_CHECK(
    layer.has_value(),
    "Either your tensor may have escaped from inside a function being vmapped and this is a user error ",
    "(see https://pytorch.org/functorch/stable/ux_limitations.html), "
    "or there is an internal functorch error in `",
    what,
    "` Please file an issue if it looks like the latter"
  )
}

// 根据输入的 tensor 创建一个批处理版本的 tensor，并返回
Tensor makeBatched(const Tensor& tensor, optional<int64_t> bdim, int64_t level) {
  // 如果 bdim 有值
  if (bdim.has_value()) {
    // 使用 TORCH_INTERNAL_ASSERT 断言 bdim 的值必须大于等于 0
    TORCH_INTERNAL_ASSERT(*bdim >= 0);
    // 使用 TORCH_INTERNAL_ASSERT 断言 bdim 的值必须小于 tensor 的维度
    TORCH_INTERNAL_ASSERT(*bdim < tensor.dim());
    // 调用 makeBatched 函数创建批处理版本的 tensor，并返回结果
    return makeBatched(tensor, bdim.value(), level);
  }
  // 如果 bdim 没有值，则直接返回原始的 tensor
  return tensor;
}

// 根据输入的 tensors 向量创建批处理版本的 tensors 向量，并返回
std::vector<Tensor> makeBatchedVector(const std::vector<Tensor>& tensors, optional<int64_t> bdim, int64_t level) {
  // 创建一个空的 Tensor 向量 res，并保留与 tensors 向量相同大小的空间
  std::vector<Tensor> res;
  res.reserve(tensors.size());
  // 遍历输入的 tensors 向量
  for (const auto & tensor : tensors) {
    // 将每个 tensor 调用 makeBatched 函数创建批处理版本，并添加到 res 向量中
    res.emplace_back(makeBatched(tensor, bdim, level));
  }
  // 返回创建好的批处理版本的 tensors 向量 res
  return res;
}

// 解封一个 tensor 的批处理版本，返回解封后的 tensor 以及批处理维度 bdim
std::tuple<Tensor, std::optional<int64_t>> unwrapTensorAtLevel(const Tensor& tensor, int64_t level) {
  // 尝试获取 tensor 的 BatchedTensorImpl 指针
  auto* batched = maybeGetBatchedImpl(tensor);
  // 如果 batched 为空指针，则返回原始的 tensor 和空的批处理维度
  if (!batched) {
    return std::make_tuple(tensor, nullopt);
  }
  // 如果 batched 的批处理级别与给定 level 相同，则返回批处理后的值和批处理维度
  if (batched->level() == level) {
    return std::make_tuple(batched->value(), batched->bdim());
  }
  // 否则返回原始的 tensor 和空的批处理维度
  return std::make_tuple(tensor, nullopt);
}

// 检查给定 tensor 是否在指定 level 上已经进行了批处理
bool isBatchedAtLevel(const Tensor& tensor, int64_t level) {
  // 调用 unwrapTensorAtLevel 函数解封 tensor，然后检查是否有批处理维度
  auto result = unwrapTensorAtLevel(tensor, level);
  return std::get<1>(result).has_value();
}

// 检查给定 maybe_tensor 是否在指定 level 上已经进行了批处理
bool isBatchedAtLevel(const std::optional<Tensor>& maybe_tensor, int64_t level) {
  // 如果 maybe_tensor 没有值，则直接返回 false
  if (!maybe_tensor.has_value()) {
    return false;
  }
  // 调用 isBatchedAtLevel 函数检查 maybe_tensor 是否在指定 level 上已经进行了批处理
  return isBatchedAtLevel(*maybe_tensor, level);
}

// 检查 ITensorListRef 中的所有 tensor 是否在指定 level 上已经进行了批处理
bool isBatchedAtLevel(ITensorListRef tensors, int64_t level) {
  // 遍历 ITensorListRef 中的所有 tensor
  for (const auto& tensor : tensors) {
    // 如果其中任何一个 tensor 在指定 level 上已经进行了批处理，则返回 true
    if (isBatchedAtLevel(tensor, level)) {
      return true;
    }
  }
  // 如果所有 tensor 都未在指定 level 上进行批处理，则返回 false
  return false;
}

// 检查 c10::List<std::optional<Tensor>> 中的所有 tensor 是否在指定 level 上已经进行了批处理
bool isBatchedAtLevel(const c10::List<std::optional<Tensor>>& maybe_tensors, int64_t level) {
  // 遍历 c10::List<std::optional<Tensor>> 中的所有 maybe_tensor
  for (const auto idx : c10::irange(0, maybe_tensors.size())) {
    const auto& maybe_tensor = maybe_tensors.get(idx);
    // 如果其中任何一个 maybe_tensor 在指定 level 上已经进行了批处理，则返回 true
    if (isBatchedAtLevel(maybe_tensor, level)) {
      return true;
    }
  }
  // 如果所有 maybe_tensor 都未在指定 level 上进行批处理，则返回 false
  return false;
}

// 检查 ArrayRef<optional<Tensor>> 中的所有 tensor 是否在指定 level 上已经进行了批处理
bool areAnyBatchedAtLevel(ArrayRef<optional<Tensor>> maybe_tensors, int64_t level) {
  // 遍历 ArrayRef<optional<Tensor>> 中的所有 maybe_tensor
  for (const auto& maybe_tensor : maybe_tensors) {
    // 如果其中任何一个 maybe_tensor 在指定 level 上已经进行了批处理，则返回 true
    if (isBatchedAtLevel(maybe_tensor, level)) {
      return true;
    }
  }
  // 如果所有 maybe_tensor 都未在指定 level 上进行批处理，则返回 false
  return false;
}

} // namespace at::functorch
```