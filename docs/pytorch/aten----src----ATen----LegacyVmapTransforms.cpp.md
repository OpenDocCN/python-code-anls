# `.\pytorch\aten\src\ATen\LegacyVmapTransforms.cpp`

```py
// 引入 ATen 库中所需的头文件
#include <ATen/LegacyVmapTransforms.h>
#include <ATen/ATen.h>
#include <ATen/core/IListRef.h>
#include <c10/util/irange.h>

// 定义 ATen 命名空间
namespace at {

// 检查 bdims 中的批处理维度是否按顺序出现在张量的前面
static bool areBdimsAtFrontInOrder(BatchDimsRef bdims) {
  // 遍历 bdims 的索引范围
  for (const auto idx : c10::irange(static_cast<int64_t>(bdims.size()))) {
    // 如果 bdims[idx] 的维度与 idx 不匹配，则返回 false
    if (bdims[idx].dim() != idx) {
      return false;
    }
  }
  // 如果所有批处理维度都按顺序在前面，则返回 true
  return true;
}

// 将 BatchedTensorImpl 的批处理维度置换到张量的前面，并返回物理张量
static Tensor permuteBatchDimsToFront(BatchedTensorImpl* batched) {
  // 获取批处理维度 bdims 和物理张量 physical_tensor
  auto bdims = batched->bdims();
  const Tensor& physical_tensor = batched->value();
  // 如果批处理维度已经按顺序在前面，则直接返回物理张量
  if (areBdimsAtFrontInOrder(bdims)) {
    return physical_tensor;
  }
  // 否则，需要重新排列张量的维度
  const auto sizes = physical_tensor.sizes();
  VmapDimVector permutation(sizes.size(), 0);
  permutation.reserve(sizes.size());
  const auto is_bdim = createBatchDimBitset(bdims);
  int64_t idx = 0;
  // 遍历 bdims，将其维度放在排列的最前面
  for (const auto& bdim : bdims) {
    permutation[idx++] = bdim.dim();
  }
  // 接着将非批处理维度放在排列的后面
  for (const auto ptr : c10::irange(sizes.size())) {
    if (is_bdim[ptr]) {
      continue;
    }
    permutation[idx++] = ptr;
  }
  // 使用排列后的维度对物理张量进行置换操作，并返回置换后的张量
  return physical_tensor.permute(permutation);
}

// 将逻辑张量 logical_tensor 转换为物理视图
VmapPhysicalView MultiBatchVmapTransform::logicalToPhysical(const Tensor& logical_tensor) {
  // 获取逻辑张量的 BatchedTensorImpl 指针
  auto* batched = maybeGetBatchedImpl(logical_tensor);
  // 断言 batched 不为空，即逻辑张量确实是一个 BatchedTensor
  TORCH_INTERNAL_ASSERT(
      batched,
      "logicalToPhysical(tensor) should only be passed a BatchedTensor");
  // 返回物理张量及其对应的批处理级别位集合
  return { permuteBatchDimsToFront(batched), createVmapLevelsBitset(batched->bdims()) };
}

// 返回物理视图中的批处理维度数量
int64_t VmapPhysicalView::numBatchDims() const {
  return levels_.count();
}

// 返回物理视图中的逻辑维度数量
int64_t VmapPhysicalView::numLogicalDims() const {
  return /*physical*/tensor_.dim() - numBatchDims();
}

// 获取物理视图中的物理维度
VmapDimVector VmapPhysicalView::getPhysicalDims(OptionalIntArrayRef opt_logical_dims) const {
  auto logical_ndim = numLogicalDims();
  // 初始化结果向量 result
  VmapDimVector result;
  result.reserve(logical_ndim);
  // 如果传入了逻辑维度 opt_logical_dims，则使用这些维度；否则使用默认维度顺序
  if (opt_logical_dims.has_value() && !opt_logical_dims.value().empty()) {
    auto logical_dims = opt_logical_dims.value();
    for (auto dim : logical_dims) {
      result.push_back(maybe_wrap_dim(dim, logical_ndim) + numBatchDims());
    }
  } else {
    for (int64_t dim = 0; dim < logical_ndim; dim++) {
      result.push_back(dim + numBatchDims());
    }
  }
  // 返回计算得到的物理维度向量
  return result;
}

// 获取物理视图中指定逻辑维度的物理维度
int64_t VmapPhysicalView::getPhysicalDim(int64_t logical_dim) const {
  auto logical_ndim = numLogicalDims();
  // 返回逻辑维度 logical_dim 对应的物理维度
  return maybe_wrap_dim(logical_dim, logical_ndim) + numBatchDims();
}

// 获取物理视图中的物理形状
VmapDimVector VmapPhysicalView::getPhysicalShape(IntArrayRef logical_shape) const {
  // 初始化结果向量 result
  VmapDimVector result;
  result.reserve(logical_shape.size() + numBatchDims());
  // 将物理张量的维度大小添加到结果向量的前部
  auto tensor_sizes = tensor_.sizes();
  result.insert(result.end(), tensor_sizes.begin(), tensor_sizes.begin() + numBatchDims());
  // 将逻辑形状的维度大小添加到结果向量的后部
  result.insert(result.end(), logical_shape.begin(), logical_shape.end());
  // 返回计算得到的物理形状向量
  return result;
}
// 计算从 levels_bitset 中的位集合中获取的前置批处理维度
static BatchDims computeFrontBatchDimsFromLevels(std::bitset<kVmapNumLevels> levels_bitset) {
  BatchDims bdims; // 创建一个空的批处理维度列表
  int64_t dim = 0; // 初始化维度计数器为0
  for (const auto level : c10::irange(kVmapNumLevels)) { // 遍历从0到kVmapNumLevels的范围
    if (!levels_bitset[level]) { // 如果 levels_bitset 中的某一位为0，则跳过该位
      continue;
    }
    bdims.emplace_back(level, dim++); // 将 level 和当前维度值 dim 添加到批处理维度列表中，并增加 dim 的值
  }
  return bdims; // 返回构建好的批处理维度列表
}

// 给定一个 Tensor 或 BatchedTensor，返回其所有 vmapped 维度前置的底层物理张量，
// 如果这些维度存在的话，以及一个包含在张量中存在的 vmap 级别的位集合。
static std::pair<Tensor,std::bitset<kVmapNumLevels>>
getPhysicalTensorAndLevels(const Tensor& self) {
  auto* batched = maybeGetBatchedImpl(self); // 获取可能的 Batched 实现指针
  if (batched) { // 如果存在 Batched 实现
    return {permuteBatchDimsToFront(batched), createVmapLevelsBitset(batched->bdims())}; // 返回前置批处理维度后的物理张量及其 vmap 级别的位集合
  }
  return {self, 0}; // 否则返回原始张量及空的位集合
}

// 给定一个 Tensor 或 BatchedTensor，创建该张量的物理视图，
// 使其具有 `requested_levels` 中的每个级别的批处理维度，
// 并具有 `requested_example_dim` 个非批处理维度。
//
// 此函数用于准备可以传递给广播操作的张量的物理视图。例如，当添加两个大小分别为 [B0, 3] 和 [B0, B1, 2, 3] 的 BatchedTensor 时，
// 其中 Bi 是批处理维度，我们必须分别对齐批处理维度和非批处理维度（以下简称为“示例”维度），
// 以生成大小为 [B0, 1, 1, 3] 和 [B0, B1, 2, 3] 的张量，使它们可以相加。
//
// 下面是在上述两个张量上使用 alignBatchDimsAtFront 的直接示例。
//
// 1) alignBatchDimsAtFront([B0, 3], requested_levels={0, 1}, requested_example_dim=2)
// 返回大小为 [B0, 1, 1, 3] 的物理视图，通过为级别 1 添加额外维度并通过添加另一个维度以填充示例维度至 2。
//
// 2) alignBatchDimsAtFront([B0, B1, 2, 3], requested_levels={0, 1}, requested_example_dim=2)
// 返回大小为 [B0, B1, 2, 3] 的物理视图
static Tensor alignBatchDimsAtFront(
    const Tensor& self, // 输入张量
    std::bitset<kVmapNumLevels> requested_levels, // 请求的批处理维度级别位集合
    int64_t requested_example_dim) { // 请求的示例维度数
  auto [physical_tensor, tensor_levels] = getPhysicalTensorAndLevels(self); // 获取输入张量的物理张量及其 vmap 级别位集合

  TORCH_INTERNAL_ASSERT(
    (tensor_levels | requested_levels) == requested_levels,
    "`requested_levels` must be a superset of `self`'s levels"); // 断言确保 requested_levels 是 self 的 levels 的超集

  auto physical_sizes = physical_tensor.sizes(); // 获取物理张量的大小

  const auto tensor_example_dim = (
    static_cast<int64_t>(physical_sizes.size())
    - /*num_batch_dims*/static_cast<int64_t>(tensor_levels.count())
  ); // 计算张量的示例维度数

  TORCH_INTERNAL_ASSERT(tensor_example_dim <= requested_example_dim); // 断言确保张量的示例维度数不大于请求的示例维度数

  if (tensor_levels == requested_levels && tensor_example_dim == requested_example_dim) {
    // 优化：如果物理张量已经是正确的形状，则不需要再次进行视图操作
    // 返回物理张量
    return physical_tensor;
  }

  // 创建一个大小为 requested_levels.count() + requested_example_dim 的向量 aligned_sizes，初始值为 1
  VmapDimVector aligned_sizes(requested_levels.count() + requested_example_dim, 1);

  // 对示例维度（非 bdim 维度）进行对齐
  // aligned_sizes[-tensor_example_dim:] = tensor_sizes[-tensor_example_dim:]
  std::copy(
      physical_sizes.rbegin(),                               // 从 physical_sizes 的倒数第 tensor_example_dim 个元素开始
      physical_sizes.rbegin() + tensor_example_dim,           // 到倒数第一个元素结束
      aligned_sizes.rbegin());                               // 复制到 aligned_sizes 的末尾对应位置

  // 对 bdims 进行对齐
  int64_t level = 0;
  int64_t tensor_dim = 0;
  for (const auto bdim : c10::irange(requested_levels.count())) {  // 对 requested_levels 中的每个元素进行循环
    // 确定 bdim 的级别
    while (!requested_levels[level]) level++;   // 找到第一个为真的 requested_levels[level]
    if (tensor_levels[level]) {                 // 如果对应的 tensor_levels[level] 为真
      aligned_sizes[bdim] = physical_sizes[tensor_dim++];  // 设置 aligned_sizes[bdim] 为 physical_sizes[tensor_dim]，然后增加 tensor_dim
    }
    level++;  // 增加 level，准备处理下一个 requested_levels 的元素
  }

  // 返回根据 aligned_sizes 对 physical_tensor 进行的视图操作
  return physical_tensor.view(aligned_sizes);
// 算法如下：
// 1. 确定 `logical_tensors` 中所有的 Vmap 集合级别。
// 2. 将所有批次维度移动到张量的前面，并添加额外维度大小为 1。此时，每个张量将具有每个集合级别的一个维度。
// 3. 计算批次大小。
// 4. 扩展每个物理张量，使其输出批次大小等于 `batch_sizes`。

VmapPhysicalViewVec
MultiBatchVmapTransform::logicalToPhysical(ITensorListRef logical_tensors) {
  // Figure out all of the collective vmap levels in `logical_tensors`.
  // 确定 `logical_tensors` 中所有的 Vmap 集合级别。
  std::bitset<kVmapNumLevels> collective_levels;
  for (const auto& logical_tensor : logical_tensors) {
    auto* batched = maybeGetBatchedImpl(logical_tensor);
    if (batched) {
      collective_levels |= createVmapLevelsBitset(batched->bdims());
    }
  }

  // Populate physical_tensors.
  // This contains a list of regular (non-Batched) Tensors where all of the
  // batch dims have been moved to the front of the tensor. Any previously
  // non-existing batch dims get added to the tensors as new dimensions of size 1.
  // 填充 physical_tensors。
  // 包含一个常规张量（非批处理）列表，其中所有批处理维度都已移至张量前面。
  // 任何之前不存在的批处理维度都作为大小为 1 的新维度添加到张量中。
  std::vector<Tensor> physical_tensors;
  int64_t num_batch_dims = collective_levels.count();
  for (const auto& logical_tensor : logical_tensors) {
    auto requested_example_dim = /*logical_dim*/logical_tensor.dim();
    auto physical_tensor = alignBatchDimsAtFront(
        logical_tensor, collective_levels, requested_example_dim);
    physical_tensors.push_back(std::move(physical_tensor));
  }

  // Compute batch_sizes
  // 计算批次大小
  VmapDimVector batch_sizes(num_batch_dims, 1);
  for (const auto& physical_tensor : physical_tensors) {
    auto physical_sizes = physical_tensor.sizes();
    for (const auto dim : c10::irange(num_batch_dims)) {
      if (physical_sizes[dim] != 1) {
        batch_sizes[dim] = physical_sizes[dim];
      }
    }
  }

  // Expand each physical_tensor so that it has batch sizes `batch_sizes`
  // 扩展每个 physical_tensor，使其具有批次大小 `batch_sizes`
  VmapPhysicalViewVec result;
  for (const auto& physical_tensor : physical_tensors) {
    VmapDimVector expanded_size(batch_sizes.begin(), batch_sizes.end());
    auto physical_sizes = physical_tensor.sizes();
    expanded_size.insert(
        expanded_size.end(),
        physical_sizes.begin() + num_batch_dims,
        physical_sizes.end());
    result.emplace_back(physical_tensor.expand(expanded_size), collective_levels);
  }
  return result;
}

// 获取逻辑张量列表的级别和最大逻辑维度
static std::pair<std::bitset<kVmapNumLevels>,int64_t>
getLevelsAndLargestLogicalDim(TensorList logical_tensors) {
  TORCH_INTERNAL_ASSERT(!logical_tensors.empty());
  std::bitset<kVmapNumLevels> levels;
  int64_t largest_logical_dim = -1;
  for (const auto& tensor : logical_tensors) {
    auto* batched = maybeGetBatchedImpl(tensor);
    if (batched) {
      levels = levels | createVmapLevelsBitset(batched->bdims());
    }
    auto tensor_logical_dim = /*logical dim*/tensor.dim();
    if (tensor_logical_dim > largest_logical_dim) {
      largest_logical_dim = tensor_logical_dim;
    }
  }
  // 返回级别和最大逻辑维度的 std::pair
  return {levels, largest_logical_dim};
}
    }
  }
  // 返回一个对象，包含 levels 和 largest_logical_dim 作为属性
  return { levels, largest_logical_dim };
} // namespace at



VmapPhysicalViewVec BroadcastingVmapTransform::logicalToPhysical(TensorList logical_tensors) {
  // 断言逻辑张量列表的大小为2，用于确保函数仅在这种情况下被测试过。请在移除此检查之前添加更多测试。
  TORCH_INTERNAL_ASSERT(
      logical_tensors.size() == 2,
      "This function has only been tested for two tensors. Please add more tests ",
      "before removing this check ");

  // 创建结果向量
  VmapPhysicalViewVec result;

  // 调用辅助函数，获取层级和最大逻辑维度
  auto [levels, largest_logical_dim] = getLevelsAndLargestLogicalDim(logical_tensors);

  // 遍历逻辑张量列表
  for (const auto& tensor : logical_tensors) {
    // 注意事项：可能不需要对张量进行对齐。
    // 例如，当添加大小分别为(B, 2)和(3, 2)的两个张量时，
    // 第一个张量是带批量维度B的批处理张量，第二个张量是常规张量，
    // 我们将返回大小为(B, 1, 2)和(1, 3, 2)的视图。然而，第二个张量上的视图是不必要的：
    // 广播语义允许添加大小为(B, 1, 2)和(3, 2)的两个张量！
    //
    // 如果这种不必要的视图成为问题，请考虑在未来优化它。这可能涉及创建一种新类型的VmapPhysicalView。
    auto aligned = alignBatchDimsAtFront(tensor, levels, largest_logical_dim);
    // 将对齐后的张量和层级添加到结果向量中
    result.emplace_back(std::move(aligned), levels);
  }
  // 返回结果向量
  return result;
}

VmapPhysicalToLogicalMap VmapPhysicalView::getPhysicalToLogicalMap() const {
  // 返回物理到逻辑映射对象，使用层级信息
  return VmapPhysicalToLogicalMap(levels_);
}

Tensor VmapPhysicalToLogicalMap::apply(const Tensor& physical_tensor) const {
  // 应用物理到逻辑映射，返回批处理后的张量
  return makeBatched(physical_tensor, computeFrontBatchDimsFromLevels(levels_));
}

void VmapPhysicalToLogicalMap::applyInplace(std::vector<Tensor>& physical_tensors) const {
  // 将每个物理张量应用物理到逻辑映射，直接修改物理张量向量
  for (auto & physical_tensor : physical_tensors) {
    physical_tensor = apply(physical_tensor);
  }
}

} // namespace at
```