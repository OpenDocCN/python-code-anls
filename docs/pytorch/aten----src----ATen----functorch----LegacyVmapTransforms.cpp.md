# `.\pytorch\aten\src\ATen\functorch\LegacyVmapTransforms.cpp`

```py
// 包含版权声明和头文件引用，声明所使用的许可证和依赖的头文件
#include <ATen/functorch/LegacyVmapTransforms.h>
#include <ATen/functorch/DynamicLayer.h>

#include <ATen/ATen.h>
#include <c10/util/irange.h>

namespace at::functorch {

// 定义一个命名空间 at::functorch，下面是命名空间中的函数和类的实现

// 接受一个 BatchedTensorImpl 对象，将所有批量维度排列到最前面，
// 然后返回张量的物理版本
static Tensor permuteBatchDimsToFront(const BatchedTensorImpl* batched) {
  // 获取批量张量的物理张量
  const Tensor& physical_tensor = batched->value();
  // 如果批量维度为 0，直接返回物理张量
  if (batched->bdim() == 0) {
    return physical_tensor;
  }
  // 获取物理张量的大小
  const auto sizes = physical_tensor.sym_sizes();
  // 创建一个排列顺序的向量，将批量维度排列到最前面
  VmapDimVector permutation(sizes.size(), 0);
  permutation.reserve(sizes.size());
  // 创建批量维度的位集合
  const auto is_bdim = createBatchDimBitset(batched->bdim());
  int64_t idx = 0;
  // 将批量维度加入排列顺序的最前面
  permutation[idx++] = batched->bdim();
  // 遍历剩余的维度，按顺序加入排列顺序中，跳过批量维度
  for (const auto ptr : c10::irange(0, sizes.size())) {
    if (is_bdim[ptr]) {
      continue;
    }
    permutation[idx++] = ptr;
  }
  // 返回根据新排列顺序排列后的物理张量
  return physical_tensor.permute(permutation);
}

// 将逻辑张量转换为物理视图，返回 VmapPhysicalView 对象
VmapPhysicalView MultiBatchVmapTransform::logicalToPhysical(const Tensor& logical_tensor) {
  // 尝试获取逻辑张量对应的 BatchedTensorImpl 对象
  auto* batched = maybeGetBatchedImpl(logical_tensor);
  // 断言确保 batched 不为空，逻辑张量应该是 BatchedTensor 类型
  TORCH_INTERNAL_ASSERT(
      batched,
      "logicalToPhysical(tensor) should only be passed a BatchedTensor");
  // 返回物理视图，包括重新排列批量维度和创建 vmap 级别的位集合
  return { permuteBatchDimsToFront(batched), createVmapLevelsBitset(batched->level()) };
}

// 返回物理视图的批量维度数量
int64_t VmapPhysicalView::numBatchDims() const {
  return levels_.count();
}

// 返回物理视图的逻辑维度数量
int64_t VmapPhysicalView::numLogicalDims() const {
  // 物理张量的维度减去批量维度的数量即为逻辑维度的数量
  return /*physical*/tensor_.dim() - numBatchDims();
}

// 根据逻辑维度数组返回对应的物理维度向量
VmapDimVector VmapPhysicalView::getPhysicalDims(IntArrayRef logical_dims) const {
  auto logical_ndim = numLogicalDims();
  // 创建结果向量，保留足够的空间存储逻辑维度对应的物理维度
  VmapDimVector result;
  result.reserve(logical_ndim);
  // 对每个逻辑维度进行转换，加上批量维度数量得到物理维度
  for (auto dim : logical_dims) {
    result.push_back(maybe_wrap_dim(dim, logical_ndim) + numBatchDims());
  }
  return result;
}

// 根据逻辑维度返回对应的物理维度
int64_t VmapPhysicalView::getPhysicalDim(int64_t logical_dim) const {
  auto logical_ndim = numLogicalDims();
  // 返回物理维度，通过逻辑维度加上批量维度数量得到
  return maybe_wrap_dim(logical_dim, logical_ndim) + numBatchDims();
}

// 根据逻辑形状返回对应的物理形状
VmapDimVector VmapPhysicalView::getPhysicalShape(IntArrayRef logical_shape) const {
  // 创建结果向量，包含物理张量的大小以及逻辑形状
  VmapDimVector result;
  result.reserve(logical_shape.size() + numBatchDims());
  // 获取物理张量的大小，并将其前 numBatchDims() 个维度加入结果中
  auto tensor_sizes = tensor_.sizes();
  result.insert(result.end(), tensor_sizes.begin(), tensor_sizes.begin() + numBatchDims());
  // 将逻辑形状的维度加入结果中
  result.insert(result.end(), logical_shape.begin(), logical_shape.end());
  return result;
}

// 命名空间 at::functorch 的结束
}
// 返回一个 SymDimVector 对象，表示物理形状，根据逻辑形状和当前 tensor 的批次维度数量
SymDimVector VmapPhysicalView::getPhysicalShape(c10::SymIntArrayRef logical_shape) const {
  SymDimVector result; // 创建一个 SymDimVector 对象，用于存储物理形状
  result.reserve(logical_shape.size() + numBatchDims()); // 预留空间，确保足够容纳逻辑形状加上批次维度数量的元素
  auto tensor_sizes = tensor_.sym_sizes(); // 获取当前 tensor 的符号化大小
  result.insert(result.end(), tensor_sizes.begin(), tensor_sizes.begin() + numBatchDims()); // 将当前 tensor 的前 numBatchDims() 个维度大小插入到 result 中
  result.insert(result.end(), logical_shape.begin(), logical_shape.end()); // 将逻辑形状的所有维度大小插入到 result 中
  return result; // 返回物理形状
}

// 根据 levels_bitset 计算前置批次维度的维度和级别，返回一个包含两个 int64_t 值的元组
static std::tuple<int64_t, int64_t> computeFrontBatchDimsFromLevels(std::bitset<kVmapNumLevels> levels_bitset) {
  int64_t level = 0; // 初始化级别为 0
  int64_t dim = 0; // 初始化维度为 0
  for (; level < kVmapNumLevels; level++) { // 遍历 levels_bitset 的每一个位
    if (!levels_bitset[level]) { // 如果当前级别位未设置
      continue; // 跳过当前循环
    }
    break; // 如果当前级别位设置了，则退出循环
  }
  return std::make_tuple(dim, level); // 返回计算出的维度和级别的元组
}

// 将 tensor 中的指定维度（如果有）移动到最前，并根据情况扩展维度大小，返回处理后的 Tensor
static Tensor moveDimToFrontAndExpand(Tensor tensor, optional<int64_t> dim, c10::SymInt size) {
  if (dim) { // 如果指定了要移动的维度
    tensor = tensor.movedim(*dim, 0); // 将指定维度移动到最前
  } else { // 如果未指定要移动的维度
    tensor = tensor.unsqueeze(0); // 在第0维度上增加一个维度
    auto expanded_sizes = tensor.sym_sizes().vec(); // 获取 tensor 的符号化大小并转为向量
    expanded_sizes[0] = size; // 更新第0维度的大小为 size
    tensor = tensor.expand_symint(expanded_sizes); // 根据 expanded_sizes 扩展 tensor 的大小
  }
  return tensor; // 返回处理后的 Tensor
}

// 多批次 Vmap 变换的逻辑到物理转换
VmapPhysicalViewVec
MultiBatchVmapTransform::logicalToPhysical(ITensorListRef logical_tensors) {
  auto cur_level = maybeCurrentDynamicLayer().value().layerId(); // 获取当前动态层的层ID
  c10::SymInt bdim_size = -1; // 初始化批次维度大小为 -1

  // 首先确定批次大小
  for (const auto& logical_tensor : logical_tensors) { // 遍历逻辑张量列表
    auto* batched = maybeGetBatchedImpl(logical_tensor); // 获取逻辑张量的批次信息
    if (!batched) { // 如果未找到批次信息
      continue; // 继续下一次循环
    }
    if (batched->level() != cur_level) { // 如果批次级别与当前级别不匹配
      continue; // 继续下一次循环
    }
    bdim_size = batched->value().sym_size(batched->bdim()); // 获取批次维度的大小
  }
  TORCH_INTERNAL_ASSERT(bdim_size != -1); // 断言批次维度大小已经确定

  std::bitset<kVmapNumLevels> levels; // 创建一个 kVmapNumLevels 位的位集合
  levels[cur_level] = true; // 设置当前级别的位为 true

  VmapPhysicalViewVec result; // 创建一个 VmapPhysicalViewVec 对象，用于存储物理视图
  for (const auto& logical_tensor : logical_tensors) { // 再次遍历逻辑张量列表
    auto* batched = maybeGetBatchedImpl(logical_tensor); // 获取逻辑张量的批次信息
    if (!batched || (batched->level() != cur_level)) { // 如果未找到批次信息或者批次级别不匹配当前级别
      // 在维度 0 上增加一个维度，并根据需要扩展到正确的形状
      c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched); // 保护功能的批次派遣键
      auto value = moveDimToFrontAndExpand(logical_tensor, {}, bdim_size); // 将逻辑张量移动到最前并扩展
      result.emplace_back(std::move(value), levels); // 将处理后的张量添加到结果中
      continue; // 继续下一次循环
    }
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched); // 保护功能的批次派遣键
    auto physical = batched->value(); // 获取批次化后的物理张量
    auto value = moveDimToFrontAndExpand(physical, batched->bdim(), bdim_size); // 将批次维度移动到最前并扩展
    result.emplace_back(std::move(value), levels); // 将处理后的张量添加到结果中
  }

  return result; // 返回逻辑到物理转换后的结果
}

// 将 tensor 中的指定维度（如果有）移动到最前，并在维度 0 上增加一个维度，返回处理后的 Tensor
static Tensor moveDimToFrontAndUnsqueeze(Tensor tensor, optional<int64_t> dim, int64_t example_ndim) {
  if (dim) { // 如果指定了要移动的维度
    // 将指定维度移动到最前并在维度 0 上增加一个维度
    tensor = tensor.movedim(*dim, 0);

    tensor = tensor.movedim(*dim, 0);
  } else { // 如果未指定要移动的维度
    tensor = tensor.unsqueeze(0); // 在维度 0 上增加一个维度
  }
  return tensor; // 返回处理后的 Tensor
}
    // 如果 dim 是一个元组，则使用 movedim 将 tensor 的维度移动到指定位置（0 表示最前面）
    tensor = tensor.movedim(*dim, 0);
  } else {
    // 否则，使用 unsqueeze 在最前面添加一个维度
    tensor = tensor.unsqueeze(0);
  }
  // 计算当前 tensor 的维度数减去 1，得到 ndim
  auto ndim = tensor.dim() - 1;
  // 如果 example_ndim 比 ndim 大，则需要继续在 tensor 的最前面添加维度
  for (int64_t i = 0; i < example_ndim - ndim; i++) {
    tensor = tensor.unsqueeze(1);
  }
  // 返回处理后的 tensor
  return tensor;
}

VmapPhysicalViewVec BroadcastingVmapTransform::logicalToPhysical(TensorList logical_tensors) {
  auto cur_level = maybeCurrentDynamicLayer().value().layerId();
  auto bdim_size = -1;

  // Figure out the batch size first
  for (const auto& logical_tensor : logical_tensors) {
    auto* batched = maybeGetBatchedImpl(logical_tensor);
    if (!batched || (batched->level() != cur_level)) {
      continue;
    }
    // 获取当前批处理级别的批次大小
    bdim_size = batched->value().size(batched->bdim());
  }
  // 确保批次大小已经被设置
  TORCH_INTERNAL_ASSERT(bdim_size != -1);

  std::bitset<kVmapNumLevels> levels;
  levels[cur_level] = true;

  // figure out the example ndim
  int64_t max_example_dim = -1;
  for (const auto& logical_tensor : logical_tensors) {
    // 计算逻辑张量的最大维度
    max_example_dim = std::max(logical_tensor.dim(), max_example_dim);
  }

  VmapPhysicalViewVec result;
  for (const auto& logical_tensor : logical_tensors) {
    auto* batched = maybeGetBatchedImpl(logical_tensor);
    if (!batched || (batched->level() != cur_level)) {
      // 如果没有批处理或者批处理级别不匹配，进行维度展开和添加维度0操作
      c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
      auto value = moveDimToFrontAndUnsqueeze(logical_tensor, {}, max_example_dim);
      result.emplace_back(std::move(value), levels);
      continue;
    }
    // 否则，处理批处理后的物理张量，执行维度操作
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    auto physical = batched->value();
    auto value = moveDimToFrontAndUnsqueeze(physical, batched->bdim(), max_example_dim);
    result.emplace_back(std::move(value), levels);
  }

  return result;
}

VmapPhysicalToLogicalMap VmapPhysicalView::getPhysicalToLogicalMap() const {
  // 返回物理到逻辑映射
  return VmapPhysicalToLogicalMap(levels_);
}

Tensor VmapPhysicalToLogicalMap::apply(const Tensor& physical_tensor) const {
  // 计算前向批处理维度，并创建批处理张量
  auto bdim_level = computeFrontBatchDimsFromLevels(levels_);
  return makeBatched(physical_tensor, std::get<0>(bdim_level), std::get<1>(bdim_level));
}

void VmapPhysicalToLogicalMap::applyInplace(std::vector<Tensor>& physical_tensors) const {
  // 应用物理到逻辑映射到一组张量
  for (const auto idx : c10::irange(0, physical_tensors.size())) {
    physical_tensors[idx] = apply(physical_tensors[idx]);
  }
}

} // namespace at::functorch
```