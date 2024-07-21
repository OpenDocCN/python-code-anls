# `.\pytorch\aten\src\ATen\functorch\BatchRulesScatterOps.cpp`

```py
// 包含头文件，这些头文件提供了所需的函数和数据结构声明
#include <ATen/functorch/BatchRulesHelper.h>
#include <ATen/Operators.h>
#include <ATen/functorch/PlumbingHelper.h>
#include <ATen/functorch/BatchedFallback.h>
#include <ATen/native/TensorAdvancedIndexing.h>
#include <ATen/native/IndexKernel.h>
#include <ATen/native/IndexingUtils.h>
#include <torch/library.h>

// 进入函数库的命名空间 at::functorch
namespace at::functorch {

// 匿名命名空间，定义静态函数

// 检查是否有任何 optional<int64_t> 对象有值
static bool any_has_value(ArrayRef<optional<int64_t>> bdims) {
  for (const auto& bdim : bdims) {
    if (bdim.has_value()) {
      return true;
    }
  }
  return false;
}

// 获取前导的未定义或空张量的数量
static int64_t get_num_leading_nones(ArrayRef<optional<Tensor>> indices) {
  int64_t result = 0;
  for (const auto& idx : indices) {
    if (!idx.has_value() || !idx->defined()) {
      result++;
    } else {
      return result;
    }
  }
  return result;
}

// 获取最大的逻辑维度索引
static int64_t get_max_index_logical_dim(
    ArrayRef<optional<Tensor>> indices,
    ArrayRef<optional<int64_t>> indices_bdims) {
  int64_t max_logical_dim = -1;
  // 断言 indices 和 indices_bdims 的大小相同且非空
  TORCH_INTERNAL_ASSERT(indices.size() == indices_bdims.size());
  TORCH_INTERNAL_ASSERT(!indices.empty());
  // 遍历索引数组
  for (const auto i : c10::irange(0, indices.size())) {
    const auto& maybe_tensor = indices[i];
    if (!maybe_tensor.has_value() || !maybe_tensor->defined()) {
      continue;
    }
    // 调用 rankWithoutBatchDim 函数获取逻辑维度
    auto logical_dim = rankWithoutBatchDim(maybe_tensor.value(), indices_bdims[i]);
    // 更新最大逻辑维度
    max_logical_dim = std::max(logical_dim, max_logical_dim);
  }
  return max_logical_dim;
}
// 定义一个函数 batchIndices，接受多个参数，并返回一个 optional Tensor 的 vector
static std::vector<optional<Tensor>> batchIndices(
  // 输入参数 indices 是一个 optional Tensor 的 ArrayRef
  ArrayRef<optional<Tensor>> indices,
  // 输入参数 indices_bdims 是一个 optional int64_t 的 ArrayRef
  ArrayRef<optional<int64_t>> indices_bdims,
  // 输入参数 batch_size 是一个 int64_t，表示批大小
  int64_t batch_size,
  // 输入参数 self_bdim 是一个 optional int64_t，表示自身张量的批维度
  optional<int64_t> self_bdim,
  // 输入参数 values_bdim 是一个 optional int64_t，默认为 nullopt
  optional<int64_t> values_bdim = nullopt) {
  // 有三种主要情况需要处理：
  // 1. self 张量被批处理，indices/values 没有被批处理
  // 在这种情况下，我们只需在 indices 前面添加一个 None，以实现跨 self 的批维度广播索引。
  //
  // 2. self 张量未被批处理，某些 indices 被批处理。
  // 在这种情况下，我们无需进行任何操作 - indices 将自动广播以与未批处理的 self 一起使用。
  //
  // 3. self 张量被批处理，某些 indices 被批处理。
  // 在这种情况下，我们只需添加一个 arange，用于沿第一维（即批维度）索引。还需要确保这与其余 indices 广播。
  //
  // 在所有三种情况下，根据高级索引是否相邻，我们需要对输出进行排列。
  // 有关更多详细信息，请参阅注释：[advanced indexing (index.Tensor) batch rule]。
  //
  // 还值得一提的是另一种情况 - 布尔张量 indices。如果我们有“批处理”的布尔张量 indices，那是无法表示的，因为每个批会导致具有不同值的张量。
  std::vector<optional<Tensor>> indices_;

  // 获取 indices 中最大的逻辑维度
  int64_t maxLogicalRank = get_max_index_logical_dim(indices, indices_bdims);
  // 检查 indices 是否被批处理
  bool indices_batched = any_has_value(indices_bdims);

  // 遍历 indices
  for (size_t i = 0; i < indices.size(); i++) {
    auto index = indices[i];
    // 如果 index 有值且元素数不为 0
    if (index.has_value() && index->numel() != 0) {
      const auto idx_bdim = indices_bdims[i];
      // 将 index 移动到逻辑维度的前面，可能需要填充批维度
      indices_.emplace_back(maybePadToLogicalRank(moveBatchDimToFront(index.value(), idx_bdim), idx_bdim, maxLogicalRank));
      // 如果 index 的数据类型为布尔型且 indices_bdims[i] 有值，则抛出运行时错误
      if (index.value().dtype() == kBool && indices_bdims[i].has_value()) {
        throw std::runtime_error("vmap: We do not support batching operators that can support dynamic shape. Attempting to batch over indexing with a boolean mask.");
      }
    } else {
      // 否则直接将 index 添加到 indices_ 中
      indices_.push_back(index);
    }
  }

  // 计算最大索引维度
  auto maxIndexDim = maxLogicalRank;
  if (indices_batched || values_bdim.has_value()) {
    maxIndexDim += 1;
  }

  // 根据条件修改 indices_
  if (!indices_batched && self_bdim.has_value()) {
    indices_.insert(indices_.begin(), nullopt);
  } else if (indices_batched && !self_bdim.has_value()) {
    // 如果 indices 被批处理而 self_bdim 没有值，则不做任何操作
  } else if (indices_batched && (self_bdim.has_value() || values_bdim.has_value())) {
    // 否则，创建一个 arange 张量，用于索引第一维，并确保与其余 indices 广播
    auto arange_index = at::arange(0, batch_size);
    while (arange_index.dim() < maxIndexDim) {
      arange_index = arange_index.unsqueeze(-1);
    }
    // TODO: 这是一个 O(N) 操作
    indices_.insert(indices_.begin(), arange_index);
  }
  // 返回处理后的 indices_
  return indices_;
}

// 定义一个“高级索引”，即一个非平凡张量的选择对象（即它不表示 : ）。
static bool is_advanced_index(const optional<Tensor>& idx) {
  // 如果 idx 没有值，则返回 false
  if (!idx.has_value()) {
    return false;
  }
  // 如果 idx 未定义，则返回 false
  if (!idx->defined()) {
    return false;
  }


继续完成后续部分的注释可能超出了单个代码块的长度限制。
    # 返回布尔值 `false`
    return false;
  }
  # 返回布尔值 `true`
  return true;
// See NOTE: [advanced indices adjacent] for definition
// 检查是否高级索引相邻，即连续出现的高级索引区域数是否不超过1
static bool are_advanced_indices_adjacent(ArrayRef<optional<Tensor>> indices) {
  int64_t num_advanced_indices_regions = 0;  // 记录高级索引区域的数量
  bool in_advanced_indices_region = false;   // 标记是否在高级索引区域内
  for (const auto& idx : indices) {
    if (!in_advanced_indices_region && is_advanced_index(idx)) {
      num_advanced_indices_regions++;  // 进入新的高级索引区域
      in_advanced_indices_region = true;
      continue;
    }
    if (in_advanced_indices_region && !is_advanced_index(idx)) {
      in_advanced_indices_region = false;  // 离开高级索引区域
      continue;
    }
  }
  return num_advanced_indices_regions <= 1;  // 返回是否高级索引区域数量不超过1的判断结果
}

// 给定一个张量 tensor[B, <first_region>, <second_region>, ...]
// 将两个区域交换，得到张量 tensor[B, <second_region>, <first_region>, ...]
//
// 具体来说，给定：
// - tensor: Tensor[B, 2, 3, 4, 5, 6, 7, 8]
// - first_region_size: 2
// - second_region_size: 3
// 产生结果：
// - result: Tensor[B, 4, 5, 6, 2, 3, 7, 8]
//                     -------  ----
//                     第二区域   第一区域
static Tensor swap_regions(const Tensor& tensor, int64_t first_region_size, int64_t second_region_size) {
  VmapDimVector permutation(tensor.dim(), 0);  // 创建一个初始排列为 [0, 1, 2, ..., dim-1]
  std::iota(permutation.begin(), permutation.end(), 0);  // 填充排列为 [0, 1, 2, ..., dim-1]
  std::rotate(
      permutation.begin() + 1,
      permutation.begin() + 1 + first_region_size,
      permutation.begin() + 1 + first_region_size + second_region_size);  // 旋转排列，交换两个区域的位置
  return tensor.permute(permutation);  // 返回根据新排列的置换后的张量
}

// 对索引进行批处理规则
std::tuple<Tensor,optional<int64_t>> index_batch_rule(
    const Tensor& self,
    optional<int64_t> self_bdim,
    ArrayRef<optional<Tensor>> indices,
    bool advanced_indices_are_adjacent) {
  if (advanced_indices_are_adjacent) {
    // Case 1
    // self: Tensor[B, 5, 6, 7, 8]
    // indices: [:, Tensor[2, 2], Tensor[2, 2], :]
    // batched_indices: [:, :, Tensor[2, 2], Tensor[2, 2], :]
    // res: Tensor[B, 5, 2, 2, 8]
    return std::make_tuple(res, 0);  // 返回结果和维度
  } else {
    // Case 2
    // self: Tensor[5, 6, 7, 8]
    // indices: [:, :, Tensor[B, 2, 2], Tensor[2, 2]]
    // batched_indices: indices (无变化)
    // res: Tensor[5, 6, B, 2, 2]
    return std::make_tuple(res, num_leading_nones);  // 返回结果和前导的 none 数量
  }

  // Case 3: self_batched and indices_batched
  TORCH_INTERNAL_ASSERT(self_batched && indices_batched);  // 断言确保 self 和 indices 都是批处理的
  if (!advanced_indices_are_adjacent) {
    // self: Tensor[B, 5, 6, 7, 8]
    // indices: [:, Tensor[B, 2, 2], :, Tensor[2, 2]]
    // batched_indices: [arange(B).expand(B, 2, 2), :, Tensor[B, 2, 2], :, Tensor[2, 2]]
    // res: Tensor[B, 2, 2, 5, 6, 8]
    return std::make_tuple(res, 0);  // 返回结果和 0
  }
}
    // 返回一个包含 res 和整数 0 的 std::tuple 对象
    return std::make_tuple(res, 0);
  }
  // 如果 num_leading_nones 等于 0，执行以下代码块
  if (num_leading_nones == 0) {
    // self: Tensor[B, 5, 6, 7, 8]，表示包含五个维度的张量
    // indices: [Tensor[B, 2, 2], Tensor[2, 2], :, :]，包含两个 Tensor 作为索引的列表
    // batched_indices: [arange(B).expand(B, 2, 2), Tensor[B, 2, 2], Tensor[2, 2], :, :]，批处理索引
    // res: Tensor[B, 2, 2, 7, 8]，结果张量的形状
    return std::make_tuple(res, 0);
  }
  // 复杂情况，indices 中的高级索引相邻，但在 batched_indices 中不再相邻
  //
  // self: Tensor[B, 5, 6, 7, 8, 9]，表示包含六个维度的张量
  // indices: [:, :, Tensor[B, 2, 3], Tensor[2, 3], :]，包含两个高级索引的列表
  // batched_indices: [arange(B).expand(B, 2, 3), :, :, Tensor[B, 2, 3], Tensor[2, 3], :]，批处理索引
  // res: Tensor[B, 2, 3, 5, 6, 9]，期望的结果张量形状
  // expected: Tensor[B, 5, 6, 2, 3, 9]，预期的结果张量形状
  //
  // 解决方案是重新排列维度，直到获得正确的形状。
  // 结果设置为 [B, <maxIndexDim>, <leading_nones>, ...]
  // 我们只需将 <leading_nones> 移动到 <maxIndexDim> 之前，以生成 [B, <leading_nones>, <maxIndexDim>, ...]
  return std::make_tuple(swap_regions(res, max_index_dim, num_leading_nones), 0);
}

// 由于代码生成器中不支持 List<optional<Tensor>>，因此需要进行索引处理
Tensor index_plumbing(const Tensor & self, const List<optional<Tensor>> & indices
) {
  // 排除 FuncTorchBatched 分发键的保护作用
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  // 获取当前可能的动态层
  auto maybe_layer = maybeCurrentDynamicLayer();
  // 检查动态层是否已经逃逸，如果逃逸则抛出异常
  vmap_check_escaped(maybe_layer, "index_plumbing");
  // 获取当前层的层级 ID
  int64_t cur_level = maybe_layer->layerId();
  // 如果张量和索引都不在当前层进行批处理，则直接使用标准索引操作
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(indices, cur_level)) {
    return at::index(self, indices);
  }
  // 解包张量到当前层级的值和批处理维度
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  // 初始化索引值和批处理维度的容器
  std::vector<optional<Tensor>> indices_value;
  std::vector<optional<int64_t>> indices_bdims;
  // 遍历索引列表中的每一个索引
  for (const auto&& indRef : indices) {
      // 获取当前索引的可选值
      optional<Tensor> ind = indRef;
      optional<Tensor> index;
      optional<int64_t> index_bdim;
      // 如果索引有值，则解包索引到当前层级的值和批处理维度
      if (ind.has_value()) {
        std::tie(index, index_bdim) = unwrapTensorAtLevel(ind.value(), cur_level);
      }
    // 将解包后的索引值和批处理维度放入容器中
    indices_value.push_back(index);
    indices_bdims.push_back(index_bdim);
  }
  // 使用批处理规则对张量和索引进行索引操作
  auto results = index_batch_rule(self_value, self_bdim, indices_value, indices_bdims);
  // 创建批处理后的张量，使用结果值和批处理维度，并指定当前层级
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

namespace {
  // 代码主要来自于 https://github.com/pytorch/pytorch/blob/fb0e27d38a8fdab4e1c14d6378c9e41cb30fd6a3
  // /aten/src/ATen/native/TensorAdvancedIndexing.cpp#L294-L312
  // 计算索引后的形状
  VmapDimVector compute_indexed_shape(const Tensor &src, TensorList indices_list)
  {
    int64_t dims_before = 0, dims_indexed = 0;
    IntArrayRef replacement_shape;
    // 遍历索引列表中的每个维度
    for (const auto dim : c10::irange(indices_list.size())) {
      // 如果当前维度的索引未定义
      if (!indices_list[dim].defined()) {
        // 如果是第一个未定义的维度
        if (dims_indexed == 0) {
          dims_before++;
        }
      } else {
        // 否则，增加已索引维度的计数，并设置替换形状为当前索引的大小
        dims_indexed++;
        replacement_shape = indices_list[dim].sizes();
      }
    }

    // 将原张量的形状复制到新的形状向量中
    auto shape = VmapDimVector(src.sizes());
    // 计算要删除的维度范围
    int64_t end = dims_before + dims_indexed;
    shape.erase(shape.begin() + dims_before, shape.begin() + end);
    // 在指定位置插入替换形状的维度
    shape.insert(shape.begin() + dims_before, replacement_shape.begin(), replacement_shape.end());
    // 返回新的形状向量
    return shape;
  }

  // 代码主要来自于 https://github.com/pytorch/pytorch/blob/fb0e27d38a8fdab4e1c14d6378c9e41cb30fd6a3
  // /aten/src/ATen/native/TensorAdvancedIndexing.cpp#L379-L405
  // 获取索引后的形状
  VmapDimVector get_indexed_shape(Tensor self, const torch::List<std::optional<at::Tensor>> &orig)
  {
    // 检查索引张量的类型
    at::native::checkIndexTensorTypes(orig);
    // 扩展 BoolTensor（掩码）或 ByteTensor（掩码）成一个或多个 LongTensor
    auto indices = at::native::expandTensors(self, orig);
    // 广播所有索引张量

together
    try {
      // 尝试扩展索引，使其与 self 的维度兼容
      indices = at::expand_outplace(indices);
    } catch (std::exception &e) {
      // 捕获异常，如果索引张量无法广播到一起，则抛出错误
      TORCH_CHECK_INDEX(false, "shape mismatch: indexing tensors could not be broadcast together"
                               " with shapes ");
    }
    // 添加缺失的空张量，使 indices 的长度与 self 的维度相同
    while (indices.size() < static_cast<size_t>(self.dim())) {
      indices.emplace_back();
    }
    // 如果非空索引不是全部连续的，将 self 和 indices 转置，使它们在前面连续
    if (!at::native::hasContiguousSubspace(indices)) {
      std::tie(self, indices) = at::native::transposeToFront(self, indices);
    }
    // 计算使用索引后的 self 的形状
    return compute_indexed_shape(self, indices);
  }

  std::tuple<Tensor, std::vector<optional<Tensor>>, Tensor>
  index_put_batch_rule_helper(const Tensor &self,
                              optional<int64_t> self_bdim,
                              ArrayRef<optional<Tensor>> indices,
                              ArrayRef<optional<int64_t>> indices_bdims,
                              const Tensor &values,
                              optional<int64_t> values_bdim,
                              optional<int64_t> opt_batch_size = {}) {

    // 将具有 batch 维度的 self 和 values 移动到前面
    Tensor self_ = moveBatchDimToFront(self, self_bdim);
    Tensor values_ = moveBatchDimToFront(values, values_bdim);
    // 对于 inplace 变种 `index_put_` 和 `_index_put_impl_`，我们在这里确定 batch_size
    // 而对于 `index_put`，则在函数外部确定
    const auto batch_size = opt_batch_size ? opt_batch_size.value() : self_.size(0);
    // 确保 self_ 和 values_ 具有指定的 batch 维度
    self_ = ensure_has_bdim(self_, self_bdim.has_value(), batch_size);
    values_ = ensure_has_bdim(values_, values_bdim.has_value(), batch_size);
    // 断言 indices 和 indices_bdims 的大小一致
    TORCH_INTERNAL_ASSERT(indices.size() == indices_bdims.size());

    // 确保 self 的 batch 维度在 0 位置
    const auto indices_ = batchIndices(indices, indices_bdims, batch_size, /*self_bdim=*/0, values_bdim);

    // 获取使用 indices_ 进行索引后的 self 的形状
    auto indexed_shape = get_indexed_shape(self_, List<optional<Tensor>>(indices_));

    // 处理 values 的广播支持
    // 例如，如果 `indexed_shape.size()` 是 5，而 `values` 的形状是 (N, 2, 3)，则以下块将将 `values` 重塑为 (N, 1, 1, 2, 3)。
    // 如果 indexed_shape 的大小大于 values_ 的维度数
    if ((int64_t)indexed_shape.size() > values_.dim()) {
      // 获取 values_ 的大小信息
      auto values_sizes = values_.sizes();

      // 计算需要添加的单位维度数（用于将值广播到 indexed_shape）
      auto n_unit_dims = indexed_shape.size() - values_sizes.size();
      // 创建新的 values_ 的形状，包括新增的单位维度
      VmapDimVector new_values_shape(values_sizes.size() + n_unit_dims);

      // 添加批处理维度
      new_values_shape[0] = batch_size;

      // 插入用于广播的单位维度
      for (const auto idx : c10::irange(n_unit_dims)) {
        // 批处理维度已经填充过了
        new_values_shape[idx + 1] = 1;
      }

      // 将剩余的维度从 values_sizes 复制到 new_values_shape
      for (const auto idx : c10::irange(1, values_sizes.size())) {
        // 批处理维度和单位维度已经填充过了
        new_values_shape[idx + n_unit_dims] = values_sizes[idx];
      }

      // 重新视图化 values_，按照新的形状
      values_ = values_.view(new_values_shape);
    }

    // 返回 self_、indices_ 和 values_ 的元组
    return std::make_tuple(self_, indices_, values_);
  }

  // 解包当前级别的 self、indices 和 values
  auto unpackSelfAndIndicesAndValuesAtCurrentLevel(const Tensor &self,
                                                   const List<optional<Tensor>> &indices,
                                                   const Tensor &values, int64_t cur_level)
  {
    // 解包 self 在当前级别的值和批处理维度
    auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);

    // 初始化 indices 的值和批处理维度的容器
    std::vector<optional<Tensor>> indices_value;
    std::vector<optional<int64_t>> indices_bdims;

    // 遍历 indices
    for (const auto &&indRef : indices) {
      // 获取当前索引的 optional<Tensor>
      optional<Tensor> ind = indRef;
      optional<Tensor> index;
      optional<int64_t> index_bdim;

      // 如果 ind 有值，则解包 index 和 index_bdim
      if (ind.has_value()) {
        std::tie(index, index_bdim) = unwrapTensorAtLevel(ind.value(), cur_level);
      }

      // 将解包后的 index 和 index_bdim 存入对应容器
      indices_value.push_back(index);
      indices_bdims.push_back(index_bdim);
    }

    // 解包 values 在当前级别的值和批处理维度
    auto [values_value, values_bdim] = unwrapTensorAtLevel(values, cur_level);

    // 返回解包后的 self_value、self_bdim、indices_value、indices_bdims、values_value、values_bdim 的元组
    return std::make_tuple(self_value, self_bdim, indices_value, indices_bdims, values_value, values_bdim);
  }
}  // namespace



void index_put__batch_rule(
    const Tensor& self,
    optional<int64_t> self_bdim,
    ArrayRef<optional<Tensor>> indices,
    ArrayRef<optional<int64_t>> indices_bdims,
    const Tensor& values,
    optional<int64_t> values_bdim,
    bool accumulate) {
  // 检查是否存在 self_bdim 值，如果不存在则抛出不兼容 vmap 错误
  if (!self_bdim.has_value()) {
    vmapIncompatibleInplaceError("index_put_");
  }
  // 调用辅助函数，获取处理后的 self、indices、values
  auto [self_, indices_, values_] = index_put_batch_rule_helper(
      self, self_bdim, indices, indices_bdims, values, values_bdim);
  // 调用 ATen 的 index_put_ 方法，对 self_ 进行索引赋值操作
  at::index_put_(self_, List<optional<Tensor>>(indices_), values_, accumulate);
}

// 因为代码生成不支持 List<optional<Tensor>>，所以进行“plumbing”处理
Tensor& index_put__plumbing(Tensor & self, const List<optional<Tensor>> & indices
, const Tensor & values, bool accumulate) {
  // 排除 DispatchKey 为 FuncTorchBatched 的情况
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  // 检查动态层是否存在，如果不存在则抛出错误
  vmap_check_escaped(maybe_layer, "index_put__plumbing");
  // 获取当前动态层的层级
  int64_t cur_level = maybe_layer->layerId();
  // 如果在当前层级下，self、indices、values 都未进行批处理，则直接调用 self 的 index_put_ 方法
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(indices, cur_level) && !isBatchedAtLevel(values, cur_level)) {
    return self.index_put_(indices, values, accumulate);
  }
  // 否则，解包 self 和 indices、values，并调用 index_put__batch_rule 进行处理
  auto [self_value, self_bdim, indices_value, indices_bdims, values_value, values_bdim] =
      unpackSelfAndIndicesAndValuesAtCurrentLevel(self, indices, values, cur_level);
  index_put__batch_rule(self_value, self_bdim, indices_value, indices_bdims, values_value, values_bdim, accumulate);
  return self;
}

void _index_put_impl__batch_rule(
    const Tensor& self,
    optional<int64_t> self_bdim,
    ArrayRef<optional<Tensor>> indices,
    ArrayRef<optional<int64_t>> indices_bdims,
    const Tensor& values,
    optional<int64_t> values_bdim,
    bool accumulate,
    bool unsafe) {
  // 检查是否存在 self_bdim 值，如果不存在则抛出不兼容 vmap 错误
  if (!self_bdim.has_value()) {
    vmapIncompatibleInplaceError("_index_put_impl_");
  }
  // 调用辅助函数，获取处理后的 self、indices、values
  auto [self_, indices_, values_] = index_put_batch_rule_helper(
      self, self_bdim, indices, indices_bdims, values, values_bdim);
  // 调用 ATen 的 _index_put_impl_ 方法，对 self_ 进行索引赋值操作
  at::_index_put_impl_(self_, List<optional<Tensor>>(indices_), values_, accumulate, unsafe);
}

// 因为代码生成不支持 List<optional<Tensor>>，所以进行“plumbing”处理
Tensor &_index_put_impl__plumbing(Tensor &self, const List<optional<Tensor>> &indices,
                                  const Tensor &values, bool accumulate, bool unsafe) {
  // 排除 DispatchKey 为 FuncTorchBatched 的情况
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  // 检查动态层是否存在，如果不存在则抛出错误
  vmap_check_escaped(maybe_layer, "_index_put_impl__plumbing");
  // 获取当前动态层的层级
  int64_t cur_level = maybe_layer->layerId();
  // 如果在当前层级下，self、indices、values 都未进行批处理，则直接调用 self 的 _index_put_impl_ 方法
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(indices, cur_level) && !isBatchedAtLevel(values, cur_level)) {
    # 调用 at::_index_put_impl_ 方法，将给定的参数传递进去，并返回结果
    return at::_index_put_impl_(self, indices, values, accumulate, unsafe);
    }
    
    # 解构赋值，从 unpackSelfAndIndicesAndValuesAtCurrentLevel 函数返回的元组中获取变量
    auto [self_value, self_bdim, indices_value, indices_bdims, values_value, values_bdim] =
        unpackSelfAndIndicesAndValuesAtCurrentLevel(self, indices, values, cur_level);
    
    # 调用 _index_put_impl__batch_rule 方法，传递解构赋值得到的参数，进行批处理规则处理
    _index_put_impl__batch_rule(self_value, self_bdim, indices_value, indices_bdims, values_value, values_bdim, accumulate, unsafe);
    
    # 返回修改后的 self 对象，表示索引操作完成
    return self;
// 定义静态函数 maybe_permute_values，用于根据索引情况对输入的 values 进行可能的重新排列
static Tensor maybe_permute_values(
    const Tensor& values, // 输入的原始数据张量
    ArrayRef<optional<Tensor>> orig_indices, // 原始索引的可选张量数组引用
    ArrayRef<optional<int64_t>> orig_indices_bdims) { // 原始索引的可选维度数组引用
  bool indices_batched = any_has_value(orig_indices_bdims); // 检查是否存在批量索引
  bool advanced_indices_are_adjacent = are_advanced_indices_adjacent(orig_indices); // 检查高级索引是否相邻
  auto num_leading_nones = get_num_leading_nones(orig_indices); // 获取前导的 None 值数量
  auto max_index_dim = get_max_index_logical_dim(orig_indices, orig_indices_bdims); // 获取逻辑维度中的最大索引维度

  // NB: values has its B dimension at the front
  // 注意：values 的 B 维度位于最前面

  // 如果索引不是批量的
  if (!indices_batched) {
    if (advanced_indices_are_adjacent) {
      // self: Tensor[B, 5, 6, 7, 8]
      // indices: [:, Tensor[2, 2], Tensor[2, 2], :]
      // batched_indices: [:, :, Tensor[2, 2], Tensor[2, 2], :]
      // 所需的 values: Tensor[B, 5, 2, 2, 8]
      return values;
    }
    // self: Tensor[B, 5, 6, 7]
    // indices: [Tensor[2, 2], :, Tensor[2, 2]]
    // batched_indices: [:, Tensor[2, 2], :, Tensor[2, 2]]
    // 所需的 values: Tensor[2, 2, B, 6]
    return values.movedim(0, max_index_dim); // 将 values 张量按照指定的维度移动
  }

  // 如果高级索引不相邻
  if (!advanced_indices_are_adjacent) {
    // self: Tensor[B, 5, 6, 7, 8]
    // indices: [:, Tensor[B, 2, 2], :, Tensor[2, 2]]
    // batched_indices: [arange(B).expand(B, 2, 2), :, Tensor[B, 2, 2], :, Tensor[2, 2]]
    // 所需的 values: Tensor[B, 2, 2, 5, 7]
    return values;
  }

  // 换句话说，在 batched_indices 中，高级索引不再相邻
  if (num_leading_nones == 0) {
    // self: Tensor[B, 5, 6, 7, 8]
    // indices: [Tensor[B, 2, 2], Tensor[2, 2], :, :]
    // batched_indices: [arange(B).expand(B, 2, 2), Tensor[B, 2, 2], Tensor[2, 2], :, :]
    // 所需的 values: Tensor[B, 2, 2, 7, 8]
    return values;
  }

  // 这是复杂的情况。在 indices 中，高级索引是相邻的。
  // 在 batched_indices 中，高级索引不再相邻
  //
  // self: Tensor[B, 5, 6, 7, 8, 9]
  // indices: [:, :, Tensor[B, 2, 3], Tensor[2, 3], :]
  // batched_indices: [arange(B).expand(B, 2, 3), :, :, Tensor[B, 2, 3], Tensor[2, 3], :]
  // 所需的 values: Tensor[B, 2, 3, 5, 6, 9]
  // 实际的 values: Tensor[B, 5, 6, 2, 3, 9]
  //
  // 解决方法是移动维度，直到获得正确的形状。
  // values 被设置为 [B, <leading_nones>, <maxIndexDim>, ...]
  // 我们只需将 <maxIndexDim> 移动到 <leading_nones> 之前，以生成
  // [B, <maxIndexDim>, <leading_nones>, ...]
  return swap_regions(values, num_leading_nones, max_index_dim); // 交换 values 张量的区域以达到所需的形状
}
    // 遍历索引列表，查找具有非空值的索引，并计算批处理大小
    for (size_t i = 0; i < indices.size(); i++) {
      // 如果索引维度标记为真，并且索引值不为空
      if (indices_bdims[i] && indices[i].has_value()) {
        // 根据索引的特定维度大小计算批处理大小
        batch_size = indices[i].value().size(*indices_bdims[i]);
        // 找到后立即结束循环
        break;
      }
    }
  }

  // 调用辅助函数获取索引放置批处理规则的返回值，解构元组
  auto [self_, indices_, values_] = index_put_batch_rule_helper(
      self, self_bdim, indices, indices_bdims, values, values_bdim, batch_size);

  // 为什么需要对 values 进行置换？
  // 详见“NOTE [Advanced indexing (index.Tensor) batch rule]”部分的详细说明，
  // 总体来说，index_put 实际上执行以下操作：
  // - result = self_.clone()
  // - result[indices_] = values
  // - 返回 result
  // 现在的问题是，result[indices_] 可能返回一个形状与 values 相同但经过置换的 Tensor。
  // 这是因为 result[indices_] 的形状取决于原始索引是否具有“相邻高级索引”，
  // 而批量处理的 indices_ 可能会改变“相邻高级索引”的属性。
  // 因此，需要对 values_ 进行可能的置换以适应这种变化。
  values_ = maybe_permute_values(values_, indices, indices_bdims);

  // 使用 index_put 函数进行索引放置操作，将结果存储在 result 中
  auto result = at::index_put(self_, List<optional<Tensor>>(indices_), values_, accumulate);
  // 返回包含 result 和 0 的元组
  return std::make_tuple(result, 0);
}

// 这里是函数 index_put_plumbing 的实现，用于处理索引赋值操作
Tensor index_put_plumbing(const Tensor & self, const List<optional<Tensor>> & indices,
                          const Tensor & values, bool accumulate) {
  // 暂时排除 FuncTorchBatched 调度键
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  // 获取当前动态层
  auto maybe_layer = maybeCurrentDynamicLayer();
  // 检查动态层是否逃逸，用于 Vmap
  vmap_check_escaped(maybe_layer, "index_put_plumbing");
  // 获取当前层级
  int64_t cur_level = maybe_layer->layerId();
  // 如果 self、indices、values 都不在当前层级的批处理中
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(indices, cur_level) && !isBatchedAtLevel(values, cur_level)) {
    // 调用 Tensor 的 index_put 方法进行索引赋值操作
    return self.index_put(indices, values, accumulate);
  }
  // 将 self、indices、values 解包到当前层级
  auto [self_value, self_bdim, indices_value, indices_bdims, values_value, values_bdim] =
      unpackSelfAndIndicesAndValuesAtCurrentLevel(self, indices, values, cur_level);
  // 调用索引赋值的批处理规则
  auto results = index_put_batch_rule(self_value, self_bdim, indices_value, indices_bdims, values_value, values_bdim, accumulate);
  // 返回结果的批处理形式
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// 匿名命名空间，定义 scatter 操作的批处理规则模板函数
template<typename Func, typename ...Args>
std::tuple<Tensor,optional<int64_t>> scatter_batch_rule(
    Func f,
    const Tensor& self, optional<int64_t> self_bdim,
    int64_t dim,
    const Tensor& index, optional<int64_t> index_bdim,
    const Scalar& value, Args... args) {
  // 获取 self 和 index 的逻辑秩（不包括批处理维度）
  auto self_logical_rank = rankWithoutBatchDim(self, self_bdim);
  auto index_logical_rank = rankWithoutBatchDim(index, index_bdim);
  // 获取批处理大小
  auto batch_size = get_bdim_size2(self, self_bdim, index, index_bdim);

  // 将 self 和 index 的批处理维度移动到最前面
  auto self_ = moveBatchDimToFront(self, self_bdim);
  auto index_ = moveBatchDimToFront(index, index_bdim);

  // 如果 self 的逻辑秩为 0，则在最后增加一个维度
  if (self_logical_rank == 0) {
    self_ = self_.unsqueeze(-1);
  }
  // 如果 index 的逻辑秩为 0，则在最后增加一个维度
  if (index_logical_rank == 0) {
    index_ = index_.unsqueeze(-1);
  }
  // 确保 self 和 index 拥有批处理维度
  self_ = ensure_has_bdim(self_, self_bdim.has_value(), batch_size);
  index_ = ensure_has_bdim(index_, index_bdim.has_value(), batch_size);
  // 获取物理维度
  auto physical_dim = getPhysicalDim(self_, /*has_batch_dim*/true, dim);

  // 调用传入的函数 f 进行 scatter 操作
  auto result = f(self_, physical_dim, index_, value, args...);
  // 如果 self 的逻辑秩为 0，则去除结果的最后一个维度
  if (self_logical_rank == 0) {
    result = result.squeeze(-1);
  }
  // 返回结果和 batch_dim
  return std::make_tuple(result, 0);
}

// 定义 scatter 操作的批处理规则模板函数（处理 src 参数的情况）
template <typename Func, typename ...Args>
inline std::tuple<Tensor,optional<int64_t>> scatter_batch_rule(
    Func f,
    const Tensor& self, optional<int64_t> self_bdim,
    int64_t dim,
    const Tensor& index, optional<int64_t> index_bdim,
    const Tensor& src, optional<int64_t> src_bdim, Args... args) {
  // 获取 self、index、src 的逻辑秩（不包括批处理维度）
  auto self_logical_rank = rankWithoutBatchDim(self, self_bdim);
  auto index_logical_rank = rankWithoutBatchDim(index, index_bdim);
  auto src_logical_rank = rankWithoutBatchDim(src, src_bdim);
  // 获取批处理大小
  auto batch_size = get_bdim_size3(self, self_bdim, index, index_bdim, src, src_bdim);

  // 将 self、index、src 的批处理维度移动到最前面
  auto self_ = moveBatchDimToFront(self, self_bdim);
  auto index_ = moveBatchDimToFront(index, index_bdim);
  auto src_ = moveBatchDimToFront(src, src_bdim);

  // 如果 self 的逻辑秩为 0，则在最后增加一个维度
  if (self_logical_rank == 0) {
    self_ = self_.unsqueeze(-1);
  }
  // 如果 index 的逻辑秩为 0，则在最后增加一个维度
  if (index_logical_rank == 0) {
    index_ = index_.unsqueeze(-1);
  }
    // 将 self_ 张量在最后一个维度上增加一个维度
    self_ = self_.unsqueeze(-1);
  }
  // 如果 index_logical_rank 等于 0，则在 index_ 张量的最后一个维度上增加一个维度
  if (index_logical_rank == 0) {
    index_ = index_.unsqueeze(-1);
  }
  // 如果 src_logical_rank 等于 0，则在 src_ 张量的最后一个维度上增加一个维度
  if (src_logical_rank == 0) {
    src_ = src_.unsqueeze(-1);
  }
  // 确保 self_ 张量具有指定的批量维度，根据 self_bdim 是否有值和给定的 batch_size
  self_ = ensure_has_bdim(self_, self_bdim.has_value(), batch_size);
  // 确保 index_ 张量具有指定的批量维度，根据 index_bdim 是否有值和给定的 batch_size
  index_ = ensure_has_bdim(index_, index_bdim.has_value(), batch_size);
  // 确保 src_ 张量具有指定的批量维度，根据 src_bdim 是否有值和给定的 batch_size
  src_ = ensure_has_bdim(src_, src_bdim.has_value(), batch_size);
  // 获取具有指定维度的物理维度，包括批量维度，根据 self_ 张量和给定的 dim
  auto physical_dim = getPhysicalDim(self_, /*has_batch_dim*/true, dim);

  // 调用函数 f 处理 self_, index_, src_ 张量及其它参数 args，返回处理结果 result
  auto result = f(self_, physical_dim, index_, src_, args...);
  // 结果 result 应该与 self 张量具有相同的形状
  if (self_logical_rank == 0) {
    // 如果 self_logical_rank 等于 0，则在 result 张量的最后一个维度上减少一个维度
    result = result.squeeze(-1);
  }
  // 返回结果 result 和整数 0 的 tuple
  return std::make_tuple(result, 0);
}

} // namespace



std::tuple<Tensor,optional<int64_t>> scatter_value_batch_rule(
    const Tensor& self, optional<int64_t> self_bdim,
    int64_t dim,
    const Tensor& index, optional<int64_t> index_bdim,
    const Scalar& value) {
  return scatter_batch_rule(ATEN_FN2(scatter, value),
                            self, self_bdim, dim, index, index_bdim, value);
}


// 定义 scatter_value_batch_rule 函数，用于处理 scatter 操作中按值散播的规则
std::tuple<Tensor,optional<int64_t>> scatter_value_batch_rule(
    const Tensor& self, optional<int64_t> self_bdim,
    int64_t dim,
    const Tensor& index, optional<int64_t> index_bdim,
    const Scalar& value) {
  // 调用 scatter_batch_rule 函数，传递 scatter 操作的函数符号、输入参数和值
  return scatter_batch_rule(ATEN_FN2(scatter, value),
                            self, self_bdim, dim, index, index_bdim, value);
}



std::tuple<Tensor,optional<int64_t>> scatter_src_batch_rule(
    const Tensor& self, optional<int64_t> self_bdim,
    int64_t dim,
    const Tensor& index, optional<int64_t> index_bdim,
    const Tensor& src, optional<int64_t> src_bdim) {
  return scatter_batch_rule(ATEN_FN2(scatter, src),
                            self, self_bdim, dim, index, index_bdim, src, src_bdim);
}


// 定义 scatter_src_batch_rule 函数，用于处理 scatter 操作中按源张量散播的规则
std::tuple<Tensor,optional<int64_t>> scatter_src_batch_rule(
    const Tensor& self, optional<int64_t> self_bdim,
    int64_t dim,
    const Tensor& index, optional<int64_t> index_bdim,
    const Tensor& src, optional<int64_t> src_bdim) {
  // 调用 scatter_batch_rule 函数，传递 scatter 操作的函数符号、输入参数和源张量
  return scatter_batch_rule(ATEN_FN2(scatter, src),
                            self, self_bdim, dim, index, index_bdim, src, src_bdim);
}



std::tuple<Tensor,optional<int64_t>> scatter_add_batch_rule(
    const Tensor& self, optional<int64_t> self_bdim,
    int64_t dim,
    const Tensor& index, optional<int64_t> index_bdim,
    const Tensor& src, optional<int64_t> src_bdim) {
  return scatter_batch_rule(ATEN_FN(scatter_add),
                            self, self_bdim, dim, index, index_bdim, src, src_bdim);
}


// 定义 scatter_add_batch_rule 函数，用于处理 scatter_add 操作的规则
std::tuple<Tensor,optional<int64_t>> scatter_add_batch_rule(
    const Tensor& self, optional<int64_t> self_bdim,
    int64_t dim,
    const Tensor& index, optional<int64_t> index_bdim,
    const Tensor& src, optional<int64_t> src_bdim) {
  // 调用 scatter_batch_rule 函数，传递 scatter_add 操作的函数符号、输入参数和源张量
  return scatter_batch_rule(ATEN_FN(scatter_add),
                            self, self_bdim, dim, index, index_bdim, src, src_bdim);
}



std::tuple<Tensor,optional<int64_t>> scatter_reduce_batch_rule(
    const Tensor& self, optional<int64_t> self_bdim,
    int64_t dim,
    const Tensor& index, optional<int64_t> index_bdim,
    const Tensor& src, optional<int64_t> src_bdim,
    const c10::string_view reduce) {
  return scatter_batch_rule(ATEN_FN2(scatter, reduce),
                            self, self_bdim, dim, index, index_bdim, src, src_bdim, reduce);
}


// 定义 scatter_reduce_batch_rule 函数，用于处理 scatter 操作中带有 reduce 参数的规则
std::tuple<Tensor,optional<int64_t>> scatter_reduce_batch_rule(
    const Tensor& self, optional<int64_t> self_bdim,
    int64_t dim,
    const Tensor& index, optional<int64_t> index_bdim,
    const Tensor& src, optional<int64_t> src_bdim,
    const c10::string_view reduce) {
  // 调用 scatter_batch_rule 函数，传递 scatter 操作的函数符号、输入参数、源张量和 reduce 参数
  return scatter_batch_rule(ATEN_FN2(scatter, reduce),
                            self, self_bdim, dim, index, index_bdim, src, src_bdim, reduce);
}



std::tuple<Tensor,optional<int64_t>> scatter_value_reduce_batch_rule(
    const Tensor& self, optional<int64_t> self_bdim,
    int64_t dim,
    const Tensor& index, optional<int64_t> index_bdim,
    const Scalar& src,
    const c10::string_view reduce) {
  return scatter_batch_rule(ATEN_FN2(scatter, value_reduce),
                            self, self_bdim, dim, index, index_bdim, src, reduce);
}


// 定义 scatter_value_reduce_batch_rule 函数，用于处理 scatter 操作中带有 value_reduce 参数的规则
std::tuple<Tensor,optional<int64_t>> scatter_value_reduce_batch_rule(
    const Tensor& self, optional<int64_t> self_bdim,
    int64_t dim,
    const Tensor& index, optional<int64_t> index_bdim,
    const Scalar& src,
    const c10::string_view reduce) {
  // 调用 scatter_batch_rule 函数，传递 scatter 操作的函数符号、输入参数、源标量和 reduce 参数
  return scatter_batch_rule(ATEN_FN2(scatter, value_reduce),
                            self, self_bdim, dim, index, index_bdim, src, reduce);
}



std::tuple<Tensor,optional<int64_t>> gather_batch_rule(
    const Tensor& self, optional<int64_t> self_bdim,
    int64_t dim,
    const Tensor& index, optional<int64_t> index_bdim,
    bool sparse_grad) {
  auto self_logical_rank = rankWithoutBatchDim(self, self_bdim);
  auto index_logical_rank = rankWithoutBatchDim(index, index_bdim);
  auto batch_size = get_bdim_size2(self, self_bdim, index, index_bdim);

  auto self_ = moveBatchDimToFront(self, self_bdim);
  auto index_ = moveBatchDimToFront(index, index_bdim);

  if (self_logical_rank == 0) {
    self_ = self_.unsqueeze(-1);
  }
  if (index_logical_rank == 0) {
    index_ = index_.unsqueeze(-1);
  }
  self_ = ensure_has_bdim(self_, self_bdim.has_value(), batch_size);
  index_ = ensure_has_bdim(index_, index_bdim.has_value(), batch_size);
  auto physical_dim = getPhysicalDim(self_, /*has_batch_dim*/true, dim);

  auto result = at::gather(self_, physical_dim, index_, sparse_grad);
  // result should have same rank as index
  if (index_logical_rank == 0) {


// 定义 gather_batch_rule 函数，用于处理 gather 操作的规则
std::tuple<Tensor,optional<int64_t>> gather_batch_rule(
    const Tensor& self, optional<int64_t> self_bdim,
    int64_t dim,
    const Tensor& index, optional<int64_t> index_bdim,
    bool sparse_grad) {
  // 计算输入张量和索引张量的逻辑维度
  auto self_logical_rank = rankWithoutBatchDim(self, self_bdim);
  auto index_logical_rank = rankWithoutBatchDim(index, index_bdim);
  // 获取批处理大小
  auto batch_size = get_bdim_size2(self, self_bdim, index, index_bdim);

  // 将批处理维度移动到张量前面
  auto self_ = moveBatchDimToFront(self, self_bdim);
  auto index_ = moveBatchDimToFront(index, index_bdim);

  // 如果输入张量或索引张量的逻辑维度为 0，需扩展维度
  if (self_logical_rank == 0) {
    self_ = self_.unsqueeze(-1);
  }
  if (index_logical_rank == 0) {
    index_ = index_.unsqueeze(-1);
  }

  // 确保张量具有批处理维度
  self_ = ensure_has_bdim(self_, self_bdim.has_value(), batch_size);
  index_ = ensure_has_bdim(index_, index_bdim.has_value(), batch_size);

  // 获取物理维度
  auto physical_dim = getPhysicalDim(self_, /*has_batch_dim*/true, dim);

  // 执行 gather 操作
    result = result.squeeze(-1);
  }
  返回从 result 中移除维度为 1 的维度后的结果
  return std::make_tuple(result, 0);
}

Tensor get_expanded_index(const Tensor& index, IntArrayRef self_size, int64_t dim) {
  if (index.dim() == 0) {
    return index.expand(self_size);
  }
  dim = maybe_wrap_dim(dim, static_cast<int64_t>(self_size.size()));

  // setup new_index_shape as [BS, 1, ..., idx_size, ..., 1]
  // to reshape index_
  auto idx_size = index.size(0);  // get non-batch size of index tensor
  Tensor index_;
  {
    VmapDimVector new_index_shape(self_size.size(), 1);
    new_index_shape[dim] = idx_size;
    index_ = index.view(new_index_shape);
  }
  // Now apply expand to index_
  {
    VmapDimVector new_index_shape = {self_size.begin(), self_size.end()};
    new_index_shape[dim] = idx_size;
    index_ = index_.expand(new_index_shape);
  }
  return index_;
}

Tensor index_select_decomp(const Tensor &self, int64_t dim, const Tensor &index)
{
  Tensor index_ = index;
  if (self.dim() > index.dim()) {
    index_ = get_expanded_index(index, self.sizes(), dim);
  }

  auto result = at::gather(self, dim, index_);

  // output of gather has same dimension as `index` while
  // output of index_select has same dimension as self
  // Eg. t = torch.tensor(1)
  //     idx = torch.tensor([0])
  //     torch.index_select(t, 0, idx) # 0-D
  //     torch.gather(t, 0, idx) # 1-D
  if (self.dim() == 0 && result.dim() != 0) {
    result = result.squeeze(-1);
  }

  return result;
}

Tensor index_copy_decomp(
    const Tensor &self, int64_t dim,
    const Tensor &index, const Tensor &source)
{
  Tensor index_ = index;
  if (self.dim() > index.dim()) {
    index_ = get_expanded_index(index, self.sizes(), dim);
  }

  return at::scatter(self, dim, index_, source);  ;
}

// Note [Fix vmap slice_scatter]
// registers a decomposition for `slice_scatter` that calls into `slice.src`
// *_scatter operators have some special semantics though, that we can't easily
// through a decomposition: slice_scatter's output needs to have the same
// size, size, strides and storage_offset as the input.
Tensor slice_scatter_decomp(const Tensor &self, const Tensor &src,
                            int64_t dim, std::optional<int64_t> start,
                            std::optional<int64_t> end, int64_t step)
{
  auto idx = at::arange(start.value_or(0), end.value_or(self.size(dim)), step, self.options().dtype(kLong));
  idx = get_expanded_index(idx, self.sizes(), dim);
  return at::scatter(self, dim, idx, src);
}

Tensor select_scatter_decomp(
    const Tensor &self, const Tensor &source,
    int64_t dim, int64_t index)
{
  // supports negative index
  index = maybe_wrap_dim(index, self.size(dim));
  auto index_ = at::scalar_tensor(index, self.options().dtype(kLong));

  return at::scatter(self, dim, index_.expand_as(self), source.unsqueeze(dim).expand_as(self));
}

std::tuple<Tensor, optional<int64_t>> diagonal_scatter_batch_rule(
    const Tensor &self, std::optional<int64_t> self_bdim,
    const Tensor &src, std::optional<int64_t> src_bdim,
    int64_t offset, int64_t dim1, int64_t dim2)
{
  // This function deals with scattering diagonal elements from `src` into `self`
  // according to specified dimensions and offsets.
}
{
  // 将批次维度移到张量前面，确保操作的一致性
  auto self_ = moveBatchDimToFront(self, self_bdim);
  auto src_ = moveBatchDimToFront(src, src_bdim);

  // 获取批次大小，用于后续保证维度的一致性
  auto batch_size = get_bdim_size2(self, self_bdim, src, src_bdim);

  // 确保张量具有批次维度，如果没有则添加
  self_ = ensure_has_bdim(self_, self_bdim.has_value(), batch_size);
  src_ = ensure_has_bdim(src_, src_bdim.has_value(), batch_size);

  // 计算没有批次维度的张量的逻辑秩
  auto self_logical_rank = rankWithoutBatchDim(self, self_bdim);

  // 根据逻辑秩调整维度，加上1以匹配PyTorch的索引
  dim1 = maybe_wrap_dim(dim1, self_logical_rank) + 1;
  dim2 = maybe_wrap_dim(dim2, self_logical_rank) + 1;

  // 调用PyTorch的对角线散播操作，并返回结果和标志0
  return std::make_tuple(at::diagonal_scatter(self_, src_, offset, dim1, dim2), 0);
}

std::tuple<Tensor,optional<int64_t>> index_add_batch_rule_impl(
    Tensor& self, optional<int64_t> self_bdim,
    int64_t dim,
    const Tensor& index, optional<int64_t> index_bdim,
    const Tensor& other, optional<int64_t> other_bdim,
    const Scalar& alpha,
    const bool inplace) {

  // 如果是原地操作且没有批次维度，则抛出错误
  if (inplace && !self_bdim.has_value()){
    vmapIncompatibleInplaceError("index_add_");
  }

  // 如果index没有批次维度，则处理标量张量的情况
  if (!index_bdim) {
    const auto self_logical_rank = rankWithoutBatchDim(self, self_bdim);
    const auto other_logical_rank = rankWithoutBatchDim(other, other_bdim);

    // 将批次维度移到张量前面，处理标量张量的情况
    auto self_ = moveBatchDimToFront(self, self_bdim);
    if (self_logical_rank == 0) {
      self_ = self_.unsqueeze(-1);
    }
    auto other_ = moveBatchDimToFront(other, other_bdim);
    if (other_logical_rank == 0) {
      other_ = other_.unsqueeze(-1);
    }

    // 根据逻辑秩调整维度，确保维度匹配
    dim = maybe_wrap_dim(dim, self_logical_rank);

    // 获取批次大小，用于后续保证维度的一致性
    const auto batch_size = get_bdim_size2(self, self_bdim, other, other_bdim);
    self_ = ensure_has_bdim(self_, self_bdim.has_value(), batch_size);
    other_ = ensure_has_bdim(other_, other_bdim.has_value(), batch_size);

    // 如果是原地操作，直接调用对应的index_add_方法
    if (inplace) {
      self_.index_add_(dim + 1, index, other_, alpha);
      if (self_logical_rank == 0) {
        self_ = self_.squeeze(-1);
      }
      return std::make_tuple(self, 0);
    }

    // 否则调用普通的index_add方法，并根据需要压缩维度
    auto result = self_.index_add(dim + 1, index, other_, alpha);
    if (self_logical_rank == 0) {
      result = result.squeeze(-1);
    }
    return std::make_tuple(result, 0);
  }

  // 处理具有批次维度的索引情况，使用循环和堆叠操作
  auto batch_size = get_bdim_size3(self, self_bdim, other, other_bdim, index, index_bdim);
  std::vector<Tensor> results;
  if (!inplace) {
    results.reserve(batch_size);
  }
  for (const auto i : c10::irange(0, batch_size)) {
    // 根据批次维度选择张量的切片
    const auto& self_slice = self_bdim.has_value() ?
      self.select(*self_bdim, i) : self;
    const auto& other_slice = other_bdim.has_value() ?
      other.select(*other_bdim, i) : other;
    const auto& index_slice = index_bdim.has_value() ?
      index.select(*index_bdim, i) : index;

    // 如果是原地操作，则直接调用index_add_方法
    if (inplace) {
      self_slice.index_add_(dim, index_slice, other_slice, alpha);
    } else {
      // 否则收集每个批次的index_add结果
      results.push_back(at::index_add(self_slice, dim, index_slice, other_slice, alpha));
    }
  }
  }
  // 关闭函数定义

  }
  // 关闭 if 语句块

  if (inplace) {
    // 如果 inplace 为真，则执行以下操作
    return std::make_tuple(at::stack(self), 0);
    // 返回一个元组，包含 self 张量的堆栈以及整数 0
  }

  // 如果 inplace 为假，则执行以下操作
  return std::make_tuple(at::stack(results), 0);
  // 返回一个元组，包含 results 张量的堆栈以及整数 0
}

void index_add__batch_rule(
    Tensor& self, optional<int64_t> self_bdim,
    int64_t dim,
    const Tensor& index, optional<int64_t> index_bdim,
    const Tensor& other, optional<int64_t> other_bdim,
    const Scalar& alpha) {
  // 调用实现函数来执行索引加法的批处理规则
  index_add_batch_rule_impl(self, self_bdim, dim, index, index_bdim, other,
                            other_bdim, alpha, true);
}

std::tuple<Tensor,optional<int64_t>> index_add_batch_rule(
    Tensor& self, optional<int64_t> self_bdim,
    int64_t dim,
    const Tensor& index, optional<int64_t> index_bdim,
    const Tensor& other, optional<int64_t> other_bdim,
    const Scalar& alpha) {
  // 克隆输入张量，保留内存格式
  auto self_ = self.clone(at::MemoryFormat::Preserve);
  // 调用实现函数来执行索引加法的批处理规则
  return index_add_batch_rule_impl(self_, self_bdim, dim, index, index_bdim,
                                   other, other_bdim, alpha, false);
}

static std::tuple<Tensor,Tensor> binary_pointwise_align(
    const Tensor & self,
    optional<int64_t> self_bdim,
    const Tensor & mask,
    optional<int64_t> mask_bdim) {
  // 计算最大逻辑秩
  auto tensor_logical_rank = rankWithoutBatchDim(self, self_bdim);
  auto other_logical_rank = rankWithoutBatchDim(mask, mask_bdim);
  auto max_logical_rank = std::max(tensor_logical_rank, other_logical_rank);

  // 将批处理维度移到前面
  auto tensor_ = moveBatchDimToFront(self, self_bdim);
  auto other_ = moveBatchDimToFront(mask, mask_bdim);

  // 如果维度不对齐，需要对齐它们
  // 示例：Tensor[B, 3] + Tensor[2, 5, 3] -> Tensor[B, 1, 1, 3] + Tensor[2, 5, 3]
  // 只有具有批处理维度的张量需要修改
  tensor_ = maybePadToLogicalRank(tensor_, self_bdim, max_logical_rank);
  other_ = maybePadToLogicalRank(other_, mask_bdim, max_logical_rank);

  return std::make_tuple(tensor_, other_);
}

std::tuple<Tensor,optional<int64_t>> masked_fill_scalar_batch_rule(
    const Tensor & self,
    optional<int64_t> self_bdim,
    const Tensor & mask,
    optional<int64_t> mask_bdim,
    const Scalar& source) {
  // 对输入张量和掩码进行二进制逐点对齐
  auto tensors = binary_pointwise_align(self, self_bdim, mask, mask_bdim);
  // 使用标量值填充掩码位置
  auto result = at::masked_fill(std::get<0>(tensors), std::get<1>(tensors), source);
  return std::make_tuple(result, 0);
}

std::tuple<Tensor,optional<int64_t>> index_fill_batch_rule_helper(
  int64_t batch_size,
  int64_t self_logical_rank,
  int64_t index_logical_rank,
  Tensor & self_,
  int64_t dim,
  Tensor & index_,
  const Scalar & value
  ){
  if (self_logical_rank != 0){
    // 创建索引偏移量
    auto index_offset = at::arange(
      batch_size,
      at::TensorOptions().dtype(index_.scalar_type()).device(index_.device())
    );
    // 如果索引逻辑秩为零，则展开维度
    if (index_logical_rank == 0){
      index_ = index_.unsqueeze(-1);
    }
    // 添加索引偏移量，使其与输入张量大小相匹配
    index_ = index_.add(index_offset.unsqueeze(-1), self_.size(dim + 1));
    // 重新整形索引维度
    index_ = reshape_dim_into(0, 0, index_);
    // 重塑自身张量维度
    self_ = reshape_dim_into(0, dim, self_);
    // 在指定维度上使用索引填充张量
    self_.index_fill_(dim, index_, value);
    // 重塑自身张量移出维度
    self_ = reshape_dim_outof(dim, batch_size, self_);
    // 返回一个包含 self_ 和 dim 的 tuple
    return std::make_tuple(self_, dim);
  }

  // 如果 self_logical_rank == 0，则批处理维度肯定是 0，我们必须对每一行应用批处理索引。
  if (index_logical_rank != 0){
    // 将 index_ 在第 0 维上进行重塑，使其成为一个批处理索引的形状
    index_ = reshape_dim_into(0, 0, index_);
  }
  // 在 dim 维度后面添加一个新的维度（unsqueeze），用于索引操作
  self_.unsqueeze_(-1);
  // 在 dim + 1 维度上使用 index_ 进行填充 value
  self_.index_fill_(dim + 1, index_, value);
  // 去除在 dim + 1 维度上的添加的维度（squeeze）
  self_.squeeze_(-1);

  // 返回一个包含处理后的 self_ 和数字 0 的 tuple
  return std::make_tuple(self_, 0);
}

std::tuple<Tensor,optional<int64_t>> index_fill_int_scalar_batch_rule_impl(
    Tensor & self, optional<int64_t> self_bdim,
    int64_t dim,
    const Tensor & index, optional<int64_t> index_bdim,
    const Scalar & value,
    const bool inplace) {
  // 计算输入张量 self 和 index 的逻辑秩（不包括批次维度）
  const auto self_logical_rank = rankWithoutBatchDim(self, self_bdim);
  const auto index_logical_rank = rankWithoutBatchDim(index, index_bdim);
  // 将批次维度移动到张量的最前面
  Tensor self_ = moveBatchDimToFront(self, self_bdim);
  Tensor index_ = moveBatchDimToFront(index, index_bdim);
  // 根据 self 的逻辑秩修正 dim 的值，确保在有效范围内
  dim = maybe_wrap_dim(dim, self_logical_rank);

  // 如果 inplace 标志为 true，但 self 没有批次维度，抛出错误
  if (inplace && !self_bdim.has_value()) {
    vmapIncompatibleInplaceError("index_fill_");
  }

  // 如果 index 没有批次维度
  if (!index_bdim) {
    // 如果 self 的逻辑秩为 0，则在最后添加一个维度
    if (self_logical_rank == 0){
      self_.unsqueeze_(-1);
    }
    // 在 self_ 上使用 index_ 在 dim+1 维度上填充值 value
    self_.index_fill_(dim + 1, index_, value);
    // 如果 self 的逻辑秩为 0，则挤压掉添加的维度
    if (self_logical_rank == 0) {
      self_.squeeze_(-1);
    }
    // 返回修改后的 self_ 和批次维度值 0
    return std::make_tuple(self_, 0);
  }

  // 获取 self 和 index 的批次大小
  auto batch_size = get_bdim_size2(self, self_bdim, index, index_bdim);
  // 确保 self_ 和 index_ 具有批次维度，根据情况添加
  self_ = ensure_has_bdim(self_, self_bdim.has_value(), batch_size);
  index_ = ensure_has_bdim(index_, index_bdim.has_value(), batch_size);

  // 如果 inplace 标志为 true
  if (inplace) {
    // 对每个批次中的元素进行循环，因为不能在不复制的情况下改变具有不兼容步幅的 self_
    for (const auto i : c10::irange(0, batch_size)) {
      const auto& self_slice = self_.select(0, i);
      const auto& index_slice = index_.select(0, i);
      self_slice.index_fill_(
        dim,
        index_slice,
        value
      );
    }
    // 返回修改后的 self_ 和批次维度值 0
    return std::make_tuple(self_, 0);
  }

  // 如果 inplace 标志为 false，则克隆 self_
  self_ = self_bdim.has_value() ? self_ : self_.clone();

  // 调用帮助函数，处理填充操作，返回结果
  return index_fill_batch_rule_helper(batch_size, self_logical_rank, index_logical_rank, self_, dim, index_, value);
}

std::tuple<Tensor,optional<int64_t>> index_fill_int_tensor_batch_rule_impl(
    Tensor & self, optional<int64_t> self_bdim,
    int64_t dim,
    const Tensor & index, optional<int64_t> index_bdim,
    const Tensor & value, optional<int64_t> value_bdim,
    const bool inplace) {
  // 计算输入张量 self、index 和 value 的逻辑秩（不包括批次维度）
  const auto self_logical_rank = rankWithoutBatchDim(self, self_bdim);
  const auto index_logical_rank = rankWithoutBatchDim(index, index_bdim);
  // 将批次维度移动到张量的最前面
  Tensor self_ = moveBatchDimToFront(self, self_bdim);
  Tensor index_ = moveBatchDimToFront(index, index_bdim);
  Tensor value_ = moveBatchDimToFront(value, value_bdim);
  // 根据 self 的逻辑秩修正 dim 的值，确保在有效范围内
  dim = maybe_wrap_dim(dim, self_logical_rank);

  // 如果 inplace 标志为 true，但 self 没有批次维度，抛出错误
  if (inplace && !self_bdim.has_value()) {
    vmapIncompatibleInplaceError("index_fill_");
  }

  // 如果 index 和 value 均没有批次维度
  if (!index_bdim && !value_bdim) {
    // 如果 self 的逻辑秩为 0，则在最后添加一个维度
    if (self_logical_rank == 0){
      self_.unsqueeze_(-1);
    }
    // 在 self_ 上使用 index_ 和 value_ 在 dim+1 维度上填充值
    self_.index_fill_(dim + 1, index_, value);
    // 如果 self 的逻辑秩为 0，则挤压掉添加的维度
    if (self_logical_rank == 0) {
      self_.squeeze_(-1);
    }
    // 返回修改后的 self_ 和批次维度值 0
    return std::make_tuple(self_, 0);
  }

  // 获取 self、index 和 value 的批次大小
  auto batch_size = get_bdim_size3(self, self_bdim, index, index_bdim, value, value_bdim);
  // 确保 self_、index_ 和 value_ 具有批次维度，根据情况添加
  self_ = ensure_has_bdim(self_, self_bdim.has_value(), batch_size);
  index_ = ensure_has_bdim(index_, index_bdim.has_value(), batch_size);

  // 如果 inplace 标志为 true 或 value 有批次维度
  if (inplace || value_bdim.has_value()) {
    // 对于 in-place 操作，使用循环因为不能改变 self_ 张量的步幅而不进行复制。
    // 如果值具有批处理维度，也需要使用循环，因为 index_fill_ 仅支持单元素张量。
    for (const auto i : c10::irange(0, batch_size)) {
      // 选择 self_ 张量的第 i 个切片
      const auto& self_slice = self_.select(0, i);
      // 选择 index_ 张量的第 i 个切片
      const auto& index_slice = index_.select(0, i);
      // 在 self_slice 上使用 index_fill_ 方法：
      // 如果 value_bdim 有值，则使用 value_.select(0, i)，否则使用 value_
      self_slice.index_fill_(
        dim,
        index_slice,
        value_bdim.has_value() ? value_.select(0, i) : value_
      );
    }
    // 返回更新后的 self_ 张量和状态值 0 的元组
    return std::make_tuple(self_, 0);
  }

  // 如果 self_ 具有批处理维度，则克隆 self_ 张量以确保正确的操作。
  self_ = self_bdim.has_value() ? self_ : self_.clone();

  // 在这里调用 value.item() 是安全的，因为 value 肯定不是批处理张量。
  // 调用 index_fill_batch_rule_helper 函数，返回其结果
  return index_fill_batch_rule_helper(batch_size, self_logical_rank, index_logical_rank, self_, dim, index_, value.item());
}

void index_fill__int_scalar_batch_rule(
    Tensor & self, optional<int64_t> self_bdim,
    int64_t dim,
    const Tensor & index, optional<int64_t> index_bdim,
    const Scalar & value) {
  // 调用具体的实现函数，用标量值填充索引指定的维度
  index_fill_int_scalar_batch_rule_impl(self, self_bdim, dim, index, index_bdim, value, true);
}

void index_fill__int_tensor_batch_rule(
    Tensor & self, optional<int64_t> self_bdim,
    int64_t dim,
    const Tensor & index, optional<int64_t> index_bdim,
    const Tensor & value, optional<int64_t> value_bdim) {
  // 调用具体的实现函数，用张量值填充索引指定的维度
  index_fill_int_tensor_batch_rule_impl(self, self_bdim, dim, index, index_bdim, value, value_bdim, true);
}

std::tuple<Tensor,optional<int64_t>> index_fill_int_scalar_batch_rule(
    const Tensor & self, optional<int64_t> self_bdim,
    int64_t dim,
    const Tensor & index, optional<int64_t> index_bdim,
    const Scalar & value) {
  // 复制输入张量以保留内存格式，然后调用具体的实现函数进行索引填充
  auto self_ = self.clone(at::MemoryFormat::Preserve);
  return index_fill_int_scalar_batch_rule_impl(self_, self_bdim, dim, index, index_bdim, value, false);
}

std::tuple<Tensor,optional<int64_t>> index_fill_int_tensor_batch_rule(
    const Tensor & self, optional<int64_t> self_bdim,
    int64_t dim,
    const Tensor & index, optional<int64_t> index_bdim,
    const Tensor & value, optional<int64_t> value_bdim) {
  // 复制输入张量以保留内存格式，然后调用具体的实现函数进行索引填充
  auto self_ = self.clone(at::MemoryFormat::Preserve);
  return index_fill_int_tensor_batch_rule_impl(self_, self_bdim, dim, index, index_bdim, value, value_bdim, false);
}

}
# 实现 ATen 库中 aten 命名空间下的 Torch 库功能
TORCH_LIBRARY_IMPL(aten, FuncTorchBatched, m) {
    # 实现 index.Tensor 的批处理逻辑
    m.impl("index.Tensor", index_plumbing);
    # 实现 index_put_ 的批处理逻辑
    m.impl("index_put_", index_put__plumbing);
    # 实现 index_put 的批处理逻辑
    m.impl("index_put", index_put_plumbing);
    # 实现 _index_put_impl_ 的批处理逻辑
    m.impl("_index_put_impl_", _index_put_impl__plumbing);
    # 实现 slice_scatter 的批处理逻辑
    m.impl("slice_scatter", slice_scatter_decomp);
    # 实现 select_scatter 的批处理逻辑
    m.impl("select_scatter", select_scatter_decomp);
    # 实现 index_copy 的批处理逻辑
    m.impl("index_copy", index_copy_decomp);
    # 实现 index_select 的批处理逻辑
    m.impl("index_select", index_select_decomp);
    # 支持标量和张量之间的 masked_fill 批处理
    VMAP_SUPPORT2(masked_fill, Scalar, masked_fill_scalar_batch_rule);
    # 支持整数标量和整数张量之间的 index_fill_ 批处理
    VMAP_SUPPORT2(index_fill_, int_Tensor, index_fill__int_tensor_batch_rule);
    # 支持整数标量和整数张量之间的 index_fill 批处理
    VMAP_SUPPORT2(index_fill_, int_Scalar, index_fill__int_scalar_batch_rule);
    # 支持整数张量和整数标量之间的 index_fill 批处理
    VMAP_SUPPORT2(index_fill, int_Tensor, index_fill_int_tensor_batch_rule);
    # 支持整数张量和整数标量之间的 index_fill 批处理
    VMAP_SUPPORT2(index_fill, int_Scalar, index_fill_int_scalar_batch_rule);
    # 支持 index_add_ 批处理
    VMAP_SUPPORT(index_add_, index_add__batch_rule);
    # 支持 index_add 批处理
    VMAP_SUPPORT(index_add, index_add_batch_rule);
    # 支持 diagonal_scatter 批处理
    VMAP_SUPPORT(diagonal_scatter, diagonal_scatter_batch_rule);
    # 支持 gather 批处理
    VMAP_SUPPORT(gather, gather_batch_rule);
    # 支持 scatter_value 批处理
    VMAP_SUPPORT2(scatter, value, scatter_value_batch_rule);
    # 支持 scatter_src 批处理
    VMAP_SUPPORT2(scatter, src, scatter_src_batch_rule);
    # 支持 scatter_add 批处理
    VMAP_SUPPORT(scatter_add, scatter_add_batch_rule);
    # 支持 scatter_reduce 批处理
    VMAP_SUPPORT2(scatter, reduce, scatter_reduce_batch_rule);
    # 支持 scatter_value_reduce 批处理
    VMAP_SUPPORT2(scatter, value_reduce, scatter_value_reduce_batch_rule);
    // as_strided_scatter 在当前情况下不支持 for 循环回退，
    // 因为 as_strided_scatter 返回的输出会匹配其输入的步幅和存储偏移量。
    // 在 for 循环回退中，每个输入张量都是大批量张量的一个切片。
    // 注册 vmapErrorFallback 作为 as_strided_scatter 的实现函数
    m.impl("as_strided_scatter", torch::CppFunction::makeFromBoxedFunction<&vmapErrorFallback>());
}

} // namespace at::functorch
```