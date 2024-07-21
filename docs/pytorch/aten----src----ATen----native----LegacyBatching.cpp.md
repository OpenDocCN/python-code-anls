# `.\pytorch\aten\src\ATen\native\LegacyBatching.cpp`

```
// 包含头文件 Tensor.h, LegacyBatchedTensorImpl.h, WrapDimUtils.h 和 LegacyVmapTransforms.h
#include <ATen/core/Tensor.h>
#include <ATen/LegacyBatchedTensorImpl.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/LegacyVmapTransforms.h>

#ifdef AT_PER_OPERATOR_HEADERS
// 根据预处理宏 AT_PER_OPERATOR_HEADERS，包含 _add_batch_dim_native.h 和 _remove_batch_dim_native.h 头文件
#include <ATen/ops/_add_batch_dim_native.h>
#include <ATen/ops/_remove_batch_dim_native.h>
#endif

// at::native 命名空间开始
namespace at::native {

// 向张量 `self` 中添加一个离散的批次维度，返回结果张量
Tensor _add_batch_dim(const Tensor& self, int64_t batch_dim, int64_t level) {
  // 调用 addBatchDim 函数添加批次维度，返回结果
  return addBatchDim(self, level, batch_dim);
}

// 检查张量 `self` 是否具有指定级别 `level` 的批次维度
static bool has_level(const Tensor& self, int64_t level) {
  // 获取张量的 BatchedTensorImpl，若不存在则返回 false
  const auto* batched = maybeGetBatchedImpl(self);
  if (!batched) {
    return false;
  }
  // 获取批次维度信息列表
  auto bdims = batched->bdims();
  // 在 bdims 中查找是否存在 level 级别的 BatchDim
  auto* it = std::find_if(bdims.begin(), bdims.end(), [&](const BatchDim& bdim) {
    return bdim.level() == level;
  });
  // 返回是否找到指定级别的批次维度
  return it != bdims.end();
}

// 移除给定级别 `level` 的批次维度，并返回移除后的 Tensor 以及原来批次维度的逻辑维度索引
// 此函数调用后一般紧跟着调用 `movedim` 函数
//
// 前提条件：在 `batched` 中必须存在级别为 `level` 的 BatchDim
//
// 返回原因是为了跟踪原批次维度所在的逻辑维度位置，以便在 vmap 中将其移动到指定的逻辑维度
// 例如，在离开 vmap 块时，如果 x 是一个具有物理索引 0 处的批次维度的 BatchedTensor，
// 我们想要将其移动到逻辑索引 1 处，如 out_dims 指定的那样。因此我们返回批次维度的索引位置，
// 以便稍后通过调用 `movedim` 将其移动到正确的位置。
static std::pair<Tensor,int64_t> remove_existing_batch_dim(
    const BatchedTensorImpl* batched, int64_t level) {
  // 获取批次维度信息列表
  auto bdims = batched->bdims();
  // 如果只有一个批次维度，则确保该维度是 level，并返回它的值和维度索引
  if (bdims.size() == 1) {
    TORCH_INTERNAL_ASSERT(bdims[0].level() == level);
    return std::make_pair(batched->value(), bdims[0].dim());
  }
  // 否则，构建新的批次维度列表，并记录要暴露的物理维度索引
  BatchDims new_bdims;
  int64_t newly_exposed_physical_dim = -1;
  new_bdims.reserve(bdims.size() - 1);
  for (const auto& bdim : bdims) {
    if (bdim.level() == level) {
      newly_exposed_physical_dim = bdim.dim();
    } else {
      new_bdims.push_back(bdim);
    }
  }
    }
  }

// 结束两个嵌套的代码块


  // 因为批处理维度中必须存在级别 `level`，所以我们应该找到一个 `newly_exposed_logical_dim`。
  TORCH_INTERNAL_ASSERT(newly_exposed_physical_dim != -1);

// 使用 TORCH_INTERNAL_ASSERT 断言 `newly_exposed_physical_dim` 不为 -1，确保物理维度被正确找到。


  int64_t num_batch_dims_before_newly_exposed_physical_dim = std::count_if(
      new_bdims.begin(), new_bdims.end(),
      [&](const BatchDim& bdim) {
        return bdim.dim() < newly_exposed_physical_dim;
      });

// 计算在 `newly_exposed_physical_dim` 之前存在的批处理维度数量，存储在 `num_batch_dims_before_newly_exposed_physical_dim` 中。


  int64_t newly_exposed_logical_dim =
      newly_exposed_physical_dim - num_batch_dims_before_newly_exposed_physical_dim;

// 计算新暴露的逻辑维度，通过减去之前的物理维度数量。


  auto result_tensor = makeBatched(batched->value(), std::move(new_bdims));

// 使用 `makeBatched` 函数创建一个批处理后的张量 `result_tensor`，使用 `new_bdims` 进行构造。


  return std::make_pair(std::move(result_tensor), newly_exposed_logical_dim);

// 返回一个由 `result_tensor` 和 `newly_exposed_logical_dim` 构成的 `pair` 对象。
// } 是对应于代码中 namespace at::native 的结束标记

// 定义一个静态函数 maybe_movedim，用于可能移动张量的维度，如果目标与源相同则返回原始张量
static Tensor maybe_movedim(const Tensor& self, int64_t src, int64_t dst) {
  auto logical_dim = self.dim(); // 获取张量的逻辑维度数
  src = maybe_wrap_dim(src, logical_dim); // 对源维度进行边界检查和可能的包装
  dst = maybe_wrap_dim(dst, logical_dim); // 对目标维度进行边界检查和可能的包装
  if (src == dst) { // 如果源维度等于目标维度，直接返回原始张量
    return self;
  }
  return self.movedim(src, dst); // 否则移动张量的维度从 src 到 dst
}

// _remove_batch_dim 函数移除张量 self 中的批量维度，根据 level 参数决定移除哪个批量维度
Tensor _remove_batch_dim(const Tensor& self, int64_t level, int64_t batch_size, int64_t out_dim) {
  if (!has_level(self, level)) { // 如果张量 self 中不存在指定的批量级别
    auto self_sizes = self.sizes(); // 获取张量 self 的尺寸
    VmapDimVector expanded_sizes(self_sizes.begin(), self_sizes.end()); // 创建一个扩展后的尺寸向量
    expanded_sizes.insert(expanded_sizes.begin() + out_dim, batch_size); // 在指定位置插入批量大小
    return self.expand(expanded_sizes); // 扩展张量 self 的尺寸并返回
  }

  // 如果张量 self 是批量化的，必须具有批量级别
  const auto* batched = maybeGetBatchedImpl(self);
  TORCH_INTERNAL_ASSERT(batched != nullptr); // 断言批量化指针不为空

  // 移除现有的批量维度，并获取新暴露的逻辑维度
  auto [self_without_bdim, newly_exposed_logical_dim] = remove_existing_batch_dim(batched, level);
  // 可能移动张量的维度，将新暴露的逻辑维度移动到指定的输出维度位置
  return maybe_movedim(self_without_bdim, newly_exposed_logical_dim, out_dim);
}

// 命名空间结束标记，对应 namespace at::native
} // namespace at::native
```