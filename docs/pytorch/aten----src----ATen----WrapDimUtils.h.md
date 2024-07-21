# `.\pytorch\aten\src\ATen\WrapDimUtils.h`

```
// 预处理命令，确保头文件只被包含一次
#pragma once

// 引入 ATen 库的相关头文件
#include <ATen/core/IListRef.h>
#include <ATen/core/Tensor.h>
#include <c10/core/TensorImpl.h>
#include <c10/core/WrapDimMinimal.h>
#include <c10/util/irange.h>

// 使用 at 命名空间
namespace at {

// 使用 c10 命名空间中的 maybe_wrap_dim 函数
// 如果 dim_post_expr 为 0 且 wrap_scalar 为 true，则 dim 必须在 [-1, 0] 范围内，适用于标量张量的特殊情况，
// 如 torch.sum(scalar_tensor, 0)。否则，dim 应在 [-dim_post_expr, dim_post_expr-1] 范围内。
using c10::maybe_wrap_dim;

// 用于处理 TensorImpl* 类型的 maybe_wrap_dim 函数重载
inline int64_t maybe_wrap_dim(int64_t dim, TensorImpl* tensor) {
  return maybe_wrap_dim(dim, tensor->dim());
}

// 用于处理 TensorList 类型的 maybe_wrap_dim 函数重载
inline int64_t maybe_wrap_dim(int64_t dim, TensorList tensors) {
  if (tensors.empty()) {
    // 如果 TensorList 为空，则无法进行包装，返回原始的 dim 值
    // 可依赖底层实现抛出错误（如果必要）
    return dim;
  }
  return maybe_wrap_dim(dim, tensors[0].dim());
}

// 用于处理 std::vector<std::vector<int64_t>> 类型的 maybe_wrap_dim 函数重载
inline int64_t maybe_wrap_dim(
    int64_t dim,
    const std::vector<std::vector<int64_t>>& tensor_sizes) {
  if (tensor_sizes.empty()) {
    // 如果 tensor_sizes 列表为空，则无法进行包装，返回原始的 dim 值
    // 可依赖底层实现抛出错误（如果必要）
    return dim;
  }
  return maybe_wrap_dim(dim, tensor_sizes[0].size());
}

// maybe_wrap_dims_n 函数定义
// 给定长度为 ndims 的维度数组 `dims`，根据 `dim_post_expr` 包装每个维度，允许使用负索引
// 如果 `wrap_scalars` 为 true，则标量张量（秩为 0）的维度允许在 [-1, 0] 范围内，
// 否则对不在 [-dim_post_expr, dim_post_expr) 范围内的维度抛出 IndexError
inline void maybe_wrap_dims_n(
    int64_t* dims,
    int64_t ndims,
    int64_t dim_post_expr,
    bool wrap_scalars = true) {
  if (dim_post_expr <= 0) {
    if (wrap_scalars) {
      // 如果 wrap_scalars 为 true，将 dim_post_expr 设为 1，使范围为 [-1, 0]
      dim_post_expr = 1;
    } else {
      // 如果 wrap_scalars 为 false 且 dim_post_expr <= 0，表示标量张量，
      // 如果 ndims 不为 0，则抛出索引错误
      TORCH_CHECK_INDEX(
          ndims == 0,
          "Dimension specified as ",
          dims[0],
          " but tensor has no dimensions");
      return;
    }
  }
  // 计算有效的最小和最大索引范围
  int64_t min = -dim_post_expr;
  int64_t max = dim_post_expr - 1;
  // 对每个维度进行遍历，检查是否在有效范围内，并按需要进行包装处理
  for (const auto i : c10::irange(ndims)) {
    auto& dim = dims[i];
    if (dim < min || dim > max) {
      // 如果维度超出有效范围，抛出索引错误
      TORCH_CHECK_INDEX(
          false,
          "Dimension out of range (expected to be in range of [",
          min,
          ", ",
          max,
          "], but got ",
          dim,
          ")");
    }
    if (dim < 0)
      // 如果维度是负数，根据 dim_post_expr 进行包装处理
      dim += dim_post_expr;
  }
}

// maybe_wrap_dims 函数模板定义
// 给定连续容器 `dims`，根据 `dim_post_expr` 包装每个维度，允许使用负索引
// 如果 `wrap_scalars` 为 true，则标量张量（秩为 0）的维度允许在 [-1, 0] 范围内，
// 否则对不在 [-dim_post_expr, dim_post_expr) 范围内的维度抛出 IndexError
template <typename Container>
inline void maybe_wrap_dims(
    Container& dims,
    int64_t dim_post_expr,
    bool wrap_scalars = true) {
    # 定义一个布尔变量 wrap_scalars，默认为 true
    bool wrap_scalars = true) {
    # 调用 maybe_wrap_dims_n 函数，将 dims 数组的数据指针、大小、dim_post_expr 表达式和 wrap_scalars 作为参数传递
    return maybe_wrap_dims_n(
        dims.data(), dims.size(), dim_post_expr, wrap_scalars);
// 原先，大小为 [0] 的张量是唯一可能为空的张量；因此，除非所有其他张量都是一维的，否则不可能连接空张量，
// 因此我们允许跳过这些张量（无论是为了包裹维度行为还是维度大小检查）。我们保持这种行为是为了向后兼容性，
// 但仅适用于这个特定的大小（即其他空大小不会被跳过）。
inline int64_t legacy_cat_wrap_dim(
    int64_t dim,
    const std::vector<std::vector<int64_t>>& tensor_sizes) {
  // 遍历张量大小的二维向量
  for (auto& sizes : tensor_sizes) {
    // 如果当前张量大小向量的长度为1，并且第一个元素为0，则跳过此张量
    if (sizes.size() == 1 && sizes[0] == 0) {
      continue;
    }
    // 否则调用 maybe_wrap_dim 函数，将 dim 包裹到指定的维度范围内，并返回结果
    return maybe_wrap_dim(dim, static_cast<int64_t>(sizes.size()));
  }
  // 如果没有符合条件的张量，则直接返回 dim
  return dim;
}

// 对于具有符号整数的张量大小的向量，同样进行包裹维度的处理
inline int64_t legacy_cat_wrap_dim_symint(
    int64_t dim,
    const std::vector<std::vector<c10::SymInt>>& tensor_sizes) {
  // 遍历张量大小的二维向量
  for (auto& sizes : tensor_sizes) {
    // 如果当前张量大小向量的长度为1
    if (sizes.size() == 1) {
      // 并且第一个元素的符号等于0（即 TORCH_GUARD_SIZE_OBLIVIOUS(sizes[0].sym_eq(0)) 为真），则跳过此张量
      if (TORCH_GUARD_SIZE_OBLIVIOUS(sizes[0].sym_eq(0))) {
        continue;
      }
    }
    // 否则调用 maybe_wrap_dim 函数，将 dim 包裹到指定的维度范围内，并返回结果
    return maybe_wrap_dim(dim, static_cast<int64_t>(sizes.size()));
  }
  // 如果没有符合条件的张量，则直接返回 dim
  return dim;
}

// 对于张量列表的引用，同样进行包裹维度的处理
inline int64_t legacy_cat_wrap_dim(
    int64_t dim,
    const MaterializedITensorListRef& tensors) {
  // 遍历张量列表
  for (const Tensor& tensor : tensors) {
    // 如果当前张量的维度为1
    if (tensor.dim() == 1) {
      // 并且第一个维度的符号大小等于0（即 TORCH_GUARD_SIZE_OBLIVIOUS(tensor.sym_sizes()[0].sym_eq(0)) 为真），则跳过此张量
      if (TORCH_GUARD_SIZE_OBLIVIOUS(tensor.sym_sizes()[0].sym_eq(0))) {
        continue;
      }
    }
    // 否则调用 maybe_wrap_dim 函数，将 dim 包裹到指定的维度范围内，并返回结果
    return maybe_wrap_dim(dim, tensor.dim());
  }
  // 如果没有符合条件的张量，则直接返回 dim
  return dim;
}

// 将负维度包裹在一个向量中
inline void wrap_all_dims(
    std::vector<int64_t>& dims_to_wrap,
    int64_t tensor_total_dims) {
  // 遍历要包裹的维度向量
  for (const auto i : c10::irange(dims_to_wrap.size())) {
    // 调用 maybe_wrap_dim 函数，将每个维度包裹到指定的维度范围内
    dims_to_wrap[i] = maybe_wrap_dim(dims_to_wrap[i], tensor_total_dims);
  }
}

// at 命名空间的结束
} // namespace at
```