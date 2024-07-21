# `.\pytorch\aten\src\ATen\native\sparse\SparseFactories.cpp`

```py
// 包含 ATen 库中的调度和迭代器功能
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/sparse/SparseFactories.h>

// 如果未定义 AT_PER_OPERATOR_HEADERS 宏，则包含通用函数和原生函数
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
// 如果定义了 AT_PER_OPERATOR_HEADERS 宏，则包含特定操作的头文件
#else
#include <ATen/ops/_spdiags_native.h>
#include <ATen/ops/_unique.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/sparse_coo_tensor.h>
#include <ATen/ops/where.h>
#endif

// 定义 at::native 命名空间
namespace at::native {

// 定义 spdiags_kernel_stub 的调度分发器
DEFINE_DISPATCH(spdiags_kernel_stub);

// spdiags 函数定义，用于创建对角线稀疏矩阵
Tensor spdiags(
    const Tensor& diagonals,
    const Tensor& offsets,
    IntArrayRef shape,
    std::optional<Layout> layout) {
  // 将 diagonals 转换为二维张量，如果原始维度为 1，则添加一维
  auto diagonals_2d = diagonals.dim() == 1 ? diagonals.unsqueeze(0) : diagonals;
  // 检查 diagonals_2d 是否为二维张量，否则抛出错误
  TORCH_CHECK(diagonals_2d.dim() == 2, "Diagonals must be vector or matrix");
  // 检查输出形状是否为二维
  TORCH_CHECK(shape.size() == 2, "Output shape must be 2d");
  // 将 offsets 转换为一维张量，如果原始维度为 0，则添加一维
  auto offsets_1d = offsets.dim() == 0 ? offsets.unsqueeze(0) : offsets;
  // 检查 offsets_1d 是否为一维张量，否则抛出错误
  TORCH_CHECK(offsets_1d.dim() == 1, "Offsets must be scalar or vector");
  // 检查 diagonals_2d 和 offsets_1d 的长度是否匹配，否则抛出错误
  TORCH_CHECK(
      diagonals_2d.size(0) == offsets_1d.size(0),
      "Number of diagonals (",
      diagonals_2d.size(0),
      ") does not match the number of offsets (",
      offsets_1d.size(0),
      ")");
  
  // 如果提供了布局参数，则进行布局检查
  if (layout) {
    // 检查输出布局是否为稀疏、稀疏Csc或稀疏Csr之一，如果不是则抛出错误信息
    TORCH_CHECK(
        (*layout == Layout::Sparse) || (*layout == Layout::SparseCsc) ||
            (*layout == Layout::SparseCsr),
        "Only output layouts (Sparse, SparseCsc, SparseCsr) are supported, got ",
        *layout);
  }
  // 检查偏移张量的数据类型是否为Long型，如果不是则抛出错误信息
  TORCH_CHECK(
      offsets_1d.scalar_type() == at::kLong,
      "Offset Tensor must have dtype Long but got ",
      offsets_1d.scalar_type());

  // 检查偏移张量中的元素数量与去重后的元素数量是否一致，如果不一致则抛出错误信息
  TORCH_CHECK(
      offsets_1d.numel() == std::get<0>(at::_unique(offsets_1d)).numel(),
      "Offset tensor contains duplicate values");

  // 计算每个对角线上的非零元素数量
  auto nnz_per_diag = at::where(
      offsets_1d.le(0),
      offsets_1d.add(shape[0]).clamp_max_(diagonals_2d.size(1)),
      offsets_1d.add(-std::min<int64_t>(shape[1], diagonals_2d.size(1))).neg());

  // 计算累积非零元素数量
  auto nnz_per_diag_cumsum = nnz_per_diag.cumsum(-1);
  const auto nnz = diagonals_2d.size(0) > 0
      ? nnz_per_diag_cumsum.select(-1, -1).item<int64_t>()
      : int64_t{0};
  
  // 计算每个对角线的结果内存偏移量
  auto result_mem_offsets = nnz_per_diag_cumsum.sub(nnz_per_diag);

  // 创建空的索引张量和值张量
  auto indices = at::empty({2, nnz}, offsets_1d.options());
  auto values = at::empty({nnz}, diagonals_2d.options());

  // 创建对角线索引张量，用于迭代时查找每次迭代中读取的对角线行
  const auto n_diag = offsets_1d.size(0);
  Tensor diag_index = at::arange(n_diag, offsets_1d.options());

  // 创建一个空的输出张量作为cpu_kernel的输出
  auto dummy = at::empty({1}, offsets_1d.options()).resize_({0});

  // 配置TensorIterator，用于迭代计算稀疏矩阵对角线操作的内核
  auto iter = TensorIteratorConfig()
                  .set_check_mem_overlap(false)
                  .add_output(dummy)
                  .add_input(diag_index)
                  .add_input(offsets_1d)
                  .add_input(result_mem_offsets)
                  .add_input(nnz_per_diag)
                  .build();
  // 调用稀疏对角线操作的内核函数
  spdiags_kernel_stub(iter.device_type(), iter, diagonals_2d, values, indices);

  // 根据layout返回相应的稀疏矩阵格式
  auto result_coo = at::sparse_coo_tensor(indices, values, shape);
  if (layout) {
    if (*layout == Layout::SparseCsr) {
      return result_coo.to_sparse_csr();
    }
    if (*layout == Layout::SparseCsc) {
      return result_coo.to_sparse_csc();
    }
  }
  // 默认返回COO格式的稀疏矩阵
  return result_coo;
}

} // namespace at::native


注释：

// 结束当前代码块，关闭命名空间 at::native
```