# `.\pytorch\aten\src\ATen\native\SparseTensorUtils.cpp`

```
#include <ATen/native/SparseTensorUtils.h>

#include <ATen/ATen.h>
#include <ATen/SparseTensorImpl.h>
#include <ATen/native/sparse/SparseStubs.h>
#include <ATen/Parallel.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/_sparse_coo_tensor_with_dims_and_tensors.h>
#include <ATen/ops/zeros.h>
#endif

namespace at::native {

DEFINE_DISPATCH(flatten_indices_stub);

} // namespace at::native

namespace at::sparse {

// NOTE [ Flatten Sparse Indices ]
// This helper function flattens a sparse indices tensor (a Tensor) into a 1D
// indices tensor. E.g.,
//   input = [[2, 4, 0],
//            [3, 1, 10]]
//   full_size = [2, 12]
//   output = [ 2 * 12 + 3, 4 * 12 + 1, 0 * 12 + 10 ] = [27, 49, 10]
//
// In other words, assuming that each `indices[i, :]` is a valid index to a
// tensor `t` of shape `full_size`. This returns the corresponding indices to
// the flattened tensor `t.reshape( prod(full_size[:indices.size(0)]), -1 )`.
// if forceClone is true, the result will forced to be a clone of self.
// if force_clone is true, the result will forced to be a clone of self.
Tensor flatten_indices(const Tensor& indices, IntArrayRef full_size, bool force_clone /*= false*/) {
  // 获取稀疏索引的维度
  int64_t sparse_dim = indices.size(0);
  // 如果稀疏维度为1，且需要强制克隆结果，则压缩第0维并返回克隆结果
  if (sparse_dim == 1) {
    if (force_clone) {
      return indices.squeeze(0).clone(at::MemoryFormat::Contiguous);
    } else {
      return indices.squeeze(0);
    }
  } else {
    // 如果稀疏索引为空，则返回全零的1D索引张量
    if (!indices.numel()) {
      return at::zeros({indices.size(1)}, indices.options().dtype(kLong));
    }
    // 否则调用 flatten_indices_stub 函数，对稀疏索引进行扁平化处理
    return at::native::flatten_indices_stub(indices.device().type(), indices, full_size.slice(0, sparse_dim));
  }
}

// Flatten sparse tensor's indices from nD to 1D, similar to NOTE [ Flatten Sparse Indices ],
// except this one allows partial flatten: only flatten on specified dims. Note that
// the flatten indices might be uncoalesced if dims_to_flatten.size() < sparse_dim.
// Also if input indices is already coalesced, the flattened indices will also be sorted.
//
// args:
//    indices: sparse tensor indices
//    sizes: sparse tensor sizes
//    dims_to_flatten: a list of dim index to flatten
//
// Ex1:
//   indices = [[2, 4, 0],
//             [3, 1, 3]]
//   sizes = [2, 12]
//   dims_to_flatten = [0, 1]
//   new_indices = [ 2 * 12 + 3, 4 * 12 + 1, 0 * 12 + 3 ] = [27, 49, 3]
//
// Ex2:
//   dims_to_flatten = [1]
//   new_indices = [ 3, 1, 3 ]  # uncoalesced
Tensor flatten_indices_by_dims(const Tensor& indices, const IntArrayRef& sizes, const IntArrayRef& dims_to_flatten){
  // 创建一个全零的新索引张量，大小与输入索引的第1维度大小相同
  Tensor new_indices = at::zeros({indices.size(1)}, indices.options());
  // 对于每个要扁平化的维度，更新新索引张量的值
  for (auto d : dims_to_flatten) {
    new_indices.mul_(sizes[d]); // 乘以当前维度的大小
    new_indices.add_(indices.select(0, d)); // 加上对应维度的索引值
  }
  return new_indices;
}

Tensor coo_to_csr(const int64_t* indices, int64_t dim, int64_t nnz) {
  /*
    Find the CSR representation for a row `indices` from the COO format

    TODO: Add implementation here
  */
}

} // namespace at::sparse
    /*
      Inputs:
        `indices` is the row pointer from COO indices
        `dim` is the row dimensionality
        `nnz` is the number of non-zeros
    
      Output:
        `csr` is a compressed row array in a CSR format
    */
    // 创建一个全零的长整型张量 csr，长度为 dim + 1，用于存储 CSR 格式的压缩行数组
    Tensor csr = at::zeros({dim + 1}, kLong);
    
    // TODO: 当正确支持零大小维度时，可以删除这个条件语句
    if (nnz > 0) {
      // 获取 csr 张量的访问器，以便直接访问和修改数据
      auto csr_accessor = csr.accessor<int64_t, 1>();
    
      // 使用并行计算将稀疏矩阵转换为 CSR 格式
      at::parallel_for(0, nnz, 10000, [&](int64_t start, int64_t end) {
        // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
        int64_t h, hp0, hp1;
        // 遍历从 start 到 end 的索引范围
        for (const auto i : c10::irange(start, end)) {
          // 获取当前行的起始和结束位置
          hp0 = indices[i];
          hp1 = (i+1 == nnz) ?  dim : indices[i+1];
          // 如果当前行不为空
          if (hp0 != hp1) {
            // 将当前行的元素标记为 i+1，csr 格式中的索引从1开始
            for (h = hp0; h < hp1; h++) {
              csr_accessor[h+1] = i+1;
            }
          }
        }
      });
    }
    // 返回 CSR 格式的稀疏矩阵表示
    return csr;
// 结束当前的命名空间 at::sparse

} // namespace at::sparse

// 定义一个函数 zeros_like_with_indices，接受一个稀疏张量作为参数，并返回一个相同形状的稀疏张量全为零的张量
Tensor zeros_like_with_indices(const Tensor& t) {
  // 内部断言，确保输入张量 t 是稀疏张量
  TORCH_INTERNAL_ASSERT(t.is_sparse());
  // 调用 _sparse_coo_tensor_with_dims_and_tensors 函数创建一个稀疏 COO 张量，并返回
  return at::_sparse_coo_tensor_with_dims_and_tensors(
      t.sparse_dim(),
      t.dense_dim(),
      t.sizes(),
      // 克隆 t 的索引张量
      t._indices().clone(),
      // 创建一个与 t._values() 具有相同选项的零张量，并扩展成与 t._values() 相同形状
      at::zeros({1}, t._values().options()).expand_as(t._values()),
      t.options(),
      // 返回 t 是否已压缩的状态
      t.is_coalesced());
}

// 定义一个函数 full_coo_indices，接受一个大小数组和选项作为参数，并返回一个 COO 格式的完整索引张量
Tensor full_coo_indices(IntArrayRef sizes, TensorOptions options) {
  // 获取 sizes 数组中的最大值
  const auto max_size = *std::max_element(sizes.begin(), sizes.end());
  // 使用 at::arange 创建一个从 0 到 max_size 的张量，使用指定选项
  const auto max_size_arange = at::arange(max_size, options);
  // 创建一个空的张量数组 stack
  std::vector<Tensor> stack;
  stack.reserve(sizes.size());
  // 遍历 sizes 数组
  for (size_t i=0; i < sizes.size(); i++) {
    // 从 max_size_arange 中选择前 sizes[i] 个元素，创建张量 a
    Tensor a = max_size_arange.narrow(-1, 0, sizes[i]);
    // 再次遍历 sizes 数组
    for (size_t j=0; j < sizes.size(); j++) {
      // 如果 i 不等于 j，则在第 j 维度上展开张量 a
      if (i != j) {
        a.unsqueeze_(j);
      }
    }
    // 将展开后的张量 a 添加到 stack 数组中
    stack.push_back(a.expand(sizes));
  }
  // 使用 at::stack 将 stack 数组中的张量堆叠在一起，并在指定维度上展平
  return at::stack(stack).flatten(1, -1);
}

// 结束当前的命名空间 at::sparse
} // namespace at::sparse
```