# `.\pytorch\aten\src\ATen\native\TensorShape.h`

```py
#pragma once
#include <ATen/core/Tensor.h>
#include <c10/util/irange.h>
#include <ATen/core/IListRef.h>

namespace at::native {

// 定义一个公共的 API，用于克隆一个张量并保持步长信息不变
TORCH_API at::Tensor clone_preserve_strides(const at::Tensor& self);

// 内联函数，用于检查是否应该跳过某个张量的串联操作
inline bool cat_should_skip_tensor(const Tensor& t) {
  return t.sym_numel() == 0 && t.dim() == 1;
}

// 内联函数，检查两个张量的形状是否兼容用于沿指定维度串联操作
inline void check_cat_shape_except_dim(const Tensor & first, const Tensor & second, int64_t dimension, int64_t index) {
  int64_t first_dims = first.dim();
  int64_t second_dims = second.dim();
  // 检查张量的维度数是否相同
  TORCH_CHECK(first_dims == second_dims, "Tensors must have same number of dimensions: got ",
              first_dims, " and ", second_dims);
  for (const auto dim : c10::irange(first_dims)) {
    if (dim == dimension) {
      continue;
    }
    int64_t first_dim_size = first.sizes()[dim];
    int64_t second_dim_size = second.sizes()[dim];
    // 检查除了指定维度外的其他维度大小是否相同
    TORCH_CHECK(first_dim_size == second_dim_size, "Sizes of tensors must match except in dimension ",
                dimension, ". Expected size ", static_cast<long long>(first_dim_size), " but got size ", static_cast<long long>(second_dim_size), " for tensor number ", index, " in the list.");
  }
}

// 内联函数，检查是否有零维张量存在于列表中
inline void check_cat_no_zero_dim(const MaterializedITensorListRef& tensors) {
  int64_t i = 0;
  for(const Tensor& t : tensors) {
    // 检查是否存在零维张量，如果存在则报错
    TORCH_CHECK(t.dim() > 0,
             "zero-dimensional tensor (at position ", i, ") cannot be concatenated");
    i++;
  }
}

// 内联函数，计算在给定维度上的张量分割数
inline int64_t get_num_splits(const Tensor& self, int64_t split_size, int64_t dim) {
  TORCH_CHECK(self.dim() != 0, "split expects at least a 1-dimensional tensor");
  TORCH_CHECK(split_size >= 0,  "split expects split_size be non-negative, but got split_size=", split_size);
  int64_t dim_size = self.size(dim);
  TORCH_CHECK(split_size > 0 || dim_size == 0,
           "split_size can only be 0 if dimension size is 0, "
           "but got dimension size of ", dim_size);
  // 计算在给定维度上的分割数
  int64_t num_splits = 1;
  if (split_size != 0) {
    num_splits = std::max<int64_t>((dim_size + split_size - 1) / split_size, 1);
  }
  return num_splits;
}

// 内联函数，检查张量列表中的所有张量是否具有相同的维度数
inline bool have_same_ndims(TensorList tensors) {
  auto ndim = tensors[0].dim();
  for (const auto tensor_idx : c10::irange(tensors.size())) {
    if(tensors[tensor_idx].dim() != ndim) {
      return false;
    }
  }
  return true;
}

// 内联函数，检查张量列表中的张量的前导维度是否匹配
inline void leading_dimension_matches(TensorList tensors, int64_t dim) {
  auto tensor_zero_size = tensors[0].sizes();
  std::vector<c10::SymInt> leading_dim_sizes(tensor_zero_size.begin(), tensor_zero_size.begin() + dim);
  for (const auto i : c10::irange(tensors.size())) {
    at::Tensor tensor = tensors[i];
    // 检查张量的前导维度是否与指定的维度匹配
    for(const auto j : c10::irange(dim)) {
      // 使用 C++ 的范围迭代器循环遍历从 0 到 dim-1 的每一个索引 j
      TORCH_CHECK(
        // 使用 TORCH_CHECK 断言，验证条件是否成立：tensor 的第 j 维度大小等于 leading_dim_sizes[j]
        tensor.size(j) == leading_dim_sizes[j],
        "_chunk_cat expects same sizes of 0,...,dim-1 dimensions for all tensors"
      );
    }
  }
} // 关闭 at::native 命名空间

inline int64_t preprocess_chunk_cat_inputs(TensorList tensors, int64_t dim, int64_t num_chunks) {
  // 检查 num_chunks 参数是否大于等于 1
  TORCH_CHECK(num_chunks >= 1, "_chunk_cat 期望正数的 num_chunks");
  // 检查输入的张量列表是否非空
  TORCH_CHECK(!tensors.empty(),
           "_chunk_cat 期望非空的输入张量列表");
  // 获取第一个张量的数据类型作为期望的数据类型
  auto expected_dtype = tensors[0].dtype();
  // 获取第一个张量的设备作为期望的设备
  auto expected_device = tensors[0].device();
  // 遍历张量列表，检查每个张量是否非空、数据类型一致，并且位于同一设备上
  for(const auto i : c10::irange(tensors.size())) {
    TORCH_CHECK(tensors[i].numel() > 0, "_chunk_cat 期望非空的张量");
    TORCH_CHECK(tensors[i].dtype() == expected_dtype, "_chunk_cat 期望所有输入张量具有相同的数据类型");
    TORCH_CHECK(tensors[i].device() == expected_device, "_chunk_cat 期望所有输入张量位于同一设备上");
  }
  // 如果输入张量具有相同的维度数，则将 dim 调整为合适的维度
  if (have_same_ndims(tensors)) {
    dim = maybe_wrap_dim(dim, tensors[0].dim());
  } else {
    // 否则，检查 dim 是否非负，并且对于每个输入张量，检查 dim 是否小于其维度数
    TORCH_CHECK(dim >= 0, "_chunk_cat 当输入张量具有不同的维度时，期望非负的 dim")
    for(const auto i : c10::irange(tensors.size())) {
      TORCH_CHECK(dim < tensors[i].ndimension(), "_chunk_cat 对所有输入张量，期望 dim < ndim");
    }
  }
  // 检查主导维度是否匹配
  leading_dimension_matches(tensors, dim);
  // 返回处理后的 dim
  return dim;
}

} // 结束 at::native 命名空间
```