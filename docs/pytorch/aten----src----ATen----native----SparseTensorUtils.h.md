# `.\pytorch\aten\src\ATen\native\SparseTensorUtils.h`

```
#pragma once

#include <ATen/Parallel.h>
#include <ATen/SparseTensorImpl.h>
#include <ATen/core/Tensor.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/tensor.h>
#endif

namespace at::sparse {

// Just for documentary purposes
using SparseTensor = Tensor; // 定义 SparseTensor 别名为 Tensor
using SparseType = Type; // 定义 SparseType 别名为 Type

// This is an internal utility function for getting at the SparseTensorImpl,
// so that we can write sparse tensor specific accessors for special fields
// in SparseTensor.  You should only use this for writing low level
// setters/getters for SparseTensorImpl fields; otherwise, you should use
// the low level setters/getters that were implemented using this.
//
// This may be called repeatedly, so make sure it's pretty cheap.
// 获取 SparseTensorImpl 对象的内部实用函数，以便为 SparseTensor 的特殊字段编写稀疏张量特定的访问器。
// 只应用于编写 SparseTensorImpl 字段的低级设置器/获取器；否则，应使用基于此实现的低级设置器/获取器。
inline SparseTensorImpl* get_sparse_impl(const SparseTensor& self) {
  TORCH_INTERNAL_ASSERT(
      self.is_sparse(), "_internal_get_SparseTensorImpl: not a sparse tensor");
  return static_cast<SparseTensorImpl*>(self.unsafeGetTensorImpl());
}

// Takes indices and values and directly puts them into the sparse tensor, no
// copy.  This used to be called THSTensor_(_move)
// 将索引和值直接放入稀疏张量中，无需复制。这曾被称为 THSTensor_(_move)
inline void alias_into_sparse(
    const SparseTensor& self,
    const Tensor& indices,
    const Tensor& values) {
  get_sparse_impl(self)->set_indices_and_values_unsafe(indices, values);
}

// Take indices and values and makes a (data) copy of them to put into the
// sparse indices/values.  This used to be called THSTensor_(_set)
// 将索引和值复制一份放入稀疏张量中。这曾被称为 THSTensor_(_set)
inline void copy_into_sparse(
    const SparseTensor& self,
    const Tensor& indices,
    const Tensor& values,
    bool non_blocking) {
  alias_into_sparse(
      self,
      indices.to(self._indices().options(), non_blocking, /*copy=*/true),
      values.to(self._values().options(), non_blocking, /*copy=*/true));
}

// TODO: put this into the public API
// 检查两个张量是否是相同的张量对象
inline bool is_same_tensor(const Tensor& lhs, const Tensor& rhs) {
  return lhs.unsafeGetTensorImpl() == rhs.unsafeGetTensorImpl();
}

// 检查两个稀疏张量是否具有相同的密度（即稀疏维度和密集维度）
inline bool is_same_density(const SparseTensor& self, const SparseTensor& src) {
  return self.sparse_dim() == src.sparse_dim() &&
      self.dense_dim() == src.dense_dim();
}

// Give us a new values tensor, with the same dimensionality
// as 'values' but with a new number of non-zero elements.
// TODO: Expose this for real in ATen, some day?
// NB: Doesn't preserve data.
// 返回一个具有与 'values' 相同维度但新的非零元素数量的新 values 张量
inline Tensor new_values_with_size_of(const Tensor& values, int64_t nnz) {
  std::vector<int64_t> size = values.sizes().vec();
  size[0] = nnz;
  return at::empty(size, values.options());
}

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
// 将稀疏索引张量（Tensor）展平为一维索引张量的辅助函数。
// 例如，
//   input = [[2, 4, 0],
//            [3, 1, 10]]
//   full_size = [2, 12]
//   output = [ 2 * 12 + 3, 4 * 12 + 1, 0 * 12 + 10 ] = [27, 49, 10]
//
// 换句话说，假设每个 `indices[i, :]` 都是形状为 `full_size` 的张量 `t` 的有效索引。这将返回对应的索引。
// Flatten sparse tensor's indices from nD to 1D, similar to NOTE [ Flatten
// Sparse Indices ], except this one allows partial flatten: only flatten on
// specified dims. Note that the flatten indices might be uncoalesced if
// dims_to_flatten.size() < sparse_dim. Also if input indices is already
// coalesced, the flattened indices will also be sorted.
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
TORCH_API Tensor flatten_indices_by_dims(
    const Tensor& indices,
    const IntArrayRef& sizes,
    const IntArrayRef& dims_to_flatten);

// Find the CSR representation for a row `indices` from the COO format
TORCH_API Tensor coo_to_csr(const int64_t* indices, int64_t dim, int64_t nnz);

// Create a new tensor filled with zeros, shaped like `t`
TORCH_API Tensor zeros_like_with_indices(const Tensor& t);

// Holder class for tensor geometry information, supporting static shape length `static_shape_max_len`
template <size_t static_shape_max_len>
class TensorGeometryHolder {
  using geometry_holder_t = std::array<int64_t, static_shape_max_len>;

 public:
  // Constructor initializing sizes and strides from given arrays
  explicit TensorGeometryHolder(
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options = {}) {
    std::copy(sizes.begin(), sizes.end(), t_sizes.begin());
    std::copy(strides.begin(), strides.end(), t_strides.begin());
  }

  // Constructor initializing sizes and strides from a tensor `t`
  explicit TensorGeometryHolder(const Tensor& t)
      : TensorGeometryHolder(t.sizes(), t.strides()) {}

  // Dereference operator returning tuple of sizes and strides
  auto operator*() const {
    return std::make_tuple(t_sizes, t_strides);
  }

 private:
  geometry_holder_t t_sizes;    // Holder for tensor sizes
  geometry_holder_t t_strides;  // Holder for tensor strides
};

// Specialization of TensorGeometryHolder for static shape length 0
template <>
class TensorGeometryHolder<0> {
  using geometry_holder_t = Tensor;

 public:
  // Constructor initializing sizes and strides from given arrays and options
  explicit TensorGeometryHolder(
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options) {
    const int64_t t_ndims = sizes.size();
    // Create a tensor for sizes and strides on CPU
    const auto cpu_options = TensorOptions(options).dtype(kLong).device(kCPU);
    Tensor t_sizes_and_strides_cpu = at::empty({2, t_ndims}, cpu_options);
    t_sizes_and_strides_cpu.select(0, 0).copy_(at::tensor(sizes, cpu_options));
    t_sizes_and_strides_cpu.select(0, 1).copy_(
        at::tensor(strides, cpu_options));
    // Convert sizes and strides tensor to specified device
    const Tensor t_sizes_and_strides =
        t_sizes_and_strides_cpu.to(options.device());
    t_sizes = t_sizes_and_strides.select(0, 0);  // Extract sizes from the tensor
    t_strides = t_sizes_and_strides.select(0, 1);


    // 从 t_sizes_and_strides 中选择索引为 0 的维度的 strides（步幅），并赋值给 t_strides
    t_strides = t_sizes_and_strides.select(0, 1);



  }

  explicit TensorGeometryHolder(const Tensor& t)
      : TensorGeometryHolder(t.sizes(), t.strides(), t.options()) {}


  // 使用给定张量 t 的大小、步幅和选项构造 TensorGeometryHolder 对象的显式构造函数
  explicit TensorGeometryHolder(const Tensor& t)
      : TensorGeometryHolder(t.sizes(), t.strides(), t.options()) {}



  auto operator*() const {
    return std::make_tuple(
        t_sizes.template data_ptr<int64_t>(),
        t_strides.template data_ptr<int64_t>());
  }


  // 定义解引用操作符 * ，返回一个包含 t_sizes 和 t_strides 指向数据的元组
  auto operator*() const {
    return std::make_tuple(
        t_sizes.template data_ptr<int64_t>(),  // 获取 t_sizes 指向数据的指针
        t_strides.template data_ptr<int64_t>());  // 获取 t_strides 指向数据的指针
  }



 private:
  geometry_holder_t t_sizes;
  geometry_holder_t t_strides;


 private:
  // 私有成员变量，分别用于保存张量的大小（sizes）和步幅（strides）
  geometry_holder_t t_sizes;    // 存储张量的大小的对象
  geometry_holder_t t_strides;  // 存储张量的步幅的对象
};

// 结束了 at::sparse 命名空间的定义

// 返回给定形状张量的所有索引。
//
// full_coo_indices(shape) 等同于
// torch.ones(shape).nonzero().transpose(-2, -1)，但速度更快。
TORCH_API Tensor full_coo_indices(IntArrayRef sizes, TensorOptions options);

// at::sparse 命名空间的结束
} // namespace at::sparse
```