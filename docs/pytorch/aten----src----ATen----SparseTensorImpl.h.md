# `.\pytorch\aten\src\ATen\SparseTensorImpl.h`

```
#pragma once

#include <ATen/Tensor.h>
#include <c10/core/TensorImpl.h>
#include <c10/core/impl/TorchDispatchModeTLS.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/resize.h>
#endif

namespace at {

// SparseTensorImpl 继承自 TensorImpl，表示稀疏张量的实现
struct TORCH_API SparseTensorImpl : public TensorImpl {
  // Stored in COO format, indices + values.

  // INVARIANTS:
  // sparse_dim: range [0, len(shape)]; sparse_dim + dense_dim = len(shape)
  // dense_dim : range [0, len(shape)]; sparse_dim + dense_dim = len(shape)
  // _indices.shape: dimensionality: 2,  shape: (sparse_dim, nnz)
  // _values.shape:  dimensionality: 1 + dense_dim.  shape: (nnz,
  // shape[sparse_dim:])

  int64_t sparse_dim_ = 0; // number of sparse dimensions
  int64_t dense_dim_ = 0; // number of dense dimensions

  Tensor indices_; // always a LongTensor
  Tensor values_;

  // A sparse tensor is 'coalesced' if every index occurs at most once in
  // the indices tensor, and the indices are in sorted order.  (This means
  // that it is very easy to convert a coalesced tensor to CSR format: you
  // need only compute CSR format indices.)
  //
  // Most math operations can only be performed on coalesced sparse tensors,
  // because many algorithms proceed by merging two sorted lists (of indices).
  bool coalesced_ = false;

  // compute_numel with integer multiplication overflow check, see gh-57542
  // 刷新稀疏张量的元素数量
  void refresh_numel() {
    TensorImpl::safe_refresh_numel();
  }

 public:
  // Public for now...
  // 显式构造函数，初始化稀疏张量的分发键集和数据类型
  explicit SparseTensorImpl(at::DispatchKeySet, const caffe2::TypeMeta);

  // 释放资源
  void release_resources() override;

  // 返回稀疏张量的非零元素数量
  int64_t nnz() const {
    return values_.size(0);
  }

  // 返回稀疏张量的符号化非零元素数量
  c10::SymInt sym_nnz() const {
    return values_.sym_size(0);
  }

  // 返回稀疏张量的稀疏维度
  int64_t sparse_dim() const {
    return sparse_dim_;
  }

  // 返回稀疏张量的密集维度
  int64_t dense_dim() const {
    return dense_dim_;
  }

  // 返回稀疏张量是否已压缩
  bool coalesced() const {
    return coalesced_;
  }

  // 返回稀疏张量的索引
  Tensor indices() const {
    return indices_;
  }

  // 返回稀疏张量的值
  Tensor values() const {
    return values_;
  }

  // 设置稀疏张量的尺寸
  void set_size(int64_t dim, int64_t new_size) override;

  // 设置稀疏张量的步长
  void set_stride(int64_t dim, int64_t new_stride) override;

  // 设置稀疏张量的存储偏移量
  void set_storage_offset(int64_t storage_offset) override;

#ifdef DEBUG
  // 检查稀疏张量是否具有存储
  bool has_storage() const override;
#endif

  // 警告：此函数不保留稀疏维度和密集维度对于索引和值的不变性
  // 直接改变稀疏张量的大小
  void raw_resize_(int64_t sparse_dim, int64_t dense_dim, IntArrayRef size) {
    TORCH_CHECK(
        allow_tensor_metadata_change(),
        "raw_resize_ ",
        err_msg_tensor_metadata_change_not_allowed);
    TORCH_CHECK(
        !has_symbolic_sizes_strides_,
        "raw_resize_ called on tensor with symbolic shape")
    set_sizes_and_strides(size, std::vector<int64_t>(size.size()));
    sparse_dim_ = sparse_dim;
    dense_dim_ = dense_dim;
    refresh_numel();
  }

  // NOTE: This function preserves invariants of sparse_dim/dense_dim with
  // respect to indices and values.
  //
  // NOTE: This function supports the following cases:
  // 1. When we keep the number of dense dimensions unchanged, and NOT shrinking
  // the size of any of the dense dimensions.
  // 2. When we keep the number of sparse dimensions unchanged, and NOT
  // shrinking the size of any of the sparse dimensions.
  // 3. When the sparse tensor has zero nnz, in which case we are free to change
  // the shapes of both its sparse and dense dimensions.
  //
  // This function DOESN'T support (and will throw an error) the following
  // cases:
  // 1. When we attempt to change the number of sparse dimensions on a non-empty
  // sparse tensor (such an operation will invalidate the indices stored).
  // 2. When we attempt to change the number of dense dimensions on a non-empty
  // sparse tensor (such an operation will behave differently from an equivalent
  // dense tensor's resize method, and for API consistency we don't support it).
  // 3. When we attempt to shrink the size of any of the dense dimensions on a
  // non-empty sparse tensor (such an operation will behave differently from an
  // equivalent dense tensor's resize method, and for API consistency we don't
  // support it).
  // 4. When we attempt to shrink the size of any of the sparse dimensions on a
  // non-empty sparse tensor (this could make some of the stored indices
  // out-of-bound and thus unsafe).
  template <typename T>
  void _resize_(int64_t sparse_dim, int64_t dense_dim, ArrayRef<T> size) {
    // 检查是否允许改变张量元数据
    TORCH_CHECK(
        allow_tensor_metadata_change(),
        "resize_ ",
        err_msg_tensor_metadata_change_not_allowed);
    // 检查张量是否具有符号形状
    TORCH_CHECK(
        !has_symbolic_sizes_strides_,
        "resize_ called on tensor with symbolic shape")
    // 检查维度数量是否匹配
    TORCH_CHECK(
        sparse_dim + dense_dim == static_cast<int64_t>(size.size()),
        "number of dimensions must be sparse_dim (",
        sparse_dim,
        ") + dense_dim (",
        dense_dim,
        "), but got ",
        size.size());
    // 如果稀疏张量的非零元素数大于零，则执行以下检查
    if (nnz() > 0) {
      // 提示消息，列出备选选项
      auto alt_options_msg =
          "You could try the following options:\n\
# 如果需要创建一个指定大小的空稀疏张量，请使用 `x = torch.sparse_coo_tensor(size)`。
# 如果需要调整此张量的大小，您有以下选项：
#   1. 对于稀疏和密集维度，保持它们的数量不变且大小不缩小，然后再尝试相同的调用。
#   2. 或者，从此稀疏张量中创建一个新的稀疏张量，其正确的索引和值。

TORCH_CHECK(
    sparse_dim == sparse_dim_,
    "修改非空稀疏张量的稀疏维度数量（从 ",
    sparse_dim_,
    " 改为 ",
    sparse_dim,
    "）不受支持。\n",
    alt_options_msg);

TORCH_CHECK(
    dense_dim == dense_dim_,
    "修改非空稀疏张量的密集维度数量（从 ",
    dense_dim_,
    " 改为 ",
    dense_dim,
    "）不受支持。\n",
    alt_options_msg);

bool shrinking_sparse_dims = false;
bool shrinking_dense_dim = false;
auto sparse_size_original = generic_sizes<T>().slice(0, sparse_dim);
auto sparse_size_new = size.slice(0, sparse_dim);

# 检查稀疏维度是否有缩小的情况
for (const auto i : c10::irange(sparse_dim)) {
    if (sparse_size_new[i] < sparse_size_original[i]) {
        shrinking_sparse_dims = true;
        break;
    }
}

auto dense_size_original = generic_sizes<T>().slice(sparse_dim);
auto dense_size_new = size.slice(sparse_dim);

# 检查密集维度是否有缩小的情况
for (const auto i : c10::irange(dense_dim)) {
    if (dense_size_new[i] < dense_size_original[i]) {
        shrinking_dense_dim = true;
        break;
    }
}

TORCH_CHECK(
    !shrinking_sparse_dims,
    "缩小非空稀疏张量的稀疏维度大小（从 ",
    sparse_size_original,
    " 改为 ",
    sparse_size_new,
    "）不受支持。\n",
    alt_options_msg);

TORCH_CHECK(
    !shrinking_dense_dim,
    "缩小非空稀疏张量的密集维度大小（从 ",
    dense_size_original,
    " 改为 ",
    dense_size_new,
    "）不受支持。\n",
    alt_options_msg);
}

auto sizes_and_strides = generic_sizes<T>();

# 检查当前张量的大小是否与给定的大小相等
const bool size_equals_sizes = std::equal(
    size.begin(),
    size.end(),
    sizes_and_strides.begin(),
    sizes_and_strides.end());

# 如果当前大小与给定大小不相等，或者稀疏维度或密集维度与原始张量不同，则进行调整
if ((!size_equals_sizes) || (sparse_dim != sparse_dim_) ||
    (dense_dim != dense_dim_)) {
  auto nnz = at::symint::sizes<T>(values())[0];
  std::vector<T> values_size = {nnz};
  auto dense_size = size.slice(sparse_dim);
  values_size.insert(
      values_size.end(), dense_size.begin(), dense_size.end());
  at::symint::resize_<T>(values_, values_size);
  at::symint::resize_<T>(indices_, {T(sparse_dim), nnz});
}

# 如果当前大小与给定大小不相等，则设置新的大小和步长
if (!size_equals_sizes) {
  set_sizes_and_strides(size, std::vector<T>(size.size()));
  }

  // 将稀疏维度和密集维度分别赋值为给定的值，并刷新张量的元素数量
  void resize_(int64_t sparse_dim, int64_t dense_dim, ArrayRef<int64_t> size) {
    return _resize_(sparse_dim, dense_dim, size);
  }

  // 将稀疏维度和密集维度分别赋值为给定的值，并刷新张量的元素数量（处理符号整数的情况）
  void resize_(
      int64_t sparse_dim,
      int64_t dense_dim,
      ArrayRef<c10::SymInt> size) {
    return _resize_(sparse_dim, dense_dim, size);
  }

  // 注意：此函数将调整稀疏张量的大小，并清空 `indices` 和 `values`。
  void resize_and_clear_(
      int64_t sparse_dim,
      int64_t dense_dim,
      IntArrayRef size) {
    TORCH_CHECK(
        allow_tensor_metadata_change(),
        "resize_and_clear_ ",
        err_msg_tensor_metadata_change_not_allowed);
    TORCH_CHECK(
        !has_symbolic_sizes_strides_,
        "resize_and_clear_ called on tensor with symbolic shape")
    TORCH_CHECK(
        sparse_dim + dense_dim == static_cast<int64_t>(size.size()),
        "number of dimensions must be sparse_dim (",
        sparse_dim,
        ") + dense_dim (",
        dense_dim,
        "), but got ",
        size.size());

    // 设置张量的大小和步幅
    set_sizes_and_strides(size, std::vector<int64_t>(size.size()));
    sparse_dim_ = sparse_dim;
    dense_dim_ = dense_dim;

    // 创建空的索引张量
    auto empty_indices = at::empty({sparse_dim, 0}, indices().options());
    // 创建空的值张量
    std::vector<int64_t> values_size = {0};
    auto dense_size = sizes().slice(sparse_dim);
    values_size.insert(values_size.end(), dense_size.begin(), dense_size.end());
    auto empty_values = at::empty(values_size, values().options());
    // 设置张量的索引和值为这些空张量
    set_indices_and_values_unsafe(empty_indices, empty_values);
    // 刷新张量的元素数量
    refresh_numel();
  }

  // 设置张量是否紧凑（coalesced）
  void set_coalesced(bool coalesced) {
    TORCH_CHECK(
        allow_tensor_metadata_change(),
        "set_coalesced ",
        err_msg_tensor_metadata_change_not_allowed);
    coalesced_ = coalesced;
  }

  // 注意：此函数仅在内部使用，不会暴露给 Python 前端
  // 设置非零元素数和缩小张量的范围
  void set_nnz_and_narrow(int64_t new_nnz) {
    TORCH_CHECK(
        allow_tensor_metadata_change(),
        "set_nnz_and_narrow ",
        err_msg_tensor_metadata_change_not_allowed);
    AT_ASSERT(new_nnz <= nnz());
    // 缩小索引和值张量的范围以适应新的非零元素数
    indices_ = indices_.narrow(1, 0, new_nnz);
    values_ = values_.narrow(0, 0, new_nnz);
    // 如果新的非零元素数小于 2，则设置张量为紧凑状态
    if (new_nnz < 2) {
      coalesced_ = true;
  }
}

// Takes indices and values and directly puts them into the sparse tensor, no
// copy. NOTE: this function is unsafe because it doesn't check whether any
// indices are out of boundaries of `sizes`, so it should ONLY be used where
// we know that the indices are guaranteed to be within bounds. This used to
// be called THSTensor_(_move) NB: This used to be able to avoid a refcount
// bump, but I was too lazy to make it happen
// 将给定的索引和值直接放入稀疏张量中，不进行复制。注意：此函数不安全，因为它不检查任何索引是否超出了 `sizes` 的边界，
// 因此只应在确保索引肯定在范围内时使用。此函数曾被称为 THSTensor_(_move) 注意：曾可以避免引用计数的增加，但我懒得去实现它。

void set_indices_and_values_unsafe(
    const Tensor& indices,
    const Tensor& values);

template <typename VariableVersion>
c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach_core(
    VariableVersion&& version_counter,
    bool allow_tensor_metadata_change) const {
  const auto mode_stack_len = c10::impl::TorchDispatchModeTLS::stack_len();
  c10::impl::PyInterpreter&& interpreter = nullptr;
  if (mode_stack_len > 0 &&
      !c10::impl::tls_is_dispatch_key_excluded(DispatchKey::Python)) {
    const auto& cur_torch_dispatch_mode_state =
        c10::impl::TorchDispatchModeTLS::get_stack_at(mode_stack_len - 1);
    interpreter = cur_torch_dispatch_mode_state->pyinterpreter();
  } else if (
      key_set_.has(DispatchKey::Python) &&
      !c10::impl::tls_is_dispatch_key_excluded(DispatchKey::Python)) {
    interpreter = pyobj_slot_.load_pyobj_interpreter();
  } else {
    // otherwise just copy the SparseTensorImpl and not the PyObject.
    // 否则只复制 SparseTensorImpl 而不是 PyObject。
    auto impl = c10::make_intrusive<SparseTensorImpl>(key_set(), dtype());
    copy_tensor_metadata(
        /*src_sparse_impl=*/this,
        /*dest_sparse_impl=*/impl.get(),
        /*version_counter=*/version_counter,
        /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
    impl->refresh_numel();
    return impl;
  }
  auto r = interpreter->detach(this);
  r->set_version_counter(std::forward<VariableVersion>(version_counter));
  r->set_allow_tensor_metadata_change(allow_tensor_metadata_change);
  return r;
}

/**
 * Return a TensorImpl that is a shallow-copy of this TensorImpl.
 *
 * For usage of `version_counter` and `allow_tensor_metadata_change`,
 * see NOTE [ TensorImpl Shallow-Copying ].
 */
// 返回一个当前 TensorImpl 的浅拷贝的 TensorImpl。
// 
// 关于 `version_counter` 和 `allow_tensor_metadata_change` 的使用，请参见注释 [ TensorImpl Shallow-Copying ]。
c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
    const c10::VariableVersion& version_counter,
    bool allow_tensor_metadata_change) const override {
  return shallow_copy_and_detach_core(
      version_counter, allow_tensor_metadata_change);
}

/**
 * Return a TensorImpl that is a shallow-copy of this TensorImpl.
 *
 * For usage of `version_counter` and `allow_tensor_metadata_change`,
 * see NOTE [ TensorImpl Shallow-Copying ].
 */
// 返回一个当前 TensorImpl 的浅拷贝的 TensorImpl。
// 
// 关于 `version_counter` 和 `allow_tensor_metadata_change` 的使用，请参见注释 [ TensorImpl Shallow-Copying ]。
c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
    c10::VariableVersion&& version_counter,
    bool allow_tensor_metadata_change) const override {
    return shallow_copy_and_detach_core(
        std::move(version_counter), allow_tensor_metadata_change);
  }

  /**
   * 对另一个 TensorImpl 进行浅拷贝到当前 TensorImpl 中。
   *
   * 为什么这个函数不检查当前 TensorImpl 的 `allow_tensor_metadata_change_`，请参见 NOTE [ TensorImpl Shallow-Copying ]。
   */
  void shallow_copy_from(const c10::intrusive_ptr<TensorImpl>& impl) override {
    AT_ASSERT(has_compatible_shallow_copy_type(impl->key_set()));
    auto sparse_impl = static_cast<const SparseTensorImpl*>(impl.get());
    copy_tensor_metadata(
        /*src_sparse_impl=*/sparse_impl,
        /*dest_sparse_impl=*/this,
        /*version_counter=*/version_counter(),
        /*allow_tensor_metadata_change=*/allow_tensor_metadata_change());
    refresh_numel();
  }

 private:
  explicit SparseTensorImpl(
      at::DispatchKeySet,
      const caffe2::TypeMeta,
      at::Tensor indices,
      at::Tensor values);

  /**
   * 从一个 TensorImpl 拷贝张量元数据字段（如 sizes / strides / storage 指针 / storage_offset）到另一个 TensorImpl。
   *
   * 关于 `version_counter` 和 `allow_tensor_metadata_change` 的使用，请参见 NOTE [ TensorImpl Shallow-Copying ]。
   */
  static void copy_tensor_metadata(
      const SparseTensorImpl* src_sparse_impl,
      SparseTensorImpl* dest_sparse_impl,
      c10::VariableVersion version_counter,
      bool allow_tensor_metadata_change) {
    TensorImpl::copy_tensor_metadata(
        src_sparse_impl,
        dest_sparse_impl,
        std::move(version_counter),
        allow_tensor_metadata_change);

    // Sparse-specific fields
    dest_sparse_impl->sparse_dim_ = src_sparse_impl->sparse_dim();
    dest_sparse_impl->dense_dim_ = src_sparse_impl->dense_dim();
    dest_sparse_impl->indices_ = src_sparse_impl->indices();
    dest_sparse_impl->values_ = src_sparse_impl->values();
    dest_sparse_impl->coalesced_ = src_sparse_impl->coalesced();
  }

  const char* tensorimpl_type_name() const override;
};

// 结束命名空间 'at'
} // namespace at
```