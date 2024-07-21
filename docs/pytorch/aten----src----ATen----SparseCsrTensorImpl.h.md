# `.\pytorch\aten\src\ATen\SparseCsrTensorImpl.h`

```py
#pragma once

#include <ATen/Tensor.h> // 包含 ATen 库的 Tensor 类
#include <c10/core/TensorImpl.h> // 包含 c10 库的 TensorImpl 类
#include <c10/core/impl/TorchDispatchModeTLS.h> // 包含 c10 库的 TorchDispatchModeTLS 类
#include <c10/util/Exception.h> // 包含 c10 库的 Exception 类

namespace at {

// 实现稀疏 CSR 张量的结构体。使用三个 1-D 张量来表示数据：`crow_indices_`、`col_indices_` 和 `values_`。
// `crow_indices_` 张量是一个整数张量，形状为 `(size(0) + 1)`，表示 CSR 张量的压缩行索引。
// `col_indices_` 张量是一个整数张量，形状为 `(nnz())`，显式存储稀疏张量每个值的列索引。
// `values_` 张量可以是任何 PyTorch 支持的数据类型，形状为 `(nnz())`。
//
// 由于 CSR 格式相对 COO 格式的主要优势是计算速度，因此需要确保这些数据结构能够平滑地与优化库（如 MKL 和 MAGMA）进行接口对接。
// 由于目前 PyTorch 的 MKL 接口使用 int32 类型的索引，因此在调用 MKL 的 SPMM 或 SPMV 等函数时，确保 `crow_indices` 和 `col_indices` 是 int32 类型非常重要。
//
// 如果不使用 MKL，则可以使用 64 位整数张量作为索引。
struct TORCH_API SparseCsrTensorImpl : public TensorImpl {
  Tensor crow_indices_; // CSR 张量的压缩行索引
  Tensor col_indices_;  // CSR 张量的列索引
  Tensor values_;       // CSR 张量的值
  Layout layout_;       // 张量的布局

 public:
  // 构造函数，初始化稀疏 CSR 张量实现
  explicit SparseCsrTensorImpl(
      at::DispatchKeySet,
      at::Device device,
      Layout layout,
      const caffe2::TypeMeta);

  // 调整稀疏 CSR 张量的大小，并保持稀疏性
  void resize_(int64_t nnz, IntArrayRef size);

  // 调整稀疏 CSR 张量的大小，并清空其内容
  void resize_and_clear_(
      int64_t sparse_dim,
      int64_t dense_dim,
      IntArrayRef size);

  // 根据源张量调整稀疏 CSR 张量的大小
  void resize_as_sparse_compressed_tensor_(const Tensor& src);

  // 设置稀疏 CSR 张量的成员张量和大小（支持 SymIntArrayRef 类型的大小）
  void set_member_tensors(
      const Tensor& crow_indices,
      const Tensor& col_indices,
      const Tensor& values,
      c10::SymIntArrayRef size);

  // 设置稀疏 CSR 张量的成员张量和大小（支持 IntArrayRef 类型的大小）
  void set_member_tensors(
      const Tensor& crow_indices,
      const Tensor& col_indices,
      const Tensor& values,
      IntArrayRef size);

  // 获取压缩行索引 `crow_indices_` 张量
  const Tensor& compressed_indices() const {
    return crow_indices_;
  }

  // 获取列索引 `col_indices_` 张量
  const Tensor& plain_indices() const {
    return col_indices_;
  }

  // 获取值 `values_` 张量
  const Tensor& values() const {
    return values_;
  }

  // 返回稀疏 CSR 张量的非零元素数量
  int64_t nnz() {
    return col_indices_.size(-1);
  }

  // 返回批次维度的大小
  inline int64_t batch_dim() const noexcept {
    return crow_indices_.dim() - 1;
  }

  // 返回稀疏维度的大小
  inline int64_t sparse_dim() const noexcept {
    return 2;
  }

  // 返回密集维度的大小
  inline int64_t dense_dim() const noexcept {
    return values_.dim() - batch_dim() - block_dim() - 1;
  }

 private:
  // 返回块维度的大小
  inline int64_t block_dim() const noexcept {
    // 返回一个布尔值：如果布局为稀疏的 BSR 或 BSC，则返回 2，否则返回 0
    return (layout_ == kSparseBsr || layout_ == kSparseBsc ? 2 : 0);
  }

 protected:
  // 返回一个 IntArrayRef，表示自定义步长
  IntArrayRef strides_custom() const override;
  // 返回一个 SymIntArrayRef，表示自定义对称步长
  SymIntArrayRef sym_strides_custom() const override;
  // 检查在指定内存格式下是否连续
  bool is_contiguous_custom(MemoryFormat) const override;

 public:
  // 设置指定维度的大小
  void set_size(int64_t dim, int64_t new_size) override;
  // 设置指定维度的步长
  void set_stride(int64_t dim, int64_t new_stride) override;
  // 设置存储偏移量
  void set_storage_offset(int64_t storage_offset) override;
  // 返回当前的布局类型
  Layout layout_impl() const override {
    return layout_;
  }
  // 设置布局类型
  void set_layout(Layout layout) {
    switch (layout) {
      // 如果布局是稀疏的 CSR、CSC、BSR 或 BSC，则设置布局
      case kSparseCsr:
      case kSparseCsc:
      case kSparseBsr:
      case kSparseBsc:
        layout_ = layout;
        break;
      // 否则抛出异常，说明布局不支持
      default:
        TORCH_CHECK(false, "unsupported layout ", layout);
    }
  }

  // 返回一个浅拷贝的 TensorImpl，根据变量版本和是否允许改变张量元数据
  template <typename VariableVersion>
  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach_core(
      VariableVersion&& version_counter,
      bool allow_tensor_metadata_change) const {
    // 获取当前的 Torch 调度模式栈长度
    const auto mode_stack_len = c10::impl::TorchDispatchModeTLS::stack_len();
    // 初始化 Python 解释器为 nullptr
    c10::impl::PyInterpreter&& interpreter = nullptr;
    // 如果 Torch 调度模式栈长度大于 0，并且 Python 调度键不被排除
    if (mode_stack_len > 0 &&
        !c10::impl::tls_is_dispatch_key_excluded(DispatchKey::Python)) {
      // 获取当前 Torch 调度模式状态
      const auto& cur_torch_dispatch_mode_state =
          c10::impl::TorchDispatchModeTLS::get_stack_at(mode_stack_len - 1);
      // 获取当前的 Python 解释器
      interpreter = cur_torch_dispatch_mode_state->pyinterpreter();
    } else if (
        key_set_.has(DispatchKey::Python) &&
        !c10::impl::tls_is_dispatch_key_excluded(DispatchKey::Python)) {
      // 如果当前的调度键集合包含 Python 调度键，并且 Python 调度键不被排除
      // 加载 Python 对象槽的 Python 解释器
      interpreter = pyobj_slot_.load_pyobj_interpreter();
    } else {
      // 否则，只复制稀疏张量实现而不复制 PyObject
      auto impl = c10::make_intrusive<SparseCsrTensorImpl>(
          key_set(), device(), layout_impl(), dtype());
      // 复制张量元数据
      copy_tensor_metadata(
          /*src_sparse_impl=*/this,
          /*dest_sparse_impl=*/impl.get(),
          /*version_counter=*/version_counter,
          /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
      // 刷新张量元素数量
      impl->refresh_numel();
      // 返回新实现的浅拷贝
      return impl;
    }
    // 否则，使用解释器分离当前的 TensorImpl
    auto r = interpreter->detach(this);
    // 设置版本计数器和是否允许张量元数据更改
    r->set_version_counter(std::forward<VariableVersion>(version_counter));
    r->set_allow_tensor_metadata_change(allow_tensor_metadata_change);
    // 返回分离后的 TensorImpl
    return r;
  }

  /**
   * 返回一个 TensorImpl 的浅拷贝。
   *
   * 有关 `version_counter` 和 `allow_tensor_metadata_change` 的使用，请参见 NOTE [ TensorImpl Shallow-Copying ]。
   */
  // 返回一个浅拷贝的 TensorImpl
  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
      const c10::VariableVersion& version_counter,
      bool allow_tensor_metadata_change) const override {
    return shallow_copy_and_detach_core(
        version_counter, allow_tensor_metadata_change);
  }



  /**
   * 返回一个当前 TensorImpl 的浅拷贝的 TensorImpl。
   *
   * 有关 `version_counter` 和 `allow_tensor_metadata_change` 的使用，请参见 NOTE [ TensorImpl Shallow-Copying ]。
   */
  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
      c10::VariableVersion&& version_counter,
      bool allow_tensor_metadata_change) const override {
    return shallow_copy_and_detach_core(
        std::move(version_counter), allow_tensor_metadata_change);
  }



 private:
  explicit SparseCsrTensorImpl(
      at::DispatchKeySet key_set,
      const caffe2::TypeMeta data_type,
      at::Tensor crow_indices,
      at::Tensor col_indices,
      at::Tensor values,
      at::Layout layout);

  const char* tensorimpl_type_name() const override;



  /**
   * 将来自一个 TensorImpl 的张量元数据字段（如 sizes / strides / storage 指针 /
   * storage_offset）复制到另一个 TensorImpl。
   *
   * 有关 `version_counter` 和 `allow_tensor_metadata_change` 的使用，请参见 NOTE
   * [ TensorImpl Shallow-Copying ]。
   */
  static void copy_tensor_metadata(
      const SparseCsrTensorImpl* src_sparse_impl,
      SparseCsrTensorImpl* dest_sparse_impl,
      c10::VariableVersion version_counter,
      bool allow_tensor_metadata_change) {
    TensorImpl::copy_tensor_metadata(
        src_sparse_impl,
        dest_sparse_impl,
        std::move(version_counter),
        allow_tensor_metadata_change);

    // Sparse-specific fields
    dest_sparse_impl->crow_indices_ = src_sparse_impl->compressed_indices();
    dest_sparse_impl->col_indices_ = src_sparse_impl->plain_indices();
    dest_sparse_impl->values_ = src_sparse_impl->values();
    dest_sparse_impl->layout_ = src_sparse_impl->layout_impl();
  }
};
} // namespace at
```