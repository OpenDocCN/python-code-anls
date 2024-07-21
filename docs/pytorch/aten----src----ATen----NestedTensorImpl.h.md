# `.\pytorch\aten\src\ATen\NestedTensorImpl.h`

```py
#pragma once
// 预处理指令，确保头文件只被编译一次

#include <ATen/MemoryOverlap.h>
// 包含 ATen 库的内存重叠检测功能头文件

#include <ATen/Tensor.h>
// 包含 ATen 库的张量（Tensor）实现头文件

#include <c10/core/DispatchKey.h>
// 包含 c10 核心库的调度键（DispatchKey）头文件

#include <c10/core/DispatchKeySet.h>
// 包含 c10 核心库的调度键集合（DispatchKeySet）头文件

#include <c10/core/MemoryFormat.h>
// 包含 c10 核心库的内存格式（MemoryFormat）头文件

#include <c10/core/TensorImpl.h>
// 包含 c10 核心库的张量实现（TensorImpl）头文件

#include <c10/util/ArrayRef.h>
// 包含 c10 实用工具库的数组引用（ArrayRef）头文件

#include <c10/util/Exception.h>
// 包含 c10 实用工具库的异常处理（Exception）头文件

#include <c10/util/Metaprogramming.h>
// 包含 c10 实用工具库的元编程（Metaprogramming）头文件

#include <c10/util/irange.h>
// 包含 c10 实用工具库的迭代范围（irange）头文件

namespace at::native {
// 定义 at::native 命名空间

struct NestedTensorImpl;
// 声明 NestedTensorImpl 结构体

inline bool nested_tensor_impl_is_contiguous(const NestedTensorImpl* nt);
// 声明内联函数，用于检查 NestedTensorImpl 是否是连续的

int64_t get_numel_from_nested_size_tensor(const at::Tensor& tensor);
// 声明函数，从嵌套尺寸张量中获取元素数量

at::Tensor construct_nested_strides(const at::Tensor& nested_size);
// 声明函数，构建嵌套张量的步幅张量

at::Tensor construct_offsets(const at::Tensor& nested_size);
// 声明函数，构建嵌套张量的偏移张量

struct TORCH_API NestedTensorImpl : public c10::TensorImpl {
  // 定义 NestedTensorImpl 结构体，继承自 c10::TensorImpl

  explicit NestedTensorImpl(
      Storage storage,
      c10::DispatchKeySet key_set,
      const caffe2::TypeMeta data_type,
      at::Tensor nested_sizes,
      at::Tensor nested_strides,
      at::Tensor storage_offsets);
  // 显式构造函数声明，初始化 NestedTensorImpl 实例

  explicit NestedTensorImpl(
      const at::Tensor& buffer,
      at::Tensor nested_sizes,
      at::Tensor nested_strides,
      at::Tensor storage_offsets);
  // 显式构造函数声明，从缓冲区和相关张量初始化 NestedTensorImpl 实例

  explicit NestedTensorImpl(
      const at::Tensor& buffer,
      const at::Tensor& nested_sizes);
  // 显式构造函数声明，从缓冲区和嵌套尺寸张量初始化 NestedTensorImpl 实例

  explicit NestedTensorImpl(
      c10::TensorImpl::ImplType impl_type,
      const at::Tensor& base_tensor,
      at::Tensor nested_sizes,
      at::Tensor nested_strides,
      at::Tensor storage_offsets);
  // 显式构造函数声明，从基础张量和相关张量初始化 NestedTensorImpl 实例

  // TODO: don't expose private implementation details like this; in
  // particular, resizing this tensor will mess up our dim() and
  // callers cannot fix it.
  const Tensor& get_nested_sizes() const {
    return nested_sizes_;
  }
  // 返回嵌套尺寸张量的常量引用，暴露了私有实现细节

  // TODO: don't expose private implementation details like this
  const Tensor& get_nested_strides() const {
    return nested_strides_;
  }
  // 返回嵌套步幅张量的常量引用，暴露了私有实现细节

  const Tensor& get_storage_offsets() const {
    return storage_offsets_;
  }
  // 返回存储偏移张量的常量引用

  // Returns nullopt if the ith dimension is irregular. The ith dimension
  // of a NestedTensor is regular if the unbound tensors match in
  // size at the (i-1)th dimension.
  std::optional<int64_t> opt_size(int64_t d) const;
  // 返回一个 optional<int64_t>，如果第 d 维度不规则，则返回 nullopt；
  // NestedTensor 的第 d 维度在未绑定的张量中在 (i-1) 维度上尺寸匹配时为规则的。

  int64_t size(int64_t d) const {
    std::optional<int64_t> optional_size = this->opt_size(d);
    // 获取第 d 维度的尺寸信息

    TORCH_CHECK(
        optional_size.has_value(),
        "Given dimension ",
        d,
        " is irregular and does not have a size.");
    // 检查 optional_size 是否有值，否则抛出异常

    return *optional_size;
    // 返回第 d 维度的尺寸
  }

  /**
   * Return a view of the nested tensor as a 1 dimensional contiguous tensor.
   *
   * The buffer tensor created by this function shares the same storage_impl as
   * the original nested tensor, and therefore can be seen as a view.
   *
   * @return A newly constructed view tensor
   */
  at::Tensor get_buffer() const {
    TORCH_CHECK(
        nested_tensor_impl_is_contiguous(this),
        "NestedTensor must be contiguous to get buffer.");
    // 检查 NestedTensor 是否是连续的，否则抛出异常

    // 返回一个新构造的视图张量
    // 调用 get_unsafe_storage_as_tensor() 函数并返回结果
    return get_unsafe_storage_as_tensor();
  }
  /**
   * 如果可能，请使用 get_buffer() 替代本函数。此函数直接将存储返回为张量，
   * 一般情况下不安全使用。如果使用本函数，调用者必须确保考虑 nested_sizes、
   * nested_strides 和 storage_offsets。
   *
   * @return 新构造的视图张量
   */
  at::Tensor get_unsafe_storage_as_tensor() const {
    // 生成 buffer 的键集合
    auto buffer_key_set_ = generate_buffer_key_set();
    // 获取 buffer 的大小
    const auto buffer_size = get_buffer_size();
    // 使用 storage_、buffer_key_set_ 和 data_type_ 构造一个 TensorImpl 对象
    auto buffer_tensor_impl = c10::make_intrusive<TensorImpl>(
        c10::TensorImpl::VIEW, Storage(storage_), buffer_key_set_, data_type_);
    // 设置 TensorImpl 对象的大小为连续的数组
    buffer_tensor_impl->set_sizes_contiguous(
        c10::makeArrayRef(static_cast<int64_t>(buffer_size)));
    // 返回一个 Tensor 对象，其持有 buffer_tensor_impl
    return Tensor(buffer_tensor_impl);
  }

  // 返回存储的总字节数除以数据类型的字节大小，作为缓冲区大小
  size_t get_buffer_size() const {
    return storage_.nbytes() / data_type_.itemsize();
  }

 protected:
  const char* tensorimpl_type_name() const override;

  // TODO: numel_custom 和 is_contiguous_custom 可以被实际实现覆盖
  // 使用真实实现覆盖 numel_custom 和 is_contiguous_custom 可能是有利可图的
  int64_t numel_custom() const override;
  c10::SymInt sym_numel_custom() const override;
  bool is_contiguous_custom(MemoryFormat) const override;
  int64_t size_custom(int64_t d) const override {
    return this->size(d);
  }
  c10::SymInt sym_size_custom(int64_t d) const override {
    return c10::SymInt{this->size(d)};
  }
  IntArrayRef sizes_custom() const override;
  c10::SymIntArrayRef sym_sizes_custom() const override;
  IntArrayRef strides_custom() const override;
  c10::SymIntArrayRef sym_strides_custom() const override;

  // 实现真实功能的一个函数
  int64_t dim_custom() const override;

  // 创建浅拷贝并分离的一个函数，使用版本计数器和允许张量元数据更改
  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
      const c10::VariableVersion& version_counter,
      bool allow_tensor_metadata_change) const override;

  // 创建浅拷贝并分离的一个函数，使用版本计数器和允许张量元数据更改
  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
      c10::VariableVersion&& version_counter,
      bool allow_tensor_metadata_change) const override;

  // 从另一个 TensorImpl 对象浅复制
  void shallow_copy_from(const c10::intrusive_ptr<TensorImpl>& impl) override {
    copy_tensor_metadata(
        /*src_impl=*/impl.get(),
        /*dest_impl=*/this,
        /*version_counter=*/version_counter(),
        /*allow_tensor_metadata_change=*/allow_tensor_metadata_change());
  }

 private:
  // Must be called after any changes to our dim() to sync the state
  // to TensorImpl.
  void refresh_dim();

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const at::Tensor nested_sizes_, nested_strides_;
  // The starting positions of the underlying tensors in contiguous buffer
  // i.e. the buffer memory offsets to get the underlying tensors
  // The reason to keep this metadata is that, without strong enough constraint
  // it cannot be derived from `nested_sizes_`
  // and `nested_strides_`:
  // 1. when buffer has blanks, e.g. [tensor1, blank, tensor2]
  //    this can happen e.g. after slicing a nested tensor
  // 2. when multiple tensors share a same memory
  // 3. when the nesting ordering is changed, e.g. [tensor1, tensor3, tensor2]
  // Some strong enough constraints are:
  // 1. every underlying tensor is contiguous in memory
  //    && nesting in ascending order
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const at::Tensor storage_offsets_;
  // NOTE: -1 here means the size is missing
  // Optional to allow it to be computed lazily from nested.
  // TODO: maybe we can remove this metadata since
  //       we can compute it from `nested_sizes_`
  mutable std::optional<std::vector<int64_t>> opt_sizes_;

  template <typename VariableVersion>
  // Shallow copy and detach core implementation for tensors
  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach_core(
      VariableVersion&& version_counter,
      bool allow_tensor_metadata_change) const;

  /**
   * Generates a non-nested key_set from a nested tensor.
   *
   * For many nested tensor kernel implementations a buffer tensor
   * is generated and redispatched to a non-nested kernel this function
   * generates the key set used by that buffer tensor
   *
   * @return Appropriate key set for non-nested tensor
   */
  inline c10::DispatchKeySet generate_buffer_key_set() const {
    // Obtain the initial key set including all current keys
    auto buffer_key_set = this->key_set();
    // Check if Autograd dispatch key is present in the key set
    const bool Autograd = buffer_key_set.has_any(c10::autograd_dispatch_keyset);
    // Remove nested tensor specific keys from the key set
    buffer_key_set = buffer_key_set -
        c10::DispatchKeySet{
            c10::DispatchKey::NestedTensor,
            c10::DispatchKey::AutogradNestedTensor};

    // Add dense tensor specific keys to the key set
    buffer_key_set =
        buffer_key_set | c10::DispatchKeySet{c10::DispatchKey::Dense};
    // Conditionally add Autograd key back to the key set based on its previous presence
    buffer_key_set = Autograd
        ? c10::DispatchKeySet{c10::DispatchKey::Autograd} | buffer_key_set
        : buffer_key_set;

    // Return the final adjusted key set for the buffer tensor
    return buffer_key_set;
  }
};

// 返回给定张量的 NestedTensorImpl 实例，若张量不是嵌套张量则返回空指针
inline NestedTensorImpl* get_nested_tensor_impl_or_null(
    const at::Tensor& tensor) {
  // 检查张量是否为嵌套张量，是则返回其对应的 NestedTensorImpl 实例
  if (tensor.is_nested()) {
    return static_cast<NestedTensorImpl*>(tensor.unsafeGetTensorImpl());
  }
  // 否则返回空指针
  return nullptr;
}

// 返回给定张量的 NestedTensorImpl 实例，若张量不是嵌套张量则抛出异常
inline NestedTensorImpl* get_nested_tensor_impl(const at::Tensor& tensor) {
  // 断言张量为嵌套张量，否则抛出异常提示信息
  TORCH_CHECK(
      tensor.is_nested(), "get_nested_tensor_impl requires a NestedTensor.");
  // 返回张量对应的 NestedTensorImpl 实例
  return static_cast<NestedTensorImpl*>(tensor.unsafeGetTensorImpl());
}

// 检查给定的 NestedTensorImpl 实例是否是连续的
inline bool nested_tensor_impl_is_contiguous(const NestedTensorImpl* nt) {
  int64_t ntensors = nt->size(0);
  if (ntensors == 0) {
    return true;
  }
  // 获取嵌套大小和步长张量
  const Tensor &sizemat = nt->get_nested_sizes(),
               &stridemat = nt->get_nested_strides();
  // 获取存储偏移指针
  const int64_t* offsets_ptr =
      nt->get_storage_offsets().const_data_ptr<int64_t>();
  int64_t orig_dim = sizemat.size(1);

  // 根据嵌套的原始维度进行不同的连续性检查
  if (orig_dim == 0) {
    // 如果是嵌套标量，检查每个标量是否连续
    for (int64_t i = 0; i < ntensors; i++) {
      if (offsets_ptr[i] != i) {
        return false;
      }
    }
  } else {
    // 如果是嵌套张量，检查每个张量的连续性
    const int64_t *sizemat_ptr = sizemat.const_data_ptr<int64_t>(),
                  *stridemat_ptr = stridemat.const_data_ptr<int64_t>();
    for (int64_t i = 0; i < ntensors; i++) {
      // 检查最后一个维度的步长是否为1
      if (stridemat_ptr[orig_dim - 1] != 1) {
        return false;
      }
      int64_t product = sizemat_ptr[orig_dim - 1];
      // 检查每个维度的步长是否正确
      for (int64_t j = orig_dim - 2; j >= 0; j--) {
        if (stridemat_ptr[j] != product) {
          return false;
        }
        product *= sizemat_ptr[j];
      }
      sizemat_ptr += orig_dim;
      stridemat_ptr += orig_dim;
    }
    // 检查存储偏移是否正确
    if (offsets_ptr[0] != 0) {
      return false;
    }
    sizemat_ptr = sizemat.const_data_ptr<int64_t>();
    stridemat_ptr = stridemat.const_data_ptr<int64_t>();
    for (int64_t i = 1; i < ntensors; i++) {
      // 检查存储偏移是否按照预期增加
      if (offsets_ptr[i] !=
          offsets_ptr[i - 1] + *sizemat_ptr * *stridemat_ptr) {
        return false;
      }
      sizemat_ptr += orig_dim;
      stridemat_ptr += orig_dim;
    }
  }
  // 若所有检查通过，则认为是连续的
  return true;
}

// 返回给定张量的嵌套大小张量的引用
inline const at::Tensor& get_nested_sizes(const at::Tensor& tensor) {
  return get_nested_tensor_impl(tensor)->get_nested_sizes();
}

} // namespace at::native
```