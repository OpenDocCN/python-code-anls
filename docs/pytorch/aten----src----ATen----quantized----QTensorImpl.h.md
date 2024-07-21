# `.\pytorch\aten\src\ATen\quantized\QTensorImpl.h`

```
#pragma once
// 预处理命令，确保头文件只被包含一次

#include <ATen/quantized/Quantizer.h>
// 引入 ATen 库中的量化器头文件
#include <c10/core/TensorImpl.h>
// 引入 c10 核心库中的 TensorImpl 头文件
#include <c10/util/Exception.h>
// 引入 c10 实用工具中的异常处理头文件

namespace at {
// 进入 at 命名空间

/**
 * QTensorImpl is a TensorImpl for Quantized Tensors, it stores Quantizer which
 * specifies the quantization scheme and parameters, for more information please
 * see ATen/quantized/Quantizer.h
 *
 * We'll use QTensor in code or documentation to refer to a Tensor with QTensorImpl.
 */
// QTensorImpl 结构体继承自 c10::TensorImpl，用于量化张量的实现
struct TORCH_API QTensorImpl : public c10::TensorImpl {
 public:
  QTensorImpl(
      Storage&& storage,
      DispatchKeySet key_set,
      const caffe2::TypeMeta data_type,
      QuantizerPtr quantizer);
  // 构造函数，接受存储、调度键集、数据类型元信息和量化器指针

  // See Note [Enum ImplType]
  QTensorImpl(
      ImplType type,
      Storage&& storage,
      DispatchKeySet key_set,
      const caffe2::TypeMeta data_type,
      QuantizerPtr quantizer);
  // 构造函数，接受实现类型、存储、调度键集、数据类型元信息和量化器指针

  // TODO: Expose in PyTorch Frontend
  QuantizerPtr quantizer() {
    return quantizer_;
  }
  // 获取量化器指针的方法

  void set_quantizer_(QuantizerPtr quantizer) {
    quantizer_ = quantizer;
  }
  // 设置量化器指针的方法

  /**
   * Return a TensorImpl that is a shallow-copy of this TensorImpl.
   *
   * For usage of `version_counter` and `allow_tensor_metadata_change`,
   * see NOTE [ TensorImpl Shallow-Copying ].
   */
  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
      const c10::VariableVersion& version_counter,
      bool allow_tensor_metadata_change) const override {
    auto impl = c10::make_intrusive<QTensorImpl>(
        Storage(storage()), key_set(), data_type_, quantizer_);
    // 创建 QTensorImpl 的浅拷贝，包括存储、调度键集、数据类型和量化器

    copy_tensor_metadata(
      /*src_impl=*/this,
      /*dest_impl=*/impl.get(),
      /*version_counter=*/version_counter,
      /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
    // 复制张量元数据到新创建的 QTensorImpl 实例中

    impl->refresh_numel();
    // 刷新张量元素数量

    impl->refresh_contiguous();
    // 刷新张量的连续性

    return impl;
    // 返回新创建的 QTensorImpl 实例
  }

  /**
   * Return a TensorImpl that is a shallow-copy of this TensorImpl.
   *
   * For usage of `version_counter` and `allow_tensor_metadata_change`,
   * see NOTE [ TensorImpl Shallow-Copying ].
   */
  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
      c10::VariableVersion&& version_counter,
      bool allow_tensor_metadata_change) const override {
    auto impl = c10::make_intrusive<QTensorImpl>(
        Storage(storage()), key_set(), data_type_, quantizer_);
    // 创建 QTensorImpl 的浅拷贝，包括存储、调度键集、数据类型和量化器

    copy_tensor_metadata(
      /*src_impl=*/this,
      /*dest_impl=*/impl.get(),
      /*version_counter=*/std::move(version_counter),
      /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
    // 复制张量元数据到新创建的 QTensorImpl 实例中

    impl->refresh_numel();
    // 刷新张量元素数量

    impl->refresh_contiguous();
    // 刷新张量的连续性

    return impl;
    // 返回新创建的 QTensorImpl 实例
  }

  /**
   * Shallow-copies data from another TensorImpl into this TensorImpl.
   *
   * For why this function doesn't check this TensorImpl's `allow_tensor_metadata_change_`,
   * see NOTE [ TensorImpl Shallow-Copying ].
   */
  void shallow_copy_from(const c10::intrusive_ptr<TensorImpl>& impl) override {
    AT_ASSERT(has_compatible_shallow_copy_type(impl->key_set()));
    // 断言当前实例与传入实例具有兼容的浅拷贝类型

    auto q_impl = static_cast<const QTensorImpl*>(impl.get());
    // 将传入实例转换为 QTensorImpl 指针
    copy_tensor_metadata(
      /*src_impl=*/q_impl,
      /*dest_impl=*/this,
      /*version_counter=*/version_counter(),
      /*allow_tensor_metadata_change=*/allow_tensor_metadata_change());
    refresh_numel();
    refresh_contiguous();
  }

 private:
  QuantizerPtr quantizer_;

  const char* tensorimpl_type_name() const override;

  /**
   * Copy the tensor metadata fields (e.g. sizes / strides / storage pointer / storage_offset)
   * from one TensorImpl to another TensorImpl.
   *
   * For usage of `version_counter` and `allow_tensor_metadata_change`, see NOTE [ TensorImpl Shallow-Copying ].
   */
  static void copy_tensor_metadata(
      const QTensorImpl* src_q_impl,
      QTensorImpl* dest_q_impl,
      const c10::VariableVersion& version_counter,
      bool allow_tensor_metadata_change) {
    // 调用基类的静态方法，复制源张量实现到目标张量实现的元数据
    TensorImpl::copy_tensor_metadata(src_q_impl, dest_q_impl, version_counter, allow_tensor_metadata_change);

    // 复制特定于 OpaqueTensorImpl 的字段
    dest_q_impl->quantizer_ = src_q_impl->quantizer_;
  }
};

} // namespace at
```