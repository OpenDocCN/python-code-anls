# `.\pytorch\aten\src\ATen\OpaqueTensorImpl.h`

```
#pragma once

#include <c10/core/MemoryFormat.h>
#include <c10/core/SymIntArrayRef.h>
#include <c10/core/TensorImpl.h>
#include <c10/util/Exception.h>

namespace at {

// An "Opaque" TensorImpl -- there are no strides and (for now)
// even data() is not supported (thus no pointer arithmetic).

// NOTE: We could allow data() in the future, but would have to ensure pointer
// arithmetic code is properly guarded.
//
// NOTE: This does not support resize_ (and other metadata-changing ops) because
// of `shallow_copy_and_detach`. We would need to define an interface to
// "shallow copy" in order to add support.

template <typename OpaqueHandle>
struct TORCH_API OpaqueTensorImpl : public TensorImpl {
  // public constructor for now...
  OpaqueTensorImpl(
      at::DispatchKeySet key_set,
      const caffe2::TypeMeta data_type,
      c10::Device device,
      OpaqueHandle opaque_handle,
      c10::IntArrayRef sizes,
      bool is_non_overlapping_and_dense = true)
      : TensorImpl(key_set, data_type, device),
        opaque_handle_(std::move(opaque_handle)) {
    set_storage_access_should_throw();  // 设置存储访问应抛出异常
    set_custom_sizes_strides(SizesStridesPolicy::CustomStrides);  // 设置自定义大小和步幅策略
    sizes_and_strides_.set_sizes(sizes);  // 设置张量的尺寸
    refresh_numel();  // 刷新张量元素的数量
    // NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer)
    is_non_overlapping_and_dense_ = is_non_overlapping_and_dense;  // 设置张量是否是非重叠和密集的标志
  }

  // Destructor doesn't call release_resources because it's
  // unnecessary; don't forget to change that if needed!
  void release_resources() override {
    TensorImpl::release_resources();  // 调用基类的资源释放方法
    opaque_handle_ = {};  // 释放不透明句柄
  }

  void set_size(int64_t dim, int64_t new_size) override {
    AT_ERROR("opaque tensors do not have set_size");  // 不支持设置张量大小的错误信息
  }

  void set_stride(int64_t dim, int64_t new_stride) override {
    AT_ERROR("opaque tensors do not have set_stride");  // 不支持设置张量步幅的错误信息
  }

  void set_storage_offset(int64_t storage_offset) override {
    AT_ERROR("opaque tensors do not have set_storage_offset");  // 不支持设置张量存储偏移的错误信息
  }

#ifdef DEBUG
  bool has_storage() const override {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        !storage_, "OpaqueTensorImpl assumes that storage_ is never set");  // 断言确保不透明张量没有存储
    return false;  // 返回假，表示没有存储
  }
#endif

  /**
   * Return a TensorImpl that is a shallow-copy of this TensorImpl.
   *
   * For usage of `version_counter` and `allow_tensor_metadata_change`,
   * see NOTE [ TensorImpl Shallow-Copying ].
   */
  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
      const c10::VariableVersion& version_counter,
      bool allow_tensor_metadata_change) const override {
    auto impl = c10::make_intrusive<OpaqueTensorImpl<OpaqueHandle>>(
        key_set(),
        dtype(),
        device(),
        opaque_handle_,
        sizes_and_strides_.sizes_arrayref());
    copy_tensor_metadata(
        /*src_opaque_impl=*/this,
        /*dest_opaque_impl=*/impl.get(),
        /*version_counter=*/version_counter,
        /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
    impl->refresh_numel();
    // 返回一个浅拷贝的TensorImpl
    return impl;
  }

  OpaqueHandle opaque_handle_;  // 不透明句柄
};

} // namespace at
  return impl;
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
  // 创建一个新的 OpaqueTensorImpl 对象作为浅拷贝
  auto impl = c10::make_intrusive<OpaqueTensorImpl<OpaqueHandle>>(
      key_set(),
      dtype(),
      device(),
      opaque_handle_,
      sizes_and_strides_.sizes_arrayref());
  // 拷贝张量元数据到新对象中
  copy_tensor_metadata(
      /*src_opaque_impl=*/this,
      /*dest_opaque_impl=*/impl.get(),
      /*version_counter=*/std::move(version_counter),
      /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
  // 刷新张量的元素数目
  impl->refresh_numel();
  // 返回新创建的浅拷贝对象
  return impl;
}

/**
 * Shallow-copies data from another TensorImpl into this TensorImpl.
 *
 * For why this function doesn't check this TensorImpl's
 * `allow_tensor_metadata_change_`, see NOTE [ TensorImpl Shallow-Copying ].
 */
void shallow_copy_from(const c10::intrusive_ptr<TensorImpl>& impl) override {
  // 断言当前张量可以进行浅拷贝
  AT_ASSERT(has_compatible_shallow_copy_type(impl->key_set()));
  // 获取源张量的具体实现类型
  auto opaque_impl =
      static_cast<const OpaqueTensorImpl<OpaqueHandle>*>(impl.get());
  // 拷贝张量元数据到当前对象中
  copy_tensor_metadata(
      /*src_impl=*/opaque_impl,
      /*dest_impl=*/this,
      /*version_counter=*/version_counter(),
      /*allow_tensor_metadata_change=*/allow_tensor_metadata_change());
  // 刷新当前张量的元素数目
  refresh_numel();
}

const OpaqueHandle& opaque_handle() const {
  // 返回不可变的透明句柄引用
  return opaque_handle_;
}

OpaqueHandle& unsafe_opaque_handle() {
  // 返回可变的透明句柄引用，不进行安全检查
  return opaque_handle_;
}

protected:
/**
 * Copy the tensor metadata fields (e.g. sizes / strides / storage pointer /
 * storage_offset) from one TensorImpl to another TensorImpl.
 *
 * For usage of `version_counter` and `allow_tensor_metadata_change`, see NOTE
 * [ TensorImpl Shallow-Copying ].
 */
static void copy_tensor_metadata(
    const OpaqueTensorImpl<OpaqueHandle>* src_opaque_impl,
    OpaqueTensorImpl<OpaqueHandle>* dest_opaque_impl,
    const c10::VariableVersion& version_counter,
    bool allow_tensor_metadata_change) {
  // 调用基类的方法复制张量元数据
  TensorImpl::copy_tensor_metadata(
      src_opaque_impl,
      dest_opaque_impl,
      version_counter,
      allow_tensor_metadata_change);

  // 复制特定于 OpaqueTensorImpl 的字段
  dest_opaque_impl->opaque_handle_ = src_opaque_impl->opaque_handle_;
}

static void copy_tensor_metadata(
    const OpaqueTensorImpl<OpaqueHandle>* src_opaque_impl,
    OpaqueTensorImpl<OpaqueHandle>* dest_opaque_impl,
    c10::VariableVersion&& version_counter,
    bool allow_tensor_metadata_change) {
  // 调用基类的方法复制张量元数据
  TensorImpl::copy_tensor_metadata(
      src_opaque_impl,
      dest_opaque_impl,
      std::move(version_counter),
      allow_tensor_metadata_change);

  // 复制特定于 OpaqueTensorImpl 的字段
  dest_opaque_impl->opaque_handle_ = src_opaque_impl->opaque_handle_;
}
    // 将源对象的不透明句柄复制给目标对象的不透明句柄
    dest_opaque_impl->opaque_handle_ = src_opaque_impl->opaque_handle_;
  }

 private:
  // 返回当前对象的张量实现类型名称为 "OpaqueTensorImpl"
  const char* tensorimpl_type_name() const override {
    return "OpaqueTensorImpl";
  }

  // 不透明句柄成员变量
  OpaqueHandle opaque_handle_;
};

} // namespace at
```