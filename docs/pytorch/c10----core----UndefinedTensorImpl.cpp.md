# `.\pytorch\c10\core\UndefinedTensorImpl.cpp`

```
#include <c10/core/UndefinedTensorImpl.h>
#include <c10/util/Exception.h>

namespace c10 {

// UndefinedTensorImpl 类的构造函数，继承自 TensorImpl 类。
// 使用 DispatchKey::Undefined 表示未定义的张量类型，不关联具体的数据类型。
// 参数中的 caffe2::TypeMeta() 表示类型元信息为空，c10::nullopt 表示没有可选值传递。
UndefinedTensorImpl::UndefinedTensorImpl()
    : TensorImpl(DispatchKey::Undefined, caffe2::TypeMeta(), c10::nullopt) {
  // 设置在访问存储时应该抛出异常
  set_storage_access_should_throw();
  // TODO: 访问未定义张量的大小是没有意义的，并且应该报错，但实际上这并没有发生！
  // 使用自定义的尺寸和步长策略，指定为 CustomStrides。
  set_custom_sizes_strides(SizesStridesPolicy::CustomStrides);
}

// 检查未定义张量是否是指定格式下的连续张量
bool UndefinedTensorImpl::is_contiguous_custom(MemoryFormat format) const {
  return is_contiguous_default(format);
}

// 返回未定义张量的步长数组，由于张量未定义，直接抛出错误信息
IntArrayRef UndefinedTensorImpl::strides_custom() const {
  TORCH_CHECK(false, "strides() called on an undefined Tensor");
}

// 返回未定义张量的对称步长数组，由于张量未定义，直接抛出错误信息
SymIntArrayRef UndefinedTensorImpl::sym_strides_custom() const {
  TORCH_CHECK(false, "sym_strides() called on an undefined Tensor");
}

#ifdef DEBUG
// 检查未定义张量是否拥有存储，仅在 DEBUG 模式下有效，否则直接返回 false
bool UndefinedTensorImpl::has_storage() const {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      !storage_, "UndefinedTensorImpl assumes that storage_ is never set");
  return false;
}
#endif

// 设置未定义张量的存储偏移，直接抛出错误信息
void UndefinedTensorImpl::set_storage_offset(int64_t) {
  TORCH_CHECK(false, "set_storage_offset() called on an undefined Tensor");
}

// 返回未定义张量的类型名称，即 "UndefinedTensorImpl"
const char* UndefinedTensorImpl::tensorimpl_type_name() const {
  return "UndefinedTensorImpl";
}

// 静态成员 _singleton 的定义和初始化
UndefinedTensorImpl UndefinedTensorImpl::_singleton;

} // namespace c10
```