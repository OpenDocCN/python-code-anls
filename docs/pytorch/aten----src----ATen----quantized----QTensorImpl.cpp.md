# `.\pytorch\aten\src\ATen\quantized\QTensorImpl.cpp`

```py
// 包含 ATen 库中的 QTensorImpl 类的头文件
#include <ATen/quantized/QTensorImpl.h>

// ATen 命名空间
namespace at {

// QTensorImpl 类的构造函数实现，接受移动语义的 storage 对象、调度键集 key_set、数据类型 data_type 和量化器 quantizer
QTensorImpl::QTensorImpl(
    Storage&& storage,
    DispatchKeySet key_set,
    const caffe2::TypeMeta data_type,
    QuantizerPtr quantizer)
    : TensorImpl(std::move(storage), std::move(key_set), data_type),  // 调用基类 TensorImpl 的构造函数
      quantizer_(std::move(quantizer)) {}  // 初始化 quantizer_ 成员变量

// QTensorImpl 类的构造函数实现，接受 ImplType 类型的 type、移动语义的 storage 对象、调度键集 key_set、数据类型 data_type 和量化器 quantizer
QTensorImpl::QTensorImpl(
    ImplType type,
    Storage&& storage,
    DispatchKeySet key_set,
    const caffe2::TypeMeta data_type,
    QuantizerPtr quantizer)
    : TensorImpl(type, std::move(storage), std::move(key_set), data_type),  // 调用基类 TensorImpl 的构造函数
      quantizer_(std::move(quantizer)) {}  // 初始化 quantizer_ 成员变量

// 返回 QTensorImpl 类型名称的常量字符指针
const char* QTensorImpl::tensorimpl_type_name() const {
  return "QTensorImpl";
}

} // namespace at
```