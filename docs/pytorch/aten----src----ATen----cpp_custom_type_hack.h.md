# `.\pytorch\aten\src\ATen\cpp_custom_type_hack.h`

```
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP```
// This section of commented-out code serves as a strong warning and reminder
// to anyone reading it. It explicitly states that the following code was a
// temporary solution, likely for embedding C++ structures into PyTorch Tensors,
// but it's unsafe and not supported. It emphasizes that using this code will
// lead to issues, and it has been replaced by more appropriate and safer methods,
// specifically referring to custom classes detailed in the provided link.

// The message instructs developers not to add more calls to the functionalities
// defined in this file and urges the use of PyTorch's recommended practices
// for handling custom classes instead.
// 包含 ATen 库的追踪模式和张量定义
#include <ATen/TracerMode.h>
#include <ATen/core/Tensor.h>

// 根据是否定义 AT_PER_OPERATOR_HEADERS 选择正确的头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#endif

// 定义自定义类型 hack 的命名空间
namespace at::cpp_custom_type_hack {

// 对模板函数进行声明，用于检查 Tensor 是否为特定类型 T
template <typename T>
[[deprecated(
    "Use custom classes instead: "
    "https://pytorch.org/tutorials/advanced/torch_script_custom_classes.html")]] bool
isa(const Tensor& packed) {
  // 返回条件：Tensor 的数据类型为 kByte，并且存储的数据删除函数为特定类型 T 对应的删除函数
  return (packed.scalar_type() == kByte) &&
      (packed.storage().data_ptr().get_deleter() ==
       caffe2::TypeMeta::Make<T>().deleteFn());
}

// 对模板函数进行声明，用于将 Tensor 强制转换为特定类型 T 的引用
template <typename T>
[[deprecated(
    "Use custom classes instead: "
    "https://pytorch.org/tutorials/advanced/torch_script_custom_classes.html")]] T&
cast(const Tensor& packed) {
  // 检查条件：Tensor 的数据类型为 kByte，且存储的数据删除函数为特定类型 T 对应的删除函数
  TORCH_CHECK(
      packed.scalar_type() == kByte, "Expected temporary cpp type wrapper");
  TORCH_CHECK(
      packed.storage().data_ptr().get_deleter() ==
          caffe2::TypeMeta::Make<T>().deleteFn(),
      "Expected temporary cpp type wrapper of type ",
      caffe2::TypeMeta::TypeName<T>());
  // 返回通过 reinterpret_cast 转换后的 T 类型指针
  return *reinterpret_cast<T*>(packed.storage().data_ptr().get());
}

// 对模板函数进行声明，用于创建包含指定类型对象的 Tensor
template <typename T>
[[deprecated(
    "Use custom classes instead: "
    "https://pytorch.org/tutorials/advanced/torch_script_custom_classes.html")]] Tensor
create(std::unique_ptr<T> ptr, TensorOptions options) {
  // 禁用追踪器分发以确保不进行追踪
  at::AutoDispatchBelowADInplaceOrView guard; // TODO: remove
  at::tracer::impl::NoTracerDispatchMode tracer_guard;

  // 释放唯一指针，并创建包含其指针和删除函数的 DataPtr
  void* raw_ptr = ptr.release();
  at::DataPtr at_ptr(
      raw_ptr, raw_ptr, caffe2::TypeMeta::Make<T>().deleteFn(), at::kCPU);

  // 创建并返回一个 Tensor，其大小为 sizeof(T)，数据类型为 kByte
  auto retval = at::empty({sizeof(T)}, options.device(kCPU).dtype(at::kByte));
  retval.storage().set_data_ptr_noswap(std::move(at_ptr));
  return retval;
}

} // namespace at::cpp_custom_type_hack
```