# `.\pytorch\aten\src\ATen\native\cuda\TensorShapeCUDA.cpp`

```
// 定义 TORCH_ASSERT_ONLY_METHOD_OPERATORS 宏，用于在 ATen 核心 Tensor.h 头文件中包含所需的头文件
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 包含 ATen 核心 Tensor.h 头文件，该文件包含了与张量操作相关的基本功能
#include <ATen/core/Tensor.h>
// 包含 ATen CUDA 上下文相关的头文件，用于管理 CUDA 设备上的张量操作
#include <ATen/cuda/CUDAContext.h>
// 包含 ATen 的 Resize.h 头文件，该文件定义了张量的大小调整操作
#include <ATen/native/Resize.h>
// 包含 ATen CUDA 版本的 Resize.h 头文件，定义了 CUDA 下的张量大小调整操作
#include <ATen/native/cuda/Resize.h>

// 如果未定义 AT_PER_OPERATOR_HEADERS 宏，则包含 ATen 的 NativeFunctions.h 头文件，该文件包含了一组张量操作的函数定义
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
// 如果定义了 AT_PER_OPERATOR_HEADERS 宏，则包含 ATen 的 set_native.h 头文件，该文件设置了本地（native）函数的实现
#else
#include <ATen/ops/set_native.h>
#endif

// ATen 命名空间下的 native 命名空间，定义了一组与本地（native）张量操作相关的函数
namespace at::native {

// 此函数用于在 CUDA 设备上设置给定张量的存储空间，并返回修改后的张量引用
// 这里的实现需要根据 CPU 和 CUDA 的不同分开处理，因为没有一致的方法来获取用于设备的分配器
Tensor& set_cuda_(Tensor& result) {
  // 获取结果张量的数据类型
  caffe2::TypeMeta dtype = result.dtype();
  // 创建一个新的存储空间对象，使用 CUDA 设备的分配器，并允许自动分配内存
  Storage storage(
      Storage::use_byte_size_t(),
      0,
      at::cuda::getCUDADeviceAllocator(),
      true);
  // 将结果张量设置为使用新的存储空间，偏移为 0，大小为空，步长为空
  result.set_(storage, 0, {0}, {});
  // 断言结果张量的数据类型与之前的数据类型相同
  TORCH_INTERNAL_ASSERT(dtype == result.dtype());
  // 返回设置后的结果张量
  return result;
}

// 在 CUDA 实现中统一？为了避免在 resize_impl_cpu_ 中进行调度，此函数用于设置给定张量的 CUDA 存储空间，并返回修改后的张量引用
Tensor& set_storage_cuda_(Tensor& result, Storage storage, int64_t storage_offset, IntArrayRef size, IntArrayRef stride) {
  // 检查设置存储空间的有效性
  checkSetStorage(result, storage, storage_offset, size, stride);

  // 设置结果张量的存储偏移量
  result.unsafeGetTensorImpl()->set_storage_offset(storage_offset);
  // 如果步长数据不为空，则将其作为可选参数传递；否则传递空值
  at::OptionalIntArrayRef stride_opt = stride.data() != nullptr ?
                                          at::OptionalIntArrayRef(stride) : c10::nullopt;
  // 调用 CUDA 下的 resize 实现函数，修改结果张量的大小和步长
  at::native::resize_impl_cuda_(result.unsafeGetTensorImpl(), size, stride_opt);
  // 返回设置后的结果张量
  return result;
}

} // namespace at::native
```