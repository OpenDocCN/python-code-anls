# `.\pytorch\aten\src\ATen\cuda\EmptyTensor.h`

```
#pragma once
// 预处理指令，确保头文件只被编译一次

#include <ATen/core/TensorBase.h>
// 包含 ATen 库中的 TensorBase 头文件

namespace at::detail {
// 进入 at::detail 命名空间

TORCH_CUDA_CPP_API TensorBase empty_cuda(
    IntArrayRef size,
    ScalarType dtype,
    std::optional<Device> device_opt,
    std::optional<c10::MemoryFormat> memory_format_opt);
// 声明一个名为 empty_cuda 的函数，返回一个 TensorBase 对象，
// 接受参数：size（整数数组引用）、dtype（标量类型）、
// device_opt（可选的设备）、memory_format_opt（可选的内存格式）

TORCH_CUDA_CPP_API TensorBase empty_cuda(
    IntArrayRef size,
    std::optional<ScalarType> dtype_opt,
    std::optional<Layout> layout_opt,
    std::optional<Device> device_opt,
    std::optional<bool> pin_memory_opt,
    std::optional<c10::MemoryFormat> memory_format_opt);
// 声明重载的 empty_cuda 函数，返回一个 TensorBase 对象，
// 接受参数：size（整数数组引用）、dtype_opt（可选的标量类型）、
// layout_opt（可选的布局）、device_opt（可选的设备）、
// pin_memory_opt（可选的是否使用钉住内存）、memory_format_opt（可选的内存格式）

TORCH_CUDA_CPP_API TensorBase empty_cuda(
    IntArrayRef size,
    const TensorOptions &options);
// 声明重载的 empty_cuda 函数，返回一个 TensorBase 对象，
// 接受参数：size（整数数组引用）、options（TensorOptions 对象的常量引用）

TORCH_CUDA_CPP_API TensorBase empty_strided_cuda(
    IntArrayRef size,
    IntArrayRef stride,
    ScalarType dtype,
    std::optional<Device> device_opt);
// 声明 empty_strided_cuda 函数，返回一个 TensorBase 对象，
// 接受参数：size（整数数组引用）、stride（整数数组引用）、
// dtype（标量类型）、device_opt（可选的设备）

TORCH_CUDA_CPP_API TensorBase empty_strided_cuda(
    IntArrayRef size,
    IntArrayRef stride,
    std::optional<ScalarType> dtype_opt,
    std::optional<Layout> layout_opt,
    std::optional<Device> device_opt,
    std::optional<bool> pin_memory_opt);
// 声明重载的 empty_strided_cuda 函数，返回一个 TensorBase 对象，
// 接受参数：size（整数数组引用）、stride（整数数组引用）、
// dtype_opt（可选的标量类型）、layout_opt（可选的布局）、
// device_opt（可选的设备）、pin_memory_opt（可选的是否使用钉住内存）

TORCH_CUDA_CPP_API TensorBase empty_strided_cuda(
    IntArrayRef size,
    IntArrayRef stride,
    const TensorOptions &options);
// 声明重载的 empty_strided_cuda 函数，返回一个 TensorBase 对象，
// 接受参数：size（整数数组引用）、stride（整数数组引用）、
// options（TensorOptions 对象的常量引用）

}  // namespace at::detail
// 结束 at::detail 命名空间
```