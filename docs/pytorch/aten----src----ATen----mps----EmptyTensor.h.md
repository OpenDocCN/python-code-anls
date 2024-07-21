# `.\pytorch\aten\src\ATen\mps\EmptyTensor.h`

```py
//  Copyright © 2022 Apple Inc.
// 包含 ATen 库中的 TensorBase 类的头文件
#pragma once
#include <ATen/core/TensorBase.h>

// ATen 命名空间中的 detail 命名空间
namespace at::detail {

// 创建一个没有初始化的 TensorBase 对象，用指定的 size 和可选的 dtype、layout、device、pin_memory_opt、memory_format_opt 参数
C10_EXPORT TensorBase empty_mps(
    IntArrayRef size,
    std::optional<ScalarType> dtype_opt,
    std::optional<Layout> layout_opt,
    std::optional<Device> device_opt,
    std::optional<bool> pin_memory_opt,
    std::optional<c10::MemoryFormat> memory_format_opt);

// 创建一个没有初始化的 TensorBase 对象，用指定的 size 和 TensorOptions 对象
C10_EXPORT TensorBase empty_mps(
    IntArrayRef size, const TensorOptions &options);

// 创建一个没有初始化的 TensorBase 对象，用指定的 size、stride 和 dtype，以及可选的 device_opt 参数
C10_EXPORT TensorBase empty_strided_mps(
    IntArrayRef size,
    IntArrayRef stride,
    ScalarType dtype,
    std::optional<Device> device_opt);

// 创建一个没有初始化的 TensorBase 对象，用指定的 size、stride 和 TensorOptions 对象
C10_EXPORT TensorBase empty_strided_mps(
    IntArrayRef size,
    IntArrayRef stride,
    const TensorOptions &options);

} // namespace at::detail
```