# `.\pytorch\aten\src\ATen\EmptyTensor.h`

```py
#pragma once
#include <ATen/core/TensorBase.h>

namespace at::detail {

// 检查给定大小数组中是否有负数元素，若有则抛出错误
inline void check_size_nonnegative(ArrayRef<int64_t> size) {
  // 遍历大小数组中的每个元素
  for (const auto& x : size) {
    // 使用 TORCH_CHECK 检查是否有负数大小，报错信息包含具体维度信息
    TORCH_CHECK(
        x >= 0,
        "Trying to create tensor with negative dimension ",
        x,
        ": ",
        size);
  }
}

// 检查给定符号整数大小数组中是否有负数元素，若有则抛出错误
inline void check_size_nonnegative(ArrayRef<c10::SymInt> size) {
  // 遍历大小数组中的每个符号整数
  for (const auto& x : size) {
    // 使用 TORCH_CHECK 调用 SymInt 的 expect_size 方法，报错信息包含具体维度信息
    TORCH_CHECK(
        x.expect_size(__FILE__, __LINE__),
        "Trying to create tensor with negative dimension ",
        x,
        ": ",
        size);
  }
}

// 计算连续存储情况下的存储字节数
TORCH_API size_t computeStorageNbytesContiguous(
    IntArrayRef sizes,
    size_t itemsize,
    size_t storage_offset = 0);

// 计算连续存储情况下的符号整数存储字节数
TORCH_API SymInt computeStorageNbytesContiguous(
    SymIntArrayRef sizes,
    const SymInt& itemsize,
    const SymInt& storage_offset = 0);

// 计算一般情况下的存储字节数
TORCH_API size_t computeStorageNbytes(
    IntArrayRef sizes,
    IntArrayRef strides,
    size_t itemsize,
    size_t storage_offset = 0);

// 计算一般情况下的符号整数存储字节数
TORCH_API SymInt computeStorageNbytes(
    SymIntArrayRef sizes,
    SymIntArrayRef strides,
    const SymInt& itemsize,
    const SymInt& storage_offset = 0);

// 创建一般情况下的空张量
TORCH_API TensorBase empty_generic(
    IntArrayRef size,
    c10::Allocator* allocator,
    c10::DispatchKeySet ks,
    ScalarType scalar_type,
    std::optional<c10::MemoryFormat> memory_format_opt);

// 创建符号整数情况下的空张量
TORCH_API TensorBase empty_generic_symint(
    SymIntArrayRef size,
    c10::Allocator* allocator,
    c10::DispatchKeySet ks,
    ScalarType scalar_type,
    std::optional<c10::MemoryFormat> memory_format_opt);

// 创建带步长的一般情况下的空张量
TORCH_API TensorBase empty_strided_generic(
    IntArrayRef size,
    IntArrayRef stride,
    c10::Allocator* allocator,
    c10::DispatchKeySet ks,
    ScalarType scalar_type);

// 创建带步长的符号整数情况下的空张量
TORCH_API TensorBase empty_strided_symint_generic(
    SymIntArrayRef size,
    SymIntArrayRef stride,
    c10::Allocator* allocator,
    c10::DispatchKeySet ks,
    ScalarType scalar_type);

// 在 CPU 上创建空张量
TORCH_API TensorBase empty_cpu(
    IntArrayRef size,
    ScalarType dtype,
    bool pin_memory = false,
    std::optional<c10::MemoryFormat> memory_format_opt = c10::nullopt);

// 在 CPU 上创建空张量，支持更多选项
TORCH_API TensorBase empty_cpu(
    IntArrayRef size,
    std::optional<ScalarType> dtype_opt,
    std::optional<Layout> layout_opt,
    std::optional<Device> device_opt,
    std::optional<bool> pin_memory_opt,
    std::optional<c10::MemoryFormat> memory_format_opt);

// 在 CPU 上创建空张量，使用 TensorOptions 选项
TORCH_API TensorBase empty_cpu(IntArrayRef size, const TensorOptions& options);

// 在 CPU 上创建带步长的空张量
TORCH_API TensorBase empty_strided_cpu(
    IntArrayRef size,
    IntArrayRef stride,
    ScalarType dtype,
    bool pin_memory = false);

// 在 CPU 上创建带步长的空张量，支持更多选项
TORCH_API TensorBase empty_strided_cpu(
    IntArrayRef size,
    IntArrayRef stride,
    std::optional<ScalarType> dtype_opt,
    std::optional<Layout> layout_opt,
    std::optional<Device> device_opt,
    std::optional<bool> pin_memory_opt);

// 在 CPU 上创建带步长的空张量，使用 TensorOptions 选项
TORCH_API TensorBase empty_strided_cpu(
    IntArrayRef size,
    IntArrayRef stride,
    const TensorOptions& options);

// 创建元数据空张量（未完整）
TORCH_API TensorBase empty_meta(
    IntArrayRef size,
    ScalarType dtype,
    std::optional<c10::MemoryFormat> memory_format_opt = c10::nullopt);


# 定义函数参数：数组引用 `size`，数据类型 `dtype`，可选的内存格式 `memory_format_opt` 默认为空
// 创建一个空的 TensorBase 对象，没有初始化数据，用于指定大小的张量
TORCH_API TensorBase empty_meta(
    IntArrayRef size,                                           // 张量的维度尺寸
    std::optional<ScalarType> dtype_opt,                        // 可选参数：张量的数据类型
    std::optional<Layout> layout_opt,                           // 可选参数：张量的布局方式
    std::optional<Device> device_opt,                           // 可选参数：张量的设备类型
    std::optional<bool> pin_memory_opt,                         // 可选参数：是否使用固定内存
    std::optional<c10::MemoryFormat> memory_format_opt);        // 可选参数：内存格式

// 创建一个空的 TensorBase 对象，没有初始化数据，用于指定大小和符号整数的张量
TORCH_API TensorBase empty_symint_meta(
    SymIntArrayRef size,                                        // 符号整数的张量维度尺寸
    std::optional<ScalarType> dtype_opt,                        // 可选参数：张量的数据类型
    std::optional<Layout> layout_opt,                           // 可选参数：张量的布局方式
    std::optional<Device> device_opt,                           // 可选参数：张量的设备类型
    std::optional<bool> pin_memory_opt,                         // 可选参数：是否使用固定内存
    std::optional<c10::MemoryFormat> memory_format_opt);        // 可选参数：内存格式

// 创建一个空的 TensorBase 对象，没有初始化数据，通过给定的 TensorOptions 指定张量的其他属性
TORCH_API TensorBase empty_meta(IntArrayRef size, const TensorOptions& options);  // 张量的维度尺寸和其他选项

// 创建一个空的 TensorBase 对象，按照给定的步长创建分布张量
TORCH_API TensorBase empty_strided_meta(
    IntArrayRef size,                                           // 张量的维度尺寸
    IntArrayRef stride,                                         // 张量的步长尺寸
    ScalarType dtype);                                          // 张量的数据类型

// 创建一个空的 TensorBase 对象，按照给定的步长创建分布张量，可以选择数据类型、布局、设备和固定内存选项
TORCH_API TensorBase empty_strided_meta(
    IntArrayRef size,                                           // 张量的维度尺寸
    IntArrayRef stride,                                         // 张量的步长尺寸
    std::optional<ScalarType> dtype_opt,                        // 可选参数：张量的数据类型
    std::optional<Layout> layout_opt,                           // 可选参数：张量的布局方式
    std::optional<Device> device_opt,                           // 可选参数：张量的设备类型
    std::optional<bool> pin_memory_opt);                        // 可选参数：是否使用固定内存

// 创建一个空的 TensorBase 对象，按照给定的步长创建分布张量，通过给定的 TensorOptions 指定张量的其他属性
TORCH_API TensorBase empty_strided_meta(
    IntArrayRef size,                                           // 张量的维度尺寸
    IntArrayRef stride,                                         // 张量的步长尺寸
    const TensorOptions& options);                              // 张量的其他选项

// 创建一个空的 TensorBase 对象，按照给定的符号整数的大小和步长创建分布张量，指定数据类型
TORCH_API TensorBase empty_strided_symint_meta(
    SymIntArrayRef size,                                        // 符号整数的张量维度尺寸
    SymIntArrayRef stride,                                      // 符号整数的张量步长尺寸
    ScalarType dtype);                                          // 张量的数据类型

// 创建一个空的 TensorBase 对象，按照给定的符号整数的大小和步长创建分布张量，可以选择数据类型、布局、设备和固定内存选项
TORCH_API TensorBase empty_strided_symint_meta(
    SymIntArrayRef size,                                        // 符号整数的张量维度尺寸
    SymIntArrayRef stride,                                      // 符号整数的张量步长尺寸
    std::optional<ScalarType> dtype_opt,                        // 可选参数：张量的数据类型
    std::optional<Layout> layout_opt,                           // 可选参数：张量的布局方式
    std::optional<Device> device_opt,                           // 可选参数：张量的设备类型
    std::optional<bool> pin_memory_opt);                        // 可选参数：是否使用固定内存

// 创建一个空的 TensorBase 对象，按照给定的符号整数的大小和步长创建分布张量，通过给定的 TensorOptions 指定张量的其他属性
TORCH_API TensorBase empty_strided_symint_meta(
    SymIntArrayRef size,                                        // 符号整数的张量维度尺寸
    SymIntArrayRef stride,                                      // 符号整数的张量步长尺寸
    const TensorOptions& options);                              // 张量的其他选项

} // namespace at::detail
```