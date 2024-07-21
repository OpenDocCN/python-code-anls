# `.\pytorch\aten\src\ATen\cuda\EmptyTensor.cpp`

```py
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/cuda/EmptyTensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/EmptyTensor.h>

// 进入 ATen 命名空间中的 detail 命名空间
namespace at::detail {

// 在 CUDA 上创建一个空的 TensorBase 对象
TensorBase empty_cuda(
    IntArrayRef size, // 张量的尺寸
    ScalarType dtype, // 张量的数据类型
    std::optional<Device> device_opt, // 设备的可选参数
    std::optional<c10::MemoryFormat> memory_format_opt) { // 内存格式的可选参数
  // 初始化全局的 CUDA 上下文
  at::globalContext().lazyInitCUDA();
  // 获取设备，如果没有提供则使用默认设备
  const auto device = device_or_default(device_opt);
  // 断言设备是 CUDA 设备
  TORCH_INTERNAL_ASSERT(device.is_cuda());
  // 设备守卫，确保在当前设备上操作
  const DeviceGuard device_guard(device);
  // 获取 CUDA 设备的内存分配器
  auto* allocator = at::cuda::getCUDADeviceAllocator();
  // 定义 CUDA DispatchKey 集合
  constexpr c10::DispatchKeySet cuda_dks(c10::DispatchKey::CUDA);
  // 调用通用的空张量创建函数 empty_generic
  return at::detail::empty_generic(
      size, allocator, cuda_dks, dtype, memory_format_opt);
}

// 在 CUDA 上创建一个空的 TensorBase 对象（重载函数）
TensorBase empty_cuda(
    IntArrayRef size, // 张量的尺寸
    std::optional<ScalarType> dtype_opt, // 数据类型的可选参数
    std::optional<Layout> layout_opt, // 布局的可选参数
    std::optional<Device> device_opt, // 设备的可选参数
    std::optional<bool> pin_memory_opt, // 是否固定内存的可选参数
    std::optional<c10::MemoryFormat> memory_format_opt) { // 内存格式的可选参数
  // 检查是否允许固定内存，仅允许在 CPU 密集型张量中固定内存
  TORCH_CHECK(!pin_memory_opt.has_value() || !*pin_memory_opt, "Only dense CPU tensors can be pinned");
  // 断言在调试模式下布局为 Strided
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(layout_or_default(layout_opt) == Layout::Strided);

  // 获取数据类型，如果没有提供则使用默认数据类型
  const auto dtype = dtype_or_default(dtype_opt);
  // 调用前一个重载的 empty_cuda 函数
  return at::detail::empty_cuda(size, dtype, device_opt, memory_format_opt);
}

// 在 CUDA 上创建一个空的 TensorBase 对象（使用 TensorOptions）
TensorBase empty_cuda(
    IntArrayRef size, // 张量的尺寸
    const TensorOptions &options) { // 张量选项
  // 调用前一个重载的 empty_cuda 函数，将选项转换为相应的参数
  return at::detail::empty_cuda(
      size,
      optTypeMetaToScalarType(options.dtype_opt()), // 数据类型选项转换为标量类型
      options.layout_opt(), // 布局选项
      options.device_opt(), // 设备选项
      options.pinned_memory_opt(), // 固定内存选项
      options.memory_format_opt()); // 内存格式选项
}

// 在 CUDA 上创建一个带步长的空张量对象
TensorBase empty_strided_cuda(
    IntArrayRef size, // 张量的尺寸
    IntArrayRef stride, // 张量的步长
    ScalarType dtype, // 张量的数据类型
    std::optional<Device> device_opt) { // 设备的可选参数
  // 初始化全局的 CUDA 上下文
  at::globalContext().lazyInitCUDA();
  // 获取设备，如果没有提供则使用默认设备
  const auto device = device_or_default(device_opt);
  // 断言设备是 CUDA 设备
  TORCH_INTERNAL_ASSERT(device.is_cuda());
  // 设备守卫，确保在当前设备上操作
  const DeviceGuard device_guard(device);
  // 获取 CUDA 设备的内存分配器
  auto* allocator = at::cuda::getCUDADeviceAllocator();
  // 定义 CUDA DispatchKey 集合
  constexpr c10::DispatchKeySet cuda_dks(c10::DispatchKey::CUDA);
  // 调用通用的带步长的空张量创建函数 empty_strided_generic
  return at::detail::empty_strided_generic(
      size, stride, allocator, cuda_dks, dtype);
}

// 在 CUDA 上创建一个带步长的空张量对象（重载函数）
TensorBase empty_strided_cuda(
    IntArrayRef size, // 张量的尺寸
    IntArrayRef stride, // 张量的步长
    std::optional<ScalarType> dtype_opt, // 数据类型的可选参数
    std::optional<Layout> layout_opt, // 布局的可选参数
    std::optional<Device> device_opt, // 设备的可选参数
    std::optional<bool> pin_memory_opt) { // 是否固定内存的可选参数
  // 检查是否允许固定内存，仅允许在 CPU 密集型张量中固定内存
  TORCH_CHECK(!pin_memory_opt.has_value() || !*pin_memory_opt, "Only dense CPU tensors can be pinned");
  // 断言在调试模式下布局为 Strided
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(layout_or_default(layout_opt) == Layout::Strided);

  // 获取数据类型，如果没有提供则使用默认数据类型
  const auto dtype = dtype_or_default(dtype_opt);
  // 调用前一个重载的 empty_strided_cuda 函数
  return at::detail::empty_strided_cuda(size, stride, dtype, device_opt);
}

// 在 CUDA 上创建一个带步长的空张量对象（使用 TensorOptions）
TensorBase empty_strided_cuda(
    IntArrayRef size, // 张量的尺寸
    IntArrayRef stride, // 张量的步长
    const TensorOptions &options) { // 张量选项
  // 调用前一个重载的 empty_strided_cuda 函数，将选项转换为相应的参数
  return at::detail::empty_strided_cuda(
      size,
      stride,
      optTypeMetaToScalarType(options.dtype_opt()), // 数据类型选项转换为标量类型
      options.layout_opt(), // 布局选项
      options.device_opt(), // 设备选项
      options.pinned_memory_opt()); // 固定内存选项
}

// 结束 ATen 命名空间中的 detail 命名空间
} // namespace at::detail
    // 使用给定的大小和步幅参数，调用 CUDA 版本的 empty_strided 函数以创建一个新的 Tensor
    const TensorOptions &options) {
      // 调用 CUDA 版本的 empty_strided 函数，传入大小(size)、步幅(stride)、数据类型(options.dtype_opt())、
      // 布局选项(options.layout_opt())、设备选项(options.device_opt())、固定内存选项(options.pinned_memory_opt())
      return at::detail::empty_strided_cuda(
          size,
          stride,
          optTypeMetaToScalarType(options.dtype_opt()), // 将选项中的数据类型转换为 ScalarType
          options.layout_opt(),  // 传入的布局选项
          options.device_opt(),  // 传入的设备选项
          options.pinned_memory_opt());  // 传入的固定内存选项
    }
}

}  // namespace at::detail
```