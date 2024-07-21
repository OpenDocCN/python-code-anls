# `.\pytorch\aten\src\ATen\native\vulkan\ops\Factory.cpp`

```py
#include <ATen/native/vulkan/ops/Factory.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

// 定义函数 _empty_affine_quantized，返回量化空张量
Tensor _empty_affine_quantized(
    const IntArrayRef sizes,                                // 接收张量大小的引用数组
    const std::optional<ScalarType> dtype,                   // 可选的张量数据类型
    const std::optional<c10::Layout> layout,                // 可选的张量布局
    const std::optional<Device> device,                     // 可选的设备
    const std::optional<bool> pin_memory,                   // 可选的内存固定标志
    const double scale,                                     // 缩放因子
    const int64_t zero_point,                               // 零点
    const optional<MemoryFormat> memory_format) {           // 可选的内存格式
  // 设置存储类型为 TEXTURE_3D
  api::StorageType storage_type = api::StorageType::TEXTURE_3D;
  // 调用 convert_quantized 函数，将张量转换为量化张量
  return convert_quantized(vTensor{
      api::context(),                                      // 获取 Vulkan 上下文
      sizes.vec(),                                          // 将大小转换为向量
      scale,                                                // 缩放因子
      zero_point,                                           // 零点
      convert_dtype(dtype ? *dtype : c10::kFloat),          // 转换数据类型
      storage_type,                                         // 存储类型
      memory_format ? get_gpu_memory_layout(storage_type, *memory_format)
                    : api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED,
  });
}

// 定义函数 empty_memory_format，返回内存格式化的张量
Tensor empty_memory_format(
    const IntArrayRef sizes,                                // 接收张量大小的引用数组
    const std::optional<ScalarType> dtype,                   // 可选的张量数据类型
    const std::optional<c10::Layout> layout,                // 可选的张量布局
    const std::optional<Device> device,                     // 可选的设备
    const std::optional<bool> pin_memory,                   // 可选的内存固定标志
    const optional<MemoryFormat> memory_format) {           // 可选的内存格式
  // 设置存储类型为 TEXTURE_3D
  api::StorageType storage_type = api::StorageType::TEXTURE_3D;
  // 调用 convert 函数，将张量转换为指定内存格式
  return convert(vTensor{
      api::context(),                                      // 获取 Vulkan 上下文
      sizes.vec(),                                          // 将大小转换为向量
      convert_dtype(dtype ? *dtype : c10::kFloat),          // 转换数据类型
      storage_type,                                         // 存储类型
      memory_format ? get_gpu_memory_layout(storage_type, *memory_format)
                    : api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED,
  });
}

// 定义函数 empty_strided，返回分步空张量
Tensor empty_strided(
    const IntArrayRef sizes,                                // 接收张量大小的引用数组
    const IntArrayRef /* strides */,                        // 可选的步长数组（未使用）
    const optional<ScalarType> dtype,                       // 可选的张量数据类型
    const optional<c10::Layout> layout,                     // 可选的张量布局
    const optional<Device> device,                          // 可选的设备
    const optional<bool> pin_memory) {                      // 可选的内存固定标志
  // 调用 empty_memory_format 函数，创建内存格式化的张量，默认使用连续内存格式
  return empty_memory_format(
      sizes, dtype, layout, device, pin_memory, c10::MemoryFormat::Contiguous);
}

#ifdef USE_VULKAN_API

// 实现 Vulkan API 的 Torch 库
TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  // 注册函数 empty_memory_format 到 Vulkan 后端
  m.impl(
      TORCH_SELECTIVE_NAME("aten::empty.memory_format"),
      at::native::vulkan::ops::empty_memory_format);
  // 注册函数 _empty_affine_quantized 到 Vulkan 后端
  m.impl(
      TORCH_SELECTIVE_NAME("aten::_empty_affine_quantized"),
      at::native::vulkan::ops::_empty_affine_quantized);
  // 注册函数 empty_strided 到 Vulkan 后端
  m.impl(
      TORCH_SELECTIVE_NAME("aten::empty_strided"),
      TORCH_FN(at::native::vulkan::ops::empty_strided));
}

#endif /* USE_VULKAN_API */

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
```