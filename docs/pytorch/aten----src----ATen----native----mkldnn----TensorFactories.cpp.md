# `.\pytorch\aten\src\ATen\native\mkldnn\TensorFactories.cpp`

```
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/native/mkldnn/MKLDNNCommon.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty_native.h>
#endif

namespace at { namespace native {

#if AT_MKLDNN_ENABLED()

// 创建一个 MKL-DNN 张量，用给定的尺寸、数据类型、布局、设备等参数
Tensor empty_mkldnn(IntArrayRef sizes, std::optional<ScalarType> dtype, std::optional<Layout> layout, std::optional<Device> device, std::optional<bool> pin_memory, std::optional<c10::MemoryFormat> optional_memory_format) {
  // 检查是否传入了 'memory_format' 参数，如果有则报错，因为与 MKL-DNN 张量不兼容
  TORCH_CHECK(
     !optional_memory_format.has_value(),
     "'memory_format' argument is incompatible with mkldnn tensor");
  
  // NOTE: int32_t dims from ideep::tensor but sizes needs int64_t
  // TODO: support int64_t dims in ideep::tensor to avoid extra conversion
  // 将 IntArrayRef 转换为 ideep::tensor::dims 类型，这里暂时使用 int32_t，未来可能支持 int64_t
  ideep::tensor::dims dst_dims (sizes.begin(), sizes.end());
  
  // 根据传入的 dtype 获取对应的 MKL-DNN 数据类型，如果未指定则默认为 f32
  auto data_type = dtype.has_value() ? get_mkldnn_dtype(dtype.value()) : ideep::tensor::data_type::f32;
  
  // 使用 ideep::tensor::dims 和数据类型创建一个 MKL-DNN 张量
  ideep::tensor it {dst_dims, data_type};
  
  // 调用 new_with_itensor_mkldnn 函数创建新的 MKL-DNN 张量，并返回
  return new_with_itensor_mkldnn(std::move(it), dtype, device);
}

#else

// 如果 MKL-DNN 功能未启用，直接抛出错误，表示不支持 MKL-DNN 构建
Tensor empty_mkldnn(IntArrayRef sizes, std::optional<ScalarType> dtype, std::optional<Layout> layout, std::optional<Device> device, std::optional<bool> pin_memory, std::optional<c10::MemoryFormat> optional_memory_format) {
  TORCH_CHECK(false, "empty_mkldnn: MKL-DNN build is disabled");
}

#endif // AT_MKLDNN_ENABLED()

}} // namespace at::native
```