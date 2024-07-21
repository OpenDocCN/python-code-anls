# `.\pytorch\aten\src\ATen\native\vulkan\ops\Convert.h`

```py
/**
 * pragma once 指令：确保当前头文件只被编译一次，防止重复包含。
 */

#ifdef USE_VULKAN_API
/**
 * 使用 Vulkan API 情况下的命名空间和功能定义。
 */
#include <ATen/native/vulkan/VulkanOpaqueTensorImpl.h>
#include <ATen/native/vulkan/api/Tensor.h>
#include <ATen/native/vulkan/api/Types.h>
#include <c10/util/accumulate.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

/**
 * 根据请求的存储类型和指定的 c10::MemoryFormat，确定适当的 GPU 内存布局限定符。
 * @param storage_type 请求的存储类型
 * @param memory_format 指定的内存格式
 * @return 返回适合的 GPU 内存布局
 */
inline api::GPUMemoryLayout get_gpu_memory_layout(
    const api::StorageType storage_type,
    const c10::MemoryFormat memory_format) {
  if (storage_type == api::StorageType::BUFFER) {
    switch (memory_format) {
      case c10::MemoryFormat::Contiguous:
        return api::GPUMemoryLayout::TENSOR_WIDTH_PACKED;
      case c10::MemoryFormat::ChannelsLast:
        return api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED;
      default:
        VK_THROW("Invalid memory format used to create vTensor!");
    }
  }
  // 对于纹理存储，始终返回打包通道维度的内存布局。当前，对于二维张量，添加了一个通道维度，
  // 以及3个零填充通道，结果形状为{4, H, W}。对于一维张量，它被展开到大小{1, 1, L}，
  // 并添加3个零填充通道，以产生最终大小{4, 1, L}。这是为了确保物理纹理位置直接对应于逻辑张量坐标
  // （因此，texelFetch(ivec3(x, y, 0), 0)将对应于tensor[y, x]）。
  //
  // TODO(ssjia): 让二维和一维张量默认使用TENSOR_WIDTH_PACKED。
  return api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED;
}

/*
 * 将c10::ScalarType转换为等效的::at::native::vulkan::api::ScalarType。
 */
static inline api::ScalarType convert_dtype(const c10::ScalarType dtype) {
#define DEFINE_CASE(ctype, vkformat, name) \
  case c10::ScalarType::name:              \
    return ::at::native::vulkan::api::ScalarType::name;

  switch (dtype) {
    VK_FORALL_SCALAR_TYPES(DEFINE_CASE)
    default:
      TORCH_CHECK(false, "Not a supported Vulkan ScalarType!");
  }
#undef DEFINE_CASE
}

/*
 * 将::at::native::vulkan::api::ScalarType转换为等效的c10::ScalarType。
 */
static inline c10::ScalarType convert_dtype(const api::ScalarType dtype) {
#define DEFINE_CASE(ctype, vkformat, name)          \
  case ::at::native::vulkan::api::ScalarType::name: \
    return c10::ScalarType::name;

  switch (dtype) {
    VK_FORALL_SCALAR_TYPES(DEFINE_CASE)
    default:
      TORCH_CHECK(false, "Not a supported c10::ScalarType!");
  }
#undef DEFINE_CASE
}

/**
 * vTensor 的实现类型别名，基于 VulkanOpaqueTensorImpl<vTensor>。
 */
using vTensorImpl = VulkanOpaqueTensorImpl<vTensor>;

/**
 * 将vTensor转换为普通Tensor。
 */
inline Tensor convert(const vTensor& tensor) {
  return at::detail::make_tensor<vTensorImpl>(
      DispatchKeySet(DispatchKey::Vulkan),
      c10::scalarTypeToTypeMeta(convert_dtype(tensor.dtype())),
      at::Device(at::kVulkan),
      tensor,
      tensor.sizes(),
      tensor.strides());
}
// 将一个量化的 vTensor 转换为普通的 Tensor
inline Tensor convert_quantized(const vTensor& tensor) {
  // 检查输入的 tensor 是否是量化的，如果不是则抛出错误信息
  TORCH_CHECK(tensor.is_quantized(), "Not a Quantized Tensor");
  
  // 使用 vTensor 的属性创建一个新的 Tensor
  return at::detail::make_tensor<vTensorImpl>(
      DispatchKeySet(DispatchKey::Vulkan),   // 使用 Vulkan DispatchKeySet
      c10::scalarTypeToTypeMeta(convert_dtype(tensor.dtype())),  // 将数据类型转换为对应的类型元信息
      at::Device(at::kVulkan),  // 指定设备为 Vulkan
      tensor,  // 使用原始的 vTensor 数据
      tensor.sizes(),  // 保持原始的大小
      tensor.strides());  // 保持原始的步幅
}

// 将普通的 Tensor 转换为 vTensor 的引用
inline vTensor& convert(const Tensor& tensor) {
  // 内部断言，检查是否是 Vulkan 的 tensor，如果不是则抛出错误信息
  TORCH_INTERNAL_ASSERT(tensor.is_vulkan(), "Vulkan tensor expected!");

  // 获取 tensor 的实现对象指针，强制转换为 vTensorImpl
  vTensorImpl* const impl =
      static_cast<vTensorImpl*>(tensor.unsafeGetTensorImpl());

  // 返回该实现对象的不安全操作句柄
  return impl->unsafe_opaque_handle();
}

// 结束 Vulkan 相关的命名空间
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
```