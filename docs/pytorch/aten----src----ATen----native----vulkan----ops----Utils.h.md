# `.\pytorch\aten\src\ATen\native\vulkan\ops\Utils.h`

```py
#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/ops/Common.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

namespace utils {

// 将输入张量从 NCHW 格式转换为 NC4HW 格式
Tensor nchw_to_nc4hw(const Tensor&);

// 创建用于临时存储的张量，基于给定的 Vulkan 张量
Tensor create_staging_tensor(const vTensor&);

// 将输入张量从 NC4HW 格式转换为 NCHW 格式
Tensor nc4hw_to_nchw(const Tensor&, IntArrayRef);

// 将一个存储缓冲区的数据复制到另一个存储缓冲区
void copy_buffer_to_buffer(
    api::Context* const context,
    api::StorageBuffer& src,
    api::StorageBuffer& dst,
    VkFence fence_handle);

// 将存储缓冲区的数据复制到 Vulkan 张量
void copy_buffer_to_vtensor(
    api::VulkanBuffer&,
    vTensor&,
    api::PipelineBarrier&);

// 将 Vulkan 张量的数据复制到存储缓冲区
void copy_vtensor_to_buffer(
    vTensor&,
    api::VulkanBuffer&,
    api::PipelineBarrier&,
    const VkFence fence_handle = VK_NULL_HANDLE);

// 将存储缓冲区的数据打包到 Vulkan 张量
void pack_buffer_to_vtensor(
    api::VulkanBuffer&,
    vTensor&,
    api::PipelineBarrier&);

// 将临时存储的数据打包到 Vulkan 张量
void pack_staging_to_vtensor(api::VulkanBuffer&, vTensor&);

// 将 Vulkan 张量的数据打包到临时存储
bool pack_vtensor_to_staging(
    vTensor&,
    api::VulkanBuffer&,
    const VkFence fence_handle = VK_NULL_HANDLE);

// 检查两个张量是否可广播
void is_broadcastable(const Tensor& input1, const Tensor& input2);

// 计算广播后的张量大小
std::vector<int64_t> broadcast_size(const Tensor& t1, const Tensor& t2);

// 从给定张量中提取指定位置的 texel 值，用于调试和单元测试
// 此函数效率较低，因为需要执行隔离操作以提取单个值
api::utils::vec4 extract_texel(
    const Tensor& tensor,
    const api::utils::ivec3& pos);

// 创建一个 api::utils::ivec2 对象，根据给定的整数数组 ints 和可选的反转标志
inline api::utils::ivec2 make_ivec2(
    const IntArrayRef ints,
    bool reverse = false) {
  // 确保整数数组的大小为 2
  VK_CHECK_COND(ints.size() == 2);
  // 根据反转标志返回相应的 ivec2 对象
  if (reverse) {
    return {
        api::utils::safe_downcast<int32_t>(ints[1]),
        api::utils::safe_downcast<int32_t>(ints[0])};
  } else {
    return {
        api::utils::safe_downcast<int32_t>(ints[0]),
        api::utils::safe_downcast<int32_t>(ints[1])};
  }
}

// 创建一个 api::utils::ivec4 对象，根据给定的整数数组 ints 和可选的反转标志
inline api::utils::ivec4 make_ivec4(
    const IntArrayRef ints,
    bool reverse = false) {
  // 确保整数数组的大小为 4
  VK_CHECK_COND(ints.size() == 4);
  // 根据反转标志返回相应的 ivec4 对象
  if (reverse) {
    return {
        api::utils::safe_downcast<int32_t>(ints[3]),
        api::utils::safe_downcast<int32_t>(ints[2]),
        api::utils::safe_downcast<int32_t>(ints[1]),
        api::utils::safe_downcast<int32_t>(ints[0]),
    };
  } else {
    return {
        api::utils::safe_downcast<int32_t>(ints[0]),
        api::utils::safe_downcast<int32_t>(ints[1]),
        api::utils::safe_downcast<int32_t>(ints[2]),
        api::utils::safe_downcast<int32_t>(ints[3]),
    };
  }
}

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
```