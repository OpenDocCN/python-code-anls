# `.\pytorch\aten\src\ATen\native\vulkan\impl\Packing.h`

```py
// 包含 Vulkan API 的头文件
#include <ATen/native/vulkan/api/api.h>

// 忽略 lint 工具对 facebook-hte-BadMemberName 的警告
// lint-ignore-every CLANGTIDY facebook-hte-BadMemberName

// 命名空间声明：at -> native -> vulkan -> packing
namespace at {
namespace native {
namespace vulkan {
namespace packing {

// 获取将 NCHW 格式张量转换为图像格式的着色器信息
api::ShaderInfo get_nchw_to_image_shader(const vTensor& v_dst);

// 获取将图像格式转换为 NCHW 格式张量的着色器信息
api::ShaderInfo get_image_to_nchw_shader(const vTensor& v_src);

// 记录将 NCHW 格式张量转换为图像格式的操作
void record_nchw_to_image_op(
    api::Context* const context,
    api::ShaderInfo& compute_shader,
    api::VulkanBuffer& src_buffer,
    vTensor& v_dst,
    api::PipelineBarrier pipeline_barrier,
    VkFence fence_handle);

// 记录将图像格式转换为 NCHW 格式张量的操作
bool record_image_to_nchw_op(
    api::Context* const context,
    api::ShaderInfo& compute_shader,
    vTensor& v_src,
    api::VulkanBuffer& dst_buffer,
    api::PipelineBarrier pipeline_barrier,
    VkFence fence_handle);

// 记录将 NCHW 格式张量转换为缓冲区格式的操作
void record_nchw_to_buffer_op(
    api::Context* const context,
    api::VulkanBuffer& src_buffer,
    vTensor& v_dst,
    api::PipelineBarrier pipeline_barrier,
    VkFence fence_handle);

// 记录将缓冲区格式转换为 NCHW 格式张量的操作
bool record_buffer_to_nchw_op(
    api::Context* const context,
    vTensor& v_src,
    api::VulkanBuffer& dst_buffer,
    api::PipelineBarrier pipeline_barrier,
    VkFence fence_handle);

// 将通道打包的图像格式转换为高度打包的张量格式
vTensor convert_image_channels_packed_to_height_packed(const vTensor& v_input);

// 将通道打包的图像格式转换为宽度打包的张量格式
vTensor convert_image_channels_packed_to_width_packed(const vTensor& v_input);

} // namespace packing
} // namespace vulkan
} // namespace native
} // namespace at
```