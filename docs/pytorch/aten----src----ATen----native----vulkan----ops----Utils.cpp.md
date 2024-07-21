# `.\pytorch\aten\src\ATen\native\vulkan\ops\Utils.cpp`

```py
/*
 * This header includes Vulkan-specific tensor packing functionalities.
 * It ensures access to Vulkan common operations and types.
 */
#include <ATen/native/vulkan/impl/Packing.h>
#include <ATen/native/vulkan/ops/Common.h>

/*
 * Conditional inclusion based on the presence of per-operator headers.
 * If AT_PER_OPERATOR_HEADERS is not defined, it includes general ATen functions.
 * Otherwise, it includes specific ATen operator headers like cat, empty, narrow, zeros.
 */
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/cat.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/narrow.h>
#include <ATen/ops/zeros.h>
#endif

namespace at {
namespace native {
namespace vulkan {
namespace ops {

namespace utils {

/*
 * Function: nchw_to_nc4hw
 *
 * This function formats an input tensor from NCHW layout to NC4HW layout, which
 * facilitates direct copying of the tensor's buffer into a GPU texture. The
 * steps involved are:
 *
 * 1. Retrieve dimensions N (batch size), C (channels), H (height), W (width)
 *    from the input tensor 'src'.
 *
 * 2. Align the number of channels (C) to the nearest higher multiple of 4 (C_aligned).
 *
 * 3. Calculate the total number of groups of 4 channels needed after alignment (NC4).
 *
 * 4. Add padding to the tensor along the channel dimension to ensure it is a multiple
 *    of 4, resulting in a new tensor 'src_padded'.
 *
 * 5. Reshape 'src_padded' to group channels into blocks of 4, resulting in a tensor
 *    'src_NC4HW' of shape {NC4, 4, H, W}.
 *
 * 6. Permute the dimensions of 'src_NC4HW' to ensure the groups of 4 channels are
 *    contiguous, resulting in the final shape {NC4, H, W, 4}.
 *
 * 7. Return a contiguous version of 'src_NC4HW', ensuring all elements are stored
 *    in a single, contiguous block of memory.
 *
 * Parameters:
 * - src: Input tensor in NCHW layout.
 *
 * Returns:
 * - Tensor: Output tensor formatted in NC4HW layout suitable for GPU texture operations.
 */
Tensor nchw_to_nc4hw(const Tensor& src) {
  uint32_t N = get_dim<Dim4D::Batch>(src.sizes());   // Retrieve batch size (N)
  uint32_t C = get_dim<Dim4D::Channel>(src.sizes()); // Retrieve number of channels (C)
  uint32_t H = get_dim<Dim4D::Height>(src.sizes());  // Retrieve height (H)
  uint32_t W = get_dim<Dim4D::Width>(src.sizes());   // Retrieve width (W)

  uint32_t C_aligned = api::utils::align_up(C, 4u);  // Align channels to multiple of 4
  uint32_t NC4 = (N * C_aligned) / 4;                // Calculate number of NC4 blocks

  // Add padding to ensure channel dimension is a multiple of 4
  Tensor padding = at::zeros({N, C_aligned - C, H, W}, src.options());
  Tensor src_padded = at::cat({src.reshape({N, C, H, W}), padding}, 1);

  // Reshape to group channels into blocks of 4 and permute dimensions for contiguity
  Tensor src_NC4HW = src_padded.reshape({NC4, 4, H, W}).permute({0, 2, 3, 1});

  // Return a contiguous version of the tensor
  return src_NC4HW.contiguous();
}

/*
 * Creates a staging tensor into which texture data, formatted in NC4HW layout,
 * can be directly copied. The shape of the staging tensor matches that produced
 * by a call to nchw_to_nc4hw().
 */
/*
 * 创建一个暂存张量，用于存储从vTensor复制的纹理数据。
 * 张量的维度是NC4HW，其中N是批量大小，C是通道数，H是高度，W是宽度。
 * NC4是C向上取整至4的倍数后的结果，以便对齐处理。
 * 使用vTensor的纹理格式对应的数据类型创建张量，而非使用options().dtype()，
 * 这是为了确保暂存张量的字节数与图像纹理中的字节数匹配。
 * 参考api::vk_format()的注释了解更多详情。
 */
Tensor create_staging_tensor(const vTensor& v_in) {
  uint32_t N = get_dim<Dim4D::Batch>(v_in.sizes());
  uint32_t C = get_dim<Dim4D::Channel>(v_in.sizes());
  uint32_t H = get_dim<Dim4D::Height>(v_in.sizes());
  uint32_t W = get_dim<Dim4D::Width>(v_in.sizes());

  uint32_t NC4 = N * api::utils::div_up(C, 4u);

  return at::empty(
      {NC4, H, W, 4},
      at::device(at::kCPU).dtype(convert_dtype(v_in.texture_dtype())));
}

/*
 * 将NC4HW格式的张量重新格式化为NCHW格式。
 * 该函数是对create_staging_tensor()创建的暂存张量进行的逆向操作，
 * 以恢复原始张量的转换。
 *
 * 需要传入原始张量的尺寸大小以完全恢复原始张量的属性。
 * 首先对张量进行反置步骤和通道分组步骤的撤销操作。
 * 然后移除填充通道。
 * 最后按原始大小和数据类型重塑张量，并返回一个连续的张量。
 */
Tensor nc4hw_to_nchw(const Tensor& t_in, IntArrayRef sizes) {
  uint32_t N = get_dim<Dim4D::Batch>(sizes);
  uint32_t C = get_dim<Dim4D::Channel>(sizes);
  uint32_t H = get_dim<Dim4D::Height>(sizes);
  uint32_t W = get_dim<Dim4D::Width>(sizes);

  uint32_t C_aligned = api::utils::align_up(C, 4u);

  Tensor t_in_padded = t_in.permute({0, 3, 1, 2}).reshape({N, C_aligned, H, W});
  Tensor t_in_shaved =
      at::narrow(t_in_padded, /*dim=*/1, /*start=*/0, /*end=*/C);

  return t_in_shaved.reshape(sizes).contiguous();
}

/*
 * 将缓冲区中的数据复制到vTensor中。
 * 确保源缓冲区和目标纹理具有相同的字节数。
 * 提交一个复制命令到Vulkan设备上下文中，从源缓冲区到目标纹理。
 * 使用给定的管线屏障控制复制过程的同步。
 */
void copy_buffer_to_vtensor(
    api::VulkanBuffer& src_buffer,
    vTensor& v_dst,
    api::PipelineBarrier& pipeline_barrier) {
  api::Context* const context = api::context();

  TORCH_CHECK(
      src_buffer.mem_size() == v_dst.gpu_nbytes(),
      "Vulkan copy_buffer_to_vtensor: source buffer and destination texture "
      "do not have the same number of bytes");

  context->submit_copy<api::VulkanBuffer, api::VulkanImage>(
      pipeline_barrier,
      src_buffer,
      v_dst.image(
          pipeline_barrier,
          api::PipelineStage::TRANSFER,
          api::MemoryAccessType::WRITE),
      v_dst.extents(),
      {0u, 0u, 0u},
      {0u, 0u, 0u},
      VK_NULL_HANDLE);
}

/*
 * 在Vulkan设备上下文中，将一个缓冲区的数据复制到另一个缓冲区中。
 * 使用给定的上下文、源缓冲区和目标缓冲区来执行复制操作。
 */
void copy_buffer_to_buffer(
    api::Context* const context,
    api::StorageBuffer& src,
    api::StorageBuffer& dst,
    // 创建一个名为 pipeline_barrier 的 api::PipelineBarrier 对象
    api::PipelineBarrier pipeline_barrier{};
    
    // 在 Vulkan API 中提交一个复制操作，涉及以下步骤和资源：
    context->submit_copy<api::VulkanBuffer, api::VulkanBuffer>(
        // 提交的是一个管线屏障操作
        pipeline_barrier,
        // 源缓冲区
        src.buffer(),
        // 目标缓冲区
        dst.buffer(),
        // 复制的详细信息：源缓冲区大小（以字节为单位）、源和目标缓冲区的偏移量
        {static_cast<uint32_t>(src.buffer().mem_size()), 0u, 0u},
        // 源缓冲区的偏移量
        {0u, 0u, 0u},
        // 目标缓冲区的偏移量
        {0u, 0u, 0u},
        // 使用给定的 VkFence 句柄来同步操作
        fence_handle);
// 检查两个张量是否可以进行广播
void is_broadcastable(const Tensor& input1, const Tensor& input2) {
    // 确保每个输入张量的维度不超过4，因为 Vulkan 只支持最多4维的张量
    TORCH_CHECK(
        input1.dim() <= 4 && input2.dim() <= 4,
        "Vulkan only supports tensors <= 4 dimensions");

    // 检查输入张量的形状是否可以广播
    // 参考 https://pytorch.org/docs/stable/notes/broadcasting.html
    // 查看广播语义的说明
    const std::string broadcast_error_msg = "Tensors are not broadcastable!";

    // 如果两个张量的批量维度不同，即它们的第一维度不相同，则报错
    if (get_dim<Dim4D::Batch>(input1) != get_dim<Dim4D::Batch>(input2)) {
    // 检查输入张量的 Batch 维度是否可以广播，如果不能则抛出错误信息
    TORCH_CHECK(
        get_dim<Dim4D::Batch>(input1) == 1 ||
            get_dim<Dim4D::Batch>(input2) == 1,
        broadcast_error_msg);
    // 如果输入张量的 Channel 维度不相等，检查是否可以广播，如果不能则抛出错误信息
    if (get_dim<Dim4D::Channel>(input1) != get_dim<Dim4D::Channel>(input2)) {
        TORCH_CHECK(
            get_dim<Dim4D::Channel>(input1) == 1 ||
                get_dim<Dim4D::Channel>(input2) == 1,
            broadcast_error_msg);
    }
    // 如果输入张量的 Height 维度不相等，检查是否可以广播，如果不能则抛出错误信息
    if (get_dim<Dim4D::Height>(input1) != get_dim<Dim4D::Height>(input2)) {
        TORCH_CHECK(
            get_dim<Dim4D::Height>(input1) == 1 ||
                get_dim<Dim4D::Height>(input2) == 1,
            broadcast_error_msg);
    }
    // 如果输入张量的 Width 维度不相等，检查是否可以广播，如果不能则抛出错误信息
    if (get_dim<Dim4D::Width>(input1) != get_dim<Dim4D::Width>(input2)) {
        TORCH_CHECK(
            get_dim<Dim4D::Width>(input1) == 1 ||
                get_dim<Dim4D::Width>(input2) == 1,
            broadcast_error_msg);
    }
// 结束 utils 命名空间
} // namespace utils

// 结束 ops 命名空间
} // namespace ops

// 结束 vulkan 命名空间
} // namespace vulkan

// 结束 native 命名空间
} // namespace native

// 结束 at 命名空间
} // namespace at
```