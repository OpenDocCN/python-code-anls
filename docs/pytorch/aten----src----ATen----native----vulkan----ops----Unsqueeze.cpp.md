# `.\pytorch\aten\src\ATen\native\vulkan\ops\Unsqueeze.cpp`

```
// 包含 Vulkan 相关的头文件
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <torch/library.h>

// 进入 Vulkan 相关操作的命名空间
namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

// 使用 Vulkan API 的通用工具函数
using namespace api::utils;

// 定义一个结构体 Block，用于存储元数据信息
struct Block final {
  ivec2 info;  // 包含两个整数信息的向量
};

// Vulkan 实现的 unsqueeze 函数，用于扩展张量的维度
Tensor unsqueeze(const at::Tensor& self, int64_t dim) {
  // 检查输入张量的维度是否超过 3 维
  TORCH_CHECK(
      self.dim() <= 3,
      "Vulkan unsqueeze only supports up to 3d tensors as input!");
  // 检查插入维度的合法性
  TORCH_CHECK(
      dim >= -self.dim() - 1 && dim <= self.dim(),
      "Vulkan unsqueeze dimension out of range expected to be in range of [",
      -self.dim() - 1,
      ",",
      self.dim(),
      "], but got ",
      dim);

  // 获取全局 Vulkan 上下文
  api::Context* const context = api::context();

  // 将输入 Tensor 转换为 vTensor
  const Tensor input = self.is_vulkan() ? self : self.vulkan();
  const vTensor& v_input = convert(input);

  // 创建输出纹理。对于 unsqueeze 操作，添加一个维度。
  std::vector<int64_t> output_size = v_input.sizes();
  if (dim < 0) {
    dim += (self.dim() + 1);
  }
  output_size.insert(output_size.begin() + dim, 1);
  // 创建输出 vTensor
  vTensor v_output{
      context,
      output_size,
      convert_dtype(self.scalar_type()),
  };

  // 需要确定如何在命令缓冲区中插入内存屏障
  api::PipelineBarrier pipeline_barrier{};

  // 全局工作项数等于输出纹理的尺寸
  uvec3 global_size = v_output.extents();
  // 自适应确定局部工作组大小，通常为 {4, 4, 4}
  uvec3 local_size = adaptive_work_group_size(global_size);

  // 当在第 0 维度进行 unsqueeze 时，只需复制元数据
  if (dim == 0) {
    const vTensor& v_self = convert(self);
    uvec3 src_offset{};
    uvec3 dst_offset{};
    context->submit_copy<api::VulkanImage, api::VulkanImage>(
        // 管道屏障
        pipeline_barrier,
        // 图像
        v_self.image(pipeline_barrier, api::PipelineStage::TRANSFER),
        v_output.image(
            pipeline_barrier,
            api::PipelineStage::TRANSFER,
            api::MemoryAccessType::WRITE),
        // 复制细节
        v_self.extents(),
        src_offset,
        dst_offset,
        // 栅栏句柄
        VK_NULL_HANDLE);
    return convert(v_output);
  }

  else {
    int channel_index = 1; // 在 3D 张量中的通道维度索引
    // 对于 1D、2D 张量，调整 dim 和 channel_index
    if (self.dim() < 3) {
      dim += (3 - self.dim());
      channel_index = 0;
    }

    // 创建参数缓冲区
    struct Block block {
      {
        // 要 unsqueeze 的维度
        static_cast<int32_t>(dim),
        // Image3D 中通道的数量，每 4 个元素为一组
        static_cast<int32_t>(
            std::ceil(static_cast<float>(output_size[channel_index]) / 4)),
      }
    };

    // 使用 block 创建统一参数缓冲区
    api::UniformParamsBuffer params(context, block);
    context->submit_compute_job(
        // 提交计算任务到上下文对象中
        VK_KERNEL(unsqueeze),  // 使用名为"unsqueeze"的 Vulkan 内核
        pipeline_barrier,      // 使用提供的管线屏障对象来控制流水线访问
        global_size,           // 指定全局工作组大小，用于并行计算
        local_size,            // 指定局部工作组大小，用于线程内并行计算
        VK_NULL_HANDLE,        // 使用空的 Vulkan 句柄来表示无关联的围栏
        v_output.image(        // 将输出图像注册到计算管线阶段，允许写访问
            pipeline_barrier,  // 使用相同的管线屏障来控制图像的访问
            api::PipelineStage::COMPUTE,  // 在计算管线阶段进行访问
            api::MemoryAccessType::WRITE),  // 允许写访问
        v_input.image(         // 将输入图像注册到计算管线阶段，用于读访问
            pipeline_barrier,  // 使用相同的管线屏障来控制图像的访问
            api::PipelineStage::COMPUTE),  // 在计算管线阶段进行访问
        params.buffer());      // 使用参数缓冲区对象提供计算所需的参数数据
    // 将计算后的输出转换为合适的格式并返回
    return convert(v_output);
  }
} // 结束 at 命名空间

#ifdef USE_VULKAN_API
// 如果定义了 USE_VULKAN_API 宏，则进入条件编译

// 实现 aten 库在 Vulkan 后端的 TORCH_LIBRARY_IMPL
TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  // 注册 unsqueeze 操作的 Vulkan 后端实现函数 unsqueeze
  m.impl(TORCH_SELECTIVE_NAME("aten::unsqueeze"), TORCH_FN(unsqueeze));
}

#endif /* USE_VULKAN_API */ // 结束条件编译部分

} // 结束 namespace
} // 结束 ops 命名空间
} // 结束 vulkan 命名空间
} // 结束 native 命名空间
} // 结束 at 命名空间
```