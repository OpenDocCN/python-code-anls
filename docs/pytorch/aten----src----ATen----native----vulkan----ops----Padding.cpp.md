# `.\pytorch\aten\src\ATen\native\vulkan\ops\Padding.cpp`

```py
// 引入Vulkan操作所需的头文件
#include <ATen/native/vulkan/ops/Common.h>
// 引入c10库中的范围工具
#include <c10/util/irange.h>
// 引入Torch库
#include <torch/library.h>

// 定义ATen命名空间
namespace at {
// 定义Vulkan相关操作的命名空间
namespace native {
namespace vulkan {
namespace ops {
namespace {

// 使用api::utils命名空间
using namespace api::utils;

// 定义pad2d函数，实现在2D张量上的填充操作
Tensor pad2d(
    const Tensor& self_arg,                 // 输入张量
    IntArrayRef padding,                    // 填充尺寸
    const api::ShaderInfo& shader_descriptor // Vulkan着色器描述符
) {
  const int pad_dim = padding.size();        // 获取填充维度
  const IntArrayRef input_size = self_arg.sizes(); // 获取输入张量尺寸
  const int input_dim = input_size.size();   // 获取输入张量维度数

  // 检查填充尺寸是否为1元组或4元组
  TORCH_CHECK(
      pad_dim == 1 || pad_dim == 4,
      "Padding sizes must be a 1-tuple or 4-tuple!");
  // 检查输入张量维度是否大于等于2
  TORCH_CHECK(input_dim >= 2, "Input tensor must have dim >= 2!");

  // 获取当前上下文环境
  api::Context* const context = api::context();

  // 初始化填充的各个方向的尺寸
  int pad_left = padding[0];
  int pad_right = padding[0];
  int pad_top = padding[0];
  int pad_bottom = padding[0];
  if (pad_dim == 4) {
    pad_right = padding[1];
    pad_top = padding[2];
    pad_bottom = padding[3];
  }

  // 获取使用Vulkan后端的输入张量，并转换为vTensor类型
  const Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
  const vTensor& v_self = convert(self);

  // 初始化输出张量的尺寸
  std::vector<int64_t> output_size(input_dim);
  for (const auto d : c10::irange(input_dim)) {
    if (d == input_dim - 1) {
      output_size[d] = input_size[d] + pad_right + pad_left;
    } else if (d == input_dim - 2) {
      output_size[d] = input_size[d] + pad_top + pad_bottom;
    } else {
      output_size[d] = input_size[d];
    }
  }

  // 创建Vulkan输出张量v_output
  vTensor v_output{
      context,
      output_size,
      v_self.dtype(),
  };

  // 定义填充块的结构
  const struct Block final {
    uvec3 extents;                         // 张量的尺寸
    uint32_t _;                            // 保留字段
    uvec4 padding;                         // 填充尺寸
  } block{
      v_output.extents(),                   // 获取输出张量的尺寸
      0u,
      {safe_downcast<uint32_t>(pad_left),   // 安全类型转换后的填充尺寸
       safe_downcast<uint32_t>(pad_right),
       safe_downcast<uint32_t>(pad_top),
       safe_downcast<uint32_t>(pad_bottom)},
  };

  // 创建Uniform参数缓冲区
  api::UniformParamsBuffer params(context, block);
  // 创建管线屏障
  api::PipelineBarrier pipeline_barrier{};

  // 提交计算作业到Vulkan上下文
  context->submit_compute_job(
      // 着色器描述符
      shader_descriptor,
      // 管线屏障
      pipeline_barrier,
      // 全局工作组尺寸
      v_output.extents(),
      // 本地工作组尺寸
      adaptive_work_group_size(v_output.extents()),
      // 栅栏句柄
      VK_NULL_HANDLE,
      // 着色器参数
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_self.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // 参数缓冲区
      params.buffer());

  // 转换并返回Vulkan输出张量
  return convert(v_output);
}

// 在反射填充模式下对2D张量进行填充
Tensor reflection_pad2d(const Tensor& self_arg, IntArrayRef padding) {
  return pad2d(self_arg, padding, VK_KERNEL(reflection_pad2d));
}

// 在复制填充模式下对2D张量进行填充
Tensor replication_pad2d(const Tensor& self_arg, IntArrayRef padding) {
  return pad2d(self_arg, padding, VK_KERNEL(replication_pad2d));
}

// 如果定义了使用Vulkan API，则继续定义相关函数
#ifdef USE_VULKAN_API
TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  // 注册 Vulkan 后端实现的 reflection_pad2d 函数
  m.impl(
      TORCH_SELECTIVE_NAME("aten::reflection_pad2d"),
      TORCH_FN(reflection_pad2d));
  // 注册 Vulkan 后端实现的 replication_pad2d 函数
  m.impl(
      TORCH_SELECTIVE_NAME("aten::replication_pad2d"),
      TORCH_FN(replication_pad2d));
}

#endif /* USE_VULKAN_API */

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
```