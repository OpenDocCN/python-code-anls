# `.\pytorch\aten\src\ATen\native\vulkan\ops\Glu.cpp`

```py
// 包含 Vulkan 相关的头文件
#include <ATen/native/vulkan/ops/Common.h>
// 包含 PyTorch 的库头文件
#include <torch/library.h>

// 定义命名空间 at::native::vulkan::ops::
namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

// 使用 api::utils 命名空间
using namespace api::utils;

// Vulkan 实现的 GLU 操作，接受四维张量作为输入，可选参数 dim 默认为 -1
Tensor glu(const at::Tensor& input_arg, const int64_t dim = -1) {
  // 检查输入张量维度是否为 4
  TORCH_CHECK(input_arg.dim() == 4, "Vulkan glu only supports 4-dim input!");
  // 检查参数 dim 是否为 1
  TORCH_CHECK(
      dim == 1,
      "Vulkan glu only supports GLU for dim = 1, but got dim = ",
      dim);
  // 检查输入张量的通道维度是否是 4 的倍数
  TORCH_CHECK(
      get_dim<Dim4D::Channel>(input_arg) % 4 == 0,
      "Vulkan glu expects channel dim to be multiple of 4!");

  // 将输入张量转换为 Vulkan 张量
  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  // 将 Vulkan 张量转换为 vTensor
  const vTensor& v_input = convert(input);
  // 获取 vTensor 的尺寸信息
  const IntArrayRef v_input_sizes = v_input.sizes();

  // 计算输出张量的通道尺寸
  auto output_ch_size = v_input.sizes()[1] / 2;

  // 获取 Vulkan API 的上下文
  api::Context* const context = api::context();

  // 创建 vTensor 作为输出
  vTensor v_output{
      context,
      {v_input_sizes[0], output_ch_size, v_input_sizes[2], v_input_sizes[3]},
      v_input.dtype(),
  };

  // 定义用于传递给着色器的 Block 结构体
  const struct Block final {
    uvec3 extents;
    int32_t chext;
  } block{v_output.extents(), safe_downcast<int32_t>(output_ch_size)};

  // 创建 UniformParamsBuffer 用于传递参数给着色器
  api::UniformParamsBuffer params(context, block);
  // 创建 PipelineBarrier 用于管线屏障操作
  api::PipelineBarrier pipeline_barrier{};

  // 提交计算作业到 Vulkan 上下文
  context->submit_compute_job(
      // 着色器描述符
      VK_KERNEL(glu_channel_mul4),
      // 管线屏障
      pipeline_barrier,
      // 全局工作组大小
      v_output.extents(),
      // 局部工作组大小
      adaptive_work_group_size(v_output.extents()),
      // 栅栏句柄
      VK_NULL_HANDLE,
      // 着色器参数
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // 参数缓冲区
      params.buffer());

  // 将 vTensor 转换回普通张量并返回
  return convert(v_output);
}

// 如果定义了 USE_VULKAN_API，注册 Vulkan 实现的 glu 函数到 ATen 库
#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::glu"), TORCH_FN(glu));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
```