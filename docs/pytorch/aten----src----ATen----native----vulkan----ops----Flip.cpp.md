# `.\pytorch\aten\src\ATen\native\vulkan\ops\Flip.cpp`

```py
// 引入 Vulkan 相关头文件
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <torch/library.h>

// 定义命名空间 at::native::vulkan::ops::
namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

// 使用 api::utils 命名空间
using namespace api::utils;

// Vulkan 特定的 flip 函数，用于在 Vulkan 上执行张量翻转操作
Tensor flip(const at::Tensor& self, const IntArrayRef dim_list) {
  // 检查输入张量的维度在 1 到 4 之间
  TORCH_CHECK(
      self.dim() >= 1 || self.dim() <= 4,
      "Vulkan flip supports up to 4d tensors as input!");

  // 获取全局 Vulkan 上下文
  api::Context* const context = api::context();

  // 将输入张量转换为 vTensor
  const Tensor input = self.is_vulkan() ? self : self.vulkan();
  const vTensor& v_input = convert(input);

  // 创建输出纹理 vTensor
  vTensor v_output{
      context,
      v_input.sizes(),
      convert_dtype(self.scalar_type()),
  };

  // 创建管线屏障对象，用于确定如何在命令缓冲区中插入内存屏障
  api::PipelineBarrier pipeline_barrier{};

  // 创建维度参数数组
  std::vector<int32_t> dim_args = {0, 0, 0, 0};
  for (const auto dim : dim_list) {
    // 检查翻转维度是否在有效范围内
    TORCH_CHECK(
        dim >= -self.dim() - 1 && dim <= self.dim(),
        "Vulkan flip dimension out of range expected to be in range of [",
        -self.dim() - 1,
        ",",
        self.dim(),
        "], but got ",
        dim);
    // 标准化维度值
    int normalized_dim = utils::normalize(dim, self.dim());

    // 如果张量维度小于4，则需要将维度值偏移
    if (self.dim() < 4) {
      normalized_dim += (4 - self.dim());
    }
    dim_args[normalized_dim] = 1;
  }

  // 创建参数块，包含四维张量的尺寸和翻转维度信息
  const struct Block final {
    uvec4 extents;
    ivec4 dims;
  } block{
      {get_dim<Dim4D::Width>(v_output),
       get_dim<Dim4D::Height>(v_output),
       get_dim<Dim4D::Channel>(v_output),
       get_dim<Dim4D::Batch>(v_output)},
      {dim_args[3], dim_args[2], dim_args[1], dim_args[0]},
  };

  // 创建统一参数缓冲区
  api::UniformParamsBuffer params(context, block);

  // 提交计算作业到 Vulkan 上下文
  context->submit_compute_job(
      // Vulkan 着色器描述符
      VK_KERNEL(flip),
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
  
  // 将 vTensor 转换为普通张量并返回
  return convert(v_output);
};

// 在使用 Vulkan API 的情况下注册 Vulkan 的 flip 实现到 aten 库中
#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::flip"), TORCH_FN(flip));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
```