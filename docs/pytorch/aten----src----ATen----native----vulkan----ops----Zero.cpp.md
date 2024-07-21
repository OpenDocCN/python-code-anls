# `.\pytorch\aten\src\ATen\native\vulkan\ops\Zero.cpp`

```
// 引入 Vulkan 相关头文件和 Torch 库
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <torch/library.h>

// 声明命名空间 at、native、vulkan、ops 中的匿名命名空间
namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

// 使用 Vulkan 相关 API 的命名空间 utils
using namespace api::utils;

// 实现 in-place 操作 zero_，将输入张量 self 全部置为零
Tensor& zero_(at::Tensor& self) {
  // 检查输入张量维度是否不超过4维
  TORCH_CHECK(self.dim() <= 4, "Vulkan zero_ supports up to 4d tensors");

  // 将 self 转换为 Vulkan 张量 v_self
  vTensor& v_self = convert(self);

  // 获取全局 Vulkan 上下文
  api::Context* const context = api::context();

  // 定义管线屏障对象，用于确定如何在命令缓冲区中插入内存屏障
  api::PipelineBarrier pipeline_barrier{};

  // 提交计算作业给 Vulkan 上下文
  context->submit_compute_job(
      // 着色器描述符
      VK_KERNEL(zero),
      // 管线屏障
      pipeline_barrier,
      // 全局工作组大小
      v_self.extents(),
      // 局部工作组大小
      adaptive_work_group_size(v_self.extents()),
      // 栅栏句柄
      VK_NULL_HANDLE,
      // 着色器参数
      v_self.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::READ | api::MemoryAccessType::WRITE));

  // 返回修改后的 self 张量
  return self;
}

// 创建全零张量 zeros，根据输入尺寸 size 创建 Vulkan 张量
Tensor zeros(
    const IntArrayRef size,
    std::optional<ScalarType> dtype,
    std::optional<c10::Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // 检查输入尺寸维度是否不超过4维
  TORCH_CHECK(size.size() <= 4, "Vulkan zeros supports up to 4d tensors");

  // 获取全局 Vulkan 上下文
  api::Context* const context = api::context();

  // 创建输出 Vulkan 张量 v_output
  vTensor v_output{
      context,
      size.vec(),
      api::ScalarType::Float, // 确定数据类型为浮点型
  };

  // 定义管线屏障对象，用于确定如何在命令缓冲区中插入内存屏障
  api::PipelineBarrier pipeline_barrier{};

  // 提交计算作业给 Vulkan 上下文
  context->submit_compute_job(
      // 着色器描述符
      VK_KERNEL(zero),
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
          api::MemoryAccessType::READ | api::MemoryAccessType::WRITE));

  // 将 Vulkan 张量 v_output 转换为普通张量并返回
  return convert(v_output);
}

// 如果定义了 USE_VULKAN_API 宏，则实现 Vulkan 版本的 aten 库
#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::zero_"), TORCH_FN(zero_));
  m.impl(TORCH_SELECTIVE_NAME("aten::zeros"), TORCH_FN(zeros));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
```