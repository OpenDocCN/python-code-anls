# `.\pytorch\aten\src\ATen\native\vulkan\ops\UnaryOp.cpp`

```
namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {
using namespace api::utils;

// 实现针对 Vulkan 张量的一元操作，通过 Vulkan API 执行计算任务
Tensor unary_op(
    const Tensor& self_arg,
    const api::ShaderInfo& shader_descriptor) {
  
  // 获取当前 Vulkan API 上下文
  api::Context* const context = api::context();

  // 如果输入张量不是 Vulkan 张量，则转换为 Vulkan 张量
  const Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
  const vTensor& v_self = convert(self);

  // 创建输出 Vulkan 张量 v_output，具有与输入张量相同的大小和数据类型
  vTensor v_output{
      context,
      v_self.sizes(),
      v_self.dtype(),
  };

  // 定义 Vulkan 计算任务的参数块
  const struct Block final {
    uvec3 extents;
    uint32_t fill0;
  } block{
      v_self.extents(),
      0,
  };

  // 创建用于参数传递的 Uniform 参数缓冲区
  api::UniformParamsBuffer params(context, block);
  // 创建管线障碍对象
  api::PipelineBarrier pipeline_barrier{};

  // 提交 Vulkan 计算作业
  context->submit_compute_job(
      // 着色器描述符
      shader_descriptor,
      // 管线障碍
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
      v_self.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // 参数缓冲区
      params.buffer());

  // 将 Vulkan 张量 v_output 转换为普通张量并返回
  return convert(v_output);
}

// 实现原地一元操作，针对 Vulkan 张量，通过 Vulkan API 执行计算任务
Tensor& unary_op_(Tensor& self_arg, const api::ShaderInfo& shader_descriptor) {
  // 检查输入张量是否为 Vulkan 张量，否则抛出错误信息
  TORCH_CHECK(
      self_arg.is_vulkan(),
      "Vulkan: In-place operator is only supported on Vulkan tensors.");

  // 获取当前 Vulkan API 上下文
  api::Context* const context = api::context();

  // 将输入张量转换为 Vulkan 张量
  vTensor& v_self = convert(self_arg);

  // 定义 Vulkan 计算任务的参数块
  const struct Block final {
    uvec3 extents;
    uint32_t fill0;
  } block{
      v_self.extents(),
      0,
  };

  // 创建用于参数传递的 Uniform 参数缓冲区
  api::UniformParamsBuffer params(context, block);
  // 创建管线障碍对象
  api::PipelineBarrier pipeline_barrier{};

  // 提交 Vulkan 计算作业
  context->submit_compute_job(
      // 着色器描述符
      shader_descriptor,
      // 管线障碍
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
          api::MemoryAccessType::READ | api::MemoryAccessType::WRITE),
      // 参数缓冲区
      params.buffer());

  // 返回原地操作后的 Vulkan 张量
  return self_arg;
}

// 实现指数函数的计算，通过调用一元操作函数 unary_op
Tensor exp(const Tensor& self_arg) {
  return unary_op(self_arg, VK_KERNEL(exp));
}

// 实现指数函数的原地计算，通过调用原地一元操作函数 unary_op_
Tensor& exp_(Tensor& self_arg) {
  return unary_op_(self_arg, VK_KERNEL(exp_inplace));
}

// 实现平方根函数的计算，通过调用一元操作函数 unary_op
Tensor sqrt(const Tensor& self_arg) {
  return unary_op(self_arg, VK_KERNEL(sqrt));
}

// 实现平方根函数的原地计算，通过调用原地一元操作函数 unary_op_
Tensor& sqrt_(Tensor& self_arg) {
  return unary_op_(self_arg, VK_KERNEL(sqrt_inplace));
}

// 实现对数函数的计算，通过调用一元操作函数 unary_op
Tensor log(const Tensor& self_arg) {
  return unary_op(self_arg, VK_KERNEL(log));
}
// 实现了对输入张量进行 in-place 操作的对数运算，并返回修改后的张量引用
Tensor& log_(Tensor& self_arg) {
  return unary_op_(self_arg, VK_KERNEL(log_inplace));
}

#ifdef USE_VULKAN_API

// 在 Vulkan 后端实现的 ATen 库中注册了对数、指数、平方根等函数的 Vulkan 实现
TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  // 注册 aten::exp 函数的 Vulkan 实现
  m.impl(TORCH_SELECTIVE_NAME("aten::exp"), TORCH_FN(exp));
  // 注册 aten::exp_ 函数的 Vulkan 实现
  m.impl(TORCH_SELECTIVE_NAME("aten::exp_"), TORCH_FN(exp_));
  // 注册 aten::sqrt 函数的 Vulkan 实现
  m.impl(TORCH_SELECTIVE_NAME("aten::sqrt"), TORCH_FN(sqrt));
  // 注册 aten::sqrt_ 函数的 Vulkan 实现
  m.impl(TORCH_SELECTIVE_NAME("aten::sqrt_"), TORCH_FN(sqrt_));
  // 注册 aten::log 函数的 Vulkan 实现
  m.impl(TORCH_SELECTIVE_NAME("aten::log"), TORCH_FN(log));
  // 注册 aten::log_ 函数的 Vulkan 实现
  m.impl(TORCH_SELECTIVE_NAME("aten::log_"), TORCH_FN(log_));
}

#endif /* USE_VULKAN_API */

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
```