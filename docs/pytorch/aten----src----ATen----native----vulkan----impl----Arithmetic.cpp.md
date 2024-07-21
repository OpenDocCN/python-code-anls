# `.\pytorch\aten\src\ATen\native\vulkan\impl\Arithmetic.cpp`

```
namespace at {
namespace native {
namespace vulkan {
namespace arithmetic {

// 根据操作类型选择相应的 Vulkan 着色器并返回其信息
api::ShaderInfo get_shader(const OpType type) {
  switch (type) {
    case OpType::ADD:
      return VK_KERNEL(add);
    case OpType::SUB:
      return VK_KERNEL(sub);
    case OpType::MUL:
      return VK_KERNEL(mul);
    case OpType::DIV:
      return VK_KERNEL(div);
    case OpType::FLOOR_DIV:
      return VK_KERNEL(floor_divide);
    case OpType::POW:
      return VK_KERNEL(pow);
  }
  // 如果操作类型无效，抛出异常
  VK_THROW("Invalid OpType");
}

// 定义包含着色器执行所需参数的结构体
struct Params final {
  api::utils::ivec4 outputSizes;    // 输出张量的尺寸信息
  api::utils::ivec4 input1Sizes;    // 输入张量1的尺寸信息
  api::utils::ivec4 input2Sizes;    // 输入张量2的尺寸信息
  float alpha;                      // 操作的 alpha 值
};

// 记录 Vulkan 计算操作的执行过程
void record_op(
    api::Context* const context,    // Vulkan 执行上下文
    const api::ShaderInfo& compute_shader,    // 计算着色器信息
    vTensor& v_in1,    // 输入张量1
    vTensor& v_in2,    // 输入张量2
    vTensor& v_dst,    // 输出张量
    const float alpha) {    // 操作的 alpha 值
  // 计算全局和局部工作组大小
  api::utils::uvec3 global_size = v_dst.extents();
  api::utils::uvec3 local_size = adaptive_work_group_size(global_size);

  // 初始化操作所需的参数块
  Params block{
      api::utils::make_ivec4(
          {dim_at<Dim4D::Width>(v_dst),
           dim_at<Dim4D::Height>(v_dst),
           dim_at<Dim4D::Channel>(v_dst),
           dim_at<Dim4D::Batch>(v_dst)}),
      api::utils::make_ivec4(
          {dim_at<Dim4D::Width>(v_in1),
           dim_at<Dim4D::Height>(v_in1),
           dim_at<Dim4D::Channel>(v_in1),
           dim_at<Dim4D::Batch>(v_in1)}),
      api::utils::make_ivec4(
          {dim_at<Dim4D::Width>(v_in2),
           dim_at<Dim4D::Height>(v_in2),
           dim_at<Dim4D::Channel>(v_in2),
           dim_at<Dim4D::Batch>(v_in2)}),
      alpha,
  };

  // 创建 Uniform 参数缓冲区
  api::UniformParamsBuffer params(context, block);
  // 创建管线屏障
  api::PipelineBarrier pipeline_barrier{};

  // 提交 Vulkan 计算任务
  context->submit_compute_job(
      // 着色器描述符
      compute_shader,
      // 管线屏障
      pipeline_barrier,
      // 全局工作组大小
      global_size,
      // 局部工作组大小
      local_size,
      // 栅栏句柄
      VK_NULL_HANDLE,
      // 着色器参数
      v_dst.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_in1.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_in2.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // 参数缓冲区
      params.buffer());
}

} // namespace arithmetic
} // namespace vulkan
} // namespace native
} // namespace at
```