# `.\pytorch\aten\src\ATen\native\vulkan\ops\Lerp.cpp`

```
// 引入Vulkan操作的通用头文件
#include <ATen/native/vulkan/ops/Common.h>
// 引入Torch库头文件
#include <torch/library.h>

// 定义在命名空间at::native::vulkan::ops内部的匿名命名空间
namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

// 使用api::utils命名空间
using namespace api::utils;

// 检查元素操作的输入张量是否符合要求
void check_inputs_elementwise_op(const Tensor& input1, const Tensor& input2) {
  // 检查通道维度是否相等
  TORCH_CHECK(
      get_dim<Dim4D::Channel>(input1) == get_dim<Dim4D::Channel>(input2),
      "Vulkan elementwise ops require channel dimension to be equal!");
  
  // 如果批次维度不相等，要求通道维度是4的倍数以便沿批次维度广播
  if (get_dim<Dim4D::Batch>(input1) != get_dim<Dim4D::Batch>(input2)) {
    TORCH_CHECK(
        get_dim<Dim4D::Channel>(input1) % 4 == 0,
        "Vulkan elementwise ops require channel to be a multiple of 4 to broadcast along batch dimension!")
  }

  // 获取输入张量的高度和宽度
  const uint32_t input1_h = get_dim<Dim4D::Height>(input1);
  const uint32_t input1_w = get_dim<Dim4D::Width>(input1);
  const uint32_t input2_h = get_dim<Dim4D::Height>(input2);
  const uint32_t input2_w = get_dim<Dim4D::Width>(input2);

  // 如果高度不相等，进行广播维度检查
  const std::string broadcast_error_msg =
      "Incompatible input dimensions for broadcasting for Vulkan elementwise op!";
  if (input1_h != input2_h) {
    if (input1_h > input2_h) {
      TORCH_CHECK(input2_h == 1, broadcast_error_msg);
      TORCH_CHECK(input2_w == input1_w || input2_w == 1, broadcast_error_msg);
    } else if (input2_h > input1_h) {
      TORCH_CHECK(input1_h == 1, broadcast_error_msg);
      TORCH_CHECK(input1_w == input2_w || input1_w == 1, broadcast_error_msg);
    }
  } else if (input1_w != input2_w) {
    if (input1_w > input2_w) {
      TORCH_CHECK(input2_w == 1, broadcast_error_msg);
    } else if (input2_w > input1_w) {
      TORCH_CHECK(input1_h == 1, broadcast_error_msg);
    }
  }
}

// 执行线性插值运算，处理标量参数
Tensor _lerp_scalar(
    const Tensor& start_arg,
    const Tensor& end_arg,
    const Scalar& weight_arg) {
  // 检查输入张量是否符合元素操作的要求
  check_inputs_elementwise_op(start_arg, end_arg);
  
  // 获取当前上下文
  api::Context* const context = api::context();

  // 根据是否为Vulkan张量，选择转换或使用原始张量
  const Tensor start = start_arg.is_vulkan() ? start_arg : start_arg.vulkan();
  const vTensor& v_start = convert(start);

  const Tensor end = end_arg.is_vulkan() ? end_arg : end_arg.vulkan();
  const vTensor& v_end = convert(end);

  // 创建Vulkan张量用于输出
  vTensor v_output{
      context,
      v_start.sizes(),
      v_start.dtype(),
  };

  // 获取标量参数的float值
  const float weight = weight_arg.to<float>();

  // 定义结构体Block，存储扩展和填充信息
  const struct Block final {
    uvec3 extents;
    uint32_t fill_0;
    uvec3 input1_extents;
    uint32_t fill_1;
    uvec3 input2_extents;
    // 定义浮点型变量 weight
    float weight;
  } block{
      // 使用 v_output 的尺寸创建输出对象的范围描述
      v_output.extents(),
      // 初始偏移量设置为 0
      0u,
      // 使用 v_start 的尺寸创建起始对象的范围描述
      v_start.extents(),
      // 初始偏移量设置为 0
      0u,
      // 使用 v_end 的尺寸创建结束对象的范围描述
      v_end.extents(),
      // 设置权重为 weight
      weight,
  };

  // 创建 UniformParamsBuffer 对象 params，用于存储块数据
  api::UniformParamsBuffer params(context, block);
  // 创建 PipelineBarrier 对象 pipeline_barrier，用于指定管线屏障
  api::PipelineBarrier pipeline_barrier{};

  // 提交计算作业给上下文 context
  context->submit_compute_job(
      // 使用 lerp_scalar 内核进行计算
      VK_KERNEL(lerp_scalar),
      // 应用管线屏障
      pipeline_barrier,
      // 全局工作组大小为 v_output 的尺寸
      v_output.extents(),
      // 使用自适应的工作组大小，基于 v_output 的尺寸
      adaptive_work_group_size(v_output.extents()),
      // 等待句柄为空
      VK_NULL_HANDLE,
      // 设置 shader 参数
      v_output.image(
          pipeline_barrier,
          // 指定计算阶段
          api::PipelineStage::COMPUTE,
          // 写内存访问权限
          api::MemoryAccessType::WRITE),
      // 传递 v_start 图像数据给 shader
      v_start.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // 传递 v_end 图像数据给 shader
      v_end.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // 使用 params 缓冲区传递参数
      params.buffer());

  // 转换并返回 v_output 的数据
  return convert(v_output);
// 结束上一个函数，开始定义名为 _lerp_scalar_ 的函数，该函数用于在 Vulkan 引擎中执行标量线性插值操作
Tensor& _lerp_scalar_(
    // 修改 self_arg 引用的 Tensor 对象，end_arg 和 weight_arg 分别是标量线性插值的终止张量和权重标量
    Tensor& self_arg,
    const Tensor& end_arg,
    const Scalar& weight_arg) {
  // 检查输入张量以确保元素操作的一致性
  check_inputs_elementwise_op(self_arg, end_arg);

  // 断言当前 self_arg 是 Vulkan 引擎支持的张量类型
  TORCH_CHECK(
      self_arg.is_vulkan(),
      "Vulkan: In-place operator is only supported on Vulkan tensors.");

  // 获取当前 Vulkan 引擎的上下文对象
  api::Context* const context = api::context();

  // 将 self_arg 转换为 Vulkan 引擎中的张量对象 v_self
  vTensor& v_self = convert(self_arg);

  // 如果 end_arg 是 Vulkan 引擎支持的张量类型，则直接使用 end_arg，否则将其转换为 Vulkan 引擎张量
  const Tensor end = end_arg.is_vulkan() ? end_arg : end_arg.vulkan();
  const vTensor& v_end = convert(end);

  // 将 weight_arg 转换为 float 类型的权重值
  const float weight = weight_arg.to<float>();

  // 定义包含张量尺寸和权重的 Block 结构体
  const struct Block final {
    uvec3 extents;           // 自身张量的尺寸
    uint32_t fill_0;         // 填充字段
    uvec3 input_extents;     // end 张量的尺寸
    float alpha;             // 权重值
  } block{
      v_self.extents(),      // 获取 self_arg 的尺寸
      0u,                    // 填充值为 0
      v_end.extents(),       // 获取 end 张量的尺寸
      weight,                // 使用给定的权重值
  };

  // 使用 Block 结构体创建 UniformParamsBuffer 对象，用于在 Vulkan 引擎中传递参数
  api::UniformParamsBuffer params(context, block);

  // 创建一个空的 PipelineBarrier 对象
  api::PipelineBarrier pipeline_barrier{};

  // 在 Vulkan 上下文中提交计算任务，执行标量线性插值操作
  context->submit_compute_job(
      // Vulkan 内核描述符，指定要执行的计算内核
      VK_KERNEL(lerp_scalar_),
      // 管道屏障，确保前后计算的内存访问正确
      pipeline_barrier,
      // 全局工作组大小，与张量的尺寸相同
      v_self.extents(),
      // 适应性工作组大小，根据张量尺寸调整
      adaptive_work_group_size(v_self.extents()),
      // 等待句柄，暂不等待
      VK_NULL_HANDLE,
      // 着色器参数，读写自身和 end 张量的内存访问权限
      v_self.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::READ | api::MemoryAccessType::WRITE),
      v_end.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // 参数缓冲区，传递 Block 结构体的参数
      params.buffer());

  // 返回已修改的 self_arg 引用
  return self_arg;
}

// 定义名为 _lerp_tensor 的函数，执行张量之间的线性插值操作
Tensor _lerp_tensor(
    // 输入参数 start_arg、end_arg 和 weight_arg 分别为起始张量、终止张量和权重张量
    const Tensor& start_arg,
    const Tensor& end_arg,
    const Tensor& weight_arg) {
  // 检查输入张量，确保元素操作的一致性
  check_inputs_elementwise_op(start_arg, end_arg);
  check_inputs_elementwise_op(start_arg, weight_arg);

  // 获取当前 Vulkan 引擎的上下文对象
  api::Context* const context = api::context();

  // 如果 start_arg 是 Vulkan 引擎支持的张量类型，则直接使用 start_arg，否则将其转换为 Vulkan 引擎张量
  const Tensor start = start_arg.is_vulkan() ? start_arg : start_arg.vulkan();
  const vTensor& v_start = convert(start);

  // 如果 end_arg 是 Vulkan 引擎支持的张量类型，则直接使用 end_arg，否则将其转换为 Vulkan 引擎张量
  const Tensor end = end_arg.is_vulkan() ? end_arg : end_arg.vulkan();
  const vTensor& v_end = convert(end);

  // 如果 weight_arg 是 Vulkan 引擎支持的张量类型，则直接使用 weight_arg，否则将其转换为 Vulkan 引擎张量
  const Tensor weight =
      weight_arg.is_vulkan() ? weight_arg : weight_arg.vulkan();
  const vTensor& v_weight = convert(weight_arg);

  // 创建一个新的 Vulkan 引擎张量 v_output，用于存储输出结果
  vTensor v_output{
      context,              // 使用当前 Vulkan 上下文对象
      v_start.sizes(),      // 使用 start_arg 的尺寸作为输出张量的尺寸
      v_start.dtype(),      // 使用 start_arg 的数据类型作为输出张量的数据类型
  };

  // 定义包含张量尺寸的 Block 结构体，用于传递给 Vulkan 引擎的计算任务
  const struct Block final {
    uvec3 extents;         // start 张量的尺寸
    uint32_t fill_0;       // 填充字段
    uvec3 input1_extents;  // end 张量的尺寸
    uint32_t fill_1;       // 填充字段
    uvec3 input2_extents;  // weight 张量的尺寸
    uint32_t fill_2;       // 填充字段
    uvec3 input3_extents;  // 输出张量的尺寸
    // 声明一个无符号32位整数变量，用于填充结构体中未使用的部分
    uint32_t fill_3;
  } block{
      // 定义一个结构体 block，初始化其成员
      v_output.extents(),    // 访问 v_output 对象的 extents() 方法，获取输出图像的大小
      0u,                    // 初始化第一个成员为 0
      v_start.extents(),     // 访问 v_start 对象的 extents() 方法，获取起始图像的大小
      0u,                    // 初始化第二个成员为 0
      v_end.extents(),       // 访问 v_end 对象的 extents() 方法，获取结束图像的大小
      0u,                    // 初始化第三个成员为 0
      v_weight.extents(),    // 访问 v_weight 对象的 extents() 方法，获取权重图像的大小
      0u,                    // 初始化第四个成员为 0
  };

  // 使用 context 和 block 创建 UniformParamsBuffer 对象 params
  api::UniformParamsBuffer params(context, block);

  // 创建空的 PipelineBarrier 对象 pipeline_barrier
  api::PipelineBarrier pipeline_barrier{};

  // 提交计算任务给 context
  context->submit_compute_job(
      // 指定计算任务使用的着色器描述符
      VK_KERNEL(lerp),
      // 指定计算任务的管线屏障
      pipeline_barrier,
      // 指定全局工作组大小
      v_output.extents(),
      // 计算并返回适应性工作组大小
      adaptive_work_group_size(v_output.extents()),
      // 指定 fence 句柄
      VK_NULL_HANDLE,
      // 指定着色器参数
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_start.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_end.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_weight.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // 指定参数缓冲区
      params.buffer());

  // 将 v_output 转换为期望的返回类型并返回
  return convert(v_output);
} // 结束命名空间 at

Tensor& _lerp_tensor_(
    Tensor& self_arg,
    const Tensor& end_arg,
    const Tensor& weight_arg) {
  check_inputs_elementwise_op(self_arg, end_arg);  // 检查输入张量的元素操作
  check_inputs_elementwise_op(self_arg, weight_arg);  // 检查输入张量的元素操作

  TORCH_CHECK(
      self_arg.is_vulkan(),
      "Vulkan: In-place operator is only supported on Vulkan tensors.");  // 检查是否为 Vulkan 张量，否则抛出错误信息

  api::Context* const context = api::context();  // 获取 Vulkan API 的上下文

  vTensor& v_self = convert(self_arg);  // 将 self_arg 转换为 vTensor

  const Tensor end = end_arg.is_vulkan() ? end_arg : end_arg.vulkan();  // 如果 end_arg 是 Vulkan 张量则直接使用，否则转换为 Vulkan 张量
  const vTensor& v_end = convert(end_arg);  // 将 end_arg 转换为 vTensor

  const Tensor weight =
      weight_arg.is_vulkan() ? weight_arg : weight_arg.vulkan();  // 如果 weight_arg 是 Vulkan 张量则直接使用，否则转换为 Vulkan 张量
  const vTensor& v_weight = convert(weight_arg);  // 将 weight_arg 转换为 vTensor

  // 定义名为 Block 的结构体
  const struct Block final {
    uvec3 extents;
    uint32_t fill_0;
    uvec3 input1_extents;
    uint32_t fill_1;
    uvec3 input2_extents;
    uint32_t fill_2;
  } block{
      v_self.extents(),  // 使用 v_self 的尺寸作为 extents
      0u,  // 填充值为 0
      v_end.extents(),  // 使用 v_end 的尺寸作为 input1_extents
      0u,  // 填充值为 0
      v_weight.extents(),  // 使用 v_weight 的尺寸作为 input2_extents
      0u,  // 填充值为 0
  };

  api::UniformParamsBuffer params(context, block);  // 使用 Vulkan API 创建 UniformParamsBuffer，传入 context 和 block

  api::PipelineBarrier pipeline_barrier{};  // 创建一个空的 PipelineBarrier 对象

  context->submit_compute_job(
      // shader descriptor
      VK_KERNEL(lerp_),  // 使用 lerp_ 作为 Vulkan kernel 的描述符
      // pipeline barrier
      pipeline_barrier,  // 使用前面创建的 pipeline_barrier
      // global work group size
      v_self.extents(),  // 使用 v_self 的尺寸作为全局工作组大小
      // local work group size
      adaptive_work_group_size(v_self.extents()),  // 根据 v_self 的尺寸计算自适应的本地工作组大小
      // fence handle
      VK_NULL_HANDLE,  // 使用空的 fence 句柄
      // shader arguments
      v_self.image(  // 获取 v_self 的图像，指定访问模式为读写
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::READ | api::MemoryAccessType::WRITE),
      v_end.image(pipeline_barrier, api::PipelineStage::COMPUTE),  // 获取 v_end 的图像，指定访问模式为计算阶段
      v_weight.image(pipeline_barrier, api::PipelineStage::COMPUTE),  // 获取 v_weight 的图像，指定访问模式为计算阶段
      // params buffer
      params.buffer());  // 使用之前创建的 params 缓冲区作为参数缓冲区

  return self_arg;  // 返回原始的 self_arg
}

Tensor lerp_scalar(
    const Tensor& start,
    const Tensor& end,
    const Scalar& weight) {
  return _lerp_scalar(start, end, weight);  // 调用 _lerp_scalar 函数并返回结果
}

Tensor& lerp_scalar_(Tensor& self, const Tensor& end, const Scalar& weight) {
  return _lerp_scalar_(self, end, weight);  // 调用 _lerp_scalar_ 函数并返回结果
}

Tensor lerp_tensor(
    const Tensor& start,
    const Tensor& end,
    const Tensor& weight) {
  if (weight.sizes().size() == 0) {  // 如果 weight 的维度大小为 0
    return _lerp_scalar(start, end, weight.item<float>());  // 调用 _lerp_scalar 函数并返回结果
  }
  return _lerp_tensor(start, end, weight);  // 调用 _lerp_tensor 函数并返回结果
}

Tensor& lerp_tensor_(Tensor& self, const Tensor& end, const Tensor& weight) {
  if (weight.sizes().size() == 0) {  // 如果 weight 的维度大小为 0
    return _lerp_scalar_(self, end, weight.item<float>());  // 调用 _lerp_scalar_ 函数并返回结果
  }
  return _lerp_tensor_(self, end, weight);  // 调用 _lerp_tensor_ 函数并返回结果
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::lerp.Scalar"), TORCH_FN(lerp_scalar));  // 注册 Vulkan 实现的标量插值函数
  m.impl(TORCH_SELECTIVE_NAME("aten::lerp_.Scalar"), TORCH_FN(lerp_scalar_));  // 注册 Vulkan 实现的标量原位插值函数
  m.impl(TORCH_SELECTIVE_NAME("aten::lerp.Tensor"), TORCH_FN(lerp_tensor));  // 注册 Vulkan 实现的张量插值函数
  m.impl(TORCH_SELECTIVE_NAME("aten::lerp_.Tensor"), TORCH_FN(lerp_tensor_));  // 注册 Vulkan 实现的张量原位插值函数
}

#endif /* USE_VULKAN_API */

} // 结束命名空间 vulkan
} // 结束命名空间 native
} // 结束命名空间 ops
} // 结束命名空间 at
```