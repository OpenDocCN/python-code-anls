# `.\pytorch\aten\src\ATen\native\vulkan\ops\QuantizedTensor.cpp`

```py
// 引入 Vulkan 相关头文件
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/QuantizedFunctions.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <torch/library.h>

// 定义 Vulkan 相关操作的命名空间
namespace at {
namespace native {
namespace vulkan {
namespace ops {

// 使用 Vulkan 相关 API 的实用工具
using namespace api::utils;

// 获取特定量化类型的着色器信息
static api::ShaderInfo get_quantize_per_tensor_shader(
    const c10::ScalarType dtype) {
  switch (dtype) {
    // 对于无符号量化 8 位整数类型，返回对应的 Vulkan 着色器信息
    case c10::ScalarType::QUInt8:
      return VK_KERNEL(quantize_per_tensor_quint8);
    // 对于有符号量化 8 位整数类型，返回对应的 Vulkan 着色器信息
    case c10::ScalarType::QInt8:
      return VK_KERNEL(quantize_per_tensor_qint8);
    // 对于有符号量化 32 位整数类型，返回对应的 Vulkan 着色器信息
    case c10::ScalarType::QInt32:
      return VK_KERNEL(quantize_per_tensor_qint32);
    // 如果 dtype 不支持 Vulkan 量化，抛出异常
    default:
      TORCH_CHECK(
          false,
          "Vulkan quantization currently not supported for dtype ",
          dtype);
  }
}

// 在 Vulkan 环境下对输入张量进行量化
Tensor quantize_per_tensor(
    const at::Tensor& input_arg,
    const double scale,
    const int64_t zero_point,
    const c10::ScalarType dtype) {
  // 获取当前量化类型对应的计算着色器
  api::ShaderInfo compute_shader = get_quantize_per_tensor_shader(dtype);

  // 获取当前 Vulkan 环境的上下文
  api::Context* const context = api::context();

  // 将输入张量转换为 Vulkan 张量，如果已经是 Vulkan 张量则保持不变
  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  // 将 Vulkan 张量转换为 Vulkan 张量结构体
  const vTensor& v_input = convert(input);

  // 创建 Vulkan 输出张量
  vTensor v_output{
      context,
      v_input.sizes(),
      scale,
      zero_point,
      convert_dtype(dtype),
  };

  // 定义 Vulkan 计算任务的块结构
  const struct Block final {
    uvec3 extents;
    uint32_t _;
    float scale;
    float _1;
    int32_t zero_point;
    int32_t _2;
  } block{
      v_output.extents(),
      0u,
      safe_downcast<float>(scale),
      0.0f,
      safe_downcast<int32_t>(zero_point),
      0u,
  };

  // 创建 Vulkan 统一参数缓冲区
  api::UniformParamsBuffer params(context, block);
  // 创建 Vulkan 管线屏障对象
  api::PipelineBarrier pipeline_barrier{};

  // 提交 Vulkan 计算任务
  context->submit_compute_job(
      // 着色器描述符
      compute_shader,
      // 屏障
      pipeline_barrier,
      // 全局工作组大小
      v_input.extents(),
      // 本地工作组大小
      adaptive_work_group_size(v_input.extents()),
      // 回调句柄
      VK_NULL_HANDLE,
      // 着色器参数
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // 参数缓冲区
      params.buffer());

  // 将 Vulkan 输出张量转换为 PyTorch 张量并返回
  return convert_quantized(v_output);
}

// 使用张量的量化参数执行张量量化
Tensor quantize_per_tensor_tensor_qparams(
    const at::Tensor& input_arg,
    const at::Tensor& scale,
    const at::Tensor& zero_point,
    const c10::ScalarType dtype) {
  // 检查量化参数张量 scale 和 zero_point 是否只包含一个元素
  TORCH_CHECK(
      (scale.numel() == 1 && zero_point.numel() == 1),
      "Only 1 element expected in scale and zero_point");
  // 调用 quantize_per_tensor 函数进行张量量化
  return quantize_per_tensor(
      input_arg, scale.item().toDouble(), zero_point.item().toLong(), dtype);
}

// 辅助函数，用于 dequantize 函数中使用 scale 和 zero_point
Tensor dequantize_helper(
    const at::Tensor& input_arg,
    const double scale,
    const int64_t zero_point,
    const c10::ScalarType dtype) {
    // 检查数据类型是否为 kFloat，如果不是则抛出错误信息
    TORCH_CHECK(dtype == kFloat, "Expected type Float");
    
    // 获取当前的 API 上下文
    api::Context* const context = api::context();
    
    // 根据输入参数选择使用 Vulkan 引擎的张量或转换后的 Vulkan 引擎张量作为输入
    const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
    // 将输入张量转换为 Vulkan 引擎张量
    const vTensor& v_input = convert(input);
    
    // 创建输出 Vulkan 引擎张量 v_output，使用当前上下文、与 v_input 相同的大小和 kFloat 类型
    vTensor v_output{
        context,
        v_input.sizes(),
        api::kFloat,
    };
    
    // 定义名为 Block 的结构体，用于存储后续需要使用的参数
    const struct Block final {
      uvec3 extents;         // 张量的尺寸
      uint32_t _;            // 未使用的占位符
      float scale;           // 缩放因子
      float _1;              // 未使用的占位符
      int32_t zero_point;    // 零点值
      int32_t _2;            // 未使用的占位符
    } block{
        v_output.extents(),                   // 使用 v_output 的尺寸作为 extents
        0u,                                   // 初始化未使用的占位符
        safe_downcast<float>(scale),          // 使用安全转换后的 scale 作为缩放因子
        0.0f,                                 // 初始化未使用的占位符
        safe_downcast<int32_t>(zero_point),   // 使用安全转换后的 zero_point 作为零点值
        0u                                    // 初始化未使用的占位符
    };
    
    // 创建 UniformParamsBuffer 对象 params，用于存储块 block 的参数
    api::UniformParamsBuffer params(context, block);
    
    // 创建 PipelineBarrier 对象 pipeline_barrier，用于管线屏障控制
    api::PipelineBarrier pipeline_barrier{};
    
    // 提交计算任务到上下文，包括：
    // - 使用的着色器描述符 VK_KERNEL(dequantize)
    // - 管线屏障 pipeline_barrier
    // - 全局工作组大小 v_input.extents()
    // - 自适应的局部工作组大小 adaptive_work_group_size(v_input.extents())
    // - 等待句柄 VK_NULL_HANDLE
    // - 着色器参数：输出图像 v_output.image、输入图像 v_input.image
    // - 参数缓冲区 params.buffer()
    context->submit_compute_job(
        VK_KERNEL(dequantize),
        pipeline_barrier,
        v_input.extents(),
        adaptive_work_group_size(v_input.extents()),
        VK_NULL_HANDLE,
        v_output.image(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        params.buffer());
    
    // 将 Vulkan 引擎张量 v_output 转换为普通张量，并返回
    return convert(v_output);
// 关闭 ops 命名空间
} // namespace ops

// 关闭 vulkan 命名空间
} // namespace vulkan

// 关闭 native 命名空间
} // namespace native

// 关闭 at 命名空间
} // namespace at
```