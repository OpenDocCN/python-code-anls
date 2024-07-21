# `.\pytorch\aten\src\ATen\native\vulkan\ops\Upsample.cpp`

```
// 包含 Vulkan 上采样相关的头文件和函数声明
#include <ATen/native/UpSample.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/QuantizedFunctions.h>
#include <torch/library.h>

// 命名空间定义开始
namespace at {
namespace native {
namespace vulkan {
namespace ops {

// 使用 Vulkan 操作相关的 API 工具
using namespace api::utils;

// Vulkan 实现的最近邻上采样函数，接受输入张量、输出尺寸、水平和垂直缩放因子
Tensor upsample_nearest2d(
    const Tensor& input_arg,                   // 输入张量
    const IntArrayRef output_sizes,             // 输出尺寸
    const std::optional<double> scales_h,       // 垂直缩放因子（可选）
    const std::optional<double> scales_w) {     // 水平缩放因子（可选）

  // 获取当前 Vulkan 上下文
  api::Context* const context = api::context();

  // 检查输入张量维度和输出尺寸是否符合要求
  TORCH_CHECK(
      (4 == input_arg.sizes().size()) && (2 == output_sizes.size()),
      "Invalid input!");

  // 根据输入张量类型选择是否转换为 Vulkan 张量
  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  // 将 Vulkan 张量转换为 Vulkan 内部表示 vTensor
  const vTensor& v_input = convert(input);
  // 获取输入张量的尺寸信息
  const auto v_input_sizes = v_input.sizes();

  // 创建输出 Vulkan 张量 v_output，根据给定的尺寸和数据类型
  vTensor v_output{
      context,
      {
          v_input_sizes[Layout::Activation4D::batch],
          v_input_sizes[Layout::Activation4D::channels],
          output_sizes[Layout::Parameter::height],
          output_sizes[Layout::Parameter::width],
      },
      v_input.dtype(),
  };

  // 如果输入张量是量化类型，则设置输出张量为量化类型，并复制量化参数
  if (v_input.is_quantized()) {
    v_output.set_is_quantized();
    v_output.set_scale(v_input.get_scale());
    v_output.set_zero_point(v_input.get_zero_point());
  }

  // 定义用于 Vulkan 计算的数据块结构
  const struct Block final {
    uvec3 extents;   // 输出张量的维度信息
    uint32_t fill0;  // 填充字段
    ivec2 iextents;  // 输入张量的宽度和高度减一
    vec2 scale;      // 计算得到的水平和垂直缩放因子
  } block{
      v_output.extents(),  // 获取输出张量的维度信息
      0u,                  // 填充字段初始化为 0
      {
          safe_downcast<int32_t>(
              input_arg.size(Layout::Activation4D::width) - 1),   // 计算输入张量宽度减一
          safe_downcast<int32_t>(
              input_arg.size(Layout::Activation4D::height) - 1),  // 计算输入张量高度减一
      },
      {
          compute_scales_value<float>(
              scales_w,
              v_input_sizes[Layout::Activation4D::width],           // 计算水平缩放因子
              output_sizes[Layout::Parameter::width]),
          compute_scales_value<float>(
              scales_h,
              v_input_sizes[Layout::Activation4D::height],          // 计算垂直缩放因子
              output_sizes[Layout::Parameter::height]),
      },
  };

  // 创建 Vulkan 统一参数缓冲区，用于存储计算所需的参数数据
  api::UniformParamsBuffer params(context, block);
  // 创建 Vulkan 管道屏障对象
  api::PipelineBarrier pipeline_barrier{};

  // 提交 Vulkan 计算作业
  context->submit_compute_job(
      // 使用的着色器描述符，根据输入张量是否量化选择不同的着色器
      v_input.is_quantized() ? VK_KERNEL(quantized_upsample_nearest2d)
                             : VK_KERNEL(upsample_nearest2d),
      // 管道屏障对象
      pipeline_barrier,
      // 全局工作组大小
      v_output.extents(),
      // 自适应的局部工作组大小
      adaptive_work_group_size(v_output.extents()),
      // 线程同步的句柄
      VK_NULL_HANDLE,
      // 着色器参数列表
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),  // 输出张量写入内存的描述
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),  // 输入张量读取内存的描述
      // 参数缓冲区
      params.buffer());

  // 将 Vulkan 张量 v_output 转换为通用张量，并返回
  return convert(v_output);
}

// 后续还有一个双线性上采样函数 upsample_bilinear2d 没有完整显示在此处
    // 获取当前线程的上下文对象
    api::Context* const context = api::context();
    
    // 检查输入参数和输出尺寸的有效性，确保输入有4个维度，输出有2个尺寸
    TORCH_CHECK(
        (4 == input_arg.sizes().size()) && (2 == output_sizes.size()),
        "Invalid input!");
    
    // 根据输入是否使用 Vulkan，选择对应的输入 Tensor
    const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
    // 将选定的输入 Tensor 转换为 Vulkan 张量
    const vTensor& v_input = convert(input);
    
    // 创建 Vulkan 输出张量 v_output，包括批次数、通道数以及指定的高度和宽度
    vTensor v_output{
        context,
        {
            get_dim<Dim4D::Batch>(v_input),
            get_dim<Dim4D::Channel>(v_input),
            output_sizes[Layout::Parameter::height],
            output_sizes[Layout::Parameter::width],
        },
        v_input.dtype(),
    };
    
    // 获取 Vulkan 输出张量的尺寸信息
    const api::utils::uvec3 output_extents = v_output.extents();
    // 定义并初始化名为 Block 的结构体，包含输出尺寸、填充值、输入尺寸和缩放比例
    const struct Block final {
      uvec3 oextents; // 输出张量的尺寸
      uint32_t padding; // 填充值
      ivec2 iextents; // 输入张量的尺寸
      vec2 scale; // 缩放比例
    } block{
        v_output.extents(), // oextents
        0u, // padding
        {
            safe_downcast<int32_t>(get_dim<Dim4D::Width>(input_arg) - 1),
            safe_downcast<int32_t>(get_dim<Dim4D::Height>(input_arg) - 1),
        }, // iextents
        {
            compute_scales_value<float>(
                scales_w,
                get_dim<Dim4D::Width>(input_arg),
                get_dim<Dim4D::Width>(v_output)),
            compute_scales_value<float>(
                scales_h,
                get_dim<Dim4D::Height>(input_arg),
                get_dim<Dim4D::Height>(v_output)),
        }, // scale
    };
    
    // 使用 Block 结构体创建 UniformParamsBuffer 对象 params
    api::UniformParamsBuffer params(context, block);
    // 初始化 PipelineBarrier 对象 pipeline_barrier
    api::PipelineBarrier pipeline_barrier{};
    // 初始化 ShaderInfo 对象 shader_desc
    api::ShaderInfo shader_desc;
    // 根据 align_corners 参数选择相应的 Vulkan 核函数描述符
    if (align_corners) {
      shader_desc = VK_KERNEL(upsample_bilinear2d_align_true);
    } else {
      shader_desc = VK_KERNEL(upsample_bilinear2d_align_false);
    }
    
    // 提交计算作业到 Vulkan 上下文，包括核函数描述符、流水线屏障、全局工作组大小、局部工作组大小、栅栏句柄、
    // 输出图像的内存访问权限、输入图像的内存访问权限以及参数缓冲区
    context->submit_compute_job(
        shader_desc,
        pipeline_barrier,
        output_extents,
        adaptive_work_group_size(output_extents),
        VK_NULL_HANDLE,
        v_output.image(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        params.buffer());
    
    // 将 Vulkan 输出张量转换为一般 Tensor，并返回
    return convert(v_output);
#ifdef USE_VULKAN_API
// 如果定义了 USE_VULKAN_API 宏，则执行以下代码块

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  // 在 Torch 的 aten 库中注册 Vulkan 实现
  m.impl(
      TORCH_SELECTIVE_NAME("aten::upsample_nearest2d"),
      TORCH_FN(upsample_nearest2d));
  // 注册 upsample_nearest2d 函数的 Vulkan 实现

  m.impl(
      TORCH_SELECTIVE_NAME("aten::upsample_bilinear2d"),
      TORCH_FN(upsample_bilinear2d));
  // 注册 upsample_bilinear2d 函数的 Vulkan 实现
}

#endif /* USE_VULKAN_API */
// 结束 USE_VULKAN_API 宏定义条件编译区块

} // namespace ops
// 结束 ops 命名空间

} // namespace vulkan
// 结束 vulkan 命名空间

} // namespace native
// 结束 native 命名空间

} // namespace at
// 结束 at 命名空间
```