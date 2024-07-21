# `.\pytorch\aten\src\ATen\native\vulkan\ops\Pool.cpp`

```py
// 引入 ATen 库的池化相关头文件
#include <ATen/native/Pool.h>
// 引入 ATen Vulkan 操作的通用头文件
#include <ATen/native/vulkan/ops/Common.h>
// 引入 Torch 库的头文件
#include <torch/library.h>

// 定义 at 命名空间
namespace at {
// 定义 native 命名空间
namespace native {
// 定义 Vulkan 命名空间
namespace vulkan {
// 定义 ops 命名空间
namespace ops {
// 匿名命名空间
namespace {

// 使用 api::utils 命名空间
using namespace api::utils;

// Vulkan 实现的自适应平均池化函数，输入 Tensor 和输出尺寸
Tensor adaptive_avg_pool2d(
    const at::Tensor& self_arg,
    const IntArrayRef output_size) {
  // 检查输入 Tensor 的维度是否为 4
  TORCH_CHECK(
      self_arg.dim() == 4,
      "Vulkan adaptive_avg_pool2d expects 4-dimensional input!");

  // 获取 Vulkan 上下文
  api::Context* const context = api::context();

  // 如果输入 Tensor 是 Vulkan 张量，则直接使用；否则转换为 Vulkan 张量
  const Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
  // 将 self 转换为 Vulkan 张量类型
  const vTensor& v_self = convert(self);

  // 创建 Vulkan 输出张量 v_output
  vTensor v_output{
      context,
      {
          self_arg.size(Layout::Activation4D::batch),
          self_arg.size(Layout::Activation4D::channels),
          output_size[Layout::Activation4D::batch],
          output_size[Layout::Activation4D::channels],
      },
      v_self.dtype(),
  };

  // 获取输出张量 v_output 和输入张量 v_self 的尺寸
  const uvec3 v_output_size = v_output.extents();
  const uvec3 v_self_size = v_self.extents();

  // 计算池化的步长 stride
  const vec2 stride{
      static_cast<float>(v_self_size.data[0u]) / v_output_size.data[0u],
      static_cast<float>(v_self_size.data[1u]) / v_output_size.data[1u],
  };

  // 定义并初始化 Vulkan 中的 Block 结构体
  const struct Block final {
    uvec3 extents;
    uint32_t _;
    vec2 kernel;
    vec2 stride;
  } block{
      v_output.extents(),
      0u,
      {
          v_self_size.data[0u] -
              (v_output_size.data[0u] - 1u) * stride.data[0u],
          v_self_size.data[1u] -
              (v_output_size.data[1u] - 1u) * stride.data[1u],
      },
      stride,
  };

  // 创建 Vulkan 统一参数缓冲区 params
  api::UniformParamsBuffer params(context, block);
  // 创建 Vulkan 管道屏障 pipeline_barrier
  api::PipelineBarrier pipeline_barrier{};

  // 提交 Vulkan 计算任务
  context->submit_compute_job(
      // Vulkan 着色器描述符
      VK_KERNEL(adaptive_avg_pool2d),
      // Vulkan 管道屏障
      pipeline_barrier,
      // 全局工作组大小
      v_output.extents(),
      // 本地工作组大小
      adaptive_work_group_size(v_output.extents()),
      // 围栏句柄
      VK_NULL_HANDLE,
      // 着色器参数
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_self.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // 参数缓冲区
      params.buffer());

  // 将 Vulkan 张量 v_output 转换为标准 Tensor 返回
  return convert(v_output);
}

// Vulkan 实现的池化函数，支持不同的池化参数和着色器描述符
Tensor pool2d(
    const Tensor& self_arg,
    const IntArrayRef kernel_arg,
    IntArrayRef stride_arg,
    const IntArrayRef padding_arg,
    const IntArrayRef dilation_arg,
    const bool ceil_mode,
    const api::ShaderInfo& shader_descriptor) {
  // 如果 stride_arg 为空，则使用 kernel_arg 作为默认值
  if (stride_arg.empty()) {
    stride_arg = kernel_arg;
  }

  // 检查 kernel_arg、stride_arg、padding_arg 是否为空
  TORCH_CHECK(!kernel_arg.empty(), "Kernel size cannot be empty!");
  TORCH_CHECK(!stride_arg.empty(), "Stride cannot be empty!");
  TORCH_CHECK(!padding_arg.empty(), "Padding cannot be empty!");

  // 定义静态的标准化函数，用于规范化参数
  static const auto normalize = [](const IntArrayRef parameter) {
    return std::array<int64_t, 2>{
        parameter[0],
        (2 == parameter.size()) ? parameter[1] : parameter[0],
    };
};
  };

  const auto input_size = self_arg.sizes();  // 获取输入张量的尺寸信息
  const auto kernel = normalize(kernel_arg);  // 对核进行归一化处理
  const auto stride = normalize(stride_arg);  // 对步长进行归一化处理
  const auto padding = normalize(padding_arg);  // 对填充进行归一化处理
  const auto dilation = normalize(dilation_arg);  // 对扩展进行归一化处理

  // 计算池化操作后输出的高度
  const int64_t output_height = pooling_output_shape(
      input_size[Layout::Activation4D::height],
      kernel[Layout::Parameter::height],
      padding[Layout::Parameter::height],
      stride[Layout::Parameter::height],
      dilation[Layout::Parameter::height],
      ceil_mode);

  // 计算池化操作后输出的宽度
  const int64_t output_width = pooling_output_shape(
      input_size[Layout::Activation4D::width],
      kernel[Layout::Parameter::width],
      padding[Layout::Parameter::width],
      stride[Layout::Parameter::width],
      dilation[Layout::Parameter::width],
      ceil_mode);

  // 检查池化操作的形状是否合法
  pool2d_shape_check(
      self_arg,
      kernel[Layout::Parameter::height],
      kernel[Layout::Parameter::width],
      stride[Layout::Parameter::height],
      stride[Layout::Parameter::width],
      padding[Layout::Parameter::height],
      padding[Layout::Parameter::width],
      dilation[Layout::Parameter::height],
      dilation[Layout::Parameter::width],
      input_size[Layout::Activation4D::channels],
      input_size[Layout::Activation4D::height],
      input_size[Layout::Activation4D::width],
      output_height,
      output_width,
      self_arg.suggest_memory_format());

  // 获取当前的上下文
  api::Context* const context = api::context();

  // 根据是否使用 Vulkan 进行选择性地获取张量
  const Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
  // 将张量转换为 Vulkan 张量
  const vTensor& v_self = convert(self);

  // 创建输出 Vulkan 张量
  vTensor v_output{
      context,
      {
          input_size[Layout::Activation4D::batch],
          input_size[Layout::Activation4D::channels],
          output_height,
          output_width,
      },
      v_self.dtype(),
  };
  // 如果输入张量是量化的，则设置输出张量也为量化形式，并复制量化参数
  if (v_self.is_quantized()) {
    v_output.set_is_quantized();
    v_output.set_scale(v_self.get_scale());
    v_output.set_zero_point(v_self.get_zero_point());
  }

  // 定义用于统一参数的 API 结构体
  api::UniformParamsBuffer params;
  // 定义最终的 Block 结构体，用于描述块信息
  const struct Block final {
    uvec3 extents;
    int32_t range;
    ivec4 kernel;
    ivec2 stride;
    ivec2 padding;
    // 定义一个名为 dilation 的 ivec2 结构，用于存储膨胀（dilation）参数
    ivec2 dilation;
  } block{
      // 定义一个结构体 block，包含以下成员：
      // - 输出张量的尺寸
      // - 卷积核大小的乘积
      // - 各种尺寸参数：卷积核的宽度和高度，以及输入张量的宽度和高度
      // - 步长参数：卷积的水平和垂直方向上的步长
      // - 填充参数：卷积的水平和垂直方向上的填充
      // - 膨胀参数：卷积的水平和垂直方向上的膨胀
      v_output.extents(),
      safe_downcast<int32_t>(
          kernel[Layout::Parameter::width] * kernel[Layout::Parameter::height]),
      {
          safe_downcast<int32_t>(kernel[Layout::Parameter::width]),
          safe_downcast<int32_t>(kernel[Layout::Parameter::height]),
          safe_downcast<int32_t>(self_arg.size(Layout::Activation4D::width)),
          safe_downcast<int32_t>(self_arg.size(Layout::Activation4D::height)),
      },
      {
          safe_downcast<int32_t>(stride[Layout::Parameter::width]),
          safe_downcast<int32_t>(stride[Layout::Parameter::height]),
      },
      {
          safe_downcast<int32_t>(padding[Layout::Parameter::width]),
          safe_downcast<int32_t>(padding[Layout::Parameter::height]),
      },
      {
          safe_downcast<int32_t>(dilation[Layout::Parameter::width]),
          safe_downcast<int32_t>(dilation[Layout::Parameter::height]),
      },
  };
  // 使用 block 结构体创建 UniformParamsBuffer 对象 params
  params = api::UniformParamsBuffer(context, block);

  // 定义一个空的 PipelineBarrier 对象 pipeline_barrier
  api::PipelineBarrier pipeline_barrier{};

  // 提交计算作业到上下文 context
  context->submit_compute_job(
      // 着色器描述符
      shader_descriptor,
      // 管线障碍
      pipeline_barrier,
      // 全局工作组大小
      v_output.extents(),
      // 自适应工作组大小
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

  // 返回转换后的 v_output
  return convert(v_output);
} // namespace at
} // namespace native
} // namespace vulkan
} // namespace ops
} // namespace ops



#ifdef USE_VULKAN_API

// 在 "aten" 库中注册 Vulkan 实现的池化函数
TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  // 注册自适应平均池化的 Vulkan 实现
  m.impl(
      TORCH_SELECTIVE_NAME("aten::_adaptive_avg_pool2d"),
      TORCH_FN(adaptive_avg_pool2d));
  // 注册平均池化的 Vulkan 实现
  m.impl(TORCH_SELECTIVE_NAME("aten::avg_pool2d"), TORCH_FN(avg_pool2d));
  // 注册最大池化的 Vulkan 实现
  m.impl(TORCH_SELECTIVE_NAME("aten::max_pool2d"), TORCH_FN(max_pool2d));
}

#endif /* USE_VULKAN_API */


这段代码的作用是在使用 Vulkan API 的情况下，在 "aten" 库中注册不同池化操作的 Vulkan 实现。
```