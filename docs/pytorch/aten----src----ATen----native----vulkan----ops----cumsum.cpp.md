# `.\pytorch\aten\src\ATen\native\vulkan\ops\cumsum.cpp`

```py
// 包含Vulkan操作的常用头文件和实用程序
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Utils.h>
// 包含Torch库的头文件
#include <torch/library.h>

// 命名空间嵌套：at -> native -> vulkan -> ops
namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

// 使用api::utils命名空间中的所有内容
using namespace api::utils;

// 设置累积和操作的 Vulkan 核函数参数
void set_cumsum_kernel_params(
    const long long num_dims,                    // 累积和操作的张量维度数
    const long long dim,                         // 累积和操作的维度索引
    const IntArrayRef v_input_sizes,             // 输入张量的尺寸数组
    api::ShaderInfo& shader_descriptor,          // Vulkan着色器描述符
    api::utils::ivec4& input_shader_extents,     // 输入着色器范围
    api::utils::ivec4& early_exit,               // 提前退出标志数组
    api::utils::ivec4& input_dim_stride,         // 输入维度步长数组
    api::utils::ivec4& input_tensor_dims) {      // 输入张量维度数组
  // 当张量维度为1时的情况处理
  if (num_dims == 1) {
    early_exit.data[0u] = 1;                    // 设置提前退出标志的第一个元素为1
    input_dim_stride.data[0u] = 1;              // 设置输入维度步长的第一个元素为1
    shader_descriptor = VK_KERNEL(cumsum_batch_height_width);  // 设置着色器描述符为累积和批次高度宽度核函数
  } else if (num_dims == 2) {
    // 对于高度和宽度维度的情况，可以重用单个着色器，并使用矢量化参数
    shader_descriptor = VK_KERNEL(cumsum_batch_height_width);  // 设置着色器描述符为累积和批次高度宽度核函数
    if (dim == 0) {
      early_exit.data[1u] = 1;                  // 设置提前退出标志的第二个元素为1
      input_dim_stride.data[1u] = 1;            // 设置输入维度步长的第二个元素为1
    } else { // dim == 1
      early_exit.data[0u] = 1;                  // 设置提前退出标志的第一个元素为1
      input_dim_stride.data[0u] = 1;            // 设置输入维度步长的第一个元素为1
    }
  } else if (num_dims == 3) {
    // 对于三维张量的情况
    for (uint32_t i = 0; i < num_dims; i++) {
      input_tensor_dims.data[i + 1] = safe_downcast<int32_t>(v_input_sizes[i]);  // 设置输入张量维度数组
    }
    if (dim == 0) {
      early_exit.data[2u] = 1;                  // 设置提前退出标志的第三个元素为1
      input_dim_stride.data[2u] = 1;            // 设置输入维度步长的第三个元素为1
      shader_descriptor = VK_KERNEL(cumsum_channel);  // 设置着色器描述符为累积和通道核函数
    } else if (dim == 1) {
      // 对于高度和宽度维度的情况，可以重用单个着色器，并使用矢量化参数
      early_exit.data[1u] = 1;                  // 设置提前退出标志的第二个元素为1
      input_dim_stride.data[1u] = 1;            // 设置输入维度步长的第二个元素为1
      shader_descriptor = VK_KERNEL(cumsum_batch_height_width);  // 设置着色器描述符为累积和批次高度宽度核函数
    } else { // dim == 2
      early_exit.data[0u] = 1;                  // 设置提前退出标志的第一个元素为1
      input_dim_stride.data[0u] = 1;            // 设置输入维度步长的第一个元素为1
      shader_descriptor = VK_KERNEL(cumsum_batch_height_width);  // 设置着色器描述符为累积和批次高度宽度核函数
    }
  } else {
    // 假设维度数为4的情况
    for (uint32_t i = 0; i < num_dims; i++) {
      input_tensor_dims.data[i] = safe_downcast<int32_t>(v_input_sizes[i]);  // 设置输入张量维度数组
    }
    if (dim == 1) {
      // 对于四维张量，沿着通道维度扫描的情况，内存布局强制使用不同的着色器算法
      input_shader_extents.data[2u] = v_input_sizes[Layout::Activation4D::batch];  // 设置输入着色器范围的第三个元素
      shader_descriptor = VK_KERNEL(cumsum_channel);  // 设置着色器描述符为累积和通道核函数
    } else {
      // 对于批次、高度和宽度维度的情况，可以重用单个着色器，并使用矢量化参数
      if (dim == 0) {
        early_exit.data[2u] = safe_downcast<int32_t>(std::ceil(v_input_sizes[Layout::Activation4D::channels] / 4.0));  // 设置提前退出标志的第三个元素
        input_dim_stride.data[2u] = safe_downcast<int32_t>(std::ceil(v_input_sizes[Layout::Activation4D::channels] / 4.0));  // 设置输入维度步长的第三个元素
      } else if (dim == 2) {
        early_exit.data[1u] = 1;                // 设置提前退出标志的第二个元素为1
        input_dim_stride.data[1u] = 1;          // 设置输入维度步长的第二个元素为1
      } else { // dim == 3
        early_exit.data[0u] = 1;                // 设置提前退出标志的第一个元素为1
        input_dim_stride.data[0u] = 1;          // 设置输入维度步长的第一个元素为1
      }
      shader_descriptor = VK_KERNEL(cumsum_batch_height_width);  // 设置着色器描述符为累积和批次高度宽度核函数
    }
  }
}

// 累积和操作的入口函数
Tensor cumsum(
    const at::Tensor& input_arg,                    // 引用输入张量
    const int64_t dim_arg,                          // 累加操作的维度
    const std::optional<ScalarType> dtype) {        // 数据类型（可选）

  TORCH_CHECK(
      input_arg.dim() >= 1 && input_arg.dim() <= 4, // 检查输入张量维度是否在1到4之间
      "Vulkan cumsum expects 1 <= input dimension <= 4, Tensor input dimensions ",
      input_arg.dim());                            // 若维度不符合要求，抛出错误信息

  TORCH_CHECK(
      dim_arg < input_arg.dim(),                    // 检查累加维度是否小于输入张量的维度
      "cumsum dim input was ",
      dim_arg,
      " out of range for Tensor input with dimensions ",
      input_arg.dim());                            // 若维度超出范围，抛出错误信息

  int64_t dim = utils::normalize(dim_arg, input_arg.dim());  // 规范化累加维度

  api::Context* const context = api::context();     // 获取 Vulkan API 上下文

  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();  // 将输入张量转换为 Vulkan 张量类型
  const vTensor& v_input = convert(input);          // 转换为 Vulkan 张量对象
  const IntArrayRef v_input_sizes = v_input.sizes(); // 获取 Vulkan 张量的大小信息

  vTensor v_output{                                // 创建 Vulkan 输出张量对象
      context,
      v_input.sizes(),
      v_input.dtype(),
  };

  const api::utils::uvec3 global_workgroup_extents = v_output.extents();  // 获取全局工作组大小
  api::utils::ivec4 input_shader_extents = {       // 输入着色器扩展尺寸，根据输入张量的尺寸进行初始化
      safe_downcast<int32_t>(v_input.extents().data[0u]),
      safe_downcast<int32_t>(v_input.extents().data[1u]),
      safe_downcast<int32_t>(v_input.extents().data[2u]),
      0 // 零填充
  };
  // early_exit 是基于全局工作组位置的退出条件
  api::utils::ivec4 early_exit = {                 // 早期退出条件，根据输入张量的尺寸进行初始化
      safe_downcast<int32_t>(v_input.extents().data[0u]),
      safe_downcast<int32_t>(v_input.extents().data[1u]),
      safe_downcast<int32_t>(v_input.extents().data[2u]),
      0 // 零填充
  };
  // batch/height/width 共享同一个着色器，按每个维度情况使用 input_dim_stride 进行矢量化
  api::utils::ivec4 input_dim_stride = {           // 输入维度步长，初始化为零填充
      0,
      0,
      0,
      0, // 零填充
  };
  api::utils::ivec4 input_tensor_dims = {           // 输入张量维度，初始化为零填充
      0,
      0,
      0,
      0,
  };
  api::ShaderInfo shader_descriptor;
  set_cumsum_kernel_params(                        // 设置累加核函数的参数
      input_arg.dim(),
      dim,
      v_input_sizes,
      shader_descriptor,
      input_shader_extents,
      early_exit,
      input_dim_stride,
      input_tensor_dims);

  const struct Block final {                       // 定义一个结构体 Block
    ivec4 input_shader_extents;
    ivec4 input_tensor_dims;
    ivec4 input_dim_stride;
    ivec4 early_exit;
  } block{                                          // 初始化结构体 Block 的实例
      input_shader_extents, input_tensor_dims, input_dim_stride, early_exit};

  api::UniformParamsBuffer params(context, block);  // 创建 UniformParamsBuffer 对象
  api::PipelineBarrier pipeline_barrier{};          // 创建 PipelineBarrier 对象

  context->submit_compute_job(                     // 提交计算作业到 Vulkan 上下文
      // shader descriptor
      shader_descriptor,
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      global_workgroup_extents,
      // local work group size
      adaptive_work_group_size(global_workgroup_extents),
      // fence handle
      VK_NULL_HANDLE,
      // shader arguments
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());

  return convert(v_output);                        // 将 Vulkan 输出张量转换为普通张量并返回
}
#ifdef USE_VULKAN_API
// 如果定义了 USE_VULKAN_API 宏，则编译以下内容

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  // 在 Torch 的 aten 库中注册 Vulkan 实现，m 是 Torch 模块对象
  m.impl(TORCH_SELECTIVE_NAME("aten::cumsum"), TORCH_FN(cumsum));
}
// 结束 TORCH_LIBRARY_IMPL 块

#endif /* USE_VULKAN_API */
// 结束条件编译块，指明不再编译以下内容

} // namespace
// 结束匿名命名空间

} // namespace ops
// 结束 ops 命名空间

} // namespace vulkan
// 结束 vulkan 命名空间

} // namespace native
// 结束 native 命名空间

} // namespace at
// 结束 at 命名空间
```