# `.\pytorch\aten\src\ATen\native\vulkan\ops\Softmax.cpp`

```py
// 引入 Vulkan 操作的常用头文件和工具函数
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

// 设置 softmax 算法的内核参数
void set_softmax_kernel_params(
    // 输入张量的维度数
    const long long num_dims,
    // 执行 softmax 的维度
    const long long softmax_dim,
    // 输入张量各维度大小的引用
    const IntArrayRef v_input_sizes,
    // Vulkan 着色器描述符
    api::ShaderInfo& shader_descriptor,
    // 输入张量的扩展信息
    api::utils::ivec4& input_shader_extents,
    // 早期退出标志向量
    api::utils::ivec4& early_exit,
    // 输入张量每维的步长
    api::utils::ivec4& input_dim_stride,
    // 输入张量维度信息
    api::utils::ivec4& input_tensor_dims) {
  
  // 如果张量维度为 1
  if (num_dims == 1) {
    // 设置早期退出标志和步长
    early_exit.data[0u] = 1;
    input_dim_stride.data[0u] = 1;
    // 选择适合的 Vulkan 内核函数
    shader_descriptor = VK_KERNEL(softmax_batch_height_width);
  
  // 如果张量维度为 2
  } else if (num_dims == 2) {
    // 针对高度、宽度维度，可以重用单个着色器，使用矢量化参数
    if (softmax_dim == 0) {
      early_exit.data[1u] = 1;
      input_dim_stride.data[1u] = 1;
      shader_descriptor = VK_KERNEL(softmax_batch_height_width);
    } else { // softmax_dim == 1
      early_exit.data[0u] = 1;
      input_dim_stride.data[0u] = 1;
      shader_descriptor = VK_KERNEL(softmax_batch_height_width);
    }
  
  // 如果张量维度为 3
  } else if (num_dims == 3) {
    // 针对高度、宽度维度，可以重用单个着色器，使用矢量化参数
    for (uint32_t i = 0; i < num_dims; i++) {
      input_tensor_dims.data[i + 1] = safe_downcast<int32_t>(v_input_sizes[i]);
    }
    if (softmax_dim == 0) {
      early_exit.data[2u] = 1;
      input_dim_stride.data[2u] = 1;
      shader_descriptor = VK_KERNEL(softmax_channel);
    } else if (softmax_dim == 1) {
      early_exit.data[1u] = 1;
      input_dim_stride.data[1u] = 1;
      shader_descriptor = VK_KERNEL(softmax_batch_height_width);
    } else { // softmax_dim == 2
      early_exit.data[0u] = 1;
      input_dim_stride.data[0u] = 1;
      shader_descriptor = VK_KERNEL(softmax_batch_height_width);
    }
  
  // 如果张量维度为 4
  } else {
    // 针对批次、高度、宽度维度，可以重用单个着色器，使用矢量化参数
    for (uint32_t i = 0; i < num_dims; i++) {
      input_tensor_dims.data[i] = safe_downcast<int32_t>(v_input_sizes[i]);
    }
    if (softmax_dim == 1) {
      // 对于 4 维张量，沿着通道维进行 softmax，内存布局需要不同的着色器算法
      input_shader_extents.data[2u] =
          v_input_sizes[Layout::Activation4D::batch];
      shader_descriptor = VK_KERNEL(softmax_channel);
    } else {
      // 如果 softmax_dim 不等于 0，进入该分支
      if (softmax_dim == 0) {
        // 根据输入数据的通道数计算并向上取整得到新的值，存储在 early_exit 和 input_dim_stride 的第三个位置上
        early_exit.data[2u] = safe_downcast<int32_t>(
            std::ceil(v_input_sizes[Layout::Activation4D::channels] / 4.0));
        input_dim_stride.data[2u] = safe_downcast<int32_t>(
            std::ceil(v_input_sizes[Layout::Activation4D::channels] / 4.0));
      } else if (softmax_dim == 2) {
        // 如果 softmax_dim 等于 2，设置 early_exit 和 input_dim_stride 的第二个位置为 1
        early_exit.data[1u] = 1;
        input_dim_stride.data[1u] = 1;
      } else { // 否则，即 softmax_dim == 3
        // 设置 early_exit 和 input_dim_stride 的第一个位置为 1
        early_exit.data[0u] = 1;
        input_dim_stride.data[0u] = 1;
      }
      // 将 shader_descriptor 设置为 softmax_batch_height_width 内核的描述符
      shader_descriptor = VK_KERNEL(softmax_batch_height_width);
    }
  }
}



Tensor softmax_internal(
    const at::Tensor& input_arg,  // 输入张量的引用
    const int64_t dim_arg,  // 软最大化操作的维度
    const bool half_to_float) {  // 是否将半精度转换为单精度

  TORCH_CHECK(
      input_arg.dim() >= 1 && input_arg.dim() <= 4,  // 检查输入张量维度范围
      "Vulkan softmax expects 1,2,3 or 4-dimensional input!");  // 错误消息提示

  int64_t dim = utils::normalize(dim_arg, input_arg.dim());  // 规范化维度值
  TORCH_CHECK(
      dim >= 0 && dim < input_arg.dim(),  // 检查规范化后的维度是否在有效范围内
      "Softmax dim input was ",
      dim,
      " out of range for Tensor input with dimensions ",
      input_arg.dim());  // 错误消息提示

  api::Context* const context = api::context();  // 获取 Vulkan 上下文

  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();  // 将输入张量转换为 Vulkan 张量
  const vTensor& v_input = convert(input);  // 转换为 Vulkan 张量表示

  vTensor v_output{  // 创建 Vulkan 输出张量对象
      context,
      v_input.sizes(),
      v_input.dtype(),
  };

  const api::utils::uvec3 global_workgroup_extents = v_output.extents();  // 获取全局工作组大小
  api::utils::ivec4 input_shader_extents = {  // 输入着色器的尺寸范围
      safe_downcast<int32_t>(v_input.extents().data[0u]),  // x 维度大小
      safe_downcast<int32_t>(v_input.extents().data[1u]),  // y 维度大小
      safe_downcast<int32_t>(v_input.extents().data[2u]),  // z 维度大小
      0 // 零填充
  };

  // early_exit 是全局工作组位置条件，用于不必要的执行退出。
  api::utils::ivec4 early_exit = {
      safe_downcast<int32_t>(v_input.extents().data[0u]),  // x 维度大小
      safe_downcast<int32_t>(v_input.extents().data[1u]),  // y 维度大小
      safe_downcast<int32_t>(v_input.extents().data[2u]),  // z 维度大小
      0 // 零填充
  };

  // 对于批次/高度/宽度，它们共享相同的着色器，由每个维度情况下的输入维度步长进行矢量化。
  api::utils::ivec4 input_dim_stride = {
      0,  // x 维度步长
      0,  // y 维度步长
      0,  // z 维度步长
      0,  // 零填充
  };

  api::utils::ivec4 input_tensor_dims = {
      0,  // 输入张量维度
      0,  // 输入张量维度
      0,  // 输入张量维度
      0,  // 输入张量维度
  };

  api::ShaderInfo shader_descriptor;
  set_softmax_kernel_params(
      input_arg.dim(),  // 输入张量的维度数
      dim,  // 软最大化操作的规范化维度
      v_input.sizes(),  // Vulkan 输入张量的尺寸
      shader_descriptor,  // 着色器描述符
      input_shader_extents,  // 输入着色器尺寸范围
      early_exit,  // 早期退出条件
      input_dim_stride,  // 输入维度步长
      input_tensor_dims);  // 输入张量维度

  const struct Block final {
    ivec4 input_shader_extents;
    ivec4 input_tensor_dims;
    ivec4 input_dim_stride;
    ivec4 early_exit;
  } block{
      input_shader_extents, input_tensor_dims, input_dim_stride, early_exit};  // 定义用于参数传递的块

  api::UniformParamsBuffer params(context, block);  // 创建统一参数缓冲区
  api::PipelineBarrier pipeline_barrier{};  // 管道屏障对象

  context->submit_compute_job(
      // 着色器描述符
      shader_descriptor,
      // 管道屏障
      pipeline_barrier,
      // 全局工作组大小
      global_workgroup_extents,
      // 自适应工作组大小
      adaptive_work_group_size(global_workgroup_extents),
      // 等待处理的句柄
      VK_NULL_HANDLE,
      // 着色器参数
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // 参数缓冲区
      params.buffer());

  return convert(v_output);  // 返回转换后的 Vulkan 输出张量
}



Tensor softmax(
    const at::Tensor& input_arg,  // 输入张量的引用
    const int64_t dim,  // 软最大化操作的维度
    // 定义一个名为 softmax 的函数，接受三个参数：input_arg 是输入的数据，dim 是维度信息，half_to_float 是一个布尔值参数，表示是否将输入从半精度转换为单精度
    const bool half_to_float) {
        // 调用 softmax_internal 函数，传入参数 input_arg, dim, half_to_float，并返回其结果
        return softmax_internal(input_arg, dim, half_to_float);
    }
} // 结束当前命名空间 'at'

Tensor log_softmax(
    const at::Tensor& input_arg, // 输入张量的常量引用
    const int64_t dim, // 指定的维度
    const bool half_to_float) { // 是否将半精度转换为单精度布尔值
  // 在计算 softmax 后，一些值变得非常小，低于 float16 的精度。这些值在 float16 中表示为 0，
  // 在应用对数函数时会导致 -inf。根据维基百科：
  // https://en.wikipedia.org/wiki/Half-precision_floating-point_format#Exponent_encoding，
  // 最小的严格正值（次正规化值）是 2^−24 ≈ 5.9605 × 10^−8。
  // 因此，我们在 softmax 输出的基础上添加 6 x 10^-8，以避免数值问题。
  float epsilon = 6e-8;
  // 调用内部 softmax 函数，并在其结果上添加 epsilon，然后应用对数函数
  return softmax_internal(input_arg, dim, half_to_float).add(epsilon).log();
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) { // 实现 Torch 库函数在 Vulkan API 下的行为
  m.impl("_softmax", TORCH_FN(softmax)); // 将 "_softmax" 实现为 softmax 函数
  m.impl("_log_softmax", TORCH_FN(log_softmax)); // 将 "_log_softmax" 实现为 log_softmax 函数
}

#endif /* USE_VULKAN_API */

} // 结束命名空间 'vulkan'
} // 结束命名空间 'native'
} // 结束命名空间 'at'
```