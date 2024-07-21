# `.\pytorch\aten\src\ATen\native\vulkan\ops\Convolution.cpp`

```py
//
// 包含 ATen 库的必要头文件
//
#include <ATen/Context.h>

//
// 包含 ATen 库的卷积相关工具函数
//
#include <ATen/native/ConvUtils.h>
//
// 包含 ATen 库的参数处理工具函数
//
#include <ATen/native/utils/ParamUtils.h>
//
// 包含 ATen Vulkan 实现的工具函数
//
#include <ATen/native/vulkan/api/Utils.h>
//
// 包含 ATen Vulkan 实现的张量打包函数
//
#include <ATen/native/vulkan/impl/Packing.h>
//
// 包含 ATen Vulkan 实现的卷积操作的公共函数
//
#include <ATen/native/vulkan/ops/Common.h>
//
// 包含 ATen Vulkan 实现的卷积操作函数
//
#include <ATen/native/vulkan/ops/Convolution.h>
//
// 包含 ATen Vulkan 实现的张量复制函数
//
#include <ATen/native/vulkan/ops/Copy.h>
//
// 包含 ATen Vulkan 实现的工具函数
//
#include <ATen/native/vulkan/ops/Utils.h>
//
// 包含 C++ 标准库的范围遍历工具
//
#include <c10/util/irange.h>

//
// 如果未启用每个运算符的单独头文件，则包含 ATen 全局操作函数
//
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
//
// 否则，包含 ATen 单独的操作函数头文件
//
#else
#include <ATen/ops/dequantize.h>
#include <ATen/ops/pad.h>
#include <ATen/ops/permute.h>
#include <ATen/ops/quantize_per_tensor.h>
#include <ATen/ops/zeros.h>
#endif

namespace at {
namespace native {
namespace vulkan {
namespace ops {

namespace conv2d {

//
// 判断卷积类型的辅助函数
//

//
// 检查是否为深度卷积
//
inline bool is_depthwise(const IntArrayRef weight_size, const int64_t groups) {
  uint32_t groups_uint = api::utils::safe_downcast<uint32_t>(groups);
  if (get_dim<DimConv2DKernel::OutChannels>(weight_size) != groups_uint) {
    return false;
  }
  if (get_dim<DimConv2DKernel::InChannels>(weight_size) != 1) {
    return false;
  }
  return true;
}

//
// 检查是否为逐点卷积
//
inline bool is_pointwise(const IntArrayRef weight_size) {
  if (get_dim<DimConv2DKernel::Width>(weight_size) != 1) {
    return false;
  }
  if (get_dim<DimConv2DKernel::Height>(weight_size) != 1) {
    return false;
  }
  return true;
}

//
// 确定使用的卷积方法
//
Conv2dMethod determine_method(
    const IntArrayRef weight_size,
    const IntArrayRef stride,
    const IntArrayRef padding,
    const IntArrayRef dilation,
    const int64_t groups,
    const bool transposed,
    const bool quantized) {
  //
  // 如果是转置卷积，返回滑动窗口卷积方法
  //
  if (transposed) {
    return Conv2dSlidingWindow;
  }
  //
  // 如果是深度卷积，返回深度卷积方法
  //
  if (is_depthwise(weight_size, groups)) {
    return Conv2dDepthwise;
  }
  //
  // 如果是逐点卷积，返回逐点卷积方法
  //
  if (is_pointwise(weight_size)) {
    return Conv2dPointwise;
  }
  //
  // 默认情况下返回滑动窗口卷积方法
  //
  return Conv2dSlidingWindow;
}

//
// 用于预打包的数据重排函数
//
/*
 * 将卷积权重张量重新排列为可以被卷积计算着色器使用的布局。此打包的目标是安排数据，使得在计算着色器中的数据访问尽可能线性化。
 * 打包模式的理由将在着色器内核代码中描述。
 *
 * 要理解此函数执行的转换，请考虑大小为{11, 1, 3, 3}的示例输入权重张量。以下转换将应用于此权重张量：
 *
 * 1. 首先，对N维度应用填充，使其成为4的倍数。在此情况下，添加了1批次，生成大小为{12, 1, 3, 3}的张量。
 *
 * 2. 接下来，展平张量的最后两个维度。通过将张量重塑为大小为{12, 1, 9}来完成此操作。
 *
 * 3. 最后，我们希望将批次维度“折叠”到通道维度中。我们首先沿着N维度分割张量，使每个分割具有4批次。通过将张量重塑为大小为{3, 4, 1, 9}来完成此操作。
 *
 * 4. 通常情况下，我们已经完成了，但是我们希望将每个结果批次垂直堆叠。通过置换N和C维度并将张量重塑为大小{4, 3, 9}来完成此操作。
 */
at::Tensor rearrange_weights_dw(const Tensor& weight_in) {
  // 克隆输入的权重张量，以便不修改原始输入
  at::Tensor weight = weight_in.clone();

  // 获取张量的各个维度大小
  uint32_t N = ops::get_dim<DimConv2DKernel::OutChannels>(weight);
  uint32_t C = ops::get_dim<DimConv2DKernel::InChannels>(weight);
  uint32_t H = ops::get_dim<DimConv2DKernel::Height>(weight);
  uint32_t W = ops::get_dim<DimConv2DKernel::Width>(weight);

  // 将N维度对齐到最接近的4的倍数，计算所需填充
  uint32_t N_aligned = api::utils::align_up(N, 4u);
  uint32_t N_padding_needed = N_aligned - N;

  // 对N维度应用常量填充，使其成为4的倍数
  weight = at::pad(weight, {0, 0, 0, 0, 0, 0, 0, N_padding_needed}, "constant", 0);

  // 将张量重塑为{N_aligned, C, H * W}，以便展平H和W维度
  weight = weight.reshape({N_aligned, C, H * W});

  // 将批次维度分割为每组4个批次
  uint32_t N4 = N_aligned / 4u;
  weight = weight.reshape({N4, 4, C, H * W});

  // 置换4个批次的顺序，使它们沿着通道维度排列，然后将结果批次垂直堆叠
  weight = weight.permute({1, 0, 2, 3}).reshape({4, N4 * C, H * W});

  // 返回连续内存中的张量
  return weight.contiguous();
}
/*
 * 重排卷积权重张量，使其能够被卷积计算着色器使用。此打包的目标是安排数据，
 * 以使计算着色器中的数据访问尽可能线性。关于打包模式背后的原理将在着色器内核代码中描述。
 *
 * 为了理解此函数执行的转换，请考虑一个大小为 {10, 7, 3, 3} 的示例输入。将对此权重张量执行以下转换：
 *
 * 1. 首先，在 N 和 C 维度上应用填充，使两者都是 4 的倍数。在本例中，添加了 2 个批次和 1 个通道的填充，
 *    生成大小为 {12, 8, 3, 3} 的张量。
 *
 * 2. 接下来，沿着 C 维度将张量分割，使每个分割具有 4 个通道。这通过将通道重塑为大小 {12, 2, (4, 3, 3)} 完成。
 *    括号表示分割的大小。
 *
 * 3. 对于每个分割，我们希望将 C 维度“折叠”到 W 维度中。假设第一个分割在 H=0 行的值为
 *    0,1,2 | 10,11,12 | 20,21,22 | 30,31,32
 *    其中 | 表示通道边界，然后目标是将这些行组合成一个行，其值为
 *    0, 10, 20, 30, 1, 11, 21, 31, 2, 12, 22, 32
 *    代码中通过置换和重塑张量来完成这一步骤，生成大小为 {12, 2, (3, 12)} 的张量。
 *
 * 4. 接下来，我们希望水平堆叠属于同一批次的分割，这通过交换中间张量的 C 和 H 维度并重塑来完成，
 *    生成大小为 {12, 3, 24} 的张量。
 *
 * 5. 现在，我们将重复类似的过程，“折叠” N 维度到 C 维度。首先沿 N 维度分割张量，使每个分割有 4 个批次。
 *    为此，将张量重塑为 {3, 4, 3, 24}。
 *
 * 6. 通常情况下，我们完成了，但我们还想将每个批次垂直堆叠在彼此上。因此最后一步是通过置换 N 和 C 维度
 *    并重塑到输出形状 {4, 9, 24}。
 *
 * 对于转置卷积，存在一些细微差异以反映着色器中的数据访问模式。首要的差异是权重张量沿 H 和 W 维度翻转。
 * 第二个主要差异是步骤 3 和 4 稍有不同，以便交替插入分割。
 */
at::Tensor rearrange_weights_2d(const Tensor& weight_in, bool tconv) {
    // 对输入权重张量进行克隆，以便在不改变原始张量的情况下进行修改
    at::Tensor weight = weight_in.clone();

    // 如果是转置卷积，沿着 H 和 W 轴翻转权重值
    if (tconv) {
  // 在 weight 张量上执行两次 flip 操作，分别在第 3 和第 2 维上进行翻转
  weight = weight.flip(3).flip(2);

  // 获取 weight 张量的各个维度大小
  uint32_t N = get_dim<DimConv2DKernel::OutChannels>(weight);
  uint32_t C = get_dim<DimConv2DKernel::InChannels>(weight);
  uint32_t H = get_dim<DimConv2DKernel::Height>(weight);
  uint32_t W = get_dim<DimConv2DKernel::Width>(weight);

  // 将 N 和 C 维度分别向上对齐到最接近的 4 的倍数
  uint32_t N_aligned = api::utils::align_up(N, 4u);
  uint32_t C_aligned = api::utils::align_up(C, 4u);

  // 计算需要填充的 C 和 N 维度的数量，使其成为 4 的倍数
  uint32_t C_padding_needed = C_aligned - C;
  uint32_t N_padding_needed = N_aligned - N;

  // 对 weight 张量进行填充，以匹配对齐后的维度
  weight = at::pad(
      weight,
      {0, 0, 0, 0, 0, C_padding_needed, 0, N_padding_needed},
      "constant",
      0);

  // 将 C 维度分割成每组 4 个通道
  uint32_t C4 = C_aligned / 4u;
  weight = weight.reshape({N_aligned, C4, 4, H, W});

  if (!tconv) {
    // 对于非转置卷积操作，将每组 4 个通道沿着宽度轴折叠
    weight = weight.permute({0, 1, 3, 4, 2}).reshape({N_aligned, C4, H, 4 * W});
    // 接下来将每组四个通道沿着宽度轴再次折叠
    weight =
        weight.permute({0, 2, 1, 3}).reshape({N_aligned, H, C_aligned * W});
  } else {
    // 对于转置卷积操作，执行与上述相同的操作，但要求从每个通道中交错提取四个批次
    weight = weight.permute({0, 3, 4, 1, 2}).reshape({N_aligned, H, W, 4 * C4});
    // 接下来将最后两个维度重新整形为单行
    weight = weight.reshape({N_aligned, H, C_aligned * W});
  }

  // 将 N 维度分割成每组 4 个样本
  uint32_t N4 = N_aligned / 4u;
  weight = weight.reshape({N4, 4, H, C_aligned * W});

  // 折叠最外层维度，使每组 4 个样本垂直堆叠
  weight = weight.permute({1, 0, 2, 3}).reshape({4, N4 * H, C_aligned * W});

  // 返回连续存储的 weight 张量
  return weight.contiguous();
/*
 * Rearranges a convolution weight tensor to a layout that can be used by
 * convolution compute shaders. The goal of this packing is to arrange the data
 * such that data access in the compute shader is as linear as possible. The
 * reasoning behind the packing pattern will be described in the shader kernel
 * code.
 *
 * The rearrangement structure is quite straightforward. Essentially we are
 * taking each texel and arranging them along the x axis.
 */
at::Tensor rearrange_bias(
    const std::optional<Tensor>& bias_in,
    const at::Tensor& weight_in,
    bool tconv) {
  // If optional is empty, just return zeros
  if (!bias_in) {
    uint32_t L = tconv ? get_dim<DimTConv2DKernel::OutChannels>(weight_in)
                       : get_dim<DimConv2DKernel::OutChannels>(weight_in);
    const uint32_t L4 = api::utils::div_up(L, 4u);

    // Create a zero tensor with shape [4, 1, L4] matching weight_in's options
    at::Tensor bias = at::zeros({4, 1, L4}, weight_in.options());
    return bias;
  }

  // Clone the bias tensor if it exists
  at::Tensor bias = bias_in->clone();

  // Bias should just be a 1D tensor
  uint32_t L = get_dim<Dim1D::Length>(bias);

  // Align L to the next multiple of 4
  uint32_t L_aligned = api::utils::align_up(L, 4u);

  // Calculate padding needed to make L_aligned a multiple of 4
  uint32_t padding_needed = L_aligned - L;

  // Pad the bias tensor with zeros at the end to achieve length L_aligned
  bias = at::pad(bias, {0, padding_needed}, "constant", 0);

  // Reshape + permute to group every 4 consecutive elements along the same
  // channel
  uint32_t L4 = L_aligned / 4u;
  bias = bias.reshape({L4, 4}).permute({1, 0});
  bias = bias.reshape({4, 1, L4});

  // Ensure the tensor memory is contiguous for efficient processing
  return bias.contiguous();
}

//
// Shader and Workgroup size determination
//

static api::ShaderInfo get_shader(
    const IntArrayRef kernel_size,
    const IntArrayRef stride,
    const IntArrayRef padding,
    const IntArrayRef dilation,
    const Conv2dMethod method,
    const bool transposed,
    const bool quantized) {
  api::ShaderInfo shader;

  // Determine shader based on whether quantization is applied
  if (quantized) {
    if (transposed) {
      shader = VK_KERNEL(quantized_conv_transpose2d);
      return shader;
    }

    switch (method) {
      case Conv2dSlidingWindow:
        shader = VK_KERNEL(quantized_conv2d);
        break;
      case Conv2dDepthwise:
        shader = VK_KERNEL(quantized_conv2d_dw);
        break;
      case Conv2dPointwise:
        shader = VK_KERNEL(quantized_conv2d_pw_2x2);
        break;
        // todo fail for quantized transposed conv
    }
    return shader;
  }

  // If not quantized, determine shader based on transposition and method
  if (transposed) {
    shader = VK_KERNEL(conv_transpose2d);
    return shader;
  }

  switch (method) {
    case Conv2dSlidingWindow:
      shader = VK_KERNEL(conv2d);
      break;
    // additional cases for other convolution methods can be added here
  }

  // Return the determined shader information
  return shader;
}
    // 如果卷积类型为深度可分离卷积(Conv2dDepthwise)
    case Conv2dDepthwise:
      // 使用深度可分离卷积的默认着色器
      shader = VK_KERNEL(conv2d_dw);
      // 如果卷积核大小为4并且第三维和第四维为3
      if (kernel_size.size() == 4 && kernel_size[2] == 3 &&
          kernel_size[3] == 3) {
        // 将着色器设置为适用于3x3输出块大小的深度可分离卷积着色器
        shader = VK_KERNEL(conv2d_dw_output_tile_3x3);
      }
      // 如果卷积核大小为4并且第三维和第四维为5
      if (kernel_size.size() == 4 && kernel_size[2] == 5 &&
          kernel_size[3] == 5) {
        // 将着色器设置为适用于5x5输出块大小的深度可分离卷积着色器
        shader = VK_KERNEL(conv2d_dw_output_tile_5x5);
      }
      break;
    
    // 如果卷积类型为逐点卷积(Conv2dPointwise)
    case Conv2dPointwise:
      // 使用逐点卷积的默认着色器
      shader = VK_KERNEL(conv2d_pw_output_tile_2x2);
      break;
    }
    
    // 返回选择的着色器
    return shader;
// 结构体定义，包含了记录操作所需的参数
struct Params final {
  // 输出张量的尺寸
  api::utils::ivec3 out_extents;
  // 填充值，未使用
  int32_t fill0;
  // 输入张量的尺寸
  api::utils::ivec3 in_extents;
  // 填充值，未使用
  int32_t fill1;
  // 叠加区域的坐标和大小
  api::utils::ivec4 overlay_region;
  // 卷积核的尺寸
  api::utils::ivec2 kernel_size;
  // 步幅
  api::utils::ivec2 stride;
  // 填充
  api::utils::ivec2 padding;
  // 膨胀
  api::utils::ivec2 dilate;
  // 输出范围的最小值和最大值
  api::utils::vec2 clamp;
};

// 记录操作的函数，执行计算任务
void record_op(
    api::Context* const context,            // 上下文环境
    api::ShaderInfo& compute_shader,        // 计算着色器信息
    vTensor& v_output,                      // 输出张量
    const vTensor& v_input,                 // 输入张量
    const vTensor& v_weight,                // 权重张量
    const vTensor& v_bias,                  // 偏置张量
    const IntArrayRef overlay_region,       // 叠加区域
    const IntArrayRef stride,               // 步幅
    const IntArrayRef padding,              // 填充
    const IntArrayRef dilation,             // 膨胀
    const float output_min,                 // 输出最小值
    const float output_max,                 // 输出最大值
    const IntArrayRef kernel_size,          // 卷积核尺寸
    const Conv2dMethod method,              // 卷积方法
    const bool transposed) {                // 是否转置
  // 初始化管线屏障
  api::PipelineBarrier pipeline_barrier{};

  // 获取全局工作组大小和自适应的局部工作组大小
  api::utils::uvec3 global_size = v_output.extents();
  api::utils::uvec3 local_size = adaptive_work_group_size(global_size);

  // 填充参数结构体
  Params block{
      api::utils::make_ivec3(v_output.extents()),  // 输出张量的尺寸
      0u,                                          // 填充值
      api::utils::make_ivec3(v_input.extents()),   // 输入张量的尺寸
      0u,                                          // 填充值
      utils::make_ivec4(overlay_region, /*reverse=*/true),  // 叠加区域（反向）
      utils::make_ivec2({kernel_size[3], kernel_size[2]}),   // 卷积核尺寸（反向）
      utils::make_ivec2(stride, /*reverse=*/true),           // 步幅（反向）
      utils::make_ivec2(padding, /*reverse=*/true),          // 填充（反向）
      utils::make_ivec2(dilation, /*reverse=*/true),         // 膨胀（反向）
      {output_min, output_max},                               // 输出范围
  };

  // 创建统一参数缓冲区
  api::UniformParamsBuffer params(context, block);

  // 提交计算任务
  context->submit_compute_job(
      // 计算着色器描述符
      compute_shader,
      // 管线屏障
      pipeline_barrier,
      // 全局工作组大小
      global_size,
      // 局部工作组大小
      local_size,
      // 围栏句柄
      VK_NULL_HANDLE,
      // 着色器参数
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_weight.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_bias.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // 参数缓冲区
      params.buffer());
}

// 结构体定义，包含了记录量化操作所需的参数
struct QParams final {
  // 缩放因子
  api::utils::vec4 scales;
  // 零点
  api::utils::ivec4 zero_points;
  // 输出张量的尺寸
  api::utils::ivec3 out_extents;
  // 填充值，未使用
  int32_t fill0;
  // 输入张量的尺寸
  api::utils::ivec3 in_extents;
  // 填充值，未使用
  int32_t fill1;
  // 叠加区域的坐标和大小
  api::utils::ivec4 overlay_region;
  // 卷积核的尺寸
  api::utils::ivec2 kernel_size;
  // 步幅
  api::utils::ivec2 stride;
  // 填充
  api::utils::ivec2 padding;
  // 膨胀
  api::utils::ivec2 dilate;
  // 输出范围的最小值和最大值
  api::utils::vec2 clamp;
};

// 记录量化操作的函数，执行计算任务
void record_quantized_op(
    api::Context* const context,            // 上下文环境
    api::ShaderInfo& compute_shader,        // 计算着色器信息
    vTensor& v_output,                      // 输出张量
    const vTensor& v_input,                 // 输入张量
    const vTensor& v_weight,                // 权重张量
    const vTensor& v_bias,                  // 偏置张量
    const IntArrayRef overlay_region,       // 叠加区域
    const IntArrayRef stride,               // 步幅
    const IntArrayRef padding,              // 填充
    const IntArrayRef dilation,             // 膨胀
    const float output_min,                 // 输出最小值
    const float output_max,                 // 输出最大值
    const IntArrayRef kernel_size,          // 卷积核尺寸
    const Conv2dMethod method,              // 卷积方法
    // 定义一个函数，接受输出、输入、权重和偏置张量，以及一些卷积操作的参数，并执行相应的计算
    const bool transposed) {
      // 创建一个管线屏障对象，用于同步计算管线的状态
      api::PipelineBarrier pipeline_barrier{};
    
      // 获取输出张量的全局大小（维度），用于确定计算的全局工作组大小
      api::utils::uvec3 global_size = v_output.extents();
      // 根据全局大小调整本地工作组大小，以优化计算性能
      api::utils::uvec3 local_size = adaptive_work_group_size(global_size);
    
      // 定义一个包含量化参数的结构体，包括输出、输入、权重和偏置的量化信息，以及卷积操作的其他参数
      QParams block{
          {
              v_output.get_scale_float(),
              v_input.get_scale_float(),
              v_weight.get_scale_float(),
              v_bias.get_scale_float(),
          },
          {
              v_output.get_zero_point_int32(),
              v_input.get_zero_point_int32(),
              v_weight.get_zero_point_int32(),
              v_bias.get_zero_point_int32(),
          },
          // 将输出张量的维度转换为ivec3结构
          api::utils::make_ivec3(v_output.extents()),
          0u,  // 未使用的参数，置零
          // 将输入张量的维度转换为ivec3结构
          api::utils::make_ivec3(v_input.extents()),
          0u,  // 未使用的参数，置零
          // 创建一个ivec4结构来表示覆盖区域和方向标志
          utils::make_ivec4(overlay_region, /*reverse=*/true),
          // 创建一个ivec2结构来表示卷积核大小（高度和宽度）
          utils::make_ivec2({kernel_size[3], kernel_size[2]}),
          // 创建一个ivec2结构来表示卷积步幅（高度和宽度）
          utils::make_ivec2(stride, /*reverse=*/true),
          // 创建一个ivec2结构来表示填充大小（高度和宽度）
          utils::make_ivec2(padding, /*reverse=*/true),
          // 创建一个ivec2结构来表示扩展大小（高度和宽度）
          utils::make_ivec2(dilation, /*reverse=*/true),
          // 创建一个包含输出最小值和最大值的向量
          {output_min, output_max},
      };
      // 创建统一参数缓冲区对象，用于将参数传递给着色器程序
      api::UniformParamsBuffer params(context, block);
    
      // 提交计算作业到图形命令上下文
      context->submit_compute_job(
          // 计算着色器程序描述符
          compute_shader,
          // 管线屏障，用于同步计算着色器的读写访问
          pipeline_barrier,
          // 全局工作组大小，用于指定计算的全局并行度
          global_size,
          // 本地工作组大小，用于指定计算的本地并行度
          local_size,
          // 围栏句柄，用于在提交作业时进行同步控制
          VK_NULL_HANDLE,
          // 着色器参数列表，包括输出、输入、权重、偏置等图像数据的描述
          v_output.image(
              pipeline_barrier,
              api::PipelineStage::COMPUTE,
              api::MemoryAccessType::WRITE),
          v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
          v_weight.image(pipeline_barrier, api::PipelineStage::COMPUTE),
          v_bias.image(pipeline_barrier, api::PipelineStage::COMPUTE),
          // 参数缓冲区对象，包含了卷积操作所需的所有参数
          params.buffer());
}

} // namespace conv2d

namespace {

using namespace api::utils;

// 将权重数据打包成 Vulkan 张量
vTensor pack_weights(
    const Tensor& weight_inp,  // 输入权重张量
    const bool transposed,      // 是否转置
    const bool quantized,       // 是否量化
    const Conv2dMethod conv_method) {  // 卷积方法
  if (weight_inp.is_vulkan()) {  // 如果输入张量已经是 Vulkan 张量，则直接转换并返回
    return convert(weight_inp);
  }

  const Tensor weight_arg = quantized ? at::dequantize(weight_inp) : weight_inp;  // 如果量化则反量化权重

  // 根据是否转置选择不同的排列方式
  const Tensor weight = transposed
      ? at::permute(weight_arg, {1, 0, 2, 3}).contiguous()  // 转置排列并保证连续性
      : weight_arg.contiguous();  // 保证权重张量的连续性

  at::Tensor weight_rearranged;
  if (conv_method == Conv2dDepthwise) {  // 如果是深度可分离卷积
    weight_rearranged = conv2d::rearrange_weights_dw(weight);  // 重新排列深度可分离卷积的权重
  } else {
    weight_rearranged = conv2d::rearrange_weights_2d(weight, transposed);  // 重新排列二维卷积的权重
  }

  // 创建 Vulkan 张量对象，并进行数据转换
  vTensor v_weight{
      api::context(),
      weight_rearranged.sizes().vec(),
      convert_dtype(weight_rearranged.scalar_type()),
      api::StorageType::TEXTURE_2D,
  };

  pack_cpu_to_vulkan(weight_rearranged, v_weight);  // 将 CPU 上的权重数据打包到 Vulkan 张量

  return v_weight;  // 返回 Vulkan 张量
}

// 将偏置数据打包成 Vulkan 张量
vTensor pack_biases(
    const std::optional<Tensor>& bias,  // 可选的偏置张量
    const Tensor& weight,               // 权重张量
    const bool transposed,              // 是否转置
    const bool quantized) {             // 是否量化
  at::Tensor bias_arg = conv2d::rearrange_bias(bias, weight, transposed);  // 重新排列偏置数据

  // 如果量化且偏置张量的数据类型是量化类型，则反量化偏置
  at::Tensor bias_rearranged =
      (quantized &&
       (bias_arg.scalar_type() == kQUInt8 || bias_arg.scalar_type() == kQInt8 ||
        bias_arg.scalar_type() == kQInt32))
      ? at::dequantize(bias_arg)
      : bias_arg;

  // 创建 Vulkan 张量对象，并进行数据转换
  vTensor v_bias{
      api::context(),
      bias_rearranged.sizes().vec(),
      convert_dtype(bias_rearranged.scalar_type()),
      api::StorageType::TEXTURE_2D,
  };

  pack_cpu_to_vulkan(bias_rearranged, v_bias);  // 将 CPU 上的偏置数据打包到 Vulkan 张量

  return v_bias;  // 返回 Vulkan 张量
}

/*
 * 计算卷积输出时叠加区域的大小。
 */
std::array<int64_t, 4> compute_overlay_region(
    const Tensor& weight,       // 权重张量
    const IntArrayRef dilation, // 扩张参数数组
    const bool transposed) {    // 是否转置
  const IntArrayRef filter = weight.sizes();  // 获取权重张量的尺寸信息

  // 计算叠加长度的 Lambda 函数
  const auto overlay_length = [](const int64_t k, const int64_t d) {
    return k + (k - 1) * (d - 1);
  };

  // 返回叠加区域的大小数组
  return {
      align_up(
          transposed ? filter[Layout::TransposedFilter::output]
                     : filter[Layout::Filter::output],
          INT64_C(4)),  // 对输出尺寸进行对齐
      align_up(
          transposed ? filter[Layout::TransposedFilter::input]
                     : filter[Layout::Filter::input],
          INT64_C(4)),  // 对输入尺寸进行对齐
      overlay_length(
          filter[Layout::Filter::height], dilation[Layout::Parameter::height]),  // 计算高度叠加长度
      overlay_length(
          filter[Layout::Filter::width], dilation[Layout::Parameter::width]),   // 计算宽度叠加长度
  };
}

// 将参数向量打包成大小为2的数组
std::array<int64_t, 2> pack_params(const std::vector<int64_t>& vector) {
  TORCH_INTERNAL_ASSERT(2u == vector.size(), "Invalid usage!");  // 断言向量大小为2

  return {
      vector[0],
      vector[1],
  };
}

// 检查权重张量是否有效
bool weight_valid(const Tensor& weight, const bool quantized) {
  if (4 != weight.ndimension()) {  // 如果权重张量维度不是4，则无效
    return false;
  }
  if (get_dim<DimConv2DKernel::Height>(weight) == 0) {  // 如果高度维度为0，则无效
    return false;
  }
  if (get_dim<DimConv2DKernel::Width>(weight) == 0) {  // 如果宽度维度为0，则无效
    return false;
  }
  return true;  // 否则有效
}
    return false;
  }
  # 检查权重张量是否在 CPU 设备上，并且不是 Vulkan 设备，如果不是，则返回 false
  if (!weight.device().is_cpu() &&
      weight.device().type() != c10::DeviceType::Vulkan) {
    return false;
  }
  # 如果启用量化，并且权重张量的数据类型既不是无符号整型8位（kQUInt8），也不是有符号整型8位（kQInt8），则返回 false
  if (quantized &&
      (weight.scalar_type() != c10::kQUInt8 &&
       weight.scalar_type() != c10::kQInt8)) {
    return false;
  }

  # 若通过了上述条件检查，则返回 true
  return true;
}

bool bias_valid(
    const std::optional<Tensor>& bias,  // 可选的偏置张量
    const Tensor& weight,               // 权重张量
    const bool transposed,              // 是否是转置卷积
    const bool quantized) {             // 是否量化
  if (!bias) {                         // 如果没有偏置张量
    return true;                       // 返回真
  }

  if (bias->ndimension() != 1) {       // 如果偏置张量维度不为1
    return false;                      // 返回假
  }
  if (!bias->device().is_cpu() &&      // 如果偏置张量设备不是 CPU 并且不是 Vulkan
      bias->device().type() != c10::DeviceType::Vulkan) {
    return false;                      // 返回假
  }
  uint32_t L = get_dim<Dim1D::Length>(*bias);  // 获取偏置张量长度维度
  uint32_t OC = transposed ? get_dim<DimTConv2DKernel::OutChannels>(weight)
                           : get_dim<DimConv2DKernel::OutChannels>(weight);  // 获取输出通道数
  if (L != OC) {                       // 如果长度维度不等于输出通道数
    return false;                      // 返回假
  }

  return true;                         // 全部条件满足，返回真
}

bool available(
    const Tensor& weight,               // 权重张量
    const std::optional<Tensor>& bias,  // 可选的偏置张量
    const IntArrayRef stride,           // 步长数组
    const IntArrayRef padding,          // 填充数组
    const IntArrayRef dilation,         // 膨胀数组
    const bool transposed,              // 是否是转置卷积
    const bool quantized,               // 是否量化
    const IntArrayRef /* output_padding */,  // 输出填充（暂时不用）
    const int64_t groups,               // 分组数
    const std::optional<Scalar>& output_min,  // 可选的输出最小值
    const std::optional<Scalar>& output_max) {  // 可选的输出最大值
  if (!weight_valid(weight, quantized)) {  // 如果权重张量不合法
    return false;                        // 返回假
  }
  if (!bias_valid(bias, weight, transposed, quantized)) {  // 如果偏置不合法
    return false;                        // 返回假
  }
  if (get_dim<Dim4D::Height>(stride) == 0 ||  // 如果步长数组高度或宽度为0
      get_dim<Dim4D::Width>(stride) == 0) {
    return false;                        // 返回假
  }
  if (transposed) {                     // 如果是转置卷积
    if (get_dim<Dim4D::Height>(dilation) != 1 ||  // 如果膨胀数组高度不为1
        get_dim<Dim4D::Width>(dilation) != 1) {
      return false;                      // 返回假
    }
  } else {                              // 如果不是转置卷积
    if (get_dim<Dim4D::Height>(dilation) == 0 ||  // 如果膨胀数组高度或宽度为0
        get_dim<Dim4D::Width>(dilation) == 0) {
      return false;                      // 返回假
    }
  }
  if (groups <= 0) {                    // 如果分组数小于等于0
    return false;                      // 返回假
  }
  if (transposed) {                    // 如果是转置卷积
    if ((get_dim<DimTConv2DKernel::OutChannels>(weight) % groups) != 0) {  // 如果输出通道数不能整除分组数
      return false;                     // 返回假
    }
  } else {                             // 如果不是转置卷积
    if ((get_dim<DimConv2DKernel::OutChannels>(weight) % groups) != 0) {  // 如果输出通道数不能整除分组数
      return false;                     // 返回假
    }
  }
  if (get_dim<DimConv2DKernel::InChannels>(weight) == 0 ||  // 如果输入通道数或输出通道数为0
      get_dim<DimConv2DKernel::OutChannels>(weight) == 0) {
    return false;                      // 返回假
  }
  if (output_min && !output_min->isFloatingPoint()) {  // 如果存在输出最小值且不是浮点数
    return false;                      // 返回假
  }
  if (output_max && !output_max->isFloatingPoint()) {  // 如果存在输出最大值且不是浮点数
    return false;                      // 返回假
  }
  return true;                         // 全部条件满足，返回真
}

bool usable(const Tensor& input, const bool quantized) {  // 可用性检查函数
  if (input.ndimension() != 4) {       // 如果输入张量维度不为4
    return false;                      // 返回假
  }
  if (input.device().type() != c10::DeviceType::Vulkan) {  // 如果输入张量设备不是 Vulkan
    return false;                      // 返回假
  }
  if (!quantized && input.scalar_type() != at::kFloat) {  // 如果非量化且输入张量类型不是浮点型
    return false;                      // 返回假
  }
  if (quantized && input.scalar_type() != c10::kQUInt8) {  // 如果量化且输入张量类型不是无符号8位整数
    return false;                      // 返回假
  }
  if (get_dim<Dim4D::Batch>(input) == 0) {  // 如果输入张量批次维度为0
    return false;                      // 返回假
  }
  if (get_dim<Dim4D::Channel>(input) == 0) {  // 如果输入张量通道维度为0
    return false;                      // 返回假
  }
  if (get_dim<Dim4D::Height>(input) == 0) {  // 如果输入张量高度维度为0
    return false;                      // 返回假
  }
  if (get_dim<Dim4D::Width>(input) == 0) {  // 如果输入张量宽度维度为0
    return false;                      // 返回假
  }
  if (input.requires_grad()) {         // 如果输入张量需要梯度
    return false;                      // 返回假
  }

  return true;                         // 全部条件满足，返回真
}

static inline std::vector<int64_t> get_conv_transpose_output_size(
    IntArrayRef input_size,            // 输入大小数组
    IntArrayRef weight_size,           // 权重大小数组
    IntArrayRef padding,               // 填充数组
    IntArrayRef output_padding,        // 输出填充数组
    // 定义函数 calculate_output_size，计算卷积操作后输出张量的大小
    std::vector<int64_t> calculate_output_size(
        // 输入张量的大小
        IntArrayRef input_size,
        // 卷积核的大小
        IntArrayRef weight_size,
        // 步长数组，默认为空数组
        IntArrayRef stride,
        // 膨胀数组，默认为空数组
        IntArrayRef dilation = IntArrayRef()) {
      
      // 获取张量的维度数量
      auto dim = input_size.size();
      
      // 创建存储输出张量大小的向量
      std::vector<int64_t> output_size(dim);
      
      // 计算输出张量的第一个维度大小，通常是批次维度
      output_size[0] = input_size[input_batch_size_dim];
      
      // 计算输出张量的第二个维度大小，通常是卷积核的输入通道维度
      output_size[1] = weight_size[weight_input_channels_dim];
      
      // 遍历计算从第三个维度开始的所有维度的输出大小
      for (const auto d : c10::irange(2, dim)) {
        // 根据卷积操作的参数计算每个维度的输出大小
        output_size[d] = stride[d - 2] * (input_size[d] - 1) + weight_size[d] -
            2 * padding[d - 2] + output_padding[d - 2];
      }
      
      // 返回计算得到的输出张量大小向量
      return output_size;
    }
} // 结束当前命名空间 conv1d

namespace conv1d {

/*
 * 对权重进行宽度打包，确保在 Vulkan 设备上执行
 */
vTensor pack_weights_using_width_packing(const Tensor& weight_arg) {
  // 复制权重到本地变量
  Tensor weight = weight_arg;

  // 如果权重在 CPU 上，转移到 Vulkan 设备上
  if (weight.is_cpu()) {
    weight = weight.vulkan();
  }

  // 检查权重是否在 Vulkan 设备上
  TORCH_CHECK(weight.is_vulkan(), "Weight must be on Vulkan device!");

  // 转换成 vTensor 格式
  vTensor v_weight = convert(weight);

  // 如果 v_weight 的 GPU 内存布局为 TENSOR_CHANNELS_PACKED，则进行通道打包到宽度打包的转换
  if (v_weight.gpu_memory_layout() ==
      api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED) {
    v_weight = packing::convert_image_channels_packed_to_width_packed(v_weight);
  }

  // 检查转换后的 v_weight 是否为 TENSOR_WIDTH_PACKED 格式
  TORCH_CHECK(
      v_weight.gpu_memory_layout() == api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
      "After packing, the v_weight must be in TENSOR_WIDTH_PACKED format");

  // 返回宽度打包后的 v_weight
  return v_weight;
}

/*
 * 这是一个完整的实现。有关算法详细信息，请参阅着色器内核代码。
 */
Tensor run_conv1d_context_impl(
    const Tensor& input_arg,
    const Tensor& weight_arg,
    const std::optional<Tensor>& bias_arg_opt,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    // 定义一个接受整数参数的函数，返回一个Tensor类型对象
    int64_t groups) {
  // 获取当前上下文的指针，用于访问API
  api::Context* const context = api::context();
  // 如果input_arg是基于Vulkan的Tensor，则直接使用它；否则转换为Vulkan Tensor
  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  // 如果weight_arg是基于Vulkan的Tensor，则直接使用它；否则转换为Vulkan Tensor
  const Tensor weight =
      weight_arg.is_vulkan() ? weight_arg : weight_arg.vulkan();

  // 获取input和weight Tensor的尺寸信息
  const IntArrayRef& input_sizes = input.sizes();
  const IntArrayRef& weight_sizes = weight.sizes();

  // 转换Tensor尺寸为int32_t类型，获取输入通道数、输出通道数和卷积核大小
  int32_t in_channels = static_cast<int32_t>(input_sizes[1]);
  int32_t out_channels = static_cast<int32_t>(weight_sizes[0]);
  int32_t kernel_size = static_cast<int32_t>(weight_sizes[2]);

  // 定义bias Tensor
  Tensor bias;
  // 如果存在bias_arg_opt参数且为Vulkan Tensor，则直接使用；否则转换为Vulkan Tensor
  if (bias_arg_opt) {
    if (bias_arg_opt->is_vulkan()) {
      bias = bias_arg_opt.value();
    } else {
      bias = bias_arg_opt.value().vulkan();
    }
  } else {
    // 如果bias_arg_opt不存在，则创建一个全零的Vulkan Tensor作为bias
    bias = at::zeros({out_channels}).vulkan();
  }

  // 检查input和weight Tensor的维度是否为3
  TORCH_CHECK(input.dim() == 3, "input must be a 3-dim tensor");
  TORCH_CHECK(weight.dim() == 3, "weight must be a 3-dim tensor");
  // 检查输入通道数是否可以被groups整除
  TORCH_CHECK(
      in_channels % groups == 0, "in_channels must be divisible by groups");
  // 检查输出通道数是否可以被groups整除
  TORCH_CHECK(
      out_channels % groups == 0, "out_channels must be divisible by groups");

  // 将input、weight和bias Tensor转换为vTensor
  const vTensor& v_input = convert(input);
  const vTensor& v_weight = convert(weight);
  const vTensor& v_bias = convert(bias);

  // 创建v_output作为输出vTensor对象，使用上下文、卷积输出大小和输入数据类型初始化
  vTensor v_output{
      context,
      conv_output_size(input_sizes, weight_sizes, padding, stride, dilation),
      v_input.dtype(),
  };

  // 定义Block结构体，存储卷积操作相关参数
  const struct Block final {
    int32_t in_length;
    int32_t kernel_size;
    int32_t stride;
    int32_t padding;
    int32_t dilation;
    int32_t in_group_size;
    int32_t out_group_size;
    int32_t batch_size;
  } block{
      static_cast<int32_t>(input_sizes[2]),
      kernel_size,
      static_cast<int32_t>(stride[0]),
      static_cast<int32_t>(padding[0]),
      static_cast<int32_t>(dilation[0]),
      static_cast<int32_t>(in_channels / groups),
      static_cast<int32_t>(out_channels / groups),
      static_cast<int32_t>(input_sizes[0]),
  };

  // 使用Block参数创建UniformParamsBuffer对象
  api::UniformParamsBuffer params(context, block);
  // 创建PipelineBarrier对象
  api::PipelineBarrier pipeline_barrier{};

  // 提交计算作业到上下文中，执行卷积操作
  context->submit_compute_job(
      // Vulkan核函数描述符
      VK_KERNEL(conv1d),
      // 管道屏障
      pipeline_barrier,
      // 全局工作组大小
      {1, static_cast<uint32_t>(out_channels), 1},
      // 局部工作组大小
      {1, 1, 1},
      // 围栏句柄
      VK_NULL_HANDLE,
      // Shader参数
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_weight.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_bias.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // 参数缓冲区
      params.buffer());

  // 将v_output转换为Tensor对象并返回
  return convert(v_output);
}
} // namespace conv1d



// 结束 conv1d 命名空间的定义

Conv2dPackedContext::Conv2dPackedContext(
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    const IntArrayRef stride_arg,
    const IntArrayRef padding_arg,
    const IntArrayRef dilation_arg,
    const bool transposed,
    const bool quantized,
    const IntArrayRef output_padding_arg,
    const int64_t groups,
    const std::optional<Scalar>& output_min,
    const std::optional<Scalar>& output_max)
    : unpacked_{c10::AnyType::get()} {



// Conv2dPackedContext 构造函数的定义，初始化列表开始

  const auto stride = expand_param_if_needed(stride_arg, "stride", 2);
  const auto padding = expand_param_if_needed(padding_arg, "padding", 2);
  const auto dilation = expand_param_if_needed(dilation_arg, "dilation", 2);
  const auto output_padding =
      expand_param_if_needed(output_padding_arg, "output_padding", 2);



  // 根据需要扩展参数 stride, padding, dilation, output_padding，并赋值给对应的变量

  TORCH_CHECK(
      available(
          weight,
          bias,
          stride,
          padding,
          dilation,
          transposed,
          quantized,
          output_padding,
          groups,
          output_min,
          output_max),
      "Vulkan::convolution not available! "
      "Reason: The provided (weight, bias, stride, padding, dilation, groups, "
      "transposed, output_padding, output_min, output_max) parameters are either "
      "invalid individually or their combination is not supported by Vulkan impl.");



  // 检查参数的有效性和组合是否支持 Vulkan 实现，抛出错误信息若不支持

  const auto method = conv2d::determine_method(
      weight.sizes(), stride, padding, dilation, groups, transposed, quantized);



  // 确定使用的卷积方法，根据给定的参数调用 conv2d::determine_method 函数

  packed_.reserve(Packed::NumArgs);
  // 为 packed_ 分配足够的空间以容纳 Packed::NumArgs 个元素

  packed_.emplace_back(
      convert(pack_weights(weight, transposed, quantized, method)));



  // 将权重 weight 打包，并将打包结果添加到 packed_ 中

  packed_.emplace_back(
      convert(pack_biases(bias, weight, transposed, quantized)));



  // 将偏置 bias 打包，并将打包结果添加到 packed_ 中

  packed_.emplace_back(compute_overlay_region(weight, dilation, transposed));



  // 计算权重的覆盖区域并将结果添加到 packed_ 中

  packed_.emplace_back(pack_params(stride));
  packed_.emplace_back(pack_params(padding));
  packed_.emplace_back(output_padding);
  packed_.emplace_back(pack_params(dilation));
  packed_.emplace_back(transposed);
  packed_.emplace_back(quantized);
  packed_.emplace_back(safe_downcast<int32_t>(groups));



  // 将各种参数打包并添加到 packed_ 中

  packed_.emplace_back(
      output_min ? output_min->template to<float>()
                 : -std::numeric_limits<float>::infinity());
  packed_.emplace_back(
      output_max ? output_max->template to<float>()
                 : +std::numeric_limits<float>::infinity());



  // 根据条件将 output_min 和 output_max 转换为 float 并添加到 packed_ 中

  packed_.emplace_back(method);
  packed_.emplace_back(weight.sizes().vec());



  // 将卷积方法和权重的大小添加到 packed_ 中

  compute_shader_ = conv2d::get_shader(
      weight.sizes(), stride, padding, dilation, method, transposed, quantized);



  // 根据给定参数获取卷积的着色器，并赋值给 compute_shader_

  if (!at::globalContext().releaseWeightsWhenPrepacking()) {



  // 如果全局上下文不释放预打包时的权重，则执行以下操作

    unpacked_.reserve(Unpacked::NumArgs);



  // 为 unpacked_ 分配足够的空间以容纳 Unpacked::NumArgs 个元素，并初始化

    unpacked_.emplace_back(weight);
    unpacked_.emplace_back(bias);
    unpacked_.emplace_back(stride_arg.vec());
    unpacked_.emplace_back(padding_arg.vec());
    unpacked_.emplace_back(dilation_arg.vec());
    unpacked_.emplace_back(transposed);
    unpacked_.emplace_back(quantized);



  // 将权重、偏置和各种参数添加到 unpacked_ 中
    # 将 output_padding_arg.vec() 的内容作为新元素添加到 unpacked_ 中
    unpacked_.emplace_back(output_padding_arg.vec());
    # 将 groups 添加到 unpacked_ 中作为新元素
    unpacked_.emplace_back(groups);
    # 将 output_min 添加到 unpacked_ 中作为新元素
    unpacked_.emplace_back(output_min);
    # 将 output_max 添加到 unpacked_ 中作为新元素
    unpacked_.emplace_back(output_max);
}

// 创建一个已打包的卷积上下文对象，将未打包的参数列表转换为打包后的上下文对象
Conv2dPackedContext Conv2dPackedContext::pack(c10::impl::GenericList unpacked) {
  // 返回一个新的Conv2dPackedContext对象，使用未打包的参数列表中的数据进行初始化
  return Conv2dPackedContext(
      unpacked.get(Unpacked::Weight).toTensor(),  // 获取并转换权重张量
      get_optional_tensor(unpacked, Unpacked::Bias),  // 获取可选的偏置张量
      unpacked.get(Unpacked::Stride).toIntVector(),  // 获取并转换步长向量
      unpacked.get(Unpacked::Padding).toIntVector(),  // 获取并转换填充向量
      unpacked.get(Unpacked::Dilation).toIntVector(),  // 获取并转换膨胀向量
      unpacked.get(Unpacked::isTransposed).toBool(),  // 获取并转换是否转置的标志
      unpacked.get(Unpacked::isQuantized).toBool(),  // 获取并转换是否量化的标志
      unpacked.get(Unpacked::OutputPadding).toIntVector(),  // 获取并转换输出填充向量
      unpacked.get(Unpacked::Groups).toInt(),  // 获取并转换分组数
      get_optional_scalar(unpacked, Unpacked::OutputMin),  // 获取可选的输出最小标量
      get_optional_scalar(unpacked, Unpacked::OutputMax));  // 获取可选的输出最大标量
}

// 创建卷积上下文对象的函数，用于标准卷积
c10::intrusive_ptr<Conv2dPackedContext> create_conv2d_context(
    Tensor&& weight,
    std::optional<Tensor>&& bias,
    std::vector<int64_t>&& stride,
    std::vector<int64_t>&& padding,
    std::vector<int64_t>&& dilation,
    const int64_t groups,
    const std::optional<Scalar>& output_min,
    const std::optional<Scalar>& output_max) {
  // 使用给定的参数创建并返回一个标准卷积的打包后的上下文对象
  return c10::make_intrusive<Conv2dPackedContext>(Conv2dPackedContext(
      weight,
      bias,
      stride,
      padding,
      dilation,
      /* transposed = */ false,  // 设置为非转置卷积
      /* quantized = */ false,  // 设置为非量化卷积
      /* output_padding_arg = */ {0},  // 输出填充为零
      groups,
      output_min,
      output_max));
}

// 创建卷积上下文对象的函数，用于转置卷积
c10::intrusive_ptr<Conv2dPackedContext> create_tconv2d_context(
    Tensor&& weight,
    std::optional<Tensor>&& bias,
    std::vector<int64_t>&& stride,
    std::vector<int64_t>&& padding,
    std::vector<int64_t>&& output_padding,
    std::vector<int64_t>&& dilation,
    const int64_t groups,
    const std::optional<Scalar>& output_min,
    const std::optional<Scalar>& output_max) {
  // 使用给定的参数创建并返回一个转置卷积的打包后的上下文对象
  return c10::make_intrusive<Conv2dPackedContext>(Conv2dPackedContext(
      weight,
      bias,
      stride,
      padding,
      dilation,
      /* transposed = */ true,  // 设置为转置卷积
      /* quantized = */ false,  // 设置为非量化卷积
      output_padding,  // 使用给定的输出填充向量
      groups,
      output_min,
      output_max));
}

// 创建卷积上下文对象的函数，用于量化卷积
c10::intrusive_ptr<Conv2dPackedContext> create_qconv2d_context(
    Tensor&& weight,
    std::optional<Tensor>&& bias,
    std::vector<int64_t>&& stride,
    std::vector<int64_t>&& padding,
    std::vector<int64_t>&& dilation,
    const int64_t groups,
    const std::optional<Scalar>& output_min,
    const std::optional<Scalar>& output_max) {
  // 使用给定的参数创建并返回一个量化卷积的打包后的上下文对象
  return c10::make_intrusive<Conv2dPackedContext>(Conv2dPackedContext(
      weight,
      bias,
      stride,
      padding,
      dilation,
      /* transposed = */ false,  // 设置为非转置卷积
      /* quantized = */ true,  // 设置为量化卷积
      /* output_padding_arg = */ {0},  // 输出填充为零
      groups,
      output_min,
      output_max));
}

// 创建卷积上下文对象的函数，用于量化转置卷积
c10::intrusive_ptr<Conv2dPackedContext> create_qtconv2d_context(
    Tensor&& weight,
    std::optional<Tensor>&& bias,
    std::vector<int64_t>&& stride,
    std::vector<int64_t>&& padding,
    std::vector<int64_t>&& output_padding,
    std::vector<int64_t>&& dilation,
    const int64_t groups,
    const std::optional<Scalar>& output_min,
    const std::optional<Scalar>& output_max) {
  // 使用给定的参数创建并返回一个量化转置卷积的打包后的上下文对象
  return c10::make_intrusive<Conv2dPackedContext>(Conv2dPackedContext(
      weight,
      bias,
      stride,
      padding,
      dilation,
      /* transposed = */ true,  // 设置为转置卷积
      /* quantized = */ true,  // 设置为量化卷积
      output_padding,  // 使用给定的输出填充向量
      groups,
      output_min,
      output_max));
}
    const std::optional<Scalar>& output_max) {

# 接收函数参数，包括输出最大值的可选标量引用

  return c10::make_intrusive<Conv2dPackedContext>(Conv2dPackedContext(
      weight,
      bias,
      stride,
      padding,
      dilation,
      /* transposed = */ true,
      /* quantized = */ true,
      output_padding,
      groups,
      output_min,
      output_max));

# 使用给定的参数创建 Conv2dPackedContext 对象，并返回该对象的智能指针
  // 这里开始一个新的代码块
  std::vector<int64_t> output_size;
  // 如果是反卷积操作，则计算输出大小
  if (transposed) {
    output_size = get_conv_transpose_output_size(
        v_input.sizes(),
        kernel_size,
        padding,
        output_padding,
        stride,
        dilation);
  } else {
    // 否则，计算正常卷积的输出大小
    output_size = conv_output_size(
        v_input.sizes(), kernel_size, padding, stride, dilation);
  }

  // 使用 Vulkan context 创建输出张量 v_output，与输入张量的大小和数据类型相同
  vTensor v_output{
      context,
      output_size,
      v_input.dtype(),
  };

  // 如果张量是量化的，则设置相应的量化参数
  if (quantized) {
    v_output.set_is_quantized();
    v_output.set_scale(scale);
    v_output.set_zero_point(zero_point);
  }

  // 如果张量是量化的，则继续进行以下操作
  if (quantized) {
    # 如果是量化操作，调用记录量化卷积操作的函数；否则调用记录普通卷积操作的函数
    if (quantized) {
        conv2d::record_quantized_op(
            context,
            conv_context->compute_shader(),
            v_output,
            v_input,
            v_weight,
            v_bias,
            overlay_region,
            stride,
            padding,
            dilation,
            output_min,
            output_max,
            kernel_size,
            method_,
            transposed);
    } else {
        conv2d::record_op(
            context,
            conv_context->compute_shader(),
            v_output,
            v_input,
            v_weight,
            v_bias,
            overlay_region,
            stride,
            padding,
            dilation,
            output_min,
            output_max,
            kernel_size,
            method_,
            transposed);
    }
    
    # 将输出张量转换并返回
    return convert(v_output);
}

/* 
   根据输入参数和卷积上下文执行卷积操作，并返回结果张量。
   这里默认使用1.0的缩放因子和0的零点值。
*/
Tensor run_conv2d_context(
    const Tensor& input_arg,
    const c10::intrusive_ptr<Conv2dPackedContext>& conv_context) {
  return run_conv2d_context_impl(input_arg, conv_context, 1.0f, 0u);
}

/* 
   根据输入参数和卷积上下文执行转置卷积操作，并返回结果张量。
   这里默认使用1.0的缩放因子和0的零点值。
*/
Tensor run_tconv2d_context(
    const Tensor& input_arg,
    const c10::intrusive_ptr<Conv2dPackedContext>& conv_context) {
  return run_conv2d_context_impl(input_arg, conv_context, 1.0f, 0u);
}

/* 
   根据输入参数、缩放因子和零点值以及卷积上下文执行量化卷积操作，并返回结果张量。
*/
Tensor run_qconv2d_context(
    const Tensor& input_arg,
    double scale,
    int64_t zero_point,
    const c10::intrusive_ptr<Conv2dPackedContext>& conv_context) {
  return run_conv2d_context_impl(input_arg, conv_context, scale, zero_point);
}

/* 
   执行量化卷积操作的函数，接受输入、权重、可选偏置等参数，并返回结果张量。
*/
Tensor quantized_conv2d(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    double out_scale,
    int64_t out_zero_point) {
  return quantized_convolution(
      input,
      weight,
      bias,
      stride,
      padding,
      dilation,
      false,
      {{0, 0}},
      groups,
      out_scale,
      out_zero_point);
}

/* 
   以下是对旧版本兼容的类和函数定义
*/

/* 
   构造函数，使用给定的卷积上下文初始化。
*/
Conv2dOpContext::Conv2dOpContext(Conv2dPackedContext conv_context)
    : conv_context_{std::move(conv_context)} {}

/* 
   静态方法，创建一个卷积上下文对象，根据给定的参数初始化。
*/
Conv2dOpContext Conv2dOpContext::create(
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    const IntArrayRef stride_arg,
    const IntArrayRef padding_arg,
    const IntArrayRef dilation_arg,
    const bool transposed,
    const IntArrayRef output_padding_arg,
    const int64_t groups,
    const std::optional<Scalar>& output_min,
    const std::optional<Scalar>& output_max) {
  return Conv2dOpContext{Conv2dPackedContext(
      weight,
      bias,
      stride_arg,
      padding_arg,
      dilation_arg,
      transposed,
      /* quantized = */ false,
      output_padding_arg,
      groups,
      output_min,
      output_max)};
}

/* 
   执行卷积操作，根据内部的卷积上下文对象和输入张量返回结果张量。
*/
Tensor Conv2dOpContext::run(const Tensor& input_arg) const {
  return run_conv2d_context(
      input_arg, c10::make_intrusive<Conv2dPackedContext>(conv_context_));
}

/* 
   解包卷积上下文对象，返回其内部存储的卷积参数状态。
*/
Conv2dOpContext::State Conv2dOpContext::unpack() const {
  const c10::impl::GenericList unpacked_ = conv_context_.unpack();

  TORCH_CHECK(unpacked_.size() > 0u, "unpacked_ does not have any elements!");

  return Conv2dOpContext::State(
      unpacked_.get(Conv2dPackedContext::Unpacked::Weight).toTensor(),
      get_optional_tensor(unpacked_, Conv2dPackedContext::Unpacked::Bias),
      unpacked_.get(Conv2dPackedContext::Unpacked::Stride).toIntVector(),
      unpacked_.get(Conv2dPackedContext::Unpacked::Padding).toIntVector(),
      unpacked_.get(Conv2dPackedContext::Unpacked::Dilation).toIntVector(),
      unpacked_.get(Conv2dPackedContext::Unpacked::Groups).toInt(),
      get_optional_scalar(unpacked_, Conv2dPackedContext::Unpacked::OutputMin),
      get_optional_scalar(unpacked_, Conv2dPackedContext::Unpacked::OutputMax));
}

/* 
   创建卷积上下文对象的函数，使用给定的权重进行初始化。
*/
c10::intrusive_ptr<Conv2dOpContext> conv2d_clamp_prepack(
    Tensor&& weight,
    // 使用 std::move 将权重 Tensor 转移至新的上下文对象
    std::optional<Tensor>&& bias,
    // 使用 std::move 将偏置 Tensor 转移至新的上下文对象
    std::vector<int64_t>&& stride,
    // 使用 std::move 将步幅向量转移至新的上下文对象
    std::vector<int64_t>&& padding,
    // 使用 std::move 将填充向量转移至新的上下文对象
    std::vector<int64_t>&& dilation,
    // 将分组数传递给新的上下文对象
    const int64_t groups,
    // 将输出最小值作为可选的标量传递给新的上下文对象
    const std::optional<Scalar>& output_min,
    // 将输出最大值作为可选的标量传递给新的上下文对象
    const std::optional<Scalar>& output_max) {
  // 返回一个 Conv2dOpContext 指针，使用 make_intrusive 创建
  return c10::make_intrusive<Conv2dOpContext>(Conv2dOpContext::create(
      std::move(weight),
      std::move(bias),
      std::move(stride),
      std::move(padding),
      std::move(dilation),
      /* transposed = */ false,
      /* output_padding = */ {0},
      groups,
      output_min,
      output_max));
}
}

Tensor conv2d_clamp_run(
    const Tensor& input,
    const c10::intrusive_ptr<Conv2dOpContext>& context) {
  return context->run(input);
}
    // 使用传入的 context 对象从预打包的卷积上下文中获取权重 Tensor
    const Tensor weight =
        context->get_val(Conv1dPackedContext::Packed::Weight).toTensor();
    // 使用传入的 context 对象从预打包的卷积上下文中获取可选的偏置 Tensor
    const std::optional<Tensor>& bias_opt =
        context->get_val(Conv1dPackedContext::Packed::Bias).toTensor();
    // 使用传入的 context 对象从预打包的卷积上下文中获取步长（stride）的整数向量
    const auto stride =
        context->get_val(Conv1dPackedContext::Packed::Stride).toIntVector();
    // 使用传入的 context 对象从预打包的卷积上下文中获取填充（padding）的整数向量
    const auto padding =
        context->get_val(Conv1dPackedContext::Packed::Padding).toIntVector();
    // 使用传入的 context 对象从预打包的卷积上下文中获取扩展率（dilation）的整数向量
    const auto dilation =
        context->get_val(Conv1dPackedContext::Packed::Dilation).toIntVector();
    // 使用传入的 context 对象从预打包的卷积上下文中获取分组（groups）的整数值
    const auto groups =
        context->get_val(Conv1dPackedContext::Packed::Groups).toInt();
    // 调用 conv1d::run_conv1d_context_impl 函数，传递获取的参数进行卷积运算
    return conv1d::run_conv1d_context_impl(
        input, weight, bias_opt, stride, padding, dilation, groups);
} // 结束 TORCH_LIBRARY_IMPL(aten, Vulkan, m) 命名空间的实现

// 在 m 模块中注册 convolution_overrideable 实现为 convolution 函数
m.impl("convolution_overrideable", convolution);

// 在 m 模块中注册 TORCH_SELECTIVE_NAME("aten::conv1d") 实现为 convolution1d 函数
m.impl(TORCH_SELECTIVE_NAME("aten::conv1d"), TORCH_FN(convolution1d));

} // 结束 ops 命名空间
} // 结束 vulkan 命名空间
} // 结束 native 命名空间
} // 结束 at 命名空间
```