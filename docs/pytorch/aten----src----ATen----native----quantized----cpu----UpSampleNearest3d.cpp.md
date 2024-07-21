# `.\pytorch\aten\src\ATen\native\quantized\cpu\UpSampleNearest3d.cpp`

```
// 定义预处理宏，用于指定只包含方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 引入 ATen 库中的核心 Tensor 类和 Dispatch 头文件
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
// 引入 ATen 库中的 UpSample 相关头文件
#include <ATen/native/UpSample.h>

// 根据是否定义了 AT_PER_OPERATOR_HEADERS 来选择性地引入不同的 ATen 函数头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_empty_affine_quantized.h>
#include <ATen/ops/_upsample_nearest_exact3d_native.h>
#include <ATen/ops/upsample_nearest3d_native.h>
#endif

// 引入 C++ 标准库中的实用工具 irange
#include <c10/util/irange.h>
// 引入 C 标准库中的字符串处理函数
#include <cstring>

// 在 at 命名空间下定义 native 子命名空间
namespace at {
namespace native {

// 定义一个函数指针类型 nn_compute_source_index_fn_t，用于调度 nearest_idx 或 nearest_exact_idx 函数
typedef int64_t (*nn_compute_source_index_fn_t)(const float, int64_t, int64_t);

// 定义一个模板函数 upsample_nearest3d_out_frame，用于进行三维最近邻上采样
template <typename scalar_t, nn_compute_source_index_fn_t nn_compute_source_index_fn>
static void upsample_nearest3d_out_frame(
    scalar_t* odata,
    scalar_t* idata,
    int64_t input_depth,
    int64_t input_height,
    int64_t input_width,
    int64_t output_depth,
    int64_t output_height,
    int64_t output_width,
    int64_t nbatch,
    int64_t channels,
    std::optional<double> scales_d,
    std::optional<double> scales_h,
    std::optional<double> scales_w) {
  
  // 计算深度、高度和宽度的缩放比例
  float depth_scale = compute_scales_value<float>(scales_d, input_depth, output_depth);
  float height_scale = compute_scales_value<float>(scales_h, input_height, output_height);
  float width_scale = compute_scales_value<float>(scales_w, input_width, output_width);

  // 计算总的通道数
  channels = channels * nbatch;

  // 如果通道数为 0 或者输出的深度、高度、宽度有任何一个为 0，则直接返回
  if (channels == 0 || output_depth == 0 || output_height == 0 || output_width == 0) {
    return;
  }

  // 将输入数据 idata 和输出数据 odata 转换为 scalar_t 类型的指针
  auto* i_p = reinterpret_cast<typename scalar_t::underlying*>(idata);
  auto* o_p = reinterpret_cast<typename scalar_t::underlying*>(odata);

  // 特殊情况：输入和输出的深度、高度、宽度都相等时，直接进行内存拷贝
  if (input_depth == output_depth && input_height == output_height && input_width == output_width) {
    std::memcpy(o_p, i_p, channels * input_depth * input_height * input_width * sizeof(typename scalar_t::underlying));
    return;
  }

  // 遍历输出的深度维度
  for (const auto d2 : c10::irange(output_depth)) {
    // 计算对应于输入深度维度的索引 d1
    const int64_t d1 = nn_compute_source_index_fn(depth_scale, d2, input_depth);

    // 遍历输出的高度维度
    for (const auto h2 : c10::irange(output_height)) {
      // 计算对应于输入高度维度的索引 h1
      const int64_t h1 = nn_compute_source_index_fn(height_scale, h2, input_height);

      // 遍历输出的宽度维度
      for (const auto w2 : c10::irange(output_width)) {
        // 计算对应于输入宽度维度的索引 w1
        const int64_t w1 = nn_compute_source_index_fn(width_scale, w2, input_width);

        // 计算输入数据中的位置 pos1 和输出数据中的位置 pos2
        const auto* pos1 = &i_p[d1 * input_height * input_width + h1 * input_width + w1];
        auto* pos2 = &o_p[d2 * output_height * output_width + h2 * output_width + w2];

        // 遍历通道数
        for (C10_UNUSED const auto c : c10::irange(channels)) {
          pos2[0] = pos1[0];
          pos1 += input_depth * input_height * input_width;
          pos2 += output_depth * output_height * output_width;
        }
      }
    }
  }
}

// 定义另一个模板函数 upsample_nearest3d_out_frame_nhwc，用于进行 NHWC 格式的三维最近邻上采样
template <typename scalar_t, nn_compute_source_index_fn_t nn_compute_source_index_fn>
static void upsample_nearest3d_out_frame_nhwc(
    scalar_t* odata,


注释部分已经包含了详细的代码解释和注释，每一行代码的作用都得到了说明。
    // 使用指针 idata 指向输入数据的起始地址，数据类型为 scalar_t
    scalar_t* idata,
    // 输入数据的深度、高度和宽度
    int64_t input_depth,
    int64_t input_height,
    int64_t input_width,
    // 输出数据的深度、高度和宽度
    int64_t output_depth,
    int64_t output_height,
    int64_t output_width,
    // 批处理大小
    int64_t nbatch,
    // 通道数
    int64_t channels,
    // 可选参数，用于缩放的系数：深度、高度、宽度
    std::optional<double> scales_d,
    std::optional<double> scales_h,
    std::optional<double> scales_w) {
  
  // 计算深度、高度、宽度的缩放比例
  float depth_scale = compute_scales_value<float>(scales_d, input_depth, output_depth);
  float height_scale = compute_scales_value<float>(scales_h, input_height, output_height);
  float width_scale = compute_scales_value<float>(scales_w, input_width, output_width);

  // 遍历每个批次的数据
  for (const auto b : c10::irange(nbatch)) {
    // 获取当前批次的输入和输出数据的起始地址
    auto* i_p = reinterpret_cast<typename scalar_t::underlying*>(idata + b * input_depth * input_height * input_width * channels);
    auto* o_p = reinterpret_cast<typename scalar_t::underlying*>(odata + b * output_depth * output_height * output_width * channels);
    
    // 特殊情况：如果输入和输出的尺寸相同，直接进行内存拷贝并返回
    if (input_depth == output_depth && input_height == output_height && input_width == output_width) {
      std::memcpy(o_p, i_p, channels * input_depth * input_height * input_width * sizeof(typename scalar_t::underlying));
      return;
    }

    // 遍历输出数据的深度
    for (const auto d2 : c10::irange(output_depth)) {
      // 根据深度缩放比例计算对应的源索引
      const int64_t d1 = nn_compute_source_index_fn(depth_scale, d2, input_depth);
      
      // 遍历输出数据的高度
      for (const auto h2 : c10::irange(output_height)) {
        // 根据高度缩放比例计算对应的源索引
        const int64_t h1 = nn_compute_source_index_fn(height_scale, h2, input_height);
        
        // 遍历输出数据的宽度
        for (const auto w2 : c10::irange(output_width)) {
          // 根据宽度缩放比例计算对应的源索引
          const int64_t w1 = nn_compute_source_index_fn(width_scale, w2, input_width);

          // 计算输入数据和输出数据在内存中的位置指针
          const auto* pos1 = &i_p[(d1 * input_height * input_width + h1 * input_width + w1)*channels];
          auto* pos2 = &o_p[(d2 * output_height * output_width + h2 * output_width + w2)*channels];
          
          // 进行内存拷贝，复制对应位置的数据
          std::memcpy(pos2, pos1, channels * sizeof(typename scalar_t::underlying));
        }
      }
    }
  }
}
// 定义一个模板函数 _upsample_nearest3d_quantized_cpu，接受一个类型为 nn_compute_source_index_fn_t 的模板参数
Tensor _upsample_nearest3d_quantized_cpu(
    // 输入张量，表示原始数据
    const Tensor& input,
    // 输出大小的数组，期望为三维
    IntArrayRef output_size,
    // 可选参数，深度方向的缩放比例
    std::optional<double> scales_d,
    // 可选参数，高度方向的缩放比例
    std::optional<double> scales_h,
    // 可选参数，宽度方向的缩放比例
    std::optional<double> scales_w) {
  // 检查输出大小数组是否为三维
  TORCH_CHECK(
      output_size.size() == 3,
      "It is expected output_size equals to 3, but got size ",
      output_size.size());

  // 检查输入张量不为空且维度为5
  TORCH_CHECK(
      input.numel() != 0 && input.dim() == 5,
      "Non-empty 5D data tensor expected but got a tensor with sizes ",
      input.sizes());

  // 提取输出大小的三个维度
  int64_t output_depth = output_size[0];
  int64_t output_height = output_size[1];
  int64_t output_width = output_size[2];

  // 提取输入张量的各个维度大小
  int64_t nbatch = input.size(0);
  int64_t channels = input.size(1);
  int64_t input_depth = input.size(2);
  int64_t input_height = input.size(3);
  int64_t input_width = input.size(4);
  
  // 断言输入张量的宽度和输出宽度大于0
  AT_ASSERT(input_width > 0 && output_width > 0);
  
  // 如果输入张量以 ChannelsLast3d 内存格式存储
  if (input.is_contiguous(c10::MemoryFormat::ChannelsLast3d)) {
    // 创建一个空的仿射量化张量作为输出
    Tensor output = at::_empty_affine_quantized(
        {nbatch, channels, output_depth, output_height, output_width},
        // 选择输入的推荐内存格式作为输出的内存格式
        input.options().memory_format(input.suggest_memory_format()),
        // 使用输入张量的量化缩放因子
        input.q_scale(),
        // 使用输入张量的量化零点
        input.q_zero_point(),
        // 无额外的量化参数
        c10::nullopt);

    // 根据输入张量的标量类型调度操作 upsample_nearest3d，并执行
    AT_DISPATCH_QINT_TYPES(input.scalar_type(), "upsample_nearest3d", [&] {
      auto* idata = static_cast<scalar_t*>(input.data_ptr());
      auto* odata = static_cast<scalar_t*>(output.data_ptr());
      // 调用 NHWC 格式下的 nearest 三维上采样方法
      upsample_nearest3d_out_frame_nhwc<scalar_t, nn_compute_source_index_fn>(
          odata,
          idata,
          input_depth,
          input_height,
          input_width,
          output_depth,
          output_height,
          output_width,
          nbatch,
          channels,
          scales_d,
          scales_h,
          scales_w);
    });
    // 返回生成的输出张量
    return output;
  } else {
    // 如果输入张量不是以 ChannelsLast3d 内存格式存储

    // 创建一个空的仿射量化张量作为输出
    Tensor output = at::_empty_affine_quantized(
        {nbatch, channels, output_depth, output_height, output_width},
        // 使用输入张量的选项作为输出的选项
        input.options(),
        // 使用输入张量的量化缩放因子
        input.q_scale(),
        // 使用输入张量的量化零点
        input.q_zero_point());

    // 强制使输入张量连续化
    auto input_contig = input.contiguous();

    // 根据连续化后的输入张量的标量类型调度操作 upsample_nearest3d，并执行
    AT_DISPATCH_QINT_TYPES(input_contig.scalar_type(), "upsample_nearest3d", [&] {
      auto* idata = static_cast<scalar_t*>(input_contig.data_ptr());
      auto* odata = static_cast<scalar_t*>(output.data_ptr());
      // 调用非 NHWC 格式下的 nearest 三维上采样方法
      upsample_nearest3d_out_frame<scalar_t, nn_compute_source_index_fn>(
          odata,
          idata,
          input_depth,
          input_height,
          input_width,
          output_depth,
          output_height,
          output_width,
          nbatch,
          channels,
          scales_d,
          scales_h,
          scales_w);
    });
    // 返回生成的输出张量
    return output;
  }
}
    // 使用量化的最近邻三维上采样方法来对输入进行处理
    return _upsample_nearest3d_quantized_cpu<nearest_neighbor_compute_source_index>(
        input, osize, scale_d, scale_h, scale_w);
}

Tensor _upsample_nearest_exact3d_quantized_cpu(
    const Tensor& input,
    IntArrayRef osize,
    std::optional<double> scale_d,
    std::optional<double> scale_h,
    std::optional<double> scale_w) {
  // 调用 _upsample_nearest3d_quantized_cpu 函数，使用精确的最近邻插值方法，
  // 并返回其结果
  return _upsample_nearest3d_quantized_cpu<nearest_neighbor_exact_compute_source_index>(
      input, osize, scale_d, scale_h, scale_w);
}

// 结束 native 命名空间
} // namespace native

// 结束 at 命名空间
} // namespace at
```