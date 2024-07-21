# `.\pytorch\aten\src\ATen\native\quantized\cpu\UpSampleNearest2d.cpp`

```
// 定义宏，仅使用方法操作符进行断言
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 包含张量相关头文件
#include <ATen/core/Tensor.h>
// 包含分发相关头文件
#include <ATen/Dispatch.h>
// 包含并行计算相关头文件
#include <ATen/Parallel.h>
// 包含上采样相关头文件
#include <ATen/native/UpSample.h>
// 包含 CPU 工具函数相关头文件
#include <ATen/native/cpu/utils.h>

// 如果未定义 AT_PER_OPERATOR_HEADERS，则包含通用函数和原生函数的头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
// 否则，包含特定的头文件用于量化和最近邻插值的操作
#else
#include <ATen/ops/_empty_affine_quantized.h>
#include <ATen/ops/_upsample_nearest_exact2d_native.h>
#include <ATen/ops/upsample_nearest2d_native.h>
#endif

// 包含 C10 库的工具函数，用于范围遍历
#include <c10/util/irange.h>

// 包含 C 标准库的字符串操作相关头文件
#include <cstring>

// 在 at 命名空间下定义 native 命名空间
namespace at {
namespace native {

// 定义一个函数指针类型，用于分发至 nearest_idx 或 nearest_exact_idx 函数
typedef int64_t (*nn_compute_source_index_fn_t)(const float, int64_t, int64_t);

// 定义模板函数，用于 2D 最近邻上采样的输出帧处理
template <typename scalar_t, nn_compute_source_index_fn_t nn_compute_source_index_fn>
static void upsample_nearest2d_out_frame(
    scalar_t* odata,
    scalar_t* idata,
    int64_t input_height,
    int64_t input_width,
    int64_t output_height,
    int64_t output_width,
    int64_t nbatch,
    int64_t channels,
    std::optional<double> scales_h,
    std::optional<double> scales_w) {
  // 计算高度和宽度的缩放比例
  float height_scale = compute_scales_value<float>(scales_h, input_height, output_height);
  float width_scale = compute_scales_value<float>(scales_w, input_width, output_width);

  // 计算通道数（扩展为批次数 * 通道数）
  channels = channels * nbatch;
  // 如果通道数为 0 或输出高度/宽度为 0，则直接返回
  if (channels == 0 || output_height == 0 || output_width == 0) {
    return;
  }
  // 将输入和输出数据转换为底层标量类型的指针
  auto* i_p = reinterpret_cast<typename scalar_t::underlying*>(idata);
  auto* o_p = reinterpret_cast<typename scalar_t::underlying*>(odata);

  // 特殊情况：如果输入和输出尺寸相同，则直接进行内存拷贝
  if (input_height == output_height && input_width == output_width) {
    std::memcpy(o_p, i_p, channels * input_height * input_width * sizeof(typename scalar_t::underlying));
    return;
  }

  // 分配输出宽度的输入偏移数组
  std::unique_ptr<int64_t []> input_offset_arr(new int64_t[output_width]);
  int64_t* input_offset = input_offset_arr.get();

  // 遍历输出宽度，计算每个输出位置对应的输入位置
  for (const auto w2 : c10::irange(output_width)) {
    const int64_t w1 = nn_compute_source_index_fn(width_scale, w2, input_width);
    input_offset[w2] = w1;
  }

  // 计算并行执行的任务粒度
  int64_t grain_size = internal::GRAIN_SIZE / std::max(int64_t{1}, output_width);
  // 并行处理每个通道和输出高度的数据
  at::parallel_for(0, channels * output_height, grain_size, [&](int64_t begin, int64_t end) {
    int64_t nc{0}, h2{0};
    // 初始化数据索引
    data_index_init(begin, nc, channels, h2, output_height);

    for (const auto i : c10::irange(begin, end)) {
      // 计算每行的输入索引 h1
      const int64_t h1 = nn_compute_source_index_fn(height_scale, h2, input_height);
      // 获取当前位置的输入和输出指针
      const auto* pos1 = &i_p[nc * input_height * input_width + h1 * input_width];
      auto* pos2 = &o_p[i * output_width];

      // 复制对应位置的数据
      for (const auto w2 : c10::irange(output_width)) {
        const int64_t w1 = input_offset[w2];
        pos2[w2] = pos1[w1];
      }

      // 更新数据索引
      data_index_step(nc, channels, h2, output_height);
    }
  });
}

// 模板函数，处理 NHWC 格式的 2D 最近邻上采样输出帧
template <typename scalar_t, nn_compute_source_index_fn_t nn_compute_source_index_fn>
static void upsample_nearest2d_out_frame_nhwc(
    scalar_t* odata,
    scalar_t* idata,
    // 定义一个函数，执行图像缩放操作，接受输入和输出的尺寸，批处理大小，通道数，
    // 以及可选的高度和宽度缩放因子
    int64_t input_height,
    int64_t input_width,
    int64_t output_height,
    int64_t output_width,
    int64_t nbatch,
    int64_t channels,
    std::optional<double> scales_h,
    std::optional<double> scales_w) {
  // 计算高度和宽度的缩放比例
  float height_scale = compute_scales_value<float>(scales_h, input_height, output_height);
  float width_scale = compute_scales_value<float>(scales_w, input_width, output_width);

  // 并行处理每个批次的输出像素
  at::parallel_for(0, nbatch * output_height * output_width, 0, [&](int64_t begin, int64_t end) {
    // 初始化变量b、h2、w2，用于索引数据
    int64_t b{0}, h2{0}, w2{0};
    // 初始化数据索引，确定当前像素在输入和输出中的位置
    data_index_init(begin, b, nbatch, h2, output_height, w2, output_width);

    // 遍历当前分块内的所有像素
    for (const auto i : c10::irange(begin, end)) {
      // 计算输入和输出数据的地址偏移
      auto* i_p = reinterpret_cast<typename scalar_t::underlying*>(idata + b * input_height * input_width * channels);
      auto* o_p = reinterpret_cast<typename scalar_t::underlying*>(odata + i * channels);

      // 计算当前输出像素在输入图像中的源像素位置
      const int64_t h1 = nn_compute_source_index_fn(height_scale, h2, input_height);
      const int64_t w1 = nn_compute_source_index_fn(width_scale, w2, input_width);

      // 指向输入和输出数据中的对应像素位置
      const auto* pos1 = &i_p[(h1 * input_width + w1) * channels];
      auto* pos2 = &o_p[0];
      // 执行数据拷贝操作，复制源像素到目标像素位置
      std::memcpy(pos2, pos1, channels * sizeof(typename scalar_t::underlying));

      // 更新数据索引，准备处理下一个像素
      data_index_step(b, nbatch, h2, output_height, w2, output_width);
    }
  });
}



template <nn_compute_source_index_fn_t nn_compute_source_index_fn>
// 定义一个模板函数，接受一个计算源索引函数作为模板参数
Tensor _upsample_nearest2d_quantized_cpu(
    const Tensor& input,
    IntArrayRef output_size,
    std::optional<double> scales_h,
    std::optional<double> scales_w) {
  // 输入参数检查，确保输出大小是二维的
  TORCH_CHECK(
      output_size.size() == 2,
      "It is expected output_size equals to 2, but got size ",
      output_size.size());

  // 输入参数检查，确保输入张量是非空的四维数据张量
  TORCH_CHECK(
      input.dim() == 4,
      "Non-empty 4D data tensor expected but got a tensor with sizes ",
      input.sizes());

  // 获取输出的高度和宽度
  int64_t output_height = output_size[0];
  int64_t output_width = output_size[1];

  // 获取输入张量的批次大小、通道数、输入高度和宽度
  int64_t nbatch = input.size(0);
  int64_t channels = input.size(1);
  int64_t input_height = input.size(2);
  int64_t input_width = input.size(3);
  // 断言输入宽度和输出宽度大于零
  AT_ASSERT(input_width > 0 && output_width > 0);

  // 如果输入张量以通道最后的内存格式存储
  if (input.is_contiguous(c10::MemoryFormat::ChannelsLast)) {
    // 创建一个空的仿射量化张量作为输出，保持输入的内存格式和量化参数
    Tensor output = at::_empty_affine_quantized(
        {nbatch, channels, output_height, output_width},
        input.options().memory_format(input.suggest_memory_format()),
        input.q_scale(),
        input.q_zero_point(),
        c10::nullopt);

    // 特殊情况：直接复制输入到输出
    if (input_height == output_height && input_width == output_width) {
      output.copy_(input);
      return output;
    }

    // 使用模板分发调用最近邻插值的计算函数
    AT_DISPATCH_QINT_TYPES(input.scalar_type(), "upsample_nearest2d", [&] {
      auto* idata = static_cast<scalar_t*>(input.data_ptr());
      auto* odata = static_cast<scalar_t*>(output.data_ptr());
      upsample_nearest2d_out_frame_nhwc<scalar_t, nn_compute_source_index_fn>(
          odata,
          idata,
          input_height,
          input_width,
          output_height,
          output_width,
          nbatch,
          channels,
          scales_h,
          scales_w);
    });
    return output;
  } else {
    // 创建一个空的仿射量化张量作为输出，使用默认选项和输入的量化参数
    Tensor output = at::_empty_affine_quantized(
        {nbatch, channels, output_height, output_width},
        input.options(),
        input.q_scale(),
        input.q_zero_point());

    // 对输入进行连续化处理
    auto input_contig = input.contiguous();

    // 使用模板分发调用最近邻插值的计算函数
    AT_DISPATCH_QINT_TYPES(input_contig.scalar_type(), "upsample_nearest2d", [&] {
      auto* idata = static_cast<scalar_t*>(input_contig.data_ptr());
      auto* odata = static_cast<scalar_t*>(output.data_ptr());
      upsample_nearest2d_out_frame<scalar_t, nn_compute_source_index_fn>(
          odata,
          idata,
          input_height,
          input_width,
          output_height,
          output_width,
          nbatch,
          channels,
          scales_h,
          scales_w);
    });
    return output;
  }
}

// 使用最近邻计算源索引函数作为参数调用 _upsample_nearest2d_quantized_cpu
Tensor upsample_nearest2d_quantized_cpu(
    const Tensor& input,
    IntArrayRef osize,
    std::optional<double> scale_h,
    std::optional<double> scale_w) {
  return _upsample_nearest2d_quantized_cpu<nearest_neighbor_compute_source_index>(input, osize, scale_h, scale_w);
}

// 下一个函数的定义未提供，无法继续解释
Tensor _upsample_nearest_exact2d_quantized_cpu(



using at::native::upsample::compute_output_size;
using at::native::upsample::get_scale_value;

// 代码片段结束
    // 接受四个参数：输入张量 input，目标输出大小 osize，可选的高度缩放比例 scale_h，可选的宽度缩放比例 scale_w
    const Tensor& input,
    IntArrayRef osize,
    std::optional<double> scale_h,
    std::optional<double> scale_w) {
    // 调用 _upsample_nearest2d_quantized_cpu 函数进行最近邻插值上采样，传入参数 input 作为输入张量，
    // osize 作为目标输出大小，scale_h 和 scale_w 分别作为高度和宽度的缩放比例
    return _upsample_nearest2d_quantized_cpu<nearest_neighbor_exact_compute_source_index>(input, osize, scale_h, scale_w);
}

} // namespace native
} // namespace at
```