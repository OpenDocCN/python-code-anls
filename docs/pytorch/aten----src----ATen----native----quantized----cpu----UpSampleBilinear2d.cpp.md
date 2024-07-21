# `.\pytorch\aten\src\ATen\native\quantized\cpu\UpSampleBilinear2d.cpp`

```py
// 定义 TORCH_ASSERT_ONLY_METHOD_OPERATORS 宏，用于指定仅使用方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 包含 ATen 库的 Tensor 类和相关头文件
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/native/UpSample.h>
#include <ATen/native/quantized/AffineQuantizer.h>
#include <ATen/native/quantized/cpu/QuantizedOps.h>
#include <ATen/native/cpu/utils.h>
#include <c10/util/irange.h>

// 如果未定义 AT_PER_OPERATOR_HEADERS 宏，则包含 ATen 的整体功能和原生函数
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
// 否则，包含特定的头文件，如 _empty_affine_quantized.h 和 upsample_bilinear2d_native.h
#else
#include <ATen/ops/_empty_affine_quantized.h>
#include <ATen/ops/upsample_bilinear2d_native.h>
#endif

// 包含 C 语言标准库头文件
#include <cstring>

// ATen 命名空间
namespace at {
// ATen 内部函数的命名空间
namespace native {
// 匿名命名空间，用于封装内部实现细节

// 预先计算在宽度上的双线性插值参数
struct UpsampleBilinearParamW {
  int64_t w1, w1p;          // w1: 下标，w1p: 是否超过边界
  float w0lambda, w1lambda; // w0lambda: w1 的权重，w1lambda: w1p 的权重

  // 构造函数，初始化插值参数
  UpsampleBilinearParamW(int64_t w1, int64_t w1p, float w0lambda, float w1lambda)
    : w1(w1)
    , w1p(w1p)
    , w0lambda(w0lambda)
    , w1lambda(w1lambda) {}
};

// 用于 native_functions.yaml 的 at::native 函数模板
template <typename scalar_t>
// 双线性插值的输出帧
static void upsample_bilinear2d_out_frame(
    Tensor& output,                     // 输出张量
    const Tensor& input,                // 输入张量
    int64_t input_height,               // 输入高度
    int64_t input_width,                // 输入宽度
    int64_t output_height,              // 输出高度
    int64_t output_width,               // 输出宽度
    int64_t nbatch,                    // 批次大小
    int64_t channels,                  // 通道数
    bool align_corners,                // 是否对齐角点
    std::optional<double> scales_h,    // 高度缩放比例（可选）
    std::optional<double> scales_w) {  // 宽度缩放比例（可选）

  auto* idata = static_cast<const scalar_t*>(input.const_data_ptr());  // 输入数据指针
  auto* odata = static_cast<scalar_t*>(output.data_ptr());             // 输出数据指针

  channels = channels * nbatch;  // 计算批次的通道数
  if (channels == 0 || output_height == 0 || output_width == 0) {
    return;  // 如果通道数或输出尺寸为零，直接返回
  }
  auto* i_p = reinterpret_cast<const typename scalar_t::underlying*>(idata);  // 输入数据底层类型指针
  auto* o_p = reinterpret_cast<typename scalar_t::underlying*>(odata);        // 输出数据底层类型指针

  // 特殊情况：直接复制数据
  if (input_height == output_height && input_width == output_width) {
    std::memcpy(
        o_p,
        i_p,
        channels * input_height * input_width *
            sizeof(typename scalar_t::underlying));  // 使用 memcpy 复制数据
    return;
  }

  // 计算高度和宽度上的像素区域缩放比例
  const auto rheight = area_pixel_compute_scale<float>(
      input_height, output_height, align_corners, scales_h);

  const auto rwidth = area_pixel_compute_scale<float>(
      input_width, output_width, align_corners, scales_w);

  // 计算输出的比例因子
  float output_scale = output.q_scale() / input.q_scale();

  // 输入和输出的量化零点
  const int64_t input_q_zero_point = input.q_zero_point();
  const int64_t output_q_zero_point = output.q_zero_point();

  // 存储宽度上的插值参数
  std::vector<UpsampleBilinearParamW> params_w;
  params_w.reserve(output_width);  // 预留输出宽度的空间
  for (const auto w2 : c10::irange(output_width)) {
    // 计算宽度上的源像素索引
    const auto w1r = area_pixel_compute_source_index<float>(
        rwidth, w2, align_corners, /*cubic=*/false);

    const int64_t w1 = w1r;  // 取整的源像素索引
    const int64_t w1p = (w1 < input_width - 1) ? 1 : 0;  // 判断是否超过边界

    const float w1lambda = w1r - w1;                      // 计算 w1 的权重
    const float w0lambda = static_cast<float>(1.) - w1lambda;  // 计算 w0 的权重
    // 将参数 w1, w1p, w0lambda, w1lambda 加入 params_w 的尾部
    params_w.emplace_back(w1, w1p, w0lambda, w1lambda);
  }

  // 相比于 'nearest' 方法，每个点需要 4 个点，并且需要额外的乘法和加法
  // 将缩放比例设置为 16
  int64_t grain_size = internal::GRAIN_SIZE / std::max(int64_t{1}, output_width) / 16;
  // 并行处理循环，范围是 0 到 channels * output_height，以 grain_size 为步长
  at::parallel_for(0, channels * output_height, grain_size, [&](int64_t begin, int64_t end) {
    // 初始化 nc 和 h2
    int64_t nc{0}, h2{0};
    data_index_init(begin, nc, channels, h2, output_height);

    // 对于范围内的每个 i
    for (const auto i : c10::irange(begin, end)) {
      // 计算 h1r，这里使用 area_pixel_compute_source_index 函数计算源索引
      const auto h1r = area_pixel_compute_source_index<float>(
          rheight, h2, align_corners, /*cubic=*/false);

      // 将 h1 转换为整数
      const int64_t h1 = h1r;
      // 如果 h1 小于 input_height - 1，则 h1p 为 1，否则为 0
      const int64_t h1p = (h1 < input_height - 1) ? 1 : 0;

      // 计算 h1lambda 和 h0lambda
      const float h1lambda = h1r - h1;
      const float h0lambda = static_cast<float>(1.) - h1lambda;

      // 指向输入数据和输出数据的指针
      const auto* i_ptr = &i_p[nc * input_height * input_width];
      auto* pos2 = &o_p[i * output_width];

      // 对于输出宽度内的每个 w2
      for (const auto w2 : c10::irange(output_width)) {
        // 获取当前 w2 对应的 param_w 参数
        const auto& param_w = params_w[w2];
        const int64_t w1 = param_w.w1;
        const int64_t w1p = param_w.w1p;
        const float w0lambda = param_w.w0lambda;
        const float w1lambda = param_w.w1lambda;

        // 指向输入数据的当前位置 pos1
        const auto* pos1 = i_ptr + h1 * input_width + w1;

        // 计算结果，包括线性插值和量化校正
        float result = h0lambda * (w0lambda * pos1[0] + w1lambda * pos1[w1p]) +
            h1lambda *
                (w0lambda * pos1[h1p * input_width] +
                 w1lambda * pos1[h1p * input_width + w1p]) - input_q_zero_point;

        // 重新量化结果并存储到输出位置 pos2[w2]
        pos2[w2] = at::native::quantize_val<scalar_t>(
                      output_scale, output_q_zero_point, result)
                      .val_;
      }

      // 更新数据索引 nc 和 h2
      data_index_step(nc, channels, h2, output_height);
    }
  });
} // namespace native
} // namespace at
```