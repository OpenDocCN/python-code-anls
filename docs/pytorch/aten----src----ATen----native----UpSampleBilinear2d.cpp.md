# `.\pytorch\aten\src\ATen\native\UpSampleBilinear2d.cpp`

```py
// Adapted from interp.cpp from Caffe util by Pauline Luc
// Originally developed by George Papandreou

// 定义宏，仅使用方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 包含 ATen 库的头文件
#include <ATen/core/Tensor.h>
#include <ATen/TensorMeta.h>
#include <ATen/native/UpSample.h>
#include <c10/util/irange.h>

// 根据不同的预处理宏选择性包含不同的头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_upsample_bilinear2d_aa.h>
#include <ATen/ops/_upsample_bilinear2d_aa_backward.h>
#include <ATen/ops/_upsample_bilinear2d_aa_backward_native.h>
#include <ATen/ops/_upsample_bilinear2d_aa_native.h>
#include <ATen/ops/upsample_bilinear2d.h>
#include <ATen/ops/upsample_bilinear2d_backward.h>
#include <ATen/ops/upsample_bilinear2d_backward_native.h>
#include <ATen/ops/upsample_bilinear2d_native.h>
#endif

// 命名空间 at::meta
namespace at::meta {

// 定义 TORCH_META_FUNC 宏，实现双线性插值的上采样
TORCH_META_FUNC(upsample_bilinear2d) (
  const Tensor& input, IntArrayRef output_size, bool align_corners, std::optional<double> scales_h, std::optional<double> scales_w
) {
  // 调用 native::upsample_2d_common_check 函数获取完整的输出尺寸
  auto full_output_size = native::upsample_2d_common_check(input.sizes(), output_size);

  // 检查输入张量的维度和大小，至少需要一个非空的 4D 数据张量
  TORCH_CHECK(
      input.numel() != 0 || c10::multiply_integers(input.sizes().begin() + 1, input.sizes().end()),
      "Non-empty 4D data tensor expected but got a tensor with sizes ",
      input.sizes());

  // 设置输出张量的格式和内存布局
  set_output_raw_strided(0, full_output_size, {}, input.options().memory_format(input.suggest_memory_format()));
}

// 定义 TORCH_META_FUNC 宏，实现双线性插值上采样的反向传播
TORCH_META_FUNC(upsample_bilinear2d_backward) (
  const Tensor& grad_output,
  IntArrayRef output_size,
  IntArrayRef input_size,
  bool align_corners,
  std::optional<double> scales_h,
  std::optional<double> scales_w
) {
  // 调用 native::upsample_2d_common_check 函数获取完整的输出尺寸
  auto full_output_size = native::upsample_2d_common_check(input_size, output_size);

  // 检查梯度输出张量的维度，必须为 4
  TORCH_CHECK(
      grad_output.dim() == 4,
      "Expected grad_output to be a tensor of dimension 4 but got: dimension ", grad_output.dim());

  // 检查梯度输出张量的每个维度与期望的输出尺寸是否一致
  for (const auto i : c10::irange(4)) {
    TORCH_CHECK(
        grad_output.size(i) == full_output_size[i],
        "Expected grad_output to have the same shape as output;",
        " output.size(", i, ") = ", full_output_size[i],
        " but got grad_output.size(", i, ") = ", grad_output.size(i));
  }

  // 设置输入张量的梯度输出张量的格式和内存布局
  set_output_raw_strided(0, input_size, {}, grad_output.options().memory_format(grad_output.suggest_memory_format()));
}

// 定义 TORCH_META_FUNC 宏，实现带反锯齿的双线性插值上采样
TORCH_META_FUNC(_upsample_bilinear2d_aa) (
  const Tensor& input, IntArrayRef output_size, bool align_corners, std::optional<double> scales_h, std::optional<double> scales_w
) {
  // 调用 native::upsample_2d_common_check 函数获取完整的输出尺寸
  auto full_output_size = native::upsample_2d_common_check(input.sizes(), output_size);

  // 检查输入张量的维度和大小，至少需要一个非空的 4D 数据张量
  TORCH_CHECK(
      input.numel() != 0 || c10::multiply_integers(input.sizes().begin() + 1, input.sizes().end()),
      "Non-empty 4D data tensor expected but got a tensor with sizes ",
      input.sizes());

  // 设置输出张量的格式和内存布局
  set_output_raw_strided(0, full_output_size, {}, input.options().memory_format(input.suggest_memory_format()));
}

} // namespace at::meta
// 定义 TORCH_META_FUNC 宏，处理 _upsample_bilinear2d_aa_backward 函数的元信息
TORCH_META_FUNC(_upsample_bilinear2d_aa_backward) (
  // 接收梯度输出、输出大小、输入大小、是否对齐角点、垂直缩放比例、水平缩放比例
  const Tensor& grad_output,
  IntArrayRef output_size,
  IntArrayRef input_size,
  bool align_corners,
  std::optional<double> scales_h,
  std::optional<double> scales_w
) {
  // 调用 native::upsample_2d_common_check 函数获取完整的输出大小
  auto full_output_size = native::upsample_2d_common_check(input_size, output_size);

  // 检查 grad_output 张量的维度是否为 4
  TORCH_CHECK(
      grad_output.dim() == 4,
      "Expected grad_output to be a tensor of dimension 4 but got: dimension ", grad_output.dim());

  // 循环检查 grad_output 的每个维度是否与完整输出大小相匹配
  for (const auto i : c10::irange(4)) {
    TORCH_CHECK(
        grad_output.size(i) == full_output_size[i],
        "Expected grad_output to have the same shape as output;",
        " output.size(", i, ") = ", full_output_size[i],
        " but got grad_output.size(", i, ") = ", grad_output.size(i));
  }

  // 使用 grad_output 的选项设置输出张量的原始步幅
  set_output_raw_strided(0, input_size, {}, grad_output.options().memory_format(grad_output.suggest_memory_format()));
}

} // namespace at::meta

// 开始定义 at::native 命名空间
namespace at::native {

// 实现 upsample_bilinear2d_out_cpu 函数
TORCH_IMPL_FUNC(upsample_bilinear2d_out_cpu) (
    // 输入张量、输出大小、是否对齐角点、垂直缩放比例、水平缩放比例、输出张量
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners,
    std::optional<double> scales_h,
    std::optional<double> scales_w,
    const Tensor& output
) {
  // 调用 upsample_bilinear2d_kernel 函数处理 CPU 上的双线性插值操作
  upsample_bilinear2d_kernel(kCPU, output, input, align_corners, scales_h, scales_w);
}

// 实现 upsample_bilinear2d_backward_out_cpu 函数
TORCH_IMPL_FUNC(upsample_bilinear2d_backward_out_cpu) (
    // 梯度输出张量、输出大小、输入大小、是否对齐角点、垂直缩放比例、水平缩放比例、梯度输入张量
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners,
    std::optional<double> scales_h,
    std::optional<double> scales_w,
    const Tensor& grad_input
) {
  // 将 grad_input 张量清零
  grad_input.zero_();
  // 调用 upsample_bilinear2d_backward_kernel 函数处理 CPU 上的双线性插值反向传播
  upsample_bilinear2d_backward_kernel(kCPU, grad_input, grad_output, align_corners, scales_h, scales_w);
}


// 实现 _upsample_bilinear2d_aa_out_cpu 函数
TORCH_IMPL_FUNC(_upsample_bilinear2d_aa_out_cpu) (
    // 输入张量、输出大小、是否对齐角点、垂直缩放比例、水平缩放比例、输出张量
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners,
    std::optional<double> scales_h,
    std::optional<double> scales_w,
    const Tensor& output
) {
  // 调用 _upsample_bilinear2d_aa_kernel 函数处理 CPU 上的双线性插值 AA 操作
  _upsample_bilinear2d_aa_kernel(kCPU, output, input, align_corners, scales_h, scales_w);
}

// 实现 _upsample_bilinear2d_aa_backward_out_cpu 函数
TORCH_IMPL_FUNC(_upsample_bilinear2d_aa_backward_out_cpu) (
    // 梯度输出张量、输出大小、输入大小、是否对齐角点、垂直缩放比例、水平缩放比例、梯度输入张量
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners,
    std::optional<double> scales_h,
    std::optional<double> scales_w,
    const Tensor& grad_input
) {
  // 将 grad_input 张量清零
  grad_input.zero_();
  // 调用 _upsample_bilinear2d_aa_backward_kernel 函数处理 CPU 上的双线性插值 AA 反向传播
  _upsample_bilinear2d_aa_backward_kernel(kCPU, grad_input, grad_output, align_corners, scales_h, scales_w);
}

// 使用 at::native::upsample 命名空间中的 compute_output_size 和 get_scale_value 函数
using at::native::upsample::compute_output_size;
using at::native::upsample::get_scale_value;

// 实现 upsample_bilinear2d 函数
Tensor upsample_bilinear2d(
    // 输入张量、可选的输出大小、是否对齐角点、可选的缩放因子
    const Tensor& input,
    at::OptionalIntArrayRef output_size,
    bool align_corners,
    std::optional<ArrayRef<double>> scale_factors) {
  // 计算输出大小
  auto osize = compute_output_size(input.sizes(), output_size, scale_factors);
  // 获取垂直和水平缩放因子的值
  auto scale_h = get_scale_value(scale_factors, 0);
  auto scale_w = get_scale_value(scale_factors, 1);
  // 调用 at::upsample_bilinear2d 函数进行双线性插值操作
  return at::upsample_bilinear2d(input, osize, align_corners, scale_h, scale_w);
}

// 实现 _upsample_bilinear2d_aa 函数
Tensor _upsample_bilinear2d_aa(
    // 输入张量、输出大小、是否对齐角点、垂直缩放比例、水平缩放比例
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners,
    std::optional<double> scales_h,
    std::optional<double> scales_w,
    const Tensor& output
) {
  // 调用 _upsample_bilinear2d_aa_kernel 函数处理双线性插值 AA 操作
  _upsample_bilinear2d_aa_kernel(kCPU, output, input, align_corners, scales_h, scales_w);
}
    # 根据给定的输出尺寸参数和缩放因子计算最终的输出尺寸
    auto osize = compute_output_size(input.sizes(), output_size, scale_factors);
    # 获取垂直方向（高度）的缩放因子
    auto scale_h = get_scale_value(scale_factors, 0);
    # 获取水平方向（宽度）的缩放因子
    auto scale_w = get_scale_value(scale_factors, 1);
    # 调用带有双线性插值的2D上采样函数，进行图像的放大
    return at::_upsample_bilinear2d_aa(input, osize, align_corners, scale_h, scale_w);
}

DEFINE_DISPATCH(upsample_bilinear2d_kernel);
// 定义了一个名为 `upsample_bilinear2d_kernel` 的分发函数

DEFINE_DISPATCH(upsample_bilinear2d_backward_kernel);
// 定义了一个名为 `upsample_bilinear2d_backward_kernel` 的分发函数

DEFINE_DISPATCH(_upsample_bilinear2d_aa_kernel);
// 定义了一个名为 `_upsample_bilinear2d_aa_kernel` 的分发函数

DEFINE_DISPATCH(_upsample_bilinear2d_aa_backward_kernel);
// 定义了一个名为 `_upsample_bilinear2d_aa_backward_kernel` 的分发函数

} // namespace at::native
// 结束了 `at::native` 命名空间的定义
```