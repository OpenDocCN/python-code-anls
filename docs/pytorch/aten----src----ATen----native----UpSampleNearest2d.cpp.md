# `.\pytorch\aten\src\ATen\native\UpSampleNearest2d.cpp`

```
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/TensorMeta.h>
#include <ATen/native/UpSample.h>
#include <c10/util/accumulate.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_upsample_nearest_exact2d.h>
#include <ATen/ops/_upsample_nearest_exact2d_backward.h>
#include <ATen/ops/_upsample_nearest_exact2d_backward_native.h>
#include <ATen/ops/_upsample_nearest_exact2d_native.h>
#include <ATen/ops/upsample_nearest2d.h>
#include <ATen/ops/upsample_nearest2d_backward.h>
#include <ATen/ops/upsample_nearest2d_backward_native.h>
#include <ATen/ops/upsample_nearest2d_native.h>
#endif

namespace at::meta {

// 定义了一个名为 upsample_nearest2d 的元函数，处理最近邻插值在二维上采样的情况
TORCH_META_FUNC(upsample_nearest2d) (
    const Tensor& input, IntArrayRef output_size, std::optional<double> scales_h, std::optional<double> scales_w
) {
  // 计算最终输出的大小，确保是有效的输出大小
  auto full_output_size = native::upsample_2d_common_check(input.sizes(), output_size);

  // 检查输入是否为非空的四维数据张量
  // 如果 batch 大小为空，可以接受，但不接受其他维度为空
  TORCH_CHECK(
      input.numel() != 0 || c10::multiply_integers(input.sizes().begin() + 1, input.sizes().end()),
      "Non-empty 4D data tensor expected but got a tensor with sizes ",
      input.sizes());

  // 设置输出的张量格式和内存布局，使用输入张量的推荐内存格式
  set_output_raw_strided(0, full_output_size, {}, input.options().memory_format(input.suggest_memory_format()));
}

// 定义了一个名为 _upsample_nearest_exact2d 的元函数，处理精确最近邻插值在二维上采样的情况
TORCH_META_FUNC(_upsample_nearest_exact2d) (
  const Tensor& input, IntArrayRef output_size, std::optional<double> scales_h, std::optional<double> scales_w
) {
  // 计算最终输出的大小，确保是有效的输出大小
  auto full_output_size = native::upsample_2d_common_check(input.sizes(), output_size);

  // 检查输入是否为非空的四维数据张量
  // 如果 batch 大小为空，可以接受，但不接受其他维度为空
  TORCH_CHECK(
      input.numel() != 0 || c10::multiply_integers(input.sizes().begin() + 1, input.sizes().end()),
      "Non-empty 4D data tensor expected but got a tensor with sizes ",
      input.sizes());

  // 设置输出的张量格式和内存布局，使用输入张量的推荐内存格式
  set_output_raw_strided(0, full_output_size, {}, input.options().memory_format(input.suggest_memory_format()));
}

// 定义了一个名为 upsample_nearest2d_backward 的元函数，处理最近邻插值在二维上采样的反向传播情况
TORCH_META_FUNC(upsample_nearest2d_backward) (
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    std::optional<double> scales_h,
    std::optional<double> scales_w
) {
  // 计算最终输出的大小，确保是有效的输出大小
  auto full_output_size = native::upsample_2d_common_check(input_size, output_size);

  // 检查梯度输出张量的维度是否为四维
  TORCH_CHECK(
      grad_output.dim() == 4,
      "Expected grad_output to be a tensor of dimension 4 but got: dimension ", grad_output.dim());

  // 检查每个维度上的输出大小是否与预期的完整输出大小匹配
  for (const auto i : c10::irange(4)) {
    TORCH_CHECK(
        grad_output.size(i) == full_output_size[i],
        "Expected grad_output to have the same shape as output;",
        " output.size(", i, ") = ", full_output_size[i],
        " but got grad_output.size(", i, ") = ", grad_output.size(i));
  }

  // 设置输出的张量格式和内存布局，使用梯度输出张量的推荐内存格式
  set_output_raw_strided(0, input_size, {}, grad_output.options().memory_format(grad_output.suggest_memory_format()));
}

} // namespace at::meta
TORCH_META_FUNC(_upsample_nearest_exact2d_backward) (
  const Tensor& grad_output,                                 // 输入参数：梯度输出张量
  IntArrayRef output_size,                                   // 输入参数：输出尺寸数组引用
  IntArrayRef input_size,                                    // 输入参数：输入尺寸数组引用
  std::optional<double> scales_h,                            // 输入参数：高度缩放比例（可选）
  std::optional<double> scales_w                             // 输入参数：宽度缩放比例（可选）
) {
  auto full_output_size = native::upsample_2d_common_check(input_size, output_size);  // 计算完整的输出尺寸

  TORCH_CHECK(
      grad_output.dim() == 4,                                // 检查：梯度输出张量维度必须为4
      "Expected grad_output to be a tensor of dimension 4 but got: dimension ", grad_output.dim());

  for (const auto i : c10::irange(4)) {
    TORCH_CHECK(
        grad_output.size(i) == full_output_size[i],          // 检查：梯度输出张量在每个维度上的尺寸与完整输出尺寸匹配
        "Expected grad_output to have the same shape as output;",
        " output.size(", i, ") = ", full_output_size[i],
        " but got grad_output.size(", i, ") = ", grad_output.size(i));
  }

  set_output_raw_strided(0, input_size, {}, grad_output.options().memory_format(grad_output.suggest_memory_format()));  // 设置输出的原始步幅
}

} // namespace at::meta

namespace at::native {

TORCH_IMPL_FUNC(upsample_nearest2d_out_cpu) (
    const Tensor& input,                                     // 输入参数：输入张量
    IntArrayRef output_size,                                 // 输入参数：输出尺寸数组引用
    std::optional<double> scales_h,                          // 输入参数：高度缩放比例（可选）
    std::optional<double> scales_w,                          // 输入参数：宽度缩放比例（可选）
    const Tensor& output                                     // 输入参数：输出张量
) {
  upsample_nearest2d_kernel(kCPU, output, input, scales_h, scales_w);  // 调用最近邻2D上采样的核心函数
}

TORCH_IMPL_FUNC(_upsample_nearest_exact2d_out_cpu) (
    const Tensor& input,                                     // 输入参数：输入张量
    IntArrayRef output_size,                                 // 输入参数：输出尺寸数组引用
    std::optional<double> scales_h,                          // 输入参数：高度缩放比例（可选）
    std::optional<double> scales_w,                          // 输入参数：宽度缩放比例（可选）
    const Tensor& output                                     // 输入参数：输出张量
) {
  _upsample_nearest_exact2d_kernel(kCPU, output, input, scales_h, scales_w);  // 调用精确最近邻2D上采样的核心函数
}

TORCH_IMPL_FUNC(upsample_nearest2d_backward_out_cpu) (
    const Tensor& grad_output,                               // 输入参数：梯度输出张量
    IntArrayRef output_size,                                 // 输入参数：输出尺寸数组引用
    IntArrayRef input_size,                                  // 输入参数：输入尺寸数组引用
    std::optional<double> scales_h,                          // 输入参数：高度缩放比例（可选）
    std::optional<double> scales_w,                          // 输入参数：宽度缩放比例（可选）
    const Tensor& grad_input                                 // 输入参数：梯度输入张量
) {
  grad_input.zero_();                                        // 将梯度输入张量清零
  upsample_nearest2d_backward_kernel(kCPU, grad_input, grad_output, scales_h, scales_w);  // 调用最近邻2D上采样反向传播的核心函数
}

TORCH_IMPL_FUNC(_upsample_nearest_exact2d_backward_out_cpu) (
    const Tensor& grad_output,                               // 输入参数：梯度输出张量
    IntArrayRef output_size,                                 // 输入参数：输出尺寸数组引用
    IntArrayRef input_size,                                  // 输入参数：输入尺寸数组引用
    std::optional<double> scales_h,                          // 输入参数：高度缩放比例（可选）
    std::optional<double> scales_w,                          // 输入参数：宽度缩放比例（可选）
    const Tensor& grad_input                                 // 输入参数：梯度输入张量
) {
  grad_input.zero_();                                        // 将梯度输入张量清零
  _upsample_nearest_exact2d_backward_kernel(kCPU, grad_input, grad_output, scales_h, scales_w);  // 调用精确最近邻2D上采样反向传播的核心函数
}

using at::native::upsample::compute_output_size;             // 引用上采样命名空间中的计算输出尺寸函数
using at::native::upsample::get_scale_value;                 // 引用上采样命名空间中的获取缩放值函数

Tensor upsample_nearest2d(
    const Tensor& input,                                     // 输入参数：输入张量
    at::OptionalIntArrayRef output_size,                     // 输入参数：输出尺寸数组引用（可选）
    std::optional<ArrayRef<double>> scale_factors            // 输入参数：缩放因子数组引用（可选）
) {
  auto osize = compute_output_size(input.sizes(), output_size, scale_factors);  // 计算输出尺寸
  auto scale_h = get_scale_value(scale_factors, 0);           // 获取高度缩放值
  auto scale_w = get_scale_value(scale_factors, 1);           // 获取宽度缩放值
  return at::upsample_nearest2d(input, osize, scale_h, scale_w);  // 调用最近邻2D上采样函数
}

Tensor _upsample_nearest_exact2d(
    const Tensor& input,                                     // 输入参数：输入张量
    at::OptionalIntArrayRef output_size,                     // 输入参数：输出尺寸数组引用（可选）
    std::optional<ArrayRef<double>> scale_factors            // 输入参数：缩放因子数组引用（可选）
) {
    std::optional<ArrayRef<double>> scale_factors) {

# 定义函数 `upsample_nearest_exact2d`，接受输入张量、输出大小和可选的尺度因子作为参数
  auto osize = compute_output_size(input.sizes(), output_size, scale_factors);

# 计算输出大小，调用 `compute_output_size` 函数，传入输入张量大小、输出大小和尺度因子
  auto scale_h = get_scale_value(scale_factors, 0);

# 获取高度方向的尺度因子，调用 `get_scale_value` 函数，传入尺度因子和索引0
  auto scale_w = get_scale_value(scale_factors, 1);

# 获取宽度方向的尺度因子，调用 `get_scale_value` 函数，传入尺度因子和索引1
  return at::_upsample_nearest_exact2d(input, osize, scale_h, scale_w);

# 调用 PyTorch 中的 `_upsample_nearest_exact2d` 函数，传入输入张量、输出大小、高度尺度因子和宽度尺度因子，并返回结果
}

DEFINE_DISPATCH(upsample_nearest2d_kernel);
// 定义名为 upsample_nearest2d_kernel 的调度器分发器

DEFINE_DISPATCH(_upsample_nearest_exact2d_kernel);
// 定义名为 _upsample_nearest_exact2d_kernel 的调度器分发器

DEFINE_DISPATCH(upsample_nearest2d_backward_kernel);
// 定义名为 upsample_nearest2d_backward_kernel 的调度器分发器

DEFINE_DISPATCH(_upsample_nearest_exact2d_backward_kernel);
// 定义名为 _upsample_nearest_exact2d_backward_kernel 的调度器分发器

} // namespace at::native
// 结束 at::native 命名空间
```