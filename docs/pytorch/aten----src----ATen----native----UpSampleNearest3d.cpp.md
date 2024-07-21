# `.\pytorch\aten\src\ATen\native\UpSampleNearest3d.cpp`

```
// 定义宏，仅使用方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 引入张量操作的头文件
#include <ATen/core/Tensor.h>
#include <ATen/TensorMeta.h>
#include <ATen/native/UpSample.h>
#include <c10/util/irange.h>

// 根据是否定义了 AT_PER_OPERATOR_HEADERS 来选择性引入不同的头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_upsample_nearest_exact3d.h>
#include <ATen/ops/_upsample_nearest_exact3d_backward.h>
#include <ATen/ops/_upsample_nearest_exact3d_backward_native.h>
#include <ATen/ops/_upsample_nearest_exact3d_native.h>
#include <ATen/ops/upsample_nearest3d.h>
#include <ATen/ops/upsample_nearest3d_backward.h>
#include <ATen/ops/upsample_nearest3d_backward_native.h>
#include <ATen/ops/upsample_nearest3d_native.h>
#endif

// 命名空间定义开始
namespace at::meta {

// 定义 TORCH_META_FUNC 宏处理的函数 upsample_nearest3d
TORCH_META_FUNC(upsample_nearest3d) (
    const Tensor& input,  // 输入张量
    IntArrayRef output_size,  // 输出大小的数组引用
    std::optional<double> scales_d,  // 可选的深度缩放因子
    std::optional<double> scales_h,  // 可选的高度缩放因子
    std::optional<double> scales_w  // 可选的宽度缩放因子
) {
  // 调用 native 命名空间的 upsample_3d_common_check 函数，获取完整的输出大小
  auto full_output_size = native::upsample_3d_common_check(input.sizes(), output_size);

  // 检查输入张量是否为空，或者批次大小以外的维度是否为空
  TORCH_CHECK(
      input.numel() != 0 || c10::multiply_integers(input.sizes().begin() + 1, input.sizes().end()),
      "Non-empty 5D data tensor expected but got a tensor with sizes ",
      input.sizes());

  // 设置输出张量的原始步长信息和内存格式
  set_output_raw_strided(0, full_output_size, {}, input.options().memory_format(input.suggest_memory_format()));
}

// 定义 TORCH_META_FUNC 宏处理的函数 _upsample_nearest_exact3d
TORCH_META_FUNC(_upsample_nearest_exact3d) (
  const Tensor& input,  // 输入张量
  IntArrayRef output_size,  // 输出大小的数组引用
  std::optional<double> scales_d,  // 可选的深度缩放因子
  std::optional<double> scales_h,  // 可选的高度缩放因子
  std::optional<double> scales_w  // 可选的宽度缩放因子
) {
  // 调用 native 命名空间的 upsample_3d_common_check 函数，获取完整的输出大小
  auto full_output_size = native::upsample_3d_common_check(input.sizes(), output_size);

  // 检查输入张量是否为空，或者批次大小以外的维度是否为空
  TORCH_CHECK(
      input.numel() != 0 || c10::multiply_integers(input.sizes().begin() + 1, input.sizes().end()),
      "Non-empty 5D data tensor expected but got a tensor with sizes ",
      input.sizes());

  // 设置输出张量的原始步长信息和内存格式
  set_output_raw_strided(0, full_output_size, {}, input.options().memory_format(input.suggest_memory_format()));
}

// 定义 TORCH_META_FUNC 宏处理的函数 upsample_nearest3d_backward
TORCH_META_FUNC(upsample_nearest3d_backward) (
    const Tensor& grad_output,  // 梯度输出张量
    IntArrayRef output_size,  // 输出大小的数组引用
    IntArrayRef input_size,  // 输入大小的数组引用
    std::optional<double> scales_d,  // 可选的深度缩放因子
    std::optional<double> scales_h,  // 可选的高度缩放因子
    std::optional<double> scales_w  // 可选的宽度缩放因子
) {
  // 调用 native 命名空间的 upsample_3d_common_check 函数，获取完整的输出大小
  auto full_output_size = native::upsample_3d_common_check(input_size, output_size);

  // 检查梯度输出张量是否为5维
  TORCH_CHECK(
      grad_output.dim() == 5,
      "Expected grad_output to be a tensor of dimension 5 but got: dimension ", grad_output.dim());

  // 检查梯度输出张量的每个维度是否与预期的输出大小相同
  for (const auto i : c10::irange(5)) {
    TORCH_CHECK(
        grad_output.size(i) == full_output_size[i],
        "Expected grad_output to have the same shape as output;",
        " output.size(", i, ") = ", full_output_size[i],
        " but got grad_output.size(", i, ") = ", grad_output.size(i));
  }

  // 设置输出张量的原始步长信息
  set_output_raw_strided(0, input_size, {}, grad_output.options());
}

// 命名空间定义结束
}
TORCH_META_FUNC(_upsample_nearest_exact3d_backward) (
  const Tensor& grad_output,                              // 输入参数：梯度输出张量
  IntArrayRef output_size,                                // 输入参数：输出尺寸的整数数组引用
  IntArrayRef input_size,                                 // 输入参数：输入尺寸的整数数组引用
  std::optional<double> scales_d,                         // 可选参数：沿深度尺度因子
  std::optional<double> scales_h,                         // 可选参数：沿高度尺度因子
  std::optional<double> scales_w                          // 可选参数：沿宽度尺度因子
) {
  auto full_output_size = native::upsample_3d_common_check(input_size, output_size);  // 计算完整的输出尺寸

  TORCH_CHECK(
      grad_output.dim() == 5,                             // 检查梯度输出张量维度是否为5
      "Expected grad_output to be a tensor of dimension 5 but got: dimension ", grad_output.dim());

  for (const auto i : c10::irange(5)) {                   // 遍历维度索引范围
    TORCH_CHECK(
        grad_output.size(i) == full_output_size[i],       // 检查梯度输出张量每个维度是否与完整输出尺寸相匹配
        "Expected grad_output to have the same shape as output;",
        " output.size(", i, ") = ", full_output_size[i],
        " but got grad_output.size(", i, ") = ", grad_output.size(i));
  }

  set_output_raw_strided(0, input_size, {}, grad_output.options());  // 设置输出的原始步幅
}

} // namespace at::meta

namespace at::native {

TORCH_IMPL_FUNC(upsample_nearest3d_out_cpu) (
    const Tensor& input,                                  // 输入参数：输入张量
    IntArrayRef output_size,                              // 输入参数：输出尺寸的整数数组引用
    std::optional<double> scales_d,                       // 可选参数：沿深度尺度因子
    std::optional<double> scales_h,                       // 可选参数：沿高度尺度因子
    std::optional<double> scales_w,                       // 可选参数：沿宽度尺度因子
    const Tensor& output                                  // 输入参数：输出张量
) {
  upsample_nearest3d_kernel(kCPU, output, input, scales_d, scales_h, scales_w);  // 调用最近邻三维上采样的 CPU 内核函数
}

TORCH_IMPL_FUNC(_upsample_nearest_exact3d_out_cpu) (
    const Tensor& input,                                  // 输入参数：输入张量
    IntArrayRef output_size,                              // 输入参数：输出尺寸的整数数组引用
    std::optional<double> scales_d,                       // 可选参数：沿深度尺度因子
    std::optional<double> scales_h,                       // 可选参数：沿高度尺度因子
    std::optional<double> scales_w,                       // 可选参数：沿宽度尺度因子
    const Tensor& output                                  // 输入参数：输出张量
) {
  _upsample_nearest_exact3d_kernel(kCPU, output, input, scales_d, scales_h, scales_w);  // 调用精确最近邻三维上采样的 CPU 内核函数
}

TORCH_IMPL_FUNC(upsample_nearest3d_backward_out_cpu) (
    const Tensor& grad_output,                            // 输入参数：梯度输出张量
    IntArrayRef output_size,                              // 输入参数：输出尺寸的整数数组引用
    IntArrayRef input_size,                               // 输入参数：输入尺寸的整数数组引用
    std::optional<double> scales_d,                       // 可选参数：沿深度尺度因子
    std::optional<double> scales_h,                       // 可选参数：沿高度尺度因子
    std::optional<double> scales_w,                       // 可选参数：沿宽度尺度因子
    const Tensor& grad_input                              // 输入参数：梯度输入张量
) {
  grad_input.zero_();                                     // 将梯度输入张量置零
  upsample_nearest3d_backward_kernel(kCPU, grad_input, grad_output, scales_d, scales_h, scales_w);  // 调用最近邻三维上采样的反向传播 CPU 内核函数
}

TORCH_IMPL_FUNC(_upsample_nearest_exact3d_backward_out_cpu) (
    const Tensor& grad_output,                            // 输入参数：梯度输出张量
    IntArrayRef output_size,                              // 输入参数：输出尺寸的整数数组引用
    IntArrayRef input_size,                               // 输入参数：输入尺寸的整数数组引用
    std::optional<double> scales_d,                       // 可选参数：沿深度尺度因子
    std::optional<double> scales_h,                       // 可选参数：沿高度尺度因子
    std::optional<double> scales_w,                       // 可选参数：沿宽度尺度因子
    const Tensor& grad_input                              // 输入参数：梯度输入张量
) {
  grad_input.zero_();                                     // 将梯度输入张量置零
  _upsample_nearest_exact3d_backward_kernel(kCPU, grad_input, grad_output, scales_d, scales_h, scales_w);  // 调用精确最近邻三维上采样的反向传播 CPU 内核函数
}

// vec variants

using at::native::upsample::compute_output_size;          // 使用上采样命名空间中的计算输出尺寸函数
using at::native::upsample::get_scale_value;              // 使用上采样命名空间中的获取尺度值函数

Tensor upsample_nearest3d(
    const Tensor& input,                                  // 输入参数：输入张量
    at::OptionalIntArrayRef output_size,                  // 可选参数：输出尺寸的整数数组引用
    std::optional<ArrayRef<double>> scale_factors        // 可选参数：尺度因子数组引用
) {
  auto osize = compute_output_size(input.sizes(), output_size, scale_factors);  // 计算输出尺寸
  auto scale_d = get_scale_value(scale_factors, 0);       // 获取深度尺度因子值
  auto scale_h = get_scale_value(scale_factors, 1);       // 获取高度尺度因子值
  auto scale_w = get_scale_value(scale_factors, 2);       // 获取宽度尺度因子值
  return at::upsample_nearest3d(input, osize, scale_d, scale_h, scale_w);  // 执行最近邻三维上采样
}

Tensor _upsample_nearest_exact3d(
    # 使用引用传递的张量作为输入
    const Tensor& input,
    # 计算最终输出的尺寸，根据给定的输出尺寸和缩放因子
    at::OptionalIntArrayRef output_size,
    # 可选参数，缩放因子数组，用于指定每个维度的缩放比例
    std::optional<ArrayRef<double>> scale_factors) {
  # 计算输出大小
  auto osize = compute_output_size(input.sizes(), output_size, scale_factors);
  # 获取缩放因子的值，分别对应深度、高度和宽度
  auto scale_d = get_scale_value(scale_factors, 0);
  auto scale_h = get_scale_value(scale_factors, 1);
  auto scale_w = get_scale_value(scale_factors, 2);
  # 使用最近邻插值方式对输入进行三维精确上采样
  return at::_upsample_nearest_exact3d(input, osize, scale_d, scale_h, scale_w);
} // namespace at::native
```