# `.\pytorch\aten\src\ATen\native\UpSampleNearest1d.cpp`

```
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_upsample_nearest_exact1d.h>
#include <ATen/ops/_upsample_nearest_exact1d_backward.h>
#include <ATen/ops/_upsample_nearest_exact1d_backward_native.h>
#include <ATen/ops/_upsample_nearest_exact1d_native.h>
#include <ATen/ops/upsample_nearest1d.h>
#include <ATen/ops/upsample_nearest1d_backward.h>
#include <ATen/ops/upsample_nearest1d_backward_native.h>
#include <ATen/ops/upsample_nearest1d_native.h>
#endif



// 如果未定义 AT_PER_OPERATOR_HEADERS 宏，则包含通用的 ATen 函数头文件
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
// 否则，包含特定的上采样最近邻方法的头文件

namespace at::meta {

// 定义上采样最近邻一维操作的元信息函数
TORCH_META_FUNC(upsample_nearest1d) (
    const Tensor& input, IntArrayRef output_size, std::optional<double> scales
) {
  // 调用通用的一维上采样检查函数，获取完整的输出大小
  auto full_output_size = native::upsample_1d_common_check(input.sizes(), output_size);

  // 检查输入张量的维度和大小是否符合要求
  TORCH_CHECK(
      (input.size(1) != 0 && input.size(2) != 0) && input.dim() == 3,
      "Non-empty 3D data tensor expected but got a tensor with sizes ",
      input.sizes());

  // 设置输出张量的原始步长方式存储
  set_output_raw_strided(0, full_output_size, {}, input.options());
}

// 定义上采样最近邻精确一维操作的元信息函数
TORCH_META_FUNC(_upsample_nearest_exact1d) (
  const Tensor& input, IntArrayRef output_size, std::optional<double> scales
) {
  // 调用通用的一维上采样检查函数，获取完整的输出大小
  auto full_output_size = native::upsample_1d_common_check(input.sizes(), output_size);

  // 检查输入张量的维度和大小是否符合要求
  TORCH_CHECK(
      (input.size(1) != 0 && input.size(2) != 0) && input.dim() == 3,
      "Non-empty 3D data tensor expected but got a tensor with sizes ",
      input.sizes());

  // 设置输出张量的原始步长方式存储
  set_output_raw_strided(0, full_output_size, {}, input.options());
}

// 定义上采样最近邻一维反向操作的元信息函数
TORCH_META_FUNC(upsample_nearest1d_backward) (
    const Tensor& grad_output, IntArrayRef output_size, IntArrayRef input_size, std::optional<double> scales
) {
  // 调用通用的一维上采样检查函数，获取完整的输出大小
  auto full_output_size = native::upsample_1d_common_check(input_size, output_size);

  // 检查梯度输出张量的维度和大小是否符合要求
  check_dim_size(grad_output, 3, 0, full_output_size[0]);
  check_dim_size(grad_output, 3, 1, full_output_size[1]);
  check_dim_size(grad_output, 3, 2, full_output_size[2]);

  // 设置输出张量的原始步长方式存储
  set_output_raw_strided(0, input_size, {}, grad_output.options());
}

// 定义上采样最近邻精确一维反向操作的元信息函数
TORCH_META_FUNC(_upsample_nearest_exact1d_backward) (
  const Tensor& grad_output, IntArrayRef output_size, IntArrayRef input_size, std::optional<double> scales
) {
  // 调用通用的一维上采样检查函数，获取完整的输出大小
  auto full_output_size = native::upsample_1d_common_check(input_size, output_size);

  // 检查梯度输出张量的维度和大小是否符合要求
  check_dim_size(grad_output, 3, 0, full_output_size[0]);
  check_dim_size(grad_output, 3, 1, full_output_size[1]);
  check_dim_size(grad_output, 3, 2, full_output_size[2]);

  // 设置输出张量的原始步长方式存储
  set_output_raw_strided(0, input_size, {}, grad_output.options());
}

} // namespace at::meta

namespace at::native {

// 实现上采样最近邻一维操作的函数
TORCH_IMPL_FUNC(upsample_nearest1d_out_cpu) (
    const Tensor& input,
    IntArrayRef output_size,
    std::optional<double> scales,
    const Tensor& output
) {
    // 实现待添加，未提供完整代码段
}



// 未提供完整的函数实现代码，需要继续完善此处的函数实现内容。
// 实现对应的上采样最近邻插值算法（1维）的 CPU 版本的输出函数
TORCH_IMPL_FUNC(_upsample_nearest_exact1d_out_cpu) (
    const Tensor& input,                       // 输入张量
    IntArrayRef output_size,                    // 输出尺寸
    std::optional<double> scales,               // 尺度参数（可选）
    const Tensor& output                       // 输出张量
) {
    // 调用相应的最近邻插值算法的 CPU 内核函数
    _upsample_nearest_exact1d_kernel(kCPU, output, input, scales);
}

// 实现对应的上采样最近邻插值算法（1维）的反向传播 CPU 版本的输出函数
TORCH_IMPL_FUNC(upsample_nearest1d_backward_out_cpu) (
    const Tensor& grad_output,                  // 梯度输出张量
    IntArrayRef output_size,                    // 输出尺寸
    IntArrayRef input_size,                     // 输入尺寸
    std::optional<double> scales,               // 尺度参数（可选）
    const Tensor& grad_input                    // 梯度输入张量
) {
    // 清零梯度输入张量
    grad_input.zero_();
    // 调用上采样最近邻插值算法反向传播 CPU 内核函数
    upsample_nearest1d_backward_kernel(kCPU, grad_input, grad_output, scales);
}

// 实现对应的上采样最近邻插值算法（1维）的精确版本反向传播 CPU 输出函数
TORCH_IMPL_FUNC(_upsample_nearest_exact1d_backward_out_cpu) (
    const Tensor& grad_output,                  // 梯度输出张量
    IntArrayRef output_size,                    // 输出尺寸
    IntArrayRef input_size,                     // 输入尺寸
    std::optional<double> scales,               // 尺度参数（可选）
    const Tensor& grad_input                    // 梯度输入张量
) {
    // 清零梯度输入张量
    grad_input.zero_();
    // 调用上采样最近邻插值算法的精确版本反向传播 CPU 内核函数
    _upsample_nearest_exact1d_backward_kernel(kCPU, grad_input, grad_output, scales);
}

// 使用命名空间 at::native

using at::native::upsample::compute_output_size;  // 引用 compute_output_size 函数
using at::native::upsample::get_scale_value;      // 引用 get_scale_value 函数

// 实现上采样最近邻插值算法（1维）的正常版本的函数
Tensor upsample_nearest1d(
    const Tensor& input,                        // 输入张量
    at::OptionalIntArrayRef output_size,        // 输出尺寸（可选）
    std::optional<ArrayRef<double>> scale_factors  // 尺度因子（可选）
) {
    // 计算输出尺寸
    auto osize = compute_output_size(input.sizes(), output_size, scale_factors);
    // 获取尺度因子值
    auto scale_w = get_scale_value(scale_factors, 0);
    // 调用 ATen 的上采样最近邻插值算法（1维）
    return at::upsample_nearest1d(input, osize, scale_w);
}

// 实现上采样最近邻插值算法（1维）的精确版本的函数
Tensor _upsample_nearest_exact1d(
    const Tensor& input,                        // 输入张量
    at::OptionalIntArrayRef output_size,        // 输出尺寸（可选）
    std::optional<ArrayRef<double>> scale_factors  // 尺度因子（可选）
) {
    // 计算输出尺寸
    auto osize = compute_output_size(input.sizes(), output_size, scale_factors);
    // 获取尺度因子值
    auto scale_w = get_scale_value(scale_factors, 0);
    // 调用 ATen 的上采样最近邻插值算法的精确版本（1维）
    return at::_upsample_nearest_exact1d(input, osize, scale_w);
}

// 定义调度分发的函数，用于上采样最近邻插值算法（1维）的 CPU 内核
DEFINE_DISPATCH(upsample_nearest1d_kernel);
DEFINE_DISPATCH(_upsample_nearest_exact1d_kernel);
DEFINE_DISPATCH(upsample_nearest1d_backward_kernel);
DEFINE_DISPATCH(_upsample_nearest_exact1d_backward_kernel);

} // namespace at::native
```