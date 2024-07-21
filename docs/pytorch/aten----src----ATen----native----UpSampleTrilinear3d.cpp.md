# `.\pytorch\aten\src\ATen\native\UpSampleTrilinear3d.cpp`

```py
// Adapted from interp.cpp from Caffe util by Pauline Luc
// Originally developed by George Papandreou
// 定义仅供方法操作符使用的宏
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/core/Tensor.h>
#include <ATen/TensorMeta.h>
#include <ATen/native/UpSample.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/upsample_trilinear3d.h>
#include <ATen/ops/upsample_trilinear3d_backward.h>
#include <ATen/ops/upsample_trilinear3d_backward_native.h>
#include <ATen/ops/upsample_trilinear3d_native.h>
#endif

// ATen 库的命名空间
namespace at::meta {

// 定义 TORCH_META_FUNC 宏用于声明 upsample_trilinear3d 函数元信息
TORCH_META_FUNC(upsample_trilinear3d) (
  const Tensor& input,
  IntArrayRef output_size,
  bool align_corners,
  std::optional<double> scales_d,
  std::optional<double> scales_h,
  std::optional<double> scales_w
) {
  // 调用 native 命名空间中的函数，进行 3D 上采样的通用检查
  auto full_output_size = native::upsample_3d_common_check(input.sizes(), output_size);

  // 检查输入张量是否非空，或者批次大小为空，但其它维度不为空
  TORCH_CHECK(
      input.numel() != 0 || c10::multiply_integers(input.sizes().begin() + 1, input.sizes().end()),
      "Non-empty 5D data tensor expected but got a tensor with sizes ",
      input.sizes());

  // 设置输出张量的原始步长，以及内存格式
  set_output_raw_strided(0, full_output_size, {}, input.options().memory_format(input.suggest_memory_format()));
}

// 定义 TORCH_META_FUNC 宏用于声明 upsample_trilinear3d_backward 函数元信息
TORCH_META_FUNC(upsample_trilinear3d_backward) (
  const Tensor& grad_output,
  IntArrayRef output_size,
  IntArrayRef input_size,
  bool align_corners,
  std::optional<double> scales_d,
  std::optional<double> scales_h,
  std::optional<double> scales_w
) {
  // 调用 native 命名空间中的函数，进行 3D 上采样的通用检查
  auto full_output_size = native::upsample_3d_common_check(input_size, output_size);

  // 检查梯度输出张量是否为五维
  TORCH_CHECK(
      grad_output.dim() == 5,
      "Expected grad_output to be a tensor of dimension 5 but got: dimension ", grad_output.dim());

  // 检查每个维度上的大小是否匹配
  for (const auto i : c10::irange(5)) {
    TORCH_CHECK(
        grad_output.size(i) == full_output_size[i],
        "Expected grad_output to have the same shape as output;",
        " output.size(", i, ") = ", full_output_size[i],
        " but got grad_output.size(", i, ") = ", grad_output.size(i));
  }

  // 设置输出张量的原始步长，以及内存格式
  set_output_raw_strided(0, input_size, {}, grad_output.options().memory_format(grad_output.suggest_memory_format()));
}

} // namespace at::meta

// ATen 库的命名空间
namespace at::native {

// 实现上采样三线性插值的 CPU 版本
TORCH_IMPL_FUNC(upsample_trilinear3d_out_cpu) (
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners,
    std::optional<double> scales_d,
    std::optional<double> scales_h,
    std::optional<double> scales_w,
    const Tensor& output
) {
  // 调用 CPU 版本的上采样三线性插值核函数
  upsample_trilinear3d_kernel(kCPU, output, input, align_corners, scales_d, scales_h, scales_w);
}

// 实现上采样三线性插值的 CPU 版本的反向传播
TORCH_IMPL_FUNC(upsample_trilinear3d_backward_out_cpu) (
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners,
    std::optional<double> scales_d,
    std::optional<double> scales_h,
    std::optional<double> scales_w,
    const Tensor& grad_input
) {
// 清零输入梯度张量
grad_input.zero_();
// 调用 CPU 上的三线性插值三维反向传播内核函数，处理输入和输出梯度
upsample_trilinear3d_backward_kernel(kCPU, grad_input, grad_output, align_corners, scales_d, scales_h, scales_w);
}

// vec 变体

// 引用上采样命名空间中的计算输出尺寸和获取比例值函数
using at::native::upsample::compute_output_size;
using at::native::upsample::get_scale_value;

// 定义三线性插值三维上采样函数，返回处理后的张量
Tensor upsample_trilinear3d(
    const Tensor& input, // 输入张量
    at::OptionalIntArrayRef output_size, // 可选的输出尺寸
    bool align_corners, // 是否对齐角点
    std::optional<ArrayRef<double>> scale_factors) { // 可选的比例因子数组引用
  auto osize = compute_output_size(input.sizes(), output_size, scale_factors); // 计算输出尺寸
  auto scale_d = get_scale_value(scale_factors, 0); // 获取深度方向的比例值
  auto scale_h = get_scale_value(scale_factors, 1); // 获取高度方向的比例值
  auto scale_w = get_scale_value(scale_factors, 2); // 获取宽度方向的比例值
  // 调用 ATen 库中的三线性插值三维上采样函数，并返回结果
  return at::upsample_trilinear3d(input, osize, align_corners, scale_d, scale_h, scale_w);
}

// 定义三线性插值三维反向传播内核分发函数
DEFINE_DISPATCH(upsample_trilinear3d_kernel);
// 定义三线性插值三维反向传播内核函数分发函数
DEFINE_DISPATCH(upsample_trilinear3d_backward_kernel);

} // namespace at::native
```