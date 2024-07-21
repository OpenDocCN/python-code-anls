# `.\pytorch\aten\src\ATen\native\UpSampleLinear1d.cpp`

```py
// Adapted from interp.cpp from Caffe util by Pauline Luc
// Originally developed by George Papandreou

// 定义宏，仅允许方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 包含必要的头文件
#include <ATen/core/Tensor.h>
#include <ATen/TensorMeta.h>
#include <ATen/TensorUtils.h>
#include <ATen/native/UpSample.h>

// 根据情况选择是否包含操作函数头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/upsample_linear1d.h>
#include <ATen/ops/upsample_linear1d_backward.h>
#include <ATen/ops/upsample_linear1d_backward_native.h>
#include <ATen/ops/upsample_linear1d_native.h>
#endif

// 定义命名空间 at::meta
namespace at::meta {

// 定义 TORCH_META_FUNC 宏，用于上采样线性插值的元函数
TORCH_META_FUNC(upsample_linear1d) (
    const Tensor& input,                // 输入张量
    IntArrayRef output_size,            // 输出大小
    bool align_corners,                 // 是否对齐角点
    std::optional<double> scales        // 可选的缩放因子
) {
  // 获取完整的输出大小
  auto full_output_size = native::upsample_1d_common_check(input.sizes(), output_size);

  // 检查输入张量的维度和是否为空
  TORCH_CHECK(
      (input.size(1) != 0 && input.size(2) != 0) && input.dim() == 3,
      "Non-empty 3D data tensor expected but got a tensor with sizes ",
      input.sizes());

  // 设置输出张量的原始步长化存储
  set_output_raw_strided(0, full_output_size, {}, input.options());
}

// 定义 TORCH_META_FUNC 宏，用于上采样线性插值的反向元函数
TORCH_META_FUNC(upsample_linear1d_backward) (
    const Tensor& grad_output,          // 梯度输出张量
    IntArrayRef output_size,            // 输出大小
    IntArrayRef input_size,             // 输入大小
    bool align_corners,                 // 是否对齐角点
    std::optional<double> scales        // 可选的缩放因子
) {
  // 获取完整的输出大小
  auto full_output_size = native::upsample_1d_common_check(input_size, output_size);

  // 检查输入大小的维度是否为3
  TORCH_CHECK(
      input_size.size() == 3,
      "It is expected input_size equals to 3, but got size ",
      input_size.size());

  // 检查梯度输出张量的维度和大小
  check_dim_size(grad_output, 3, 0, full_output_size[0]);
  check_dim_size(grad_output, 3, 1, full_output_size[1]);
  check_dim_size(grad_output, 3, 2, full_output_size[2]);

  // 设置输出张量的原始步长化存储
  set_output_raw_strided(0, input_size, {}, grad_output.options());
}

} // namespace at::meta

// 定义命名空间 at::native
namespace at::native {

// 定义 TORCH_IMPL_FUNC 宏，实现 CPU 上的上采样线性插值
TORCH_IMPL_FUNC(upsample_linear1d_out_cpu) (
    const Tensor& input,                // 输入张量
    IntArrayRef output_size,            // 输出大小
    bool align_corners,                 // 是否对齐角点
    std::optional<double> scales,       // 可选的缩放因子
    const Tensor& output                // 输出张量
) {
  // 调用 CPU 上的上采样线性插值核函数
  upsample_linear1d_kernel(kCPU, output, input, align_corners, scales);
}

// 定义 TORCH_IMPL_FUNC 宏，实现 CPU 上的上采样线性插值反向传播
TORCH_IMPL_FUNC(upsample_linear1d_backward_out_cpu) (
    const Tensor& grad_output,          // 梯度输出张量
    IntArrayRef output_size,            // 输出大小
    IntArrayRef input_size,             // 输入大小
    bool align_corners,                 // 是否对齐角点
    std::optional<double> scales,       // 可选的缩放因子
    const Tensor& grad_input            // 梯度输入张量
) {
  // 将梯度输入张量清零
  grad_input.zero_();
  // 调用 CPU 上的上采样线性插值反向传播核函数
  upsample_linear1d_backward_kernel(kCPU, grad_input, grad_output, align_corners, scales);
}

// 使用命名空间 at::native::upsample 中的函数
// 定义上采样线性插值函数，支持向量化变体
Tensor upsample_linear1d(
    const Tensor& input,                            // 输入张量
    at::OptionalIntArrayRef output_size,            // 输出大小
    bool align_corners,                             // 是否对齐角点
    std::optional<ArrayRef<double>> scale_factors   // 可选的缩放因子数组
) {
  // 计算输出大小
  auto osize = compute_output_size(input.sizes(), output_size, scale_factors);
  // 获取缩放因子
  auto scale_w = get_scale_value(scale_factors, 0);
  // 调用上采样线性插值函数
  return at::upsample_linear1d(input, osize, align_corners, scale_w);
}

// 定义上采样线性插值核函数的分发宏
DEFINE_DISPATCH(upsample_linear1d_kernel);

} // namespace at::native
DEFINE_DISPATCH(upsample_linear1d_backward_kernel);


// 定义一个名为 upsample_linear1d_backward_kernel 的调度分发器
DEFINE_DISPATCH(upsample_linear1d_backward_kernel);



} // namespace at::native


// 结束命名空间 at::native
```