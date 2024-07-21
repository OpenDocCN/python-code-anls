# `.\pytorch\aten\src\ATen\native\Activation.cpp`

```
// 定义宏，指定仅使用方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 包含激活函数相关的头文件
#include <ATen/native/Activation.h>

// 包含张量相关的核心头文件
#include <ATen/core/Tensor.h>
// 包含分发机制相关的头文件
#include <ATen/Dispatch.h>
// 包含张量迭代器相关的头文件
#include <ATen/TensorIterator.h>
// 包含张量操作相关的头文件
#include <ATen/TensorOperators.h>
// 包含数学操作类型相关的头文件
#include <ATen/OpMathType.h>
// 包含并行处理相关的头文件
#include <ATen/Parallel.h>
// 包含标量操作相关的头文件
#include <ATen/ScalarOps.h>

// 如果编译器定义了 C10_MOBILE 和 USE_XNNPACK，则包含 XNNPACK 引擎相关的头文件
#if defined(C10_MOBILE) && defined(USE_XNNPACK)
#include <ATen/native/xnnpack/Engine.h>
#endif

// 包含分布助手相关的头文件
#include <ATen/core/DistributionsHelper.h>

// 包含 c10 库中的工具函数范围头文件
#include <c10/util/irange.h>
// 包含 c10 库中的标量类型头文件
#include <c10/core/ScalarType.h>

// 如果启用了 MKLDNN 支持，则包含 MKLDNN 相关的头文件
#if AT_MKLDNN_ENABLED()
#include <ATen/native/mkldnn/MKLDNNCommon.h>
#include <ATen/native/mkldnn/Utils.h>
#endif

// 如果未定义 AT_PER_OPERATOR_HEADERS 宏，则包含函数操作和原生函数相关的头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
// 否则，包含单独操作的原生函数头文件
#else
#include <ATen/ops/celu_native.h>
#include <ATen/ops/clamp.h>
#include <ATen/ops/clamp_min.h>
#include <ATen/ops/elu.h>
#include <ATen/ops/elu_backward_native.h>
#include <ATen/ops/elu_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/gelu_backward_native.h>
#include <ATen/ops/gelu_native.h>
#include <ATen/ops/hardshrink_backward_native.h>
#include <ATen/ops/hardshrink_native.h>
#include <ATen/ops/hardsigmoid_backward_native.h>
#include <ATen/ops/hardsigmoid_native.h>
#include <ATen/ops/hardswish_backward_native.h>
#include <ATen/ops/hardswish_native.h>
#include <ATen/ops/hardtanh.h>
#include <ATen/ops/hardtanh_backward_native.h>
#include <ATen/ops/hardtanh_native.h>
#include <ATen/ops/infinitely_differentiable_gelu_backward_native.h>
#include <ATen/ops/leaky_relu.h>
#include <ATen/ops/leaky_relu_backward.h>
#include <ATen/ops/leaky_relu_backward_native.h>
#include <ATen/ops/leaky_relu_native.h>
#include <ATen/ops/log_sigmoid_backward_native.h>
#include <ATen/ops/log_sigmoid_forward.h>
#include <ATen/ops/log_sigmoid_forward_native.h>
#include <ATen/ops/log_sigmoid_native.h>
#include <ATen/ops/mish_backward_native.h>
#include <ATen/ops/mish_native.h>
#include <ATen/ops/prelu_native.h>
#include <ATen/ops/_prelu_kernel.h>
#include <ATen/ops/_prelu_kernel_native.h>
#include <ATen/ops/_prelu_kernel_backward_native.h>
#include <ATen/ops/relu6_native.h>
#include <ATen/ops/relu_native.h>
#include <ATen/ops/rrelu_native.h>
#include <ATen/ops/rrelu_with_noise.h>
#include <ATen/ops/rrelu_with_noise_backward_native.h>
#include <ATen/ops/rrelu_with_noise_native.h>
#include <ATen/ops/selu_native.h>
#include <ATen/ops/sigmoid.h>
#include <ATen/ops/silu_backward_native.h>
#include <ATen/ops/silu_native.h>
#include <ATen/ops/softplus.h>
#include <ATen/ops/softplus_backward_native.h>
#include <ATen/ops/softplus_native.h>
#include <ATen/ops/softshrink_backward_native.h>
#include <ATen/ops/softshrink_native.h>
#include <ATen/ops/tanh.h>
#include <ATen/ops/threshold_backward_native.h>
#include <ATen/ops/threshold_native.h>
#endif

// at 命名空间下的 meta 命名空间
namespace at::meta {
// 计算 `result = self <= threshold ? value : other` 的元函数
// 其中 other 在 threshold() 中是 self，在 threshold_backward() 中是 grad
// 定义名为 threshold 的 Torch 元函数，计算 self <= threshold ? value : other
TORCH_META_FUNC(threshold)(const Tensor& self, const Scalar& threshold, const Scalar& value) {
  // 获取可能的输出张量
  const Tensor& result = maybe_get_output();
  // 构建张量迭代器配置
  build(TensorIteratorConfig()
    .set_check_mem_overlap(false)  // threshold 是幂等的，因此内存重叠是可以接受的
    .add_output(result)             // 添加输出张量 result
    .add_const_input(self)          // 添加常量输入 self
    .add_const_input(self)          // 添加常量输入 self（其他）
    .allow_cpu_scalars(true)       // 允许 CPU 标量
    .promote_inputs_to_common_dtype(true)  // 提升输入为公共数据类型
    .cast_common_dtype_to_outputs(true)    // 将公共数据类型转换为输出类型
    .enforce_safe_casting_to_output(true)); // 强制安全转换到输出类型
}

// 定义名为 threshold_backward 的 Torch 元函数，用于计算梯度反向传播
// other 在 threshold() 中是 self，在 threshold_backward() 中是 grad
TORCH_META_FUNC(threshold_backward)(const Tensor& grad, const Tensor& self, const Scalar& threshold) {
  // 获取可能的输出张量 gradInput
  const Tensor& gradInput = maybe_get_output();
  // 构建张量迭代器配置
  build(TensorIteratorConfig()
    .set_check_mem_overlap(false)  // threshold 是幂等的，因此内存重叠是可以接受的
    .add_output(gradInput)          // 添加输出张量 gradInput
    .add_const_input(self)          // 添加常量输入 self
    .add_const_input(grad)          // 添加常量输入 grad（其他）
    .allow_cpu_scalars(true)       // 允许 CPU 标量
    .promote_inputs_to_common_dtype(true)  // 提升输入为公共数据类型
    .cast_common_dtype_to_outputs(true)    // 将公共数据类型转换为输出类型
    .enforce_safe_casting_to_output(true)); // 强制安全转换到输出类型
}

// 定义名为 elu 的 Torch 元函数，执行 ELU（指数线性单元）操作
TORCH_META_FUNC(elu) (
  const Tensor& self, const Scalar& alpha, const Scalar& scale, const Scalar& input_scale
) {
  // 使用 self 构建一元操作
  build_unary_op(maybe_get_output(), self);
}

// 定义名为 elu_backward 的 Torch 元函数，执行 ELU 反向传播操作
TORCH_META_FUNC(elu_backward) (
  const Tensor& grad_output,
  const Scalar& alpha,
  const Scalar& scale,
  const Scalar& input_scale,
  bool is_result,
  const Tensor& self_or_result
) {
  // 检查是否结果或 self_or_result 中的 alpha 小于 0.0
  TORCH_CHECK(
    !is_result || alpha.to<double>() >= 0.0,
    "In-place elu backward calculation is triggered with a negative slope which is not supported. "
    "This is caused by calling in-place forward function with a negative slope, "
    "please call out-of-place version instead.");
  
  // 使用 grad_output 和 self_or_result 构建借用二元操作
  build_borrowing_binary_op(maybe_get_output(), grad_output, self_or_result);
}

// 定义名为 silu 的 Torch 元函数，执行 SiLU（Sigmoid Linearity）操作
TORCH_META_FUNC(silu) (const Tensor& self) {
  // 使用 self 构建一元操作
  build_unary_op(maybe_get_output(), self);
}

// 定义名为 silu_backward 的 Torch 元函数，执行 SiLU 反向传播操作
TORCH_META_FUNC(silu_backward) (
  const Tensor& grad_output, const Tensor& input
) {
  // 使用 grad_output 和 input 构建借用二元操作
  build_borrowing_binary_op(maybe_get_output(), grad_output, input);
}

// 定义名为 mish 的 Torch 元函数，执行 Mish（Mish Activation）操作
TORCH_META_FUNC(mish) (const Tensor& self) {
  // 使用 self 构建一元操作
  build_unary_op(maybe_get_output(), self);
}

// 定义名为 softplus 的 Torch 元函数，执行 Softplus 操作
TORCH_META_FUNC(softplus) (
  const Tensor& self, const Scalar& beta, const Scalar& threshold
) {
  // 使用 self 构建一元操作
  build_unary_op(maybe_get_output(), self);
}

// 定义名为 softplus_backward 的 Torch 元函数，执行 Softplus 反向传播操作
TORCH_META_FUNC(softplus_backward) (
  const Tensor& grad_output,
  const Tensor& self,
  const Scalar& beta,
  const Scalar& threshold
) {
  // 使用 grad_output 和 self 构建借用二元操作
  build_borrowing_binary_op(maybe_get_output(), grad_output, self);
}

// 定义名为 leaky_relu 的 Torch 元函数，执行 Leaky ReLU 操作
TORCH_META_FUNC(leaky_relu) (
  const Tensor& self, const Scalar& negval
) {
  // 使用 self 构建一元操作
  build_unary_op(maybe_get_output(), self);
}

// 注意：Leaky ReLU 反向传播计算不支持带有负斜率的原位调用。
// 原因是，对于原位前向调用，前向结果将保存到自动求导节点而不是输入本身，
// 在计算反向梯度时，无法知道
// 实现 leaky_relu 激活函数的反向传播。当输入的斜率为负时，判断原始输入是否为正。
// 例如，如果 forward 是 2，斜率是 -0.2，那么此节点的原始输入可能是 2 或者 -10，因此在这种情况下无法正确计算反向梯度。
TORCH_META_FUNC(leaky_relu_backward) (
  const Tensor& grad_output,
  const Tensor& self_or_result,
  const Scalar& negval,
  bool is_result
) {
  // 检查是否启用了原地操作，如果斜率为负则报错，不支持使用负斜率进行原地操作的 leakyReLU 反向计算。
  TORCH_CHECK(
    !is_result || negval.to<double>() >= 0.0,
    "In-place leakyReLU 反向计算中触发了带有负斜率的情况，这是不被支持的。"
    "这是因为使用负斜率调用了原地前向函数，请改用非原地版本。"
    "如果确实需要支持带有负斜率的原地 leakyReLU 反向计算，请在 https://github.com/pytorch/pytorch 上提交问题。"
  );

  // 使用 build_borrowing_binary_op 函数构建操作，处理可能的输出和输入之间的二元操作。
  build_borrowing_binary_op(maybe_get_output(), self_or_result, grad_output);
}

// 实现 hardsigmoid 激活函数的前向计算。
TORCH_META_FUNC(hardsigmoid) (const Tensor& self) {
  // 使用 build_unary_op 函数构建操作，处理可能的输出和输入之间的一元操作。
  build_unary_op(maybe_get_output(), self);
}

// 实现 hardsigmoid 激活函数的反向传播。
TORCH_META_FUNC(hardsigmoid_backward) (const Tensor& grad_output, const Tensor& self) {
  // 使用 build_borrowing_binary_op 函数构建操作，处理可能的输出和输入之间的二元操作。
  build_borrowing_binary_op(maybe_get_output(), grad_output, self);
}

// 实现 hardshrink 激活函数的前向计算。
TORCH_META_FUNC(hardshrink) (const Tensor& self, const Scalar& lambd) {
  // 使用 build_unary_op 函数构建操作，处理可能的输出和输入之间的一元操作。
  build_unary_op(maybe_get_output(), self);
}

// 实现 hardshrink 激活函数的反向传播。
TORCH_META_FUNC(hardshrink_backward) (
  const Tensor& grad,
  const Tensor& self,
  const Scalar& lambd
) {
  // 使用 build_borrowing_binary_op 函数构建操作，处理可能的输出和输入之间的二元操作。
  build_borrowing_binary_op(maybe_get_output(), grad, self);
}

// 检查 softshrink 函数的 lambd 参数是否合法。
static inline void softshrink_check(const Scalar& lambd) {
  double lamb = lambd.to<double>();
  TORCH_CHECK(lamb >= 0, "lambda 必须大于等于 0，但发现为 ", lamb, "。");
}

// 实现 softshrink 激活函数的前向计算。
TORCH_META_FUNC(softshrink) (
  const Tensor& self,
  const Scalar& lambd
) {
  // 检查 lambd 参数是否合法
  softshrink_check(lambd);
  // 使用 build_unary_op 函数构建操作，处理可能的输出和输入之间的一元操作。
  build_unary_op(maybe_get_output(), self);
}

// 实现 softshrink 激活函数的反向传播。
TORCH_META_FUNC(softshrink_backward) (
  const Tensor& grad,
  const Tensor& self,
  const Scalar& lambd
) {
  // 使用 build_borrowing_binary_op 函数构建操作，处理可能的输出和输入之间的二元操作。
  build_borrowing_binary_op(maybe_get_output(), grad, self);
}

// 实现 gelu 激活函数的前向计算。
TORCH_META_FUNC(gelu) (const Tensor& self, c10::string_view approximate) {
  // 使用 build_unary_op 函数构建操作，处理可能的输出和输入之间的一元操作。
  build_unary_op(maybe_get_output(), self);
}

// 实现 gelu 激活函数的反向传播。
TORCH_META_FUNC(gelu_backward) (
  const Tensor& grad,
  const Tensor& self,
  c10::string_view approximate
) {
  // 使用 build_borrowing_binary_op 函数构建操作，处理可能的输出和输入之间的二元操作。
  build_borrowing_binary_op(maybe_get_output(), grad, self);
}

} // namespace at::meta

namespace at::native {

// SELU 激活函数的 alpha 和 scale 值的常量定义
static const double SELU_ALPHA = 1.6732632423543772848170429916717;
static const double SELU_SCALE = 1.0507009873554804934193349852946;

// 定义各种激活函数的分发函数，通过调用这些函数来执行对应的操作。
DEFINE_DISPATCH(elu_stub);
DEFINE_DISPATCH(elu_backward_stub);
DEFINE_DISPATCH(softplus_stub);
DEFINE_DISPATCH(softplus_backward_stub);
DEFINE_DISPATCH(log_sigmoid_cpu_stub);
DEFINE_DISPATCH(log_sigmoid_backward_stub);
DEFINE_DISPATCH(threshold_stub);
DEFINE_DISPATCH(hardtanh_backward_stub);
DEFINE_DISPATCH(hardsigmoid_stub);
DEFINE_DISPATCH(hardsigmoid_backward_stub);
DEFINE_DISPATCH(hardswish_stub);
DEFINE_DISPATCH(hardswish_backward_stub);
DEFINE_DISPATCH(hardshrink_stub);
DEFINE_DISPATCH(softshrink_stub);
# 定义调度器函数，用于指向对应的实现函数
DEFINE_DISPATCH(shrink_backward_stub);
DEFINE_DISPATCH(leaky_relu_stub);
DEFINE_DISPATCH(leaky_relu_backward_stub);
DEFINE_DISPATCH(silu_stub);
DEFINE_DISPATCH(silu_backward_stub);
DEFINE_DISPATCH(mish_stub);
DEFINE_DISPATCH(mish_backward_stub);
DEFINE_DISPATCH(prelu_stub);
DEFINE_DISPATCH(prelu_backward_stub);

# 实现 elu_out 函数，计算 ELU 激活函数的输出
TORCH_IMPL_FUNC(elu_out) (
  const Tensor& self, const Scalar& alpha, const Scalar& scale, const Scalar& input_scale, const Tensor& result
) {
  # 调用对应设备上的 elu_stub 函数，计算 ELU 激活函数的输出
  elu_stub(device_type(), *this, alpha, scale, input_scale);
}

# 实现 elu_backward_out 函数，计算 ELU 激活函数的反向传播
TORCH_IMPL_FUNC(elu_backward_out) (
  const Tensor& grad_output,
  const Scalar& alpha,
  const Scalar& scale,
  const Scalar& input_scale,
  bool is_result,
  const Tensor& self_or_result,
  const Tensor& grad_input
) {
  # 调用对应设备上的 elu_backward_stub 函数，计算 ELU 激活函数的反向传播
  elu_backward_stub(device_type(), *this, alpha, scale, input_scale, is_result);
}

# 实现 silu_out 函数，计算 SiLU 激活函数的输出
TORCH_IMPL_FUNC(silu_out) (
  const Tensor& self, const Tensor& result
) {
  # 调用对应设备上的 silu_stub 函数，计算 SiLU 激活函数的输出
  silu_stub(device_type(), *this);
}

# 实现 silu_backward_out 函数，计算 SiLU 激活函数的反向传播
TORCH_IMPL_FUNC(silu_backward_out) (
  const Tensor& grad_output, const Tensor& input, const Tensor& grad_input
) {
  # 调用对应设备上的 silu_backward_stub 函数，计算 SiLU 激活函数的反向传播
  silu_backward_stub(device_type(), *this);
}

# 实现 mish_out 函数，计算 Mish 激活函数的输出
TORCH_IMPL_FUNC(mish_out) (
  const Tensor& self, const Tensor& result
) {
  # 调用对应设备上的 mish_stub 函数，计算 Mish 激活函数的输出
  mish_stub(device_type(), *this);
}

# 实现 softplus_out 函数，计算 Softplus 激活函数的输出
TORCH_IMPL_FUNC(softplus_out) (
  const Tensor& self, const Scalar& beta, const Scalar& threshold, const Tensor& result
) {
  # 调用对应设备上的 softplus_stub 函数，计算 Softplus 激活函数的输出
  softplus_stub(device_type(), *this, beta, threshold);
}

# 实现 softplus_backward_out 函数，计算 Softplus 激活函数的反向传播
TORCH_IMPL_FUNC(softplus_backward_out) (
  const Tensor& grad_output,
  const Tensor& self,
  const Scalar& beta,
  const Scalar& threshold,
  const Tensor& grad_input
) {
  # 调用对应设备上的 softplus_backward_stub 函数，计算 Softplus 激活函数的反向传播
  softplus_backward_stub(device_type(), *this, beta, threshold);
}

# 实现 leaky_relu_out 函数，计算 Leaky ReLU 激活函数的输出
TORCH_IMPL_FUNC(leaky_relu_out) (
  const Tensor& self, const Scalar& negval, const Tensor& result
) {
  # 调用对应设备上的 leaky_relu_stub 函数，计算 Leaky ReLU 激活函数的输出
  leaky_relu_stub(device_type(), *this, negval);
}

# 实现 leaky_relu_backward_out 函数，计算 Leaky ReLU 激活函数的反向传播
TORCH_IMPL_FUNC(leaky_relu_backward_out) (
  const Tensor& grad_output,
  const Tensor& self_or_result,
  const Scalar& negval,
  bool is_result,
  const Tensor& grad_input
) {
  # 调用对应设备上的 leaky_relu_backward_stub 函数，计算 Leaky ReLU 激活函数的反向传播
  leaky_relu_backward_stub(device_type(), *this, negval);
}

# 实现 hardsigmoid_out 函数，计算 Hardsigmoid 激活函数的输出
TORCH_IMPL_FUNC(hardsigmoid_out) (
  const Tensor& self, const Tensor& result
) {
  # 调用对应设备上的 hardsigmoid_stub 函数，计算 Hardsigmoid 激活函数的输出
  hardsigmoid_stub(device_type(), *this);
}

# 实现 hardsigmoid_backward_out 函数，计算 Hardsigmoid 激活函数的反向传播
TORCH_IMPL_FUNC(hardsigmoid_backward_out) (
  const Tensor& grad_output, const Tensor& self, const Tensor& grad_input
) {
  # 调用对应设备上的 hardsigmoid_backward_stub 函数，计算 Hardsigmoid 激活函数的反向传播
  hardsigmoid_backward_stub(device_type(), *this);
}

# 实现 hardshrink_out 函数，计算 Hardshrink 激活函数的输出
TORCH_IMPL_FUNC(hardshrink_out) (
  const Tensor & self, const Scalar& lambd, const Tensor& result
) {
  # 调用对应设备上的 hardshrink_stub 函数，计算 Hardshrink 激活函数的输出
  hardshrink_stub(device_type(), *this, lambd);
}

# 实现 hardshrink_backward_out 函数，计算 Hardshrink 激活函数的反向传播
TORCH_IMPL_FUNC(hardshrink_backward_out) (
  const Tensor & grad, const Tensor & self, const Scalar& lambd, const Tensor& grad_input
) {
  # 调用对应设备上的 shrink_backward_stub 函数，计算 Hardshrink 激活函数的反向传播
  shrink_backward_stub(device_type(), *this, lambd);
}

# 实现 softshrink_out 函数，计算 Softshrink 激活函数的输出
TORCH_IMPL_FUNC(softshrink_out) (
  const Tensor & self, const Scalar& lambd, const Tensor& result
) {
  # 调用对应设备上的 softshrink_stub 函数，计算 Softshrink 激活函数的输出
  softshrink_stub(device_type(), *this, lambd);
}

# 实现 softshrink_backward_out 函数，计算 Softshrink 激活函数的反向传播
TORCH_IMPL_FUNC(softshrink_backward_out) (
  const Tensor & grad, const Tensor & self, const Scalar& lambd, const Tensor& grad_input
) {
  # 调用对应设备上的 shrink_backward_stub 函数，计算 Softshrink 激活函数的反向传播
  shrink_backward_stub(device_type(), *this, lambd);
}
// 调用 shrink_backward_stub 函数，该函数可能用于执行后向传播时的某些计算
) {
  shrink_backward_stub(device_type(), *this, lambd);
}

#if AT_MKLDNN_ENABLED()
// 判断是否可以使用 MKLDNN 加速，条件包括全局 MKLDNN 开启以及输入张量符合要求
static bool use_mkldnn(const Tensor& input) {
  if (!at::globalContext().userEnabledMkldnn()) {
    return false;
  }
  if (!input.is_contiguous() || input.numel() <= 1) {
    return false;
  }
  return (input.is_mkldnn()) || // 输入张量为 MKLDNN 张量
    (input.device().is_cpu() &&
    (((input.scalar_type() == kBFloat16) && mkldnn_bf16_device_check()) ||
    (input.scalar_type() == kFloat))); // 输入为稠密布局且为 bfloat16/float32 类型
}
#endif

// 实现 gelu_out_cpu 函数，根据输入张量和参数选择是否使用 MKLDNN 加速执行
TORCH_IMPL_FUNC(gelu_out_cpu) (
  const Tensor& self, c10::string_view approximate, const Tensor& result
) {
auto approximate_type = get_gelutype_enum(approximate);
#if AT_MKLDNN_ENABLED()
  // 如果可以使用 MKLDNN 并且近似类型为 None，则使用 MKLDNN 执行 gelu 函数
  if (use_mkldnn(self) && (approximate_type == GeluType::None)) {
    const ideep::tensor& x = itensor_from_tensor(self, /*from_const_data_ptr*/true);
    ideep::tensor y = itensor_from_tensor(result);
    ideep::eltwise_forward::compute(
      x, y, ideep::algorithm::eltwise_gelu_erf, ideep::prop_kind::forward_training, /*alpha*/ 0.0);
  } else {
    // 否则使用 CPU 上的 GeluKernel 函数
    GeluKernel(kCPU, *this, approximate_type);
  }
#else
  // 如果未启用 MKLDNN，则直接使用 CPU 上的 GeluKernel 函数
  GeluKernel(kCPU, *this, approximate_type);
#endif
}

// 实现 gelu_backward_out_cpu 函数，根据输入张量和参数选择是否使用 MKLDNN 加速执行反向传播
TORCH_IMPL_FUNC(gelu_backward_out_cpu) (
  const Tensor& grad, const Tensor& self, c10::string_view approximate, const Tensor& grad_input
) {
auto approximate_type = get_gelutype_enum(approximate);
#if AT_MKLDNN_ENABLED()
  // 如果可以使用 MKLDNN 并且近似类型为 None，则使用 MKLDNN 执行 gelu 反向传播
  if (use_mkldnn(self) && (approximate_type == GeluType::None)) {
    const ideep::tensor& x = itensor_from_tensor(self, /*from_const_data_ptr*/true);
    ideep::tensor grady = itensor_from_tensor(grad, /*from_const_data_ptr*/true);
    ideep::tensor gradx = itensor_from_tensor(grad_input);
    ideep::eltwise_backward::compute(x, grady, gradx,
      ideep::algorithm::eltwise_gelu_erf, /*alpha*/ 0.0);
  } else {
    // 否则使用 CPU 上的 GeluBackwardKernel 函数
    GeluBackwardKernel(kCPU, *this, approximate_type);
  }
#else
  // 如果未启用 MKLDNN，则直接使用 CPU 上的 GeluBackwardKernel 函数
  GeluBackwardKernel(kCPU, *this, approximate_type);
#endif
}

// 执行 hardtanh 函数，对输入张量执行硬切线函数操作并返回结果张量
Tensor hardtanh(const Tensor& self, const Scalar& min, const Scalar& max) {
  // 创建与输入张量相同形状的空张量 result
  Tensor result = at::empty_like(self);
  // 调用 hardtanh_out 函数对输入张量执行硬切线函数操作，结果存储在 result 中并返回
  return at::hardtanh_out(result, self, min, max);
}

// 执行 hardtanh_out 函数，在给定输出张量 result 的情况下对输入张量执行硬切线函数操作
Tensor& hardtanh_out(const Tensor& self, const Scalar& min, const Scalar& max, Tensor& result) {
  // 检查输入张量是否为布尔类型，硬切线函数不支持布尔类型输入
  TORCH_CHECK(self.scalar_type() != at::kBool,
  "Bool inputs not supported for hardtanh");
  // 保留边界不导致类型提升的传统行为
  Scalar min_, max_;
  // 如果输入张量为整数类型但不包括布尔类型，则需要特别处理
  if (at::isIntegralType(self.scalar_type(), /*include_bool*/false)) {
    int64_t minval = min.toLong();
    int64_t maxval = max.toLong();
    // 对于无符号整数类型且边界为负数的情况，无法执行硬切线函数
    TORCH_CHECK(self.dtype() != at::kByte || (minval >= 0 &&
       maxval >=0), "cannot do hardtanh on an unsigned type with negative limits");
    min_ = minval;
    max_ = maxval;
  } else {
    min_ = min;
    max_ = max;
  }
  // 调用 clamp_out 函数对输入张量执行边界限制操作，并将结果存储在 result 中返回
  return at::clamp_out(result, self, min_, max_);
}

// 执行 hardtanh_ 函数，在原地对输入张量执行硬切线函数操作
Tensor& hardtanh_(Tensor& self, const Scalar& min, const Scalar& max) {
  // 调用 hardtanh_out 函数，在原地对输入张量执行硬切线函数操作
  return at::hardtanh_out(self, self, min, max);
}
Tensor& hardtanh_backward_out(const Tensor& grad_output, const Tensor& self, const Scalar& min, const Scalar& max, Tensor& grad_input) {
  // 借用 Tensor 迭代器，执行二元操作，将 grad_output 和 self 作为输入，grad_input 作为输出
  auto iter = TensorIterator::borrowing_binary_op(grad_input, grad_output, self);
  // 调用硬切线反向传播的 stub 函数，传入设备类型、迭代器以及最小值和最大值
  hardtanh_backward_stub(iter.device_type(), iter, min, max);
  // 返回更新后的 grad_input
  return grad_input;
}

Tensor hardtanh_backward(const Tensor& grad_output, const Tensor& self, const Scalar& min, const Scalar& max) {
  // 定义一个空的 Tensor 用于存储结果
  Tensor result;
  // 借用 Tensor 迭代器，执行二元操作，将 grad_output 和 self 作为输入，result 作为输出
  auto iter = TensorIterator::borrowing_binary_op(result, grad_output, self);
  // 调用硬切线反向传播的 stub 函数，传入设备类型、迭代器以及最小值和最大值
  hardtanh_backward_stub(iter.device_type(), iter, min, max);
  // 返回计算结果的输出
  return iter.output();
}

Tensor hardswish(const Tensor& self) {
  #if defined(C10_MOBILE) && defined(USE_XNNPACK)
  // 如果是移动设备且使用了 XNNPACK，检查是否可使用 XNNPACK 实现的 hardswish 函数
  if (xnnpack::use_hardswish(self)) {
    // 如果可以，则直接调用 XNNPACK 的 hardswish 函数
    return xnnpack::hardswish(self);
  }
  #endif
  // 定义一个空的 Tensor 用于存储结果
  Tensor result;
  // 借用 Tensor 迭代器，执行一元操作，将 self 作为输入，result 作为输出
  auto iter = TensorIterator::unary_op(result, self);
  // 调用硬 swish 的 stub 函数，传入设备类型和迭代器
  hardswish_stub(iter.device_type(), iter);
  // 返回计算结果的输出
  return iter.output();
}

Tensor& hardswish_out(const Tensor& self, Tensor& result) {
  // 借用 Tensor 迭代器，执行一元操作，将 self 作为输入，result 作为输出
  auto iter = TensorIterator::unary_op(result, self);
  // 调用硬 swish 的 stub 函数，传入设备类型和迭代器
  hardswish_stub(iter.device_type(), iter);
  // 返回更新后的 result
  return result;
}

Tensor& hardswish_(Tensor& self) {
  #if defined(C10_MOBILE) && defined(USE_XNNPACK)
  // 如果是移动设备且使用了 XNNPACK，检查是否可使用 XNNPACK 实现的 hardswish 函数
  if (xnnpack::use_hardswish(self)) {
    // 如果可以，则直接调用 XNNPACK 的 inplace hardswish 函数
    xnnpack::hardswish_(self);
    // 返回更新后的 self
    return self;
  }
  #endif
  // 借用 Tensor 迭代器，执行一元操作，将 self 作为输入和输出
  auto iter = TensorIterator::unary_op(self, self);
  // 调用硬 swish 的 stub 函数，传入设备类型和迭代器
  hardswish_stub(iter.device_type(), iter);
  // 返回更新后的 self
  return self;
}

Tensor hardswish_backward(const Tensor& grad_output, const Tensor& self) {
  // 定义一个空的 Tensor 用于存储梯度输入
  Tensor grad_input;
  // 借用 Tensor 迭代器，执行二元操作，将 grad_output 和 self 作为输入，grad_input 作为输出
  auto iter = TensorIterator::borrowing_binary_op(grad_input, grad_output, self);
  // 调用硬 swish 反向传播的 stub 函数，传入设备类型和迭代器
  hardswish_backward_stub(iter.device_type(), iter);
  // 返回计算结果的输出
  return iter.output();
}

Tensor relu(const Tensor & self) {
  // 检查输入 Tensor 的数据类型不能是布尔型，否则抛出异常
  TORCH_CHECK(self.scalar_type() != at::kBool, "Boolean inputs not supported for relu");
  // 使用 clamp_min 函数将所有小于 0 的值置为 0，并返回处理后的 Tensor
  return at::clamp_min(self, 0);
}

Tensor & relu_(Tensor & self) {
  // 检查输入 Tensor 的数据类型不能是布尔型，否则抛出异常
  TORCH_CHECK(self.scalar_type() != at::kBool, "Boolean inputs not supported for relu");
  // 使用 inplace 方式将所有小于 0 的值置为 0，并返回更新后的 self
  return at::clamp_min_(self, 0);
}

Tensor selu(const Tensor & self) {
  // 使用 elu 函数，传入 SELU 的 alpha 和 scale 值，计算并返回结果
  return at::elu(self, SELU_ALPHA, SELU_SCALE);
}

Tensor relu6(const Tensor & self) {
  // 使用 hardtanh 函数，设置最小值为 0，最大值为 6，实现 relu6 激活函数
  return at::hardtanh(self, /*min_val=*/0, /*max_val=*/6);
}

Tensor & selu_(Tensor & self) {
  // 使用 inplace 方式调用 elu 函数，传入 SELU 的 alpha 和 scale 值，计算并返回更新后的 self
  return at::elu_(self, SELU_ALPHA, SELU_SCALE);
}

Tensor & relu6_(Tensor & self) {
  // 使用 inplace 方式调用 hardtanh 函数，设置最小值为 0，最大值为 6，实现 relu6 激活函数，并返回更新后的 self
  return at::hardtanh_(self, /*min_val=*/0, /*max_val=*/6);
}

Tensor celu(const Tensor & self, const Scalar& alpha) {
  // 检查 alpha 值不能为 0，否则抛出异常
  TORCH_CHECK(alpha.to<double>() != 0,
      "ZeroDivisionError: alpha cannot be 0 for CELU");
  // 计算 alpha 的倒数
  double inv_alpha = 1. / alpha.to<double>();
  // 使用 elu 函数，传入 alpha、scale 和 inv_alpha 值，计算并返回结果
  return at::elu(self, alpha, Scalar(1.0), Scalar(inv_alpha));
}

Tensor & celu_(Tensor & self, const Scalar& alpha) {
  // 检查 alpha 值不能为 0，否则抛出异常
  TORCH_CHECK(alpha.to<double>() != 0,
      "ZeroDivisionError: alpha cannot be 0 for CELU");
  // 计算 alpha 的倒数
  double inv_alpha = 1. / alpha.to<double>();
  // 使用 inplace 方式调用 elu 函数，传入 alpha、scale 和 inv_alpha 值，计算并返回更新后的 self
  return at::elu_(self, alpha, Scalar(1.0), Scalar(inv_alpha));
}

Tensor math_silu_backward(
    const Tensor& grad_output,
    // 对输入的张量进行 sigmoid 操作，将结果保存在 input_sigmoid 中
    auto input_sigmoid = at::sigmoid(input);
    // 计算反向传播的梯度，使用链式法则进行计算：
    // grad_output * (input_sigmoid * (1 + input * (1 - input_sigmoid)))
    // 其中，grad_output 是反向传播中的输入梯度，input 是 sigmoid 操作的输入张量
    return grad_output * (input_sigmoid * (1 + input * (1 - input_sigmoid)));
}

// 定义了 mish 激活函数的反向传播操作，计算输入和输出梯度
Tensor mish_backward(
    const Tensor& grad_output,  // 输入参数：输出梯度
    const Tensor& input) {      // 输入参数：输入数据
  // 创建一个与输入数据类型一致的空张量，用于存储输入梯度
  Tensor grad_input = at::empty({0}, input.options());
  // 创建一个张量迭代器，执行二元操作：grad_input 和 grad_output 与 input 之间的操作
  auto iter = TensorIterator::binary_op(grad_input, grad_output, input);
  // 调用底层的 stub 函数执行 mish 激活函数的反向传播
  mish_backward_stub(iter.device_type(), iter);
  // 返回计算得到的输入梯度
  return grad_input;
}

// 定义了 math_mish 激活函数的反向传播操作
Tensor math_mish_backward(
    const Tensor& grad_output,  // 输入参数：输出梯度
    const Tensor& input) {      // 输入参数：输入数据
  // 计算输入数据的 tanh(softplus(input)) 和 sigmoid(input)
  auto input_tanh_softplus = at::tanh(at::softplus(input));
  auto input_sigmoid = at::sigmoid(input);
  // 返回根据 math_mish 函数的导数计算得到的梯度
  return grad_output * (input_tanh_softplus + (input * input_sigmoid * (1 - input_tanh_softplus * input_tanh_softplus)));
}

// 定义了 _rrelu_with_noise_train 函数模板，用于在训练中计算 RReLU 激活函数及其噪声
template <typename scalar_t>
inline void _rrelu_with_noise_train(
    Tensor& output,                  // 输出参数：计算结果
    const Tensor& input,             // 输入参数：输入数据
    const Tensor& noise,             // 输入参数：噪声数据
    const Scalar& lower_,            // 输入参数：下界
    const Scalar& upper_,            // 输入参数：上界
    std::optional<Generator> generator) {  // 输入参数：随机数生成器（可选）
  using opmath_t = at::opmath_type<scalar_t>;
  // 将下界和上界转换为对应类型的数值
  opmath_t lower = lower_.to<opmath_t>();
  opmath_t upper = upper_.to<opmath_t>();
  // 使输出张量连续存储，准备进行操作
  Tensor tmp_tensor = output.contiguous();
  // 获取输出数据指针及输入数据和噪声数据的常量指针
  scalar_t* output_data = tmp_tensor.data_ptr<scalar_t>();
  const scalar_t* input_data = input.const_data_ptr<scalar_t>();
  scalar_t* noise_data = noise.data_ptr<scalar_t>();
  // 获取默认的 CPU 随机数生成器，并加锁，以确保线程安全
  auto gen  = at::get_generator_or_default<CPUGeneratorImpl>(generator, detail::getDefaultCPUGenerator());
  std::lock_guard<std::mutex> lock(gen->mutex_);
  // 遍历输入数据的每个元素
  for (const auto i : c10::irange(input.numel())) {
    // 若输入数据小于等于 0，则按照一定规则生成随机数 r，并更新输出和噪声数据
    if (input_data[i] <= 0) {
      at::uniform_real_distribution<double> uniform(lower, upper);
      const opmath_t r = (opmath_t)uniform(gen);
      output_data[i] = input_data[i] * r;
      noise_data[i] = r;
    } else {
      // 若输入数据大于 0，则更新噪声数据为 1，输出数据不变
      noise_data[i] = 1;
      output_data[i] = input_data[i];
    }
  }
  // 如果输出张量不是连续存储的，则将计算结果复制回输出张量
  if (!output.is_contiguous()) {
    output.copy_(tmp_tensor);
  }
}

// 定义了 rrelu_with_noise_out_cpu 函数，根据训练状态计算 RReLU 激活函数及其噪声
Tensor& rrelu_with_noise_out_cpu(const Tensor& self,  // 输入参数：自身张量
    const Tensor& noise,           // 输入参数：噪声张量
    const Scalar& lower,           // 输入参数：下界
    const Scalar& upper,           // 输入参数：上界
    bool training,                 // 输入参数：训练状态
    std::optional<Generator> generator,  // 输入参数：随机数生成器（可选）
    Tensor& output) {              // 输出参数：计算结果张量
  // 检查自身张量和噪声张量的形状是否匹配
  TORCH_CHECK(self.sym_sizes() == noise.sym_sizes(), "noise tensor shape must match self tensor shape. Got self.shape = ", self.sym_sizes(), " noise.shape = ", noise.sym_sizes());
  if (training) {
    // 如果处于训练状态，则根据自身类型调度 _rrelu_with_noise_train 函数进行计算
    AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::BFloat16, self.scalar_type(), "rrelu_with_noise_out_cpu", [&] {
      _rrelu_with_noise_train<scalar_t>(output, self.contiguous(), noise, lower, upper, generator);
    });
    // 返回计算结果张量
    return output;
  } else {
    // 如果非训练状态，则计算 negative_slope，并调用 leaky_relu_out 函数计算结果
    auto lower_tensor = scalar_to_tensor(lower);
    auto upper_tensor = scalar_to_tensor(upper);
    auto negative = (lower_tensor + upper_tensor) / 2;
    Scalar negative_slope = negative.item();
    return at::leaky_relu_out(output, self, negative_slope);
  }
}

// 定义了 rrelu_with_noise_cpu 函数，根据训练状态计算 RReLU 激活函数及其噪声
Tensor rrelu_with_noise_cpu(
    const Tensor& self,           // 输入参数：自身张量
    const Tensor& noise,          // 输入参数：噪声张量
    const Scalar& lower,          // 输入参数：下界
    const Scalar& upper,          // 输入参数：上界
    bool training,                // 输入参数：训练状态
    # 使用 std::optional 包装的 Generator 对象（如果提供）来生成随机数
    std::optional<Generator> generator) {
  # 创建一个与 self 张量相同形状的新张量 output，但使用旧的内存布局格式
  auto output = at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  # 调用 CPU 上的 rrelu_with_noise_out_cpu 函数，传入参数 self 张量、噪声张量、下界、上界、训练标志、移动语义的 generator、以及输出张量 output
  return at::native::rrelu_with_noise_out_cpu(
      self, noise, lower, upper, training, std::move(generator), output);
}

Tensor& rrelu_with_noise_cpu_(
    Tensor& self,
    const Tensor& noise,
    const Scalar& lower,
    const Scalar& upper,
    bool training,
    std::optional<Generator> generator) {
  // 调用 rrelu_with_noise_out_cpu 函数来执行 RReLU 激活函数操作
  return at::native::rrelu_with_noise_out_cpu(
      self, noise, lower, upper, training, std::move(generator), self);
}

Tensor rrelu_with_noise_backward(
    const Tensor& grad_output,
    const Tensor& self_or_result,
    const Tensor& noise,
    const Scalar& lower,
    const Scalar& upper,
    bool training,
    bool is_result) {
  if (training) {
    // 如果处于训练模式，则返回噪声和梯度输出的乘积作为反向传播的结果
    return noise * grad_output;
  } else {
    // 计算 lower 和 upper 的中间值，然后调用 leaky_relu_backward 函数进行反向传播计算
    auto l = lower.toDouble();
    auto u = upper.toDouble();
    auto mid = (l + u) / 2.;
    return at::leaky_relu_backward(grad_output, self_or_result, mid, is_result);
  }
}

Tensor rrelu(const Tensor & self, const Scalar& lower, const Scalar& upper, bool training, std::optional<Generator> generator) {
  // 检查 lower 和 upper 是否满足条件
  TORCH_CHECK(lower.to<double>() <= upper.to<double>(), "Lower bound should be less than or equal to the upper bound")
  // 调用 rrelu_with_noise 函数，返回 RReLU 激活后的张量
  return at::rrelu_with_noise(self, at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT), lower, upper, training, std::move(generator));
}

Tensor & rrelu_(Tensor & self, const Scalar& lower, const Scalar& upper, bool training, std::optional<Generator> generator) {
  // 检查 lower 和 upper 是否满足条件
  TORCH_CHECK(lower.to<double>() <= upper.to<double>(), "Lower bound should be less than or equal to the upper bound")
  // 调用 rrelu_with_noise_ 函数，直接在原地修改 self 张量并返回
  return at::rrelu_with_noise_(self, at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT), lower, upper, training, std::move(generator));
}

TORCH_IMPL_FUNC(threshold_out)(const Tensor& self, const Scalar& threshold, const Scalar& value, const Tensor& result) {
  // 调用 threshold_stub 函数处理阈值化操作
  threshold_stub(device_type(), *this, threshold, value);
}

TORCH_IMPL_FUNC(threshold_backward_out)(const Tensor& grad, const Tensor& self, const Scalar& threshold, const Tensor& gradInput) {
  // 调用 threshold_stub 函数处理阈值化反向传播操作
  threshold_stub(device_type(), *this, threshold, 0);
}

Tensor prelu(const Tensor& self, const Tensor& weight_) {
  // 检查权重张量是否已定义
  TORCH_INTERNAL_ASSERT(weight_.defined());
  auto self_dim = self.dim();
  // 检查张量类型是否相同
  TORCH_CHECK(self.scalar_type() == weight_.scalar_type(),
              "prelu: Type promoting not supported. Got ",
              self.scalar_type(), " and ", weight_.scalar_type());
  if (weight_.sym_numel() != 1) {
    // 如果权重不是标量，则检查输入张量维度
    TORCH_CHECK(self_dim > 0, "Not allow zero-dim input tensor.");

    auto channel_size = self_dim > 1 ? self.sym_size(1) : 1; // channel_size 默认为 1
    // 检查参数数量和输入通道大小是否匹配
    TORCH_CHECK(channel_size == weight_.sym_numel(),
      "Mismatch of parameter numbers and input channel size. Found parameter numbers = ", weight_.numel(),
      " and channel size = ", channel_size, ".");
  }

  TORCH_CHECK(
    weight_.dim() <= 1,
    "prelu: Expected `weight` to be a scalar or 1D tensor, but got: ndim = ", weight_.dim());
  // 调整权重张量以便广播到输入张量的维度
  auto weight = weight_;
  if (self_dim != weight.dim()) {
    SymDimVector dim_w(self_dim, 1);
    # 如果 self_dim 大于 1，则更新 dim_w 数组的第二个元素为 weight_ 的元素数量
    if (self_dim > 1) {
      dim_w[1] = weight_.sym_numel();
    }
    # 将 weight 张量重塑为指定形状，这里的 reshape_symint() 方法会返回一个视图，
    # 在 CPU/CUDA 上始终如此，但某些后端（如 MKLDNN）不支持视图操作
    weight = weight.reshape_symint(dim_w);
  }
  # 调用 _prelu_kernel 函数，传入 self 和处理后的 weight 张量作为参数，并返回结果
  return at::_prelu_kernel(self, weight);
Tensor _prelu_kernel(const Tensor& self, const Tensor& weight) {
  // 创建一个与输入张量self相同形状和数据类型的空张量result
  auto result = at::empty_like(self);
  // 构建张量迭代器配置，用于执行element-wise操作，将结果写入result中
  auto iter = TensorIteratorConfig()
    .add_output(result)  // 将result作为输出
    .add_const_input(self)  // 将self作为常量输入
    .add_const_input(weight)  // 将weight作为常量输入
    .build();  // 构建迭代器
  // 调用prelu_stub函数执行PReLU操作，根据设备类型执行相应的操作
  prelu_stub(iter.device_type(), iter);
  // 返回PReLU操作的结果张量result
  return result;
}

std::tuple<Tensor, Tensor> _prelu_kernel_backward(const Tensor& grad_out, const Tensor& self, const Tensor& weight) {
  // 创建一个与输入张量self形状和数据类型相同的空梯度张量grad_self
  Tensor grad_self = at::empty({0}, self.options());
  // 创建一个与输入张量weight形状和数据类型相同的空梯度张量grad_weight
  Tensor grad_weight = at::empty({0}, weight.options());
  // 构建张量迭代器配置，用于执行PReLU反向传播操作，计算self和weight的梯度
  auto iter = TensorIteratorConfig()
    .add_output(grad_self)  // 将grad_self作为输出
    .add_output(grad_weight)  // 将grad_weight作为输出
    .add_const_input(self)  // 将self作为常量输入
    .add_const_input(weight)  // 将weight作为常量输入
    .add_const_input(grad_out)  // 将grad_out作为常量输入
    .build();  // 构建迭代器
  // 调用prelu_backward_stub函数执行PReLU反向传播操作，根据设备类型执行相应的操作
  prelu_backward_stub(iter.device_type(), iter);
  // 返回self和weight的梯度，以元组形式返回
  return {grad_self, grad_weight};
}

Tensor infinitely_differentiable_gelu_backward(
    const Tensor& grad,
    const Tensor& self) {
  // 定义常量kAlpha，值为M_2_SQRTPI * M_SQRT1_2 * 0.5，用于GELU反向传播计算
  constexpr double kAlpha = M_2_SQRTPI * M_SQRT1_2 * 0.5;
  // 计算GELU函数的CDF部分，用于计算梯度
  Tensor cdf = (1.0 + (self * M_SQRT1_2).erf_()).mul_(0.5);
  // 计算GELU函数的PDF部分，用于计算梯度
  Tensor pdf = (-0.5 * self * self).exp_();
  // 返回GELU函数的反向传播梯度，计算公式为：grad * [CDF(self) + alpha * self * PDF(self)]
  return cdf.addcmul_(self, pdf, kAlpha).mul_(grad);
}

std::tuple<Tensor, Tensor> log_sigmoid_forward_cpu(const Tensor& input) {
  // 创建一个与输入张量input形状和数据类型相同的空张量result，用于存储log-sigmoid的输出
  auto result = at::empty_like(input, at::MemoryFormat::Contiguous);
  // 创建一个与输入张量input形状和数据类型相同的空张量buffer，用于临时存储计算过程中的中间结果
  auto buffer = at::empty_like(input, at::MemoryFormat::Contiguous);
  // 调用log_sigmoid_cpu_stub函数执行log-sigmoid的CPU实现，将结果存储在result和buffer中
  log_sigmoid_cpu_stub(kCPU, result, buffer, input.contiguous());
  // 返回log-sigmoid操作的结果result和buffer，以元组形式返回
  return std::make_tuple(result, buffer);
}

std::tuple<Tensor&, Tensor&> log_sigmoid_forward_out_cpu(const Tensor& input, Tensor& result, Tensor& buffer) {
  // 调整result的形状与输入张量input相同
  result.resize_as_(input);
  // 调整buffer的形状与输入张量input相同，并确保是内存连续的
  buffer.resize_as_(input, at::MemoryFormat::Contiguous);
  // 检查buffer是否是内存连续的，要求log_sigmoid的输出参数需要是内存连续的
  TORCH_CHECK(buffer.is_contiguous(), "Contiguous buffer required for log_sigmoid with out parameter");
  // 如果result不是内存连续的，则创建一个内存连续的临时张量result_tmp
  Tensor result_tmp = result.is_contiguous() ? result : at::empty_like(result, at::MemoryFormat::Contiguous);
  // 调用log_sigmoid_cpu_stub函数执行log-sigmoid的CPU实现，将结果存储在result_tmp和buffer中
  log_sigmoid_cpu_stub(kCPU, result_tmp, buffer, input.contiguous());
  // 如果result不是内存连续的，则将内存连续的结果复制回result
  if (!result.is_contiguous()) {
    result.copy_(result_tmp);
  }
  // 返回log-sigmoid操作的结果result和buffer，以元组形式返回
  return std::forward_as_tuple(result, buffer);
}

Tensor & log_sigmoid_out(const Tensor & self, Tensor & output) {
  // 创建一个与输入张量self形状和数据类型相同的空张量buffer
  Tensor buffer = at::empty({0}, self.options());
  // 调用log_sigmoid_forward_out函数执行log_sigmoid操作，将结果存储在output中，并返回output
  return std::get<0>(at::log_sigmoid_forward_out(output, buffer, self));
}

Tensor log_sigmoid(const Tensor & self) {
  // 调用log_sigmoid_forward函数执行log_sigmoid操作，返回结果
  return std::get<0>(at::log_sigmoid_forward(self));
}

Tensor log_sigmoid_backward_cuda(const Tensor& grad_output, const Tensor& input, const Tensor& buffer) {
  // 创建一个与grad_output形状和数据类型相同的空张量grad_input，用于存储log_sigmoid的反向传播梯度
  auto grad_input = at::empty_like(grad_output);
  // 构建张量迭代器配置，用于执行log_sigmoid的反向传播，计算grad_input
  auto iter = at::TensorIteratorConfig()
      .add_output(grad_input)  // 将grad_input作为输出
      .add_const_input(input)  // 将input作为常量输入
      .add_const_input(grad_output)  // 将grad_output作为常量输入
      .build();  // 构建迭代器
  // 调用log_sigmoid_backward_stub函数执行log_sigmoid的CUDA实现，根据设备类型执行相应的操作
  log_sigmoid_backward_stub(kCUDA, iter);
  // 返回log_sigmoid的反向传播梯度grad_input
  return iter.output();
}
Tensor log_sigmoid_backward_cpu(const Tensor& grad_output, const Tensor& input, const Tensor& buffer) {
  // 创建一个和 grad_output 相同形状的新张量 grad_input
  auto grad_input = at::empty_like(grad_output);
  // 配置张量迭代器，指定输出为 grad_input，输入为 input、buffer、grad_output
  auto iter = at::TensorIteratorConfig()
      .add_output(grad_input)
      .add_const_input(input)
      .add_const_input(buffer)
      .add_const_input(grad_output)
      .build();
  // 调用 log_sigmoid_backward_stub 函数处理 CPU 上的张量迭代器 iter
  log_sigmoid_backward_stub(kCPU, iter);
  // 返回处理后的 grad_input 张量
  return iter.output();
}

Tensor& log_sigmoid_backward_cuda_out(const Tensor& grad_output, const Tensor& input,
                                      const Tensor& buffer, Tensor& grad_input) {
  // 配置张量迭代器，指定输出为传入的 grad_input，输入为 input 和 grad_output
  auto iter = TensorIteratorConfig()
      .add_output(grad_input)
      .add_const_input(input)
      .add_const_input(grad_output)
      .build();
  // 调用 log_sigmoid_backward_stub 函数处理 CUDA 上的张量迭代器 iter
  log_sigmoid_backward_stub(kCUDA, iter);
  // 返回处理后的 grad_input 张量的引用
  return grad_input;
}

Tensor& log_sigmoid_backward_cpu_out(const Tensor& grad_output,
    const Tensor& input,
    const Tensor& buffer,
    Tensor& grad_input) {
  // 配置张量迭代器，指定输出为传入的 grad_input，输入为 input、buffer、grad_output
  auto iter = TensorIteratorConfig()
      .add_output(grad_input)
      .add_const_input(input)
      .add_const_input(buffer)
      .add_const_input(grad_output)
      .build();
  // 调用 log_sigmoid_backward_stub 函数处理 CPU 上的张量迭代器 iter
  log_sigmoid_backward_stub(kCPU, iter);
  // 返回处理后的 grad_input 张量的引用
  return grad_input;
}

DEFINE_DISPATCH(GeluKernel);
DEFINE_DISPATCH(GeluBackwardKernel);

}  // namespace at::native


注释说明了每个函数的作用及其内部的关键步骤，确保代码的每一部分都被清晰地解释了其功能。
```