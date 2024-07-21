# `.\pytorch\aten\src\ATen\native\quantized\cpu\qelu.cpp`

```
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/core/ivalue.h>
#include <torch/library.h>
#include <ATen/native/quantized/cpu/QuantizedOps.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/_empty_affine_quantized.h>
#endif

namespace at {
namespace native {

// 定义分发函数 qelu_stub，用于处理量化 ELU 操作
DEFINE_DISPATCH(qelu_stub);

// quantized_elu 函数实现，对输入进行量化 ELU 运算
static Tensor quantized_elu(
    const Tensor& qx, double output_scale, int64_t output_zero_point, const Scalar& alpha, const Scalar& scale, const Scalar& input_scale) {
  
  // 创建一个空的量化张量 qy，与输入 qx 的尺寸和选项相同
  Tensor qy = at::_empty_affine_quantized(qx.sizes(), qx.options(), output_scale, output_zero_point);
  
  // 调用 qelu_stub 进行量化 ELU 运算，结果存储在 qy 中
  qelu_stub(qx.device().type(), qx, alpha, scale, input_scale, qy);
  
  // 返回量化 ELU 运算的结果张量 qy
  return qy;
}

// quantized_celu 函数实现，对输入进行量化 CELU 运算
static Tensor quantized_celu(const Tensor& qx, double output_scale, int64_t output_zero_point, const Scalar& alpha) {
  
  // 检查 alpha 是否为零，如果是，则抛出异常
  TORCH_CHECK(alpha.to<double>() != 0,
      "ZeroDivisionError: alpha cannot be 0 for CELU");
  
  // 计算 alpha 的倒数
  double inv_alpha = 1. / alpha.to<double>();
  
  // 调用 quantized_elu 函数进行量化 ELU 运算，使用 inv_alpha 作为 scale 参数
  return quantized_elu(qx, output_scale, output_zero_point, alpha, Scalar(1.0), Scalar(inv_alpha));
}

// 在 quantized 命名空间下注册量化操作的实现
TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
  
  // 将 quantized_elu 函数注册为 "quantized::elu" 的实现
  m.impl(TORCH_SELECTIVE_NAME("quantized::elu"), quantized_elu);
  
  // 将 quantized_celu 函数注册为 "quantized::celu" 的实现
  m.impl(TORCH_SELECTIVE_NAME("quantized::celu"), quantized_celu);
}

}}  // namespace at::native
```