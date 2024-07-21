# `.\pytorch\aten\src\ATen\native\mkldnn\Gelu.cpp`

```
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 定义预处理指令，启用只包含方法操作符的模式

#include <ATen/core/Tensor.h>
#include <ATen/Config.h>
#include <ATen/native/Activation.h>
// 包含头文件，声明 ATen 核心张量和配置相关的内容，以及激活函数的原生实现

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/gelu_native.h>
#include <ATen/ops/gelu_backward_native.h>
#endif
// 条件编译：根据 AT_PER_OPERATOR_HEADERS 的定义决定是否包含通用的原生函数头文件或者具体的 gelu 相关原生操作的头文件

#if !AT_MKLDNN_ENABLED()
// 如果未启用 MKLDNN 支持，则进行以下内容

namespace at { namespace native {

Tensor mkldnn_gelu(const Tensor& input, c10::string_view approximate) {
  TORCH_CHECK(false, "mkldnn_gelu: ATen not compiled with MKLDNN support");
  // 检查并报错：ATen 未编译支持 MKLDNN
}

Tensor mkldnn_gelu_backward(const Tensor& grad_output, const Tensor& input, c10::string_view approximate) {
  TORCH_CHECK(false, "mkldnn_gelu_backward: ATen not compiled with MKLDNN support");
  // 检查并报错：ATen 未编译支持 MKLDNN
}

}}

#else // AT_MKLDNN_ENABLED
// 如果启用了 MKLDNN 支持，则进行以下内容

#include <ATen/native/mkldnn/MKLDNNCommon.h>
#include <ATen/native/mkldnn/Utils.h>
// 包含 MKLDNN 相关的头文件，通用功能和工具函数

namespace at { namespace native {

Tensor mkldnn_gelu(const Tensor& input, c10::string_view approximate) {
  if (input.scalar_type() == ScalarType::BFloat16) {
    TORCH_CHECK(mkldnn_bf16_device_check(),
        "mkldnn_gelu: bf16 path needs the cpu support avx512bw, avx512vl and avx512dq");
    // 如果输入是 BFloat16 类型，检查是否支持 MKLDNN 的 bf16 路径
  }
  TORCH_CHECK(get_gelutype_enum(approximate) == GeluType::None,
                  "mkldnn_gelu: fast, approximate gelu is not supported");
  // 检查所选的 Gelu 类型是否为 None，即不支持快速或近似的 Gelu

  const ideep::tensor& x = itensor_from_tensor(input);
  // 将 ATen 的张量转换为 MKLDNN 的 ideep::tensor
  ideep::tensor y;
  ideep::eltwise_forward::compute(
      x, y, ideep::algorithm::eltwise_gelu_erf, ideep::prop_kind::forward_training, /*alpha*/ 0.0);
  // 执行 MKLDNN 中的 eltwise_gelu_erf 算法的前向计算，存储结果在 y 中
  return new_with_itensor_mkldnn(std::move(y), optTypeMetaToScalarType(input.options().dtype_opt()),
                                 input.options().device_opt());
  // 将 MKLDNN 的输出 y 转换回 ATen 的 Tensor，以便保留数据类型和设备信息
}

Tensor mkldnn_gelu_backward(const Tensor& grad_output, const Tensor& input, c10::string_view approximate) {
  TORCH_CHECK(get_gelutype_enum(approximate) == GeluType::None,
                  "mkldnn_gelu_backward: fast, approximate gelu is not supported");
  // 检查所选的 Gelu 类型是否为 None，即不支持快速或近似的 Gelu

  const ideep::tensor& x = itensor_from_tensor(input);
  // 将 ATen 的输入张量转换为 MKLDNN 的 ideep::tensor
  ideep::tensor grady = itensor_from_tensor(grad_output);
  // 将 ATen 的梯度输出张量转换为 MKLDNN 的 ideep::tensor
  ideep::tensor gradx;
  ideep::eltwise_backward::compute(x, grady, gradx,
      ideep::algorithm::eltwise_gelu_erf, /*alpha*/ 0.0);
  // 执行 MKLDNN 中的 eltwise_gelu_erf 算法的反向计算，存储结果在 gradx 中
  return new_with_itensor_mkldnn(std::move(gradx),
                                 optTypeMetaToScalarType(grad_output.options().dtype_opt()),
                                 grad_output.options().device_opt());
  // 将 MKLDNN 的输出 gradx 转换回 ATen 的 Tensor，以便保留数据类型和设备信息
}

}}

#endif // AT_MKLDNN_ENABLED
// 结束条件编译块，AT_MKLDNN_ENABLED 结束
```