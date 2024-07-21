# `.\pytorch\aten\src\ATen\native\mkldnn\Prelu.cpp`

```py
#if !AT_MKLDNN_ENABLED()

namespace at { namespace native {

// 在没有启用 MKLDNN 支持时，定义 mkldnn_prelu 函数，抛出错误信息
Tensor mkldnn_prelu(const Tensor& input, const Tensor& weight) {
  TORCH_CHECK(false, "mkldnn_prelu: ATen not compiled with MKLDNN support");
}

// 在没有启用 MKLDNN 支持时，定义 mkldnn_prelu_backward 函数，抛出错误信息
std::tuple<Tensor, Tensor> mkldnn_prelu_backward(const Tensor& grad_output, const Tensor& input, const Tensor& weight) {
  TORCH_CHECK(false, "mkldnn_prelu_backward: ATen not compiled with MKLDNN support");
}

}}

#else // AT_MKLDNN_ENABLED

#include <ATen/native/mkldnn/MKLDNNCommon.h>
#include <ATen/native/mkldnn/Utils.h>

namespace at { namespace native {

// 当启用 MKLDNN 支持时，实现 mkldnn_prelu 函数
Tensor mkldnn_prelu(const Tensor& input, const Tensor& weight) {
  // 如果输入张量的数据类型为 BFloat16，则检查是否支持 mkldnn_bf16 设备特性
  if (input.scalar_type() == ScalarType::BFloat16) {
    TORCH_CHECK(mkldnn_bf16_device_check(),
        "mkldnn_relu: bf16 path needs the cpu support avx512bw, avx512vl and avx512dq");
  }

  // 将输入张量转换为 MKLDNN 格式的 tensor
  const ideep::tensor& x = itensor_from_mkldnn(input);
  // 将权重张量转换为 MKLDNN 格式的 tensor
  const ideep::tensor& w = itensor_from_tensor(weight);

  ideep::tensor y;
  // 执行 MKLDNN 的 PReLU 前向计算
  ideep::prelu_forward::compute(
      x, w, y, ideep::prop_kind::forward_training);
  
  // 将计算结果转换为 Torch Tensor 格式并返回
  return new_with_itensor_mkldnn(std::move(y), optTypeMetaToScalarType(input.options().dtype_opt()),
                                 input.options().device_opt());
}

// 当启用 MKLDNN 支持时，实现 mkldnn_prelu_backward 函数
std::tuple<Tensor, Tensor> mkldnn_prelu_backward(const Tensor& grad_output, const Tensor& input, const Tensor& weight) {
  // 将输入张量转换为 MKLDNN 格式的 tensor
  const ideep::tensor& x = itensor_from_mkldnn(input);
  // 将权重张量转换为 MKLDNN 格式的 tensor
  const ideep::tensor& w = itensor_from_tensor(weight);
  // 将梯度输出张量转换为 MKLDNN 格式的 tensor
  const ideep::tensor grady = itensor_from_mkldnn(grad_output);
  
  ideep::tensor gradx;
  ideep::tensor gradw;

  // 执行 MKLDNN 的 PReLU 反向计算
  ideep::prelu_backward::compute(
      x, w, grady, gradx, gradw, ideep::prop_kind::backward);

  // 根据权重张量是否为 MKLDNN 格式，选择相应的返回格式
  if (weight.is_mkldnn()) {
    return std::make_tuple(
        new_with_itensor_mkldnn(std::move(gradx),
                                optTypeMetaToScalarType(grad_output.options().dtype_opt()),
                                grad_output.options().device_opt()),
        new_with_itensor_mkldnn(std::move(gradw),
                                optTypeMetaToScalarType(weight.options().dtype_opt()),
                                weight.options().device_opt()));
  } else {
    return std::make_tuple(
        new_with_itensor_mkldnn(std::move(gradx),
                                optTypeMetaToScalarType(grad_output.options().dtype_opt()),
                                grad_output.options().device_opt()),
        mkldnn_to_dense(new_with_itensor_mkldnn(std::move(gradw),
                                                optTypeMetaToScalarType(weight.options().dtype_opt()),
                                                weight.options().device_opt())));
  }
}
}}

#endif // AT_MKLDNN_ENABLED
```