# `.\pytorch\aten\src\ATen\native\mkldnn\SoftMax.cpp`

```py
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Config.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_softmax_native.h>         // for mkldnn_softmax
#endif

#if !AT_MKLDNN_ENABLED()

namespace at {
namespace native {

// 如果未启用 MKLDNN 支持，定义 mkldnn_softmax 函数，抛出错误信息
Tensor mkldnn_softmax(
    const Tensor& self,
    const int64_t dim,
    const bool half_to_float) {
  TORCH_CHECK(false, "mkldnn_softmax: ATen not compiled with MKLDNN support");
}

} // namespace native
} // namespace at

#else // AT_MKLDNN_ENABLED

#include <ATen/native/mkldnn/MKLDNNCommon.h>

namespace at {
namespace native {

// 如果启用了 MKLDNN 支持，定义 mkldnn_softmax 函数
Tensor mkldnn_softmax(
    const Tensor& self,
    const int64_t dim,
    const bool half_to_float) {
  // 检查是否要进行半精度到单精度转换，如果是则抛出错误信息
  TORCH_CHECK(
      !half_to_float,
      "softmax with half to float conversion is not supported on Mkldnn");
  // 将维度转换为正常范围内的值
  const int64_t wrapped_dim = maybe_wrap_dim(dim, self.dim());
  // 从 MKLDNN 张量转换为 ideep::tensor
  ideep::tensor& x = itensor_from_mkldnn(self);
  // 创建一个空的 ideep::tensor y，并执行 softmax 计算
  ideep::tensor y;
  ideep::softmax_forward::compute(x, y, wrapped_dim);
  // 根据转换后的 ideep::tensor 创建新的 ATen Tensor，并返回
  return new_with_itensor_mkldnn(std::move(y), optTypeMetaToScalarType(self.options().dtype_opt()),
                                 self.options().device_opt());
}

} // namespace native
} // namespace at

#endif // AT_MKLDNN_ENABLED
```