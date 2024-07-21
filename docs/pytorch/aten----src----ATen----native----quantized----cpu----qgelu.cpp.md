# `.\pytorch\aten\src\ATen\native\quantized\cpu\qgelu.cpp`

```py
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/native/quantized/cpu/QuantizedOps.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/gelu_native.h>
#endif

namespace at {
namespace native {

// 定义调度分发的函数指针，用于量化 GELU 操作的 CPU 实现
DEFINE_DISPATCH(qgelu_stub);

// 实现量化 GELU 的 CPU 版本
Tensor gelu_quantized_cpu(const Tensor& qx, c10::string_view approximate) {
  // 声明输出 Tensor 变量
  Tensor qy;
  // 调用分发函数指针，执行量化 GELU 操作，根据输入 Tensor 的设备类型选择执行路径
  qgelu_stub(qx.device().type(), qx, qy, get_gelutype_enum(approximate));
  // 返回计算后的输出 Tensor
  return qy;
}

// 实现量化 GELU 的原位版本，修改输入 Tensor 自身
Tensor& gelu_quantized_cpu_(Tensor& self, c10::string_view approximate) {
  // 调用非原位版本的量化 GELU 实现，获得输出 Tensor
  Tensor qy = gelu_quantized_cpu(self, approximate);
  // 将输出 Tensor 的值复制到输入 Tensor 中，以实现原位修改
  self.copy_(qy);
  // 返回修改后的输入 Tensor 自身
  return self;
}

}}  // namespace at::native
```