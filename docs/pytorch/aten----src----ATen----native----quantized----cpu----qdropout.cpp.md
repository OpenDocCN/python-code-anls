# `.\pytorch\aten\src\ATen\native\quantized\cpu\qdropout.cpp`

```py
// 包含 PyTorch ATen 库中的头文件
#include <ATen/ATen.h>
// 包含 PyTorch ATen 库中的原生函数头文件
#include <ATen/NativeFunctions.h>
// 包含 PyTorch 的库头文件
#include <torch/library.h>
// 包含 ATen 库中的量化器头文件
#include <ATen/quantized/Quantizer.h>
// 包含 ATen 库中量化操作的原生 CPU 实现头文件
#include <ATen/native/quantized/cpu/QuantizedOps.h>

// ATen 命名空间
namespace at {
// ATen 原生命名空间
namespace native {

// 定义一个分发函数指针 qdropout_stub
DEFINE_DISPATCH(qdropout_stub);

// 定义一个函数 quantized_dropout，用于量化的 dropout 操作
static Tensor quantized_dropout(
    const Tensor& qx, double output_scale, int64_t output_zero_point, const Scalar& p, bool training) {
  // 简单地返回输入的量化 Tensor qx，暂未实现 dropout 功能
  return qx;
}

// 定义一个 Torch 库实现，用于注册量化 CPU 上的 dropout 函数
TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
  // 将 quantized_dropout 函数实现注册到 quantized::dropout 上
  m.impl(TORCH_SELECTIVE_NAME("quantized::dropout"), quantized_dropout);
}

}}  // namespace at::native
```