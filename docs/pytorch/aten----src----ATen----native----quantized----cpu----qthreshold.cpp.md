# `.\pytorch\aten\src\ATen\native\quantized\cpu\qthreshold.cpp`

```py
// 定义宏，用于在 Torch 库中仅包含方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 包含 ATen 核心 Tensor 类定义头文件
#include <ATen/core/Tensor.h>
// 包含 Torch 库的主头文件
#include <torch/library.h>
// 包含量化操作的 CPU 实现头文件
#include <ATen/native/quantized/cpu/QuantizedOps.h>

// 如果未定义每个操作符的头文件，则包含通用操作函数头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
// 否则，包含指定的空操作和阈值函数头文件
#else
#include <ATen/ops/_empty_affine_quantized.h>
#include <ATen/ops/threshold_native.h>
#endif

// 包含标准算法库
#include <algorithm>

// 在 at 命名空间下的 native 命名空间中定义内容
namespace at {
namespace native {

// 定义分派函数，用于分发 qthreshold_stub
DEFINE_DISPATCH(qthreshold_stub);

// 量化阈值核心实现函数
// 参数包括输入量化 Tensor qx、阈值 threshold 和值 value
static Tensor quantized_threshold_impl(
    const Tensor& qx,
    const Scalar& threshold,
    const Scalar& value) {
  // 创建一个和 qx 相同大小和选项的空量化 Tensor qy
  Tensor qy = at::_empty_affine_quantized(
    qx.sizes(), qx.options(), qx.q_scale(), qx.q_zero_point());
  // 调用分派函数 qthreshold_stub 执行量化阈值操作
  qthreshold_stub(qx.device().type(), qx, threshold, value, qy);
  // 返回处理后的量化 Tensor qy
  return qy;
}

// CPU 上的量化阈值函数
// 参数包括输入量化 Tensor qx、阈值 threshold 和值 value
Tensor threshold_quantized_cpu(
    const Tensor& qx,
    const Scalar& threshold,
    const Scalar& value) {
  Tensor qy;
  // 根据 qx 的数据类型调度对应的量化阈值实现
  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "threshold", [&]() {
    // 调用量化阈值实现函数 quantized_threshold_impl
    qy = quantized_threshold_impl(qx, threshold, value);
  });
  // 返回处理后的量化 Tensor qy
  return qy;
}

// 定义 Torch 库的实现，用于量化 CPU 中的阈值操作
TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
  // 注册 quantized::threshold 的实现函数为 threshold_quantized_cpu
  m.impl(TORCH_SELECTIVE_NAME("quantized::threshold"), TORCH_FN(threshold_quantized_cpu));
}

} // namespace native
} // namespace at
```