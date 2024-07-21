# `.\pytorch\aten\src\ATen\native\UnfoldBackward.cpp`

```
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/UnfoldBackward.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/unfold_backward_native.h>
#include <ATen/ops/zeros.h>
#endif

namespace at::native {

// 定义一个分发函数，用于处理 unfold_backward 操作的不同设备类型
DEFINE_DISPATCH(unfold_backward_stub);

// 实现 unfold_backward 函数，用于计算输入张量的反向展开梯度
Tensor unfold_backward(
  const Tensor& grad,              // 输入张量的梯度
  IntArrayRef input_sizes,         // 输入张量的大小
  int64_t dim,                     // 展开操作的维度
  int64_t size,                    // 展开的尺寸
  int64_t step                     // 展开的步长
) {
  // 创建一个与输入张量相同大小的全零张量，用于存储计算结果
  auto grad_input = at::zeros(input_sizes, grad.options());

  // 如果步长大于等于尺寸，则直接对输入张量进行展开操作并复制梯度
  if (step >= size) {
    auto gI_unfolded = grad_input.unfold(dim, size, step);  // 在指定维度上进行展开操作
    gI_unfolded.copy_(grad);  // 将梯度复制到展开后的张量上
    return grad_input;  // 返回展开后的梯度张量
  }

  // 否则调用预定义的分发函数处理展开操作
  unfold_backward_stub(
    grad.device().type(),  // 获取梯度张量所在设备的类型
    grad_input,            // 输出张量，用于存储展开后的梯度
    grad,                  // 输入张量的梯度
    dim, size, step        // 展开操作的维度、尺寸和步长
  );

  return grad_input;  // 返回展开后的梯度张量
}

} // namespace at::native
```