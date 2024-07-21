# `.\pytorch\aten\src\ATen\native\FusedAdagrad.cpp`

```py
// 定义宏，仅包含 Torch 断言方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 包含 ATen 库中所需的头文件
#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/FusedAdagrad.h>

// 如果未定义 AT_PER_OPERATOR_HEADERS，则包含常规的 ATen 函数头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
// 否则，包含特定于操作符的头文件
#else
#include <ATen/ops/_fused_adagrad.h>
#include <ATen/ops/_fused_adagrad_native.h>
#endif

// 命名空间定义：at 命名空间下的 native 子命名空间
namespace at {
namespace native {

// 定义 CPU 下的 _fused_adagrad_kernel_ 函数
void _fused_adagrad_kernel_cpu_(
    at::TensorList params,                   // 参数张量列表
    at::TensorList grads,                    // 梯度张量列表
    at::TensorList state_sums,               // 状态和张量列表
    at::TensorList state_steps,              // 状态步长张量列表
    const double lr,                         // 学习率
    const double lr_decay,                   // 学习率衰减
    const double weight_decay,               // 权重衰减
    const double eps,                        // 用于数值稳定性的小常数
    const bool maximize,                     // 是否最大化优化
    const std::optional<at::Tensor>& grad_scale,  // 梯度缩放张量的可选项
    const std::optional<at::Tensor>& found_inf) {  // 发现无穷值的可选项
  // 获取梯度缩放张量的指针
  const float* grad_scale_ptr =
      grad_scale.has_value() ? grad_scale->data_ptr<float>() : nullptr;
  // 获取发现无穷值张量的指针
  const float* found_inf_ptr =
      found_inf.has_value() ? found_inf->data_ptr<float>() : nullptr;
  
  // 如果发现无穷值张量存在且其值为 1.0，则直接返回
  if (found_inf_ptr && *found_inf_ptr == 1.0) {
      return;
  }
  
  // 获取参数张量列表的大小
  size_t n_tensors = params.size();
  
  // 使用 Torch 的断言检查张量列表的大小是否一致
  TORCH_CHECK(grads.size() == n_tensors);
  TORCH_CHECK(state_sums.size() == n_tensors);
  TORCH_CHECK(state_steps.size() == n_tensors);
  
  // 遍历每个张量，并调用 fused_adagrad_stub 函数进行优化更新
  for (size_t i = 0; i < n_tensors; i++) {
    fused_adagrad_stub(
      kCPU,
      params[i],
      grads[i],
      state_sums[i],
      state_steps[i],
      lr,
      lr_decay,
      weight_decay,
      eps,
      maximize,
      grad_scale_ptr);
  }
}

// 定义分发器，用于派发 fused_adagrad_stub 函数
DEFINE_DISPATCH(fused_adagrad_stub);

} // namespace native
} // namespace at
```