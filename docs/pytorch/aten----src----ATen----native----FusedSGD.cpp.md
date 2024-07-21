# `.\pytorch\aten\src\ATen\native\FusedSGD.cpp`

```
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_fused_sgd.h>
#include <ATen/ops/_fused_sgd_native.h>
#endif



// 如果未定义 AT_PER_OPERATOR_HEADERS 宏，则包含标准的 ATen 函数和本地函数头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
// 否则，包含用于融合SGD的特定操作头文件
#else
#include <ATen/ops/_fused_sgd.h>
#include <ATen/ops/_fused_sgd_native.h>
#endif



namespace at {

namespace native {

// 定义了一个内部的 CPU 实现函数 _fused_sgd_kernel_cpu_
void _fused_sgd_kernel_cpu_(
    at::TensorList params,  // 参数张量列表
    at::TensorList grads,   // 梯度张量列表
    at::TensorList momentum_buffer_list,  // 动量缓存张量列表
    const double weight_decay,  // 权重衰减
    const double momentum,      // 动量
    const double lr,            // 学习率
    const double dampening,     // 阻尼
    const bool nesterov,        // 是否使用 Nesterov 动量
    const bool maximize,        // 是否最大化
    const bool is_first_step,   // 是否第一步
    const std::optional<at::Tensor>& grad_scale,  // 梯度缩放张量的可选引用
    const std::optional<at::Tensor>& found_inf) {  // 是否发现无穷的可选引用
  const float* grad_scale_ptr =
      grad_scale.has_value() ? grad_scale->data_ptr<float>() : nullptr;  // 获取梯度缩放张量的数据指针
  const float* found_inf_ptr =
      found_inf.has_value() ? found_inf->data_ptr<float>() : nullptr;  // 获取发现无穷张量的数据指针
  if (found_inf_ptr && *found_inf_ptr == 1.0) {  // 如果发现无穷张量存在且值为1.0，则返回
      return;
  }
  size_t n_tensors = params.size();  // 获取参数张量列表的大小
  TORCH_CHECK(grads.size() == n_tensors);  // 断言梯度张量列表的大小与参数张量列表相同
  bool no_momentum_buffer = momentum == 0.0;  // 检查是否没有动量缓存
  if (no_momentum_buffer) {
    TORCH_CHECK(momentum_buffer_list.size() == 0);  // 断言动量缓存列表为空
  } else {
    TORCH_CHECK(momentum_buffer_list.size() == n_tensors);  // 否则，断言动量缓存列表的大小与参数张量列表相同
  }
  for (size_t i = 0; i < n_tensors; i++){  // 遍历参数张量列表
    fused_sgd_stub(
      kCPU,  // 使用 CPU 设备
      params[i],  // 当前参数张量
      grads[i],   // 当前梯度张量
      no_momentum_buffer ? Tensor() : momentum_buffer_list[i],  // 动量缓存张量，如果没有动量则为空张量
      weight_decay,  // 权重衰减
      momentum,      // 动量
      lr,            // 学习率
      dampening,     // 阻尼
      nesterov,      // 是否使用 Nesterov 动量
      maximize,      // 是否最大化
      is_first_step, // 是否第一步
      grad_scale_ptr);  // 梯度缩放指针
  }
}

// 重载 _fused_sgd_kernel_cpu_，支持使用张量 lr 作为学习率
void _fused_sgd_kernel_cpu_(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList momentum_buffer_list,
    const double weight_decay,
    const double momentum,
    const at::Tensor& lr,          // 学习率张量
    const double dampening,
    const bool nesterov,
    const bool maximize,
    const bool is_first_step,
    const std::optional<at::Tensor>& grad_scale,
    const std::optional<at::Tensor>& found_inf) {
    // 调用上面定义的 _fused_sgd_kernel_cpu_，将 lr 张量转换为双精度浮点数
    _fused_sgd_kernel_cpu_(
        params, grads, momentum_buffer_list, weight_decay,
        momentum, lr.item<double>(), dampening, nesterov,
        maximize, is_first_step, grad_scale, found_inf
    );
}

// 定义 fused_sgd_stub 的调度分发函数
DEFINE_DISPATCH(fused_sgd_stub);

} // namespace native
} // namespace at
```