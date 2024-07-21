# `.\pytorch\aten\src\ATen\native\FusedAdam.cpp`

```py
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/FusedAdam.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_fused_adam.h>
#include <ATen/ops/_fused_adam_native.h>
#include <ATen/ops/_fused_adamw.h>
#include <ATen/ops/_fused_adamw_native.h>
#endif

// 进入ATen命名空间
namespace at {

// 进入native子命名空间
namespace native {

// 实现CPU版本的_fused_adam_kernel_函数
void _fused_adam_kernel_cpu_(
    at::TensorList params,                  // 参数张量列表
    at::TensorList grads,                   // 梯度张量列表
    at::TensorList exp_avgs,                // 指数平均张量列表
    at::TensorList exp_avg_sqs,             // 平方指数平均张量列表
    at::TensorList max_exp_avg_sqs,         // 最大平方指数平均张量列表
    at::TensorList state_steps,             // 状态步长张量列表
    const double lr,                        // 学习率
    const double beta1,                     // beta1参数
    const double beta2,                     // beta2参数
    const double weight_decay,              // 权重衰减参数
    const double eps,                       // 用于数值稳定性的小值
    const bool amsgrad,                     // 是否使用amsgrad
    const bool maximize,                    // 是否最大化优化目标
    const std::optional<at::Tensor>& grad_scale,  // 梯度缩放张量的可选参数
    const std::optional<at::Tensor>& found_inf) {  // 是否发现无穷的可选参数
  const float* grad_scale_ptr =            // 初始化grad_scale_ptr，指向梯度缩放张量数据
      grad_scale.has_value() ? grad_scale->data_ptr<float>() : nullptr;
  const float* found_inf_ptr =            // 初始化found_inf_ptr，指向发现无穷张量数据
      found_inf.has_value() ? found_inf->data_ptr<float>() : nullptr;
  if (found_inf_ptr && *found_inf_ptr == 1.0) {  // 如果发现无穷并且值为1.0，直接返回
      return;
  }
  size_t n_tensors = params.size();        // 获取参数张量列表的大小
  TORCH_CHECK(grads.size() == n_tensors);  // 检查梯度张量列表大小与参数张量列表大小是否一致
  TORCH_CHECK(exp_avgs.size() == n_tensors);  // 检查指数平均张量列表大小与参数张量列表大小是否一致
  TORCH_CHECK(exp_avg_sqs.size() == n_tensors);  // 检查平方指数平均张量列表大小与参数张量列表大小是否一致
  if (amsgrad) {                           // 如果使用amsgrad
    TORCH_CHECK(max_exp_avg_sqs.size() == n_tensors);  // 检查最大平方指数平均张量列表大小与参数张量列表大小是否一致
  } else {
    TORCH_CHECK(max_exp_avg_sqs.size() == 0);  // 否则，最大平方指数平均张量列表大小应为0
  }
  TORCH_CHECK(state_steps.size() == n_tensors);  // 检查状态步长张量列表大小与参数张量列表大小是否一致
  at::Tensor max_exp_avg_sq = at::Tensor();  // 初始化最大平方指数平均张量
  for (size_t i = 0; i < n_tensors; i++){   // 遍历参数张量列表
    if (amsgrad) max_exp_avg_sq = max_exp_avg_sqs[i];  // 如果使用amsgrad，获取当前最大平方指数平均张量
    fused_adam_stub(                       // 调用fused_adam_stub函数执行优化步骤
      kCPU,
      params[i],                           // 当前参数张量
      grads[i],                            // 当前梯度张量
      exp_avgs[i],                         // 当前指数平均张量
      exp_avg_sqs[i],                      // 当前平方指数平均张量
      max_exp_avg_sq,                      // 当前最大平方指数平均张量
      state_steps[i],                      // 当前状态步长张量
      lr,                                  // 学习率
      beta1,                               // beta1参数
      beta2,                               // beta2参数
      weight_decay,                        // 权重衰减参数
      eps,                                 // 数值稳定性参数
      amsgrad,                             // 是否使用amsgrad
      maximize,                            // 是否最大化优化目标
      grad_scale_ptr,                      // 梯度缩放指针
      ADAM_MODE::ORIGINAL);                // ADAM优化模式设为原始模式
  }
}

// 重载_fused_adam_kernel_cpu_函数，使用Tensor类型的学习率
void _fused_adam_kernel_cpu_(
    at::TensorList params,                  // 参数张量列表
    at::TensorList grads,                   // 梯度张量列表
    at::TensorList exp_avgs,                // 指数平均张量列表
    at::TensorList exp_avg_sqs,             // 平方指数平均张量列表
    at::TensorList max_exp_avg_sqs,         // 最大平方指数平均张量列表
    at::TensorList state_steps,             // 状态步长张量列表
    const at::Tensor& lr,                   // 学习率张量
    const double beta1,                     // beta1参数
    const double beta2,                     // beta2参数
    const double weight_decay,              // 权重衰减参数
    const double eps,                       // 数值稳定性参数
    const bool amsgrad,                     // 是否使用amsgrad
    const bool maximize,                    // 是否最大化优化目标
    const std::optional<at::Tensor>& grad_scale,  // 梯度缩放张量的可选参数
    const std::optional<at::Tensor>& found_inf) {  // 是否发现无穷的可选参数
  _fused_adam_kernel_cpu_(                 // 调用前面定义的_fused_adam_kernel_cpu_函数
    params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps,
    lr.item<double>(),                     // 提取学习率张量的double值
    beta1, beta2, weight_decay, eps, amsgrad, maximize, grad_scale, found_inf);
}

// 实现CPU版本的_fused_adamw_kernel_函数
void _fused_adamw_kernel_cpu_(
    at::TensorList params,                  // 参数张量列表
    at::TensorList grads,                   // 梯度张量列表
    at::TensorList exp_avgs,                // 指数平均张量列表
    at::TensorList exp_avg_sqs,             // 平方指数平均张量列表
    at::TensorList max_exp_avg_sqs,         // 最大平方指数平均张量列表
    at::TensorList state_steps,             // 状态步长张量列表
    const double lr,                        // 学习率
    const double beta1,                     // beta1参数


这些注释解释了每个函数、变量和条件的作用和含义，确保代码的功能和逻辑清晰可读。
    const double beta2,
    const double weight_decay,
    const double eps,
    const bool amsgrad,
    const bool maximize,
    const std::optional<at::Tensor>& grad_scale,
    const std::optional<at::Tensor>& found_inf) {
  // 检查是否传入了 grad_scale 和 found_inf，获取它们的指针
  const float* grad_scale_ptr =
      grad_scale.has_value() ? grad_scale->data_ptr<float>() : nullptr;
  const float* found_inf_ptr =
      found_inf.has_value() ? found_inf->data_ptr<float>() : nullptr;
  // 如果 found_inf_ptr 存在且其值为 1.0，则直接返回，不执行后续操作
  if (found_inf_ptr && *found_inf_ptr == 1.0) {
      return;
  }
  // 获取参数张量的数量
  size_t n_tensors = params.size();
  // 检查各个张量列表的长度是否与参数张量的数量相匹配
  TORCH_CHECK(grads.size() == n_tensors);
  TORCH_CHECK(exp_avgs.size() == n_tensors);
  TORCH_CHECK(exp_avg_sqs.size() == n_tensors);
  // 根据是否使用 AMSGrad 进行检查
  if (amsgrad) {
    TORCH_CHECK(max_exp_avg_sqs.size() == n_tensors);
  } else {
    TORCH_CHECK(max_exp_avg_sqs.size() == 0);
  }
  TORCH_CHECK(state_steps.size() == n_tensors);
  // 初始化 max_exp_avg_sq 为空张量
  at::Tensor max_exp_avg_sq = at::Tensor();
  // 遍历参数张量列表，对每个张量执行 fused_adam_stub 函数
  for (size_t i = 0; i < n_tensors; i++){
    // 如果使用 AMSGrad，则将 max_exp_avg_sq 设置为对应的 max_exp_avg_sqs[i]
    if (amsgrad) max_exp_avg_sq = max_exp_avg_sqs[i];
    // 调用 fused_adam_stub 函数，执行 fused Adam 更新
    fused_adam_stub(
      kCPU,
      params[i],
      grads[i],
      exp_avgs[i],
      exp_avg_sqs[i],
      max_exp_avg_sq,
      state_steps[i],
      lr,
      beta1,
      beta2,
      weight_decay,
      eps,
      amsgrad,
      maximize,
      grad_scale_ptr,
      ADAM_MODE::ADAMW);
  }
// 以下是 _fused_adamw_kernel_cpu_ 函数的实现，用于执行融合的 AdamW 算法的 CPU 版本

// 此函数接收多个参数，分别是模型参数、梯度、指数平均值、指数平均平方值、最大指数平均平方值、状态步数、学习率 lr 等
void _fused_adamw_kernel_cpu_(
    at::TensorList params,                       // 模型参数列表
    at::TensorList grads,                        // 梯度列表
    at::TensorList exp_avgs,                     // 指数平均值列表
    at::TensorList exp_avg_sqs,                  // 指数平均平方值列表
    at::TensorList max_exp_avg_sqs,              // 最大指数平均平方值列表
    at::TensorList state_steps,                  // 状态步数列表
    const at::Tensor& lr,                        // 学习率张量
    const double beta1,                          // AdamW 算法的 beta1 参数
    const double beta2,                          // AdamW 算法的 beta2 参数
    const double weight_decay,                   // 权重衰减参数
    const double eps,                            // epsilon 参数，用于数值稳定性
    const bool amsgrad,                          // 是否使用 AMSGrad 变种
    const bool maximize,                         // 是否最大化优化目标
    const std::optional<at::Tensor>& grad_scale, // 可选的梯度缩放张量
    const std::optional<at::Tensor>& found_inf   // 可选的发现无穷大梯度的张量
) {
    // 调用内部函数 _fused_adamw_kernel_cpu_，将学习率 lr 转换为 double 类型后传递给该函数
    _fused_adamw_kernel_cpu_(params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, lr.item<double>(), beta1, beta2, weight_decay, eps, amsgrad, maximize, grad_scale, found_inf);
}

// 定义了一个名为 fused_adam_stub 的派发函数
DEFINE_DISPATCH(fused_adam_stub);

// 结束 _fused_adamw_kernel_cpu_ 函数的实现
```