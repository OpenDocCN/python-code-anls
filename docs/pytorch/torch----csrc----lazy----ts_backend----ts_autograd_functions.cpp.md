# `.\pytorch\torch\csrc\lazy\ts_backend\ts_autograd_functions.cpp`

```
//`
#include <ATen/Operators.h> // 包含 ATen 的运算符头文件
#include <ATen/native/CPUFallback.h> // 包含 CPU 备用功能的头文件
#include <torch/csrc/lazy/ts_backend/ts_autograd_functions.h> // 包含 TensorScript 后端的自动求导功能头文件
#include <torch/csrc/lazy/ts_backend/ts_eager_fallback.h> // 包含 TensorScript 后端的 eager 备用功能头文件

namespace torch {
namespace lazy {

// 实现 MaxPool3d 自动求导的前向计算函数
at::Tensor MaxPool3dAutogradFunctionTS::forward(
    torch::autograd::AutogradContext* ctx, // 自动求导上下文指针
    at::Tensor self, // 输入张量
    at::IntArrayRef kernel_size, // 卷积核大小
    at::IntArrayRef stride, // 步长
    at::IntArrayRef padding, // 填充
    at::IntArrayRef dilation, // 膨胀率
    bool ceil_mode) { // 是否使用 ceil 模式
  ctx->saved_data["kernel_size"] = kernel_size; // 保存 kernel_size 参数
  ctx->saved_data["stride"] = stride; // 保存 stride 参数
  ctx->saved_data["padding"] = padding; // 保存 padding 参数
  ctx->saved_data["dilation"] = dilation; // 保存 dilation 参数
  ctx->saved_data["ceil_mode"] = ceil_mode; // 保存 ceil_mode 参数
  auto results = at::native::
      call_fallback_fn<&ltc_eager_fallback, ATEN_OP(max_pool3d_with_indices)>::
          call(self, kernel_size, stride, padding, dilation, ceil_mode); // 调用备用函数进行前向计算
  ctx->save_for_backward({self, std::get<1>(results)}); // 保存计算所需的变量，以便反向传播
  return std::get<0>(results); // 返回前向计算的结果
}

// 实现 MaxPool3d 自动求导的反向计算函数
torch::autograd::variable_list MaxPool3dAutogradFunctionTS::backward(
    torch::autograd::AutogradContext* ctx, // 自动求导上下文指针
    torch::autograd::variable_list grad_output) { // 输入梯度
  auto kernel_size = ctx->saved_data["kernel_size"].toIntList().vec(); // 从上下文中获取 kernel_size 参数
  auto stride = ctx->saved_data["stride"].toIntList().vec(); // 从上下文中获取 stride 参数
  auto padding = ctx->saved_data["padding"].toIntList().vec(); // 从上下文中获取 padding 参数
  auto dilation = ctx->saved_data["dilation"].toIntList().vec(); // 从上下文中获取 dilation 参数
  auto ceil_mode = ctx->saved_data["ceil_mode"].toBool(); // 从上下文中获取 ceil_mode 参数
  auto saved = ctx->get_saved_variables(); // 获取保存的变量
  auto self = saved[0]; // 输入张量
  at::Tensor grad; // 初始化梯度张量
  auto indices = saved[1]; // 获取保存的索引张量
  grad = at::native::call_fallback_fn<
      &ltc_eager_fallback,
      ATEN_OP(max_pool3d_with_indices_backward)>::
      call(
          grad_output[0], // 输入梯度
          self, // 输入张量
          kernel_size, // 卷积核大小
          stride, // 步长
          padding, // 填充
          dilation, // 膨胀率
          ceil_mode, // 是否使用 ceil 模式
          indices); // 索引张量

  at::Tensor undef; // 初始化未定义张量
  torch::autograd::variable_list grad_inputs = {
      grad, undef, undef, undef, undef, undef}; // 构造输入梯度列表
  return grad_inputs; // 返回输入梯度列表
}

} // namespace lazy
} // namespace torch
```