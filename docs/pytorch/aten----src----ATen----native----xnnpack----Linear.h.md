# `.\pytorch\aten\src\ATen\native\xnnpack\Linear.h`

```
#pragma once
// 如果定义了 USE_XNNPACK 宏，则编译以下代码块

#ifdef USE_XNNPACK
// 包含 ATen 库中相关的头文件，用于张量操作
#include <ATen/Tensor.h>
// 包含 ATen 库中 XNNPACK 模块的通用功能
#include <ATen/native/xnnpack/Common.h>
// 包含 ATen 库中 XNNPACK 模块的运算上下文
#include <ATen/native/xnnpack/OpContext.h>

// 进入 ATen 库的 native::xnnpack 命名空间下的 internal::linear 命名空间
namespace at::native::xnnpack {
namespace internal::linear {

// 创建线性层预打包操作上下文的函数声明
c10::intrusive_ptr<xnnpack::LinearOpContext> createLinearClampPrePackOpContext(
    Tensor weight,
    std::optional<Tensor> bias,
    const std::optional<Scalar>& output_min,
    const std::optional<Scalar>& output_max);

// 运行带 Clamp 功能的线性层操作的函数声明
Tensor linear_clamp_run(const Tensor& input, const c10::intrusive_ptr<xnnpack::LinearOpContext>& op_context);

// 解包预打包尺寸信息的函数声明
IValue unpack_prepacked_sizes_linear(const IValue& ivalue);

// 创建线性层上下文的函数声明，用于初始化权重、偏置及输出范围
ContextLinear create(
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    const float output_min,
    const float output_max);

// 运行线性层操作的函数声明，接受上下文和输入张量作为参数
Tensor run(const ContextLinear& context, const Tensor& input);

} // namespace internal::linear

// 判断是否使用线性层的函数声明，根据输入张量、权重和偏置
bool use_linear(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias);

// 执行线性层操作的函数声明，接受输入张量、权重和偏置作为参数
Tensor linear(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias);

} // namespace at::native::xnnpack

#endif /* USE_XNNPACK */
// 结束条件：如果未定义 USE_XNNPACK 宏，则结束当前代码块
```