# `.\pytorch\aten\src\ATen\native\xnnpack\Convolution.h`

```py
#pragma once

#ifdef USE_XNNPACK

#include <ATen/Tensor.h>
#include <ATen/native/xnnpack/Common.h>
#include <ATen/native/xnnpack/OpContext.h>

namespace at::native::xnnpack {
namespace internal::convolution2d {

// 创建 Conv2dOpContext 对象，用于卷积操作的预打包和约束处理
c10::intrusive_ptr<xnnpack::Conv2dOpContext>
    createConv2dClampPrePackOpContext(
        Tensor weight,
        std::optional<Tensor> bias,
        std::vector<int64_t> stride,
        std::vector<int64_t> padding,
        std::vector<int64_t> dilation,
        int64_t groups,
        const std::optional<Scalar>& output_min,
        const std::optional<Scalar>& output_max);

// 创建 TransposeConv2dOpContext 对象，用于转置卷积操作的预打包和约束处理
c10::intrusive_ptr<xnnpack::TransposeConv2dOpContext>
    createConv2dTransposeClampPrePackOpContext(
        Tensor weight,
        std::optional<Tensor> bias,
        std::vector<int64_t> stride,
        std::vector<int64_t> padding,
        std::vector<int64_t> output_padding,
        std::vector<int64_t> dilation,
        int64_t groups,
        const std::optional<Scalar>& output_min,
        const std::optional<Scalar>& output_max);

// 执行卷积操作，输入数据 input 和预打包的卷积上下文 op_context
Tensor conv2d_clamp_run(
    const Tensor& input,
    const c10::intrusive_ptr<xnnpack::Conv2dOpContext>& op_context);

// 解包预打包的卷积尺寸信息
IValue
unpack_prepacked_sizes_conv2d(const IValue& ivalue);

// 执行转置卷积操作，输入数据 input 和预打包的转置卷积上下文 op_context
Tensor conv2d_transpose_clamp_run(
    const Tensor& input,
    const c10::intrusive_ptr<xnnpack::TransposeConv2dOpContext>& op_context);

// 创建 ContextConv2D 对象，用于存储卷积运行时的上下文信息
ContextConv2D create(
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    const IntArrayRef padding,
    const IntArrayRef output_padding,
    const IntArrayRef stride,
    const IntArrayRef dilation,
    const int64_t groups,
    const bool transposed,
    const float output_min,
    const float output_max);

// 运行卷积操作，使用给定的 ContextConv2D 对象 context 和输入数据 input
Tensor run(ContextConv2D& context, const Tensor& input);

} // namespace internal::convolution2d

// 执行卷积操作，使用给定的输入数据 input、权重 weight、偏置 bias、填充 padding、步幅 stride、膨胀 dilation 和分组 groups
Tensor convolution2d(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const IntArrayRef padding,
    const IntArrayRef stride,
    const IntArrayRef dilation,
    const int64_t groups);
} // namespace at::native::xnnpack

#endif /* USE_XNNPACK */


注释中解释了每个函数的作用和参数意义，确保代码的每一行都被准确地注释说明了其功能和上下文。
```