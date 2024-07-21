# `.\pytorch\aten\src\ATen\native\mkldnn\ConvPrepack.h`

```
#pragma once
// 只有一次的预处理指令，确保头文件只被编译一次

#include <ATen/Tensor.h>
// 包含 ATen 库中的 Tensor 类定义

#include <ATen/native/mkldnn/Common.h>
// 包含 ATen 库中 MKLDNN 相关功能的通用定义

#include <ATen/native/mkldnn/OpContext.h>
// 包含 ATen 库中 MKLDNN 操作上下文的定义

#if AT_MKLDNN_ENABLED()
// 如果编译器支持 MKLDNN

namespace at {
namespace native {
namespace mkldnn {
namespace internal {
namespace convolution {

c10::intrusive_ptr<mkldnn::ConvOpContext> createConvPrePackOpContext(
    Tensor weight,
    std::optional<Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    int64_t groups,
    std::vector<int64_t> input_size,
    std::string attr);
// 创建 MKLDNN 卷积预打包操作的上下文对象，并返回指向其的智能指针

Tensor conv_run(
    const Tensor& input,
    const c10::intrusive_ptr<mkldnn::ConvOpContext>& op_context);
// 运行 MKLDNN 卷积操作，使用给定的输入张量和操作上下文

ContextConv create(
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    const IntArrayRef padding,
    const IntArrayRef stride,
    const IntArrayRef dilation,
    const int64_t groups,
    const IntArrayRef input_size,
    const ideep::attr_t& attr);
// 创建 MKLDNN 卷积操作的上下文对象，用于后续运行

Tensor run(ContextConv& context, const Tensor& input);
// 运行给定上下文的 MKLDNN 卷积操作，使用指定输入张量

void run(ContextConv& context, const Tensor& input, void* output);
// 运行给定上下文的 MKLDNN 卷积操作，输出结果存储在给定的指针中

} // namespace convolution
} // namespace internal
} // namespace mkldnn
} // namespace native
} // namespace at

#endif // AT_MKLDNN_ENABLED()
// 结束条件编译块，检查 MKLDNN 功能是否启用
```