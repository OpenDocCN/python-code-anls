# `.\pytorch\aten\src\ATen\native\mkldnn\OpContext.cpp`

```
#if AT_MKLDNN_ENABLED()
// 如果 MKLDNN 被启用，则包含以下内容

#include <ATen/native/mkldnn/ConvPrepack.h>
// 包含 MKLDNN 的卷积预打包头文件

namespace at {
namespace native {
namespace mkldnn {

c10::intrusive_ptr<ConvOpContext> MkldnnConvOpContext::create_context(
    at::Tensor&& weight,
    std::optional<at::Tensor>&& bias,
    std::vector<int64_t>&& padding,
    std::vector<int64_t>&& stride,
    std::vector<int64_t>&& dilation,
    int64_t groups,
    std::vector<int64_t>&& input_size,
    const ideep::attr_t& attr) {
  // 创建 MKLDNN 卷积操作上下文，返回指向该上下文的智能指针
  auto op_context = mkldnn::internal::convolution::create(
      weight, bias, padding, stride, dilation, groups, input_size, attr);

  // 使用移动语义创建 MkldnnConvOpContext 对象
  auto conv_op_context = c10::make_intrusive<MkldnnConvOpContext>(
      std::move(weight),
      std::move(bias),
      std::move(padding),
      std::move(stride),
      std::move(dilation),
      groups,
      std::move(input_size),
      std::move(op_context));

  return conv_op_context; // 返回 MKLDNN 卷积操作上下文对象
}

Tensor MkldnnConvOpContext::run(const Tensor& input) {
  // 运行 MKLDNN 内部的卷积操作，并返回结果张量
  return mkldnn::internal::convolution::run(op_context_, input);
}

void MkldnnConvOpContext::run(const Tensor& input, void* output) {
  // 运行 MKLDNN 内部的卷积操作，并将结果输出到给定的内存位置
  mkldnn::internal::convolution::run(op_context_, input, output);
}

} // namespace mkldnn
} // namespace native
} // namespace at

#endif // AT_MKLDNN_ENABLED()
// 结束条件编译指令，如果 MKLDNN 未启用，则结束命名空间和代码块
```