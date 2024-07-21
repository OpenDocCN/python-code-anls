# `.\pytorch\aten\src\ATen\native\AdaptiveAveragePooling.cpp`

```py
// 定义宏，仅启用 Torch 的断言方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 引入必要的头文件
#include <ATen/core/Tensor.h>
#include <ATen/native/AdaptivePooling.h>
#include <ATen/native/xnnpack/Engine.h>
#include <c10/util/irange.h>

// 根据条件引入不同的头文件，控制操作符的选择
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_adaptive_avg_pool2d.h>
#include <ATen/ops/_adaptive_avg_pool2d_backward_native.h>
#include <ATen/ops/_adaptive_avg_pool2d_native.h>
#include <ATen/ops/adaptive_avg_pool2d_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/mkldnn_adaptive_avg_pool2d.h>
#endif

// 命名空间 at::native 中的实现细节
namespace at::native {

// 匿名命名空间，用于实现内部函数
namespace {

// 定义 CPU 下的自适应平均池化操作
void adaptive_avg_pool2d_out_cpu_template(
  at::Tensor& output,
  at::Tensor const& input,
  IntArrayRef output_size)
{
  // 检查输出大小是否为二维
  TORCH_CHECK(output_size.size() == 2, "adaptive_avg_pool2d: output_size must be 2");

  // 检查输入张量的维度是否为3D或4D
  int64_t ndim = input.dim();
  TORCH_CHECK((ndim == 3 || ndim == 4),
    "adaptive_avg_pool2d(): Expected 3D or 4D tensor, but got ", input.sizes());

  // 检查非批处理维度是否大于0
  for (const auto i : {-2, -1}) {
    TORCH_CHECK(input.size(i) > 0,
      "adaptive_avg_pool2d(): Expected input to have non-zero size for non-batch dimensions, "
      "but input has sizes ", input.sizes(), " with dimension ", i + ndim, " being "
      "empty");
  }

  // 检查输出张量的数据类型与输入张量的数据类型是否一致
  TORCH_CHECK(input.dtype() == output.dtype(),
    "expected dtype ", input.dtype(), " for `output` but got dtype ", output.dtype());

  // 获取输入张量中的通道数和输出高度、宽度
  int64_t channels  = input.size(-3);
  int64_t output_height = output_size[0];
  int64_t output_width = output_size[1];

  // 根据输入张量的维度调整输出张量的大小
  if (ndim == 3) {
    output.resize_({channels, output_height, output_width});
  } else {
    int64_t nbatch = input.size(0);
    output.resize_({nbatch, channels, output_height, output_width}, input.suggest_memory_format());
  }

  // 若输出张量为空，则直接返回
  if (output.numel() == 0) {
    return;
  }

  // 调用 CPU 下的自适应平均池化内核函数
  adaptive_avg_pool2d_kernel(kCPU, output, input, output_size);
}

// 定义 CPU 下的自适应平均池化反向传播操作
Tensor& adaptive_avg_pool2d_backward_out_cpu_template(
  Tensor& grad_input,
  const Tensor& grad_output,
  const Tensor& input)
{
  // 检查梯度输出张量的维度是否正确
  int64_t ndim = grad_output.ndimension();
  for (const auto i : c10::irange(1, ndim)) {
    TORCH_CHECK(grad_output.size(i) > 0,
      "adaptive_avg_pool2d_backward(): Expected grad_output to have non-zero size for non-batch dimensions, "
      "but grad_output has sizes ", grad_output.sizes(), " with dimension ", i, " being "
      "empty");
  }

  // 检查输入张量的维度是否为3D或4D
  TORCH_CHECK((ndim == 3 || ndim == 4),
    "adaptive_avg_pool2d_backward(): Expected 3D or 4D tensor, but got ", input.sizes());

  // 检查输入和梯度输出张量的数据类型是否一致
  TORCH_CHECK(input.dtype() == grad_output.dtype(),
    "expected dtype ", input.dtype(), " for `grad_output` but got dtype ", grad_output.dtype());
  TORCH_CHECK(input.dtype() == grad_input.dtype(),
    "expected dtype ", input.dtype(), " for `grad_input` but got dtype ", grad_input.dtype());

  // 调整梯度输入张量的大小和内存格式，然后清零
  grad_input.resize_(input.sizes(), input.suggest_memory_format());
  grad_input.zero_();

  // 返回梯度输入张量的引用
  return grad_input;
}
    // 调用 adaptive_avg_pool2d_backward_kernel 函数来计算 CPU 上的反向传播梯度
    adaptive_avg_pool2d_backward_kernel(kCPU, grad_input, grad_output);
    // 返回计算得到的输入梯度 grad_input
    return grad_input;
} // namespace

// 在 CPU 上执行自适应平均池化操作，并将结果写入输出张量
Tensor& adaptive_avg_pool2d_out_cpu(const Tensor& input,
  IntArrayRef output_size,
  Tensor& output)
{
  // 调用模板函数实现自适应平均池化操作
  adaptive_avg_pool2d_out_cpu_template(
    output, input, output_size);
  return output;  // 返回处理后的输出张量
}

// 在 CPU 上执行自适应平均池化操作，并返回结果张量
Tensor adaptive_avg_pool2d_cpu(
  at::Tensor const& input,
  IntArrayRef output_size)
{
  // 创建一个空的输出张量，与输入张量的选项相匹配
  auto output = at::empty({0}, input.options());
  // 调用模板函数实现自适应平均池化操作
  adaptive_avg_pool2d_out_cpu_template(
    output, input, output_size);
  return output;  // 返回处理后的输出张量
}

// 在具有符号整数数组大小的输入上执行自适应平均池化操作
Tensor adaptive_avg_pool2d_symint(at::Tensor const& input, SymIntArrayRef output_size) {
  // 检查输出大小数组的长度必须为2
  TORCH_CHECK(output_size.size() == 2, "adaptive_avg_pool2d: output_size must be 2");
  // 检查输出大小数组的元素必须大于或等于0
  TORCH_CHECK(
      (output_size[0] >= 0 && output_size[1] >= 0),
      "adaptive_avg_pool2d: elements of output_size must be greater than or equal to 0 ",
      "but received {", output_size[0], ", ", output_size[1], "}");

  // 如果输入张量是 MKL-DNN 类型，则调用相应的 MKL-DNN 自适应平均池化函数
  if (input.is_mkldnn()) {
    return at::mkldnn_adaptive_avg_pool2d(input, C10_AS_INTARRAYREF_SLOW(output_size));
  }

  // 如果输入张量不是量化的，并且输出大小为1x1，并且不是 XPU 类型
  if (!input.is_quantized() && output_size[0] == 1 && output_size[1] == 1 && !input.is_xpu()) {
    // 在这种情况下，自适应池化实际上只是计算平均值，可以更有效地完成
    // 使用 mean 函数计算输入张量的平均值，保持维度为 -1 和 -2
    Tensor out = input.mean({-1, -2}, /* keepdim = */ true);
    // 如果输入张量建议的内存格式是 ChannelsLast
    if (input.suggest_memory_format() == at::MemoryFormat::ChannelsLast) {
      // 确保输入张量维度为4，因为维度为3时不支持 channels_last
      const auto n = input.sym_size(0);
      const auto c = input.sym_size(1);
      // 将输出张量设置为符合 channels_last 格式的步幅视图
      out.as_strided__symint({n, c, 1, 1}, {c, 1, c, c});
    }
    return out;  // 返回处理后的输出张量
  } else {
    // 否则调用 _adaptive_avg_pool2d_symint 函数来执行自适应平均池化操作
    return _adaptive_avg_pool2d_symint(input, output_size);
  }
}

// 在 CPU 上执行自适应平均池化操作的反向传播，并返回输入梯度张量
Tensor adaptive_avg_pool2d_backward_cpu(
  const Tensor& grad_output,
  const Tensor& input)
{
  // 创建一个空的输入梯度张量，与输入张量的选项相匹配
  auto grad_input = at::empty({0}, input.options());
  // 调用模板函数实现自适应平均池化操作的反向传播
  adaptive_avg_pool2d_backward_out_cpu_template(
    grad_input, grad_output, input);
  return grad_input;  // 返回计算得到的输入梯度张量
}

// 定义自适应平均池化操作的分发函数
DEFINE_DISPATCH(adaptive_avg_pool2d_kernel);
// 定义自适应平均池化操作反向传播的分发函数
DEFINE_DISPATCH(adaptive_avg_pool2d_backward_kernel);

} // namespace at::native
```