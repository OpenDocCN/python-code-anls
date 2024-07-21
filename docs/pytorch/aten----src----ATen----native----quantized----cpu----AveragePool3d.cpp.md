# `.\pytorch\aten\src\ATen\native\quantized\cpu\AveragePool3d.cpp`

```
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/native/Pool.h>
#include <ATen/native/quantized/cpu/init_qnnpack.h>
#include <ATen/native/quantized/cpu/QnnpackUtils.h>
#include <ATen/native/quantized/cpu/QuantizedOps.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_empty_affine_quantized.h>
#include <ATen/ops/avg_pool3d_native.h>
#endif

#include <vector>

namespace at {
namespace native {

// 定义了一个名为qavg_pool3d_nhwc_stub的分发函数
DEFINE_DISPATCH(qavg_pool3d_nhwc_stub);

namespace {

// 获取池化核大小的函数
inline std::tuple<int, int, int> get_kernel(IntArrayRef kernel_size) {
  // 检查核大小必须是一个整数或三个整数组成的元组
  TORCH_CHECK(
      kernel_size.size() == 1 || kernel_size.size() == 3,
      "avg_pool3d: kernel_size must either be a single int, or a tuple of three ints");
  // 获取深度、高度和宽度
  const int kD = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kH = kernel_size.size() == 1
      ? kD
      : safe_downcast<int, int64_t>(kernel_size[1]);
  const int kW = kernel_size.size() == 1
      ? kD
      : safe_downcast<int, int64_t>(kernel_size[2]);
  return std::make_tuple(kW, kH, kD);
}

// 获取步长的函数
inline std::tuple<int, int, int> get_stride(IntArrayRef stride, int kW, int kH, int kD) {
  // 检查步长可以为空，或者是一个整数，或者是三个整数组成的元组
  TORCH_CHECK(
      stride.empty() || stride.size() == 1 || stride.size() == 3,
      "avg_pool3d: stride must either be omitted, a single int, or a tuple of three ints");
  // 获取深度、高度和宽度的步长
  const int dD = stride.empty() ? kD : safe_downcast<int, int64_t>(stride[0]);
  const int dH = stride.empty()
      ? kH
      : stride.size() == 1 ? dD : safe_downcast<int, int64_t>(stride[1]);
  const int dW = stride.empty()
      ? kW
      : stride.size() == 1 ? dD : safe_downcast<int, int64_t>(stride[2]);
  return std::make_tuple(dW, dH, dD);
}

// 获取填充的函数
inline std::tuple<int, int, int> get_padding(IntArrayRef padding) {
  // 检查填充必须是一个整数或三个整数组成的元组
  TORCH_CHECK(
      padding.size() == 1 || padding.size() == 3,
      "avg_pool3d: padding must either be a single int, or a tuple of three ints");
  // 获取宽度、高度和深度的填充
  const int padD = safe_downcast<int, int64_t>(padding[0]);
  const int padH =
      padding.size() == 1 ? padD : safe_downcast<int, int64_t>(padding[1]);
  const int padW =
      padding.size() == 1 ? padD : safe_downcast<int, int64_t>(padding[2]);
  return std::make_tuple(padW, padH, padD);
}

// 获取输出形状的函数
std::vector<int64_t> get_output_shape(
    const Tensor& input_,
    int kW,
    int kH,
    int kD,
    int dW,
    int dH,
    int dD,
    int padW,
    int padH,
    int padD,
    // 根据输入张量的维度和参数计算池化层输出的维度信息
    bool ceil_mode) {
      // 计算批次大小，若输入张量为5维则取最后一维的大小，否则默认为1
      const int64_t nbatch = input_.ndimension() == 5 ? input_.size(-5) : 1;
      // 获取输入张量的输入平面数
      const int64_t nInputPlane = input_.size(-4);
      // 获取输入张量的深度
      const int64_t inputDepth = input_.size(-3);
      // 获取输入张量的高度
      const int64_t inputHeight = input_.size(-2);
      // 获取输入张量的宽度
      const int64_t inputWidth = input_.size(-1);
      // 根据输入张量的深度和池化核的深度等参数计算池化后的深度
      const int64_t outputDepth =
          pooling_output_shape<int64_t>(inputDepth, kD, padD, dD, 1, ceil_mode);
      // 根据输入张量的高度和池化核的高度等参数计算池化后的高度
      const int64_t outputHeight =
          pooling_output_shape<int64_t>(inputHeight, kH, padH, dH, 1, ceil_mode);
      // 根据输入张量的宽度和池化核的宽度等参数计算池化后的宽度
      const int64_t outputWidth =
          pooling_output_shape<int64_t>(inputWidth, kW, padW, dW, 1, ceil_mode);
      // 如果输入张量的维度为4，则返回包含输入平面数和池化后的深度、高度、宽度的元组
      if (input_.ndimension() == 4) {
        return {nInputPlane, outputDepth, outputHeight, outputWidth};
      }
      // 否则返回包含批次大小、输入平面数和池化后的深度、高度、宽度的元组
      return {nbatch, nInputPlane, outputDepth, outputHeight, outputWidth};
    }
} // namespace

// 定义模板函数：在输入张量上执行三维平均池化操作，返回处理后的张量
template <typename scalar_t>
Tensor q_avg_pool3d(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    std::optional<int64_t> divisor_override) {

  // 获取池化核大小
  auto [kW, kH, kD] = get_kernel(kernel_size);
  // 获取步幅大小
  auto [dW, dH, dD] = get_stride(stride, kW, kH, kD);
  // 获取填充大小
  auto [padW, padH, padD] = get_padding(padding);

  // 获取输入张量的批量大小、输入通道数、深度、高度和宽度
  const int64_t nbatch = input.ndimension() == 5 ? input.size(-5) : 1;
  const int64_t nInputPlane = input.size(-4);
  const int64_t inputDepth = input.size(-3);
  const int64_t inputHeight = input.size(-2);
  const int64_t inputWidth = input.size(-1);

  // 检查 divisor_override 是否有效，不能为零
  TORCH_CHECK(
      !divisor_override.has_value() || divisor_override.value() != 0,
      "divisor must be not zero");

  // 计算输出形状
  auto output_shape =
      get_output_shape(input, kW, kH, kD, dW, dH, dD, padW, padH, padD, ceil_mode);
  const int64_t outputDepth = output_shape[output_shape.size() - 3];
  const int64_t outputHeight = output_shape[output_shape.size() - 2];
  const int64_t outputWidth = output_shape[output_shape.size() - 1];

  // 将输入张量转换为 ChannelsLast3d 内存格式
  auto input_nhwc = input.contiguous(MemoryFormat::ChannelsLast3d);

  // 创建空的量化输出张量
  auto output = at::_empty_affine_quantized(
      output_shape,
      input_nhwc.options().memory_format(input_nhwc.suggest_memory_format()),
      input_nhwc.q_scale(),
      input_nhwc.q_zero_point(),
      c10::nullopt);

  // 调用底层函数进行量化三维平均池化
  qavg_pool3d_nhwc_stub(
      input_nhwc.device().type(),
      input_nhwc,
      output,
      nbatch,
      nInputPlane,
      inputWidth,
      inputHeight,
      inputDepth,
      outputWidth,
      outputHeight,
      outputDepth,
      kW,
      kH,
      kD,
      dW,
      dH,
      dD,
      padW,
      padH,
      padD,
      count_include_pad,
      divisor_override);

  // 返回量化输出张量
  return output;
}

} // namespace

// 在 native 命名空间下定义量化三维平均池化的 CPU 版本函数
Tensor avg_pool3d_quantized_cpu(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    std::optional<int64_t> divisor_override) {
  Tensor output;
  // 根据输入张量的数据类型调度相应的量化三维平均池化函数
  AT_DISPATCH_QINT_TYPES(input.scalar_type(), "avg_pool3d_quantized_cpu", [&]() {
    output = q_avg_pool3d<scalar_t>(
        input,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override);
  });
  // 返回处理后的输出张量
  return output;
}

} // namespace native
} // namespace at
```