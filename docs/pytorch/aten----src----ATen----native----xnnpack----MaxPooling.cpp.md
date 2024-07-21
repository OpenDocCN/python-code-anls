# `.\pytorch\aten\src\ATen\native\xnnpack\MaxPooling.cpp`

```py
  // 如果定义了宏 USE_XNNPACK，编译以下代码块

  // 包含 XNNPACK 相关头文件
#include <ATen/native/Pool.h>
#include <ATen/native/utils/Factory.h>
#include <ATen/native/xnnpack/Common.h>
#include <ATen/native/xnnpack/Engine.h>
#include <ATen/native/xnnpack/Pooling.h>

// 进入命名空间 at::native::xnnpack
namespace at::native::xnnpack {

// 支持 NHWC 和 NCHW 格式的 FP32 最大池化，支持任意
// - 核大小
// - 填充
// - 步长
// - 膨胀

// 定义函数 use_max_pool2d，用于执行最大池化操作
bool use_max_pool2d(
    const Tensor& input,             // 输入张量
    const IntArrayRef kernel_,       // 核大小数组
    const IntArrayRef padding_,      // 填充数组
    IntArrayRef stride_,             // 步长数组
    const IntArrayRef dilation_,     // 膨胀数组
    const bool ceil_mode,            // 是否使用 ceil 模式
    const float output_min,          // 输出最小值
    const float output_max) {        // 输出最大值

  // 使用 internal 命名空间
  using namespace internal;

  // 确保不处理非常规配置
  if (kernel_.empty() || padding_.empty() || dilation_.empty()) {
    return false;   // 返回 false 表示不支持当前配置
  }

  // 如果步长数组为空，将步长设为核大小
  if (stride_.empty()) {
    stride_ = kernel_;
  }

  // 标准化参数
  const internal::pooling::Parameters parameters{
    kernel_,        // 核大小
    padding_,       // 填充
    stride_,        // 步长
  };

  // Here are the list of conditions required for this code path to be taken:
  // * Input must be 4D CPU float tensor with no gradients.
  // * Kernel must be a 2D IntArrayRef containing two positive numbers.
  //   Furthermore, 1x1 kernels are not valid as XNNPACK prohibits their use.
  // * Padding must be a 2D IntArrayRef containing two non-negative numbers.
  // * Stride must be a 2D IntArrayRef containing two positive numbers.
  // * Dilation must be a 2D IntArrayRef containing two positive numbers.
  // * Ceil mode is not supported and must be disabled.
  // * output_max must be greater than output_min.
  //   Namely, setting both output_min and output_max to 0 is not valid usage.
  // * Finally, application of this operator to the input tensor with the given
  //   max pool 2d parameters must result in an output tensor with a valid shape.

  // Calculate the output height for the PT (PyTorch) implementation of max pooling.
  const int64_t pt_outputHeight = pooling_output_shape(
      input.size(Layout::Activation4D::height),         // Input height dimension
      parameters.kernel[Layout::Parameter::height],     // Kernel height
      parameters.padding[Layout::Parameter::height],    // Padding height
      parameters.stride[Layout::Parameter::height],     // Stride height
      parameters.dilation[Layout::Parameter::height],   // Dilation height
      ceil_mode);                                       // Whether to use ceil mode

  // Calculate the output width for the PT (PyTorch) implementation of max pooling.
  const int64_t pt_outputWidth = pooling_output_shape(
      input.size(Layout::Activation4D::width),          // Input width dimension
      parameters.kernel[Layout::Parameter::width],      // Kernel width
      parameters.padding[Layout::Parameter::width],     // Padding width
      parameters.stride[Layout::Parameter::width],      // Stride width
      parameters.dilation[Layout::Parameter::width],    // Dilation width
      ceil_mode);                                       // Whether to use ceil mode

  // Calculate the output height for the XNNPACK implementation of max pooling.
  const int64_t xnnpack_outputHeight = pooling_output_shape(
      input.size(Layout::Activation4D::height),         // Input height dimension
      parameters.kernel[Layout::Parameter::height],     // Kernel height
      parameters.padding[Layout::Parameter::height],    // Padding height
      parameters.stride[Layout::Parameter::height],     // Stride height
      parameters.dilation[Layout::Parameter::height],   // Dilation height
      false);                                           // XNNPACK does not support ceil mode

  // Calculate the output width for the XNNPACK implementation of max pooling.
  const int64_t xnnpack_outputWidth = pooling_output_shape(
      input.size(Layout::Activation4D::width),          // Input width dimension
      parameters.kernel[Layout::Parameter::width],      // Kernel width
      parameters.padding[Layout::Parameter::width],     // Padding width
      parameters.stride[Layout::Parameter::width],      // Stride width
      parameters.dilation[Layout::Parameter::width],    // Dilation width
      false);                                           // XNNPACK does not support ceil mode

  // Check if the output heights and widths from PT and XNNPACK implementations are equal.
  const bool output_size_eq = (pt_outputHeight == xnnpack_outputHeight) &&
                              (pt_outputWidth == xnnpack_outputWidth);
    // 检查是否为 XNNPACK 可用，并且以下条件均满足时返回 true：

    // 输入
    (4 == input.dim()) &&                                 // 输入张量维度为4
    (input.device().is_cpu()) &&                          // 输入张量存储设备为CPU
    (kFloat == input.scalar_type()) &&                    // 输入张量元素类型为浮点数
    !input.requires_grad() &&                             // 输入张量不需要梯度计算

    // 卷积核
    (2 == parameters.kernel.size()) &&                    // 卷积核尺寸为2
    (parameters.kernel[Layout::Parameter::height] > 0) && // 卷积核高度大于0
    (parameters.kernel[Layout::Parameter::width] > 0) &&  // 卷积核宽度大于0
    ((parameters.kernel[Layout::Parameter::height] *
      parameters.kernel[Layout::Parameter::width]) > 1) && // 卷积核总元素个数大于1

    // 填充
    (2 == parameters.padding.size()) &&                   // 填充尺寸为2
    (parameters.padding[Layout::Parameter::height] >= 0) && // 垂直方向填充值大于等于0
    (parameters.padding[Layout::Parameter::width] >= 0) && // 水平方向填充值大于等于0

    // 步长
    (2 == parameters.stride.size()) &&                    // 步长尺寸为2
    (parameters.stride[Layout::Parameter::height] > 0) && // 垂直方向步长大于0
    (parameters.stride[Layout::Parameter::width] > 0) &&  // 水平方向步长大于0

    // 空洞率
    (2 == parameters.dilation.size()) &&                  // 空洞率尺寸为2
    (parameters.dilation[Layout::Parameter::height] > 0) && // 垂直方向空洞率大于0
    (parameters.dilation[Layout::Parameter::width] > 0) && // 水平方向空洞率大于0

    // 是否使用 ceil 模式以及输出尺寸是否相等
    (!ceil_mode || output_size_eq) &&                     // 如果不使用 ceil 模式或者输出尺寸相等

    // 输出的最小值和最大值
    (output_max > output_min) &&                          // 输出最大值大于输出最小值

    // 输出
    (pooling_output_shape(
      input.size(Layout::Activation4D::height),           // 激活层高度
      parameters.kernel[Layout::Parameter::height],       // 卷积核高度
      parameters.padding[Layout::Parameter::height],      // 填充高度
      parameters.stride[Layout::Parameter::height],       // 步长高度
      parameters.dilation[Layout::Parameter::height],     // 空洞率高度
      ceil_mode) > 0) &&                                  // 是否使用 ceil 模式

    (pooling_output_shape(
      input.size(Layout::Activation4D::width),            // 激活层宽度
      parameters.kernel[Layout::Parameter::width],        // 卷积核宽度
      parameters.padding[Layout::Parameter::width],       // 填充宽度
      parameters.stride[Layout::Parameter::width],        // 步长宽度
      parameters.dilation[Layout::Parameter::width],      // 空洞率宽度
      ceil_mode) > 0) &&                                  // 是否使用 ceil 模式

    // 总体返回结果为 true
    true;
} // namespace at::native::xnnpack
// 结束 at::native::xnnpack 命名空间定义

#endif /* USE_XNNPACK */
// 如果定义了 USE_XNNPACK 宏，则结束条件编译指令块
```