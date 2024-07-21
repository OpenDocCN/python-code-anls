# `.\pytorch\aten\src\ATen\native\NaiveDilatedConvolution.cpp`

```
// 定义宏，用于在编译时只包含方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 引入 ATen 库的核心 Tensor 类和相关头文件
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorUtils.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/native/CPUBlas.h>
#include <ATen/native/DilatedConvolutionUtils.h>
#include <ATen/native/im2col.h>
#include <ATen/native/vol2col.h>
#include <c10/util/accumulate.h>
#include <c10/util/irange.h>
#include <tuple>

// 如果没有定义 AT_PER_OPERATOR_HEADERS，则引入更高级别的 ATen 函数和原生函数
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
// 如果定义了 AT_PER_OPERATOR_HEADERS，则引入特定操作符的头文件
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/slow_conv_dilated2d_native.h>
#include <ATen/ops/slow_conv_dilated3d_native.h>
#endif

// 定义 ATen 的命名空间 at::native
namespace at::native {

// 定义匿名命名空间，用于限定作用域
namespace {

// 将超体积转换为列（hyper-volume to column），在 CPU 上操作
template <typename Dtype, int64_t dim>
void hvol2col(
    const Dtype* data_hvol,                  // 超体积数据的指针
    const int channels,                      // 数据通道数
    const IntArrayRef input_size,            // 输入尺寸的数组引用
    const IntArrayRef output_size,           // 输出尺寸的数组引用
    const IntArrayRef kernel_size,           // 卷积核尺寸的数组引用
    const IntArrayRef stride_size,           // 步幅尺寸的数组引用
    const IntArrayRef pad_size,              // 填充尺寸的数组引用
    const IntArrayRef dilation_size,         // 膨胀尺寸的数组引用
    Dtype* data_col,                         // 列数据的指针
    bool is_channels_last = false            // 是否最后一个维度是通道维度的标志位，默认为假
) {
  // 如果维度为 3，则调用 vol2col 函数转换
  if (dim == 3) {
    vol2col<Dtype>(
        data_hvol,                          // 超体积数据的指针
        channels,                           // 数据通道数
        input_size[0],                      // 输入尺寸的第一个维度
        input_size[1],                      // 输入尺寸的第二个维度
        input_size[2],                      // 输入尺寸的第三个维度
        output_size[0],                     // 输出尺寸的第一个维度
        output_size[1],                     // 输出尺寸的第二个维度
        output_size[2],                     // 输出尺寸的第三个维度
        kernel_size[0],                     // 卷积核尺寸的第一个维度
        kernel_size[1],                     // 卷积核尺寸的第二个维度
        kernel_size[2],                     // 卷积核尺寸的第三个维度
        pad_size[0],                        // 填充尺寸的第一个维度
        pad_size[1],                        // 填充尺寸的第二个维度
        pad_size[2],                        // 填充尺寸的第三个维度
        stride_size[0],                     // 步幅尺寸的第一个维度
        stride_size[1],                     // 步幅尺寸的第二个维度
        stride_size[2],                     // 步幅尺寸的第三个维度
        dilation_size[0],                   // 膨胀尺寸的第一个维度
        dilation_size[1],                   // 膨胀尺寸的第二个维度
        dilation_size[2],                   // 膨胀尺寸的第三个维度
        data_col                            // 列数据的指针
    );
  }
  // 如果维度为 2，则调用 im2col 函数转换
  if (dim == 2) {
    im2col<Dtype>(
        data_hvol,                          // 超体积数据的指针
        channels,                           // 数据通道数
        input_size[0],                      // 输入尺寸的第一个维度
        input_size[1],                      // 输入尺寸的第二个维度
        output_size[0],                     // 输出尺寸的第一个维度
        output_size[1],                     // 输出尺寸的第二个维度
        kernel_size[0],                     // 卷积核尺寸的第一个维度
        kernel_size[1],                     // 卷积核尺寸的第二个维度
        pad_size[0],                        // 填充尺寸的第一个维度
        pad_size[1],                        // 填充尺寸的第二个维度
        stride_size[0],                     // 步幅尺寸的第一个维度
        stride_size[1],                     // 步幅尺寸的第二个维度
        dilation_size[0],                   // 膨胀尺寸的第一个维度
        dilation_size[1],                   // 膨胀尺寸的第二个维度
        data_col,                           // 列数据的指针
        is_channels_last                    // 是否最后一个维度是通道维度的标志位
    );
  }
}

// 将列转换为超体积（column to hyper-volume），在 CPU 上操作
template <typename Dtype, int64_t dim>
void col2hvol(
    const Dtype* data_col,                  // 列数据的指针
    const int channels,                     // 数据通道数
    const IntArrayRef input_size,           // 输入尺寸的数组引用
    const IntArrayRef output_size,          // 输出尺寸的数组引用
    const IntArrayRef kernel_size,          // 卷积核尺寸的数组引用
    const IntArrayRef stride_size,          // 步幅尺寸的数组引用
    const IntArrayRef pad_size,             // 填充尺寸的数组引用
    const IntArrayRef dilation_size,        // 膨胀尺寸的数组引用
    Dtype* data_hvol,                      // 超体积数据的指针
    bool is_channels_last = false           // 是否最后一个维度是通道维度的标志位，默认为假
) {
  // 如果维度为 3，则调用 col2vol 函数转换
  if (dim == 3) {
    col2vol<Dtype>(
        data_col,                           // 列数据的指针
        channels,                           // 数据通道数
        input_size[0],                      // 输入尺寸的第一个维度
        input_size[1],                      // 输入尺寸的第二个维度
        input_size[2],                      // 输入尺寸的第三个维度
        output_size[0],                     // 输出尺寸的第一个维度
        output_size[1],                     // 输出尺寸的第二个维度
        output_size[2],                     // 输出尺寸的第三个维度
        kernel_size[0],                     // 卷积核尺寸的第一个维度
        kernel_size[1],                     // 卷积核尺寸的第二个维度
        kernel_size[2],                     // 卷积核尺寸的第三个维度
        pad_size[0],                        // 填充尺寸的第一个维度
        pad_size[1],                        // 填充尺寸的第二个维度
        pad_size[2],                        // 填充尺寸的第三个维度
        stride_size[0],                     // 步
    # 调用 col2im 函数，进行反向操作，将列形式的数据转换为输入图像的形式
    col2im<Dtype>(
        data_col,                # 列形式的数据，用于转换为输入图像的形式
        channels,                # 图像的通道数
        input_size[0],           # 输入图像的高度尺寸
        input_size[1],           # 输入图像的宽度尺寸
        output_size[0],          # 输出图像的高度尺寸
        output_size[1],          # 输出图像的宽度尺寸
        kernel_size[0],          # 卷积核的高度尺寸
        kernel_size[1],          # 卷积核的宽度尺寸
        pad_size[0],             # 填充尺寸的垂直方向大小
        pad_size[1],             # 填充尺寸的水平方向大小
        stride_size[0],          # 步幅尺寸的垂直方向大小
        stride_size[1],          # 步幅尺寸的水平方向大小
        dilation_size[0],        # 膨胀尺寸的垂直方向大小
        dilation_size[1],        # 膨胀尺寸的水平方向大小
        data_hvol,               # 输出的图像数据存储位置
        is_channels_last         # 是否按通道为最后一个维度的顺序存储数据
    );
/*
   结束 slow_conv_dilated_all_cpu_template 函数定义
 */
void slow_conv_dilated_location_check(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const Tensor& grad_output) {
  // 检查用户提供的张量参数的数据位置
  checkBackend("slow_conv_dilated_location_check", {input, weight}, Backend::CPU);
  // 如果有定义偏置张量，检查其数据位置
  if (bias.defined()) {
    checkBackend("slow_conv_dilated_location_check", {bias}, Backend::CPU);
  }
  // 如果有定义梯度输出张量，检查其数据位置
  if (grad_output.defined()) {
    checkBackend("slow_conv_dilated_location_check", {grad_output}, Backend::CPU);
  }
  // 不检查其它张量参数的数据位置，因为它们根据输入选项分配，因此这些张量的数据位置始终与输入张量相同。
}

/*
  slow_conv_dilated_all_cpu_template

  主工作函数。计算输出张量、梯度输入张量、梯度权重张量和/或梯度偏置张量（如果有定义），分别赋值。
 */

template <int64_t dim>
void slow_conv_dilated_all_cpu_template(
    Tensor& output,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const Tensor& grad_output,
    Tensor& grad_input,
    Tensor& grad_weight,
    Tensor& grad_bias,
    IntArrayRef kernel_size,
    IntArrayRef stride_size,
    IntArrayRef pad_size,
    IntArrayRef dilation_size,
    bool is_channels_last = false) {
  // 检查输入、权重、偏置和梯度输出张量的数据位置
  slow_conv_dilated_location_check(input, weight, bias, grad_output);
  // 获取输入张量的选项
  auto options = input.options();
  // 获取输入张量后维度的大小：
  auto input_size = input.sizes().slice(2);
  // 获取输出张量后维度的大小：
  auto output_size = internal::get_output_size<dim>(
      input, kernel_size, stride_size, pad_size, dilation_size);
  // 获取批次大小、输入平面数、输出平面数
  int64_t batchSize = input.size(0);
  int64_t nInputPlane = weight.size(1);
  int64_t nOutputPlane = weight.size(0);
  // 临时缓冲区
  Tensor columns = at::empty({0}, options);
  // 如果输出、梯度权重或梯度输入张量已经定义
  if (output.defined() || grad_weight.defined() || grad_input.defined()) {
    const int64_t m = c10::multiply_integers(kernel_size);
    const int64_t n = c10::multiply_integers(output_size);
    // 根据通道是否末尾重设 columns 张量大小
    if (is_channels_last) {
      columns.resize_({n, m * nInputPlane});
    } else {
      columns.resize_({nInputPlane * m, n});
    }
  }
  // 初始化操作
  if (grad_weight.defined()) {
    grad_weight.zero_();
  }
  if (grad_bias.defined()) {
    grad_bias.zero_();
  }
  if (output.defined() && !bias.defined()) {
    output.zero_();
  }
  // 辅助张量和变量
  Tensor grad_output_n;
  std::vector<int64_t> dims(dim);
  std::iota(dims.begin(), dims.end(), 1);

  // 根据输入张量的标量类型调度，执行慢卷积扩展的操作
  AT_DISPATCH_FLOATING_TYPES_AND3(
      at::ScalarType::Long, at::ScalarType::BFloat16, at::ScalarType::Half, input.scalar_type(), "slow_conv_dilated<>", [&] {
    // 对于批次中的每个元素，执行以下操作：
    // (这里应该有具体的操作代码，这里只是占位符)
  });

} // 结束 slow_conv_dilated_all_cpu_template 函数定义

} // 结束 namespace
    // 从可选的张量中借用偏置张量（如果存在），并确保其可用
    c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
    // 获取实际的偏置张量引用
    const Tensor& bias = *bias_maybe_owned;
    
    // 确定是否使用通道最后的内存布局格式
    bool use_channels_last = thnn_conv_use_channels_last(input, weight);
    auto memory_format = use_channels_last ? at::MemoryFormat::ChannelsLast : at::MemoryFormat::Contiguous;
    
    // 创建一个未定义的张量用于形状检查
    Tensor undefined;
    // 执行卷积的形状检查，确保输入、权重和偏置的形状满足要求
    internal::slow_conv_dilated_shape_check<2>(
        input,
        weight,
        bias,
        undefined,
        kernel_size,
        stride_size,
        pad_size,
        dilation_size);
    
    // 判断输入张量是否为批处理张量
    auto is_batch = input.dim() == 4;
    // 获取输入张量的选项
    auto options = input.options();
    
    // 计算输出张量的大小
    auto output_size = internal::get_output_size<2>(
        input, weight, kernel_size, stride_size, pad_size, dilation_size);
    
    // 根据内存布局格式进行必要的张量操作
    const Tensor input_ =
        (is_batch ? input.contiguous(memory_format) : input.contiguous().unsqueeze(0));
    const Tensor weight_ = weight.contiguous(memory_format);
    const Tensor bias_ = (bias.defined() ? bias.contiguous() : undefined);
    
    // 创建空的输出张量
    Tensor output = at::empty(output_size, options.memory_format(memory_format));
    // 如果输入是批处理张量，对输出张量进行扩展以匹配批处理的维度
    Tensor output_ = (is_batch ? output : output.unsqueeze(0));
    
    // 执行具体的慢卷积计算模板
    slow_conv_dilated_all_cpu_template<2>(
        output_,
        input_,
        weight_,
        bias_,
        undefined,
        undefined,
        undefined,
        undefined,
        kernel_size,
        stride_size,
        pad_size,
        dilation_size,
        use_channels_last);
    
    // 返回计算得到的输出张量
    return output;
}



// 定义一个静态函数，用于在 CPU 上计算 3D 扩张卷积的慢速版本的前向传播
Tensor slow_conv_dilated3d_cpu(
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size, const std::optional<Tensor>& bias_opt,
    IntArrayRef stride_size,
    IntArrayRef pad_size,
    IntArrayRef dilation_size) {
  
  // 将可选的偏置张量转换为 MaybeOwned<Tensor>，并确保获取其中的偏置张量
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  // 创建一个未定义的张量对象
  Tensor undefined;

  // 进行 3D 扩张卷积的形状检查
  internal::slow_conv_dilated_shape_check<3>(
      input,
      weight,
      bias,
      undefined,
      kernel_size,
      stride_size,
      pad_size,
      dilation_size);

  // 检查输入张量是否具有批处理维度
  auto is_batch = input.dim() == 5;
  
  // 获取输入张量的选项
  auto options = input.options();

  // 计算输出张量的大小
  auto output_size = internal::get_output_size<3>(
      input, weight, kernel_size, stride_size, pad_size, dilation_size);

  // 如果输入张量是批处理的，则保持连续性；否则在第 0 维度插入批处理维度
  const Tensor input_ =
      (is_batch ? input.contiguous() : input.contiguous().unsqueeze(0));

  // 权重张量保持连续性
  const Tensor weight_ = weight.contiguous();

  // 如果偏置张量已定义，则保持连续性；否则使用未定义张量
  const Tensor bias_ = (bias.defined() ? bias.contiguous() : undefined);

  // 创建一个空的输出张量，使用与输入相同的选项
  Tensor output = at::empty(output_size, options);

  // 如果输入张量是批处理的，则保持连续性；否则在第 0 维度插入批处理维度
  Tensor output_ = (is_batch ? output : output.unsqueeze(0));

  // 调用模板函数，在 CPU 上执行 3D 扩张卷积的计算
  slow_conv_dilated_all_cpu_template<3>(
      output,
      input_,
      weight_,
      bias_,
      undefined,
      undefined,
      undefined,
      undefined,
      kernel_size,
      stride_size,
      pad_size,
      dilation_size);

  // 返回计算得到的输出张量
  return output;
}

// 定义一个静态函数，用于在 CPU 上计算 2D 扩张卷积的慢速版本的反向传播
static std::tuple<Tensor, Tensor, Tensor> slow_conv_dilated2d_backward_cpu(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size,
    IntArrayRef stride_size,
    IntArrayRef pad_size,
    IntArrayRef dilation_size,
    // 定义函数，计算梯度并返回，参数包括输入张量、权重张量、梯度输出、卷积核大小、步长大小、填充大小、膨胀大小以及输出掩码数组
    const std::array<bool, 3ul> output_mask) {
      // 确定是否使用通道最后的存储顺序
      bool use_channels_last = thnn_conv_use_channels_last(input, weight);
      // 根据使用的存储顺序选择内存格式
      auto memory_format = use_channels_last ? at::MemoryFormat::ChannelsLast : at::MemoryFormat::Contiguous;
    
      // 定义未定义的张量
      Tensor undefined;
    
      // 检查卷积的形状，支持膨胀的卷积
      internal::slow_conv_dilated_shape_check<2>(
          input,
          weight,
          undefined,
          grad_output,
          kernel_size,
          stride_size,
          pad_size,
          dilation_size);
    
      // 检查输入张量是否是批量的
      auto is_batch = input.dim() == 4;
      // 获取梯度输出的选项
      auto options = grad_output.options();
    
      // 如果是批量的，将梯度输出调整为相应的内存格式，并在必要时插入批量维度
      const Tensor grad_output_ =
          (is_batch ? grad_output.contiguous(memory_format)
                    : grad_output.contiguous().unsqueeze(0));
      const Tensor input_ =
          (is_batch ? input.contiguous(memory_format) : input.contiguous().unsqueeze(0));
      const Tensor weight_ = weight.contiguous(memory_format);
    
      // 根据输出掩码数组计算梯度，仅计算对应输出掩码为true的部分
      Tensor grad_input =
          (output_mask[0] ? at::empty(input.sizes(), options.memory_format(memory_format)) : undefined);
      Tensor grad_weight =
          (output_mask[1] ? at::empty(weight.sizes(), options.memory_format(memory_format)) : undefined);
      Tensor grad_bias =
          (output_mask[2] ? at::empty(weight.size(0), options) : undefined);
    
      // 根据是否需要梯度计算，调整梯度输入的形状
      Tensor grad_input_ =
          (output_mask[0] ? (is_batch ? grad_input : grad_input.unsqueeze(0))
                          : undefined);
    
      // 调用模板函数计算所有梯度，CPU版本，支持膨胀的卷积
      slow_conv_dilated_all_cpu_template<2>(
          undefined,
          input_,
          weight_,
          undefined,
          grad_output_,
          grad_input,
          grad_weight,
          grad_bias,
          kernel_size,
          stride_size,
          pad_size,
          dilation_size,
          use_channels_last);
    
      // 返回计算得到的梯度张量，按顺序为输入梯度、权重梯度、偏置梯度
      return std::tie(grad_input, grad_weight, grad_bias);
    }
} // 结束 at::native 命名空间

static std::tuple<Tensor, Tensor, Tensor> slow_conv_dilated3d_backward_cpu(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size,
    IntArrayRef stride_size,
    IntArrayRef pad_size,
    IntArrayRef dilation_size,
    const std::array<bool, 3ul> output_mask) {
  // 声明一个未定义的 Tensor 对象
  Tensor undefined;
  // 进行慢速三维膨胀卷积的形状检查
  internal::slow_conv_dilated_shape_check<3>(
      input,
      weight,
      undefined,
      grad_output,
      kernel_size,
      stride_size,
      pad_size,
      dilation_size);
  // 检查输入张量是否有批处理维度
  auto is_batch = input.dim() == 5;
  // 获取梯度输出的选项
  auto options = grad_output.options();
  // 如果有批处理维度，保持张量连续性；否则，在不影响原始张量的情况下插入批处理维度
  const Tensor grad_output_ =
      (is_batch ? grad_output.contiguous()
                : grad_output.contiguous().unsqueeze(0));
  const Tensor input_ =
      (is_batch ? input.contiguous() : input.contiguous().unsqueeze(0));
  const Tensor weight_ = weight.contiguous();
  // 根据输出掩码仅计算相应的梯度
  Tensor grad_input =
      (output_mask[0] ? at::empty(input.sizes(), options) : undefined);
  Tensor grad_weight =
      (output_mask[1] ? at::empty(weight.sizes(), options) : undefined);
  Tensor grad_bias =
      (output_mask[2] ? at::empty(weight.size(0), options) : undefined);
  Tensor grad_input_ =
      (output_mask[0] ? (is_batch ? grad_input : grad_input.unsqueeze(0))
                      : undefined);
  // 调用慢速三维膨胀卷积的 CPU 模板函数
  slow_conv_dilated_all_cpu_template<3>(
      undefined,
      input_,
      weight_,
      undefined,
      grad_output_,
      grad_input,
      grad_weight,
      grad_bias,
      kernel_size,
      stride_size,
      pad_size,
      dilation_size);
  // 返回梯度张量的元组
  return std::tie(grad_input, grad_weight, grad_bias);
}

// 注册所有 CPU 分发的慢速二维膨胀卷积反向函数
REGISTER_ALL_CPU_DISPATCH(slow_conv_dilated2d_backward_stub, &slow_conv_dilated2d_backward_cpu);
// 注册所有 CPU 分发的慢速三维膨胀卷积反向函数
REGISTER_ALL_CPU_DISPATCH(slow_conv_dilated3d_backward_stub, &slow_conv_dilated3d_backward_cpu);
```