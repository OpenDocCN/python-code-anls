# `.\pytorch\aten\src\ATen\native\xnnpack\Convolution.cpp`

```py
// #ifdef USE_XNNPACK 宏定义条件编译，检查是否启用了 XNNPACK 库

#include <vector> // 引入标准库中的 vector 头文件

#include <ATen/native/ConvUtils.h> // 引入 PyTorch 的卷积工具函数头文件
#include <ATen/native/utils/Factory.h> // 引入 PyTorch 的工厂函数头文件
#include <ATen/native/utils/ParamUtils.h> // 引入 PyTorch 的参数工具函数头文件
#include <ATen/native/xnnpack/Common.h> // 引入 PyTorch XNNPACK 模块的通用函数头文件
#include <ATen/native/xnnpack/Convolution.h> // 引入 PyTorch XNNPACK 模块的卷积函数头文件
#include <ATen/native/xnnpack/Engine.h> // 引入 PyTorch XNNPACK 模块的引擎头文件
#include <c10/util/irange.h> // 引入 C10 库中的整数范围迭代器头文件

namespace at::native::xnnpack { // 定义命名空间 at::native::xnnpack

namespace internal { // 内部命名空间 internal

namespace convolution2d { // 卷积操作命名空间 convolution2d

namespace { // 匿名命名空间，限定内部变量和函数的作用域

// 支持 NHWC 和 NCHW 格式的 FP32 卷积，支持任意有效的
// - 卷积核大小
// - 填充
// - 步幅
// - 膨胀
// - 分组

// TODO: 解耦并改进错误处理和消息。
bool available(
    const Tensor& weight, // 权重张量
    const at::OptionalIntArrayRef bias_sizes_opt, // 可选的偏置大小
    const IntArrayRef padding, // 填充大小数组
    const IntArrayRef stride, // 步幅大小数组
    const IntArrayRef dilation, // 膨胀大小数组
    const int64_t groups, // 分组数量
    const bool transposed, // 是否转置卷积
    const float output_min, // 输出最小值
    const float output_max) { // 输出最大值

  // 检查是否 XNNPACK 可用，并且...
  return xnnpack::available() &&
         // 权重张量
         (4 == weight.ndimension()) && // 权重张量的维度为4
         (weight.size(Layout::Filter::height) > 0) && // 权重张量高度维度大于0
         (weight.size(Layout::Filter::width) > 0) && // 权重张量宽度维度大于0
         (weight.device().is_cpu()) && // 权重张量位于 CPU 设备上
         (kFloat == weight.scalar_type()) && // 权重张量数据类型为 float
         // 偏置
         (bias_sizes_opt.has_value() ? ((1 == bias_sizes_opt->size()) &&
                ((transposed ? (weight.size(Layout::Filter::input) ==
                                ((*bias_sizes_opt)[0] / groups))
                  : (weight.size(Layout::Filter::output) == ((*bias_sizes_opt)[0])))))
            : true) && // 如果有偏置，验证其大小是否匹配
         // 填充
         (padding[Layout::Parameter::height] >= 0) && // 高度方向填充大小非负
         (padding[Layout::Parameter::width] >= 0) && // 宽度方向填充大小非负
         // 步幅
         (stride[Layout::Parameter::height] > 0) && // 高度方向步幅大于0
         (stride[Layout::Parameter::width] > 0) && // 宽度方向步幅大于0
         // 膨胀
         (dilation[Layout::Parameter::height] > 0) && // 高度方向膨胀大于0
         (dilation[Layout::Parameter::width] > 0) && // 宽度方向膨胀大于0
         // 分组
         (groups > 0) && // 分组数大于0
         // 输入
         (weight.size(Layout::Filter::input) > 0) && // 输入通道数大于0
         // 输出
         (weight.size(Layout::Filter::output) > 0) && // 输出通道数大于0
         // 输出 - 分组
         ((weight.size(Layout::Filter::output) % groups) == 0) && // 输出通道数可以被分组整除
         // 输出最小值 / 最大值
         (output_max > output_min) && // 输出最大值大于输出最小值
         true; // 返回 true
}

// TODO: 解耦并改进错误处理和消息。
bool usable(const Tensor& input) { // 检查输入张量是否可用

  // 输入
  return (4 == input.ndimension()) && // 输入张量的维度为4
         (input.device().is_cpu()) && // 输入张量位于 CPU 设备上
         (kFloat == input.scalar_type()) && // 输入张量数据类型为 float
         (input.size(Layout::Activation4D::batch) >= 0) && // 批次维度大小大于等于0
         (input.size(Layout::Activation4D::channels) > 0) && // 通道数大于0
         (input.size(Layout::Activation4D::height) > 0) && // 高度大于0
         (input.size(Layout::Activation4D::width) > 0) && // 宽度大于0
         !input.requires_grad() && // 输入张量不需要梯度
         true; // 返回 true
}

Tensor create_and_run(
    const Tensor& input, // 输入张量
    const Tensor& weight, // 权重张量
    const Tensor& bias, // 偏置张量
    const IntArrayRef padding, // 填充大小数组
    const IntArrayRef output_padding, // 输出填充大小数组
    const IntArrayRef stride, // 步幅大小数组
    const IntArrayRef dilation,        // dilation：整数数组引用，用于卷积的膨胀系数
    const int64_t groups,             // groups：整数，卷积操作中的分组数
    const bool transposed,            // transposed：布尔值，指示是否为转置卷积操作
    const float output_min,           // output_min：浮点数，输出张量的最小值
    const float output_max) {         // output_max：浮点数，输出张量的最大值

  auto op_context = create(            // 创建一个操作上下文对象，使用以下参数：
      weight,                         // weight：权重张量
      bias,                           // bias：偏置张量
      padding,                        // padding：填充设置
      output_padding,                 // output_padding：输出填充设置
      stride,                         // stride：步幅设置
      dilation,                       // dilation：膨胀系数设置
      groups,                         // groups：分组数设置
      transposed,                     // transposed：是否为转置卷积
      output_min,                     // output_min：输出张量的最小值
      output_max);                    // output_max：输出张量的最大值

  return run(op_context, input);      // 运行操作上下文对象，使用输入张量作为参数
}
}

// XNNPack's deconvolution operator expects weights to be indexed in the following order:
//   * Groups
//   * Group Output Channels
//   * Kernel Height
//   * Kernel Width
//   * Group Input Channels
//
// (ref: https://github.com/google/XNNPACK/blob/ecd8311c8fd3d9ab47edbc3df5f2b5de7dabe75f/test/deconvolution-operator-tester.h#L678)
//
// This function takes in a contiguous NHWC pytorch tensor (e.g. MemoryFormat == ChannelsLast) and rearranges the weights in preparation for use with xnnpack.
// By default, for pytorch, transpose conv2d weights are {input_channels, output_Channels_per_group, kernel_height, kernel_width}.
// In addition, it condenses the tensor from 5 to 4 dimensions as expected by the rest of the pytorch framework by combining the groups and input_channels dimension.
const Tensor reorder_weights_for_transpose_conv(const Tensor& weight_nhwc,
    int num_groups) {

  TORCH_CHECK(weight_nhwc.size(0) % num_groups == 0, "The number of groups cannot be satisfied by the provided weight tensor.");

  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  // 计算每个组内的输入通道数
  int input_channels_per_group = weight_nhwc.size(0) / num_groups;
  // 计算每个组内的输出通道数
  int output_channels_per_group = weight_nhwc.size(1);
  // 卷积核的宽度
  int kernel_width = weight_nhwc.size(3);
  // 卷积核的高度
  int kernel_height = weight_nhwc.size(2);

  // 计算不同维度的偏移量，以便重新排列权重
  int o_offset = 1;
  int h_offset = (output_channels_per_group);
  int w_offset = (output_channels_per_group)*(kernel_height);
  int i_offset = (output_channels_per_group)*(kernel_height)*(kernel_width);
  int g_offset = (output_channels_per_group)*(kernel_height)*(kernel_width)*(input_channels_per_group);

  // 创建一个新的张量来存储重新排列后的权重
  Tensor reordered = mobile::empty_with_tail_padding(
     weight_nhwc.sizes(),
     weight_nhwc.options().dtype(),
     MemoryFormat::ChannelsLast,
     weight_nhwc.opt_names());

  // 获取重新排列后的张量的指针和原始权重张量的指针
  float* out_ptr = reordered.data_ptr<float>();
  float* in_ptr = weight_nhwc.data_ptr<float>();

  // 遍历所有组、输出通道、卷积核位置、输入通道位置，重新排列权重
  int out_index = 0;
  for (const auto g : c10::irange(num_groups)) {
    for (const auto o : c10::irange(output_channels_per_group)) {
      for (const auto w : c10::irange(kernel_width)) {
        for (const auto h : c10::irange(kernel_height)) {
          for (const auto i : c10::irange(input_channels_per_group)) {
            // 计算原始权重张量中的索引位置，并将其放入重新排列后的张量中
            int in_index = (g*g_offset) + (i*i_offset) + (h*h_offset) + (w*w_offset) + (o*o_offset);
            out_ptr[out_index] = in_ptr[in_index];
            out_index++;
          }
        }
      }
    }
  }

  // 返回重新排列后的权重张量
  return reordered;
}

} // namespace

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
  // 如果需要，根据需要扩展填充参数到2维度
  const auto padding_expanded = expand_param_if_needed(padding, "padding", 2);
  // 如果需要，根据需要扩展输出填充参数到2维度
  const auto output_padding_expanded = expand_param_if_needed(output_padding, "output_padding", 2);
  // 如果需要，根据需要扩展步长参数到2维度
  const auto stride_expanded = expand_param_if_needed(stride, "stride", 2);
  // 如果需要，根据需要扩展膨胀参数到2维度
  const auto dilation_expanded = expand_param_if_needed(dilation, "dilation", 2);
  // 将权重张量转换为ChannelsLast内存格式，确保连续性
  const Tensor weight_nhwc = weight.contiguous(MemoryFormat::ChannelsLast);

  // 检查xnnpack::convolution是否可用，验证给定的参数组合是否支持
  TORCH_CHECK(
      available(
          weight_nhwc,
          (bias.has_value() && bias->defined()) ? at::OptionalIntArrayRef(bias->sizes()) : c10::nullopt,
          padding_expanded,
          stride_expanded,
          dilation_expanded,
          groups,
          transposed,
          output_min,
          output_max),
      "xnnpack::convolution not available! "
      "Reason: The provided (weight, bias, padding, stride, dilation, groups, transposed, output_min, output_max) "
      "parameters are either invalid individually or their combination is not supported by XNNPACK.");

  // XNNPACK卷积操作符的声明和初始化
  xnn_operator_t convolution_op{};
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  // xnn_status类型的创建状态
  xnn_status create_status;
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  // 用于存储权重张量的大小的数组，初始化为4个维度大小
  std::array<int64_t, 4> weight_sizes;

  // 如果是转置卷积，重新排序权重张量以备转置卷积使用，并获取各维度大小
  if (transposed) {
    const Tensor weight_reordered = reorder_weights_for_transpose_conv(weight_nhwc, groups);
    for (const auto i : c10::irange(4)) {
      weight_sizes[i] = weight_reordered.size(i);
    }
    // 调用 xnn_create_deconvolution2d_nhwc_f32 函数创建反卷积操作
    create_status = xnn_create_deconvolution2d_nhwc_f32(
      padding_expanded[Layout::Parameter::height],                    // 输出填充顶部
      padding_expanded[Layout::Parameter::width],                     // 输出填充右侧
      padding_expanded[Layout::Parameter::height],                    // 输出填充底部
      padding_expanded[Layout::Parameter::width],                     // 输出填充左侧
      weight_reordered.size(Layout::Filter::height),                  // 卷积核高度
      weight_reordered.size(Layout::Filter::width),                   // 卷积核宽度
      stride_expanded[Layout::Parameter::height],                     // 高度上的子采样
      stride_expanded[Layout::Parameter::width],                      // 宽度上的子采样
      dilation_expanded[Layout::Parameter::height],                   // 高度上的膨胀率
      dilation_expanded[Layout::Parameter::width],                    // 宽度上的膨胀率
      groups,                                                         // 分组数
      weight_reordered.size(Layout::Filter::output) / groups,         // 每组的输入通道数
      weight_reordered.size(Layout::Filter::input),                   // 每组的输出通道数
      weight_reordered.size(Layout::Filter::output),                  // 输入数据的像素步长
      weight_reordered.size(Layout::Filter::input) * groups,          // 输出数据的像素步长
      weight_reordered.data_ptr<float>(),                             // 卷积核数据指针
      (bias && bias->defined())
          ? bias->contiguous().data_ptr<float>()
          : nullptr,                                                  // 偏置数据指针，如果没有偏置则为 nullptr
      output_min,                                                     // 输出的最小值
      output_max,                                                     // 输出的最大值
      0u,                                                             // 标志位（这里为 0）
      nullptr,                                                        // XNN 运行时缓存
      nullptr,                                                        // XNN 权重缓存
      &convolution_op);                                               // 卷积操作符
  } else {
    // 对于非创建反卷积的情况，获取权重的尺寸信息
    for (const auto i : c10::irange(4)) {
      weight_sizes[i] = weight_nhwc.size(i);
    }
    create_status = xnn_create_convolution2d_nhwc_f32(
      padding_expanded[Layout::Parameter::height],                    // 设置输入的顶部填充值
      padding_expanded[Layout::Parameter::width],                     // 设置输入的右侧填充值
      padding_expanded[Layout::Parameter::height],                    // 设置输入的底部填充值
      padding_expanded[Layout::Parameter::width],                     // 设置输入的左侧填充值
      weight_nhwc.size(Layout::Filter::height),                       // 获取卷积核的高度
      weight_nhwc.size(Layout::Filter::width),                        // 获取卷积核的宽度
      stride_expanded[Layout::Parameter::height],                     // 设置高度上的子采样步长
      stride_expanded[Layout::Parameter::width],                      // 设置宽度上的子采样步长
      dilation_expanded[Layout::Parameter::height],                   // 设置高度上的膨胀率
      dilation_expanded[Layout::Parameter::width],                    // 设置宽度上的膨胀率
      groups,                                                         // 设置卷积操作的分组数
      weight_nhwc.size(Layout::Filter::input),                        // 获取每个分组的输入通道数
      weight_nhwc.size(Layout::Filter::output) / groups,              // 获取每个分组的输出通道数
      weight_nhwc.size(Layout::Filter::input) * groups,               // 设置输入像素步长
      weight_nhwc.size(Layout::Filter::output),                       // 设置输出像素步长
      weight_nhwc.data_ptr<float>(),                                  // 设置卷积核数据指针
      (bias && bias->defined())
          ? bias->contiguous().data_ptr<float>()
          : nullptr,                                                  // 设置偏置数据指针，如果不存在则为nullptr
      output_min,                                                     // 设置输出的最小值
      output_max,                                                     // 设置输出的最大值
      0u,                                                             // 设置标志位
      nullptr,                                                        // 设置缓存数据结构
      nullptr,                                                        // 设置权重缓存数据结构
      &convolution_op);                                               // 返回创建的卷积操作符
  }

  TORCH_CHECK(
      xnn_status_success == create_status,
      (transposed ? "xnn_create_deconvolution2d_nhwc_f32 failed!"    // 检查卷积或反卷积操作创建是否成功
                  : "xnn_create_convolution2d_nhwc_f32 failed!"));    // 如果是反卷积，则返回反卷积创建失败信息，否则返回卷积创建失败信息

  return ContextConv2D{
      Operator(convolution_op),                                       // 返回卷积操作符
      weight_sizes,                                                   // 返回卷积核大小信息
      {padding_expanded[0], padding_expanded[1]},                      // 返回填充扩展后的高度和宽度
      {output_padding_expanded[0], output_padding_expanded[1]},        // 返回输出填充扩展后的高度和宽度
      {stride_expanded[0], stride_expanded[1]},                        // 返回步长扩展后的高度和宽度
      {dilation_expanded[0], dilation_expanded[1]},                    // 返回膨胀率扩展后的高度和宽度
      transposed,                                                     // 返回是否为反卷积
      groups                                                          // 返回分组数
  };
}

Tensor run(
    ContextConv2D& context,
    const Tensor& input) {
  using namespace internal;

  const Tensor padded_input_nhwc = mobile::allocate_padded_contiguous_if_needed(
      input, MemoryFormat::ChannelsLast);

  TORCH_CHECK(
      usable(padded_input_nhwc),
      "XNNPACK Convolution not usable! "
      "Reason: The provided input tensor is either invalid or unsupported by XNNPACK.");

  Tensor output;
  if (context.transposed_) {
    // 计算转置卷积的输出尺寸，并创建对应形状的空输出张量
    output = mobile::empty_with_tail_padding(
      conv_input_size(padded_input_nhwc.sizes(),
        context.weight_size_,
        context.padding_,
        context.output_padding_,
        context.stride_,
        context.dilation_,
        context.groups_),
      padded_input_nhwc.options().dtype(),
      MemoryFormat::ChannelsLast,
      padded_input_nhwc.opt_names());
  } else {
    // 计算正常卷积的输出尺寸，并创建对应形状的空输出张量
    output = mobile::empty_with_tail_padding(
      conv_output_size(
          padded_input_nhwc.sizes(),
          context.weight_size_,
          context.padding_,
          context.stride_,
          context.dilation_),
      padded_input_nhwc.options().dtype(),
      MemoryFormat::ChannelsLast,
      padded_input_nhwc.opt_names());
  }

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  xnn_status setup_status;

  /*
   * Input Pointer Caching:
   * Previously, we cached the input/output pointers and dimension parameters
   * so that if the same pointers and parameters are used, this setup could be
   * skipped.
   * However, XNNPack has integrated offsets with its indirection buffer, so the
   * buffer does not need to be recalculated even if activation tensor pointer
   * changes as long as tensor dimensions are the same. Thus, the aforementioned
   * manual caching is not needed here.
   */

  if (context.transposed_) {
    // 设置反卷积操作的参数并调整内存布局
    setup_status = xnn_reshape_deconvolution2d_nhwc_f32(
      context.op.get(),
      padded_input_nhwc.size(Layout::Activation4D::batch),   // batch_size
      padded_input_nhwc.size(Layout::Activation4D::height),  // input_height
      padded_input_nhwc.size(Layout::Activation4D::width),   // input_width
      context.output_padding_[0],                            // adjustment_height
      context.output_padding_[1],                            // adjustment_width
      nullptr,                                               // output_height_out
      nullptr,                                               // output_width_out
      caffe2::pthreadpool_());                               // threadpool

    // 配置反卷积操作，设置输入和输出张量
    setup_status = xnn_setup_deconvolution2d_nhwc_f32(
      context.op.get(),                                      // operator
      padded_input_nhwc.data_ptr<float>(),                   // input
      output.data_ptr<float>());                             // output
  } else {
    size_t workspace_size = SIZE_MAX;
    size_t workspace_alignment = SIZE_MAX;
    // 调用 XNN 库中的函数设置 NHWC 格式的二维卷积操作
    setup_status = xnn_reshape_convolution2d_nhwc_f32(
      context.op.get(),
      padded_input_nhwc.size(Layout::Activation4D::batch),   // 获取输入的批大小
      padded_input_nhwc.size(Layout::Activation4D::height),  // 获取输入的高度
      padded_input_nhwc.size(Layout::Activation4D::width),   // 获取输入的宽度
      &workspace_size,                                       // 输出工作空间大小
      &workspace_alignment,                                  // 输出工作空间对齐方式
      nullptr,                                               // 输出高度（未使用，设为 nullptr）
      nullptr,                                               // 输出宽度（未使用，设为 nullptr）
      caffe2::pthreadpool_());                               // 使用的线程池类型

    // 调用 XNN 库中的函数设置 NHWC 格式的二维卷积操作
    setup_status = xnn_setup_convolution2d_nhwc_f32(
      context.op.get(),                                      // 操作符
      nullptr,                                               // 工作空间（未使用，设为 nullptr）
      padded_input_nhwc.data_ptr<float>(),                   // 输入数据指针
      output.data_ptr<float>());                             // 输出数据指针

  }

  // 检查设置操作是否成功，若失败则抛出异常并提示是卷积还是反卷积操作失败
  TORCH_CHECK(
      xnn_status_success == setup_status,
      (context.transposed_ ? "xnn_setup_deconvolution2d_nhwc_f32 failed!"
                            : "xnn_setup_convolution2d_nhwc_f32 failed!"));

  // 运行之前设置好的操作
  const xnn_status run_status = xnn_run_operator(
      context.op.get(),         // 操作符
      caffe2::pthreadpool_());  // 使用的线程池类型

  // 内部断言，检查运行操作是否成功，若失败则抛出异常
  TORCH_INTERNAL_ASSERT(
      xnn_status_success == run_status,
      "xnn_run_operator failed!");

  // 返回连续存储（contiguous）的输出数据，建议使用与输入相同的内存格式
  return output.contiguous(input.suggest_memory_format());
} // namespace convolution2d
} // namespace internal

bool use_convolution2d(
    const Tensor& input,
    const Tensor& weight,
    const at::OptionalIntArrayRef bias_sizes_opt,
    const IntArrayRef padding,
    const IntArrayRef stride,
    const IntArrayRef dilation,
    const int64_t groups,
    const bool transposed) {
  // 检查给定的卷积参数是否可用，并确定是否可以使用卷积操作
  return internal::convolution2d::available(
            weight,
            bias_sizes_opt,
            padding,
            stride,
            dilation,
            groups,
            transposed,
            ContextConv2D::kMin,
            ContextConv2D::kMax) &&
         internal::convolution2d::usable(input);
}



Tensor conv2d_clamp_run(
    const Tensor& input,
    const c10::intrusive_ptr<xnnpack::Conv2dOpContext>& op_context) {
  // 运行给定的 Conv2dOpContext 对象来执行卷积操作
  return op_context->run(input);
}



IValue
unpack_prepacked_sizes_conv2d(const IValue& ivalue) {
  // 将输入的 IValue 转换为 xnnpack::Conv2dOpContext 类型的自定义类对象
  auto op_context = ivalue.toCustomClass<xnnpack::Conv2dOpContext>();
  // 调用 OpContext 的 unpack 方法，返回解压后的元组
  const auto tuple = op_context->unpack();
  // 提取元组中的偏置项
  const auto& bias = std::get<1>(tuple);
  // 构造新的 IValue，包含解压后的尺寸信息及偏置信息（如果存在）
  return IValue(std::make_tuple(
      std::get<0>(tuple).sizes(),
      (bias && bias->defined()) ? at::OptionalIntArrayRef(bias->sizes()) : c10::nullopt,
      std::get<2>(tuple),
      std::get<3>(tuple),
      std::get<4>(tuple),
      std::get<5>(tuple)));
}



Tensor conv2d_transpose_clamp_run(
    const Tensor& input,
    const c10::intrusive_ptr<xnnpack::TransposeConv2dOpContext>& op_context) {
  // 运行给定的 TransposeConv2dOpContext 对象来执行转置卷积操作
  return op_context->run(input);
}



c10::intrusive_ptr<xnnpack::Conv2dOpContext>
    createConv2dClampPrePackOpContext(
        Tensor weight,
        std::optional<Tensor> bias,
        std::vector<int64_t> stride,
        std::vector<int64_t> padding,
        std::vector<int64_t> dilation,
        int64_t groups,
        const std::optional<Scalar>& output_min,
        const std::optional<Scalar>& output_max) {
  // 调用 XNNPackConv2dOpContext 的静态方法创建一个预打包的 Conv2dOpContext 对象
  return xnnpack::XNNPackConv2dOpContext::create_context(
      std::move(weight),
      std::move(bias),
      std::move(padding),
      std::move(stride),
      std::move(dilation),
      groups,
      output_min,
      output_max);
}



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
        const std::optional<Scalar>& output_max) {
  // 调用 XNNPackTransposeConv2dOpContext 的静态方法创建一个预打包的 TransposeConv2dOpContext 对象
  return xnnpack::XNNPackTransposeConv2dOpContext::create_context(
      std::move(weight),
      std::move(bias),
      std::move(padding),
      std::move(output_padding),
      std::move(stride),
      std::move(dilation),
      groups,
      output_min,
      output_max);
}
// 定义一个名为 convolution2d 的函数，用于执行二维卷积操作
Tensor convolution2d(
    const Tensor& input,         // 输入张量
    const Tensor& weight,        // 卷积核张量
    const Tensor& bias,          // 偏置张量
    const IntArrayRef padding,   // 填充参数数组
    const IntArrayRef stride,    // 步幅参数数组
    const IntArrayRef dilation,  // 膨胀参数数组
    const int64_t groups) {      // 分组数

  // 调用内部函数，创建并执行二维卷积操作
  return internal::convolution2d::create_and_run(
      input,                     // 输入张量
      weight,                    // 卷积核张量
      bias,                      // 偏置张量
      padding,                   // 填充参数数组
      {0, 0},                    // 输出填充
      stride,                    // 步幅参数数组
      dilation,                  // 膨胀参数数组
      groups,                    // 分组数
      false,                     // 非转置卷积
      ContextConv2D::kMin,       // 卷积上下文最小值
      ContextConv2D::kMax);      // 卷积上下文最大值
}

} // namespace at::native::xnnpack

#endif /* USE_XNNPACK */
```