# `.\pytorch\aten\src\ATen\native\mkldnn\ConvPrepack.cpp`

```
// 包含必要的头文件
#include <vector>

#include <ATen/native/ConvUtils.h>
#include <ATen/native/mkldnn/Common.h>
#include <ATen/native/mkldnn/ConvPrepack.h>
#include <ATen/native/mkldnn/MKLDNNCommon.h>
#include <ATen/native/mkldnn/OpContext.h>
#include <ATen/native/utils/Factory.h>
#include <ATen/native/utils/ParamUtils.h>
#include <c10/util/irange.h>

// 如果使用了 MKLDNN，才进入该命名空间
#if AT_MKLDNN_ENABLED()

namespace at {
namespace native {
namespace mkldnn {
namespace internal {
namespace convolution {

// 创建预打包的卷积操作上下文
c10::intrusive_ptr<mkldnn::ConvOpContext> createConvPrePackOpContext(
    Tensor weight,                             // 卷积权重张量
    std::optional<Tensor> bias,                // 可选的卷积偏置张量
    std::vector<int64_t> stride,               // 卷积步长
    std::vector<int64_t> padding,              // 卷积填充
    std::vector<int64_t> dilation,             // 卷积扩展
    int64_t groups,                            // 卷积分组
    std::vector<int64_t> input_size,           // 输入张量大小
    std::string attr) {                        // 卷积属性字符串
  // 查找给定属性在映射中的位置
  auto it = fusion_attr_map.find(attr);
  // 检查属性是否存在，如果不存在则抛出异常
  TORCH_CHECK(it != fusion_attr_map.end(), "Fusion behavior undefined.");
  // 获取属性对应的操作属性
  ideep::attr_t op_attr = it->second;

  // 创建 MKLDNN 卷积操作上下文，并返回指针
  return mkldnn::MkldnnConvOpContext::create_context(
      std::move(weight),       // 移动权重张量
      std::move(bias),         // 移动偏置张量
      std::move(padding),      // 移动填充信息
      std::move(stride),       // 移动步长信息
      std::move(dilation),     // 移动扩展信息
      groups,                  // 卷积分组数
      std::move(input_size),   // 移动输入大小信息
      op_attr);                // 操作属性
}

// 创建上下文卷积操作
ContextConv create(
    const Tensor& weight,                      // 卷积权重张量
    const std::optional<Tensor>& bias,         // 可选的卷积偏置张量
    const IntArrayRef padding,                 // 卷积填充
    const IntArrayRef stride,                  // 卷积步长
    const IntArrayRef dilation,                // 卷积扩展
    const int64_t groups,                      // 卷积分组
    const IntArrayRef input_size,


继续注释下面的部分...
    // 定义一个函数，接受权重张量和一些卷积参数，返回一个 ContextConv 结构体
    const ideep::attr_t& attr) {
      // 获取权重张量的维度数
      auto k = weight.ndimension();
      // 计算张量的空间维度数，通常是 k 减去两个，这是标准卷积的做法
      int64_t dim = k - 2;
      // 根据需要扩展填充参数，并确保维度为 dim
      const auto padding_expanded = expand_param_if_needed(padding, "padding", dim);
      // 根据需要扩展步幅参数，并确保维度为 dim
      const auto stride_expanded = expand_param_if_needed(stride, "stride", dim);
      // 根据需要扩展膨胀参数，并确保维度为 dim
      const auto dilation_expanded =
          expand_param_if_needed(dilation, "dilation", dim);
      // 根据需要扩展输入大小参数，并确保维度为 k
      const auto input_size_expanded =
          expand_param_if_needed(input_size, "input_size", k);
    
      // 进入自动求导模式的排除分发键保护
      c10::impl::ExcludeDispatchKeyGuard edkg(c10::autograd_dispatch_keyset);
      // 将稠密权重张量转换为 IDeep 的张量视图
      auto w = itensor_view_from_dense(weight);
      // 如果输入是 nhwc 而 w 是 nchw，则输出警告信息（TODO：如果输入是 nhwc 而权重张量是 nchw 的情况应该如何处理）
      bool is_channels_last =
          weight.suggest_memory_format() == at::MemoryFormat::ChannelsLast;
      // 获取期望的权重描述符，用于卷积操作
      ideep::tensor::desc expected_weight_desc =
          ideep::convolution_forward::expected_weights_desc(
              w.get_dims(),
              w.get_data_type(),
              {stride_expanded.begin(), stride_expanded.end()},
              {padding_expanded.begin(), padding_expanded.end()},
              {padding_expanded.begin(), padding_expanded.end()},
              {dilation_expanded.begin(), dilation_expanded.end()},
              groups,
              ideep::algorithm::convolution_direct,
              ideep::prop_kind::forward,
              /*x_dtype*/ w.get_data_type(),
              {input_size_expanded.begin(), input_size_expanded.end()},
              attr,
              is_channels_last);
    
      // 初始化经打包的权重张量
      ideep::tensor packed_weight;
      packed_weight.init(expected_weight_desc);
      // 将 w 的数据填充到打包的权重张量中
      packed_weight.feed_from(w);
    
      // 返回 ContextConv 结构体，其中包括打包后的权重、偏置（如果存在）、填充、步幅、膨胀、分组和属性
      return ContextConv{
          std::move(packed_weight),
          bias.has_value() ? c10::make_optional(*bias) : c10::nullopt,
          {padding_expanded.begin(), padding_expanded.end()},
          {stride_expanded.begin(), stride_expanded.end()},
          {dilation_expanded.begin(), dilation_expanded.end()},
          groups,
          attr};
    }
}

static void _mkldnn_convolution_out(
    const ideep::tensor& x,  // 输入张量 x，作为卷积的输入
    ideep::tensor& y,        // 输出张量 y，接收卷积操作的结果
    const ideep::tensor& w,  // 卷积核张量 w，用于卷积操作
    const std::optional<ideep::tensor>& b,  // 可选的偏置张量 b，用于卷积操作
    IntArrayRef padding,      // 填充参数，用于卷积操作
    IntArrayRef stride,       // 步长参数，用于卷积操作
    IntArrayRef dilation,     // 膨胀参数，用于卷积操作
    IntArrayRef output_sizes, // 输出尺寸参数，用于卷积操作
    int64_t groups,           // 分组数，用于卷积操作
    const ideep::attr_t& attr = ideep::attr_t()) {  // 属性对象，用于卷积操作的额外属性
  if (b.has_value()) {
    ideep::convolution_forward::compute_v2(
        x,                            // 输入张量 x
        w,                            // 卷积核张量 w
        b.value(),                    // 偏置张量 b
        {output_sizes.cbegin(), output_sizes.cend()},  // 输出尺寸
        y,                            // 输出张量 y
        {stride.begin(), stride.end()},              // 步长
        {dilation.begin(), dilation.end()},          // 膨胀
        {padding.begin(), padding.end()},            // 填充
        {padding.begin(), padding.end()},            // 填充
        groups,                       // 分组数
        ideep::scale_t(),            // 规模因子 (scale factor)
        ideep::scale_t(),            // 规模因子 (scale factor)
        ideep::scale_t(),            // 规模因子 (scale factor)
        ideep::zero_point_t(),       // 零点 (zero point)
        ideep::zero_point_t(),       // 零点 (zero point)
        attr);                       // 属性对象
  } else {
    ideep::convolution_forward::compute_v2(
        x,                            // 输入张量 x
        w,                            // 卷积核张量 w
        {output_sizes.cbegin(), output_sizes.cend()},  // 输出尺寸
        y,                            // 输出张量 y
        {stride.begin(), stride.end()},              // 步长
        {dilation.begin(), dilation.end()},          // 膨胀
        {padding.begin(), padding.end()},            // 填充
        {padding.begin(), padding.end()},            // 填充
        groups,                       // 分组数
        ideep::scale_t(),            // 规模因子 (scale factor)
        ideep::scale_t(),            // 规模因子 (scale factor)
        ideep::scale_t(),            // 规模因子 (scale factor)
        ideep::zero_point_t(),       // 零点 (zero point)
        ideep::zero_point_t(),       // 零点 (zero point)
        attr);                       // 属性对象
  }
}

static void mkldnn_convolution_out(
    const Tensor& input,          // 输入 PyTorch 张量 input
    ideep::tensor& mkldnn_output, // 输出的 IDEEP 张量 mkldnn_output
    const ideep::tensor& mkldnn_weight,  // 卷积核 IDEEP 张量 mkldnn_weight
    const std::optional<Tensor>& bias_opt,  // 可选的偏置张量 bias_opt
    IntArrayRef padding,          // 填充参数
    IntArrayRef stride,           // 步长参数
    IntArrayRef dilation,         // 膨胀参数
    IntArrayRef output_sizes,     // 输出尺寸参数
    int64_t groups,               // 分组数
    const ideep::attr_t& attr = ideep::attr_t()) {  // 属性对象，默认为空属性
  c10::MaybeOwned<Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias_opt);  // 从可选的 Tensor 中获取偏置
  const Tensor& bias = *bias_maybe_owned;         // 获取偏置张量

  c10::impl::ExcludeDispatchKeyGuard edkg(c10::autograd_dispatch_keyset);  // 排除自动求导分发键集合

  const ideep::tensor mkldnn_input = itensor_from_tensor(input);  // 将 PyTorch 张量转换为 IDEEP 张量
  std::optional<ideep::tensor> mkldnn_bias{c10::nullopt};         // 可选的 IDEEP 偏置张量，初始为空
  if (bias.defined()) {
    mkldnn_bias = itensor_from_tensor(bias);  // 如果偏置张量已定义，则转换为 IDEEP 张量
  }

  _mkldnn_convolution_out(
      mkldnn_input,         // 输入 IDEEP 张量
      mkldnn_output,        // 输出 IDEEP 张量
      mkldnn_weight,        // 卷积核 IDEEP 张量
      mkldnn_bias,          // 可选的偏置 IDEEP 张量
      padding,              // 填充参数
      stride,               // 步长参数
      dilation,             // 膨胀参数
      output_sizes,         // 输出尺寸参数
      groups,               // 分组数
      attr);                // 属性对象
}

static std::vector<int64_t> get_output_sizes(
    ContextConv& context,     // 卷积上下文对象
    const Tensor& input) {    // 输入 PyTorch 张量
  const ideep::tensor& mkldnn_weight = context.weight_packed_;  // 获取打包后的卷积核 IDEEP 张量
  IntArrayRef padding = context.padding_;                        // 获取填充参数
  IntArrayRef stride = context.stride_;                          // 获取步长参数
  IntArrayRef dilation = context.dilation_;                      // 获取膨胀参数

  auto kernel_size = mkldnn_weight.get_dims();  // 获取卷积核尺寸

  std::vector<int64_t> input_size = input.sizes().vec();  // 获取输入张量尺寸
  return conv_output_size(input_size, kernel_size, padding, stride, dilation);  // 计算卷积输出尺寸
}
Tensor run(ContextConv& context, const Tensor& input) {
  // 获取输出大小的向量
  std::vector<int64_t> output_sizes = get_output_sizes(context, input);
  // 创建一个空的输出 Tensor，根据输入的内存格式选择合适的内存格式
  auto output = at::empty(
      output_sizes,
      input.options().memory_format(input.suggest_memory_format()));

  // 检查输入的内存格式是否为 ChannelsLast
  bool is_channels_last =
      input.suggest_memory_format() == at::MemoryFormat::ChannelsLast;
  // 创建一个 ideep::tensor 对象
  ideep::tensor y;

  // 排除自动求导的调度键，使用 autograd_dispatch_keyset
  c10::impl::ExcludeDispatchKeyGuard edkg(c10::autograd_dispatch_keyset);
  // 将 output 转换为 ideep::tensor 类型
  ideep::tensor mkldnn_output = itensor_from_tensor(output);

  // 根据输入的内存格式选择执行不同的操作
  if (is_channels_last) {
    // 执行 MKL-DNN 卷积操作，输出到 mkldnn_output
    mkldnn_convolution_out(
        input,
        mkldnn_output,
        context.weight_packed_,
        context.at_bias_,
        context.padding_,
        context.stride_,
        context.dilation_,
        output_sizes,
        context.groups_,
        context.attr_);
  } else {
    // 执行 MKL-DNN 卷积操作，输出到 y
    mkldnn_convolution_out(
        input,
        y,
        context.weight_packed_,
        context.at_bias_,
        context.padding_,
        context.stride_,
        context.dilation_,
        output_sizes,
        context.groups_,
        context.attr_);
    // 将 y 的数据填充到 mkldnn_output 中
    mkldnn_output.feed_from(y);
  }
  // 返回输出 Tensor
  return output;
}

void run(ContextConv& context, const Tensor& input, void* output) {
  // 获取输出大小的向量
  std::vector<int64_t> output_sizes = get_output_sizes(context, input);

  // 检查输入的内存格式是否为 ChannelsLast
  bool is_channels_last =
      input.suggest_memory_format() == at::MemoryFormat::ChannelsLast;
  // 创建一个 ideep::tensor 对象
  ideep::tensor y;

  // 根据输入的内存格式选择输出的 ideep::tag
  ideep::tag o_tag = is_channels_last ? ideep::tag::nhwc : ideep::tag::nchw;
  // 创建输出的 ideep::tensor::desc 描述对象
  ideep::tensor::desc o_desc = {
      output_sizes, get_mkldnn_dtype(input.scalar_type()), o_tag};
  // 使用给定的输出地址创建 ideep::tensor 对象
  ideep::tensor mkldnn_output = {o_desc, output};

  // 根据输入的内存格式选择执行不同的操作
  if (is_channels_last) {
    // 执行 MKL-DNN 卷积操作，输出到 mkldnn_output
    mkldnn_convolution_out(
        input,
        mkldnn_output,
        context.weight_packed_,
        context.at_bias_,
        context.padding_,
        context.stride_,
        context.dilation_,
        output_sizes,
        context.groups_,
        context.attr_);
  } else {
    // 执行 MKL-DNN 卷积操作，输出到 y
    mkldnn_convolution_out(
        input,
        y,
        context.weight_packed_,
        context.at_bias_,
        context.padding_,
        context.stride_,
        context.dilation_,
        output_sizes,
        context.groups_,
        context.attr_);
    // 将 y 的数据填充到 mkldnn_output 中
    mkldnn_output.feed_from(y);
  }
}

Tensor conv_run(
    const Tensor& input,
    const c10::intrusive_ptr<mkldnn::ConvOpContext>& op_context) {
  // 调用传入的 op_context 对象的 run 方法，并返回结果
  return op_context->run(input);
}
```