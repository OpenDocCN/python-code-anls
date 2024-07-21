# `.\pytorch\aten\src\ATen\native\mkldnn\Conv.cpp`

```py
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Config.h>
#include <torch/library.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/native/utils/ParamUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#include <ATen/Functions.h>
#else
#include <ATen/ops/_add_relu_native.h>
#include <ATen/ops/_to_dense_native.h>
#include <ATen/ops/convolution.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/mkldnn_convolution_native.h>
#endif

#if !AT_MKLDNN_ENABLED()

// 在没有启用 MKLDNN 支持时定义的命名空间
namespace at { namespace native {

// 当调用 mkldnn_convolution 函数时，抛出错误信息，提示 ATen 未编译支持 MKLDNN
Tensor mkldnn_convolution(
    const Tensor& input, const Tensor& weight, const std::optional<Tensor>& bias_opt,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups) {
  TORCH_CHECK(false, "mkldnn_convolution_forward: ATen not compiled with MKLDNN support");
}

// 注册函数桩，用于避免 CPU 分发的注册
REGISTER_NO_CPU_DISPATCH(mkldnn_convolution_backward_stub);
REGISTER_NO_CPU_DISPATCH(mkldnn_convolution_transpose_stub);
REGISTER_NO_CPU_DISPATCH(mkldnn_convolution_transpose_backward_stub);

}}

#else // AT_MKLDNN_ENABLED

#include <ATen/native/mkldnn/MKLDNNCommon.h>
#include <ATen/native/mkldnn/Utils.h>
#include <ATen/native/ConvUtils.h>
#include <c10/util/irange.h>

// 当启用 MKLDNN 支持时，定义的命名空间和函数

namespace at { namespace native {

// 按照 native/Convolution.cpp 中的规则检查前向传播的形状，不支持转置
static void check_shape_forward(const Tensor& input,
                                const Tensor& weight,
                                const Tensor& bias,
                                const IntArrayRef& padding,
                                const IntArrayRef& stride,
                                const IntArrayRef& dilation,
                                const int64_t groups) {
#define MKLDNN_CONV_ARG_CHECK(IT, OP) std::any_of(IT.begin(), IT.end(), [](auto x) { return x OP 0; })
  // 检查填充参数是否有负值
  auto is_padding_neg = MKLDNN_CONV_ARG_CHECK(padding, <);
  // 检查步幅参数是否有非正值
  auto is_stride_nonpos = MKLDNN_CONV_ARG_CHECK(stride, <=);
  // 检查扩展参数是否有非正值
  auto is_dilation_nonpos = MKLDNN_CONV_ARG_CHECK(dilation, <=);
#undef MKLDNN_CONV_ARG_CHECK
// 取消定义 MKLDNN_CONV_ARG_CHECK 宏

TORCH_CHECK(!is_padding_neg, "negative padding is not supported");
// 检查是否存在负的 padding 值，如果有则抛出错误

TORCH_CHECK(!is_stride_nonpos, "non-positive stride is not supported");
// 检查是否存在非正的 stride 值，如果有则抛出错误

TORCH_CHECK(!is_dilation_nonpos, "non-positive dilation is not supported");
// 检查是否存在非正的 dilation 值，如果有则抛出错误

TORCH_CHECK(groups > 0, "non-positive groups is not supported");
// 检查 groups 是否大于 0，如果不是则抛出错误

int64_t k = input.ndimension();
// 获取输入张量的维度数

const IntArrayRef& weight_sizes = weight.sizes();
// 获取权重张量的大小信息

int64_t weight_dim = weight_sizes.size();
// 获取权重张量的维度数

TORCH_CHECK(weight_dim == k,
            "Expected ", weight_dim, "-dimensional input for ", weight_dim,
            "-dimensional weight ", weight_sizes, ", but got ", k, "-dimensional input of size ",
            input.sizes(), " instead");
// 检查输入张量的维度数是否与权重张量的维度数相同，如果不同则抛出错误

TORCH_CHECK(weight_sizes[0] >= groups,
            "Given groups=", groups, ", expected weight to be at least ", groups,
            " at dimension 0, but got weight of size ", weight_sizes, " instead");
// 检查权重张量的第一个维度大小是否至少为 groups，如果不是则抛出错误

TORCH_CHECK(weight_sizes[0] % groups == 0,
            "Given groups=", groups, ", expected weight to be divisible by ",
            groups, " at dimension 0, but got weight of size [", weight_sizes,
            "] instead");
// 检查权重张量的第一个维度大小是否能被 groups 整除，如果不能则抛出错误

TORCH_CHECK(input.size(1) == (weight_sizes[1] * groups),
            "Given groups=", groups, ", weight of size ", weight_sizes,
            ", expected input", input.sizes(), " to have ",
            (weight_sizes[1] * groups), " channels, but got ", input.size(1),
            " channels instead");
// 检查输入张量的第二个维度（通道数）是否与权重张量的第二个维度乘以 groups 相等，如果不相等则抛出错误

TORCH_CHECK(!bias.defined() || (bias.ndimension() == 1 && bias.size(0) == weight_sizes[0]),
            "Given weight of size ", weight_sizes,
            ", expected bias to be 1-dimensional with ", weight_sizes[0], " elements",
            ", but got bias of size ", bias.sizes(), " instead");
// 检查是否定义了偏置项 bias，并且偏置项的维度为1且大小与权重张量的第一个维度大小相等，如果不符合则抛出错误

std::vector<int64_t> input_shape;
// 定义存储输入形状信息的向量

std::vector<int64_t> kernel_shape;
// 定义存储卷积核形状信息的向量

bool kernel_size_correct = true;
// 标志变量，表示卷积核大小是否正确，默认为 true

for (const auto i : c10::irange(2, k)) {
  input_shape.push_back(input.size(i) + 2 * padding[i-2]);
  // 计算经过 padding 后的每个维度的输入形状

  // 计算考虑 dilation 后的每个维度的卷积核形状，并记录日志
  kernel_shape.push_back(dilation[i-2] * (weight_sizes[i]-1) + 1);

  if (input_shape.back() < kernel_shape.back()) {
    kernel_size_correct = false;
  }
  // 如果任何一个维度的输入形状小于对应的卷积核形状，则将 kernel_size_correct 置为 false
}

TORCH_CHECK(input_shape.size() == kernel_shape.size(), "Inconsistent shape between Input and Kernel");
// 检查输入形状向量和卷积核形状向量的大小是否相同，如果不同则抛出错误

if (!kernel_size_correct) {
  // 如果卷积核大小不正确
  std::ostringstream input_ss;
  std::ostringstream kernel_ss;
  std::string separator = "";

  for (int i = 0, len = input_shape.size(); i < len; ++i) {
    input_ss << separator << input_shape[i];
    kernel_ss << separator << kernel_shape[i];
    separator = " x ";
  }

  TORCH_CHECK(false, "Calculated padded input size per channel: (", input_ss.str(), "). "
              "Kernel size: (", kernel_ss.str(), "). Kernel size can't be greater than actual input size");
  // 构造错误信息，指出每个通道的填充后的输入大小和卷积核大小之间的不匹配，并抛出错误
}
// 定义宏 `MKLDNNTensor`，用于创建 MKLDNN 张量并指定选项
#define MKLDNNTensor(itensor, options)                                  \
  new_with_itensor_mkldnn(                                              \
      std::move(itensor),                                               \
      optTypeMetaToScalarType(options.dtype_opt()),                     \
      options.device_opt())

// 注意事项 [MKLDNN Convolution Memory Formats]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// MKLDNN 中的卷积操作有三种内存格式：
//
// 如果从 PyTorch 传递的内存格式（即用户布局）与 MKLDNN 使用的内部布局不同，需要进行 `reorder`；
// 否则，如果用户布局与内部布局相同，MKLDNN 使用现有 CPU 张量上的内存 `view`。
//
// 1. NCHW（CPU 张量，连续布局）
//    输入数据重排：NCHW（用户布局）-> Blocked（内部布局）
//    权重数据重排：OIHW（用户布局）-> Blocked（内部布局）
//    输出数据重排：Blocked（内部布局）-> NCHW（用户布局）
//
// 2. NHWC（CPU 张量，通道在最后）
//    输入数据视图：NHWC（用户布局）-> NHWC（内部布局）
//    权重数据重排：OHWI（用户布局）-> Blocked（内部布局）
//    输出数据视图：NHWC（内部布局）-> NHWC（用户布局）
//
// 3. Blocked（MKLDNN 张量）：
//    通过显式将张量转换为 mkldnn 格式，例如 `x.to_mkldnn()`，
//    Blocked 格式将在层间传播。输入和输出将采用 Blocked 格式。
//
//    对于推理情况，可以通过预打包权重为 Blocked 格式
//    （以减少权重重排开销）：model = torch.utils.mkldnn.to_mkldnn(model)
//
//    对于训练情况，grad_output 可以是 CPU 张量或 MKLDNN 张量，
//    但是权重/偏置和 grad_weight/grad_bias 总是 CPU 张量。
//

// 内联函数，用于确定 MKLDNN 卷积操作的内存格式
static inline at::MemoryFormat mkldnn_convolution_memory_format(int64_t dims, bool is_channels_last) {
   auto memory_format =  at::MemoryFormat::Contiguous;
   if (is_channels_last) {
      memory_format = dims == 4 ? at::MemoryFormat::ChannelsLast : at::MemoryFormat::ChannelsLast3d;
   }
   return memory_format;
}

// MKLDNN 卷积操作的具体实现
static void _mkldnn_convolution_out (
    const Tensor& input_t,
    const Tensor& weight_t,
    const Tensor& bias,
    std::vector<int64_t>& output_sizes,
    ideep::tensor& y,
    IntArrayRef stride,
    IntArrayRef dilation,
    IntArrayRef padding,
    int64_t groups,
    bool is_channels_last,
    const ideep::attr_t& op_attr) {
  // 确定输入数据的内存格式
  auto memory_format = mkldnn_convolution_memory_format(input_t.ndimension(), is_channels_last);
  // 如果输入数据已经是 MKLDNN 格式，则使用它；否则进行内存格式的连续化处理
  auto input = input_t.is_mkldnn() ? input_t : input_t.contiguous(memory_format);
  // 如果权重数据已经是 MKLDNN 格式，则使用它；否则进行内存格式的连续化处理
  auto weight = weight_t.is_mkldnn() ? weight_t : weight_t.contiguous(memory_format);
  // 将输入数据和权重数据转换为 ideep::tensor 类型
  const ideep::tensor x = itensor_from_tensor(input, /*from_const_data_ptr*/true);
  const ideep::tensor w = itensor_from_tensor(weight, /*from_const_data_ptr*/true);
  // 如果定义了偏置，则将偏置数据转换为 ideep::tensor 类型
  if (bias.defined()) {
    const ideep::tensor b = itensor_from_tensor(bias, /*from_const_data_ptr*/true);
    // 按照给定的参数执行 MKLDNN 卷积操作，包括步长、膨胀、填充、分组和操作属性
    # 如果 b 不为空，则执行带偏置项的深度卷积前向计算
    ideep::convolution_forward::compute_v3(
        x,                              # 输入张量 x
        w,                              # 卷积核张量 w
        b,                              # 偏置张量 b
        {output_sizes.cbegin(), output_sizes.cend()},  # 输出大小的范围
        y,                              # 输出张量 y
        {stride.begin(), stride.end()},  # 步幅的范围
        {dilation.begin(), dilation.end()},  # 膨胀率的范围
        {padding.begin(), padding.end()},    # 填充的范围
        {padding.begin(), padding.end()},    # 填充的范围（第二次）
        groups,                         # 卷积操作的分组数
        is_channels_last,               # 是否为通道在后的布局
        op_attr                         # 操作属性
    );
  } else {
    # 如果 b 为空，则执行不带偏置项的深度卷积前向计算
    ideep::convolution_forward::compute_v3(
        x,                              # 输入张量 x
        w,                              # 卷积核张量 w
        {output_sizes.cbegin(), output_sizes.cend()},  # 输出大小的范围
        y,                              # 输出张量 y
        {stride.begin(), stride.end()},  # 步幅的范围
        {dilation.begin(), dilation.end()},  # 膨胀率的范围
        {padding.begin(), padding.end()},    # 填充的范围
        {padding.begin(), padding.end()},    # 填充的范围（第二次）
        groups,                         # 卷积操作的分组数
        is_channels_last,               # 是否为通道在后的布局
        op_attr                         # 操作属性
    );
  }
// 定义了一个静态函数 _mkldnn_convolution，用于执行 MKL-DNN 卷积操作
static Tensor _mkldnn_convolution(
    // 输入张量
    const Tensor& input_t,
    // 卷积核张量
    const Tensor& weight_t,
    // 可选的偏置张量
    const std::optional<Tensor>& bias_opt,
    // 填充参数
    IntArrayRef padding,
    // 步幅参数
    IntArrayRef stride,
    // 膨胀参数
    IntArrayRef dilation,
    // 分组数
    int64_t groups,
    // 是否使用通道为最后一个维度
    bool use_channels_last,
    // 属性字符串，默认为 "none"
    c10::string_view attr = "none",
    // 标量值列表
    torch::List<std::optional<at::Scalar>> scalars =
        torch::List<std::optional<at::Scalar>>(),
    // 算法类型，可选
    std::optional<c10::string_view> algorithm = c10::nullopt) {
  
  // 初始化操作属性为默认值
  ideep::attr_t op_attr = ideep::attr_t();
  
  // 如果属性不为 "none"，根据属性查找对应的融合操作属性
  if (attr != "none") {
    auto it = fusion_unary_attr_map().find(attr);
    // 检查是否找到对应的属性映射，否则报错
    TORCH_CHECK(
        it != fusion_unary_attr_map().end(), "Fusion behavior undefined.");
    // 调用属性映射函数生成操作属性
    op_attr = it->second(scalars, algorithm);
  }
  
  // 使用 at::borrow_from_optional_tensor 函数从可选的 bias_opt 中获取可能拥有的 Tensor 引用
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  // 将获取到的偏置 Tensor 引用赋值给常量引用 bias
  const Tensor& bias = *bias_maybe_owned;
  
  // 检查输入张量的低精度情况，用于 MKL-DNN 卷积
  mkldnn_check_low_precision(input_t.scalar_type(), "mkldnn_convolution");

  // 计算输入张量的维度，排除批量和通道维度
  int64_t dim = input_t.ndimension() - 2;
  
  // 根据需要扩展填充参数、步幅参数和膨胀参数的维度
  const auto padding_expanded = expand_param_if_needed(padding, "padding", dim);
  const auto stride_expanded = expand_param_if_needed(stride, "stride", dim);
  const auto dilation_expanded = expand_param_if_needed(dilation, "dilation", dim);
  
  // 检查输入张量、卷积核张量和偏置的形状
  check_shape_forward(input_t, weight_t, bias, padding_expanded, stride_expanded, dilation_expanded, groups);
  
  // 确定 MKL-DNN 卷积操作的内存格式
  auto memory_format =
      mkldnn_convolution_memory_format(input_t.ndimension(), use_channels_last);
  
  // 计算卷积的输出大小
  auto output_sizes = conv_output_size(input_t.sizes(), weight_t.sizes(), padding_expanded, stride_expanded, dilation_expanded);
  
  // 根据输入张量的选项创建一个空的输出张量
  auto output = at::empty({0}, input_t.options());
  
  // 定义一个 ideep::tensor 对象 y
  ideep::tensor y;
  
  // 如果使用通道为最后一个维度，重新调整输出张量的大小和内存格式
  if (use_channels_last) {
    output.resize_(output_sizes, memory_format);
    y = itensor_from_tensor(output);
  }
  
  // 调用 _mkldnn_convolution_out 函数执行 MKL-DNN 卷积操作
  _mkldnn_convolution_out(
      input_t,
      weight_t,
      bias,
      output_sizes,
      y,
      stride_expanded,
      dilation_expanded,
      padding_expanded,
      groups,
      use_channels_last,
      op_attr);
  
  // 如果输入张量是 MKL-DNN 张量，则返回相应的 MKLDNNTensor
  if (input_t.is_mkldnn()) {
    return MKLDNNTensor(y, input_t.options());
  } else if (!use_channels_last) {
    // 如果不使用通道为最后一个维度，则将 MKL-DNN 张量转换为稠密张量
    return mkldnn_to_dense(MKLDNNTensor(y, input_t.options()));
  } else {
    // 否则直接返回输出张量
    return output;
  }
}

// 定义了一个公共接口 mkldnn_convolution，用于执行 MKL-DNN 卷积操作
Tensor mkldnn_convolution(
    // 输入张量
    const Tensor& input_t,
    // 卷积核张量
    const Tensor& weight_t,
    // 可选的偏置张量
    const std::optional<Tensor>& bias_opt,
    // 填充参数
    IntArrayRef padding,
    // 步幅参数
    IntArrayRef stride,
    // 膨胀参数
    IntArrayRef dilation,
    // 分组数
    int64_t groups) {
  
  // 根据输入和卷积核张量判断是否使用通道为最后一个维度
  bool use_channels_last = mkldnn_conv_use_channels_last(input_t, weight_t);
  
  // 调用 _mkldnn_convolution 执行实际的 MKL-DNN 卷积操作，并返回结果
  return _mkldnn_convolution(
      input_t,
      weight_t,
      bias_opt,
      padding,
      stride,
      dilation,
      groups,
      use_channels_last);
}

// 定义了一个匿名命名空间，内部实现了 pointwise MKL-DNN 卷积操作
namespace {
Tensor mkldnn_convolution_pointwise(
    // 输入张量
    const Tensor& input_t,
    // 卷积核张量
    const Tensor& weight_t,
    // 可选的偏置张量
    const std::optional<Tensor>& bias_opt,
    // 填充参数
    IntArrayRef padding,
    // 步幅参数
    IntArrayRef stride,
    // 膨胀参数
    IntArrayRef dilation,
    // 分组数
    int64_t groups,
    // 指定的属性字符串
    c10::string_view attr,
    // 算法类型，可选
    std::optional<c10::string_view> algorithm = c10::nullopt) {
    // 创建 ExcludeDispatchKeyGuard 对象，确保在 autograd_dispatch_keyset 上排除分发键
    c10::impl::ExcludeDispatchKeyGuard edkg(c10::autograd_dispatch_keyset);
    // 检查是否使用 channels_last 布局，条件为 weight_t 是 mkldnn 类型或者 mkldnn_conv_use_channels_last 返回 true
    bool use_channels_last =
        weight_t.is_mkldnn() || mkldnn_conv_use_channels_last(input_t, weight_t);
    // 调用 _mkldnn_convolution 函数执行 MKL-DNN 卷积操作，并返回结果
    return _mkldnn_convolution(
        input_t,             // 输入张量
        weight_t,            // 权重张量
        bias_opt,            // 可选的偏置张量
        padding,             // 填充大小
        stride,              // 步幅大小
        dilation,            // 膨胀大小
        groups,              // 分组数
        use_channels_last,   // 是否使用 channels_last 布局
        attr,                // 附加属性
        scalars,             // 标量列表
        algorithm);          // 可选的算法
}

// 合并卷积、二元操作和一元操作以获得良好的性能，其操作为：output=unary_op(binary_op(conv(input_t, ...), other_t, alpha))。
// binary_attr 表示二元操作类型，可以是 "add" 或其他二元操作。unary_attr 表示一元操作类型，可以是 "relu" 或其他一元操作；如果为空，则表示没有一元操作。unary_scalars 和 unary_algorithm 是一元操作的参数，例如 "hardtanh" 具有标量参数，"gelu" 具有算法参数。
Tensor mkldnn_convolution_pointwise_binary(
    const Tensor& input_t,                     // 输入张量
    const Tensor& other_t,                     // 第二个输入张量
    const Tensor& weight_t,                    // 权重张量
    const std::optional<Tensor>& bias_opt,     // 可选的偏置张量
    IntArrayRef padding,                       // 填充参数
    IntArrayRef stride,                        // 步幅参数
    IntArrayRef dilation,                      // 膨胀参数
    int64_t groups,                            // 分组数
    c10::string_view binary_attr,              // 二元操作类型
    std::optional<at::Scalar> alpha,           // 可选的 alpha 参数
    std::optional<c10::string_view> unary_attr, // 可选的一元操作类型
    torch::List<std::optional<at::Scalar>> unary_scalars, // 一元操作的标量参数列表
    std::optional<c10::string_view> unary_algorithm        // 可选的一元操作算法
) {
  // 检查输入张量维度是否为 4 或 5
  TORCH_CHECK(
      input_t.ndimension() == 4 || input_t.ndimension() == 5,
      "mkldnn_convolution_pointwise_binary: currently only support 2d and 3d")

  // 检查 alpha 值是否为 1.0 或未指定
  TORCH_CHECK(
      !alpha.has_value() || alpha.value().to<float>() == 1.0,
      "mkldnn_convolution_pointwise_binary: the alpha value should be none or 1.0");

  // 获取偏置张量的引用
  c10::MaybeOwned<Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  // 确保输入具有相同的类型（设备、布局、数据类型），设备为 CPU，数据类型为 float、bfloat16 或 half
  check_mkldnn_binary_fusion_inputs(input_t, other_t, weight_t, bias);

  // 计算维度
  int64_t dim = input_t.ndimension() - 2;
  // 根据需要扩展填充参数、步幅参数和膨胀参数
  const auto padding_expanded = expand_param_if_needed(padding, "padding", dim);
  const auto stride_expanded = expand_param_if_needed(stride, "stride", dim);
  const auto dilation_expanded = expand_param_if_needed(dilation, "dilation", dim);
  // 检查输入张量、权重张量、偏置张量的形状是否匹配
  check_shape_forward(
      input_t, weight_t, bias, padding_expanded, stride_expanded, dilation_expanded, groups);

  // 计算卷积操作后的输出大小
  auto output_sizes = conv_output_size(
      input_t.sizes(), weight_t.sizes(), padding_expanded, stride_expanded, dilation_expanded);
  // TODO: 支持广播二元融合。
  // 检查输出大小是否与 other_t 张量的大小相同
  TORCH_CHECK(
      output_sizes == other_t.sizes(),
      "Binary Fusion's inputs should have same shape");

  // 只有在 channels_last 路径下才调用融合路径
  // TODO: 对于 groups > 1 的情况，OneDNN 优化不佳，将在下一个 OneDNN 发布中启用
  bool use_channels_last =
      weight_t.is_mkldnn() || mkldnn_conv_use_channels_last(input_t, weight_t);
  // 可以融合的条件为：groups == 1 且使用 channels_last
  bool can_be_fused = groups == 1 && use_channels_last;

  // 设置默认的一元操作属性值为 "none"
  c10::string_view unary_attr_value = "none";
  ideep::algorithm unary_alg;
  // 如果存在一元操作类型，则查找对应的算法
  if (unary_attr.has_value()) {
    auto it_unary = fusion_unary_alg_map().find(unary_attr.value());
    // 目前仅支持 conv+binary+relu
    // TODO: 支持更多的一元操作
    ```
    // 检查一元操作融合映射中是否存在对应操作，若不存在则抛出异常
    TORCH_CHECK(
        it_unary != fusion_unary_alg_map().end(),
        "Unary Fusion behavior undefined.");
    // 获取一元操作属性值
    unary_attr_value = unary_attr.value();
    // 获取一元操作算法
    unary_alg = it_unary->second;
  }
  // 在二元操作融合映射中查找二元操作属性
  auto it_binary = fusion_binary_alg_map().find(binary_attr);
  // 检查二元操作融合映射中是否存在对应操作，若不存在则抛出异常
  TORCH_CHECK(
      it_binary != fusion_binary_alg_map().end(),
      "Binary Fusion behavior undefined.");
  // 排除自动微分的派遣键集，确保不会影响融合操作
  c10::impl::ExcludeDispatchKeyGuard edkg(c10::autograd_dispatch_keyset);
  // 如果可以进行融合
  if (can_be_fused) {
    // 获取 MKL-DNN 卷积操作的内存格式
    auto memory_format =
        mkldnn_convolution_memory_format(input_t.ndimension(), true);
    // 使输入张量连续化，并采用指定的内存格式
    auto input = input_t.contiguous(memory_format);
    // 如果权重张量已经是 MKL-DNN 格式，则直接使用，否则转换为连续的指定内存格式
    auto weight =
        weight_t.is_mkldnn() ? weight_t : weight_t.contiguous(memory_format);
    // 使其他输入张量连续化，并采用指定的内存格式
    auto other = other_t.contiguous(memory_format);
    // 创建一个与其他张量相同大小的空张量
    auto output = at::empty_like(other);
    // 将输入张量转换为 MKL-DNN 张量
    const ideep::tensor x = itensor_from_tensor(input);
    // 将权重张量转换为 MKL-DNN 张量
    const ideep::tensor w = itensor_from_tensor(weight);
    // 将其他张量转换为 MKL-DNN 张量
    const ideep::tensor z = itensor_from_tensor(other);
    // 将输出张量转换为 MKL-DNN 张量
    ideep::tensor y = itensor_from_tensor(output);
    // 获取其他张量的大小
    auto output_size = other.sizes().vec();
    // 设置张量的格式标签为 NHWC（默认）
    ideep::tag format_tag = ideep::tag::nhwc;
    // 如果输入张量的维度为 5，则设置格式标签为 NDHWC
    if (input_t.ndimension() == 5) {
      format_tag = ideep::tag::ndhwc;
    }
    // 创建 MKL-DNN 张量的描述符，使用指定的大小、数据类型和格式标签
    auto other_desc = ideep::tensor::desc(
        output_size, get_mkldnn_dtype(weight.scalar_type()), format_tag);

    // 设置操作属性
    ideep::attr_t op_attr;
    // 创建后操作对象
    ideep::post_ops po;
    // 添加二元操作到后操作对象中，使用指定的描述符
    po.append_binary(it_binary->second, other_desc);
    // 如果一元操作属性值不为 "none"，则添加一元操作到后操作对象中
    if (unary_attr_value != "none") {
      po.append_eltwise(unary_alg, 0.f, 0.f);
    }
    // 设置后操作属性的后操作集
    op_attr.set_post_ops(po);

    // 如果存在偏置张量
    if (bias.defined()) {
      // 将偏置张量转换为 MKL-DNN 张量
      const ideep::tensor b = itensor_from_tensor(bias);
      // 执行 MKL-DNN 卷积前向计算，使用二元、一元后操作，设置操作属性
      ideep::convolution_forward::compute_binary(
          x,
          z,
          w,
          b,
          output_size,
          y,
          stride_expanded,
          dilation_expanded,
          padding_expanded,
          padding_expanded,
          groups,
          /* is_channels_last */ true,
          op_attr);
    } else {
      // 执行 MKL-DNN 卷积前向计算，使用二元后操作，设置操作属性
      ideep::convolution_forward::compute_binary(
          x,
          z,
          w,
          output_size,
          y,
          stride_expanded,
          dilation_expanded,
          padding_expanded,
          padding_expanded,
          groups,
          /* is_channels_last */ true,
          op_attr);
    }
    // 返回输出张量
    return output;
  } else {
    // 如果不能进行融合，则执行回退情况
    // 当输入不是通道最后或具有不同的数据类型时，MKL-DNN 融合可能会有性能回退。
    Tensor output;
    // 如果权重张量已经是 MKL-DNN 格式，调用 _mkldnn_convolution 函数
    if (weight_t.is_mkldnn()) {
      output = _mkldnn_convolution(
          input_t, weight_t, bias, padding_expanded, stride_expanded, dilation, groups, true);
    } else {
      // 否则调用 ATen 库的卷积函数
      output = at::convolution(
          input_t, weight_t, bias, stride_expanded, padding_expanded, dilation_expanded, false, 0, groups);
    }
    // 如果二元操作为 "add" 并且一元操作属性值不为 "none"，则执行原位加操作并返回结果
    if (binary_attr == "add" && unary_attr_value != "none") {
      output = at::native::add_relu_(output, other_t);
      return output;
    }
    // 如果二元操作为 "add"，则执行原位加操作
    if (binary_attr == "add") {
      output.add_(other_t);
    }
    } else if (binary_attr == "sub") {
      // 如果 binary_attr 等于 "sub"，则执行张量减法操作
      output.sub_(other_t);
    } else if (binary_attr == "mul") {
      // 如果 binary_attr 等于 "mul"，则执行张量乘法操作
      output.mul_(other_t);
    } else {
      // 对于其他情况，执行张量除法操作
      output.div_(other_t);
    }
    // 如果 unary_attr_value 不等于 "none"，则执行 ReLU 激活函数操作
    if (unary_attr_value != "none") {
      output.relu_();
    }
    // 返回处理后的张量作为函数的输出结果
    return output;
  }
}

// 合并卷积、二元操作和一元操作以提高性能，执行以下操作：
// other_t = unary_op(binary_op(conv(input_t, ...), other_t, alpha))
// binary_attr 表示二元操作类型，可以是 "add" 或其他二元操作。
// unary_attr 表示一元操作类型，可以是 "relu" 或其他一元操作；如果为 none，则表示没有一元后操作。
// unary_scalars 和 unary_algorithm 是一元操作的参数，例如 "hardtanh" 的标量参数或 "gelu" 的算法参数。

Tensor& mkldnn_convolution_pointwise_binary_(
    Tensor& other_t,                 // 输出张量
    const Tensor& input_t,           // 输入张量
    const Tensor& weight_t,          // 权重张量
    const std::optional<Tensor>& bias_opt,  // 可选的偏置张量
    IntArrayRef padding,             // 填充大小数组
    IntArrayRef stride,              // 步幅大小数组
    IntArrayRef dilation,            // 膨胀大小数组
    int64_t groups,                  // 分组数
    c10::string_view binary_attr,    // 二元操作类型
    std::optional<at::Scalar> alpha, // 可选的 alpha 参数
    std::optional<c10::string_view> unary_attr,  // 可选的一元操作类型
    torch::List<std::optional<at::Scalar>> unary_scalars,
                                      // 一元操作的标量参数列表
    std::optional<c10::string_view> unary_algorithm) {
    // 检查输入张量的维度是否为4或5，表示支持2D或3D的卷积操作
    TORCH_CHECK(
        input_t.ndimension() == 4 || input_t.ndimension() == 5,
        "mkldnn_convolution_add_: currently only support 2d and 3d")
    // 检查二元操作是否为加法
    TORCH_CHECK(
        binary_attr == "add",
        "mkldnn_convolution_pointwise_binary_: only support binary op fusion")
    // 检查 alpha 值是否为1.0或未提供
    TORCH_CHECK(
        !alpha.has_value() || alpha.value().to<float>() == 1.0,
        "mkldnn_convolution_pointwise_binary: the alpha value for the binary op should be none(meaning 1.0) or 1.0");
    // 检查是否支持无操作或者后续操作为ReLU
    TORCH_CHECK(
        !unary_attr.has_value() || unary_attr.value() == "relu",
        "mkldnn_convolution_pointwise_binary: only support none or relu unary op fusion after binary op");

    // 获取 bias 引用，确保其存在
    c10::MaybeOwned<Tensor> bias_maybe_owned =
        at::borrow_from_optional_tensor(bias_opt);
    const Tensor& bias = *bias_maybe_owned;

    // 确保输入张量具有相同的类型（设备、布局、数据类型），设备为CPU，数据类型为float、bfloat16或half
    check_mkldnn_binary_fusion_inputs(input_t, other_t, weight_t, bias);
    // 计算卷积操作的维度数
    int64_t dim = input_t.ndimension() - 2;
    // 根据需要扩展填充参数
    const auto padding_expanded = expand_param_if_needed(padding, "padding", dim);
    // 根据需要扩展步幅参数
    const auto stride_expanded = expand_param_if_needed(stride, "stride", dim);
    // 根据需要扩展膨胀参数
    const auto dilation_expanded = expand_param_if_needed(dilation, "dilation", dim);
    // 检查输入张量的形状是否符合要求
    check_shape_forward(
        input_t, weight_t, bias, padding, stride, dilation, groups);

    // 计算卷积输出的预期尺寸
    auto output_sizes = conv_output_size(
        input_t.sizes(), weight_t.sizes(), padding_expanded, stride_expanded, dilation_expanded);
    // 检查输出张量的尺寸与 other_t 的尺寸是否相同
    TORCH_CHECK(
        output_sizes == other_t.sizes(),
        "Add Fusion's inputs should have same shape");

    // 检查是否可以执行融合路径，要求权重张量为MKLDNN格式或通道最后格式的连续张量
    bool can_be_fused = (weight_t.is_mkldnn() ||
                         mkldnn_conv_use_channels_last(input_t, weight_t)) &&
        (other_t.is_contiguous(at::MemoryFormat::ChannelsLast) ||
         other_t.is_contiguous(at::MemoryFormat::ChannelsLast3d));

    // 设置自动求导分发键的排除保护
    c10::impl::ExcludeDispatchKeyGuard edkg(c10::autograd_dispatch_keyset);
    if (can_be_fused) {
        // 将 other_t 转换为ideep::tensor格式
        ideep::tensor y = itensor_from_tensor(other_t);
        ideep::attr_t op_attr;
        // 如果存在一元属性，设置为residual；否则设置为fuse_sum
        if (unary_attr.has_value()) {
            op_attr = ideep::attr_t::residual();
        } else {
            op_attr = ideep::attr_t::fuse_sum();
        }
        // 执行MKLDNN卷积操作，并将结果写入 y
        _mkldnn_convolution_out(
            input_t,
            weight_t,
            bias,
            output_sizes,
            y,
            stride_expanded,
            dilation_expanded,
            padding_expanded,
            groups,
            true,
            op_attr);
    } else {
        // 回退情况，如果输入不是通道最后格式或具有不同的数据类型，OneDNN融合可能性能回退
        Tensor output;
        if (weight_t.is_mkldnn()) {
            output = _mkldnn_convolution(
                input_t, weight_t, bias, padding_expanded, stride_expanded, dilation_expanded, groups, true);
    } else {
      // 如果有 unary_attr 值，对 output 进行 add_relu_ 操作，并将结果赋给 other_t
      output = at::convolution(
          input_t, weight_t, bias, stride_expanded, padding_expanded, dilation_expanded, false, 0, groups);
    }
    if (unary_attr.has_value()) {
      // 如果 unary_attr 有值，则对 other_t 和 output 执行 add_relu_ 操作
      other_t = at::native::add_relu_(other_t, output);
    } else {
      // 如果 unary_attr 没有值，则直接将 output 加到 other_t 上
      other_t.add_(output);
    }
  }
  // 返回最终计算结果的 other_t 张量
  return other_t;
}

// 计算转置卷积权重的原始尺寸
std::vector<int64_t> _original_deconv_weight_size(
    const Tensor& weight_t,   // 输入的权重张量
    int64_t groups) {         // 卷积分组数
  // 检查权重张量是否为MKLDNN格式或元数据张量
  TORCH_CHECK(weight_t.is_mkldnn() || weight_t.is_meta(), "expects weight_t to be mkldnn or meta tensor");
  
  // 权重张量的维度大小
  auto dim = weight_t.sizes().size();
  TORCH_CHECK(dim > 2);  // 确保维度大于2，即至少有输入和输出通道

  // 初始化存储原始权重尺寸的向量
  std::vector<int64_t> weight_IOHW_sizes(dim);

  // 根据分组数(groups)不同计算权重的原始尺寸
  if (groups > 1) {
    weight_IOHW_sizes[0] = weight_t.sizes()[1] * groups;    // 输出通道大小
    weight_IOHW_sizes[1] = weight_t.sizes()[0] / groups;    // 输入通道大小
  } else {
    weight_IOHW_sizes[0] = weight_t.sizes()[1];             // 输出通道大小
    weight_IOHW_sizes[1] = weight_t.sizes()[0];             // 输入通道大小
  }

  // 复制其他维度的尺寸
  for (const auto d : c10::irange(2, dim)) {
    weight_IOHW_sizes[d] = weight_t.sizes()[d];
  }

  // 返回原始权重尺寸向量
  return weight_IOHW_sizes;
}


// 执行MKLDNN格式的转置卷积操作
Tensor _mkldnn_convolution_transpose(
    const Tensor& input_t,                         // 输入张量
    const Tensor& weight_t,                        // 权重张量
    const std::optional<Tensor>& bias_opt,         // 可选的偏置张量
    IntArrayRef padding,                           // 填充
    IntArrayRef output_padding,                    // 输出填充
    IntArrayRef stride,                            // 步幅
    IntArrayRef dilation,                          // 膨胀
    int64_t groups,                                // 分组数
    bool use_channels_last,                        // 是否使用通道为最后一个维度
    c10::string_view attr = "none",                // 属性名称
    torch::List<std::optional<at::Scalar>> scalars = 
        torch::List<std::optional<at::Scalar>>(),  // 标量列表
    std::optional<c10::string_view> algorithm = c10::nullopt) {  // 算法选项
  ideep::attr_t op_attr = ideep::attr_t();         // 定义IDEep属性对象

  // 如果指定了属性，则检查其在融合操作中的定义
  if (attr != "none") {
    auto it = fusion_unary_attr_map().find(attr);
    TORCH_CHECK(it != fusion_unary_attr_map().end(), "Fusion behavior undefined.");
  // 从迭代器中获取操作属性，并使用给定的标量和算法执行操作
  op_attr = it->second(scalars, algorithm);

  // 从可选的张量中借用数据，创建一个不确定所有权的张量引用
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  // 检查输入张量的低精度情况，用于特定于 mkldnn 的反卷积操作
  mkldnn_check_low_precision(input_t.scalar_type(), "mkldnn_convolution_transpose");

  // 如果权重张量是 mkldnn 格式，则获取原始反卷积权重大小，否则获取标准张量的大小
  std::vector<int64_t> weight_IOHW_sizes = weight_t.is_mkldnn() ? _original_deconv_weight_size(weight_t, groups) : weight_t.sizes().vec();

  // 确定内存格式以用于 mkldnn 反卷积操作的输入张量
  auto memory_format =
      mkldnn_convolution_memory_format(input_t.ndimension(), use_channels_last);

  // 如果输入张量是 mkldnn 格式，则直接使用，否则进行内存连续化操作
  auto input = input_t.is_mkldnn() ? input_t : input_t.contiguous(memory_format);
  auto weight = weight_t.is_mkldnn() ? weight_t : weight_t.contiguous(memory_format);

  // 计算输入张量的维度
  int64_t dim = input.ndimension() - 2;
  // 根据需要扩展填充参数
  const auto padding_expanded = expand_param_if_needed(padding, "padding", dim);
  // 根据需要扩展步长参数
  const auto stride_expanded = expand_param_if_needed(stride, "stride", dim);
  // 根据需要扩展膨胀参数
  const auto dilation_expanded = expand_param_if_needed(dilation, "dilation", dim);
  // 根据需要扩展输出填充参数
  const auto output_padding_expanded = expand_param_if_needed(output_padding, "output_padding", dim);
  // 计算输出张量的大小
  auto output_sizes = conv_input_size(input.sizes(), weight_IOHW_sizes, padding_expanded, output_padding_expanded, stride_expanded, dilation_expanded, groups);
  // 创建一个空的输出张量
  auto output = at::empty({0}, input.options());

  // 将输入张量转换为 mkldnn 的 ideep::tensor 对象
  const ideep::tensor x = itensor_from_tensor(input, /*from_const_data_ptr*/true);

  // 将权重张量转换为 mkldnn 的 ideep::tensor 对象
  ideep::tensor w = itensor_from_tensor(weight, /*from_const_data_ptr*/true);
  // 如果权重张量不是 mkldnn 格式，进行转置以适配 mkldnn 的要求
  if (!weight.is_mkldnn()) {
    // mkldnn 的反卷积要求权重张量在逻辑上的顺序是 OIHW 或 OIDHW，而 PyTorch 是 IOHW 或 IODHW，通过转置进行调整
    w.transpose_(0, 1);
  }

  // 定义输出张量的 ideep::tensor 对象
  ideep::tensor y;
  // 如果使用了通道最后的内存格式，则调整输出张量的大小和内存格式
  if (use_channels_last) {
    output.resize_(output_sizes, memory_format);
    y = itensor_from_tensor(output);
  }

  // 如果存在偏置，则将偏置张量转换为 ideep::tensor 对象，并执行 mkldnn 反卷积计算
  if (bias.defined()) {
    const ideep::tensor b = itensor_from_tensor(bias, /*from_const_data_ptr*/true);
    ideep::convolution_transpose_forward::compute_v3(
        x,
        w,
        b,
        output_sizes,
        y,
        stride_expanded,
        padding_expanded,
        padding_r(padding_expanded, output_padding_expanded),
        dilation.vec(),
        groups,
        use_channels_last,
        op_attr);
  } else {
    // 如果没有偏置，则直接执行 mkldnn 反卷积计算
    ideep::convolution_transpose_forward::compute_v3(
        x,
        w,
        output_sizes,
        y,
        stride_expanded,
        padding_expanded,
        padding_r(padding_expanded, output_padding_expanded),
        dilation.vec(),
        groups,
        use_channels_last,
        op_attr);
  }

  // 如果输入张量是 mkldnn 格式，则将 ideep::tensor 转换为 MKLDNNTensor 对象并返回
  if (input.is_mkldnn()) {
    return MKLDNNTensor(y, input.options());
  } else if (!use_channels_last) {
    // 如果未使用通道最后的内存格式，则将 ideep::tensor 转换为稠密张量并返回
    return mkldnn_to_dense(MKLDNNTensor(y, input.options()));
  } else {
    // 否则直接返回输出张量
    return output;
  }
}

// 定义一个函数 mkldnn_convolution_transpose_pointwise，接受多个参数并返回 Tensor 类型
Tensor mkldnn_convolution_transpose_pointwise(
    const Tensor& input_t,  // 输入张量
    const Tensor& weight_t,  // 权重张量
    const std::optional<Tensor>& bias_opt,  // 可选的偏置张量
    IntArrayRef padding,  // 填充数组
    IntArrayRef output_padding,  // 输出填充数组
    IntArrayRef stride,  // 步幅数组
    IntArrayRef dilation,  // 扩张数组
    int64_t groups,  // 分组数
    c10::string_view attr,  // 属性字符串视图
    torch::List<std::optional<at::Scalar>> scalars,  // 标量列表
    std::optional<c10::string_view> algorithm) {  // 可选的算法字符串视图

  // 临时排除自动求导键
  c10::impl::ExcludeDispatchKeyGuard edkg(c10::autograd_dispatch_keyset);

  // 确定是否使用通道最后格式
  bool use_channels_last =
      weight_t.is_mkldnn() || mkldnn_conv_use_channels_last(input_t, weight_t);

  // 调用实际的转置卷积函数并返回结果
  return _mkldnn_convolution_transpose(
      input_t,
      weight_t,
      bias_opt,
      padding,
      output_padding,
      stride,
      dilation,
      groups,
      use_channels_last,
      attr,
      scalars,
      algorithm
  );
}

// 定义一个函数 mkldnn_convolution_transpose_pointwise_meta，接受多个参数并返回 Tensor 类型
Tensor mkldnn_convolution_transpose_pointwise_meta(
    const Tensor& input_t,  // 输入张量
    const Tensor& weight_t,  // 权重张量
    const std::optional<Tensor>& bias_opt,  // 可选的偏置张量
    IntArrayRef padding,  // 填充数组
    IntArrayRef output_padding,  // 输出填充数组
    IntArrayRef stride,  // 步幅数组
    IntArrayRef dilation,  // 扩张数组
    int64_t groups,  // 分组数
    c10::string_view attr,  // 属性字符串视图
    torch::List<std::optional<at::Scalar>> scalars,  // 标量列表
    std::optional<c10::string_view> algorithm) {  // 可选的算法字符串视图

  // 计算原始反卷积权重的大小
  std::vector<int64_t> weight_IOHW_sizes = _original_deconv_weight_size(weight_t, groups);

  // 计算输入张量维度
  int64_t dim = input_t.ndimension() - 2;

  // 根据需要扩展填充、步幅、扩张和输出填充参数
  const auto padding_expanded = expand_param_if_needed(padding, "padding", dim);
  const auto stride_expanded = expand_param_if_needed(stride, "stride", dim);
  const auto dilation_expanded = expand_param_if_needed(dilation, "dilation", dim);
  const auto output_padding_expanded = expand_param_if_needed(output_padding, "output_padding", dim);

  // 计算卷积输入大小
  auto output_sizes = conv_input_size(input_t.sizes(), weight_IOHW_sizes, padding_expanded , output_padding_expanded , stride_expanded , dilation_expanded , groups);

  // 创建并返回一个空的张量，用于存储结果
  auto output = at::empty(output_sizes, input_t.options());
  return output;
}

// 定义一个函数 mkldnn_convolution_backward_input，接受多个参数并返回 Tensor 类型
Tensor mkldnn_convolution_backward_input(
    IntArrayRef input_size,  // 输入大小数组
    const Tensor& grad_output,  // 梯度输出张量
    const Tensor& weight,  // 权重张量
    IntArrayRef padding,  // 填充数组
    IntArrayRef stride,  // 步幅数组
    IntArrayRef dilation,  // 扩张数组
    int64_t groups,  // 分组数
    bool bias_defined,  // 是否定义了偏置
    bool is_channels_last) {  // 是否通道最后

  // 创建一个空的梯度输入张量
  auto grad_input = at::empty({0}, grad_output.options());

  // 从密集张量中获取梯度输出和权重张量的 ideep::tensor 视图
  auto grad_y = itensor_from_tensor(grad_output, /*from_const_data_ptr*/true);
  auto w = itensor_view_from_dense(weight, /*from_const_data_ptr*/true);

  // 定义一个用于反卷积的 ideep::tensor
  ideep::tensor grad_x;

  // 如果使用通道最后格式，则根据情况设置内存格式和大小
  if (is_channels_last) {
    auto memory_format = mkldnn_convolution_memory_format(grad_output.ndimension(), is_channels_last);
    grad_input.resize_(input_size, memory_format);
    grad_x = itensor_from_tensor(grad_input);
  }

  // 执行反卷积数据的计算
  ideep::convolution_backward_data::compute_v2(
      grad_y,
      w,
      input_size.vec(),
      grad_x,
      stride.vec(),
      dilation.vec(),
      padding.vec(),
      padding.vec(),
      groups,
      is_channels_last);

  // 如果梯度输出张量是 MKLDNN 格式的，则...
  if (grad_output.is_mkldnn()) {
  // 如果条件为真，即不是按照通道最后的顺序存储
  return MKLDNNTensor(grad_x, grad_output.options());
} else if (!is_channels_last){
  // 如果条件为假，并且不是按照通道最后的顺序存储
  return mkldnn_to_dense(MKLDNNTensor(grad_x, grad_output.options()));
} else {
  // 如果以上条件都不满足，则返回梯度输入本身
  return grad_input;
}
}

std::tuple<Tensor, Tensor> mkldnn_convolution_backward_weights(
    IntArrayRef weight_size,                            // 定义权重大小的整数数组引用
    const Tensor& grad_output,                          // 梯度输出张量的常量引用
    const Tensor& input,                                // 输入张量的常量引用
    IntArrayRef padding,                                // 填充大小的整数数组引用
    IntArrayRef stride,                                 // 步幅大小的整数数组引用
    IntArrayRef dilation,                               // 膨胀大小的整数数组引用
    int64_t groups,                                     // 卷积组数
    bool bias_defined,                                  // 是否定义了偏置
    bool is_channels_last) {                            // 是否为通道最后格式

  const ideep::tensor grad_y = itensor_from_tensor(grad_output, /*from_const_data_ptr*/true);  // 将梯度输出转换为ideep张量
  const ideep::tensor x = itensor_from_tensor(input, /*from_const_data_ptr*/true);              // 将输入转换为ideep张量

  ideep::tensor grad_w, grad_b;                         // 定义权重梯度和偏置梯度的ideep张量对象
  if (bias_defined) {
    // 计算带有偏置的反向权重更新
    ideep::convolution_backward_weights::compute_v2(
        x,
        grad_y,
        weight_size.vec(),
        grad_w,
        grad_b,
        stride.vec(),
        dilation.vec(),
        padding.vec(),
        padding.vec(),
        groups,
        is_channels_last);
  } else {
    // 计算不带偏置的反向权重更新
    ideep::convolution_backward_weights::compute_v2(
        x,
        grad_y,
        weight_size.vec(),
        grad_w,
        stride.vec(),
        dilation.vec(),
        padding.vec(),
        padding.vec(),
        groups,
        is_channels_last);
  }

  if (!is_channels_last) {
    // 如果不是通道最后格式，则将结果转换为稠密张量格式并返回
    return std::make_tuple(
        mkldnn_to_dense(MKLDNNTensor(grad_w, grad_output.options())),
        bias_defined ? mkldnn_to_dense(MKLDNNTensor(grad_b, grad_output.options())) : Tensor());
  } else {
    // 如果是通道最后格式，则根据内存格式转换后返回
    auto memory_format = mkldnn_convolution_memory_format(grad_output.ndimension(), is_channels_last);
    return std::make_tuple(
        mkldnn_to_dense(MKLDNNTensor(grad_w, grad_output.options())).to(memory_format),
        bias_defined ? mkldnn_to_dense(MKLDNNTensor(grad_b, grad_output.options())) : Tensor());
  }
}

std::tuple<Tensor, Tensor, Tensor> mkldnn_convolution_backward(
    const Tensor& input_t, const Tensor& grad_output_t, const Tensor& weight_t,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, std::array<bool,3> output_mask)
{
  bool is_channels_last = mkldnn_conv_use_channels_last(input_t, weight_t);  // 判断是否使用通道最后格式
  auto memory_format = mkldnn_convolution_memory_format(input_t.ndimension(), is_channels_last);  // 获取卷积内存格式
  Tensor grad_output = grad_output_t.is_mkldnn() ? grad_output_t : grad_output_t.contiguous(memory_format);  // 确保梯度输出张量连续性

  Tensor input = input_t.is_mkldnn() ? input_t : input_t.contiguous(memory_format);  // 确保输入张量连续性
  Tensor weight = weight_t.is_mkldnn() ? weight_t : weight_t.contiguous(memory_format);  // 确保权重张量连续性
  int64_t dim = input.ndimension() - 2;  // 计算卷积维度
  const auto padding_expanded = expand_param_if_needed(padding, "padding", dim);  // 根据维度扩展填充参数
  const auto stride_expanded = expand_param_if_needed(stride, "stride", dim);     // 根据维度扩展步幅参数
  const auto dilation_expanded = expand_param_if_needed(dilation, "dilation", dim);  // 根据维度扩展膨胀参数
  Tensor grad_input, grad_weight, grad_bias;

  if (output_mask[0]) {
    // 如果需要计算输入的梯度，则调用对应的反向输入卷积函数
    grad_input = mkldnn_convolution_backward_input(
      input.sizes(), grad_output, weight, padding_expanded, stride_expanded, dilation_expanded, groups, output_mask[2], is_channels_last);
  }
  if (output_mask[1] || output_mask[2]) {
    // 如果需要计算权重或者偏置的梯度，则继续进行相关计算
    // 使用 MKL-DNN 库进行卷积反向传播计算，计算卷积权重的梯度和偏置的梯度
    std::tie(grad_weight, grad_bias) = mkldnn_convolution_backward_weights(
      weight.sizes(), grad_output, input, padding_expanded, stride_expanded, dilation_expanded, groups, output_mask[2], is_channels_last);
    // 返回包含梯度的元组，包括输入的梯度、权重的梯度和偏置的梯度
    }
    // 返回包含输入的梯度、权重的梯度和偏置的梯度的元组
    return std::make_tuple(grad_input, grad_weight, grad_bias);
}

}

// 注册所有 CPU 分发函数，将 mkldnn_convolution_backward_stub 映射到 mkldnn_convolution_backward
REGISTER_ALL_CPU_DISPATCH(mkldnn_convolution_backward_stub, &mkldnn_convolution_backward);

// 命名空间开始
namespace {

// 实现卷积转置操作
Tensor mkldnn_convolution_transpose(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias_opt,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups)
{
  // 确定是否使用 channels_last 内存布局
  bool use_channels_last = mkldnn_conv_use_channels_last(input, weight);
  
  // 调用内部函数执行 MKLDNN 卷积转置
  return _mkldnn_convolution_transpose(
      input,
      weight,
      bias_opt,
      padding,
      output_padding,
      stride,
      dilation,
      groups,
      use_channels_last
  );
}

// 计算卷积转置操作对输入的反向传播
Tensor mkldnn_convolution_transpose_backward_input(
    IntArrayRef input_size,
    const Tensor& grad_output,
    const Tensor& weight,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool bias_defined,
    bool is_channels_last)
{
  // 创建一个空的梯度输入张量
  auto grad_input = at::empty({0}, grad_output.options());

  // 获取梯度输出张量的 ideep::tensor 表示
  auto grad_y = itensor_from_tensor(grad_output, /*from_const_data_ptr*/true);
  // 获取权重张量的 ideep::tensor 视图，并进行转置
  auto w = itensor_view_from_dense(weight, /*from_const_data_ptr*/true).transpose_(0, 1);

  ideep::tensor grad_x;
  if (is_channels_last) {
    // 根据 channels_last 决定内存格式
    auto memory_format = mkldnn_convolution_memory_format(grad_output.ndimension(), is_channels_last);
    grad_input.resize_(input_size, memory_format);
    grad_x = itensor_from_tensor(grad_input);
  }
  
  // 执行 MKLDNN 卷积转置的反向数据计算
  ideep::convolution_transpose_backward_data::compute_v3(
      grad_y,
      w,
      input_size.vec(),
      grad_x,
      stride.vec(),
      padding.vec(),
      padding_r(padding, output_padding),
      dilation.vec(),
      groups,
      is_channels_last);

  // 如果梯度输出是 MKLDNN 张量，则返回相应的 MKLDNN 张量
  if (grad_output.is_mkldnn()) {
    return MKLDNNTensor(grad_x, grad_output.options());
  } else if (!is_channels_last) {
    // 如果不是 channels_last，则将 MKLDNN 张量转换为普通张量后返回
    return mkldnn_to_dense(MKLDNNTensor(grad_x, grad_output.options()));
  } else {
    // 否则直接返回梯度输入张量
    return grad_input;
  }
}

// 计算卷积转置操作对权重的反向传播
std::tuple<Tensor,Tensor> mkldnn_convolution_transpose_backward_weights(
    IntArrayRef weight_size,
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool bias_defined,
    bool is_channels_last)
{
  // 获取梯度输出张量的 ideep::tensor 表示
  auto grad_y = itensor_from_tensor(grad_output, /*from_const_data_ptr*/true);
  // 获取输入张量的 ideep::tensor 表示
  auto x = itensor_from_tensor(input, /*from_const_data_ptr*/true);

  ideep::tensor grad_w, grad_b;
  if (bias_defined) {
    // 执行带偏置的 MKLDNN 卷积转置的权重反向传播计算
    ideep::convolution_transpose_backward_weights::compute_v3(
        x,
        grad_y,
        weight_size.vec(),
        grad_w,
        grad_b,
        stride.vec(),
        padding.vec(),
        padding_r(padding, output_padding),
        dilation.vec(),
        groups,
        is_channels_last);
  } else {
    // 执行不带偏置的 MKLDNN 卷积转置的权重反向传播计算
    ideep::convolution_transpose_backward_weights::compute_v3(
        x,
        grad_y,
        weight_size.vec(),
        grad_w,
        stride.vec(),
        padding.vec(),
        padding_r(padding, output_padding),
        dilation.vec(),
        groups,
        is_channels_last);
  }

  // 返回计算得到的权重梯度和偏置梯度
  return std::make_tuple(
      MKLDNNTensor(grad_w, input.options()),
      MKLDNNTensor(grad_b, input.options())
  );
}
  // 调用深度学习库ideep中的反卷积权重反向传播计算函数
  ideep::convolution_transpose_backward_weights::compute_v3(
      x,                                  // 输入张量x
      grad_y,                             // 梯度张量grad_y
      weight_size.vec(),                  // 权重大小的向量
      grad_w,                             // 权重梯度grad_w
      stride.vec(),                       // 步长的向量
      padding.vec(),                      // 填充的向量
      padding_r(padding, output_padding), // 组合填充和输出填充的对象padding_r
      dilation.vec(),                     // 膨胀的向量
      groups,                             // 分组数
      is_channels_last);                  // 是否以通道为最后一维

  // 如果不是以通道为最后一维的存储方式
  if (!is_channels_last) {
    // 返回以密集张量表示的权重梯度和（如果定义了偏置）以密集张量表示的偏置梯度
    return std::make_tuple(
        mkldnn_to_dense(MKLDNNTensor(grad_w, grad_output.options())), // 转换为密集张量的权重梯度
        bias_defined ? mkldnn_to_dense(MKLDNNTensor(grad_b, grad_output.options())) : Tensor()); // 如果定义了偏置，则转换为密集张量的偏置梯度，否则返回空张量
  } else {
    // 否则，根据输出张量的维度和是否以通道为最后一维，确定内存格式
    auto memory_format = mkldnn_convolution_memory_format(grad_output.ndimension(), is_channels_last);
    // 返回以内存格式表示的权重梯度和（如果定义了偏置）以密集张量表示的偏置梯度
    return std::make_tuple(
        mkldnn_to_dense(MKLDNNTensor(grad_w, grad_output.options())).to(memory_format), // 转换为指定内存格式的权重梯度
        bias_defined ? mkldnn_to_dense(MKLDNNTensor(grad_b, grad_output.options())) : Tensor()); // 如果定义了偏置，则转换为密集张量的偏置梯度，否则返回空张量
  }
}
}

// 定义函数 mkldnn_convolution_transpose_backward，接受多个输入张量和参数，返回三个张量的元组
std::tuple<Tensor, Tensor, Tensor> mkldnn_convolution_transpose_backward(
    const Tensor& input_t, const Tensor& grad_output_t, const Tensor& weight_t,
    IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    std::array<bool,3> output_mask)
{
  // 检查输入张量是否按通道为最后一个维度的格式存储
  bool is_channels_last = mkldnn_conv_use_channels_last(input_t, weight_t);
  // 获取卷积操作的内存格式
  auto memory_format = mkldnn_convolution_memory_format(input_t.ndimension(), is_channels_last);
  // 如果 grad_output_t 是 MKLDNN 张量，则直接使用，否则转换成指定的内存格式
  Tensor grad_output = grad_output_t.is_mkldnn() ? grad_output_t : grad_output_t.contiguous(memory_format);
  // 如果 input_t 是 MKLDNN 张量，则直接使用，否则转换成指定的内存格式
  auto input = input_t.is_mkldnn() ? input_t : input_t.contiguous(memory_format);
  // 如果 weight_t 是 MKLDNN 张量，则直接使用，否则转换成指定的内存格式
  auto weight = weight_t.is_mkldnn() ? weight_t : weight_t.contiguous(memory_format);
  // 计算输入张量的维度
  int64_t dim = input.ndimension() - 2;
  // 根据维度扩展 padding、stride、dilation 和 output_padding 参数
  const auto padding_expanded = expand_param_if_needed(padding, "padding", dim);
  const auto stride_expanded = expand_param_if_needed(stride, "stride", dim);
  const auto dilation_expanded = expand_param_if_needed(dilation, "dilation", dim);
  const auto output_padding_expanded = expand_param_if_needed(output_padding, "output_padding", dim);

  // 定义 grad_input、grad_weight 和 grad_bias 张量
  Tensor grad_input, grad_weight, grad_bias;
  // 如果 output_mask 的第一个元素为 true，则计算反向传播的输入梯度
  if (output_mask[0]) {
    grad_input = mkldnn_convolution_transpose_backward_input(
        input.sizes(), grad_output, weight, padding_expanded , output_padding_expanded , stride_expanded , dilation_expanded , groups, output_mask[2], is_channels_last);
  }
  // 如果 output_mask 的第二个或第三个元素为 true，则计算反向传播的权重梯度和偏置梯度
  if (output_mask[1] || output_mask[2]) {
    std::tie(grad_weight, grad_bias) = mkldnn_convolution_transpose_backward_weights(
        weight.sizes(), grad_output, input, padding_expanded , output_padding_expanded , stride_expanded , dilation_expanded , groups, output_mask[2], is_channels_last);
  }
  // 返回计算得到的 grad_input、grad_weight 和 grad_bias 张量的元组
  return std::make_tuple(grad_input, grad_weight, grad_bias);
}
}

// 注册 CPU 下的 MKLDNN 反卷积函数
REGISTER_ALL_CPU_DISPATCH(mkldnn_convolution_transpose_stub, &mkldnn_convolution_transpose);
// 注册 CPU 下的 MKLDNN 反卷积反向传播函数
REGISTER_ALL_CPU_DISPATCH(mkldnn_convolution_transpose_backward_stub, &mkldnn_convolution_transpose_backward);

// 实现 MKLDNN 库的 CPU 版本
TORCH_LIBRARY_IMPL(mkldnn, CPU, m) {
  // 注册 _convolution_pointwise 函数到 mkldnn::_convolution_pointwise
  m.impl(
      TORCH_SELECTIVE_NAME("mkldnn::_convolution_pointwise"),
      TORCH_FN(mkldnn_convolution_pointwise));
  // 注册 _convolution_pointwise.binary 函数到 mkldnn::_convolution_pointwise.binary
  m.impl(
      TORCH_SELECTIVE_NAME("mkldnn::_convolution_pointwise.binary"),
      TORCH_FN(mkldnn_convolution_pointwise_binary));
  // 注册 _convolution_pointwise_.binary 函数到 mkldnn::_convolution_pointwise_.binary
  m.impl(
      TORCH_SELECTIVE_NAME("mkldnn::_convolution_pointwise_.binary"),
      TORCH_FN(mkldnn_convolution_pointwise_binary_));
  // 注册 _convolution_transpose_pointwise 函数到 mkldnn::_convolution_transpose_pointwise
  m.impl(
      TORCH_SELECTIVE_NAME("mkldnn::_convolution_transpose_pointwise"),
      TORCH_FN(mkldnn_convolution_transpose_pointwise));
}
TORCH_LIBRARY_IMPL(mkldnn, MkldnnCPU, m) {
  // 定义 Torch 库的实现，用于 MKL-DNN CPU 加速
  m.impl(
      TORCH_SELECTIVE_NAME("mkldnn::_convolution_pointwise"),
      // 注册点对点卷积的实现函数
      TORCH_FN(mkldnn_convolution_pointwise));
  m.impl(
      TORCH_SELECTIVE_NAME("mkldnn::_convolution_pointwise.binary"),
      // 注册二进制点对点卷积的实现函数
      TORCH_FN(mkldnn_convolution_pointwise_binary));
  m.impl(
      TORCH_SELECTIVE_NAME("mkldnn::_convolution_pointwise_.binary"),
      // 注册带下划线的二进制点对点卷积的实现函数
      TORCH_FN(mkldnn_convolution_pointwise_binary_));
  m.impl(
      TORCH_SELECTIVE_NAME("mkldnn::_convolution_transpose_pointwise"),
      // 注册转置点对点卷积的实现函数
      TORCH_FN(mkldnn_convolution_transpose_pointwise));
}

TORCH_LIBRARY_IMPL(mkldnn, Meta, m) {
  // 定义 Torch 库的实现，用于 MKL-DNN 的元操作
  m.impl(
      TORCH_SELECTIVE_NAME("mkldnn::_convolution_transpose_pointwise"),
      // 注册转置点对点卷积的元操作实现函数
      TORCH_FN(mkldnn_convolution_transpose_pointwise_meta));
}
```