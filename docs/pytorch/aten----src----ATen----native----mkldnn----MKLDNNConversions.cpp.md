# `.\pytorch\aten\src\ATen\native\mkldnn\MKLDNNConversions.cpp`

```py
    // 定义预处理器宏，用于仅支持方法操作符
    #define TORCH_ASSERT_ONLY_METHOD_OPERATORS
    // 包含 ATen 库的配置文件
    #include <ATen/Config.h>
    // 包含 ATen 核心张量类定义
    #include <ATen/core/Tensor.h>
    // 包含 MKLDNN 相关的通用函数和定义
    #include <ATen/native/mkldnn/MKLDNNCommon.h>
    // 包含 MKLDNN 的实用函数
    #include <ATen/native/mkldnn/Utils.h>
    // 包含 ATen 原生工具的参数处理函数
    #include <ATen/native/utils/ParamUtils.h>
    // 包含 Torch 库的头文件
    #include <torch/library.h>

    // 如果未定义每个运算符头文件，则包含一般操作函数的头文件
    #ifndef AT_PER_OPERATOR_HEADERS
    #include <ATen/Functions.h>
    #include <ATen/NativeFunctions.h>
    // 否则，包含特定运算符的头文件
    #else
    #include <ATen/ops/_to_dense_native.h>
    #include <ATen/ops/empty.h>
    #include <ATen/ops/empty_like.h>
    #include <ATen/ops/empty_native.h>
    #include <ATen/ops/from_blob.h>
    #include <ATen/ops/mkldnn_reorder_conv2d_weight_native.h>
    #include <ATen/ops/mkldnn_reorder_conv3d_weight_native.h>
    #include <ATen/ops/to_mkldnn_native.h>
    #include <ATen/ops/zeros.h>
    #endif

    // 进入 ATen native 命名空间
    namespace at { namespace native {

    // 如果 MKLDNN 被启用，则编写以下函数
    #if AT_MKLDNN_ENABLED()

    // 定义函数 mkldnn_to_dense，将 MKLDNN 张量转换为稠密 CPU 张量
    Tensor mkldnn_to_dense(const Tensor& mkldnn_tensor, std::optional<ScalarType> dtype, std::optional<bool> masked_grad) {
      // 检查输入 MKLDNN 张量的标量类型是否支持转换
      TORCH_CHECK(mkldnn_tensor.scalar_type() == ScalarType::Float ||
                  mkldnn_tensor.scalar_type() == ScalarType::BFloat16 ||
                  mkldnn_tensor.scalar_type() == ScalarType::Half ||
                  mkldnn_tensor.scalar_type() == ScalarType::Byte ||
                  mkldnn_tensor.scalar_type() == ScalarType::Char,
                  "mkldnn_to_dense expects float, bfloat16, half, uint8, int8 tensor input");

      // 从 MKLDNN 张量获取 ideep::tensor 引用
      ideep::tensor& stensor = itensor_from_mkldnn(mkldnn_tensor);
      // 获取 ideep 张量的维度
      auto dims = stensor.get_dims();
      // 确定目标数据类型，若未提供则使用输入 MKLDNN 张量的数据类型
      auto data_type = dtype.has_value() ? dtype.value() : mkldnn_tensor.scalar_type();
      // 检查目标数据类型是否支持转换为 CPU 张量
      TORCH_CHECK(data_type == ScalarType::Float ||
                  data_type == ScalarType::BFloat16 ||
                  data_type == ScalarType::Half ||
                  data_type == ScalarType::Byte ||
                  data_type == ScalarType::Char,
                  "mkldnn tensor only can be converted to be a float, bfloat16, Half, uint8, int8 cpu tensor")

      // 若输入为 uint8 或 int8，则不应更改数据类型
      if (mkldnn_tensor.scalar_type() == ScalarType::Byte || mkldnn_tensor.scalar_type() == ScalarType::Char) {
        // 对于 int8 和 uint8 输入，不应更改数据类型
        TORCH_CHECK(mkldnn_tensor.scalar_type() == data_type,
                "For int8, uint8 mkldnn_tensor input, we should not change the data type.");
      }

      // 注意：ideep::tensor 的维度是 int32_t，但 ATen 要求的是 int64_t
      // 创建一个空的 CPU 张量，维度由 ideep::tensor 的维度转换而来
      Tensor cpu_tensor = at::empty(
        std::vector<int64_t>(dims.begin(), dims.end()),
    mkldnn_tensor.options().layout(c10::kStrided).dtype(data_type));
    # 设置 mkldnn_tensor 的选项，包括布局和数据类型

  if (stensor.is_empty()) return cpu_tensor;
  # 如果 stensor 是空的，直接返回 cpu_tensor

  auto pub_tensor =
      data_type == ScalarType::Float
      ? stensor.to_public(cpu_tensor.template data_ptr<float>(),
                          ideep::tensor::data_type::f32)
      : (data_type == ScalarType::BFloat16
         ? stensor.to_public(cpu_tensor.template data_ptr<BFloat16>(),
                         ideep::tensor::data_type::bf16)
         : (data_type == ScalarType::Half
            ? stensor.to_public(cpu_tensor.template data_ptr<Half>(),
                            ideep::tensor::data_type::f16)
          : (data_type == ScalarType::Byte
              ? stensor.to_public(cpu_tensor.template data_ptr<uint8_t>(),
                              ideep::tensor::data_type::u8)
              : stensor.to_public(cpu_tensor.template data_ptr<int8_t>(),
                              ideep::tensor::data_type::s8)
            )
           )
      );
  # 根据不同的数据类型，将 stensor 转换为公共格式的 pub_tensor

  cpu_tensor.as_strided_(dims, pub_tensor.get_strides());
  # 使用 pub_tensor 的步幅信息对 cpu_tensor 进行重构

  // Make sure that NC11 strides follow formula of contiguous tensor.
  // 确保 NC11 步幅遵循连续张量的公式。
  # 注释: 确保 NC11 步幅满足连续张量的要求。

  return cpu_tensor.contiguous().resize_(dims, c10::MemoryFormat::Contiguous);
  # 返回保证连续性的 cpu_tensor，并按给定的尺寸和内存格式调整大小
}

Tensor dense_to_mkldnn(const Tensor& cpu_tensor, std::optional<ScalarType> dtype) {
    // 检查输入的 CPU tensor 是否在 CPU 设备上
    TORCH_CHECK(cpu_tensor.device().is_cpu(),
               "dense_to_mkldnn expects CPU tensor input");
    // 检查输入的 CPU tensor 是否是连续布局
    TORCH_CHECK(cpu_tensor.layout() == Layout::Strided,
               "dense_to_mkldnn expects strided tensor input");
    // 检查输入的 CPU tensor 数据类型是否为支持的类型之一
    TORCH_CHECK(cpu_tensor.scalar_type() == ScalarType::Float ||
                cpu_tensor.scalar_type() == ScalarType::BFloat16 ||
                cpu_tensor.scalar_type() == ScalarType::Half ||
                cpu_tensor.scalar_type() == ScalarType::Byte ||
                cpu_tensor.scalar_type() == ScalarType::Char,
               "dense_to_mkldnn expects float, bfloat16, half, uint8, int8 tensor input");
    // 检查输入的 CPU tensor 维度是否小于等于5
    TORCH_CHECK(cpu_tensor.dim() <= 5,
               "Can't convert cpu tensor with the number of dimensions > 5");
    // NOTE: 禁止直接从非连续（或通道末尾）转换为 `ideep::tensor`。
    auto cpu_tensor_cont = cpu_tensor.contiguous();
    // 确定要转换的数据类型，如果未指定则使用输入 tensor 的数据类型
    auto data_type = dtype.has_value() ? dtype.value() : cpu_tensor.scalar_type();
    if (cpu_tensor.scalar_type() == ScalarType::Byte || cpu_tensor.scalar_type() == ScalarType::Char) {
        // 对于 int8 和 uint8 输入，不应更改数据类型。
        TORCH_CHECK(cpu_tensor.scalar_type() == data_type,
                "For int8, uint8 cpu_tensor input, we should not change the data type.");
    }
    // 检查目标数据类型是否为支持的类型之一
    TORCH_CHECK(data_type == ScalarType::Float ||
                data_type == ScalarType::BFloat16 ||
                data_type == ScalarType::Half ||
                data_type == ScalarType::Byte ||
                data_type == ScalarType::Char,
                "cpu tensor only can be converted to be a float, bfloat16, half, uint8, int8 mkldnn tensor")
    // 创建一个空的 MKL-DNN tensor，与输入 tensor 具有相同的大小和设备等属性
    Tensor mkldnn_tensor = empty_mkldnn(cpu_tensor_cont.sizes(), data_type,
                                        cpu_tensor_cont.options().layout_opt(), cpu_tensor_cont.options().device_opt(),
                                        cpu_tensor_cont.options().pinned_memory_opt());
    // 从 MKL-DNN tensor 获取底层的 ideep::tensor 引用
    ideep::tensor& dtensor = itensor_from_mkldnn(mkldnn_tensor);
    // 根据输入 tensor 的数据类型，将数据复制到 MKL-DNN tensor 中
    if (cpu_tensor.scalar_type() == ScalarType::Float) {
        dtensor.feed_from(dtensor.get_dims(),
                          ideep::tensor::data_type::f32,
                          (cpu_tensor_cont.template data_ptr<float>()));
    } else if (cpu_tensor.scalar_type() == ScalarType::BFloat16) {
        dtensor.feed_from(dtensor.get_dims(),
                          ideep::tensor::data_type::bf16,
                          cpu_tensor_cont.template data_ptr<BFloat16>());
    } else if (cpu_tensor.scalar_type() == ScalarType::Half) {
        dtensor.feed_from(dtensor.get_dims(),
                          ideep::tensor::data_type::f16,
                          cpu_tensor_cont.template data_ptr<Half>());
    } else if (cpu_tensor.scalar_type() == ScalarType::Byte) {
        dtensor.feed_from(dtensor.get_dims(),
                          ideep::tensor::data_type::u8,
                          cpu_tensor_cont.template data_ptr<uint8_t>());
    } else {
    // 检查 CPU 上的张量是否是 int8 类型，否则抛出错误信息
    TORCH_CHECK(cpu_tensor.scalar_type() == ScalarType::Char,
            "Expect int8 input of cpu_tensor");
    // 使用 IDEEP 库中的张量对象 dtensor，根据给定的维度和数据类型 s8，填充数据
    dtensor.feed_from(dtensor.get_dims(),
                      ideep::tensor::data_type::s8,
                      cpu_tensor_cont.template data_ptr<int8_t>());
  }
  // 返回创建好的 MKL-DNN 张量对象
  return mkldnn_tensor;
// Mkldnn tensor has special non-public format for conv2d weights
// (dense_to_mkldnn only converts dense tensor to mkldnn tensor with
// public format). Ideep conv kernel will do implicit reorder if the
// weight is not already in this optimized format. By the time I'm
// writing this note, we are seeing ~20% perf cost of doing the
// on-the-fly reorder.
// 定义函数 mkldnn_reorder_conv2d_weight，用于重新排序二维卷积权重张量
Tensor mkldnn_reorder_conv2d_weight(
    const Tensor& self,                     // 输入参数：表示原始权重张量
    IntArrayRef padding,                    // 输入参数：填充数组
    IntArrayRef stride,                     // 输入参数：步幅数组
    IntArrayRef dilation,                   // 输入参数：膨胀数组
    int64_t groups,                         // 输入参数：分组数
    c10::OptionalArrayRef<int64_t> input_size) {  // 输入参数：可选的输入大小数组

  mkldnn_check_low_precision(self.scalar_type(), "mkldnn_reorder_conv2d_weight");  // 检查低精度
  const auto padding_expanded = expand_param_if_needed(padding, "padding", 2);     // 根据需要扩展填充参数
  const auto stride_expanded = expand_param_if_needed(stride, "stride", 2);        // 根据需要扩展步幅参数
  const auto dilation_expanded = expand_param_if_needed(dilation, "dilation", 2);  // 根据需要扩展膨胀参数

  ideep::dims src_dims = ideep::dims();                     // 创建空的 ideep::dims 对象
  bool is_channels_last = false;                            // 是否采用通道最后的内存格式
  auto memory_format = at::MemoryFormat::Contiguous;         // 默认使用连续的内存格式
  if (input_size.has_value()) {
    src_dims = input_size.value().vec();                    // 获取输入大小，并转换为向量形式
    // 如果有输入大小，始终使用通道最后的内存格式
    is_channels_last = true;
    memory_format = at::MemoryFormat::ChannelsLast;
  }

  auto self_ = self.is_mkldnn() ? self : self.contiguous(memory_format);  // 如果不是 mkldnn 格式，则转换为指定的内存格式
  auto w = itensor_from_tensor(self_);                      // 将 Tensor 转换为 ideep::tensor

  // Legacy mkldnn conv2d jitted module may contain a 5-d weight with an extra
  // dimension when groups > 1, having dimension [g, o/g, i, h, w] instead of
  // [o, i, h, w]. Ideally we should reorder the weight back in serialization.
  // For backward compatibility, we squash the first two dims (g * o/g) back to
  // its original form.
  // 处理遗留的 mkldnn conv2d 模块，当 groups > 1 时，权重可能是 5 维的，
  // 其中第一维为组数 g，第二维为 o/g，需要将其重新调整为标准形式 [o, i, h, w]
  if (w.ndims() == 5) {
    auto wdims = w.get_dims();                             // 获取当前张量的维度
    w.reshape({wdims[0] * wdims[1], wdims[2], wdims[3], wdims[4]});  // 重新调整为标准形式
  }

  // 创建卷积前向传播期望的权重描述符
  auto desc = ideep::convolution_forward::expected_weights_desc(
      w.get_dims(),                                         // 权重张量的维度
      w.get_data_type(),                                    // 权重张量的数据类型
      stride_expanded,                                      // 扩展后的步幅数组
      padding_expanded,                                     // 扩展后的填充数组
      padding_expanded,                                     // 扩展后的填充数组（这里使用了两次，可能是错误）
      dilation_expanded,                                    // 扩展后的膨胀数组
      groups,                                               // 分组数
      ideep::algorithm::convolution_direct,                 // 卷积算法
      ideep::prop_kind::forward,                            // 前向传播
      w.get_data_type(),                                    // 输出数据类型
      src_dims,                                             // 输入大小
      ideep::attr_t(),                                      // 属性
      is_channels_last);                                    // 是否通道最后的内存格式

  ideep::tensor result;                                     // 创建结果张量对象
  result.init(desc);                                        // 根据描述符初始化结果张量
  result.feed_from(w);                                      // 将 w 张量数据填充到结果张量中

  // 使用新的 itensor_mkldnn 创建一个新的 Tensor 对象，并返回
  return new_with_itensor_mkldnn(std::move(result), optTypeMetaToScalarType(self.options().dtype_opt()),
                                 self.options().device_opt());
}
  // 检查张量的低精度，并确保不是低精度类型
  mkldnn_check_low_precision(self.scalar_type(), "mkldnn_reorder_conv3d_weight");

  // 根据需要扩展填充参数，使其成为长度为 3 的数组
  const auto padding_expanded = expand_param_if_needed(padding, "padding", 3);

  // 根据需要扩展步幅参数，使其成为长度为 3 的数组
  const auto stride_expanded = expand_param_if_needed(stride, "stride", 3);

  // 根据需要扩展膨胀参数，使其成为长度为 3 的数组
  const auto dilation_expanded = expand_param_if_needed(dilation, "dilation", 3);

  // 初始化输入数据的维度为空数组
  ideep::dims src_dims = ideep::dims();

  // 是否是通道最后的存储格式，默认为否
  bool is_channels_last = false;

  // 内存格式，默认为连续存储
  auto memory_format = at::MemoryFormat::Contiguous;

  // 如果提供了输入尺寸
  if (input_size.has_value()) {
    // 将输入尺寸转换为向量形式
    src_dims = input_size.value().vec();
    
    // 如果存在输入尺寸，则始终使用通道最后的存储格式
    is_channels_last = true;

    // 内存格式设置为三维通道最后
    memory_format = at::MemoryFormat::ChannelsLast3d;
  }

  // 如果输入张量是 MKL-DNN 张量，则使用原始张量 self，否则转换为指定的内存格式
  auto self_ = self.is_mkldnn() ? self : self.contiguous(memory_format);

  // 从张量获取 MKL-DNN 的 tensor 表示
  auto w = itensor_from_tensor(self_);

  // 根据期望的权重描述，创建卷积前向操作的描述符
  auto desc = ideep::convolution_forward::expected_weights_desc(
      w.get_dims(),
      w.get_data_type(),
      stride_expanded,
      padding_expanded,
      padding_expanded,
      dilation_expanded,
      groups,
      ideep::algorithm::convolution_direct,
      ideep::prop_kind::forward,
      w.get_data_type(),
      src_dims,
      ideep::attr_t(),
      is_channels_last);

  // 初始化一个新的 tensor 用于存储操作的结果
  ideep::tensor result;
  result.init(desc);

  // 将 w 的数据填充到 result 中
  result.feed_from(w);

  // 使用 MKL-DNN 创建一个新的 tensor，并根据原张量的 dtype 和设备选项转换为相应类型
  return new_with_itensor_mkldnn(std::move(result), optTypeMetaToScalarType(self.options().dtype_opt()), self.options().device_opt());
}
// 定义静态函数 mkldnn_reorder_conv_weight，用于重新排列卷积权重张量
static Tensor mkldnn_reorder_conv_weight(
    const Tensor& self, // 输入参数 self，表示原始权重张量
    IntArrayRef padding, // 输入参数 padding，表示填充数组的引用
    IntArrayRef stride, // 输入参数 stride，表示步长数组的引用
    IntArrayRef dilation, // 输入参数 dilation，表示膨胀数组的引用
    int64_t groups, // 输入参数 groups，表示卷积分组数量
    c10::OptionalArrayRef<int64_t> input_size) { // 输入参数 input_size，表示输入尺寸的可选数组引用
  TORCH_CHECK((self.dim() == 4 || self.dim() == 5), "mkldnn_reorder_conv_weight only supports conv2d and conv3d");
  // 检查 self 张量的维度是否为 4 或 5，如果不是则抛出错误
  if (self.dim() == 4) {
    // 如果 self 的维度为 4，则调用 mkldnn_reorder_conv2d_weight 处理函数
    return at::native::mkldnn_reorder_conv2d_weight(self, padding, stride, dilation, groups, input_size);
  } else {
    // 如果 self 的维度为 5，则调用 mkldnn_reorder_conv3d_weight 处理函数
    return at::native::mkldnn_reorder_conv3d_weight(self, padding, stride, dilation, groups, input_size);
  }
}

// 定义静态函数 mkldnn_reorder_linear_weight，用于重新排列线性层权重张量
static Tensor mkldnn_reorder_linear_weight(
    const Tensor& self, // 输入参数 self，表示原始权重张量
    std::optional<int64_t> batch_size_opt) { // 输入参数 batch_size_opt，表示可选的批量大小
  mkldnn_check_low_precision(self.scalar_type(), "mkldnn_reorder_linear_weight");
  // 检查 self 张量的低精度情况
  auto out_features = self.size(0); // 计算输出特征数量
  auto in_features = self.size(1); // 计算输入特征数量
  auto self_ = self.contiguous(); // 创建 self 的连续版本
  auto w = itensor_from_tensor(self_); // 从 self_ 创建 ideep::tensor w
  ideep::dims input_size; // 定义输入尺寸
  auto dtype = w.get_data_type(); // 获取 w 的数据类型
  if (batch_size_opt.has_value()) {
    input_size = {batch_size_opt.value(), in_features}; // 如果存在批量大小值，则设置输入尺寸
  }
  // 创建期望的权重描述 packed_desc
  auto packed_desc = ideep::inner_product_forward::expected_weights_desc(
      {out_features, in_features}, // 输出特征和输入特征维度
      input_size, // 输入尺寸
      /* weight dtype */ dtype, // 权重数据类型
      /* src dtype */ dtype); // 源数据类型
  ideep::tensor result; // 创建 ideep::tensor result
  result.init(packed_desc); // 使用 packed_desc 初始化 result
  result.feed_from(w); // 将 w 的数据填充到 result 中
  // 返回一个新的带有 ideep::tensor 数据的 Tensor 对象
  return new_with_itensor_mkldnn(std::move(result), optTypeMetaToScalarType(self.options().dtype_opt()), self.options().device_opt());
}

// 定义静态函数 get_conv_transpose_expected_weights_desc，用于获取卷积转置期望的权重描述
static ideep::tensor::desc get_conv_transpose_expected_weights_desc(
    const ideep::tensor::dims& weights_dims, // 输入参数 weights_dims，表示权重维度
    ideep::tensor::data_type w_dtype, // 输入参数 w_dtype，表示权重数据类型
    const ideep::tensor::dims& strides, // 输入参数 strides，表示步长维度
    const ideep::tensor::dims& padding_l, // 输入参数 padding_l，表示左填充维度
    const ideep::tensor::dims& padding_r, // 输入参数 padding_r，表示右填充维度
    const ideep::tensor::dims& dilates, // 输入参数 dilates，表示膨胀维度
    int groups, // 输入参数 groups，表示卷积分组数量
    bool channels_last, // 输入参数 channels_last，表示是否通道在最后
    ideep::algorithm aalgorithm, // 输入参数 aalgorithm，表示算法类型
    ideep::data_type x_dtype, // 输入参数 x_dtype，表示源数据类型
    const ideep::dims& src_dims) { // 输入参数 src_dims，表示源维度
  if (channels_last) {
    // 如果 channels_last 为真，则返回通道在最后的卷积转置期望权重描述
    return ideep::convolution_transpose_forward::expected_weights_desc<true>(
        weights_dims, // 权重维度
        w_dtype, // 权重数据类型
        strides, // 步长维度
        padding_l, // 左填充维度
        padding_r, // 右填充维度
        dilates, // 膨胀维度
        groups, // 卷积分组数量
        aalgorithm, // 算法类型
        ideep::prop_kind::forward, // 属性类型（前向）
        src_dims); // 源维度
  } else {
    // 如果 channels_last 为假，则返回通道在前的卷积转置期望权重描述
    return ideep::convolution_transpose_forward::expected_weights_desc<false>(
        weights_dims, // 权重维度
        w_dtype, // 权重数据类型
        strides, // 步长维度
        padding_l, // 左填充维度
        padding_r, // 右填充维度
        dilates, // 膨胀维度
        groups, // 卷积分组数量
        aalgorithm, // 算法类型
        ideep::prop_kind::forward, // 属性类型（前向）
        src_dims); // 源维度
  }
}

// 定义静态函数 mkldnn_reorder_conv_transpose_weight，用于重新排列卷积转置权重张量
static Tensor mkldnn_reorder_conv_transpose_weight(
    const Tensor& self, // 输入参数 self，表示原始权重张量
    IntArrayRef padding, // 输入参数 padding，表示填充数组的引用
    IntArrayRef output_padding, // 输入参数 output_padding，表示输出填充数组的引用
    IntArrayRef stride, // 输入参数 stride，表示步长数组的引用
    IntArrayRef dilation, // 输入参数 dilation，表示膨胀数组的引用
    int64_t groups, // 输入参数 groups，表示卷积分组数量
    // 检查输入张量的维度是否为4或5，仅支持conv_transpose2d和conv_transpose3d操作
    TORCH_CHECK(
        (self.dim() == 4 || self.dim() == 5),
        "mkldnn_reorder_conv_transpose_weight only supports conv_transpose2d and conv_transpose3d");
    
    // 排除自动求导键集以加速操作
    c10::impl::ExcludeDispatchKeyGuard edkg(c10::autograd_dispatch_keyset);
    
    // 检查低精度，确保张量的数据类型适用于MKL-DNN操作
    mkldnn_check_low_precision(
        self.scalar_type(), "mkldnn_reorder_conv_transpose_weight");
    
    // 计算空间维度，维度减去2用于卷积转置操作
    int64_t pdim = self.dim() - 2;
    
    // 根据需要扩展padding参数
    const auto padding_expanded =
        expand_param_if_needed(padding, "padding", pdim);
    // 根据需要扩展stride参数
    const auto stride_expanded = expand_param_if_needed(stride, "stride", pdim);
    // 根据需要扩展dilation参数
    const auto dilation_expanded =
        expand_param_if_needed(dilation, "dilation", pdim);
    // 根据需要扩展output_padding参数
    const auto output_padding_expanded =
        expand_param_if_needed(output_padding, "output_padding", pdim);
    
    // 初始化源张量的维度信息为空
    ideep::dims src_dims = ideep::dims();
    // 是否使用通道为最后维度的内存格式
    bool is_channels_last = false;
    // 默认使用连续的内存格式
    auto memory_format = at::MemoryFormat::Contiguous;
    
    // 如果提供了输入大小信息
    if (input_size.has_value()) {
      // 获取输入大小向量
      src_dims = input_size.value().vec();
      // 如果有输入大小信息，则强制使用通道为最后维度的内存格式
      is_channels_last = true;
      // 根据张量维度设置内存格式为ChannelsLast或ChannelsLast3d
      memory_format = self.dim() == 4 ? at::MemoryFormat::ChannelsLast
                                      : at::MemoryFormat::ChannelsLast3d;
    }
    
    // 将张量转换为指定的内存格式（连续或通道为最后维度）
    auto self_ = self.contiguous(memory_format);
    // 从PyTorch张量创建MKL-DNN张量
    ideep::tensor w = itensor_from_tensor(self_);
    
    // 获取期望的卷积转置权重描述符
    auto expected_desc = get_conv_transpose_expected_weights_desc(
        w.get_dims(),
        w.get_data_type(),
        stride_expanded,
        padding_expanded,
        padding_r(padding_expanded, output_padding_expanded),
        dilation_expanded,
        groups,
        is_channels_last,
        ideep::algorithm::deconvolution_direct,
        w.get_data_type(),
        src_dims);
    
    // 如果groups大于1，交换权重描述符的第1和第2维度
    if (groups > 1) {
      expected_desc = expected_desc.transpose(1, 2);
    } else {
      // 否则交换权重描述符的第0和第1维度
      expected_desc = expected_desc.transpose(0, 1);
    }
    
    // 初始化结果张量
    ideep::tensor result;
    result.init(expected_desc);
    // 转置原始权重张量的第0和第1维度
    w.transpose_(0, 1);
    // 将转置后的权重张量填充到结果张量中，标记为反卷积权重
    result.feed_from(w, /*is_deconv_weights*/true);
    
    // 使用MKL-DNN张量创建新的PyTorch张量，并返回
    return new_with_itensor_mkldnn(std::move(result), optTypeMetaToScalarType(self.options().dtype_opt()),
                                   self.options().device_opt());
}

static std::tuple<ideep::tensor, ideep::tensor> get_lstm_packed_weights(
    const at::Tensor& weight_ih,                          // 输入参数：输入到隐藏层的权重张量
    const at::Tensor& weight_hh,                          // 输入参数：隐藏到隐藏层的权重张量
    const at::Tensor& weight2,                            // 输入参数：第二个权重张量
    const at::Tensor& weight3,                            // 输入参数：第三个权重张量
    int64_t layer_feature_size,                           // 输入参数：层特征大小
    int64_t hidden_size,                                  // 输入参数：隐藏单元大小
    bool has_biases,                                      // 输入参数：是否有偏置
    int64_t num_layers,                                   // 输入参数：层数
    bool bidirectional,                                   // 输入参数：是否双向
    int64_t time_step,                                    // 输入参数：时间步数
    int64_t batch_size,                                   // 输入参数：批量大小
    bool reverse) {                                       // 输入参数：是否反向

  ideep::tensor cached_weight_ih, cached_weight_hh;       // 缓存的权重张量对象

  int64_t num_gates = 4;                                  // 门的数量
  int64_t num_bias_gates = 4;                             // 偏置门的数量
  std::vector<int64_t> output_sizes = {time_step, batch_size, hidden_size};  // 输出大小的向量

  auto dtype = get_mkldnn_dtype(weight_ih.scalar_type());  // 获取权重张量数据类型
  ideep::tensor::desc src_layer_desc({time_step, batch_size, layer_feature_size}, dtype, ideep::format_tag::tnc);  // 源层描述
  ideep::tensor::desc src_iter_desc({1, 1, batch_size, hidden_size}, dtype, ideep::format_tag::ldnc);                // 源迭代器描述
  ideep::tensor::desc src_iter_c_desc({1, 1, batch_size, hidden_size}, dtype, ideep::format_tag::ldnc);              // 源迭代器C描述
  ideep::tensor::desc bias_desc({1, 1, num_bias_gates, hidden_size}, dtype, ideep::format_tag::ldgo);                // 偏置描述

  ideep::tensor::desc dst_layer_desc({time_step, batch_size, hidden_size}, dtype, ideep::format_tag::tnc);           // 目标层描述
  ideep::tensor::desc dst_iter_desc({1, 1, batch_size, hidden_size}, dtype, ideep::format_tag::ldnc);                // 目标迭代器描述
  ideep::tensor::desc dst_iter_c_desc({1, 1, batch_size, hidden_size}, dtype, ideep::format_tag::ldnc);              // 目标迭代器C描述

  ideep::tensor src_layer(src_layer_desc);                 // 源层张量
  ideep::tensor src_iter(src_iter_desc);                   // 源迭代器张量
  ideep::tensor src_iter_c(src_iter_c_desc);               // 源迭代器C张量
  ideep::tensor bias(bias_desc);                           // 偏置张量

  auto w1 = itensor_view_from_dense(
      weight_ih,
      {{1, 1, layer_feature_size, num_gates, hidden_size},
        get_mkldnn_dtype(weight_ih.scalar_type()),
        ideep::format_tag::ldgoi});                       // 从稠密张量创建视图w1

  auto w2 = itensor_view_from_dense(
      weight_hh,
      {{1, 1, hidden_size, num_gates, hidden_size},
        get_mkldnn_dtype(weight_hh.scalar_type()),
        ideep::format_tag::ldgoi});                       // 从稠密张量创建视图w2

  auto [packed_desc_ih, packed_desc_hh] =
      ideep::lstm_forward_inference::expected_weights_desc(
          output_sizes,
          src_layer,
          src_iter,
          src_iter_c,
          w1,
          w2,
          bias,
          reverse);                                        // 获取期望的LSTM前向推理权重描述

  cached_weight_ih.init(packed_desc_ih);                    // 初始化缓存的权重张量ih
  cached_weight_hh.init(packed_desc_hh);                    // 初始化缓存的权重张量hh

  cached_weight_ih.feed_from(w1);                           // 从w1填充缓存的权重ih
  cached_weight_hh.feed_from(w2);                           // 从w2填充缓存的权重hh

  return std::make_tuple(cached_weight_ih, cached_weight_hh);  // 返回缓存的权重ih和hh的元组
}

static bool should_use_plain_format(ideep::tensor w) {
#if defined(IDEEP_VERSION_MAJOR) && IDEEP_VERSION_MAJOR>=3
  return w.get_desc().is_opaque() || w.get_desc().is_plain();  // 如果描述是不透明或平面格式，返回true
# else
  return w.get_desc().is_rnn_packed() || w.get_desc().is_plain();  // 如果描述是RNN打包或平面格式，返回true
#endif
}

static std::vector<Tensor> mkldnn_reorder_mkldnn_rnn_layer_weight(
 Tensor weight0,                                           // 输入参数：权重0张量
 Tensor weight1,                                           // 输入参数：权重1张量
 int64_t hidden_size,                                      // 输入参数：隐藏单元大小
 bool reverse,                                              // 输入参数：是否反向
 bool has_biases,                                           // 输入参数：是否有偏置
 bool batch_first,                                          // 输入参数：是否批量优先
 c10::OptionalArrayRef<int64_t> input_size) {               // 输入参数：可选的输入大小数组引用

  std::vector<int64_t> input_size_value;                    // 输入大小值向量
  int64_t time_step, batch_size;                            // 时间步数和批量大小
  if (input_size.has_value()) {                             // 如果输入大小有值，
    // 获取输入大小的向量值
    input_size_value = input_size.value().vec();
    // 根据是否批处理优先(batch_first)确定时间索引
    int64_t time_index = batch_first ? 1 : 0;
    // 根据是否批处理优先(batch_first)确定批大小索引
    int64_t batch_size_index = batch_first ? 0 : 1;

    // 根据时间索引获取时间步长
    time_step = input_size_value[time_index];
    // 根据批大小索引获取批大小
    batch_size = input_size_value[batch_size_index];
  } else {
    // 如果没有输入值，则提供默认值
    // 默认时间步长为5
    time_step = 5;
    // 默认批大小为10
    batch_size = 10;
  }

  // 声明用于存储打包后权重的张量
  at::Tensor packed_w1, packed_w2;

  // 获取权重的特征大小
  int64_t feature_size = weight0.size(-1);

  // 调用函数获取打包后的LSTM权重w1_和w2_
  auto [w1_, w2_] = get_lstm_packed_weights(
    weight0,
    weight1,
    at::zeros(
      weight0.sizes(),
      weight0.options()),
    at::zeros(
      weight1.sizes(),
      weight1.options()),
    feature_size,
    hidden_size,
    has_biases, // 是否有偏置
    1, // 层数
    false, // 是否双向
    time_step,
    batch_size,
    reverse);

  // 根据条件判断是否使用普通格式处理w1_
  if (should_use_plain_format(w1_)) {
    packed_w1 = weight0;
  } else {
    // 使用特定格式创建张量packed_w1
    packed_w1 = new_with_itensor_mkldnn(std::move(w1_), optTypeMetaToScalarType(weight0.options().dtype_opt()), weight0.options().device_opt());
  }

  // 根据条件判断是否使用普通格式处理w2_
  if (should_use_plain_format(w2_)) {
    packed_w2 = weight1;
  } else {
    // 使用特定格式创建张量packed_w2
    packed_w2 = new_with_itensor_mkldnn(std::move(w2_), optTypeMetaToScalarType(weight1.options().dtype_opt()), weight1.options().device_opt());
  }

  // 返回打包后的权重张量packed_w1和packed_w2
  return {packed_w1, packed_w2};
#else

// 如果未启用 MKL-DNN 构建，则定义以下函数来抛出错误消息
Tensor mkldnn_to_dense(const Tensor& mkldnn_tensor, std::optional<ScalarType> dtype, std::optional<bool> masked_grad) {
  // 抛出错误消息，指示 MKL-DNN 构建已禁用
  TORCH_CHECK(false, "MKL-DNN build is disabled");
}

// 如果未启用 MKL-DNN 构建，则定义以下函数来抛出错误消息
Tensor dense_to_mkldnn(const Tensor& cpu_tensor, std::optional<ScalarType> dtype) {
  // 抛出错误消息，指示 MKL-DNN 构建已禁用
  TORCH_CHECK(false, "MKL-DNN build is disabled");
}

// 如果未启用 MKL-DNN 构建，则定义以下函数来抛出错误消息
Tensor mkldnn_reorder_conv2d_weight(
    const Tensor& self,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    c10::OptionalArrayRef<int64_t> input_size) {
  // 抛出错误消息，指示 MKL-DNN 构建已禁用
  TORCH_CHECK(false, "mkldnn_reorder_conv2d_weight: MKL-DNN build is disabled");
}

// 如果未启用 MKL-DNN 构建，则定义以下函数来抛出错误消息
Tensor mkldnn_reorder_conv3d_weight(
    const Tensor& self,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    c10::OptionalArrayRef<int64_t> input_size) {
  // 抛出错误消息，指示 MKL-DNN 构建已禁用
  TORCH_CHECK(false, "mkldnn_reorder_conv3d_weight: MKL-DNN build is disabled");
}

#endif // AT_MKLDNN_ENABLED()

#if AT_MKL_ENABLED() && AT_MKLDNN_ENABLED()

// 如果启用了 MKL 和 MKL-DNN，则包含 MKL 头文件
#include <mkl.h>

// 定义静态函数以重新排列线性层权重
static Tensor mkl_reorder_linear_weight(
    const Tensor& weight,
    ```
    // 检查权重张量的数据类型是否为 float，否则抛出错误信息
    TORCH_CHECK(
        weight.scalar_type() == ScalarType::Float,
        "reorder_linear_weight: weight's dtype should be float");
    
    // 临时排除自动求导的调度键，以便进行非自动求导操作
    c10::impl::ExcludeDispatchKeyGuard edkg(c10::autograd_dispatch_keyset);
    
    // 获取权重张量的维度信息
    auto M = batch_size; // M 表示批次大小
    auto N = weight.size(0); // N 表示权重张量的第一维大小
    auto K = weight.size(1); // K 表示权重张量的第二维大小
    
    // 计算用于打包权重的缓冲区大小，单位是 float
    int64_t pack_size =
        (int64_t)(cblas_sgemm_pack_get_size(CblasBMatrix, M, N, K) / sizeof(float) + 1);
    
    // 创建一个空的 MKLDNN 张量来存储打包后的权重
    auto packed_weight = empty_mkldnn(
        {pack_size, 1}, // 设置张量的形状为 (pack_size, 1)
        weight.scalar_type(), // 使用权重张量的数据类型
        weight.options().layout_opt(), // 使用权重张量的布局选项
        weight.options().device_opt(), // 使用权重张量的设备选项
        weight.options().pinned_memory_opt()); // 使用权重张量的固定内存选项
    
    // 将 MKLDNN 张量转换为 IDEEP 张量以便后续操作
    ideep::tensor& mkl_weight = itensor_from_mkldnn(packed_weight);
    
    // 使权重张量连续化以确保能够正确使用 cblas_sgemm_pack 函数
    auto weight_ = weight.contiguous();
    
    // 从密集张量获取 IDEEP 张量的视图
    const ideep::tensor orig_w = itensor_view_from_dense(weight_);
    
    // 使用 cblas 函数将原始权重数据打包到 MKLDNN 张量中
    cblas_sgemm_pack(
        CblasRowMajor, // 使用行主序存储
        CblasBMatrix, // 使用 B 矩阵作为打包类型
        CblasTrans, // 对原始权重进行转置
        M, // 行数 M
        N, // 列数 N
        K, // 原始权重的列数 K
        1.0f, // 缩放因子
        (float*)(orig_w.get_data_handle()), // 原始权重数据的指针
        K, // 原始权重的列数
        (float*)(mkl_weight.get_data_handle())); // 目标 MKLDNN 张量的数据指针
    
    // 返回打包后的 MKLDNN 张量
    return packed_weight;
}

TORCH_LIBRARY_IMPL(mkl, CPU, m) {
  // 定义 Torch 库的实现，使用 mkl 库在 CPU 上
  m.impl(
    TORCH_SELECTIVE_NAME("mkl::_mkl_reorder_linear_weight"),
    TORCH_FN(mkl_reorder_linear_weight));
}

#endif // AT_MKL_ENABLED && AT_MKLDNN_ENABLED

}}
```