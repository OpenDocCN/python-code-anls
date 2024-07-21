# `.\pytorch\aten\src\ATen\native\quantized\cpu\qconv_dynamic.cpp`

```
#ifdef USE_PYTORCH_QNNPACK

template <int kSpatialDim>
// 定义一个模板类 PackedConvWeightsQnnp，实现动态应用
at::Tensor PackedConvWeightsQnnp<kSpatialDim>::apply_dynamic(
    // 接收一个输入张量 input 和一个布尔值 reduce_range
    const at::Tensor& input,
    bool reduce_range) {
  // 如果 reduce_range 为 true，则发出警告，因为当前 qnnpack 版本不正确处理此选项
  if (reduce_range) {
    TORCH_WARN("Currently, qnnpack incorrectly ignores reduce_range when it is set to true; this may change in a future release.");
  }

  // 如果输入张量为空，设置默认的 x_min 和 x_max 为 0
  float x_min = 0;
  float x_max = 0;
  // 否则...
  if (input.numel() > 0) {
    // 计算输入张量的最小值作为 x_min
    x_min = input.min().item<float>();
    // 计算输入张量的最大值，并转换为浮点数
    x_max = input.max().item<float>();
  }

  // 输入张量被量化为8位无符号整数值
  static constexpr int precision = 8;
  static constexpr bool is_signed = false;

  // 计算输入张量的量化参数：缩放因子和零点
  auto q_params = quant_utils::ChooseQuantizationParams(
      /*min=*/x_min,  // 最小值为 x_min
      /*max=*/x_max,  // 最大值为 x_max
      /*qmin=*/is_signed ? -(1 << (precision - 1)) : 0,  // 如果有符号，则量化最小值为 -(1 << (precision - 1))，否则为 0
      /*qmax=*/is_signed ? ((1 << (precision - 1)) - 1) : (1 << precision) - 1,  // 如果有符号，则量化最大值为 ((1 << (precision - 1)) - 1)，否则为 (1 << precision) - 1
      /*preserve_sparsity=*/false,  // 不保留稀疏性
      /*force_scale_power_of_two=*/false,  // 不强制缩放因子为2的幂
      /*reduce_range=*/false); // 注意：这里设置为 false，而不是 reduce_range，用于 qnnpack

  // 对输入进行量化
  at::Tensor q_input = at::quantize_per_tensor(
      input, q_params.scale, q_params.zero_point, c10::kQUInt8);

  // 应用量化操作
  at::Tensor out =
      apply_impl<false>(q_input, q_params.scale, q_params.zero_point);

  // 返回反量化的输出张量，TODO: 优化的内核会直接输出 fp32，因此这一步可能不必要
  return at::dequantize(out);
// note: this works for both Conv and ConvT due to transpose()
// 定义了一个模板类 QConvDynamicInt8，用于动态量化的卷积操作，支持多维空间维度 kSpatialDim
template <int kSpatialDim>
class QConvDynamicInt8 final {
 public:
  // 静态方法 run，接受输入张量 input，压缩后的权重 packed_weight，以及是否缩减范围 reduce_range
  static at::Tensor run(
      at::Tensor input,
      const c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>>&
          packed_weight,
      bool reduce_range) {
    // 调用 packed_weight 对象的 apply_dynamic 方法进行动态量化卷积操作，并返回结果张量
    return packed_weight->apply_dynamic(input, reduce_range);
  }
};

// note: this works for both Conv and ConvT due to transpose()
// 定义了一个特化类 QConv1dDynamicInt8，用于动态量化的一维卷积操作
class QConv1dDynamicInt8 final {
 public:
  // 静态方法 run，接受输入张量 input，压缩后的权重 packed_weight，以及是否缩减范围 reduce_range
  static at::Tensor run(
      at::Tensor input,
      const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
      bool reduce_range) {
    at::Tensor output;
    // 在维度 quant_utils::kConv1dSqueezeDim + 2 上对输入张量进行扩展，由 N, C, L 变为 N, C, 1, L
    input = input.unsqueeze(quant_utils::kConv1dSqueezeDim + 2);
    // 调用 packed_weight 对象的 apply_dynamic 方法进行动态量化的一维卷积操作，并返回结果张量
    output = packed_weight->apply_dynamic(input, reduce_range);
    // 在维度 quant_utils::kConv1dSqueezeDim + 2 上对输出张量进行压缩，由 N, C, 1, L 变为 N, C, L
    return output.squeeze_(quant_utils::kConv1dSqueezeDim + 2);
  }
};
// 定义 TORCH_LIBRARY_IMPL 宏，注册 quantized 模块的 CPU 实现
TORCH_LIBRARY_IMPL(quantized, CPU, m) {
    // 注册 quantized::conv1d_dynamic 函数，使用 QConv1dDynamicInt8::run 实现
    m.impl(
        TORCH_SELECTIVE_NAME("quantized::conv1d_dynamic"),
        TORCH_FN(QConv1dDynamicInt8::run));
    // 注册 quantized::conv2d_dynamic 函数，使用 QConvDynamicInt8<2>::run 实现
    m.impl(
        TORCH_SELECTIVE_NAME("quantized::conv2d_dynamic"),
        TORCH_FN(QConvDynamicInt8<2>::run));
    // 注册 quantized::conv3d_dynamic 函数，使用 QConvDynamicInt8<3>::run 实现
    m.impl(
        TORCH_SELECTIVE_NAME("quantized::conv3d_dynamic"),
        TORCH_FN(QConvDynamicInt8<3>::run));

    // transpose
    // 注册 quantized::conv_transpose1d_dynamic 函数，使用 QConv1dDynamicInt8::run 实现
    m.impl(
        TORCH_SELECTIVE_NAME("quantized::conv_transpose1d_dynamic"),
        TORCH_FN(QConv1dDynamicInt8::run));
    // 注册 quantized::conv_transpose2d_dynamic 函数，使用 QConvDynamicInt8<2>::run 实现
    m.impl(
        TORCH_SELECTIVE_NAME("quantized::conv_transpose2d_dynamic"),
        TORCH_FN(QConvDynamicInt8<2>::run));
    // 注册 quantized::conv_transpose3d_dynamic 函数，使用 QConvDynamicInt8<3>::run 实现
    m.impl(
        TORCH_SELECTIVE_NAME("quantized::conv_transpose3d_dynamic"),
        TORCH_FN(QConvDynamicInt8<3>::run));
}

// 结束 quantized 命名空间
} // namespace quantized

// 结束 native 命名空间
} // namespace native

// 结束 at 命名空间
} // namespace at
```