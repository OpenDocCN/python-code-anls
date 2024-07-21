# `.\pytorch\aten\src\ATen\native\quantized\cpu\LinearUnpackImpl.cpp`

```
#ifdef USE_PYTORCH_QNNPACK
// 定义 unpack 方法，解压缩量化后的线性权重和偏置
std::tuple<at::Tensor, std::optional<at::Tensor>> PackedLinearWeightsQnnp::
    unpack() {
  // 如果原始权重已定义，直接返回原始权重和偏置
  if (orig_weight.defined()) {
    return std::tuple<at::Tensor, std::optional<at::Tensor>>(
        orig_weight, bias_);
  } else {
    // 否则，进行解压缩操作，参考 QnnpackUtils.h 中的 make_zero_points_and_scales_tensor 函数详细机制
    // 参考链接：https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/quantized/cpu/QnnpackUtils.h#L469
    // w_scales 和 w_zero_points 与原始的缩放因子和零点有所不同，包括填充和类型转换等
    at::Tensor weight_origin;

    // 获取权重缩放因子数据
    float* weight_scales_data = w_scales.data_ptr<float>();
    // 如果量化方案为每张量仿射量化
    if (q_scheme == c10::kPerTensorAffine) {
      // 创建一个仿射量化的空张量，指定大小、设备、数据类型、缩放因子和零点调整
      weight_origin = at::_empty_affine_quantized(
          weight_sizes,
          at::device(c10::kCPU).dtype(c10::kQInt8),
          static_cast<double>(weight_scales_data[0]),
          (int64_t)w_zero_points[0] - 128);
    // 如果量化方案为每通道仿射量化（Per Channel Affine）
    } else if (q_scheme == c10::kPerChannelAffine) {
      // 从给定的数据指针创建张量 scales，使用 kPaddingChannels 调整通道数
      auto scales = at::from_blob(
          weight_scales_data,
          w_scales.sizes()[0] - kPaddingChannels,
          device(c10::kCPU).dtype(c10::kFloat));

      // 创建一个空张量 zero_points，用于存储零点值，调整通道数并指定数据类型为 int64_t
      at::Tensor zero_points = at::empty(
          w_zero_points.size() - kPaddingChannels, at::device(c10::kCPU).dtype(c10::kLong));
      
      // 遍历 zero_points 的每个元素，计算调整后的零点值并赋值
      for (const auto i : c10::irange(zero_points.numel())) {
        zero_points[i] = ((int64_t)w_zero_points[i] - 128);
      }

      // 使用仿射量化的方式创建 weight_origin 张量，基于 scales 和 zero_points
      weight_origin = at::_empty_per_channel_affine_quantized(
                          weight_sizes,
                          scales,
                          zero_points.toType(c10::kLong),
                          0, // 输出通道轴为 0
                          device(c10::kCPU).dtype(c10::kQInt8))
                          .contiguous();
    } else {
      // 如果量化方案不支持，则抛出内部断言错误信息
      TORCH_INTERNAL_ASSERT(false, "Unsupported quantization scheme.");
    }

    // 将 weight_origin 转换为 int8_t 指针类型
    int8_t* weight_ptr_int8 =
        reinterpret_cast<int8_t*>(weight_origin.data_ptr<c10::qint8>());

    // 解包权重 w，传递零点数据和 int8 指针
    w->unpackWeights(w_zero_points.data(), weight_ptr_int8);

    // 对 weight_ptr_int8 指向的数据进行减去 128 的操作，详细参考给定的 GitHub 链接
    auto wt_numel = weight_origin.numel();
    for (const auto i : c10::irange(wt_numel)) {
      weight_ptr_int8[i] = (int8_t)(weight_ptr_int8[i] - 128);
    }

    // 返回包含 weight_origin 和 bias_ 的 std::tuple
    return std::tuple<at::Tensor, std::optional<at::Tensor>>(
        weight_origin, bias_);
  }
#if AT_MKLDNN_ENABLED()
// 如果启用了 MKL-DNN 加速
std::tuple<at::Tensor, std::optional<at::Tensor>> PackedLinearWeightsOnednn::unpack() {
    // 返回原始权重和偏置的元组
    return std::tuple<at::Tensor, std::optional<at::Tensor>>(
        orig_weight_, orig_bias_);
}
#endif // #if AT_MKLDNN_ENABLED()
```