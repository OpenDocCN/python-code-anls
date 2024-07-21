# `.\pytorch\aten\src\ATen\native\ao_sparse\quantized\cpu\qlinear_deserialize.cpp`

```
/**
 * @brief Deserialize packed linear weight parameters.
 *
 * This function deserializes packed linear weight parameters from a serialized format.
 * It constructs an object representing these parameters using the provided tensor.
 * 
 * @param tensor Tensor containing serialized parameters.
 * @return Pointer to an intrusive_ptr of LinearPackedParamsBase representing the deserialized parameters.
 */
c10::intrusive_ptr<LinearPackedParamsBase> PackedLinearWeight::deserialize(



    at::Tensor tensor
    // 定义函数，接受一个 BCSR 序列化对象作为参数
    const BCSRSerializationType& serialized) {
      // 从序列化对象中提取输出特征块大小
      const int64_t out_features_block_size =
          std::get<out_features_block_size_index>(serialized);
      // 从序列化对象中提取输入特征块大小
      const int64_t in_features_block_size =
          std::get<in_features_block_size_index>(serialized);
      // 根据序列化对象中的量化方案索引确定量化方案，如果为真则使用每张量仿射量化，否则使用每通道仿射量化
      const c10::QScheme q_scheme = std::get<quantization_scheme_index>(serialized)
          ? c10::kPerTensorAffine
          : c10::kPerChannelAffine;
      // 从序列化对象中提取输出通道数
      const int64_t output_channels =
          std::get<num_output_channels_index>(serialized);
      // 从序列化对象中提取输入通道数
      const int64_t input_channels = std::get<num_input_channels_index>(serialized);
      // 解包未平铺的 BCSR，然后以平铺形式打包
      at::Tensor weight_origin;
      // 从序列化对象中提取权重零点张量
      const at::Tensor weight_zero_points =
          std::get<weight_zero_point_index>(serialized);
      // 根据量化方案选择相应的初始化函数来创建权重张量
      if (q_scheme == c10::kPerTensorAffine) {
        // 使用仿射量化方式创建空的张量 weight_origin
        weight_origin = at::_empty_affine_quantized(
            {output_channels, input_channels},
            at::device(c10::kCPU).dtype(c10::kQInt8),
            // 从序列化对象中提取权重比例尺，并使用其数据指针的第一个元素
            std::get<weight_scales_index>(serialized).data_ptr<float>()[0],
            // 使用权重零点数据张量的第一个元素
            weight_zero_points.data_ptr<int8_t>()[0]);
      } else if (q_scheme == c10::kPerChannelAffine) {
    // 创建一个空的张量 weight_origin，用于存储量化后的权重数据
    weight_origin = at::_empty_per_channel_affine_quantized(
        {output_channels, input_channels},  // 输出通道数和输入通道数作为形状参数
        std::get<weight_scales_index>(serialized),  // 获取权重缩放因子
        weight_zero_points,  // 权重零点
        0,  // 输出通道轴为0
        device(c10::kCPU).dtype(c10::kQInt8));  // 在CPU上创建 qint8 类型的张量

  }

  // 加载权重值的张量 loaded_weight_values
  const at::Tensor loaded_weight_values =
      std::get<weight_values_index>(serialized);
  // 获取加载权重值张量的数据指针
  const uint8_t* loaded_weight_values_ptr =
      loaded_weight_values.data_ptr<uint8_t>();
  // 加载权重值张量的元素数量
  const int64_t loaded_weight_values_size = loaded_weight_values.numel();
  // 减去 128，因为我们序列化时使用了 +128，这样有利于 QNNPack 减小内存占用
  // 创建一个 int8 类型的权重值向量 weight_values
  std::vector<int8_t> weight_values(loaded_weight_values_size);
  // 对加载的权重值进行转换，将 uint8_t 类型转换为 int8_t 类型，并减去 128
  std::transform(
      loaded_weight_values_ptr,
      loaded_weight_values_ptr + loaded_weight_values_size,
      weight_values.begin(),
      [](uint8_t v) {
        return static_cast<int8_t>(static_cast<int16_t>(v) - 128);
      });

  // 获取行块索引的张量和列块索引的张量
  const at::Tensor row_block_indices =
      std::get<row_block_indices_index>(serialized);
  const at::Tensor col_block_indices =
      std::get<col_block_indices_index>(serialized);
  // 解压非后端特定的非平铺 BCSR 格式，然后打包成 Fbgemm 平铺 BCSR 格式
  // 因为当前不存在未平铺的 Fbgemm BCSR 格式
  unpack_bcsr(
      reinterpret_cast<int8_t*>(weight_origin.data_ptr<c10::qint8>()),
      AT_DISPATCH_INTEGRAL_TYPES(
          row_block_indices.scalar_type(),
          "packed_linear_weight_fbgemm_setup_bcsr",
          [&] {
            return ao::sparse::BCSR(
                std::move(weight_values),  // 移动权重值向量
                unwrap_vector<scalar_t, int32_t>(
                    std::get<row_block_indices_index>(serialized)),  // 行块索引
                unwrap_vector<scalar_t, int32_t>(
                    std::get<col_block_indices_index>(serialized)));  // 列块索引
          }),
      output_channels,  // 输出通道数
      input_channels,  // 输入通道数
      out_features_block_size,  // 输出特征块大小
      in_features_block_size,  // 输入特征块大小
      weight_zero_points.data_ptr<int8_t>(),  // 权重零点的数据指针
      q_scheme == c10::kPerTensorAffine);  // 是否是按张量的仿射量化方案

  // 返回预打包的线性权重对象，其中包括权重原点、偏置项、输出特征块大小和输入特征块大小
  return PackedLinearWeight::prepack(
      weight_origin,
      std::get<bias_index>(serialized),  // 偏置项
      out_features_block_size,  // 输出特征块大小
      in_features_block_size);  // 输入特征块大小
// 如果定义了 USE_PYTORCH_QNNPACK 宏，则编译以下代码块

// 从 BCSRSerializationType 反序列化 PackedLinearWeightQnnp 对象，返回一个指向 LinearPackedParamsBase 的智能指针
c10::intrusive_ptr<LinearPackedParamsBase> PackedLinearWeightQnnp::deserialize(
    const BCSRSerializationType& serialized) {
  return c10::make_intrusive<PackedLinearWeightQnnp>(serialized);
}

// 模板：针对不同的 INDICES_DTYPE 类型，定义 UnsignedIndicesTypeTrait 结构体
template <typename INDICES_DTYPE>
struct UnsignedIndicesTypeTrait {
  // 静态断言：如果 INDICES_DTYPE 的大小为 0，则抛出错误信息
  static_assert(
      sizeof(INDICES_DTYPE) == 0,
      "Invalid dtype for UnsignedIndicesTypeTrait");
};

// 对 int32_t 类型特化：定义 UnsignedIndicesTypeTrait<int32_t>::t 为 uint32_t
template <>
struct UnsignedIndicesTypeTrait<int32_t> {
  using t = uint32_t;
};

// 对 int16_t 类型特化：定义 UnsignedIndicesTypeTrait<int16_t>::t 为 uint16_t
template <>
struct UnsignedIndicesTypeTrait<int16_t> {
  using t = uint16_t;
};

// 对 int8_t 类型特化：定义 UnsignedIndicesTypeTrait<int8_t>::t 为 uint8_t
template <>
struct UnsignedIndicesTypeTrait<int8_t> {
  using t = uint8_t;
};

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
// PackedLinearWeightQnnp 类的构造函数，根据反序列化的 BCSRSerializationType 对象初始化成员变量
PackedLinearWeightQnnp::PackedLinearWeightQnnp(
    const BCSRSerializationType& serialized)
    : LinearPackedParamsBase(
          std::get<out_features_block_size_index>(serialized),  // 初始化基类 LinearPackedParamsBase 的 out_features_block_size_
          std::get<in_features_block_size_index>(serialized)),  // 初始化基类 LinearPackedParamsBase 的 in_features_block_size_
      orig_bias_(std::get<bias_index>(serialized)),              // 初始化 orig_bias_，从反序列化对象获取偏置值
      q_scheme_(                                                   // 初始化量化方案 q_scheme_
          std::get<quantization_scheme_index>(serialized)           // 根据反序列化对象获取量化方案索引
              ? c10::kPerTensorAffine                              // 如果量化方案索引非零，使用 PerTensorAffine
              : c10::kPerChannelAffine),                            // 否则使用 PerChannelAffine
      output_channels_(std::get<num_output_channels_index>(serialized)),  // 初始化输出通道数 output_channels_
      input_channels_(std::get<num_input_channels_index>(serialized)) {   // 初始化输入通道数 input_channels_

  const int64_t serialization_version =                                 // 获取序列化版本号
      std::get<serialization_version_index>(serialized);
  // 检查序列化版本是否与当前支持的版本兼容
  TORCH_CHECK(
      serialization_version <= SPARSE_LINEAR_PACKED_PARAM_SERIALIZATION_VERSION,
      "Attempted to deserialize sparse qlinear packed params with an ",
      "incompatible serialization version (",
      serialization_version,
      " > ",
      SPARSE_LINEAR_PACKED_PARAM_SERIALIZATION_VERSION,
      ")");

  // 如果存在原始偏置值 orig_bias_
  if (orig_bias_.has_value()) {
    bias_ = orig_bias_.value();  // 将 orig_bias_ 的值赋给成员变量 bias_

    // 检查偏置值的维度和大小是否符合预期
    TORCH_CHECK(
        (bias_.ndimension() == 1 && bias_.size(0) == output_channels_),
        "ao::sparse::qlinear_deserialize (qnnpack): Given weight of size ",
        "{",
        output_channels_,
        ", ",
        input_channels_,
        "}",
        ", expected bias to be 1-dimensional with ",
        output_channels_,
        " elements",
        ", but got bias of size ",
        bias_.sizes(),
        " instead");
  } else {
  bias_ = at::zeros(output_channels_, at::device(at::kCPU).dtype(at::kFloat));
}

// Pad amount (8) comes from make_zero_points_and_scales_tensor
// https://github.com/pytorch/pytorch/blob/f8c1acea1e78573c04cd18893c4abff9eea64b03/aten/src/ATen/native/quantized/cpu/qnnpack_utils.h#L468
// 计算填充的输出通道数，增加了8个额外的元素
const int64_t output_channels_padded = output_channels_ + 8;

// 初始化权重缩放因子向量，大小为output_channels_padded
w_scales_ = at::empty(
    {output_channels_padded}, at::device(at::kCPU).dtype(at::kFloat));
float* w_scales_data_ptr = w_scales_.mutable_data_ptr<float>();

// 使用值1填充w_scales_中从output_channels_到output_channels_padded之间的元素
std::fill_n(
    w_scales_data_ptr + output_channels_,
    output_channels_padded - output_channels_,
    1); // Pad with 1

// 初始化权重零点向量，大小为output_channels_padded，所有元素用0填充
w_zero_points_ =
    std::vector<uint8_t>(output_channels_padded, 0); // Pad with 0;

// 获取序列化数据中的原始权重缩放因子和零点数据指针
const float* w_scales_orig_data_ptr =
    std::get<weight_scales_index>(serialized).data_ptr<float>();
const int8_t* w_zp_orig_data_ptr =
    std::get<weight_zero_point_index>(serialized).data_ptr<int8_t>();

// 定义一个函数对象add_128，将int8_t类型的值加上128转换为uint8_t类型
const std::function<uint8_t(int8_t)> add_128 = [](int8_t v) {
  return static_cast<uint8_t>(static_cast<int16_t>(v) + 128);
};

// 根据量化方案q_scheme_的不同，对权重缩放因子和零点进行初始化
if (q_scheme_ == at::kPerTensorAffine) {
  // 对于每个通道使用相同的缩放因子和零点
  std::fill_n(w_scales_data_ptr, output_channels_, w_scales_orig_data_ptr[0]);
  std::fill_n(
      w_zero_points_.begin(), output_channels_, w_zp_orig_data_ptr[0] + 128);
} else if (q_scheme_ == at::kPerChannelAffine) {
  // 每个通道使用不同的缩放因子和零点
  std::copy(
      w_scales_orig_data_ptr,
      w_scales_orig_data_ptr + output_channels_,
      w_scales_data_ptr);
  std::transform(
      w_zp_orig_data_ptr,
      w_zp_orig_data_ptr + output_channels_,
      w_zero_points_.begin(),
      add_128);
} else {
  // 不支持的量化方案
  TORCH_CHECK(false, "Unsupported quantization scheme.");
}

// 从序列化数据中获取块索引和权重值
deserialized_bcsr_row_block_indices_ =
    std::get<row_block_indices_index>(serialized);
deserialized_bcsr_col_block_indices_ =
    std::get<col_block_indices_index>(serialized);
deserialized_bcsr_weight_values_ = std::get<weight_values_index>(serialized);
#define AT_DISPATCH_CASE_BCSR_INDICES_TYPES(...)      \
  AT_DISPATCH_CASE(at::ScalarType::Char, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Int, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::Short, __VA_ARGS__)

定义宏 `AT_DISPATCH_CASE_BCSR_INDICES_TYPES`，用于根据不同的标量类型分发操作。


#define AT_DISPATCH_BCSR_INDICES_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                   \
      TYPE, NAME, AT_DISPATCH_CASE_BCSR_INDICES_TYPES(__VA_ARGS__))

定义宏 `AT_DISPATCH_BCSR_INDICES_TYPES`，用于根据给定的类型 `TYPE`，调用 `AT_DISPATCH_CASE_BCSR_INDICES_TYPES` 宏来执行分发操作。


  bcsr_matrix_ = AT_DISPATCH_BCSR_INDICES_TYPES(
      deserialized_bcsr_row_block_indices_.scalar_type(),
      "packed_linear_weight_qnnp_setup_bcsr",
      [&] {
        using unsigned_t = UnsignedIndicesTypeTrait<scalar_t>::t;
        return qnnpack::generateBlockCSRMatrix<unsigned_t>(
            reinterpret_cast<unsigned_t*>(
                deserialized_bcsr_col_block_indices_.data_ptr<scalar_t>()),
            reinterpret_cast<unsigned_t*>(
                deserialized_bcsr_row_block_indices_.data_ptr<scalar_t>()),
            deserialized_bcsr_weight_values_.data_ptr<uint8_t>(),
            deserialized_bcsr_col_block_indices_.numel(),
            deserialized_bcsr_row_block_indices_.numel(),
            deserialized_bcsr_weight_values_.numel(),
            out_features_block_size_,
            in_features_block_size_);
      });

使用 `AT_DISPATCH_BCSR_INDICES_TYPES` 宏，根据 `deserialized_bcsr_row_block_indices_` 的标量类型，调用 `qnnpack::generateBlockCSRMatrix` 函数生成一个块压缩稀疏行矩阵 (`bcsr_matrix_`)。


#undef AT_DISPATCH_CASE_BCSR_INDICES_TYPES
#undef AT_DISPATCH_BCSR_INDICES_TYPES

取消定义之前定义的宏 `AT_DISPATCH_CASE_BCSR_INDICES_TYPES` 和 `AT_DISPATCH_BCSR_INDICES_TYPES`。


#endif // USE_PYTORCH_QNNPACK

结束条件编译指令，检查是否使用了 PyTorch QNNPACK。


} // namespace sparse
} // namespace ao

关闭 `sparse` 和 `ao` 命名空间。
```