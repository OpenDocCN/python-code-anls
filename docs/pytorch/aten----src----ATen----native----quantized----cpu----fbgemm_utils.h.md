# `.\pytorch\aten\src\ATen\native\quantized\cpu\fbgemm_utils.h`

```py
#pragma once

#include <ATen/Tensor.h>
#include <ATen/native/quantized/PackedParams.h>
#include <ATen/native/quantized/cpu/EmbeddingPackedParams.h>
#include <c10/core/QScheme.h>
#include <c10/util/irange.h>

#ifdef USE_FBGEMM
#include <fbgemm/Fbgemm.h>
#include <fbgemm/FbgemmFP16.h>
#include <fbgemm/QuantUtils.h>

// 定义一个结构体 PackedLinearWeight，继承自 LinearPackedParamsBase
// 用于存储量化线性层的参数，包括预先打包的权重矩阵 w，可选的偏置 bias_，
// 列偏移 col_offsets，权重缩放因子 w_scale，权重零点 w_zp，量化方案 q_scheme
// 这些参数在预处理步骤中用于量化推断中的计算优化
struct TORCH_API PackedLinearWeight : public LinearPackedParamsBase {
  PackedLinearWeight(
      std::unique_ptr<fbgemm::PackBMatrix<int8_t>> w,
      std::optional<at::Tensor> bias,
      std::vector<int32_t> col_offsets,
      std::vector<float> w_scale,
      std::vector<int32_t> w_zp,
      c10::QScheme q_scheme)
      : w(std::move(w)),
        bias_(std::move(bias)),
        col_offsets(std::move(col_offsets)),
        w_scale(std::move(w_scale)),
        w_zp(std::move(w_zp)),
        q_scheme(std::move(q_scheme)) {}

  std::unique_ptr<fbgemm::PackBMatrix<int8_t>> w;  // 打包后的权重矩阵
  std::optional<at::Tensor> bias_;  // 可选的偏置项
  std::vector<int32_t> col_offsets;  // 列偏移量
  std::vector<float> w_scale;  // 权重缩放因子
  std::vector<int32_t> w_zp;  // 权重零点
  c10::QScheme q_scheme;  // 量化方案

  // 应用量化线性层到输入张量，返回输出张量
  at::Tensor apply(
      at::Tensor input,
      double output_scale,
      int64_t output_zero_point) override;

  // 应用带 ReLU 激活函数的量化线性层到输入张量，返回输出张量
  at::Tensor apply_relu(
      at::Tensor input,
      double output_scale,
      int64_t output_zero_point) override;

  // 在输出张量上应用量化线性层，结果保存在提供的输出张量中
  at::Tensor& apply_out(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point,
      at::Tensor& output) override;

  // 在输出张量上应用带 ReLU 激活函数的量化线性层，结果保存在提供的输出张量中
  at::Tensor& apply_relu_out(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point,
      at::Tensor& output) override;

  // 使用输入量化参数应用量化线性层到输入张量，返回输出张量（输出为 float32）
  at::Tensor apply_with_input_q_dq_qweight_dq_output_fp32(
      at::Tensor input,
      double input_scale,
      int64_t input_zero_point) override;

  // 使用输入量化参数应用带 ReLU 激活函数的量化线性层到输入张量，返回输出张量（输出为 float32）
  at::Tensor apply_with_input_q_dq_qweight_dq_relu_output_fp32(
      at::Tensor input,
      double input_scale,
      int64_t input_zero_point) override;

  // 动态量化模式下应用量化线性层到输入张量，返回输出张量
  at::Tensor apply_dynamic(at::Tensor input, bool reduce_range = false)
      override;

  // 动态量化模式下应用带 ReLU 激活函数的量化线性层到输入张量，返回输出张量
  at::Tensor apply_dynamic_relu(at::Tensor input, bool reduce_range = false)
      override;

  // 解包方法，返回权重张量和可选的偏置张量
  std::tuple<at::Tensor, std::optional<at::Tensor>> unpack() override;

  // 返回可选的偏置项
  std::optional<at::Tensor> bias() override {
    // 返回私有成员 bias_
    return bias_;
  }

  // 预打包线性层参数，返回线性层参数的指针
  static c10::intrusive_ptr<LinearPackedParamsBase> prepack(
      at::Tensor weight,
      std::optional<at::Tensor> bias);

 private:
  // 实现应用函数，应用于没有融合 ReLU 的情况
  template <bool ReluFused>
  at::Tensor& apply_impl(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point,
      at::Tensor& output);

  // 应用函数，使用输入量化、权重量化和输出量化
  template <bool ReluFused>
  at::Tensor apply_with_input_q_dq_qweight_dq_output_fp32_impl(
      const at::Tensor& input,
      double input_scale,
      int64_t input_zero_point);

  // 动态应用函数实现，用于处理动态输入的情况
  template <bool ReluFused>
  at::Tensor apply_dynamic_impl(at::Tensor input, bool reduce_range = false);


这段代码是一个 C++ 类的声明部分，包含了几个成员函数和模板函数的声明。注释详细描述了每个函数的作用和用途。
};

// PackedLinearWeightFp16 结构体，继承自 LinearPackedParamsBase
struct TORCH_API PackedLinearWeightFp16 : public LinearPackedParamsBase {
  // 构造函数，接受一个包含权重矩阵的 unique_ptr 和一个可选的偏置 Tensor
  PackedLinearWeightFp16(
      std::unique_ptr<fbgemm::PackedGemmMatrixFP16> w,
      std::optional<at::Tensor> bias)
      : w(std::move(w)), bias_(std::move(bias)) {}

  // 权重矩阵的 unique_ptr
  std::unique_ptr<fbgemm::PackedGemmMatrixFP16> w;
  // 可选的偏置 Tensor
  std::optional<at::Tensor> bias_;

  // 应用函数，未实现，用于线性操作
  at::Tensor apply(
      at::Tensor /*input*/,
      double /*output_scale*/,
      int64_t /*output_zero_point*/) override {
    TORCH_INTERNAL_ASSERT(false);
  }
  
  // 应用函数，未实现，用于带ReLU的线性操作
  at::Tensor apply_relu(
      at::Tensor /*input*/,
      double /*output_scale*/,
      int64_t /*output_zero_point*/) override {
    TORCH_INTERNAL_ASSERT(false);
  }

  // 动态应用函数声明，用于动态量化线性操作
  at::Tensor apply_dynamic(at::Tensor input, bool reduce_range = false)
      override;
  // 动态应用函数声明，用于带ReLU的动态量化线性操作
  at::Tensor apply_dynamic_relu(at::Tensor input, bool reduce_range = false)
      override;

  // 动态应用函数声明，用于在现有输出上进行动态量化线性操作
  at::Tensor& apply_dynamic_out(
      const at::Tensor& input,
      at::Tensor& output,
      bool reduce_range = false) override;
  // 动态应用函数声明，用于在现有输出上进行带ReLU的动态量化线性操作
  at::Tensor& apply_dynamic_relu_out(
      const at::Tensor& input,
      at::Tensor& output,
      bool reduce_range = false) override;

  // 解包函数声明，用于返回权重和偏置
  std::tuple<at::Tensor, std::optional<at::Tensor>> unpack() override;

  // 获取偏置的函数，返回可选的偏置 Tensor
  std::optional<at::Tensor> bias() override {
    return bias_;
  }

  // 预打包静态方法，用于将权重和偏置打包成 LinearPackedParamsBase 对象
  static c10::intrusive_ptr<LinearPackedParamsBase> prepack(
      at::Tensor weight,
      std::optional<at::Tensor> bias);

  // 设置偏置的方法，更新偏置成员变量
  void set_bias(std::optional<at::Tensor> bias) override;

 private:
  // 模板函数声明，用于实现动态量化线性操作
  template <bool ReluFused>
  at::Tensor& apply_dynamic_impl(const at::Tensor& input, at::Tensor& output);
};

// 模板类 PackedLinearWeightFp16 的默认参数为 kSpatialDim = 2
template <int kSpatialDim = 2>
// 定义一个结构 PackedConvWeight，继承自 ConvPackedParamsBase，其中 kSpatialDim 是维度参数
struct TORCH_API PackedConvWeight : public ConvPackedParamsBase<kSpatialDim> {
  // 构造函数，初始化 PackedConvWeight 对象
  PackedConvWeight(
      // 权重的打包器对象，用于加速卷积操作
      std::unique_ptr<fbgemm::PackWeightsForConv<kSpatialDim>> w,
      // 可选的偏置张量
      std::optional<at::Tensor> bias,
      // 卷积步长列表
      torch::List<int64_t> stride,
      // 卷积填充列表
      torch::List<int64_t> padding,
      // 输出填充列表（对于转置卷积）
      torch::List<int64_t> output_padding,
      // 卷积扩展列表
      torch::List<int64_t> dilation,
      // 卷积组数
      int64_t groups,
      // 是否转置卷积
      uint8_t transpose,
      // 用于矩阵计算的列偏移量
      std::vector<int32_t> col_offsets,
      // 卷积核的形状
      std::vector<int64_t> kernel,
      // 权重的缩放因子
      std::vector<float> w_scale,
      // 权重的零点
      std::vector<int32_t> w_zp,
      // 量化方案
      c10::QScheme q_scheme)
      // 使用成员初始化列表初始化成员变量
      : w(std::move(w)),
        bias(std::move(bias)),
        stride_(std::move(stride)),
        padding_(std::move(padding)),
        output_padding_(std::move(output_padding)),
        dilation_(std::move(dilation)),
        groups_(groups),
        transpose_(transpose),
        col_offsets(std::move(col_offsets)),
        kernel(std::move(kernel)),
        w_scale(std::move(w_scale)),
        w_zp(std::move(w_zp)),
        q_scheme(q_scheme) {}

  // 带有 kSpatialDim 维度的权重打包器对象
  std::unique_ptr<fbgemm::PackWeightsForConv<kSpatialDim>> w;
  // 可选的偏置张量
  std::optional<at::Tensor> bias;
  // 卷积步长列表
  torch::List<int64_t> stride_;
  // 卷积填充列表
  torch::List<int64_t> padding_;
  // 输出填充列表（对于转置卷积）
  torch::List<int64_t> output_padding_;
  // 卷积扩展列表
  torch::List<int64_t> dilation_;
  // 卷积组数
  int64_t groups_;
  // 是否转置卷积
  uint8_t transpose_;
  // 用于矩阵计算的列偏移量
  std::vector<int32_t> col_offsets;
  // 卷积核的形状
  std::vector<int64_t> kernel;
  // 权重的缩放因子
  std::vector<float> w_scale;
  // 权重的零点
  std::vector<int32_t> w_zp;
  // 量化方案
  c10::QScheme q_scheme;

  // 应用带有输出缩放和零点的卷积操作
  at::Tensor apply(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point) override;

  // 应用带有ReLU激活函数的卷积操作
  at::Tensor apply_relu(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point) override;

  // 动态应用卷积操作，可以减少范围
  at::Tensor apply_dynamic(
    const at::Tensor& input,
    bool reduce_range) override;

  // 解压缩打包的参数
  std::tuple<at::Tensor, std::optional<at::Tensor>> unpack() override;

  // 静态方法：预打包卷积参数
  static c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>> prepack(
      at::Tensor weight,
      std::optional<at::Tensor> bias,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> output_padding,
      torch::List<int64_t> dilation,
      int64_t groups,
      bool transpose);

  // 获取偏置数据的指针
  const float* GetBiasData(at::Tensor* bias);

  // 获取量化参数
  void GetQuantizationParams(
      float act_scale,
      float out_scale,
      std::vector<float>* output_multiplier_float,
      std::vector<float>* act_times_w_scale);

  // 返回卷积步长列表
  torch::List<int64_t> stride() const override {
    return stride_;
  }

  // 返回卷积填充列表
  torch::List<int64_t> padding() const override {
    return padding_;
  }

  // 返回输出填充列表（对于转置卷积）
  torch::List<int64_t> output_padding() const override {
    return output_padding_;
  }

  // 返回卷积扩展列表
  torch::List<int64_t> dilation() const override {
    return dilation_;
  }

  // 返回卷积组数
  int64_t groups() const override {
    return groups_;
  }

  // 返回是否转置卷积
  bool transpose() const override {
    // 返回 transpose_ 的布尔值转换结果
    return (bool)transpose_;
  }

 private:
  // 模板函数，根据模板参数 ReluFused 的值选择是否使用 ReLU 激活函数
  template <bool ReluFused>
  // 对输入张量应用量化操作的实现
  at::Tensor apply_impl(
      const at::Tensor& input,         // 输入张量
      double output_scale,             // 输出的量化比例因子
      int64_t output_zero_point);      // 输出的零点偏移值
};

// PackWeight: Convert the weight from uint8 to int8.
inline void convert_uint8_int8(
    int len,
    const uint8_t* src_uint8,
    int8_t* dst_int8) {
  // Iterate over the elements and convert uint8 to int8
  for (const auto i : c10::irange(len)) {
    dst_int8[i] = static_cast<int8_t>(static_cast<int32_t>(src_uint8[i]) - 128);
  }
}

// UnpackWeight: Convert the weight from int8 to uint8.
inline void convert_int8_uint8(
    int len,
    const int8_t* src_int8,
    uint8_t* dst_uint8) {
  // Iterate over the elements and convert int8 to uint8
  for (const auto i : c10::irange(len)) {
    dst_uint8[i] = static_cast<uint8_t>(static_cast<int32_t>(src_int8[i]) + 128);
  }
}

namespace at {
namespace native {
namespace fbgemm_utils {

template <int kSpatialDim = 2>
// Construct and return fbgemm::conv_param_t structure for convolution parameters
fbgemm::conv_param_t<kSpatialDim> MakeFbgemmConvParam(
    int N,
    int C,
    int M,
    const std::vector<int>& image_shape,
    int groups,
    const std::vector<int>& kernels,
    const std::vector<int>& strides,
    const std::vector<int>& pads,
    const std::vector<int>& dilations,
    const std::vector<int>& output_padding = std::vector<int>(kSpatialDim, 0),
    bool transposed = false);

// TODO: Remove functions below when ChannelsLast3d is ready.

// MakeStridedQTensorCPU: Create a strided quantized tensor on CPU
Tensor MakeStridedQTensorCPU(
    const IntArrayRef& sizes,
    const IntArrayRef& strides,
    const TensorOptions& options,
    QuantizerPtr quantizer);

// MakeEmptyAffineQuantizedChannelsLast3dTensor: Create an empty affine quantized tensor with ChannelsLast3d format
Tensor MakeEmptyAffineQuantizedChannelsLast3dTensor(
    int64_t N,
    int64_t C,
    int64_t D,
    int64_t H,
    int64_t W,
    const TensorOptions& options,
    double scale,
    int64_t zero_point);

// MakeEmptyPerChannelAffineQuantizedChannelsLast3dTensor: Create an empty per-channel affine quantized tensor with ChannelsLast3d format
Tensor MakeEmptyPerChannelAffineQuantizedChannelsLast3dTensor(
    int64_t N,
    int64_t C,
    int64_t D,
    int64_t H,
    int64_t W,
    const TensorOptions& options,
    const Tensor& scales,
    const Tensor& zero_points);

// ConvertToChannelsLast3dTensor: Convert tensor to ChannelsLast3d format
Tensor ConvertToChannelsLast3dTensor(const Tensor& src);

template <int kSpatialDim = 2>
// TransposeConvTensorUnpackConversion: Perform unpacking conversion for transposed convolution tensor
Tensor TransposeConvTensorUnpackConversion(const Tensor& src, int groups);

template <int kSpatialDim>
// ConvertConvWeightsToChannelLastTensor: Convert convolution weights to ChannelLast format tensor
Tensor ConvertConvWeightsToChannelLastTensor(
    const at::Tensor& src,
    int groups,
    bool transpose);
} // namespace fbgemm_utils
} // namespace native
} // namespace at

#endif // USE_FBGEMM

struct TORCH_API PackedEmbeddingBagWeight : public EmbeddingPackedParamsBase {
  // PackedEmbeddingBagWeight: Structure to hold packed embedding bag weights
  PackedEmbeddingBagWeight(
      at::Tensor packed_w,
      std::vector<float> w_scale,
      std::vector<float> w_zp,
      int64_t bit_rate,
      c10::QScheme q_scheme,
      int64_t version)
      : packed_w(std::move(packed_w)),
        w_scale(std::move(w_scale)),
        w_zp(std::move(w_zp)),
        bit_rate_(bit_rate),
        q_scheme(q_scheme),
        version_(version) {
    // NOLINTNEXTLINE(clang-analyzer-cplusplus.Move)
    // Ensure packed_w is contiguous
    if (!packed_w.is_contiguous()) {
      packed_w = packed_w.contiguous();
  }
  // 结束匿名命名空间

  at::Tensor packed_w;
  // 定义一个 ATen 张量 packed_w，用于存储压缩后的权重

  std::vector<float> w_scale;
  // 定义一个 float 类型的向量 w_scale，用于存储权重的缩放因子

  std::vector<float> w_zp;
  // 定义一个 float 类型的向量 w_zp，用于存储权重的零点值

  int64_t bit_rate_;
  // 定义一个 int64_t 类型的变量 bit_rate_，用于存储比特率信息

  c10::QScheme q_scheme;
  // 定义一个 c10::QScheme 类型的变量 q_scheme，表示量化方案

  int64_t version_;
  // 定义一个 int64_t 类型的变量 version_，用于存储版本号信息

  // 覆盖基类的方法，返回比特率
  int64_t bit_rate() const override {
    return bit_rate_;
  }

  // 覆盖基类的方法，返回版本号
  int64_t version() const override {
    return version_;
  }

  // 覆盖基类的方法，实现字节量化的嵌入操作
  at::Tensor embeddingbag_byte(
      const at::Tensor& indices,
      const std::optional<at::Tensor>& offsets,
      bool pruned_weights,
      const std::optional<at::Tensor>& per_sample_weights_,
      const std::optional<at::Tensor>& compressed_indices_mapping,
      bool include_last_offset,
      bool is_embedding_op) override;
      // 返回一个 ATen 张量，表示字节量化的嵌入袋操作

  // 覆盖基类的方法，实现4比特量化的嵌入操作
  at::Tensor embeddingbag_4bit(
      const at::Tensor& indices,
      const std::optional<at::Tensor>& offsets,
      bool pruned_weights,
      const std::optional<at::Tensor>& per_sample_weights_,
      const std::optional<at::Tensor>& compressed_indices_mapping,
      bool include_last_offset,
      bool is_embedding_op) override;
      // 返回一个 ATen 张量，表示4比特量化的嵌入袋操作
};



# 这行代码是一个独立的分号，通常用于结束语句或分隔代码块
# 在这段代码中，它单独出现，没有前面的代码，也没有后续的代码，可能是代码编辑或复制粘贴过程中的残留物
# 在实际程序中，这种情况可能需要进一步确认或清理，以确保代码结构的完整性和正确性
```