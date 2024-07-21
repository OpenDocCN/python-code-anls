# `.\pytorch\aten\src\ATen\native\quantized\cpu\qconv_prepack.cpp`

```py
// 定义宏，用于仅使用方法操作符的 Torch 断言
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 包含标准库头文件
#include <utility>
#include <vector>

// 包含 ATen 库相关头文件
#include <ATen/core/Tensor.h>
#include <ATen/core/List.h>
#include <ATen/Context.h>
#include <ATen/native/quantized/PackedParams.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/cpu/init_qnnpack.h>
#include <ATen/native/quantized/cpu/QnnpackUtils.h>
#include <ATen/native/quantized/cpu/OnednnUtils.h>
#include <ATen/native/quantized/cpu/QuantUtils.h>
#include <torch/library.h>
#include <ATen/native/mkldnn/MKLDNNCommon.h>

// 根据条件包含 ATen 库函数头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/zeros.h>
#endif

// 包含 C10 实用工具的头文件
#include <c10/util/irange.h>

// 根据条件定义使用 FBGEMM
#ifdef USE_FBGEMM
// 声明模板类 PackedConvWeight 的成员函数
template <int kSpatialDim>
c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>> PackedConvWeight<kSpatialDim>::
    prepack(
        at::Tensor weight,                                     // 输入参数：权重张量
        std::optional<at::Tensor> bias,                        // 输入参数：可选的偏置张量
        torch::List<int64_t> stride,                           // 输入参数：步长列表
        torch::List<int64_t> padding,                          // 输入参数：填充列表
        torch::List<int64_t> output_padding,                   // 输入参数：输出填充列表
        torch::List<int64_t> dilation,                         // 输入参数：膨胀列表
        int64_t groups,                                        // 输入参数：分组数
        bool transpose) {                                      // 输入参数：是否转置卷积标志

  TORCH_CHECK(
      weight.ndimension() == kSpatialDim + 2,                 // 检查权重张量维度是否正确
      "Weights are expected to have ",
      kSpatialDim + 2,
      " dimensions");

  TORCH_CHECK(
      stride.size() == kSpatialDim,                           // 检查步长列表长度是否正确
      "stride should contain ",
      kSpatialDim,
      " elements for ",
      kSpatialDim,
      "D convolution.");

  TORCH_CHECK(
      padding.size() == kSpatialDim,                          // 检查填充列表长度是否正确
      "Specify front/top/left padding only. "
      "end/bottom/right padding assumed to be equal to front/top/left");

  TORCH_CHECK(
      !transpose || output_padding.size() == kSpatialDim,     // 检查输出填充列表长度是否正确（仅在转置卷积时）
      "quantized::conv_prepack: Specify top/left output padding "
      "only. bottom/right padding assumed to be equal to top/left");

  TORCH_CHECK(
      dilation.size() == kSpatialDim,                         // 检查膨胀列表长度是否正确
      "dilation should contain ",
      kSpatialDim,
      " elements for ",
      kSpatialDim,
      "D convolution.");

  const int input_channels = transpose ? weight.size(0)       // 计算输入通道数，根据是否转置选择不同的维度
                                       : weight.size(1) * groups;

  // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
  const int output_channels = transpose ? weight.size(1) * groups
                                        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
                                        : weight.size(0);

  const int kernel_d = kSpatialDim == 2 ? 1 : weight.size(2);  // 计算核的深度维度大小
  const int kernel_h = weight.size(kSpatialDim);               // 计算核的高度维度大小
  const int kernel_w = weight.size(kSpatialDim + 1);           // 计算核的宽度维度大小

  // mini-batch doesn't have any impact on how we pack weights
  // so we pass it as 1
  // Input image height/width also don't have any impact on how we pack
  // weights so we can pass any values
  const fbgemm::conv_param_t<kSpatialDim> conv_p =             // 创建用于FBGEMM卷积参数的结构体
      at::native::fbgemm_utils::MakeFbgemmConvParam<kSpatialDim>(
          1,                                                    // 虚拟的批处理大小
          input_channels,                                       // 输入通道数
          output_channels,                                      // 输出通道数
          kSpatialDim == 2 ? std::vector<int>{28, 28}           // 图像尺寸（在二维情况下虚拟）
                           : std::vector<int>{28, 28, 28},
          groups,                                               // 分组数
          kSpatialDim == 2 ? std::vector<int>{kernel_h, kernel_w}  // 卷积核大小（根据维度数量选择不同的尺寸）
                           : std::vector<int>{kernel_d, kernel_h, kernel_w},
          std::vector<int>(stride.begin(), stride.end()),       // 步长转换为标准向量
          std::vector<int>(padding.begin(), padding.end()),     // 填充转换为标准向量
          std::vector<int>(dilation.begin(), dilation.end()),   // 膨胀转换为标准向量
          std::vector<int>(output_padding.begin(), output_padding.end()),  // 输出填充转换为标准向量
          transpose);                                           // 是否转置卷积标志

  const auto qtype = weight.qscheme();                         // 获取权重的量化类型
  std::vector<int32_t> zero_points;                            // 初始化零点向量
  if (qtype == c10::kPerTensorAffine) {                        // 根据量化类型设置零点
    zero_points = {static_cast<int32_t>(weight.q_zero_point())};
  } else if (qtype == c10::kPerChannelAffine) {
    // 针对每个通道量化的情况，暂时省略部分代码
  // 检查是否需要转置，如果是则抛出错误，暂时不支持逆卷积的逐通道量化
  TORCH_CHECK(
      !transpose,
      "Per Channel Quantization is currently disabled for transposed conv");

  // 初始化 zero_points 数组，大小为输出通道数
  zero_points.resize(output_channels);

  // 遍历每个输出通道，将权重的逐通道零点值存入 zero_points 数组
  for (const auto i : c10::irange(output_channels)) {
    zero_points[i] = weight.q_per_channel_zero_points()[i].item<int32_t>();
  }
} else {
  // 如果量化方案不支持，则抛出错误
  TORCH_CHECK(false, "Unsupported qscheme: ", toString(qtype));
}

// FBGEMM 要求权重以通道为最后一个维度
// 当 ChannelsLast3d 准备好时，应该修改此处
// 对于卷积和逆卷积，FBGEMM 需要的权重排列是 G OC/G kDim0 ... kDimN IC/G
// 而 PyTorch 中的排列是 {out_c, in_c/groups, kH, kW}
// (逆卷积为 {in_c, out_c/groups, kH, kW})
const at::Tensor weight_nhwc =
    at::native::fbgemm_utils::ConvertConvWeightsToChannelLastTensor<kSpatialDim>(weight, groups, transpose);
const int8_t* weight_data_int8 =
        reinterpret_cast<int8_t*>(weight_nhwc.data_ptr<c10::qint8>());

// 计算列偏移量，类似于 fbgemm::col_offsets_with_zero_pt_s8acc32_ref
// 注意偏移量包括列的和以及标量项 weight_zero_point * KDim
const int input_channels_per_group = input_channels / groups;
const int output_channels_per_group = output_channels / groups;
const int inner_size =
    kernel_d * kernel_h * kernel_w * input_channels_per_group;

// 遍历每个组和每个输出通道组，计算列偏移量
for (const auto g : c10::irange(groups)) {
  for (const auto i : c10::irange(output_channels_per_group)) {
    const int c = g * output_channels_per_group + i;
    int32_t sum = 0;

    // 计算权重数据的列和
    for (const auto j : c10::irange(inner_size)) {
      sum += static_cast<int32_t>(weight_data_int8[c * inner_size + j]);
    }

    // 根据量化方案设置列偏移量
    if (qtype == c10::kPerTensorAffine) {
      col_offsets[c] = sum - zero_points[0] * inner_size;
    } else {
      col_offsets[c] = sum - zero_points[c] * inner_size;
    }
  }
}

// 初始化 scales 数组，根据量化方案不同进行设置
std::vector<float> scales;
if (qtype == c10::kPerTensorAffine) {
  scales = {static_cast<float>(weight.q_scale())};
} else if (qtype == c10::kPerChannelAffine) {
  // 遍历每个输出通道，将权重的逐通道缩放因子存入 scales 数组
  scales.resize(output_channels);
  for (const auto i : c10::irange(output_channels)) {
    scales[i] = weight.q_per_channel_scales()[i].item<float>();
  }
}

// 如果存在偏置，则检查其维度和大小是否正确
std::optional<at::Tensor> bias_contig;
if (bias.has_value()) {
  at::Tensor bias_vec = bias.value();
  TORCH_CHECK(bias_vec.dim() == 1, "bias should be a vector (1D Tensor)");
  TORCH_CHECK(
      bias_vec.size(0) == output_channels,
      "bias should have K elements: " + std::to_string(output_channels));
    bias_contig = bias->contiguous();

# 将bias指针所指向的Tensor进行内存连续化处理，返回连续化后的Tensor

  }

  auto ret_ptr = c10::make_intrusive<PackedConvWeight<kSpatialDim>>(
      // 使用c10库创建一个包含卷积权重信息的PackedConvWeight对象，其中：
      PackedConvWeight<kSpatialDim>{
          // 使用fbgemm库的函数PackWeightsForConv对卷积权重进行打包处理，生成唯一指针
          std::make_unique<fbgemm::PackWeightsForConv<kSpatialDim>>(
              conv_p, weight_data_int8),
          // 传入上述处理的bias的连续化Tensor
          bias_contig,
          // 卷积的步长
          stride,
          // 卷积的填充
          padding,
          // 输出的填充
          output_padding,
          // 卷积的膨胀
          dilation,
          // 分组卷积的组数
          groups,
          // 是否为转置卷积
          transpose,
          // 矩阵偏移量
          col_offsets,
          // 卷积的维度数，若为二维，定义为向量内的整数{kernel_h, kernel_w}，否则定义为向量内的整数{kernel_d, kernel_h, kernel_w}
          kSpatialDim == 2 ? std::vector<int64_t>{kernel_h, kernel_w}
                           : std::vector<int64_t>{kernel_d, kernel_h, kernel_w},
          // 规模
          scales,
          // 零点
          zero_points,
          // 数据类型
          qtype});

  // 返回创建的PackedConvWeight对象的智能指针
  return ret_ptr;
#ifdef USE_PYTORCH_QNNPACK
// 根据空间维度kSpatialDim实例化PackedConvWeightsQnnp类模板
template <int kSpatialDim>
// 实现PackedConvWeightsQnnp类的prepack方法
c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>> PackedConvWeightsQnnp<
    kSpatialDim>::
    prepack(
        at::Tensor weight,
        std::optional<at::Tensor> bias_in,
        torch::List<int64_t> stride,
        torch::List<int64_t> padding,
        torch::List<int64_t> output_padding,
        torch::List<int64_t> dilation,
        int64_t groups,
        bool transpose) {
  // 检查空间维度是否为2或3，QNNPACK仅支持2D和3D卷积
  TORCH_CHECK(
      kSpatialDim == 2 || kSpatialDim == 3,  // 1D is packed as 2d, hence we don't need other checks
      "QNNPACK packing only supports 2D / 3D convolution.");
  // 检查权重张量维度是否符合预期的空间维度+2
  TORCH_CHECK(
      weight.ndimension() == kSpatialDim + 2,
      "quantized::conv_prepack (qnnpack): Weights are expected to have ",
      kSpatialDim + 2, " dimensions, found shape ", weight.sizes());
  // 检查步幅列表大小是否等于空间维度
  TORCH_CHECK(
      stride.size() == kSpatialDim,
      "quantized::conv_prepack (qnnpack): ",
      kSpatialDim, "D convolution expects stride to have ",
      kSpatialDim, " elements.");
  // 检查填充列表大小是否等于空间维度
  TORCH_CHECK(
      padding.size() == kSpatialDim,
      "quantized::conv_prepack (qnnpack): Specify top/left input padding "
      "only. bottom/right padding assumed to be equal to top/left");
  // 如果是转置卷积，检查输出填充列表大小是否等于空间维度
  TORCH_CHECK(
      !transpose || output_padding.size() == kSpatialDim,
      "quantized::conv_prepack (qnnpack): Specify top/left output padding "
      "only. bottom/right padding assumed to be equal to top/left");
  // 检查扩展列表大小是否等于空间维度
  TORCH_CHECK(
      dilation.size() == kSpatialDim,
      "quantized::conv_prepack (qnnpack): ",
      kSpatialDim, "D convolution expects dilation to have ",
      kSpatialDim, " elements.");

  // 初始化QNNPACK库
  at::native::initQNNPACK();

  // QNNPACK期望的权重张量格式为{out_c, kH, kW, in_c/groups}，
  // 而PyTorch的排列为{out_c, in_c/groups, kH, kW}
  // （对于转置卷积，排列为{in_c, out_c/groups, kH, kW}）
  const auto out_ch = transpose ? weight.size(1) * groups : weight.size(0);
  const uint32_t kernel_d = kSpatialDim == 3 ? weight.size(2) : 1;
  const uint32_t kernel_h = weight.size(kSpatialDim);
  const uint32_t kernel_w = weight.size(kSpatialDim + 1);

  at::Tensor bias_fp32;
  // 如果提供了偏置值，则使用该值
  if (bias_in.has_value()) {
    bias_fp32 = bias_in.value();
  } else {
    // 初始化一个与权重数据类型相同的全零张量作为偏置项，使用 at::kFloat 类型
    bias_fp32 = at::zeros(out_ch, weight.options().dtype(at::kFloat));
  }

  // 检查偏置项是否未定义或者定义为一维张量且大小为 out_ch
  TORCH_CHECK(
      !bias_fp32.defined() ||
          (bias_fp32.ndimension() == 1 && bias_fp32.size(0) == out_ch),
      "quantized::conv2d_prepack (qnnpack): expected bias to be 1-dimensional "
      "with ",
      out_ch,
      " elements",
      ", but got bias of size ",
      bias_fp32.sizes(),
      " instead. "
      "(weight dimensions: ",
      weight.sizes(), " , transpose: ",
      (transpose ? "True)." : "False).")
  );

  // 再次检查偏置项是否未定义或者定义为一维张量且大小为 out_ch
  TORCH_CHECK(
      !bias_fp32.defined() ||
          (bias_fp32.ndimension() == 1 && bias_fp32.size(0) == out_ch),
      "quantized::conv3d_prepack (qnnpack): expected bias to be 1-dimensional "
      "with ",
      out_ch,
      " elements",
      ", but got bias of size ",
      bias_fp32.sizes(),
      " instead. "
      "(weight dimensions: ",
      weight.sizes(), " , transpose: ",
      (transpose ? "True)." : "False).")
  );

  // 将权重张量转换为连续内存格式，根据二维或三维空间维度选择内存布局格式
  auto weight_contig = weight.contiguous(
      kSpatialDim == 2 ? c10::MemoryFormat::ChannelsLast
                       : c10::MemoryFormat::ChannelsLast3d);
  // 检查权重是否采用分通道量化方案
  const bool is_per_channel = weight_contig.qscheme() == at::kPerChannelAffine;
  // 根据空间维度选择相应的卷积核维度数组
  auto kernel_dim = kSpatialDim == 2
      ? std::vector<int64_t>{kernel_h, kernel_w}
      : std::vector<int64_t>{kernel_d, kernel_h, kernel_w};
  // 调用函数生成权重的零点和缩放因子张量
  auto [w_zero_points, w_scales] =
      make_zero_points_and_scales_tensor(weight_contig, transpose, groups);
  // 创建一个包含预打包卷积权重的智能指针，初始化预打包权重为 nullptr，
  // 具体的预打包操作将在运算符首次运行时执行，详见 qconv.cpp
  auto ret_ptr = c10::intrusive_ptr<PackedConvWeightsQnnp<kSpatialDim>>::make(
      nullptr, /* PrePackConvWeights */
      weight_contig, /* int8_t weight */
      bias_fp32.contiguous(), /* fp32 bias */
      stride,
      padding,
      output_padding,
      dilation,
      groups,
      transpose,
      c10::nullopt, /* input_scale */
      kernel_dim,
      w_scales,
      std::move(w_zero_points),
      is_per_channel);

  // 返回预打包的卷积权重智能指针
  return ret_ptr;
#if AT_MKLDNN_ENABLED()
// 定义 prepack 函数模板，用于 ONEDNN 加速的卷积权重打包
template <int kSpatialDim>
c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>> PackedConvWeightsOnednn<
    kSpatialDim>::
    prepack(
        at::Tensor weight,  // 权重张量
        std::optional<at::Tensor> bias,  // 可选的偏置张量
        torch::List<int64_t> stride,  // 步长列表
        torch::List<int64_t> padding,  // 填充列表
        torch::List<int64_t> output_padding,  // 输出填充列表
        torch::List<int64_t> dilation,  // 膨胀列表
        int64_t groups,  // 组数
        bool transpose) {  // 是否转置

  // 检查权重张量的维度是否符合预期
  TORCH_CHECK(
      weight.ndimension() == kSpatialDim + 2,
      "Weights are expected to have ", kSpatialDim + 2, " dimensions");

  // 检查步长列表的长度是否正确
  TORCH_CHECK(
      stride.size() == kSpatialDim,
      "stride should contain ", kSpatialDim, " elements for ",
      kSpatialDim, "D convolution.");

  // 检查填充列表的长度是否正确
  TORCH_CHECK(
      padding.size() == kSpatialDim,
      "Specify front/top/left padding only. "
      "end/bottom/right padding assumed to be equal to front/top/left");

  // 检查是否转置的情况下输出填充列表的长度是否正确
  TORCH_CHECK(
      !transpose || output_padding.size() == kSpatialDim,
      "quantized::conv_prepack: Specify top/left output padding "
      "only. bottom/right padding assumed to be equal to top/left");

  // 检查膨胀列表的长度是否正确
  TORCH_CHECK(
      dilation.size() == kSpatialDim,
      "dilation should contain ", kSpatialDim, " elements for ",
      kSpatialDim, "D convolution.");

  // 如果是转置操作，检查所有输出填充值是否为零
  TORCH_CHECK(
      !transpose || std::all_of(output_padding.begin(), output_padding.end(), [](int i) { return i==0; }),
      "quantized::conv_prepack: ONEDNN only supports zero output_padding.");

  // Weight
  // 权重张量的格式：对于卷积 [OC IC//group KH KW]；对于反卷积 [IC OC//group KH KW]
  auto dims = weight.sizes().vec();  // 获取权重张量的维度
  auto strides = stride.vec();  // 获取步长列表
  auto padding_l = padding.vec();  // 获取填充列表
  auto padding_r = padding.vec();  // 复制填充列表
  auto dilates = dilation.vec();  // 获取膨胀列表
  auto op_attr = ideep::attr_t();  // 定义操作属性
  std::vector<int32_t> wgt_zero_points;  // 权重零点
  ideep::scale_t wgt_scales;  // 权重缩放因子
  const int output_channels = transpose ? weight.size(1) * groups  // 计算输出通道数
                                        : weight.size(0);
  const auto qtype = weight.qscheme();  // 获取量化类型
  if (qtype == c10::kPerTensorAffine) {
    // 对于每个张量的仿射量化，检查零点是否为零
    TORCH_CHECK(
        weight.q_zero_point()==0,
        "quantized::qconv_prepack: ONEDNN only supports symmetric quantization of weight,"
        " whose zero point must be 0.");
    wgt_zero_points = std::vector<int32_t>(1, weight.q_zero_point());  // 设置权重零点
#if IDEEP_PREREQ(3, 1, 0, 1)
    wgt_scales = ideep::scale_t(1, weight.q_scale());  // 设置权重缩放因子
#elif IDEEP_PREREQ(3, 1, 0, 0)
    wgt_scales = ideep::scale_t(1, 1.0/weight.q_scale()); // ONEDNN 和 PyTorch 的缩放因子是倒数关系
#else
    TORCH_CHECK(false, "Unexpected IDeep version to do qconv weight prepack.");
#endif
  }
#endif // AT_MKLDNN_ENABLED()
#endif
  } else if (qtype == c10::kPerChannelAffine) {
    // 检查是否禁用了转置卷积的分通道量化
    TORCH_CHECK(
        !transpose,
        "Per Channel Quantization is currently disabled for transposed conv");
    // 初始化权重的零点和量化因子的容器，大小为输出通道数
    wgt_zero_points.resize(output_channels);
    wgt_scales.resize(output_channels);
    // 遍历每个输出通道
    for (int i = 0; i < output_channels; ++i) {
      // 获取每个通道的零点值
      wgt_zero_points[i] = weight.q_per_channel_zero_points()[i].item<int32_t>();
      // 检查权重的零点是否为0，因为ONEDNN仅支持权重的对称量化，零点必须为0
      TORCH_CHECK(
          wgt_zero_points[i]==0,
          "quantized::qconv_prepack: ONEDNN only supports symmetric quantization of weight,"
          " whose zero point must be 0.");
#if IDEEP_PREREQ(3, 1, 0, 1)
      // 获取每个通道的量化因子
      wgt_scales[i] = weight.q_per_channel_scales()[i].item<float>();
#elif IDEEP_PREREQ(3, 1, 0, 0)
      // 获取每个通道的量化因子的倒数，因为ONEDNN和PyTorch的量化因子是倒数关系
      wgt_scales[i] = 1.0f / weight.q_per_channel_scales()[i].item<float>(); // Scales of ONEDNN and PyTorch are reciprocal
#else
      // 如果版本不匹配，报错
      TORCH_CHECK(false, "Unexpected IDeep version to do qconv weight prepack.");
#endif
    }
  } else {
    // 如果量化方案不支持，则报错
    TORCH_CHECK(false, "Unsupported qscheme: ", toString(qtype));
  }

  // 设置运行时源张量的零点
  op_attr.set_zero_points_mask(DNNL_ARG_SRC, /* zero_points_mask= */0);
  // 初始化权重的副本和描述符
  at::Tensor weight_copy;
  ideep::tensor::desc w_desc;
  ideep::dims dims_iohw, dims_giohw;
  ideep::tag w_tag = ideep::tag::any;
  const bool with_groups = groups > 1;
  if (transpose) {
    // 如果是转置卷积，根据模板参数生成期望的权重描述符
    w_desc = ideep::convolution_transpose_forward::expected_weights_desc<true, false>(
        dims, dnnl::memory::data_type::s8,
        strides, padding_l, padding_r, dilates, groups,
        dnnl::algorithm::deconvolution_direct, dnnl::prop_kind::forward_inference,
        ideep::dims(), op_attr);
    // convolution_transpose_forward::expected_weights_desc() 返回的格式为 [i, o, ...]
    // ONEDNN要求的格式为 [o, i, ...]，因此需要进行维度置换
    dims_iohw = w_desc.get_dims();
    dims_giohw = with_groups ? ideep::utils::group_dims(dims_iohw, groups) : dims_iohw;
    // 创建维度置换的索引向量
    std::vector<int64_t> perms(dims_giohw.size(), 0); // for permutation of weight
    std::iota(perms.begin(), perms.end(), 0);
    std::swap(perms[with_groups], perms[with_groups + 1]);
    // 根据置换索引向量对权重进行维度置换
    weight_copy = weight.reshape(dims_giohw).permute(c10::IntArrayRef(perms)).clone();
  } else {
    // 如果不是转置卷积，生成期望的权重描述符并直接克隆权重
    w_desc = ideep::convolution_forward::expected_weights_desc(
        dims, dnnl::memory::data_type::s8,
        strides, padding_l, padding_r, dilates, groups,
        dnnl::algorithm::convolution_direct, dnnl::prop_kind::forward_inference,
        dnnl::memory::data_type::u8, ideep::dims(), op_attr, /*is_channels_last=*/true);
    weight_copy = weight.clone();
  }
  // 如果存在分组卷积，设置权重的标签
  if (with_groups) {
    w_tag = kSpatialDim == 2 ? ideep::tag::goihw : ideep::tag::goidhw;
  } else {
    // 根据空间维度判断使用的 IDEEP 标签，2维时使用 oihw，否则使用 oidhw
    w_tag = kSpatialDim == 2 ? ideep::tag::oihw : ideep::tag::oidhw;
  }
  // 获取权重张量的维度信息，如果有分组，则使用分组后的维度信息
  ideep::dims w_dims = with_groups ? ideep::utils::group_dims(w_desc.get_dims(), groups)
                                   : w_desc.get_dims();
  // 创建 IDEEP 张量对象 wgt，使用给定的维度、数据类型和标签，并初始化为给定的数据指针
  ideep::tensor wgt = ideep::tensor(
      ideep::tensor::desc({w_dims, dnnl::memory::data_type::s8, w_tag}, groups),
      weight_copy.data_ptr());
  // 设置权重张量的缩放因子，用于后续的 feed_from() 操作
  wgt.set_scale(wgt_scales);
  // 初始化一个 IDEEP 张量 exp_wgt，使用与 w_desc 相同的描述
  ideep::tensor exp_wgt;
  exp_wgt.init(w_desc);
  // 同样设置 exp_wgt 的缩放因子，以便 feed_from() 操作使用
  exp_wgt.set_scale(wgt_scales);
  // 将 wgt 数据复制到 exp_wgt 中，根据 transpose 参数决定是否转置
  exp_wgt.feed_from(wgt, transpose); // 期望 wgt 数据格式为 [OC IC KH KW]
  // 使用移动语义初始化一个指向 packed_weight_p 的 IDEEP 张量指针
  ideep::tensor * packed_weight_p = new ideep::tensor(std::move(exp_wgt));
  // 设置 packed_weight_p 的缩放因子
  packed_weight_p->set_scale(wgt_scales);
  // 设置 packed_weight_p 的零点
  packed_weight_p->set_zero_point(wgt_zero_points);
  // 使用 unique_ptr 管理 packed_weight_p，确保资源释放
  std::unique_ptr<ideep::tensor> weight_ptr(packed_weight_p);
  // 处理偏置
  std::optional<ideep::tensor> onednn_bias{c10::nullopt};
  // 如果存在偏置项
  if (bias.has_value()) {
    // 获取偏置向量
    at::Tensor bias_vec = bias.value();
    // 检查偏置向量的维度，应为 1
    TORCH_CHECK(bias_vec.dim() == 1, "bias should be a vector (1D Tensor)");
    // 检查偏置向量的长度是否与输出通道数相匹配
    TORCH_CHECK(
        bias_vec.size(0) == output_channels,
        "bias should have K elements: " + std::to_string(output_channels));
    // 创建 IDEEP 张量描述对象 bias_desc，并使用偏置数据初始化 packed_bias
    auto bias_desc = ideep::tensor::desc(bias.value().sizes().vec(), dnnl::memory::data_type::f32);
    ideep::tensor packed_bias;
    packed_bias.init(bias_desc, bias.value().data_ptr());
    // 使用 std::optional 管理 packed_bias
    onednn_bias = std::optional<ideep::tensor>(packed_bias);
  }
  // 创建 PackedConvWeightsOnednn 对象的智能指针 ret_ptr
  auto ret_ptr = c10::make_intrusive<PackedConvWeightsOnednn<kSpatialDim>>(
      PackedConvWeightsOnednn<kSpatialDim>{
        std::move(weight_ptr),
        onednn_bias,
        weight,
        bias,
        stride,
        padding,
        output_padding,
        dilation,
        groups,
        transpose
      });
  // 返回 PackedConvWeightsOnednn 对象的智能指针
  return ret_ptr;
}

// 实例化模板 PackedConvWeightsOnednn，分别为二维和三维情况
template struct PackedConvWeightsOnednn<2>;
template struct PackedConvWeightsOnednn<3>;

// 将量化卷积的权重打包为 Mkldnn 张量
at::Tensor _qconv_prepack_onednn(
    at::Tensor weight, // 从 CPU 后端获取的权重，而非 QuantizedCPU
    at::Tensor weight_scales, // 权重的零点在 onednn 中必须为 0
    double input_scale, // 输入的缩放因子
    int64_t input_zero_point, // 输入的零点
    torch::List<int64_t> stride, // 卷积步长
    torch::List<int64_t> padding, // 卷积填充
    torch::List<int64_t> dilation, // 卷积膨胀
    int64_t groups, // 卷积组数
    std::optional<torch::List<int64_t>> input_shape) { // 可选的输入形状
  int kSpatialDim = weight.ndimension() - 2; // 计算空间维度
  TORCH_CHECK(
      weight.ndimension() == kSpatialDim + 2,
      "Weights are expected to have ", kSpatialDim + 2, " dimensions"); // 检查权重张量的维度是否正确
  TORCH_CHECK(
      stride.size() == (decltype(stride.size()))kSpatialDim,
      "stride should contain ", kSpatialDim, " elements for ",
      kSpatialDim, "D convolution."); // 检查卷积步长的维度是否正确
  TORCH_CHECK(
      padding.size() == (decltype(padding.size()))kSpatialDim,
      "Specify front/top/left padding only. "
      "end/bottom/right padding assumed to be equal to front/top/left"); // 检查卷积填充的维度是否正确
  TORCH_CHECK(
      dilation.size() == (decltype(dilation.size()))kSpatialDim,
      "dilation should contain ", kSpatialDim, " elements for ",
      kSpatialDim, "D convolution."); // 检查卷积膨胀的维度是否正确

  bool is_1d = (1 == kSpatialDim); // 是否为一维卷积
  auto x_dims = input_shape.has_value()?input_shape.value().vec():ideep::dims(); // 获取输入形状的向量表示
  if (is_1d) {
    if (input_shape.has_value()) {
      // N, C, L -> N, C, 1, L
      x_dims.insert(x_dims.begin() + 2, 1); // 在输入形状中插入一个维度 1
    }
    if (weight.dim() == 3) {
      weight = weight.unsqueeze(quant_utils::kConv1dSqueezeDim + 2); // 如果权重是三维，添加一个维度
    }
    stride = quant_utils::MakeArgForConv1d(stride, 1); // 调整一维卷积的步长参数
    padding = quant_utils::MakeArgForConv1d(padding, 0); // 调整一维卷积的填充参数
    dilation = quant_utils::MakeArgForConv1d(dilation, 1); // 调整一维卷积的膨胀参数
    kSpatialDim += 1; // 更新空间维度为一维
  }
  auto w_dims = weight.sizes().vec(); // 获取权重张量的大小向量
  auto strides = stride.vec(); // 获取步长的向量表示
  auto padding_l = padding.vec(); // 获取左侧填充的向量表示
  auto padding_r = padding.vec(); // 获取右侧填充的向量表示（假设与左侧填充相同）
  auto dilates = dilation.vec(); // 获取膨胀的向量表示
  auto op_attr = ideep::attr_t(); // 创建一个属性对象

  ideep::scale_t weights_scales(weight_scales.numel()); // 根据权重规模创建缩放对象

  if (weight_scales.ndimension() == 0) {
    // 如果权重是每个张量量化的，则权重缩放为标量张量
    TORCH_CHECK(
        weight_scales.numel() == 1,
        "Weight is quant per tensor, weight scale expects 1 element but got ", weight_scales.numel(), " elements.");
#if IDEEP_PREREQ(3, 1, 0, 1)
    weights_scales[0] = weight_scales.item().toDouble(); // 使用权重缩放项的双精度值
#elif IDEEP_PREREQ(3, 1, 0, 0)
    weights_scales[0] = 1.0 / weight_scales.item().toDouble(); // ONEDNN 和 PyTorch 的缩放是倒数关系
#else
    TORCH_CHECK(false, "Unexpected IDeep version to do qconv weight prepack."); // 不支持的 IDeep 版本
#endif
  } else {
    // 如果权重是每个通道量化的
    for (int i = 0; i < weight_scales.numel(); ++i) {
#if IDEEP_PREREQ(3, 1, 0, 1)
      weights_scales[i] = weight_scales[i].item().toDouble(); // 使用每个通道的权重缩放项的双精度值
#elif IDEEP_PREREQ(3, 1, 0, 0)
      weights_scales[i] = 1.0 / weight_scales[i].item().toDouble(); // ONEDNN 和 PyTorch 的缩放是倒数关系
#else
      TORCH_CHECK(false, "Unexpected IDeep version to do qconv weight prepack."); // 不支持的 IDeep 版本
#endif
    }
  }
#else
      TORCH_CHECK(false, "Unexpected IDeep version to do qconv weight prepack.");
#endif
    }
  }


  // 如果不满足特定的IDEep版本要求，抛出错误
  else {
    TORCH_CHECK(false, "Unexpected IDeep version to do qconv weight prepack.");
  }



  if (input_scale != 1.0f) {
    op_attr.set_scales_mask(DNNL_ARG_SRC, /* src_scales_mask= */0);
  }
  if (input_zero_point != 0) {
    op_attr.set_zero_points_mask(DNNL_ARG_SRC, /* src_zero_points_mask= */0);
  }


  // 如果输入的缩放因子不为1.0，则设置源张量的缩放因子掩码为0
  if (input_scale != 1.0f) {
    op_attr.set_scales_mask(DNNL_ARG_SRC, /* src_scales_mask= */0);
  }
  // 如果输入的零点值不为0，则设置源张量的零点值掩码为0
  if (input_zero_point != 0) {
    op_attr.set_zero_points_mask(DNNL_ARG_SRC, /* src_zero_points_mask= */0);
  }



  at::Tensor weight_copy;
  ideep::tensor::desc w_desc;
  ideep::dims dims_iohw, dims_giohw;
  ideep::tag w_tag = ideep::tag::any;
  const bool with_groups = groups > 1;
  w_desc = ideep::convolution_forward::expected_weights_desc(
      w_dims, dnnl::memory::data_type::s8,
      strides, padding_l, padding_r, dilates, groups,
      dnnl::algorithm::convolution_direct, dnnl::prop_kind::forward_inference,
      dnnl::memory::data_type::u8, x_dims, op_attr, /*is_channels_last=*/true);


  // 初始化变量和描述符，用于存储权重数据
  at::Tensor weight_copy;
  // 权重的描述符，用于IDEep库的操作
  ideep::tensor::desc w_desc;
  // 存储维度的变量
  ideep::dims dims_iohw, dims_giohw;
  // 权重的标签，默认为任意标签
  ideep::tag w_tag = ideep::tag::any;
  // 是否使用分组卷积，判断条件为groups是否大于1
  const bool with_groups = groups > 1;
  // 根据给定参数生成权重的描述符，用于卷积操作的期望权重描述
  w_desc = ideep::convolution_forward::expected_weights_desc(
      w_dims, dnnl::memory::data_type::s8,
      strides, padding_l, padding_r, dilates, groups,
      dnnl::algorithm::convolution_direct, dnnl::prop_kind::forward_inference,
      dnnl::memory::data_type::u8, x_dims, op_attr, /*is_channels_last=*/true);



  // Note: Weight in Conv1D will unsqueeze into Conv2D in previous step
  weight_copy = weight.clone(c10::MemoryFormat::Contiguous);


  // 注意：在先前的步骤中，Conv1D中的权重将被展开成Conv2D
  // 克隆权重张量并保证其内存格式为Contiguous（连续的）
  weight_copy = weight.clone(c10::MemoryFormat::Contiguous);



  if (with_groups) {
    w_tag = kSpatialDim == 2 ? ideep::tag::goihw : ideep::tag::goidhw;
  } else {
    w_tag = kSpatialDim == 2 ? ideep::tag::oihw : ideep::tag::oidhw;
  }
  ideep::dims wei_dims = with_groups ? ideep::utils::group_dims(w_desc.get_dims(), groups)
                                  : w_desc.get_dims();
  ideep::tensor wgt = ideep::tensor(
      ideep::tensor::desc({wei_dims, dnnl::memory::data_type::s8, w_tag}, groups),
      weight_copy.data_ptr());


  // 如果使用了分组卷积，则根据空间维度选择相应的权重标签
  if (with_groups) {
    w_tag = kSpatialDim == 2 ? ideep::tag::goihw : ideep::tag::goidhw;
  } else {
    w_tag = kSpatialDim == 2 ? ideep::tag::oihw : ideep::tag::oidhw;
  }
  // 计算权重张量的维度，如果有分组则重新计算维度
  ideep::dims wei_dims = with_groups ? ideep::utils::group_dims(w_desc.get_dims(), groups)
                                  : w_desc.get_dims();
  // 创建IDEep张量对象，用于存储权重数据
  ideep::tensor wgt = ideep::tensor(
      ideep::tensor::desc({wei_dims, dnnl::memory::data_type::s8, w_tag}, groups),
      weight_copy.data_ptr());



  wgt.set_scale(weights_scales); // Scales are needed for feed_from().


  // 设置权重张量的缩放因子，feed_from()方法需要使用这些缩放因子
  wgt.set_scale(weights_scales);



  ideep::tensor exp_wgt;
  exp_wgt.init(w_desc);
  exp_wgt.set_scale(weights_scales); // Also for feed_from()
  exp_wgt.feed_from(wgt, /*transposed*/false); // expect wgt to be in [OC IC KH KW] format


  // 初始化期望的权重张量，并设置其缩放因子
  ideep::tensor exp_wgt;
  exp_wgt.init(w_desc);
  exp_wgt.set_scale(weights_scales);
  // 将实际的权重张量数据复制到期望的权重张量中，不进行转置
  exp_wgt.feed_from(wgt, /*transposed*/false); // expect wgt to be in [OC IC KH KW] format



  auto packed_weight = at::native::new_with_itensor_mkldnn(
      std::move(exp_wgt),
      c10::optTypeMetaToScalarType(weight_copy.options().dtype_opt()),
      weight_copy.options().device_opt());

  return packed_weight;
}


  // 使用MKLDNN的张量创建函数，基于期望的权重张量创建新的MKLDNN张量
  auto packed_weight = at::native::new_with_itensor_mkldnn(
      std::move(exp_wgt),
      c10::optTypeMetaToScalarType(weight_copy.options().dtype_opt()),
      weight_copy.options().device_opt());

  // 返回打包后的权重张量
  return packed_weight;
}



#endif // #if AT_MKLDNN_ENABLED()

namespace at {
namespace native {
namespace {

template <int kSpatialDim = 2>
class QConvPackWeightInt8 final {
 public:
  static c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>> run_conv(
      Tensor weight,
      std::optional<Tensor> bias,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> dilation,
      int64_t groups) {
    torch::List<int64_t> output_padding;
    output_padding.reserve(kSpatialDim);
    for (C10_UNUSED const auto idx : c10::irange(kSpatialDim)) {
      output_padding.push_back((int64_t)0);
    }
    return _run(weight, bias, stride, padding, output_padding, dilation, groups,
                /*transpose=*/false);
  }

  static c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>> run_deconv(
      Tensor weight,
      std::optional<Tensor> bias,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> output_padding,
      torch::List<int64_t> dilation,
      int64_t groups) {
    // 调用私有静态方法 `_run`，执行卷积操作，并返回卷积参数的包装对象
    return _run(weight, bias, stride, padding, output_padding, dilation, groups,
                /*transpose=*/true);
  }

 private:
  // 静态方法 `_run`，返回卷积参数的包装对象
  static c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>> _run(
      // 权重张量
      Tensor weight,
      // 可选的偏置张量
      std::optional<Tensor> bias,
      // 步长列表
      torch::List<int64_t> stride,
      // 填充列表
      torch::List<int64_t> padding,
      // 输出填充列表
      torch::List<int64_t> output_padding,
      // 膨胀列表
      torch::List<int64_t> dilation,
      // 分组数
      int64_t groups,
      // 是否转置操作
      bool transpose) {
    // 获取全局上下文
    auto& ctx = at::globalContext();
#ifdef USE_FBGEMM
  // 检查是否启用了 FBGEMM，且上下文的量化引擎为 X86
  if (ctx.qEngine() == at::QEngine::X86) {
#if AT_MKLDNN_ENABLED()
    // 检查是否启用了 MKLDNN，确定是否应该使用 ONEDNN 进行量化
    bool use_onednn = onednn_utils::should_use_onednn_quant(
          weight, transpose, groups, output_padding);
    // 如果应该使用 ONEDNN，返回使用 ONEDNN 进行预打包的权重
    if (use_onednn) {
      return PackedConvWeightsOnednn<kSpatialDim>::prepack(
          weight, bias, stride, padding, output_padding, dilation, groups, transpose);
    }
#endif
    // 返回使用 FBGEMM 进行预打包的权重
    return PackedConvWeight<kSpatialDim>::prepack(
        weight, bias, stride, padding, output_padding, dilation, groups, transpose);
  } // x86
#endif // defined(USE_FBGEMM) || AT_MKLDNN_ENABLED()

#ifdef USE_FBGEMM
    // 如果量化引擎为 FBGEMM，返回使用 FBGEMM 进行预打包的权重
    if (ctx.qEngine() == at::QEngine::FBGEMM) {
      return PackedConvWeight<kSpatialDim>::prepack(
          weight, bias, stride, padding, output_padding, dilation, groups,
          transpose);
    }
#endif

#ifdef USE_PYTORCH_QNNPACK
    // 如果量化引擎为 QNNPACK，返回使用 QNNPACK 进行预打包的权重
    if (ctx.qEngine() == at::QEngine::QNNPACK) {
      return PackedConvWeightsQnnp<kSpatialDim>::prepack(
          weight, bias, stride, padding, output_padding, dilation, groups,
          transpose);
    }
#endif

#if AT_MKLDNN_ENABLED()
    // 如果量化引擎为 ONEDNN，返回使用 ONEDNN 进行预打包的权重
    if (ctx.qEngine() == at::QEngine::ONEDNN) {
      return PackedConvWeightsOnednn<kSpatialDim>::prepack(
        weight, bias, stride, padding, output_padding, dilation, groups,
            transpose);
    }
#endif

    // 如果没有找到适合的引擎，抛出错误信息
    TORCH_CHECK(
        false,
        "Didn't find engine for operation quantized::conv2d_prepack ",
        toString(ctx.qEngine()));
  }
};



class QConv1dPackWeightInt8 final {
 public:
  // 运行 1D 卷积的权重打包函数
  static c10::intrusive_ptr<ConvPackedParamsBase<2>> run_conv(
      Tensor weight,
      std::optional<Tensor> bias,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> dilation,
      int64_t groups) {
    // 设置输出填充为默认值
    const torch::List<int64_t> output_padding({0});
    // 调用内部运行函数，传递参数并指定不进行转置操作
    return _run(std::move(weight), std::move(bias), stride, padding, output_padding, dilation, groups,
                /*transpose=*/false);
  }

  // 运行 1D 反卷积的权重打包函数
  static c10::intrusive_ptr<ConvPackedParamsBase<2>> run_deconv(
      Tensor weight,
      std::optional<Tensor> bias,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> output_padding,
      torch::List<int64_t> dilation,
      int64_t groups) {
    // 调用内部运行函数，传递参数并指定进行转置操作
    return _run(std::move(weight), std::move(bias), stride, padding, output_padding, dilation, groups,
                /*transpose=*/true);
  }

 private:
  // 内部运行函数，用于执行权重打包操作
  static c10::intrusive_ptr<ConvPackedParamsBase<2>> _run(
      Tensor weight,
      std::optional<Tensor> bias,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> output_padding,
      torch::List<int64_t> dilation,
      int64_t groups,
      bool transpose) {
    // 获取全局上下文
    auto& ctx = at::globalContext();
    // 如果权重是 3 维的，扩展为 4 维
    if (weight.dim() == 3) {
      weight = weight.unsqueeze(quant_utils::kConv1dSqueezeDim + 2);
    }
    // 对步幅参数进行 1D 卷积的处理
    stride = quant_utils::MakeArgForConv1d(stride, 1);
    // 对填充参数进行 1D 卷积的处理
    padding = quant_utils::MakeArgForConv1d(padding, 0);
    # 使用 quant_utils 模块的 MakeArgForConv1d 函数来设置 output_padding 为 0
    output_padding = quant_utils::MakeArgForConv1d(output_padding, 0);
    # 使用 quant_utils 模块的 MakeArgForConv1d 函数来设置 dilation 为 1
    dilation = quant_utils::MakeArgForConv1d(dilation, 1);
#ifdef USE_FBGEMM
    // 如果使用 FBGEMM 引擎
    if (ctx.qEngine() == at::QEngine::X86) {
#if AT_MKLDNN_ENABLED()
        // 检查是否应该使用 OneDNN 进行量化
        bool use_onednn = onednn_utils::should_use_onednn_quant(
            weight, transpose, groups, output_padding);
        // 如果应该使用 OneDNN，则调用相应的预打包函数
        if (use_onednn) {
            return PackedConvWeightsOnednn<2>::prepack(
                weight, bias, stride, padding, output_padding, dilation, groups,
                transpose);
        }
#endif
        // 否则，调用 FBGEMM 的预打包函数
        return PackedConvWeight<2>::prepack(
            std::move(weight), std::move(bias), stride, padding, output_padding, dilation, groups,
            transpose);
    } // x86
#endif

#ifdef USE_FBGEMM
    // 如果使用 FBGEMM 引擎
    if (ctx.qEngine() == at::QEngine::FBGEMM) {
        // 调用 FBGEMM 的预打包函数
        return PackedConvWeight<2>::prepack(
            std::move(weight), std::move(bias), stride, padding, output_padding, dilation, groups,
            transpose);
    }
#endif

#ifdef USE_PYTORCH_QNNPACK
    // 如果使用 QNNPACK 引擎
    if (ctx.qEngine() == at::QEngine::QNNPACK) {
        // 调用 QNNPACK 的预打包函数
        return PackedConvWeightsQnnp<2>::prepack(
            std::move(weight), std::move(bias), stride, padding, output_padding, dilation, groups,
            transpose);
    }
#endif

#if AT_MKLDNN_ENABLED()
    // 如果使用 ONEDNN 引擎
    if (ctx.qEngine() == at::QEngine::ONEDNN) {
        // 调用 OneDNN 的预打包函数
        return PackedConvWeightsOnednn<2>::prepack(
            weight, bias, stride, padding, output_padding, dilation, groups,
            transpose);
    }
#endif

    // 如果以上条件都不满足，抛出错误
    TORCH_CHECK(
        false,
        "Didn't find engine for operation quantized::conv1d_prepack ",
        toString(ctx.qEngine()));
}
};

class QConvPrepackOneDNN final {
 public:
  // 运行 OneDNN 引擎的量化卷积预打包
  static at::Tensor run_conv(
    at::Tensor weight, // 从 CPU 后端传入的权重
    at::Tensor weight_scales, // 权重的零点必须为 0，用于 OneDNN
    double input_scale, // 输入的量化比例
    int64_t input_zero_point, // 输入的零点
    torch::List<int64_t> stride, // 步长
    torch::List<int64_t> padding, // 填充
    torch::List<int64_t> dilation, // 空洞率
    int64_t groups, // 组数
    std::optional<torch::List<int64_t>> input_shape) { // 可选的输入形状
#if AT_MKLDNN_ENABLED()
    // 如果支持 MKLDNN，则调用 OneDNN 的量化卷积预打包函数
    return _qconv_prepack_onednn(
        weight, weight_scales, input_scale, input_zero_point,
        stride, padding, dilation, groups, input_shape);
#else
    // 否则抛出未实现错误
    TORCH_CHECK(false, "Unimplemented as onednn is not available.")
#endif
  }
};
// 定义 Torch 库实现：quantized 命名空间下的 QuantizedCPU 模块
TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
  // Conv
  // conv_prepack 已废弃，请使用 conv2d_prepack 进行二维卷积。
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv_prepack"), TORCH_FN(QConvPackWeightInt8<2>::run_conv));
  // Conv1d
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv1d_prepack"), TORCH_FN(QConv1dPackWeightInt8::run_conv));
  // Conv2d
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv2d_prepack"), TORCH_FN(QConvPackWeightInt8<2>::run_conv));
  // Conv3d
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv3d_prepack"), TORCH_FN(QConvPackWeightInt8<3>::run_conv));
  // ConvTranspose1d
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose1d_prepack"), TORCH_FN(QConv1dPackWeightInt8::run_deconv));
  // ConvTranspose2d
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose2d_prepack"), TORCH_FN(QConvPackWeightInt8<2>::run_deconv));
  // ConvTranspose3d
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose3d_prepack"), TORCH_FN(QConvPackWeightInt8<3>::run_deconv));
}

// 定义 Torch 库实现：_quantized 命名空间下的 QuantizedCPU 模块
TORCH_LIBRARY_IMPL(_quantized, QuantizedCPU, m) {
  // Conv2d
  m.impl(TORCH_SELECTIVE_NAME("_quantized::conv2d_prepack"), TORCH_FN(QConvPackWeightInt8<2>::run_conv));
  // Conv3d
  m.impl(TORCH_SELECTIVE_NAME("_quantized::conv3d_prepack"), TORCH_FN(QConvPackWeightInt8<3>::run_conv));
  // ConvTranspose1d
  m.impl(TORCH_SELECTIVE_NAME("_quantized::conv_transpose1d_prepack"), TORCH_FN(QConv1dPackWeightInt8::run_deconv));
  // ConvTranspose2d
  m.impl(TORCH_SELECTIVE_NAME("_quantized::conv_transpose2d_prepack"), TORCH_FN(QConvPackWeightInt8<2>::run_deconv));
  // ConvTranspose3d
  m.impl(TORCH_SELECTIVE_NAME("_quantized::conv_transpose3d_prepack"), TORCH_FN(QConvPackWeightInt8<3>::run_deconv));
}

// 定义 Torch 库实现：onednn 命名空间下的 CPU 模块
TORCH_LIBRARY_IMPL(onednn, CPU, m) {
  // 用于 PyTorch 2.0 导出的量化新操作定义
  // Conv Prepack
  m.impl(TORCH_SELECTIVE_NAME("onednn::qconv_prepack"), TORCH_FN(QConvPrepackOneDNN::run_conv));
}

} // namespace native
} // namespace at
```