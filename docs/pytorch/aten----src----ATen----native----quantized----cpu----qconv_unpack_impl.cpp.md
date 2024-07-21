# `.\pytorch\aten\src\ATen\native\quantized\cpu\qconv_unpack_impl.cpp`

```
// 包含必要的头文件：元组、向量、ATen张量库、Torch库，以及量化相关的CPU工具和库
#include <tuple>
#include <vector>
#include <ATen/ATen.h>
#include <torch/library.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/cpu/QnnpackUtils.h>
#include <ATen/native/quantized/cpu/OnednnUtils.h>
#include <ATen/native/quantized/cpu/QuantUtils.h>
#include <ATen/native/quantized/PackedParams.h>

#ifdef USE_FBGEMM
// 对于kSpatialDim维度的PackedConvWeight模板函数的unpack方法
template <int kSpatialDim>
std::tuple<at::Tensor, std::optional<at::Tensor>> PackedConvWeight<
    kSpatialDim>::unpack() {
  // 获取打包后的权重指针
  auto* packed_weights_p = w.get();
  // 输出通道数
  const int output_channels = packed_weights_p->outputChannels();
  // 输入通道数
  const int input_channels = packed_weights_p->inputChannels();
  // 分组数
  const int groups = packed_weights_p->groups();

  // 如果是2维空间维度，则kernel_d为1，否则取kernel数组的第一个元素
  const int kernel_d = kSpatialDim == 2 ? 1 : kernel[0];
  // kernel_h为kernel数组的倒数第二个元素
  const int kernel_h = kernel[kSpatialDim - 2];
  // kernel_w为kernel数组的最后一个元素
  const int kernel_w = kernel[kSpatialDim - 1];

  // 每组的通道数
  const int C_per_G = input_channels / groups;

  // 用于存储解包后权重的张量
  // 如果量化方案为kPerTensorAffine
  at::Tensor unpacked_weights;
  if (q_scheme == c10::kPerTensorAffine) {
    // 如果是2维空间维度
    unpacked_weights = kSpatialDim == 2
        ? at::_empty_affine_quantized(
              {output_channels, C_per_G, kernel_h, kernel_w},  // 形状为[输出通道数, 输入通道数/组数, kernel_h, kernel_w]
              device(c10::kCPU)                                // CPU设备
                  .dtype(c10::kQInt8)                           // 量化整数8位
                  .memory_format(c10::MemoryFormat::ChannelsLast),  // 内存格式为ChannelsLast
              w_scale[0],  // 权重比例因子
              w_zp[0],     // 权重零点
              c10::nullopt)
        // 否则，调用FBGEMM工具库的函数创建3维ChannelsLast格式的空的量化张量
        : at::native::fbgemm_utils::
              MakeEmptyAffineQuantizedChannelsLast3dTensor(
                  output_channels,
                  C_per_G,
                  kernel_d,
                  kernel_h,
                  kernel_w,
                  device(c10::kCPU).dtype(c10::kQInt8),  // 量化整数8位
                  w_scale[0],                          // 权重比例因子
                  w_zp[0]);                            // 权重零点
  } else if (q_scheme == c10::kPerChannelAffine) {
    // 如果是每通道的仿射量化方案
    TORCH_CHECK(
        !transpose(),  // 检查是否进行了转置，若是则抛出异常
        "Per Channel Quantization is currently disabled for transposed conv");
    // 从原始数组创建张量存储比例因子
    auto scales = at::from_blob(
        w_scale.data(), w_scale.size(), device(c10::kCPU).dtype(c10::kFloat));
    // 从原始数组创建张量存储零点
    auto zero_points = at::from_blob(
        w_zp.data(), w_zp.size(), device(c10::kCPU).dtype(c10::kInt));
    // 检查空间维度是否为2，选择不同的量化方式创建解包后的权重张量
    unpacked_weights = kSpatialDim == 2
        ? at::_empty_per_channel_affine_quantized(
              {output_channels, C_per_G, kernel_h, kernel_w},  // 创建指定形状的张量
              scales.toType(c10::kDouble),                     // 用给定的标量创建缩放因子张量
              zero_points.toType(c10::kLong),                  // 用给定的标量创建零点张量
              0, /* The output channel axis is 0 */             // 输出通道轴为0
              device(c10::kCPU).dtype(c10::kQInt8),            // 在CPU上创建QInt8类型的张量
              c10::MemoryFormat::ChannelsLast)                 // 内存格式设置为ChannelsLast
        : at::native::fbgemm_utils::
              MakeEmptyPerChannelAffineQuantizedChannelsLast3dTensor(
                  output_channels,                             // 输出通道数
                  C_per_G,                                     // 每组通道数
                  kernel_d,                                    // 卷积核深度
                  kernel_h,                                    // 卷积核高度
                  kernel_w,                                    // 卷积核宽度
                  device(c10::kCPU).dtype(c10::kQInt8),        // 在CPU上创建QInt8类型的张量
                  scales.toType(c10::kDouble),                 // 用给定的标量创建缩放因子张量
                  zero_points.toType(c10::kLong));             // 用给定的标量创建零点张量
    
    } else {
      // 如果量化方案不支持，抛出错误信息
      TORCH_CHECK(false, "Unsupported qscheme: ", toString(q_scheme));
    }
    
    // 将解包后的权重张量转换为int8_t类型指针
    int8_t* unpacked_weights_p =
        reinterpret_cast<int8_t*>(unpacked_weights.data_ptr<c10::qint8>());
    
    // 使用打包权重的指针来填充打包权重对象
    packed_weights_p->unpack(unpacked_weights_p);
    
    // 如果需要转置权重张量
    if(transpose()){
      // 对解包后的权重张量进行转置，以适应给定的空间维度和组数
      unpacked_weights =
          at::native::fbgemm_utils::TransposeConvTensorUnpackConversion<
              kSpatialDim>(unpacked_weights, groups);
    }
    
    // 返回解包后的权重张量和可选的偏置张量的元组
    return std::tuple<at::Tensor, std::optional<at::Tensor>>(
        unpacked_weights, bias);
#ifdef USE_FBGEMM
// 如果使用了 FBGEMM，定义模板特化的 PackedConvWeight<2> 的 unpack 方法
template std::tuple<at::Tensor, std::optional<at::Tensor>> PackedConvWeight<2>::unpack();
// 如果使用了 FBGEMM，定义模板特化的 PackedConvWeight<3> 的 unpack 方法
template std::tuple<at::Tensor, std::optional<at::Tensor>> PackedConvWeight<3>::unpack();
#endif // USE_FBGEMM

#ifdef USE_PYTORCH_QNNPACK
// 如果使用了 PYTORCH_QNNPACK，定义模板特化的 PackedConvWeightsQnnp<kSpatialDim>::unpack 方法
template <int kSpatialDim>
std::tuple<at::Tensor, std::optional<at::Tensor>> PackedConvWeightsQnnp<kSpatialDim>::unpack() {
  // 检查空间维度是否为2
  TORCH_CHECK(
      kSpatialDim == 2,
      "QNNPACK only supports conv2d_unpack right "
      "now.");
  // 检查原始权重是否已定义
  TORCH_CHECK(
        orig_weight.defined(),
        "Cannot unpack weights. "
        "Call at::globalContext()::setReleaseOriginalWeights(false) before packing or loading to enable unpacking.");
  // 返回原始权重和偏置的元组
  return std::tuple<at::Tensor, std::optional<at::Tensor>>(orig_weight, bias);
}

// 如果使用了 PYTORCH_QNNPACK，定义模板特化的 PackedConvWeightsQnnp<2> 的 unpack 方法
template std::tuple<at::Tensor, std::optional<at::Tensor>> PackedConvWeightsQnnp<2>::unpack();
// 如果使用了 PYTORCH_QNNPACK，定义模板特化的 PackedConvWeightsQnnp<3> 的 unpack 方法
template std::tuple<at::Tensor, std::optional<at::Tensor>> PackedConvWeightsQnnp<3>::unpack();
#endif // USE_PYTORCH_QNNPACK

#if AT_MKLDNN_ENABLED()
// 如果启用了 MKLDNN，定义模板特化的 PackedConvWeightsOnednn<kSpatialDim>::unpack 方法
template <int kSpatialDim>
std::tuple<at::Tensor, std::optional<at::Tensor>> PackedConvWeightsOnednn<kSpatialDim>::unpack() {
  // 返回原始权重的克隆和原始偏置的元组
  return std::tuple<at::Tensor, std::optional<at::Tensor>>(
      orig_weight_.clone(), orig_bias_);
}

// 如果启用了 MKLDNN，定义模板特化的 PackedConvWeightsOnednn<2> 的 unpack 方法
template std::tuple<at::Tensor, std::optional<at::Tensor>> PackedConvWeightsOnednn<2>::unpack();
// 如果启用了 MKLDNN，定义模板特化的 PackedConvWeightsOnednn<3> 的 unpack 方法
template std::tuple<at::Tensor, std::optional<at::Tensor>> PackedConvWeightsOnednn<3>::unpack();
#endif // #if AT_MKLDNN_ENABLED()
```