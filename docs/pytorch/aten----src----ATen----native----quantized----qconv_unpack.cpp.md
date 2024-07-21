# `.\pytorch\aten\src\ATen\native\quantized\qconv_unpack.cpp`

```
/*
The dispatch registrations at the end of this file applies to fbgemm, qnnpack, and cudnn backends.
The correct unpack backend function is determined using runtime polymorphism through the packed_weight pointer,
which is of type intrusive_ptr<ConvPackedParamsBase<kSpatialDim>> and points to either a PackedConvWeightsQnnp,
PackedConvWeights (Fbgemm), or PackedConvWeightsCudnn at runtime, which all inherit from ConvPackedParamsBase.
The implementations for the unpack functions can be found in /cpu/qconv_unpack_impl.cpp, for fbgemm&qnnpack
and /cudnn/ConvUnpackImpl.cpp, for cudnn.
*/

#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <tuple>

#include <ATen/core/Tensor.h>
#include <ATen/core/List.h>
#include <ATen/core/ivalue.h>
#include <torch/library.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/cpu/QnnpackUtils.h>
#include <ATen/native/quantized/cpu/OnednnUtils.h>
#include <ATen/native/quantized/cpu/QuantUtils.h>
#include <ATen/native/quantized/PackedParams.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/_empty_affine_quantized.h>
#include <ATen/ops/_empty_per_channel_affine_quantized.h>
#include <ATen/ops/from_blob.h>
#endif

template <int kSpatialDim = 2>
int register_conv_params();

extern template int register_conv_params<2>();
extern template int register_conv_params<3>();

namespace at {
namespace native {
namespace {

/*
 * QConvPackWeightInt8 expects its input tensor to be in shape
 * [output_channels, kernel_height, kernel_width, input_channels/Groups]
 * Therefore, the unpacking of packed weight tensor using QConvUnpackWeightsInt8
 * results in a tensor of the same shape.
 */

template <int kSpatialDim = 2>
class QConvUnpackWeightsInt8 final {
 public:
  // Static method to unpack packed weight tensor based on runtime engine
  static std::tuple<at::Tensor, std::optional<at::Tensor>> run(
      const c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>>& packed_weight) {
    auto& ctx = at::globalContext();

#ifdef USE_FBGEMM
    // Check if FBGEMM or X86 engine is used and unpack accordingly
    if (ctx.qEngine() == at::QEngine::FBGEMM ||
        ctx.qEngine() == at::QEngine::X86) {
      return packed_weight->unpack();
    }
#endif

#ifdef USE_PYTORCH_QNNPACK
    // Check if QNNPACK engine is used and unpack for Conv2d only
    if (ctx.qEngine() == at::QEngine::QNNPACK) {
      TORCH_CHECK(
          kSpatialDim == 2,
          "quantized::conv2d_unpack (qnnpack): QNNPACK only supports Conv2d "
          "now.");
      return packed_weight->unpack();
    }
#endif

#if AT_MKLDNN_ENABLED()
    // Check if ONEDNN engine is used and unpack accordingly
    if (ctx.qEngine() == at::QEngine::ONEDNN) {
      return packed_weight->unpack();
    }
#endif

    // Throw error if no matching engine is found
    TORCH_CHECK(
        false,
        "Didn't find engine for operation quantized::conv2d_unpack ",
        toString(ctx.qEngine()));
  }
};

class QConv1dUnpackWeightsInt8 final {
 public:
  // Static method to unpack packed weight tensor for 1D convolution
  static std::tuple<at::Tensor, std::optional<at::Tensor>> run(
      const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight) {
    auto& ctx = at::globalContext();
    at::Tensor weight;
    std::optional<at::Tensor> bias;
#ifdef USE_FBGEMM
    // 检查上下文中的量化引擎是否为 FBGEMM 或 X86
    if (ctx.qEngine() == at::QEngine::FBGEMM ||
        ctx.qEngine() == at::QEngine::X86) {
      // 如果是，解压缩打包的权重和偏置
      std::tie(weight, bias) = packed_weight->unpack();
      // 在权重张量上挤压移除指定维度（kConv1dSqueezeDim + 2）
      weight = weight.squeeze_(quant_utils::kConv1dSqueezeDim + 2);
      // 返回解压缩后的权重和偏置的元组
      return std::tuple<at::Tensor, std::optional<at::Tensor>>(weight, bias);
    }
#ifdef USE_PYTORCH_QNNPACK
    // 如果使用 QNNPACK 引擎，则执行以下操作
    if (ctx.qEngine() == at::QEngine::QNNPACK) {
      // 解压缩权重和偏置
      std::tie(weight, bias) = packed_weight->unpack();
      // 克隆权重张量
      at::Tensor new_weight = weight.clone();
      // 压缩维度为 quant_utils::kConv1dSqueezeDim + 2 的权重张量
      new_weight = new_weight.squeeze_(quant_utils::kConv1dSqueezeDim + 2);
      // 返回新的权重和偏置的元组
      return std::tuple<at::Tensor, std::optional<at::Tensor>>(new_weight, bias);
    }
#endif

#if AT_MKLDNN_ENABLED()
    // 如果启用了 MKLDNN
    if (ctx.qEngine() == at::QEngine::ONEDNN) {
      // 解压缩权重和偏置
      std::tie(weight, bias) = packed_weight->unpack();
      // 克隆权重张量
      at::Tensor new_weight = weight.clone();
      // 压缩维度为 quant_utils::kConv1dSqueezeDim + 2 的权重张量
      new_weight.squeeze_(quant_utils::kConv1dSqueezeDim + 2);
      // 返回新的权重和偏置的元组
      return std::tuple<at::Tensor, std::optional<at::Tensor>>(new_weight, bias);
    }
#endif

    // 如果未找到匹配的引擎，抛出错误
    TORCH_CHECK(
        false,
        "Didn't find engine for operation quantized::conv1d_unpack ",
        toString(ctx.qEngine()));
  }
};

// 用于处理卷积操作的步幅
template <int kSpatialDim = 2>
class QConvStride final {
 public:
  static torch::List<int64_t> run(
      const c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>>& packed_weight) {
    return packed_weight->stride();
  }
};

// 用于处理卷积操作的填充
template <int kSpatialDim = 2>
class QConvPadding final {
 public:
  static torch::List<int64_t> run(
      const c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>>& packed_weight) {
    return packed_weight->padding();
  }
};

// 用于处理卷积操作的输出填充
template <int kSpatialDim = 2>
class QConvOutputPadding final {
 public:
  static torch::List<int64_t> run(
      const c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>>& packed_weight) {
    return packed_weight->output_padding();
  }
};

// 用于处理卷积操作的膨胀率
template <int kSpatialDim = 2>
class QConvDilation final {
 public:
  static torch::List<int64_t> run(
      const c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>>& packed_weight) {
    return packed_weight->dilation();
  }
};

// 用于处理卷积操作的分组
template <int kSpatialDim = 2>
class QConvGroups final {
 public:
  static int64_t run(
      const c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>>& packed_weight) {
    return packed_weight->groups();
  }
};

// 用于处理卷积操作的转置
template <int kSpatialDim = 2>
class QConvTranspose final {
 public:
  static int64_t run(
      const c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>>& packed_weight) {
    return packed_weight->transpose();
  }
};

// 解包量化预装载的卷积参数的大小
IValue
unpack_quantized_prepacked_sizes_conv2d(const IValue& ivalue) {
  auto params = ivalue.toCustomClass<ConvPackedParamsBase<2>>();
  auto [weight, bias] = params->unpack();
  at::OptionalIntArrayRef bias_sizes = c10::nullopt;
  // 如果存在偏置且已定义，则获取其大小
  if (bias && bias->defined()) {
    bias_sizes = bias->sizes();
  }
  // 返回包含权重大小、偏置大小（如果存在）、步幅、填充、膨胀率、分组的元组
  return IValue(std::make_tuple(
      weight.sizes(),
      bias_sizes,
      params->stride(),
      params->padding(),
      params->dilation(),
      params->groups()));
}

} // namespace native
} // namespace at
```