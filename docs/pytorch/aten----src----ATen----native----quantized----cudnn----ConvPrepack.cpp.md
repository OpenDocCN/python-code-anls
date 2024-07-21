# `.\pytorch\aten\src\ATen\native\quantized\cudnn\ConvPrepack.cpp`

```py
#ifdef USE_CUDA
#include <ATen/cuda/CUDAConfig.h>  // 用于获取 AT_CUDNN_ENABLED 的定义

#if AT_CUDNN_ENABLED()

#include <ATen/ATen.h>
#include <torch/library.h>
#include <ATen/native/quantized/cpu/QuantUtils.h>
#include <ATen/native/quantized/cudnn/utils.h>
#include <ATen/native/quantized/PackedParams.h>
#include <ATen/quantized/Quantizer.h>
#include <c10/core/QScheme.h>
#include <c10/util/irange.h>
#include <torch/library.h>

#include <array>
#include <vector>

// 在 kSpatialDim 默认为 2 的情况下注册卷积参数
template <int kSpatialDim = 2>
int register_conv_params();

// 对 kSpatialDim 为 2 的情况进行外部模板实例化
extern template int register_conv_params<2>();
// 对 kSpatialDim 为 3 的情况进行外部模板实例化
extern template int register_conv_params<3>();

// 实现 CUDA 版本的量化卷积权重预打包函数
template <int kSpatialDim>
c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>> PackedConvWeightCudnn<
    kSpatialDim>::
    prepack(
        at::Tensor weight,
        std::optional<at::Tensor> bias,
        torch::List<int64_t> stride,
        torch::List<int64_t> padding,
        torch::List<int64_t> output_padding,
        torch::List<int64_t> dilation,
        int64_t groups,
        bool transpose) {
  // TODO: 需要查看在 Conv.cpp 中如何实现卷积操作的 groups 参数
  TORCH_CHECK(groups == 1, "Quantized cudnn conv2d is currently limited to groups = 1; received groups =", groups);
  TORCH_CHECK(weight.qscheme() == c10::kPerTensorAffine, "Unsupported qscheme: ", toString(weight.qscheme()));
  TORCH_CHECK(
      kSpatialDim == 2,  // 1D 被打包成 2D，因此我们不需要进行其他检查
      "cuDNN packing only supports 2D convolution.");
  TORCH_CHECK(
      weight.ndimension() == kSpatialDim + 2,
      "Weights are expected to have ",
      kSpatialDim + 2,
      " dimensions");
  TORCH_CHECK(
      stride.size() == kSpatialDim,
      "stride should contain ",
      kSpatialDim,
      " elements for ",
      kSpatialDim,
      "D convolution.");
  TORCH_CHECK(
      padding.size() == kSpatialDim,
      "quantized::conv_prepack (cudnn): Specify front/top/left padding only. "
      "end/bottom/right padding assumed to be equal to front/top/left");
  TORCH_CHECK(
      !transpose || output_padding.size() == kSpatialDim,
      "quantized::conv_prepack: Specify top/left output padding "
      "only. bottom/right padding assumed to be equal to top/left");
  TORCH_CHECK(
      dilation.size() == kSpatialDim,
      "quantized::conv_prepack (cudnn): dilation should contain ",
      kSpatialDim,
      " elements for ",
      kSpatialDim,
      "D convolution.");
  TORCH_CHECK(!transpose, "cudNN quantized conv prepack expects transpose = false")
  // 计算未填充输出通道的数量
  const int num_unpadded_output_channels = weight.size(0);
  // 获取权重的量化类型
  const auto qtype = weight.qscheme();
  if (bias.has_value()) {
    TORCH_CHECK(bias.value().dim() == 1, "bias should be a vector (1D Tensor)");
    TORCH_CHECK(
        bias.value().size(0) == num_unpadded_output_channels,
        "bias should have K elements: " + std::to_string(num_unpadded_output_channels));
    // TODO: 我们稍后会创建一个广播的偏置张量，因此这里可能不需要使其连续。


继续完成剩余代码的注释。
    // 当 NVIDIA 添加适当的广播支持后，我们将重新访问这段代码
    // bias_contig = bias->contiguous();
  }

  // cudnn v8.4.0 要求 conv2d 的 int8 权重张量的输入和输出通道数是4的倍数。如果不是，
  // 我们需要显式地将其填充到4的倍数，因为 cudnn 目前不支持填充。
  // TODO: 当 cudnn 在其运算符中启用填充时，可以移除我们的填充操作；
  // 目前，仅支持 groups=1 的填充。
  // TODO: 实现 groups > 1 的填充支持
  auto num_input_channels = weight.size(1);
  int8_t num_output_slices2pad = (4 - num_unpadded_output_channels % 4) % 4;
  int8_t num_input_slices2pad = (4 - num_input_channels % 4) % 4;
  if (num_output_slices2pad != 0 || num_input_slices2pad != 0) {
    // 第二个参数是填充值的初始化列表。每个维度有两个值。
    // 参考 https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html 获取更多细节
    // 使用常数填充方式，对 weight 进行填充
    weight = at::pad(weight, {0, 0, 0, 0, 0, num_input_slices2pad, 0, num_output_slices2pad}, "constant", 0);
    // 如果存在偏置，则对偏置进行填充
    if (bias.has_value()) {
      bias.value() = at::pad(bias.value(), {0, num_output_slices2pad}, "constant", 0);
    }
  }

  // 创建一个 PackedConvWeightCudnn 的智能指针 ret_ptr
  auto ret_ptr = c10::make_intrusive<PackedConvWeightCudnn<kSpatialDim>>(
          weight.to(c10::MemoryFormat::ChannelsLast), // TODO: 这里假设是2D的，可能需要更通用的方法？
          bias,
          stride,
          padding,
          output_padding,
          dilation,
          groups,
          transpose,
          qtype,
          num_unpadded_output_channels);
  // 返回 ret_ptr
  return ret_ptr;
    if (weight.dim() == 3) {
      // 如果权重张量的维度是3，表示当前使用的是Conv2d内核进行Conv1d操作，
      // 通过在输入和权重张量中添加一个虚拟的宽度维度（尺寸为1）来实现。
      // 输出通道，输入通道/组，L -> 输出通道，输入通道/组，1，L
      weight = weight.unsqueeze(-2);
    }
    // 使用quant_utils::MakeArgForConv1d函数将stride参数转换为Conv1d的格式
    stride = quant_utils::MakeArgForConv1d(stride, 1);
    // 使用quant_utils::MakeArgForConv1d函数将padding参数转换为Conv1d的格式
    padding = quant_utils::MakeArgForConv1d(padding, 0);
    // 使用quant_utils::MakeArgForConv1d函数将output_padding参数转换为Conv1d的格式
    output_padding = quant_utils::MakeArgForConv1d(output_padding, 0);
    // 使用quant_utils::MakeArgForConv1d函数将dilation参数转换为Conv1d的格式
    dilation = quant_utils::MakeArgForConv1d(dilation, 1);
    # 调用 PackedConvWeightCudnn<2>::prepack 方法，对卷积权重进行打包预处理
    return PackedConvWeightCudnn<2>::prepack(
        # 将给定的权重传递给方法
        weight, 
        # 将给定的偏置传递给方法
        bias, 
        # 传递卷积的步幅参数
        stride, 
        # 传递卷积的填充参数
        padding, 
        # 传递卷积输出的填充参数
        output_padding, 
        # 传递卷积的扩张参数
        dilation, 
        # 传递卷积的分组参数
        groups,
        # 传递是否进行转置操作的参数
        transpose);
  }
};

// 定义 TORCH_LIBRARY_IMPL 宏，注册 quantized 库中的 CUDA 加速的函数实现
TORCH_LIBRARY_IMPL(quantized, QuantizedCUDA, m) {
    // 注册二维卷积的参数
    register_conv_params<2>();
    // 注册三维卷积的参数
    register_conv_params<3>();
    // 实现 quantized::conv1d_prepack 的 CUDA 版本，使用 QConv1dPackWeightInt8Cudnn::run_conv 函数
    m.impl(TORCH_SELECTIVE_NAME("quantized::conv1d_prepack"), TORCH_FN(QConv1dPackWeightInt8Cudnn::run_conv));
    // 实现 quantized::conv2d_prepack 的 CUDA 版本，使用 QConvPackWeightInt8Cudnn<2>::run_conv 函数
    m.impl(TORCH_SELECTIVE_NAME("quantized::conv2d_prepack"), TORCH_FN(QConvPackWeightInt8Cudnn<2>::run_conv));
}

// 结束 quantized 命名空间
} // namespace
// 结束 native 命名空间
} // namespace native
// 结束 at 命名空间
} // namespace at

// 如果 AT_CUDNN_ENABLED 宏被定义，则结束此代码块
#ifndef AT_CUDNN_ENABLED
#endif // AT_CUDNN_ENABLED

// 如果 USE_CUDA 宏被定义，则结束此代码块
#ifndef USE_CUDA
#endif // USE_CUDA
```