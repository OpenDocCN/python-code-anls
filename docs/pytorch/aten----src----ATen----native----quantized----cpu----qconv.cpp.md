# `.\pytorch\aten\src\ATen\native\quantized\cpu\qconv.cpp`

```
// 定义宏以启用仅用于方法操作符的 Torch 断言
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 包含必要的 C++ 标准库头文件
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

// 包含 ATen 库的相关头文件
#include <ATen/core/Tensor.h>
#include <ATen/core/List.h>
#include <ATen/Context.h>
#include <ATen/Parallel.h>
#include <ATen/TensorOperators.h>
#include <ATen/SmallVector.h>
#include <ATen/native/quantized/PackedParams.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/cpu/QnnpackUtils.h>
#include <ATen/native/quantized/cpu/XnnpackUtils.h>
#include <ATen/native/quantized/cpu/OnednnUtils.h>
#include <ATen/native/quantized/ConvUtils.h>
#include <ATen/native/quantized/cpu/QuantUtils.h>
#include <caffe2/utils/threadpool/pthreadpool-cpp.h>
#include <torch/library.h>
#include <ATen/quantized/Quantizer.h>
#include <ATen/native/mkldnn/MKLDNNCommon.h>

// 根据宏定义选择包含的 ATen 头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_empty_affine_quantized.h>
#include <ATen/ops/_empty_affine_quantized_native.h>
#include <ATen/ops/_empty_per_channel_affine_quantized_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/quantize_per_channel_native.h>
#include <ATen/ops/quantize_per_tensor_native.h>
#include <ATen/ops/zeros.h>
#endif

// 包含 C10 实用工具的范围函数
#include <c10/util/irange.h>

// 定义一个匿名命名空间，限定作用域以防止全局命名冲突
namespace {
// 用于最大矩阵尺寸的健全性检查的常量声明
constexpr int64_t kReasonableMaxDim = 1000000;
} // namespace

// 定义一个模板函数 ConvDimChecks，用于卷积维度检查
template <int kSpatialDim = 2>
bool ConvDimChecks(
    int64_t act_dims,           // 激活张量的维度
    int64_t stride_dims,        // 步幅张量的维度
    int64_t padding_dims,       // 填充张量的维度
    int64_t output_padding_dims,// 输出填充张量的维度
    int64_t dilation_dims,      // 扩张张量的维度
    std::string func_name,      // 函数名称字符串
    bool transpose = false) {   // 是否进行转置的标志位，默认为 false
  // 使用 Torch 的断言检查激活张量维度是否为 kSpatialDim + 2
  TORCH_CHECK(
      act_dims == kSpatialDim + 2,
      func_name,
      kSpatialDim,
      "d(): Expected activation tensor to have ",
      kSpatialDim + 2,
      " dimensions, got ",
      act_dims);
  // 使用 Torch 的断言检查步幅张量维度是否为 kSpatialDim
  TORCH_CHECK(
      stride_dims == kSpatialDim,
      func_name,
      kSpatialDim,
      "d(): Expected stride tensor to have ",
      kSpatialDim,
      " dimensions, got ",
      stride_dims);
  // 使用 Torch 的断言检查填充张量维度是否为 kSpatialDim
  TORCH_CHECK(
      padding_dims == kSpatialDim,
      func_name,
      kSpatialDim,
      "d(): Expected padding tensor to have ",
      kSpatialDim,
      " dimensions, got ",
      padding_dims);
  // 如果进行转置操作，则使用 Torch 的断言检查输出填充张量维度是否为 kSpatialDim
  TORCH_CHECK(
      !transpose || (output_padding_dims == kSpatialDim),
      func_name,
      kSpatialDim,
      "d(): Expected output padding tensor to have ",
      kSpatialDim,
      " dimensions, got ",
      output_padding_dims);
  // 使用 Torch 的断言检查扩张张量维度是否为 kSpatialDim
  TORCH_CHECK(
      dilation_dims == kSpatialDim,
      func_name,
      kSpatialDim,
      "d(): Expected dilation tensor to have ",
      kSpatialDim,
      " dimensions, got ",
      dilation_dims);
  // 返回 true 表示维度检查通过
  return true;
}
// 计算反卷积层输出的形状大小
inline int64_t compute_deconv_shape(int64_t input,
                                    int64_t kernel,
                                    int64_t stride,
                                    int64_t input_padding,
                                    int64_t output_padding,
                                    int64_t dilation) {
  // 使用反卷积计算公式计算输出的大小
  int64_t out = (input - 1) * stride - 2 * input_padding
                + dilation * (kernel - 1) + output_padding + 1;
  return out;
}

// 创建反卷积操作的输出形状
template <int64_t kSpatialDim>
at::SmallVector<int64_t, kSpatialDim + 2> MakeDeConvOutputShape(
    int64_t N, int64_t M,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& kernel,
    const torch::List<int64_t>& stride,
    const torch::List<int64_t>& input_padding,
    const torch::List<int64_t>& output_padding,
    const torch::List<int64_t>& dilation) {
  // 初始化输出形状的向量
  at::SmallVector<int64_t, kSpatialDim + 2> output_shape;
  output_shape.resize(kSpatialDim + 2);
  output_shape[0] = N;  // 批大小
  output_shape[1] = M;  // 输出通道数
  // 遍历空间维度，计算每个维度的反卷积输出形状
  for (const auto idx : c10::irange(kSpatialDim)) {
    output_shape[idx + 2] = compute_deconv_shape(input_shape[idx],
                                                 kernel[idx],
                                                 stride[idx],
                                                 input_padding[idx],
                                                 output_padding[idx],
                                                 dilation[idx]);
    // 检查输出维度是否大于零
    TORCH_CHECK(output_shape[idx + 2] > 0,
                "Output dimension is zero for ", idx, " axis;"
                " kernel: ", kernel[idx],
                ", stride: ", stride[idx],
                ", input padding: ", input_padding[idx],
                ", output padding: ", output_padding[idx],
                ", dilation: ", dilation[idx])
    // 检查输出维度是否在合理的最大值内
    TORCH_CHECK(output_shape[idx + 2] < kReasonableMaxDim,
                "Output dimension is beyond reasonable maximum for ", idx,
                " axis;"
                " kernel: ", kernel[idx],
                ", stride: ", stride[idx],
                ", input padding: ", input_padding[idx],
                ", output padding: ", output_padding[idx],
                ", dilation: ", dilation[idx]);
  }
  return output_shape;
}

#ifdef USE_FBGEMM

// 根据指定的空间维度创建卷积输出形状（特化为2维的情况）
template <>
at::SmallVector<int64_t, 4> MakeConvOutputShape<2>(
    int N,
    int M,
    const std::array<int, 2>& output_image_shape) {
  // 返回由输入参数构成的卷积输出形状向量
  return {N, M, output_image_shape[0], output_image_shape[1]};
}

// 根据指定的空间维度创建卷积输出形状（特化为3维的情况）
template <>
at::SmallVector<int64_t, 5> MakeConvOutputShape<3>(
    int N,
    int M,
    const std::array<int, 3>& output_image_shape) {
  // 返回由输入参数构成的卷积输出形状向量
  return {N,
          M,
          output_image_shape[0],
          output_image_shape[1],
          output_image_shape[2]};
}

#endif // USE_FBGEMM

#ifdef USE_PYTORCH_QNNPACK

// 如果使用PyTorch QNNPACK，可能会有额外的实现

#endif // USE_PYTORCH_QNNPACK
// 声明一个模板函数 MakeInputShape，返回一个大小为 kSpatialDim 的 int64_t 数组，表示输入的形状
template <size_t kSpatialDim>
std::array<int64_t, kSpatialDim> MakeInputShape(
    int64_t D,  // 深度（对于三维数据，通常代表高度）
    int64_t H,  // 高度
    int64_t W); // 宽度

// 特化模板函数 MakeInputShape，当 kSpatialDim 为 2 时的实现，返回一个含有 H 和 W 的数组
template <>
std::array<int64_t, 2> MakeInputShape(int64_t /*D*/, int64_t H, int64_t W) {
  return {H, W};
}

// 特化模板函数 MakeInputShape，当 kSpatialDim 为 3 时的实现，返回一个含有 D、H 和 W 的数组
template <>
std::array<int64_t, 3> MakeInputShape(int64_t D, int64_t H, int64_t W) {
  return {D, H, W};
}

#endif // USE_PYTORCH_QNNPACK

#ifdef USE_FBGEMM
// 定义模板类 PackedConvWeight 的成员函数 GetBiasData，返回指向偏置数据的指针
template <int kSpatialDim>
const float* PackedConvWeight<kSpatialDim>::GetBiasData(at::Tensor* bias_ptr) {
  const float* bias_data = nullptr;
  if (bias.has_value()) {
    *bias_ptr = bias.value(); // 将 bias 的值赋给 bias_ptr 指向的 Tensor
    TORCH_CHECK(
        bias_ptr->dtype() == at::kFloat,
        "[QConv3D] The 'bias' tensor must have 'torch.float' dtype");
    *bias_ptr = bias_ptr->contiguous(); // 将 bias_ptr 指向的 Tensor 转换为连续的存储方式
    TORCH_CHECK(bias_ptr->dim() == 1, "bias should be a vector (1D Tensor)");
    const int M = w->outputChannels(); // 获取权重 w 的输出通道数
    TORCH_CHECK(bias_ptr->size(0) == M, "bias should have ", M, " elements.");
    bias_data = bias_ptr->data_ptr<float>(); // 获取 bias_ptr 指向的 Tensor 数据的 float 指针
  }
  return bias_data; // 返回偏置数据的指针，如果没有偏置则返回 nullptr
}

// 定义模板类 PackedConvWeight 的成员函数 GetQuantizationParams，计算量化参数
template <int kSpatialDim>
void PackedConvWeight<kSpatialDim>::GetQuantizationParams(
    float act_scale, // 激活值的缩放比例
    float out_scale, // 输出值的缩放比例
    std::vector<float>* output_multiplier_float, // 输出的浮点数乘子数组
    std::vector<float>* act_times_w_scale) { // 激活值乘以权重缩放比例数组
  if (q_scheme == c10::kPerTensorAffine) { // 如果量化方案是按张量的仿射方式
    *act_times_w_scale = {(act_scale * w_scale[0])}; // 计算激活值乘以权重缩放比例数组的第一个元素
    *output_multiplier_float = {act_times_w_scale->front() / out_scale}; // 计算输出乘子数组的第一个元素
  } else if (q_scheme == c10::kPerChannelAffine) { // 如果量化方案是按通道的仿射方式
    const int M = w->outputChannels(); // 获取权重 w 的输出通道数
    output_multiplier_float->resize(M); // 调整输出乘子数组和激活值乘以权重缩放比例数组的大小为 M
    act_times_w_scale->resize(M);
    for (const auto i : c10::irange(M)) {
      act_times_w_scale->at(i) = (act_scale * w_scale[i]); // 计算每个通道的激活值乘以权重缩放比例
      output_multiplier_float->at(i) = act_times_w_scale->at(i) / out_scale; // 计算每个通道的输出乘子
    }
  } else {
    TORCH_CHECK(false, "[QConv", kSpatialDim, "D] Unknown quantization scheme"); // 如果未知的量化方案，则报错
  }
}

// 定义模板类 PackedConvWeight 的成员函数 apply，应用卷积操作并返回结果 Tensor
template <int kSpatialDim>
at::Tensor PackedConvWeight<kSpatialDim>::apply(
    const at::Tensor& input,
    double output_scale,
    int64_t output_zero_point) {
  return apply_impl<false>(input, output_scale, output_zero_point); // 调用模板成员函数 apply_impl，不包含 ReLU 激活
}

// 定义模板类 PackedConvWeight 的成员函数 apply_relu，应用带 ReLU 激活的卷积操作并返回结果 Tensor
template <int kSpatialDim>
at::Tensor PackedConvWeight<kSpatialDim>::apply_relu(
    const at::Tensor& input,
    double output_scale,
    int64_t output_zero_point) {
  return apply_impl<true>(input, output_scale, output_zero_point); // 调用模板成员函数 apply_impl，包含 ReLU 激活
}

// 定义模板类 PackedConvWeight 的成员函数 apply_impl，执行卷积操作并返回结果 Tensor
template <int kSpatialDim>
template <bool kReluFused>
at::Tensor PackedConvWeight<kSpatialDim>::apply_impl(
    const at::Tensor& act,
    double output_scale,
    int64_t output_zero_point) {
  // 定义一个函数，用于执行量化卷积操作，支持二维和三维空间维度
  int64_t output_zero_point) {
  // 量化卷积核心代码基于 NHWC（通道在最后）布局设计
  // 理想情况下，应与 conv2d 的行为兼容，并保留输入布局（进行必要的向上转换）

  // 为了更加健壮，当前强制输出布局始终为 NHWC（通道在最后），以优化性能
  //
  // 当完整的内存格式支持到位时，这种做法可能会改变
  // 参见 https://github.com/pytorch/pytorch/issues/23403
  const std::string func_name = transpose() ? "quantized::conv_transpose"
                                            : "quantized::conv";
  
  // 检查当前 CPU 是否支持 FBGEMM
  TORCH_CHECK(
      fbgemm::fbgemmSupportedCPU(), "Your CPU does not support FBGEMM.");
  
  // 检查激活数据的数据类型是否为 c10::kQUInt8
  TORCH_CHECK(act.scalar_type() == c10::kQUInt8,
                func_name,
                "(FBGEMM): Expected activation data type ",
                toString(c10::kQUInt8),
                " but got ",
                toString(act.scalar_type()));

  // 执行卷积维度检查，确保输入参数的一致性
  ConvDimChecks<kSpatialDim>(
      act.ndimension(), stride().size(), padding().size(),
      output_padding().size(), dilation().size(), func_name, transpose());

  // 获取输入张量的维度信息
  const int N = act.size(0);        // 批量大小
  const int C = act.size(1);        // 输入通道数
  const int D = kSpatialDim == 2 ? 1 : act.size(2);  // 深度维度
  const int H = act.size(kSpatialDim);               // 高度维度
  const int W = act.size(kSpatialDim + 1);           // 宽度维度

  // 将输入张量转换为 NHWC 格式的张量，若为三维则调用专用转换函数
  const at::Tensor act_ndhwc = kSpatialDim == 2
      ? act.contiguous(c10::MemoryFormat::ChannelsLast)
      : at::native::fbgemm_utils::ConvertToChannelsLast3dTensor(act);

  // 获取转换后的 NHWC 格式数据的指针
  const uint8_t* act_data =
      reinterpret_cast<uint8_t*>(act_ndhwc.data_ptr<c10::quint8>());
  
  // 获取打包后的卷积权重数据的指针
  auto* pack_w = w.get();

  // 获取卷积输出的通道数
  const int M = pack_w->outputChannels();

  // 获取卷积核的尺寸
  const int kernel_d = kSpatialDim == 2 ? 1 : kernel[0];
  const int kernel_h = kernel[kSpatialDim - 2];
  const int kernel_w = kernel[kSpatialDim - 1];

  // 获取填充尺寸
  const int pad_d = kSpatialDim == 2 ? 0 : padding_[0];
  const int pad_h = padding_[kSpatialDim - 2];
  const int pad_w = padding_[kSpatialDim - 1];

  // 获取步幅尺寸
  const int stride_d = kSpatialDim == 2 ? 1 : stride_[0];
  const int stride_h = stride_[kSpatialDim - 2];
  const int stride_w = stride_[kSpatialDim - 1];

  // 获取膨胀尺寸
  const int dilation_d = kSpatialDim == 2 ? 1 : dilation_[0];
  const int dilation_h = dilation_[kSpatialDim - 2];
  const int dilation_w = dilation_[kSpatialDim - 1];

  // 获取输出填充尺寸
  const int output_padding_d = kSpatialDim == 2 ? 0 : output_padding_[0];
  const int output_padding_h = output_padding_[kSpatialDim - 2];
  const int output_padding_w = output_padding_[kSpatialDim - 1];

  // 若为二维卷积，执行以下代码块
  if (kSpatialDim == 2) {
    // 检查输入通道数是否与权重中的输入通道数相匹配，若不匹配则抛出错误信息
    TORCH_CHECK(
        C == pack_w->inputChannels(),
        "[QConv2D] Given groups=", groups_,
        ", weight of size ", M,
        ", ", kernel_h,
        ", ", kernel_w,
        ", ", pack_w->inputChannels(),
        ", expected input (NCHW) ", N,
        ", ", C,
        ", ", H,
        ", ", W,
        " to have ", pack_w->inputChannels(),
        " channels, but got ", C,
        " channels instead");
  } else {
    // 检查输入通道数是否与权重中的输入通道数相匹配，若不匹配则抛出错误信息
    TORCH_CHECK(
        C == pack_w->inputChannels(),
        "[QConv3D] Given groups=", groups_,
        ", weight of size ", M,
        ", ", kernel_d,
        ", ", kernel_h,
        ", ", kernel_w,
        ", ", pack_w->inputChannels(),
        ", expected input (NCDHW) ", N,
        ", ", C,
        ", ", D,
        ", ", H,
        ", ", W,
        " to have ", pack_w->inputChannels(),
        " channels, but got ", C,
        " channels instead");
  }

  // 根据空间维度创建卷积参数对象 conv_p
  fbgemm::conv_param_t<kSpatialDim> conv_p =
      at::native::fbgemm_utils::MakeFbgemmConvParam<kSpatialDim>(
          N, // 批处理大小
          C, // 输入通道数
          M, // 输出通道数
          kSpatialDim == 2 ? std::vector<int>{H, W} : std::vector<int>{D, H, W}, // 空间尺寸
          groups_, // 分组数
          kSpatialDim == 2 ? std::vector<int>{kernel_h, kernel_w} : std::vector<int>{kernel_d, kernel_h, kernel_w}, // 卷积核尺寸
          kSpatialDim == 2 ? std::vector<int>{stride_h, stride_w} : std::vector<int>{stride_d, stride_h, stride_w}, // 步幅
          kSpatialDim == 2 ? std::vector<int>{pad_h, pad_w} : std::vector<int>{pad_d, pad_h, pad_w}, // 填充
          kSpatialDim == 2 ? std::vector<int>{dilation_h, dilation_w} : std::vector<int>{dilation_d, dilation_h, dilation_w}, // 扩张
          kSpatialDim == 2 ? std::vector<int>{output_padding_h, output_padding_w} : std::vector<int>{output_padding_d, output_padding_h, output_padding_w}, // 输出填充
          transpose()); // 是否转置

  // 使用 act 对象的量化比例和零点值来初始化激活尺度和零点值
  const float act_scale = act.q_scale();
  const int32_t act_zero_point = act.q_zero_point();

  at::Tensor bias;
  // 获取偏置数据的指针
  const float* bias_data = GetBiasData(&bias);

  // 检查权重尺度和零点的大小是否匹配
  TORCH_CHECK(
      w_scale.size() == w_zp.size(),
      "Weight scales and zero points vectors should have the same size.");

  // 初始化输出乘子和激活与权重尺度的乘积
  std::vector<float> output_multiplier_float;
  std::vector<float> act_times_w_scale;
  GetQuantizationParams(
      act_scale, output_scale, &output_multiplier_float, &act_times_w_scale);

  // 初始化输出形状的小型向量
  at::SmallVector<int64_t, kSpatialDim + 2> output_shape;
  if (transpose()) {
    // 根据给定的空间维度 kSpatialDim 创建反卷积输出的形状
    output_shape = MakeDeConvOutputShape<kSpatialDim>(
        N,
        M,
        kSpatialDim == 2 ? std::vector<int64_t>{H, W} : std::vector<int64_t>{D, H, W},
        kernel,
        stride(),
        padding(),
        output_padding(),
        dilation());

    // 如果使用直接卷积实现，需要在模型初始化阶段计算权重矩阵的列偏移
    // 我们需要知道输出矩阵的形状来计算直接卷积的列偏移
    // 因此不能在权重打包函数内部调用它，类似于其他量化卷积实现方式
    if (pack_w->getPackedWForDirectconv().get() &&
        pack_w->getPackedWForDirectconv().get()->is_first_call()) {
          // 调用列偏移函数，用于 s8acc32_DirectConvT
          pack_w->getPackedWForDirectconv().get()->col_offsets_with_zero_pt_s8acc32_DirectConvT(
              conv_p,
              w_zp.data(),
              col_offsets,
              M);
    }
  } else {
    // 根据给定的空间维度 kSpatialDim 创建卷积输出的形状
    output_shape = MakeConvOutputShape<kSpatialDim>(N, M, conv_p.OUT_DIM);
  }

  // 如果输出张量的任何维度小于或等于0，抛出异常
  if (N > 0) {
    TORCH_CHECK(
        std::all_of(
            output_shape.begin(),
            output_shape.end(),
            [](int64_t i) { return i > 0; }),
        "[QConv",
        kSpatialDim,
        "D] each dimension of output tensor should be greater than 0");
  }

  // 根据空间维度 kSpatialDim 创建量化的输出张量
  at::Tensor output = kSpatialDim == 2
      ? at::_empty_affine_quantized(
            output_shape,
            device(c10::kCPU)
                .dtype(c10::kQUInt8)
                .memory_format(c10::MemoryFormat::ChannelsLast),
            output_scale,
            output_zero_point,
            c10::nullopt)
      : at::native::fbgemm_utils::MakeEmptyAffineQuantizedChannelsLast3dTensor(
            output_shape[0],
            output_shape[1],
            output_shape[2],
            output_shape[3],
            output_shape[4],
            device(c10::kCPU).dtype(c10::kQUInt8),
            output_scale,
            output_zero_point);

  // 创建一个与输出张量具有相同大小和选项的空缓冲区张量
  at::Tensor buffer =
      at::empty(output.sizes(), output.options().dtype(c10::kInt));

  // 获取并行任务的数量
  const int num_tasks = at::get_num_threads();

  // 并行执行一个无操作的任务，用于占位符
  at::parallel_for(0, num_tasks, 1, [&](int64_t begin, int64_t end) {
    fbgemm::DoNothing<> kNoOpObj{};
    for (const auto task_id : c10::irange(begin, end)) {
      // 遍历任务 ID 的范围，执行下面的代码块
      if (q_scheme == c10::kPerTensorAffine) {
        // 如果量化方案为每张量仿射量化
        fbgemm::ReQuantizeOutput<
            kReluFused,
            fbgemm::QuantizationGranularity::TENSOR,
            float>
            output_proc_obj(
                kNoOpObj,
                output_multiplier_float.data(),
                output_zero_point,
                act_zero_point,
                w_zp.data(),
                nullptr, /* row offset buffer */
                col_offsets.data(),
                bias_data,
                M,
                groups_,
                act_times_w_scale.data());
        // 使用设置的量化参数创建输出处理对象
        fbgemm::fbgemmConv<decltype(output_proc_obj), kSpatialDim, int32_t>(
            conv_p,
            act_data,
            *pack_w,
            reinterpret_cast<uint8_t*>(output.data_ptr<c10::quint8>()),
            buffer.data_ptr<int32_t>(),
            output_proc_obj,
            task_id /* thread_id*/,
            num_tasks /* num_threads */);
        // 调用 FBGEMM 库的卷积函数，使用输出处理对象对数据进行处理
      } else if (q_scheme == c10::kPerChannelAffine) {
        // 如果量化方案为每通道仿射量化
        fbgemm::ReQuantizeOutput<
            kReluFused,
            fbgemm::QuantizationGranularity::OUT_CHANNEL,
            float>
            output_proc_obj(
                kNoOpObj,
                output_multiplier_float.data(),
                output_zero_point,
                act_zero_point,
                w_zp.data(),
                nullptr, /* row offset buffer */
                col_offsets.data(),
                bias_data,
                M,
                groups_,
                act_times_w_scale.data());
        // 使用设置的量化参数创建输出处理对象
        fbgemm::fbgemmConv<decltype(output_proc_obj), kSpatialDim, int32_t>(
            conv_p,
            act_data,
            *pack_w,
            reinterpret_cast<uint8_t*>(output.data_ptr<c10::quint8>()),
            buffer.data_ptr<int32_t>(),
            output_proc_obj,
            task_id /* thread_id*/,
            num_tasks /* num_threads */);
        // 调用 FBGEMM 库的卷积函数，使用输出处理对象对数据进行处理
      }
    }
  });

  return output;



    for (const auto task_id : c10::irange(begin, end)) {
      // Iterate over the range of task IDs and execute the following block of code for each.

      if (q_scheme == c10::kPerTensorAffine) {
        // If the quantization scheme is per tensor affine
        fbgemm::ReQuantizeOutput<
            kReluFused,
            fbgemm::QuantizationGranularity::TENSOR,
            float>
            output_proc_obj(
                kNoOpObj,
                output_multiplier_float.data(),
                output_zero_point,
                act_zero_point,
                w_zp.data(),
                nullptr, /* row offset buffer */
                col_offsets.data(),
                bias_data,
                M,
                groups_,
                act_times_w_scale.data());
        // Create an output processing object with specified quantization parameters

        fbgemm::fbgemmConv<decltype(output_proc_obj), kSpatialDim, int32_t>(
            conv_p,
            act_data,
            *pack_w,
            reinterpret_cast<uint8_t*>(output.data_ptr<c10::quint8>()),
            buffer.data_ptr<int32_t>(),
            output_proc_obj,
            task_id /* thread_id*/,
            num_tasks /* num_threads */);
        // Call the FBGEMM library's convolution function using the output processing object
      } else if (q_scheme == c10::kPerChannelAffine) {
        // If the quantization scheme is per channel affine
        fbgemm::ReQuantizeOutput<
            kReluFused,
            fbgemm::QuantizationGranularity::OUT_CHANNEL,
            float>
            output_proc_obj(
                kNoOpObj,
                output_multiplier_float.data(),
                output_zero_point,
                act_zero_point,
                w_zp.data(),
                nullptr, /* row offset buffer */
                col_offsets.data(),
                bias_data,
                M,
                groups_,
                act_times_w_scale.data());
        // Create an output processing object with specified quantization parameters

        fbgemm::fbgemmConv<decltype(output_proc_obj), kSpatialDim, int32_t>(
            conv_p,
            act_data,
            *pack_w,
            reinterpret_cast<uint8_t*>(output.data_ptr<c10::quint8>()),
            buffer.data_ptr<int32_t>(),
            output_proc_obj,
            task_id /* thread_id*/,
            num_tasks /* num_threads */);
        // Call the FBGEMM library's convolution function using the output processing object
      }
    }
  });

  return output;
}

template at::Tensor PackedConvWeight<2>::apply(
    const at::Tensor& act,
    double output_scale,
    int64_t output_zero_point);
# 定义 PackedConvWeight<2> 类的 apply 方法的模板特化，接受一个张量 act，以及输出比例和零点作为参数，返回一个张量。

template at::Tensor PackedConvWeight<2>::apply_relu(
    const at::Tensor& act,
    double output_scale,
    int64_t output_zero_point);
# 定义 PackedConvWeight<2> 类的 apply_relu 方法的模板特化，接受一个张量 act，以及输出比例和零点作为参数，返回一个张量。

template at::Tensor PackedConvWeight<3>::apply(
    const at::Tensor& act,
    double output_scale,
    int64_t output_zero_point);
# 定义 PackedConvWeight<3> 类的 apply 方法的模板特化，接受一个张量 act，以及输出比例和零点作为参数，返回一个张量。

template at::Tensor PackedConvWeight<3>::apply_relu(
    const at::Tensor& act,
    double output_scale,
    int64_t output_zero_point);
# 定义 PackedConvWeight<3> 类的 apply_relu 方法的模板特化，接受一个张量 act，以及输出比例和零点作为参数，返回一个张量。

template at::Tensor PackedConvWeight<2>::apply_impl<false>(
    const at::Tensor& act,
    double output_scale,
    int64_t output_zero_point);
# 定义 PackedConvWeight<2> 类的 apply_impl 方法的模板特化（模板参数为 false），接受一个张量 act，以及输出比例和零点作为参数，返回一个张量。

template at::Tensor PackedConvWeight<3>::apply_impl<false>(
  const at::Tensor& act,
  double output_scale,
  int64_t output_zero_point);
# 定义 PackedConvWeight<3> 类的 apply_impl 方法的模板特化（模板参数为 false），接受一个张量 act，以及输出比例和零点作为参数，返回一个张量。

#endif // USE_FBGEMM
# 如果定义了 USE_FBGEMM 宏，则结束条件编译区段。

#ifdef USE_PYTORCH_QNNPACK
# 如果定义了 USE_PYTORCH_QNNPACK 宏，则开始条件编译区段。

#ifdef USE_XNNPACK
# 如果定义了 USE_XNNPACK 宏，则开始条件编译区段。

template <int kSpatialDim>
template <typename scalar_t, bool kReluFused>
at::Tensor PackedConvWeightsQnnp<kSpatialDim>::apply_impl_xnnp(
# 定义 PackedConvWeightsQnnp<kSpatialDim> 类的 apply_impl_xnnp 方法，接受 kSpatialDim 维度、scalar_t 类型和 kReluFused 布尔值作为模板参数，返回一个张量。
    const at::Tensor& act, double output_scale, int64_t output_zero_point) {
  using underlying_t = typename scalar_t::underlying;

  // 使用互斥锁保护共享资源，确保线程安全操作
  std::lock_guard<std::mutex> lock(qnnp_mutex_);

  // 根据是否转置选择合适的函数名称，用于日志记录或错误消息
  const std::string func_name = transpose()
      ? "quantized::conv_transpose (xnnpack)"
      : "quantized::conv (xnnpack)";
  
  // 检查空间维度是否为2，因为XNNPACK不支持3D卷积
  TORCH_CHECK(
      kSpatialDim == 2,
      func_name, ": xnnpack does not currently support 3d convolution.");

  /*
   * 注意:
   * [de]conv_prepack 在预打包（prepack()）期间为权重（值、缩放因子和零点）准备，假设激活类型为uint8_t。
   * 但实际情况可能并非如此。解决方案可能涉及让预打包程序了解输入的量化类型。但目前，这些组件还不能将模型级信息传递给预打包函数。
   * 因此，在此函数中，如果发现输入的量化类型不是uint8_t，我们必须对权重进行处理。这包括在必要时复制和转换uint8_t为int8_t。
   * 此外，由于XNNPACK在撰写本文时不支持quint8_t的逐通道权重，我们添加以下断言以确保不会出现该情况。
   * 在处理权重时采取捷径，这意味着当我们在XNNPACK中启用缺失功能时，需要重新审视和修复某些权重处理逻辑。
   *
   * 下表总结了如何处理权重:
   *
   * .-------------------------------------------------------------------------.
   * | input_qdtype |              uint8_t            |            int8_t      |
   * | per_channel  |       yes       |       no      |      yes     |    no   |
   * |-------------------------------------------------------------------------|
   * | zero_points  | at::zeros()*    | orig_zp + 128 | at:zeros()** | orig_zp |
   * | scale        |            dtype = float, 不需要更改                      |
   * | values       |        总是在传递给XNNPACK之前进行处理                    |
   * .-------------------------------------------------------------------------.
   *
   * 注释: * - 对于uint8_t + per_channel的零点: XNNPACK中不支持，需要在支持后进行修复。
   * ** - 对于int8_t的零点: 对称量化意味着XNNPACK将忽略核心的零点。
   */

  // 如果scalar_t的底层类型是quint8，则执行以下逻辑
  if constexpr (std::is_same_v<underlying_t, c10::quint8>) {
    // 检查是否支持逐通道量化，如果不支持则抛出错误信息
    TORCH_CHECK(!per_channel(),
      func_name, ": xnnpack does not currently have per_channel support with activation dtype of c10::quint8."
  );

// 更多检查
ConvDimChecks<kSpatialDim>(
    act.ndimension(),                       // 激活张量的维度
    stride().size(),                        // 卷积核的步长大小
    padding().size(),                       // 填充的大小
    output_padding().size(),                // 输出填充的大小
    dilation().size(),                      // 膨胀的大小
    func_name,                              // 函数名称
    transpose());                           // 是否转置

const int64_t N = act.size(0);              // 获取激活张量的 batch 大小
const int64_t H = act.size(2);              // 获取激活张量的高度
const int64_t W = act.size(3);              // 获取激活张量的宽度
const int64_t D = 1;                        // 固定深度为1
const int64_t M = bias.size(0);             // 获取偏置张量的大小

const auto act_nhwc = act.contiguous(c10::MemoryFormat::ChannelsLast);   // 获取通道在最后的激活张量
const auto act_input_scale = act_nhwc.q_scale();                         // 获取激活张量的量化比例

auto status = xnn_status_invalid_state;     // 初始化状态为无效状态

// 如果操作符不存在，或者输入的量化比例不可用或不等于激活张量的量化比例，则创建一个新的操作符
if (!xnnp_convolution_op ||
    (!input_scale.has_value() || input_scale.value() != act_input_scale)) {
  xnn_operator_t xnnp_op = nullptr;         // 定义 XNN 操作符为空

  // 更新输入的量化比例以便缓存操作符
  input_scale = act_input_scale;

  // 创建一个空的张量用于装载权重
  const at::Tensor weight_contig =
      orig_weight.contiguous(c10::MemoryFormat::ChannelsLast);          // 获取通道在最后的原始权重张量
  const float* w_scales_data = w_scales.const_data_ptr<float>();        // 获取权重的量化比例数据
  underlying_t w_zp = 0;                                                // 初始化权重的零点
  at::Tensor weight_tensor;

  if (!per_channel()) {
    w_zp = static_cast<underlying_t>(
        weight_contig.q_zero_point() +
        (std::is_same<underlying_t, uint8_t>::value ? 128 : 0));        // 计算权重的零点值

    // 创建一个仿射量化的权重张量
    weight_tensor = at::native::empty_affine_quantized(
        weight_contig.sizes(),
        c10::CppTypeToScalarType<scalar_t>::value,
        c10::nullopt /* layout */,
        c10::kCPU,
        c10::nullopt /* pin_memory */,
        w_scales_data[0],
        w_zp,
        c10::MemoryFormat::ChannelsLast);
  } else { /* per_channel */
    // 创建一个按通道仿射量化的权重张量
    weight_tensor = at::native::empty_per_channel_affine_quantized(
        weight_contig.sizes(),
        w_scales,
        at::zeros(w_scales.sizes(), at::kInt), /* 参考上述关于 w_zp 的注释 */
        weight_contig.q_per_channel_axis(),
        c10::CppTypeToScalarType<scalar_t>::value,
        c10::nullopt /* layout */,
        c10::kCPU,
        c10::nullopt /* pin_memory */,
        c10::MemoryFormat::ChannelsLast);
  }

  // 复制原始权重并处理必要的数据类型转换
  at::native::xnnp_utils::q8_copy_int8_weight_and_add_offset<scalar_t>(
      weight_contig, weight_tensor);

  // 将权重张量转换为通道在最后的格式，以便 XNN 使用
  const at::Tensor xnnp_weight =
      at::native::xnnp_utils::convert_conv_weights_to_channel_last_tensor<
          kSpatialDim>(weight_tensor, groups(), transpose());

  // 设置输出的最小值，考虑是否与 ReLU 合并
  auto output_min = kReluFused
      ? activationLimits<underlying_t>(output_scale, output_zero_point, Activation::RELU).first
      : std::numeric_limits<underlying_t>::min();
    auto output_max = kReluFused
        // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
        ? activationLimits<underlying_t>(output_scale, output_zero_point, Activation::RELU).second
        : std::numeric_limits<underlying_t>::max();

// 根据 kReluFused 是否为真来选择输出的最大值，若为真，则使用 RELU 激活函数的上限作为输出的最大值，否则使用 underlying_t 类型的最大值。


    // Original bias was float, so we requantize it here.
    at::Tensor qbias = quant_utils::QuantizeBias(per_channel(), bias, weight_contig, act_input_scale);

// 将原始的浮点偏置重新量化为 qbias 张量，使用给定的量化参数和偏置值。


    status = at::native::xnnp_utils::xnnp_create_convolution2d_nhwc(
        padding()[0],
        padding()[1],
        padding()[0],
        padding()[1],
        kernel_[0],
        kernel_[1],
        stride()[0],
        stride()[1],
        dilation()[0],
        dilation()[1],
        groups(),
        !transpose() ? orig_weight.size(1) : orig_weight.size(0) / groups(),
        !transpose() ? orig_weight.size(0) / groups() : orig_weight.size(1),
        !transpose() ? orig_weight.size(1) * groups() : orig_weight.size(0),
        !transpose() ? orig_weight.size(0) : orig_weight.size(1) * groups(),
        act_nhwc.q_zero_point(),
        act_input_scale,
        w_zp, /* will be ignored for Q[SC]8, see comment
                above about w_zp*/
        w_scales_data,
        reinterpret_cast<const underlying_t*>(
            xnnp_weight.template data_ptr<scalar_t>()),
        reinterpret_cast<int32_t*>(qbias.template data_ptr<c10::qint32>()),
        output_zero_point,
        output_scale,
        output_min,
        output_max,
        0,
        &xnnp_op,
        per_channel(),
        transpose());

// 调用 XNNPack 库的二维卷积操作创建函数 `xnnp_create_convolution2d_nhwc`，传入卷积的各种参数和张量数据，生成卷积操作符 `xnnp_op`。


    xnnp_convolution_op = xnnpack_operator(xnnp_op);
    TORCH_CHECK(
        status == xnn_status_success,
        func_name,
        ": xnn create operator failed(",
        status,
        ")");
  }

// 将 XNNPack 创建的卷积操作符 `xnnp_op` 转换为 `xnnp_convolution_op`，并检查创建过程中的状态 `status` 是否成功，若失败则输出失败信息和状态码。


  at::SmallVector<int64_t, kSpatialDim + 2> output_shape;
  const auto input_shape = MakeInputShape<kSpatialDim>(D, H, W);
  if (transpose()) {
    output_shape = MakeDeConvOutputShape<kSpatialDim>(
        N, M, {H, W}, kernel_, stride(), padding(), output_padding(), dilation());
  } else {
    output_shape = at::native::quantized::MakeConvOutputShape<kSpatialDim>(
        N, M, input_shape, kernel_, stride(), padding(), dilation());
  }

// 根据卷积是否转置，计算输出的形状 `output_shape`，使用不同的函数 `MakeDeConvOutputShape` 或 `MakeConvOutputShape`。


  if (act_nhwc.numel() > 0) {

// 检查激活张量 `act_nhwc` 的元素数量是否大于 0，如果是则继续执行下一步操作。
    TORCH_CHECK(
        std::all_of(
            output_shape.begin(),
            output_shape.end(),
            [](int64_t i) { return i > 0; }),
        func_name, ": ", kSpatialDim, "d (xnnpack): each dimension of output tensor should be greater than 0.")
  }

  // Allocate output Tensor and a buffer for XNNPACK to use
  // 为输出张量分配内存，并为 XNNPACK 分配使用的缓冲区
  at::Tensor output = at::native::empty_affine_quantized(
      output_shape,
      c10::CppTypeToScalarType<scalar_t>::value,
      c10::nullopt /* layout */,
      c10::kCPU,
      c10::nullopt /* pin_memory */,
      output_scale,
      output_zero_point,
      c10::MemoryFormat::ChannelsLast);

  // Reshape the operator
  // 重塑操作符
  status = at::native::xnnp_utils::xnnp_reshape_convolution2d_nhwc(
      xnnp_convolution_op.get(),
      N,
      H,
      W,
      caffe2::pthreadpool_(),
      per_channel(),
      transpose(),
      output_padding()[0],
      output_padding()[1]);

  TORCH_CHECK(
      status == xnn_status_success,
      func_name,
      ": xnn setup operator failed(",
      status,
      ")");

  // Setup the operator
  // 设置操作符
  status = at::native::xnnp_utils::xnnp_setup_convolution2d_nhwc(
      xnnp_convolution_op.get(),
      reinterpret_cast<const underlying_t*>(act_nhwc.template data_ptr<scalar_t>()),
      reinterpret_cast<underlying_t*>(output.template data_ptr<scalar_t>()),
      per_channel(),
      transpose());

  TORCH_CHECK(
      status == xnn_status_success,
      func_name,
      ": xnn setup operator failed(",
      status,
      ")");

  // Run the operator
  // 运行操作符
  status = xnn_run_operator(
      xnnp_convolution_op.get(), /* xnn_operator_t op */
      caffe2::pthreadpool_()); /* pthreadpool_t threadpool */

  TORCH_CHECK(
      status == xnn_status_success,
      func_name,
      ": xnn run operator failed(",
      status,
      ")");

  return output;
// 结束条件标记，表明这是一个 C++ 的预处理器指令，用于条件编译
}

#endif // USE_XNNPACK

// 模板类的成员函数定义的开始，模板参数为 kSpatialDim，该模板还包含另一个模板参数 kReluFused
template <int kSpatialDim>
template <bool kReluFused>
// 函数签名：应用量化卷积权重和偏置到输入张量 act 上，返回处理后的张量
at::Tensor PackedConvWeightsQnnp<kSpatialDim>::apply_impl(
    const at::Tensor& act, // 输入参数：量化的激活张量
    double output_scale, // 输出量化参数：输出的缩放因子
    int64_t output_zero_point) { // 输出量化参数：输出的零点

  // QNNPack 不是线程安全的，使用互斥锁确保线程安全
  std::lock_guard<std::mutex> lock(qnnp_mutex_);

  // 确定函数名称，根据是否转置决定是卷积还是转置卷积
  const std::string func_name = transpose() ? "quantized::conv_transpose"
                                            : "quantized::conv";

  // 断言：如果启用了融合的 ReLU 并且进行了转置操作，则抛出错误
  TORCH_CHECK(!(kReluFused && transpose()),
              kSpatialDim == 2,
              func_name, kSpatialDim,
              "d (qnnpack): ConvTranspose cannot be fused with ReLU.");

  // 断言：确保输入的激活张量的数据类型是 uint8
  TORCH_CHECK(act.scalar_type() == c10::kQUInt8,
              func_name,
              "(qnnpack): Expected activation data type ",
              toString(c10::kQUInt8),
              " but got ",
              toString(act.scalar_type()));

  // 检查卷积操作的维度是否合法，例如输入、输出通道数、填充等参数
  ConvDimChecks<kSpatialDim>(
      act.ndimension(), stride().size(), padding().size(),
      output_padding().size(), dilation().size(), func_name, transpose());

  // 获取预先打包的权重指针
  auto* pack_w = w.get();

  // TODO：当更新预打包以实际执行打包时，可以替换为 packB->getOutputChannels()
  // 输出通道索引，根据是否转置卷积来决定索引位置
  const int out_ch_idx = transpose() ? 1 : 0;
  // 输出通道数，从偏置张量中获取
  const auto out_ch = bias.size(0);

  // 输入张量的维度信息，按 NCHW 或 NDHWC 语义排列
  const int N = act.size(0); // batch size
  const int C = act.size(1); // input channels
  const int D = kSpatialDim == 3 ? act.size(2) : 1; // depth (for 3D convolutions)
  const int H = act.size(kSpatialDim); // height
  const int W = act.size(kSpatialDim + 1); // width
  const int M = out_ch; // 输出通道数

  // 指定通道的内存布局格式，根据 kSpatialDim 不同选择不同的格式
  const auto channels_last = kSpatialDim == 2
      ? c10::MemoryFormat::ChannelsLast
      : c10::MemoryFormat::ChannelsLast3d;

  // 将输入张量转换为 NDHWC 的内存格式，以便 QNNPack 使用
  const at::Tensor act_ndhwc = act.contiguous(channels_last);

  // 计算输出的最小和最大值，如果启用了融合的 ReLU，则进行相应的激活限制
  auto output_min = kReluFused
      ? activationLimits<uint8_t>(output_scale, output_zero_point, Activation::RELU).first
      : std::numeric_limits<uint8_t>::min();
  auto output_max = kReluFused
      ? activationLimits<uint8_t>(output_scale, output_zero_point, Activation::RELU).second
      : std::numeric_limits<uint8_t>::max();

  // 获取输入张量的量化比例
  double act_input_scale = act_ndhwc.q_scale();

  // 如果没有指定输入的量化比例或者与输入张量的量化比例不匹配，则抛出错误
  if (!input_scale.has_value() || input_scale.value() != act_input_scale) {
    // 断言：输出通道数必须与权重和偏置的大小匹配
    TORCH_CHECK(M == (transpose() ? groups() : 1) * orig_weight.size(out_ch_idx),
        "Output channel size of weight and bias must match.");
    // 断言：输入通道数必须与权重和偏置的大小匹配
    TORCH_CHECK(C == (transpose() ? 1 : groups()) * orig_weight.size(1 - out_ch_idx),
        "Input channel size of weight and bias must match.");

    // 获取原始权重，并将其调整为 uint8 类型（从 int8 类型）
    auto weight_contig = orig_weight.contiguous(channels_last);
    // 获取偏置张量并转换为 float32 类型
    auto bias_fp32 = bias;


这段代码涉及到卷积操作的量化和输入数据的预处理，包括输入数据的格式转换、权重和偏置的读取，以及一些错误检查和线程安全性的处理。
    // 将 weight_contig 的数据指针强制转换为 int8_t 类型指针
    int8_t* w_data =
        reinterpret_cast<int8_t*>(weight_contig.template data_ptr<c10::qint8>());

    // 获取 weight_scales 的 float 数据指针
    float* weight_scales_data = w_scales.data_ptr<float>();

    // 调用 generate_requantization_scales 函数计算重新量化比例尺，用于后续的 qnnpack 后端处理
    // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
    generate_requantization_scales(
        w_scales, act_input_scale, output_scale, requantization_scales);

    // TODO Kimish, 不论是否按通道处理，都分配 affine_quantized。
    // 此分配仅用于打包权重，打包完成后将被释放。应保持一致性，需要修复。
    // 创建一个空的仿射量化张量 qnnp_weight，用于存储量化后的权重数据
    at::Tensor qnnp_weight = at::_empty_affine_quantized(
        weight_contig.sizes(),
        at::device(c10::kCPU).dtype(c10::kQUInt8).memory_format(channels_last),
        weight_scales_data[0],
        w_zero_points[0],
        c10::nullopt);

    // 获取 qnnp_weight 的 quint8 数据指针
    auto* qnnp_w_data = qnnp_weight.template data_ptr<c10::quint8>();

    // 获取 weight_contig 的元素总数
    auto wt_numel = weight_contig.numel();

    // 将原始数据偏移量为 128，存入 qnnp_weight 中，进行量化
    for (const auto i : c10::irange(wt_numel)) {
      qnnp_w_data[i] = static_cast<c10::quint8>(w_data[i] + 128);
    }

    // 对原始偏置进行量化，转换为 qint32 类型的张量 qbias
    at::Tensor qbias = quant_utils::QuantizeBias(convolution_op->per_channel, bias_fp32, weight_contig, act_input_scale);

    // 更新输入比例尺，以避免再次打包
    input_scale = act_input_scale;

    // 重置 w 指针，并使用 qnnpack::PrePackConvWeights 类重新包装卷积权重
    w.reset();
    w = std::make_unique<qnnpack::PrePackConvWeights>(
        convolution_op.get(),
        w_zero_points.data(),
        reinterpret_cast<uint8_t*>(qnnp_w_data),
        reinterpret_cast<int32_t*>(qbias.template data_ptr<c10::qint32>()));

    // 将 pack_w 指向 w 的内容
    pack_w = w.get();

    // 如果在预打包时释放权重
    if (at::globalContext().releaseWeightsWhenPrepacking()) {
        // 在移动设备上，通过重置 intrusive_ptr 释放原始权重
        // 在此之后调用 unpack 将触发断言
        orig_weight.reset();
    }

    // 将填充缓冲区设置为零点，仅当 zero_buffer_size 存在时执行
    if (zero_buffer_size) {
      memset(
          convolution_op->zero_buffer,
          act_ndhwc.q_zero_point(),
          zero_buffer_size);
    }
  }

  // 断言 pack_w 不为 nullptr，确保打包后的权重有效
  TORCH_INTERNAL_ASSERT(pack_w != nullptr, "Packed Weights are NULL");

  // 初始化输出形状为一个 SmallVector
  at::SmallVector<int64_t, kSpatialDim + 2> output_shape;

  // 构造输入形状
  const auto input_shape = MakeInputShape<kSpatialDim>(D, H, W);

  // 根据是否转置，确定输出形状
  if (transpose()) {
    output_shape = MakeDeConvOutputShape<kSpatialDim>(
        N,
        M,
        kSpatialDim == 2 ? std::vector<int64_t>{H, W} : std::vector<int64_t>{D, H, W},
        kernel_,
        stride(),
        padding(),
        output_padding(),
        dilation());
  } else {
    output_shape = at::native::quantized::MakeConvOutputShape<kSpatialDim>(
        N, M, input_shape, kernel_, stride(), padding(), dilation());
  }

  // 如果 act_ndhwc 张量的元素数量大于 0
  if (act_ndhwc.numel() > 0) {
    // 使用 TORCH_CHECK 确保输出形状的每个维度都大于零
    TORCH_CHECK(
        std::all_of(
            output_shape.begin(),
            output_shape.end(),
            [](int64_t i) { return i > 0; }),
        func_name,
        kSpatialDim,
        "d (qnnpack): each dimension of output tensor should "
        "be greater than 0.")
  }

  // 分配输出 Tensor 和 QNNPACK 使用的缓冲区
  at::Tensor output = at::native::empty_affine_quantized(
      output_shape,
      c10::kQUInt8,
      c10::nullopt /* layout */,
      c10::kCPU,
      c10::nullopt /* pin_memory */,
      output_scale,
      output_zero_point,
      channels_last);

  // 定义 QNNPACK 运行状态变量
  pytorch_qnnp_status run_status;
  // 如果需要转置，使用 qnnpackDeConv 运行 QNNPACK 反卷积操作
  if (transpose()) {
    run_status = qnnpack::qnnpackDeConv(
        convolution_op.get(),
        pack_w->getPackedWeights(),
        N,
        H,
        W,
        act_ndhwc.q_zero_point(),
        reinterpret_cast<uint8_t*>(act_ndhwc.template data_ptr<c10::quint8>()),
        w_zero_points.data(),
        requantization_scales.data(),
        output.q_zero_point(),
        output_min,
        output_max,
        reinterpret_cast<uint8_t*>(output.template data_ptr<c10::quint8>()),
        caffe2::pthreadpool_());
  } else {
    // 否则，使用 qnnpackConv 运行 QNNPACK 卷积操作
    run_status = qnnpack::qnnpackConv(
        convolution_op.get(),
        pack_w->getPackedWeights(),
        N,
        D,
        H,
        W,
        act_ndhwc.q_zero_point(),
        reinterpret_cast<uint8_t*>(act_ndhwc.template data_ptr<c10::quint8>()),
        w_zero_points.data(),
        requantization_scales.data(),
        output.q_zero_point(),
        output_min,
        output_max,
        reinterpret_cast<uint8_t*>(output.template data_ptr<c10::quint8>()),
        caffe2::pthreadpool_());
  }

  // 使用 TORCH_INTERNAL_ASSERT 检查 QNNPACK 运行状态是否成功
  TORCH_INTERNAL_ASSERT(
      run_status == pytorch_qnnp_status_success,
      "failed to run quantized::conv2d (qnnpack) operator");

  // 返回计算后的输出 Tensor
  return output;
#ifdef USE_XNNPACK
// 检查是否可以使用 XNNPACK 库加速卷积操作
static bool can_use_xnnp(
    c10::ScalarType dtype,               // 数据类型
    int kSpatialDim,                     // 空间维度
    bool per_channel,                    // 是否按通道量化
    bool transpose) {                    // 是否转置操作
  // 如果 XNNPACK 库不可用，则返回 false
  if (!at::native::xnnpack::available()) {
    return false;
  }
  // 检查是否支持的数据类型是 c10::kQInt8
  bool supported_dtypes = dtype == c10::kQInt8;
  // 检查是否为不支持的配置
  bool invalid_config =
      (kSpatialDim != 2 /* No support for 3d convolution */
        || (dtype == c10::kQInt8 && transpose &&
            per_channel)); /* int8_t deconv does not support per-channel */
  // 如果数据类型支持且配置无效，则报错并返回 false
  if (supported_dtypes && invalid_config) {
    const std::string func_name =
        transpose ? "quantized::conv_transpose" : "quantized::conv";
    TORCH_CHECK(
        false,
        func_name,
        " (xnnpack): Unsupported conv config for dtype KQInt8");
  }
  // 返回是否支持的数据类型且配置有效
  return supported_dtypes && !invalid_config;
}
#endif  // USE_XNNPACK

// 应用卷积权重到输入张量的模板方法
template <int kSpatialDim>
at::Tensor PackedConvWeightsQnnp<kSpatialDim>::apply(
    const at::Tensor& input,             // 输入张量
    double output_scale,                 // 输出缩放因子
    int64_t output_zero_point) {         // 输出零点
#ifdef USE_XNNPACK
  // 如果可以使用 XNNPACK 加速，则调用对应实现函数
  if (can_use_xnnp(input.scalar_type(), kSpatialDim, per_channel(), transpose())) {
    return apply_impl_xnnp<c10::qint8, false>(
        input, output_scale, output_zero_point);
  } /* fall through for unsupported types, configs, or shapes */
#endif // USE_XNNPACK
  // 否则，调用默认实现函数
  return apply_impl<false>(input, output_scale, output_zero_point);
}

// 应用带 ReLU 激活函数的卷积权重到输入张量的模板方法
template <int kSpatialDim>
at::Tensor PackedConvWeightsQnnp<kSpatialDim>::apply_relu(
    const at::Tensor& input,             // 输入张量
    double output_scale,                 // 输出缩放因子
    int64_t output_zero_point) {         // 输出零点
#ifdef USE_XNNPACK
  // 如果可以使用 XNNPACK 加速，则调用对应实现函数
  if (can_use_xnnp(input.scalar_type(), kSpatialDim, per_channel(), transpose())) {
    return apply_impl_xnnp<c10::qint8, true>(
        input, output_scale, output_zero_point);
  } /* fall through for unsupported types, configs, or shapes */
#endif // USE_XNNPACK
  // 否则，调用默认实现函数
  return apply_impl<true>(input, output_scale, output_zero_point);
}

// 实例化模板方法，kSpatialDim = 2
template at::Tensor PackedConvWeightsQnnp<2>::apply(
    const at::Tensor& act,
    double output_scale,
    int64_t output_zero_point);

// 实例化模板方法，带 ReLU，kSpatialDim = 2
template at::Tensor PackedConvWeightsQnnp<2>::apply_relu(
    const at::Tensor& act,
    double output_scale,
    int64_t output_zero_point);

// 实例化模板方法，kSpatialDim = 3
template at::Tensor PackedConvWeightsQnnp<3>::apply(
    const at::Tensor& act,
    double output_scale,
    int64_t output_zero_point);

// 实例化模板方法，带 ReLU，kSpatialDim = 3
template at::Tensor PackedConvWeightsQnnp<3>::apply_relu(
    const at::Tensor& act,
    double output_scale,
    int64_t output_zero_point);

// 实例化默认实现模板方法，kSpatialDim = 2，不带 ReLU
template at::Tensor PackedConvWeightsQnnp<2>::apply_impl<false>(
    const at::Tensor& act,
    double output_scale,
    int64_t output_zero_point);

// 实例化默认实现模板方法，kSpatialDim = 3，不带 ReLU
template at::Tensor PackedConvWeightsQnnp<3>::apply_impl<false>(
  const at::Tensor& act,
  double output_scale,
  int64_t output_zero_point);

#endif // USE_PYTORCH_QNNPACK

#if AT_MKLDNN_ENABLED()
// 应用卷积权重到输入张量的模板方法，使用 MKL-DNN 库
template <int kSpatialDim>
at::Tensor PackedConvWeightsOnednn<kSpatialDim>::apply(
    const at::Tensor& input,             // 输入张量
    double output_scale,
    // 调用模板函数 apply_impl，传入参数 input 作为输入数据，c10::nullopt 表示没有指定附加参数，
    // output_scale 表示输出的缩放因子，output_zero_point 表示输出的零点偏移量，
    // 并且指定模板参数为 false，表示不使用 inplace 操作。
    return apply_impl<false>(input, c10::nullopt, output_scale, output_zero_point);
// 应用 ReLU 激活函数到输入张量并返回处理后的张量
template <int kSpatialDim>
at::Tensor PackedConvWeightsOnednn<kSpatialDim>::apply_relu(
    const at::Tensor& input,
    double output_scale,
    int64_t output_zero_point) {
  // 调用 apply_impl 函数，指定使用 ReLU 激活函数
  return apply_impl<true>(input, c10::nullopt, output_scale, output_zero_point);
}

// 将输入张量加到累加器张量上，并返回处理后的张量
template <int kSpatialDim>
at::Tensor PackedConvWeightsOnednn<kSpatialDim>::apply_add(
    const at::Tensor& input,
    const at::Tensor& accum,
    double output_scale,
    int64_t output_zero_point) {
  // 检查 kSpatialDim 是否为 2，只支持二维卷积加法
  TORCH_CHECK(kSpatialDim == 2, " Currently, only conv2d with add is supported.");
  // 调用 apply_impl 函数，不使用 ReLU 激活函数
  return apply_impl<false>(input, accum, output_scale, output_zero_point);
}

// 将输入张量加到累加器张量上，并应用 ReLU 激活函数，返回处理后的张量
template <int kSpatialDim>
at::Tensor PackedConvWeightsOnednn<kSpatialDim>::apply_add_relu(
    const at::Tensor& input,
    const at::Tensor& accum,
    double output_scale,
    int64_t output_zero_point) {
  // 检查 kSpatialDim 是否为 2，只支持二维卷积加法并应用 ReLU
  TORCH_CHECK(kSpatialDim == 2, " Currently, only conv2d add relu is supported.");
  // 调用 apply_impl 函数，使用 ReLU 激活函数
  return apply_impl<true>(input, accum, output_scale, output_zero_point);
}

// 实际执行卷积操作的函数模板
template <int kSpatialDim>
template <bool kReluFused>
at::Tensor PackedConvWeightsOnednn<kSpatialDim>::apply_impl(
    const at::Tensor& act,
    const std::optional<at::Tensor>& accum,
    double output_scale,
    int64_t output_zero_point) {
  // 构造函数名字的基础部分
  std::string func_name = "quantized::conv";
  // 如果是转置卷积，添加 "_transpose" 到函数名字
  if (transpose()) {
    func_name += "_transpose";
  }
  // 添加空间维度到函数名字后面，如 "2d" 或 "3d"
  func_name += std::to_string(kSpatialDim) + "d";

  // 是否有累加器作为额外输入，用于卷积加法融合
  bool has_accum = accum.has_value() ? true : false;
  if (has_accum) {
    auto& ctx = at::globalContext();
    func_name += "_add";
    // 如果是转置卷积，不支持转置卷积与加法融合
    TORCH_CHECK(
      !transpose(),
      "Didn't support transposed conv for conv with add ",
      c10::toString(ctx.qEngine()));
  }

  // 如果启用了 ReLU，添加 "_relu" 到函数名字
  if (kReluFused) {
    func_name += "_relu";
  }

  // 执行维度检查，确保输入张量和卷积参数正确
  ConvDimChecks<kSpatialDim>(
      act.ndimension(), stride().size(), padding().size(),
      output_padding().size(), dilation().size(), func_name, transpose());
  
  // 检查输入张量的数据类型必须为 QUint8
  TORCH_CHECK(act.scalar_type() == c10::ScalarType::QUInt8,
      func_name, " (ONEDNN): data type of input should be QUint8.");

  // 处理输入张量的内存布局，根据 kSpatialDim 不同选择不同的存储格式
  auto act_contig = act.contiguous(kSpatialDim == 2 ? c10::MemoryFormat::ChannelsLast : c10::MemoryFormat::ChannelsLast3d);
  auto src_dims = act_contig.sizes().vec();
  auto src_data_type = dnnl::memory::data_type::u8;
  // 根据 kSpatialDim 不同选择不同的格式标签
  auto src_desc = ideep::tensor::desc(src_dims, src_data_type,
      kSpatialDim == 2 ? ideep::format_tag::nhwc : ideep::format_tag::ndhwc);
  // 创建 ONEDNN 的张量对象，并指向输入张量的数据
  ideep::tensor src(src_desc, act_contig.data_ptr());

  // 获取权重和偏置张量的引用，并检查是否有偏置项
  ideep::tensor& weights = *(weight_.get());
  bool with_bias = bias_.has_value();
  const auto& kernel_size = weights.get_dims();

  // 获取输入张量的尺寸，并计算输出张量的尺寸
  const std::vector<int64_t>& input_size = src.get_dims();
  std::vector<int64_t> output_sizes;
  if (transpose()) {
    // 如果是转置卷积，预计算输出张量的尺寸
    // 预先打包的权重格式: [o, i, ...]
    const int N = act.size(0); // batch size
    const int C = act.size(1); // input channels
    const int M = weights.get_dim(0); // output channels
    // 计算输入的深度，根据输入的空间维度确定
    const int D = kSpatialDim == 2 ? 1 : act.size(2); // input depth
    // 计算输入的高度
    const int H = act.size(kSpatialDim); // input height
    // 计算输入的宽度
    const int W = act.size(kSpatialDim + 1); // input width
    // 获取卷积核的高度
    const int KH = weights.get_dim(kSpatialDim); // kernel height
    // 获取卷积核的宽度
    const int KW = weights.get_dim(kSpatialDim + 1); // kernel width
    // 获取卷积核的深度，根据输入的空间维度确定
    const int KD = kSpatialDim == 2 ? 1 : weights.get_dim(2); // kernel depth
    // 检查输入通道数是否符合预期，对应权重的形状是 [output, input, ...]
    TORCH_CHECK(C == groups() * weights.get_dim(1), // weight: [o, i, ...]
                func_name, " (ONEDNN): input channel number should be ",
                groups() * weights.get_dim(1), ", but got ", C);
    // 计算输出形状，根据不同的空间维度情况调用不同的函数
    auto output_shape = MakeDeConvOutputShape<kSpatialDim>(
        N,
        M,
        kSpatialDim == 2 ? std::vector<int64_t>{H, W} : std::vector<int64_t>{D, H, W},
        kSpatialDim == 2 ? std::vector<int64_t>{KH, KW} : std::vector<int64_t>{KD, KH, KW},
        stride(),
        padding(),
        output_padding(),
        dilation());
    // 将输出形状转换为标准容器
    output_sizes = c10::IntArrayRef(output_shape).vec();
  } else {
    // 如果未使用 ONEDNN，则使用 PyTorch 原生的卷积输出大小计算方法
    output_sizes = at::native::conv_output_size(input_size, kernel_size, padding().vec(), stride().vec(), dilation().vec());
  }
  // 将输出大小转换为 ONEDNN 的格式
  ideep::dims dst_dims = ideep::dims({output_sizes.cbegin(), output_sizes.cend()});
  // 创建空的量化张量作为输出
  at::Tensor output = at::_empty_affine_quantized(
      dst_dims,
      device(c10::kCPU)
          .dtype(c10::kQUInt8)
          .memory_format(kSpatialDim == 2 ?
              c10::MemoryFormat::ChannelsLast :
              c10::MemoryFormat::ChannelsLast3d),
      output_scale,
      output_zero_point,
      c10::nullopt);
  // 如果输出张量元素数为 0，则直接返回空的输出张量
  if (output.numel() == 0) {
    return output;
  }
  // 创建 ONEDNN 张量 dst
  ideep::tensor dst;
  // 创建累加的连续张量
  at::Tensor accum_contig;
  // 如果有累加张量
  if (has_accum) {
    // 创建 ONEDNN 张量描述符
    auto dst_desc = ideep::tensor::desc(dst_dims, src_data_type,
        kSpatialDim == 2 ? ideep::format_tag::nhwc : ideep::format_tag::ndhwc);
    // 获取累加张量的连续副本
    accum_contig = accum.value().contiguous(kSpatialDim == 2 ? c10::MemoryFormat::ChannelsLast : c10::MemoryFormat::ChannelsLast3d);
    // 检查累加张量的数据类型与输出张量是否相同
    TORCH_CHECK(accum_contig.dtype() == output.dtype(), "The output tensor should have same dtype as the accum tensor.");
    // 当与求和操作融合时，dst 张量将共享数据指针作为累加张量
    dst.init(dst_desc, accum_contig.data_ptr());
  } else {
    dst = ideep::tensor({dst_dims, ideep::tensor::data_type::u8, {output.strides().cbegin(), output.strides().cend()}},
                      output.data_ptr());
  }

  // Parameters
  // 获取步长参数
  const ideep::dims& strides = stride().vec();
  // 获取膨胀参数
  const ideep::dims& dilates = dilation().vec();
  // 获取左填充参数
  const ideep::dims& padding_l = padding().vec();
  // 获取右填充参数
  const ideep::dims& padding_r = padding().vec();
  // 获取输入的量化比例
  double input_scale = act.q_scale();
  // 获取输入的零点
  int64_t input_zp = act.q_zero_point();
  // ONEDNN 和 PyTorch 的量化比例是互倒的
  // 设置输入的比例尺度
  const ideep::scale_t& src_scales = ideep::scale_t(1, 1.0/input_scale);
  // 获取权重的比例尺度
  const ideep::scale_t& weights_scales = weights.get_scale();
  // 计算输出的比例尺度的倒数
  double inv_output_scale = 1.0/output_scale;
  // 设置输入的零点数组
  const ideep::zero_point_t src_zero_points = ideep::zero_point_t(1, input_zp);
  // 设置输出的零点数组
  const ideep::zero_point_t dst_zero_points = ideep::zero_point_t(1, output_zero_point);

  // 设置操作属性
  ideep::attr_t op_attr;
  // 如果有累加，则使用累加的量化参数
  float sum_scale = has_accum ? accum.value().q_scale() : 1.0;
  int32_t sum_zero_point = has_accum ? accum.value().q_zero_point() : 0;
  if (has_accum) {
    // 仅告诉我们有这些后处理操作，实际的值（如比例尺度和零点）稍后设置。
    op_attr = kReluFused ? ideep::attr_t::residual_with_sum_zero_point() : ideep::attr_t::fuse_sum();
    // 设置累加的比例尺度
    const ideep::scale_t accum_scale = ideep::scale_t(1, 1.0/sum_scale);
    // 设置累加的零点数组
    const ideep::zero_point_t accum_zero_points = ideep::zero_point_t(1, sum_zero_point);
    // 使用累加的值设置目标的比例尺度和零点
    dst.set_scale(accum_scale);
    dst.set_zero_point(accum_zero_points);
  } else if (kReluFused) {
    // 如果开启了融合ReLU，则设置操作属性为融合ReLU
    op_attr = ideep::attr_t::fuse_relu();
  }

  // 如果需要偏置并且偏置被外部修改过（如量化偏置校正），则更新预打包的偏置。
  if (with_bias && bias_.value().get_data_handle() != orig_bias_.value().data_ptr()) {
    // 初始化偏置以匹配原始偏置的描述和数据指针
    bias_.value().init(bias_.value().get_desc(), orig_bias_.value().data_ptr());
  }
  // 获取偏置的引用
  const auto& b = with_bias ? bias_.value() : ideep::tensor();
  // 获取当前线程数
  int num_threads = at::get_num_threads();
  // 如果需要转置操作
  if (transpose()) {
    // 当首次调用时初始化原语缓存，之后不再更新
    // 创建原语缓存的键
    PrimitiveCacheKey cache_key = std::make_tuple(
        input_scale, input_zp, src_dims, output_scale, output_zero_point, num_threads, sum_scale, sum_zero_point);
    // 使用 C++11 中的 call_once 函数保证以下代码块只被执行一次
    c10::call_once(*cache_initialized_flag, [&](){
        // 创建 DeconvParams 对象，用于存储反卷积操作的参数
        DeconvParams params;
        // 准备反卷积操作所需的参数
        ideep::convolution_transpose_forward::prepare(
            params, src, weights, b, dst_dims, dst,
            strides, padding_l, padding_r, dilates, groups(),
            src_scales, weights_scales, ideep::scale_t(1, inv_output_scale),
            src_zero_points, dst_zero_points, op_attr,
            dnnl::algorithm::deconvolution_direct,
            dnnl::prop_kind::forward_inference,
            ideep::u8s8, ideep::engine::cpu_engine());
        // 将当前反卷积操作的参数和 cache_key 缓存到 DeconvPrimitiveCache 中
        get_deconv_cache() = DeconvPrimitiveCache(cache_key, params);
        // 检查权重描述是否与预期相符，如果不符则重新排列权重
        auto expected_weight_desc = ideep::tensor::desc(params.pd.weights_desc(), groups());
        weights = weights.reorder_if_differ_in(expected_weight_desc);
    });
    // 如果缓存命中，则使用缓存中的参数执行反卷积计算
    if (get_deconv_cache().hit(cache_key)) {
        // 获取缓存中保存的反卷积参数
        DeconvParams& params = get_deconv_cache().get_params();
        // 执行反卷积操作的计算，传入参数并更新目标张量 dst
        ideep::convolution_transpose_forward::compute<false, false>(
            params, src, weights, b, dst);
    } else {
        // 如果缓存未命中，则重新执行反卷积操作的计算路径
        ideep::convolution_transpose_forward::compute(
            src, weights, b, dst_dims, dst,
            strides, padding_l, padding_r, dilates,
            groups(), src_scales, weights_scales,
            ideep::scale_t(1, inv_output_scale),
            src_zero_points, dst_zero_points, op_attr,
            dnnl::algorithm::deconvolution_direct,
            dnnl::prop_kind::forward_inference,
            ideep::u8s8, ideep::engine::cpu_engine());
    }
  } else {  // 如果未进行转置操作
    // 创建 ConvParams 对象，用于存储卷积操作的参数
    PrimitiveCacheKey cache_key = std::make_tuple(
        input_scale, input_zp, src_dims, output_scale, output_zero_point, num_threads, sum_scale, sum_zero_point);
    c10::call_once(*cache_initialized_flag, [&](){
        ConvParams params;
        // 准备卷积操作所需的参数
        ideep::convolution_forward::prepare(
            params, src, weights, b, dst_dims, dst,
            strides, dilates, padding_l, padding_r, groups(),
            src_scales, weights_scales, ideep::scale_t(1, inv_output_scale),
            src_zero_points, dst_zero_points,
            op_attr, dnnl::algorithm::convolution_direct,
            dnnl::prop_kind::forward_inference,
            ideep::u8s8, ideep::engine::cpu_engine());
        // 将当前卷积操作的参数和 cache_key 缓存到 ConvPrimitiveCache 中
        get_conv_cache() = ConvPrimitiveCache(cache_key, params);
        // 检查权重描述是否与预期相符，如果不符则重新排列权重
        auto expected_weight_desc = ideep::tensor::desc(params.pd.weights_desc(), groups());
        weights = weights.reorder_if_differ_in(expected_weight_desc);
    });
    // 如果缓存命中，则使用缓存中的参数执行卷积计算
    // 否则，重新执行卷积操作的计算路径
    if (get_conv_cache().hit(cache_key)) {
        auto& params = get_conv_cache().get_params();
        ideep::convolution_forward::compute<false, false>(params, src, weights, b, dst);
    } else {
      // 如果没有累加的情况，执行以下操作：
      // 使用 ideep 库进行卷积操作的前向推理计算，将计算结果存储在 dst 中。
      ideep::convolution_forward::compute(
          src, weights, b, dst_dims, dst,
          strides, dilates, padding_l, padding_r, groups(),
          src_scales, weights_scales, ideep::scale_t(1, inv_output_scale),
          src_zero_points, dst_zero_points, op_attr,
          dnnl::algorithm::convolution_direct,
          dnnl::prop_kind::forward_inference,
          ideep::u8s8, ideep::engine::cpu_engine());
    }
  }
  if (has_accum) {
    // 如果执行了累加操作：
    // 当与求和操作融合时，累加张量共享数据指针作为输出的 dst 张量。
    // 将输出的尺度（scale）和零点（zero point）重设为 accum_contig 中的值。
    set_quantizer_(accum_contig, at::make_per_tensor_affine_quantizer(
        output_scale, output_zero_point, accum_contig.scalar_type()));
    // 返回累加后的张量 accum_contig
    return accum_contig;
  } else {
    // 如果没有累加操作，直接返回输出张量 output。
    return output;
  }
}

template at::Tensor PackedConvWeightsOnednn<2>::apply(
    const at::Tensor& act,
    double output_scale,
    int64_t output_zero_point);

template at::Tensor PackedConvWeightsOnednn<2>::apply_relu(
    const at::Tensor& act,
    double output_scale,
    int64_t output_zero_point);

template at::Tensor PackedConvWeightsOnednn<3>::apply(
    const at::Tensor& act,
    double output_scale,
    int64_t output_zero_point);

template at::Tensor PackedConvWeightsOnednn<3>::apply_relu(
    const at::Tensor& act,
    double output_scale,
    int64_t output_zero_point);

static at::Tensor _quantized_convolution_onednn(
    at::Tensor act, // contains quantized values but not QTensor
    double act_scale,
    int64_t act_zero_point,
    at::Tensor weight, // MKLDNN tensor with quantized values
    at::Tensor weight_scales,
    at::Tensor weight_zero_points,
    std::optional<at::Tensor> bias, // Bias is not packed into MKLDNN tensor
    torch::List<int64_t> stride,
    torch::List<int64_t> padding,
    torch::List<int64_t> dilation,
    bool transposed,
    int64_t groups,
    double output_scale,
    int64_t output_zero_point,
    std::optional<at::Tensor> accum, // accum to fused with conv add
    double accum_scale,
    int64_t accum_zero_point,
    std::optional<c10::ScalarType> output_dtype,
    std::optional<c10::string_view> binary_attr,
    std::optional<at::Scalar> binary_alpha,
    std::optional<c10::string_view> unary_attr,
    torch::List<std::optional<at::Scalar>> unary_scalars,
    std::optional<c10::string_view> unary_algorithm) {
  /*********************************/
  /*          Checks               */
  /*********************************/
  // Due to constant folding inside Inductor freeze,
  // https://github.com/pytorch/pytorch/blob/b99d605a3070de35677cc43f0196c2f2e807b822/torch/ao/quantization/fx/_decomposed.py#L62-L63
  // The inversion of scale (inv_scale = 1.0 / scale) will be folded.
  // Thus, we can only obtain inv_scale from the quant node used as
  // the output_scale of this operation.

  // Check if the output type is fp32 or bf16
  bool fp32_output = output_dtype.has_value() && (output_dtype.value() == c10::kFloat);
  bool bfloat16_output = output_dtype.has_value() && (output_dtype.value() == c10::kBFloat16);
  
  // Ensure that for fp32 or bf16 output, the output_scale must be 1.0
  if (fp32_output || bfloat16_output) {
    // When the output type is fp32 or bf16, oneDNN expects op_attr not to set scales and zero points.
    // Therefore, default values of output_scale as 1.0 and output_zero_point as 0 are used,
    // because when output_scale is 1.0, invoking op_attr.set_scales in ideep is skipped;
    // when output_zero_point is 0, invoking op_attr.set_zero_points in ideep is skipped.
    TORCH_CHECK(output_scale == 1.0, " (ONEDNN): fp32 or bf16 output, output_scale must be 1.0.");


注释：
    TORCH_CHECK(output_zero_point == 0,  " (ONEDNN): fp32 or bf16 output, output_zero_point must be 0");
  }
  
  // 计算空间维度数
  int kSpatialDim = act.dim() - 2;
  // 判断是否为一维卷积
  bool is_1d = (1 == kSpatialDim);

  // 检查是否存在二元后操作
  bool has_binary_post_op = binary_attr.has_value() && binary_attr.value() != "none";
  // 检查是否存在一元后操作
  bool has_unary_post_op = unary_attr.has_value() && unary_attr.value() != "none";
  // 检查是否有累积后操作和是否是求和
  // has_accum_postop_sum: 除了卷积之外的额外输入，用于卷积后操作求和融合。
  bool has_accum_postop_sum = has_binary_post_op && binary_attr.value() == "sum";

  if (has_accum_postop_sum) {
    // 对于后操作求和，累积张量不应为空
    TORCH_CHECK(accum.has_value(), "For post op sum, accum tensor should not be empty.");
    // 确保累积张量是连续的
    TORCH_CHECK(
      accum.value().is_contiguous(
        kSpatialDim == 2
        ? c10::MemoryFormat::ChannelsLast
        : c10::MemoryFormat::ChannelsLast3d
      ),
      "For post op sum, accum tensor must be contiguous."
    );
    // 如果输出是fp32或bfloat16，则检查累积的缩放因子和零点
    if (fp32_output || bfloat16_output) {
      TORCH_CHECK(accum_scale == 1.0,  " (ONEDNN): fp32 or bf16 output, accum_scale must be 1.0.");
      TORCH_CHECK(accum_zero_point == 0,  " (ONEDNN): fp32 or bf16 output, accum_zero_point must be 0");
      // 确保累积张量的标量类型是float或bfloat16
      TORCH_CHECK((accum.value().scalar_type() == c10::kFloat) || (accum.value().scalar_type() == c10::kBFloat16), "The accum tensor should be KFloat or KBFloat.");
    }
  }

  // 设定函数名称为quantized::packed_weights_conv，并根据空间维度附加后操作信息
  std::string func_name = "quantized::packed_weights_conv";
  func_name += std::to_string(kSpatialDim) + "d";
  if (has_binary_post_op) {
    func_name += binary_attr.value().data();
  }
  if (has_unary_post_op) {
    func_name += unary_attr.value().data();
  }

  // 如果是一维卷积，则将空间维度增加到2
  if (kSpatialDim == 1) {
    kSpatialDim += 1;
  }
  // 检查权重张量是否为MKLDNN格式
  TORCH_CHECK(
    weight.is_mkldnn(),
    func_name, ": Weight should be prepacked as an MKLDNN tensor"
  );
  // 如果是转置卷积，则报错，暂不支持
  if (transposed) {
    TORCH_CHECK(
      false,
      func_name, ": to support transposed convolution."
    );
  }
  // 如果是一维卷积，调整输入张量的形状
  if (is_1d) {
    // N, C, L -> N, C, 1, L
    act = act.unsqueeze(quant_utils::kConv1dSqueezeDim + 2);
    stride = quant_utils::MakeArgForConv1d(stride, 1);
    padding = quant_utils::MakeArgForConv1d(padding, 0);
    dilation = quant_utils::MakeArgForConv1d(dilation, 1);
  }
  // 检查输入张量的数据类型是否为uint8（无符号字符）
  TORCH_CHECK(
    act.scalar_type() == c10::ScalarType::Byte,
    func_name, ": Input tensor should have uint8 (unsigned char) data type");
  // 检查权重张量的数据类型是否为int8（字符）
  TORCH_CHECK(
    weight.scalar_type() == c10::ScalarType::Char,
    func_name, ": Weight tensor should have int8 (char) data type");
  // 检查权重张量的维度是否正确
  TORCH_CHECK(
    weight.ndimension() == kSpatialDim + 2,
    func_name, ": Weights are expected to have ", kSpatialDim + 2, " dimensions");
  // 检查步长参数的大小是否正确
  TORCH_CHECK(
    stride.size() == (decltype(stride.size()))kSpatialDim,
    func_name, ": stride should contain ", kSpatialDim, " elements for ",
    kSpatialDim, "D convolution.");
  // 检查填充参数的大小是否正确
  TORCH_CHECK(
    padding.size() == (decltype(padding.size()))kSpatialDim,
    func_name, ": Specify front/top/left padding only. "
    "end/bottom/right padding assumed to be equal to front/top/left");
    dilation.size() == (decltype(dilation.size()))kSpatialDim,
    func_name, ": dilation should contain ", kSpatialDim, " elements for ",
    kSpatialDim, "D convolution.");

  // Parameters



    // 检查 dilation 的尺寸是否等于 kSpatialDim 的类型所表示的大小
    dilation.size() == (decltype(dilation.size()))kSpatialDim,
    // 输出函数名和错误信息，指出 dilation 应包含 kSpatialDim 个元素，用于 kSpatialDim 维度卷积
    func_name, ": dilation should contain ", kSpatialDim, " elements for ",
    kSpatialDim, "D convolution.");

  // Parameters
    // 参数说明部分
#if IDEEP_PREREQ(3, 1, 0, 1)
  // 1. 如果由观察器生成的权重比例尺应该具有 float32 数据类型
  //    https://github.com/pytorch/pytorch/blob/d2c24eca8a60c56b31ca967a44d5cc4522802aa6/torch/ao/quantization/observer.py#L323
  // 2. 如果从量化张量获取的权重比例尺，如在 UT 中所做的那样，其数据类型为 double
  //    https://github.com/pytorch/pytorch/blob/d2fa3f608b5e4f582a8aaf752f10efe4ca72a7d0/aten/src/ATen/quantized/Quantizer.cpp#L69
  TORCH_CHECK(
    weight_scales.scalar_type() == c10::ScalarType::Double || weight_scales.scalar_type() == c10::ScalarType::Float,
    "weight_scales should be with data type Double or float");
  if (weight_scales.scalar_type() == c10::ScalarType::Double) {
    // 对于情况 2，我们将其从 double 转换为 float，因为 ideep::scale_t 是 std::vector<float> 的别名
    weight_scales = weight_scales.to(c10::ScalarType::Float);
  }
  TORCH_CHECK(
    weight_scales.ndimension() == 0 ||
    (weight_scales.strides().size() == 1 || weight_scales.stride(0) == 1),
    "weight_scales should be scalar tensor or contiguous 1D tensor.");
  // 创建一个 weights_scales 对象，使用 weight_scales 的数据指针初始化，范围是 [data_ptr, data_ptr + numel]
  ideep::scale_t weights_scales(weight_scales.data_ptr<float>(), weight_scales.data_ptr<float>() + weight_scales.numel());
#elif IDEEP_PREREQ(3, 1, 0, 0)
  // TODO (leslie): 在这里优化性能：
  // 1. 移除权重比例尺的倒数，我们已经在 Ideep 中完成了权重比例尺的倒数：
  //    https://github.com/intel/ideep/blob/3c90e365526e19c110371d23831678a7e9d4353d/include/ideep/operators/conv.hpp#L163-L168
  // 2. 移除两次权重比例尺的内存复制：
  //   2.1 weights_scales 的输入是 PyTorch 的 Dense 张量，我们将其转换为 vector<float>
  //   2.2 OneDNN 流提交将 weights_scales 从 vector 转换为 ideep::tensor
  //   https://github.com/intel/ideep/blob/3c90e365526e19c110371d23831678a7e9d4353d/include/ideep/operators/conv.hpp#L1855-L1860
  // 我们应该能够直接将 weights_scales 从 PyTorch Dense Tensor 转换为能够共享相同数据指针的 IDeep Tensor。
  ideep::scale_t weights_scales(weight_scales.numel());
  if (weight_scales.ndimension() == 0) {
    // 权重是每个张量量化的情况，然后 weight_scales 将是一个标量 Tensor
    weights_scales[0] = 1.0 / weight_scales.item().toDouble(); // ONEDNN 和 PyTorch 的比例尺是倒数关系
  } else {
    // 权重是每个通道量化的情况
    for (int i = 0; i < weight_scales.numel(); ++i) {
      weights_scales[i] = 1.0 / weight_scales[i].item().toDouble();
    }
  }
#else
  TORCH_CHECK(false, "Unexpected IDeep version to do qconv calculation.");
#endif

const ideep::zero_point_t src_zero_points = ideep::zero_point_t(1, act_zero_point);
const ideep::zero_point_t dst_zero_points = ideep::zero_point_t(1, output_zero_point);

// 将 weight 转换为 IDeep 的 itensor
auto packed_weight = at::native::itensor_from_mkldnn(weight);

// Bias
ideep::tensor onednn_bias;
const int output_channels = weight.size(0);
bool with_bias = bias.has_value();

at::Tensor bias_val_float;
if (with_bias) {
    // 对于 int8-mixed-bf16，我们还将使用 float32 类型的偏置
    bias_val_float = bias.value().to(at::kFloat);
    // 检查偏置是否是一个一维向量（1D Tensor）
    TORCH_CHECK(bias_val_float.dim() == 1, "bias should be a vector (1D Tensor)");
    // 检查偏置向量是否有 K 个元素，其中 K 等于输出通道数
    TORCH_CHECK(
        bias_val_float.size(0) == output_channels,
        "bias should have K elements: " + std::to_string(output_channels));
    // 创建 bias 的 IDEEP 张量描述符，使用 float32 数据类型
    auto bias_desc = ideep::tensor::desc(bias_val_float.sizes().vec(), dnnl::memory::data_type::f32);
    // 初始化 IDEEP 的偏置张量，使用 bias_val_float 的数据指针
    onednn_bias.init(bias_desc, bias_val_float.data_ptr());
  }

  // 如果需要偏置，则使用 onednn_bias，否则使用空的 IDEEP 张量
  const auto& expected_bias = with_bias ? onednn_bias : ideep::tensor();

  /*********************************/
  /*        Computation            */
  /*********************************/
  // src
  // 使 act 张量在内存中是连续的，并且根据 kSpatialDim 的值选择内存格式
  auto act_contig = act.contiguous(kSpatialDim == 2 ?
                                   c10::MemoryFormat::ChannelsLast :
                                   c10::MemoryFormat::ChannelsLast3d);
  // 获取 act_contig 的维度大小
  auto src_dims = act_contig.sizes().vec();
  // 定义 src 张量的数据类型为 uint8
  auto src_data_type = dnnl::memory::data_type::u8;
  // 创建 src 张量的 IDEEP 描述符，根据 kSpatialDim 的值选择格式标签
  auto src_desc = ideep::tensor::desc(src_dims, src_data_type,
      kSpatialDim == 2 ? ideep::format_tag::nhwc : ideep::format_tag::ndhwc);
  // 初始化 IDEEP 的 src 张量，使用 act_contig 的数据指针
  ideep::tensor src;
  src.init(src_desc, act_contig.data_ptr());
  // dst
  // 获取输入大小作为 input_size
  const std::vector<int64_t>& input_size = src.get_dims();
  // 获取 packed_weight 的大小作为 kernel_size
  const auto& kernel_size = packed_weight.get_dims();
  // 计算输出大小
  std::vector<int64_t> output_sizes;
  output_sizes = at::native::conv_output_size(input_size, kernel_size, padding.vec(), stride.vec(), dilation.vec());
  // 将输出大小转换为 IDEEP 的 dims 类型
  ideep::dims dst_dims = ideep::dims({output_sizes.cbegin(), output_sizes.cend()});
  // 如果 has_accum_postop_sum 为真，则使用 accum.value() 作为输出；否则根据数据类型和内存格式创建空的输出张量
  at::Tensor output = has_accum_postop_sum ?
    accum.value() :
    at::empty(
      dst_dims,
      device(c10::kCPU)
          .dtype(fp32_output ? c10::kFloat : (bfloat16_output ? c10::kBFloat16 : c10::kByte))
          .memory_format(kSpatialDim == 2 ?
              c10::MemoryFormat::ChannelsLast :
              c10::MemoryFormat::ChannelsLast3d)
    );
  // 如果输出张量的元素数为 0，则直接返回空的输出张量
  if (output.numel() == 0) {
    return output;
  }
  // 将稠密张量 output 转换为 IDEEP 的 dst 张量
  ideep::tensor dst = at::native::itensor_view_from_dense(output);
  // 定义一个静态的 dummy_accum_desc 作为 IDEEP 张量的描述符
  static ideep::tensor::desc dummy_accum_desc;
  // 创建 IDEEP 操作的属性，通过给定的后操作参数
  ideep::attr_t op_attr = onednn_utils::create_attr_by_post_op(
    binary_attr.has_value() ? binary_attr.value() : "none",
    binary_alpha.has_value() ? binary_alpha.value().to<double>() : 1.0,
    accum_scale,
    accum_zero_point,
    dummy_accum_desc,
    unary_attr.has_value() ? unary_attr.value() : "none",
    unary_scalars,
    unary_algorithm.has_value() ? unary_algorithm.value() : ""
  );
#if IDEEP_PREREQ(3, 1, 0, 0)
  // 使用 oneDNN 的 API 替代 ideep 中的 prepare/compute 函数，以减少集成开销。
  // ideep 中的函数较重，因为其具有复杂的数据结构，统一 API。
  // 要求 oneDNN 版本 >= 3.1.0。
  使用 ideep 命名空间中的 tensor 类
  auto weight_grouped = packed_weight.make_grouped_weights(groups, /* is_deconv */false);
  创建 tensor 对象 weights_desc，描述 weight_grouped 的维度、数据类型为 s8，格式标签为 any。
  如果 groups 大于 1，则将 weights_desc 转换为分组格式。
  获取目标 tensor dst 的描述信息。
  根据是否有 bias 决定创建 bias_desc 的 tensor::desc，数据类型为 f32，格式标签为 any，或者创建一个空的描述。
  如果 act_scale 不等于 1.0f，则设置 op_attr 的源缩放因子掩码。
  如果 act_zero_point 不等于 0，则设置 op_attr 的源零点掩码。
  计算每个分组中的输出通道数 oc_per_group，为 weight_grouped 的第一个维度除以 groups。
  计算 weight_scales 的缩放因子掩码 wei_scale_mask。
  设置 op_attr 的权重缩放因子掩码。
  如果 output_scale 不等于 1.0f，则设置 op_attr 的目标缩放因子掩码。
  如果 output_zero_point 不等于 0，则设置 op_attr 的目标零点掩码。
  设置 op_attr 的 scratchpad 模式为用户模式。
  创建 CPU 引擎 engine。
  获取与 dilation.vec() 兼容的 dilates_dnnl。
  根据是否有 bias 决定创建 convolution_forward 的原始描述 primitive_desc：
      - 如果有 bias，使用包括 bias_desc 的描述；
      - 如果没有 bias，只使用 src_desc、weights_desc、dst_desc 等描述。
  创建 convolution_forward 的原语 primitive。
  
  // 如果需要，重新排序 weight_grouped 以匹配 primitive_desc 中的权重描述。
  创建 tensor scratchpad，用于存储 scratchpad 的内容。
  创建 ideep::exec_args args，准备执行卷积的参数。
  插入参数 DNNL_ARG_SRC、DNNL_ARG_WEIGHTS、DNNL_ARG_DST、DNNL_ARG_SCRATCHPAD 到 args 中。
  如果有 bias，插入参数 DNNL_ARG_BIAS 到 args 中。
  创建 tensor src_scales_t，表示输入数据的缩放因子。
  创建 tensor wei_scales_t，表示权重的缩放因子。
  创建 tensor dst_scales_t，表示输出数据的缩放因子。
  创建 tensor src_zp_t，表示输入数据的零点。
  创建 tensor dst_zp_t，表示输出数据的零点。
  如果 act_scale 不等于 1.0f，
    # 将src_scales_t插入到args中，对应的键为DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC
    args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, src_scales_t});
  }
  # 如果输出缩放因子output_scale不为1.0，则将dst_scales_t插入到args中，键为DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST
  if (output_scale != 1.0f) {
    args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, dst_scales_t});
  }
  # 将wei_scales_t插入到args中，键为DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS
  args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, wei_scales_t});
  # 如果激活零点act_zero_point不为0，则将src_zp_t插入到args中，键为DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC
  if (act_zero_point != 0) {
    args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC, src_zp_t});
  }
  # 如果输出零点output_zero_point不为0，则将dst_zp_t插入到args中，键为DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST
  if (output_zero_point != 0) {
    args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST, dst_zp_t});
  }
  # 执行primitive对象的execute方法，在默认流ideep::stream::default_stream()上执行，传递参数args
  primitive.execute(ideep::stream::default_stream(), args);
/*
 * FBGEMM uses vpmaddubsw instruction to multiply activations (uint8_t) and
 * weights (int8_t).
 *
 * https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maddubs_epi16&expand=3284,3530
 *
 * vpmaddubsw operates on a vector of activations and a vector of
 * weights. If these vectors are
 *
 *    A (uint8_t) = a0, a1, a2, a3 ...
 *
 * and
 *
 *    B (int8_t)  = b0, b1, b2, b3 ...
 *
 * the result of this instruction is an int16_t vector with values
 *
 *    C (int16_t) = a0*b0 + a1*b1, a2*b2 + a3*b3 ...
 *
 * For large values of A and/or B the result (a0*b0 + a1*b1) might not fit into
 * an int16_t number. So the instruction saturates them to max (or min) possible
 * value of an int16_t number. Such behavior is expected for the
 * implementation below.
 *
 * For example, a0 = 255, a1 = 255, b0 = 127 and b1 = 127 the actual result
 * 64770 overflows for an int16_t number (-32768, 32767) so the returned result
 * is 32767.
 *
 */
template <int kSpatialDim, bool kReluFused>
class QConvInt8 final {
 public:
  // 定义 QConvInt8 类，用于执行量化整数卷积操作
  static Tensor run(
      Tensor act,  // 输入的激活张量
      const c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>>& packed_weight,  // 压缩的卷积参数
      double output_scale,  // 输出的量化比例因子
      int64_t output_zero_point) {  // 输出的量化零点值
    // 如果 kReluFused 为真，则调用 packed_weight 对象的 apply_relu 方法，
    // 并返回其结果，这里使用了激活函数 act，输出缩放因子 output_scale，
    // 输出零点 output_zero_point
    if (kReluFused) {
      return packed_weight->apply_relu(act, output_scale, output_zero_point);
    } else {
      // 如果 kReluFused 不为真，则调用 packed_weight 对象的 apply 方法，
      // 并返回其结果，这里使用了激活函数 act，输出缩放因子 output_scale，
      // 输出零点 output_zero_point
      return packed_weight->apply(act, output_scale, output_zero_point);
    }
  }
};

// 模板类定义：整数量化卷积和加法类，支持指定空间维度和是否融合ReLU
template <int kSpatialDim, bool kReluFused>
class QConvAddInt8 final {
 public:
  // 静态方法：执行整数量化卷积加法操作
  static Tensor run(
      Tensor act,  // 激活张量
      Tensor accum,  // 累积张量
      const c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>>& packed_weight,  // 打包的卷积参数
      double output_scale,  // 输出缩放因子
      int64_t output_zero_point) {  // 输出零点
    auto& ctx = at::globalContext();
    
    // 如果使用MKLDNN引擎，执行相应的操作
#if AT_MKLDNN_ENABLED()
    if (ctx.qEngine() == at::QEngine::ONEDNN) {
      if (kReluFused) {
        // 如果融合了ReLU，调用ONEDNN上的带ReLU的卷积加法操作
        return dynamic_cast<PackedConvWeightsOnednn<kSpatialDim>*>(packed_weight.get())->apply_add_relu(
          act, accum, output_scale, output_zero_point);
      } else {
        // 否则调用ONEDNN上的卷积加法操作
        return dynamic_cast<PackedConvWeightsOnednn<kSpatialDim>*>(packed_weight.get())->apply_add(
          act, accum, output_scale, output_zero_point);
      }
    }
#endif
    
    // 如果不使用MKLDNN引擎，则报错并指出未找到相应引擎
    TORCH_CHECK(
    false,
    "Didn't find engine for operation quantized::conv2d_add.",
    toString(ctx.qEngine()));
  }
};

// 模板类定义：一维整数量化卷积类，支持是否融合ReLU
template <bool kReluFused>
class QConv1dInt8 final {
 public:
  // 静态方法：执行一维整数量化卷积操作
  static Tensor run(
      Tensor act,  // 激活张量
      const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,  // 打包的卷积参数
      double output_scale,  // 输出缩放因子
      int64_t output_zero_point) {  // 输出零点
    at::Tensor output;
    
    // 将输入张量从 N, C, L 维度扩展到 N, C, 1, L 维度
    act = act.unsqueeze(quant_utils::kConv1dSqueezeDim + 2);
    
    if (kReluFused) {
      // 如果融合了ReLU，调用打包的卷积参数的带ReLU的卷积操作
      output = packed_weight->apply_relu(act, output_scale, output_zero_point);
    } else {
      // 否则调用打包的卷积参数的卷积操作
      output = packed_weight->apply(act, output_scale, output_zero_point);
    }
    
    // 将输出张量从 N, C, 1, L 维度压缩回到 N, C, L 维度
    return output.squeeze_(quant_utils::kConv1dSqueezeDim + 2);
  }
};

// 用于保持向后兼容性的模板类定义：整数量化卷积类，支持指定空间维度和是否融合ReLU
template <int kSpatialDim, bool kReluFused>
class QConvInt8ForBC final {
 public:
  // 静态方法：执行整数量化卷积操作（保持向后兼容性）
  static Tensor run(
      Tensor act,  // 激活张量
      const c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>>& packed_weight,  // 打包的卷积参数
      torch::List<int64_t> /*stride*/,  // 步长（已移除）
      torch::List<int64_t> /*padding*/,  // 填充（已移除）
      torch::List<int64_t> /*dilation*/,  // 膨胀（已移除）
      int64_t /*groups*/,  // 分组（已移除）
      double output_scale,  // 输出缩放因子
      int64_t output_zero_point) {  // 输出零点
    if (kReluFused) {
      // 如果融合了ReLU，发出警告并调用带ReLU的卷积操作
      TORCH_WARN_ONCE(
          "Arguments [stride, padding, dilation, groups] in ops.quantized.conv" +
              std::to_string(kSpatialDim),
          "d_relu, have been removed, please update your model to remove these arguments.");
      return packed_weight->apply_relu(act, output_scale, output_zero_point);
    } else {
      // 否则发出警告并调用卷积操作
      TORCH_WARN_ONCE(
          "Arguments [stride, padding, dilation, groups] in ops.quantized.conv",
          std::to_string(kSpatialDim),
          "d, have been removed, please update your model to remove these arguments.");
      return packed_weight->apply(act, output_scale, output_zero_point);
    }
  }
};
class QConvoneDNN final {
 public:
  static at::Tensor run_pointwise(
      at::Tensor act, // 输入的激活张量，包含量化值但不是量化张量
      double act_scale, // 激活张量的量化比例因子
      int64_t act_zero_point, // 激活张量的零点
      at::Tensor weight, // 输入的权重张量，包含量化值但不是量化张量
      at::Tensor weight_scales, // 权重张量的量化比例因子
      at::Tensor weight_zero_points, // 权重张量的零点
      std::optional<at::Tensor> bias, // 可选的偏置张量
      torch::List<int64_t> stride, // 卷积的步长
      torch::List<int64_t> padding, // 卷积的填充
      torch::List<int64_t> dilation, // 卷积的扩张
      int64_t groups, // 卷积的分组数
      double output_scale, // 输出张量的量化比例因子
      int64_t output_zero_point, // 输出张量的零点
      std::optional<c10::ScalarType> output_dtype, // 可选的输出数据类型
      c10::string_view attr, // 单目操作的属性名称
      torch::List<std::optional<at::Scalar>> scalars, // 单目操作的标量参数列表
      std::optional<c10::string_view> algorithm) { // 可选的算法名称
#if AT_MKLDNN_ENABLED()
    if (act.dim() == 3 || act.dim() == 5) {
      // 对于 Conv1D/3D 后处理检查
      TORCH_CHECK(
        attr == "none",
        "quantized pointwise conv",
        act.dim()-2,
        "d doesn't support unary_post_op fusion. Got unary_post_op: ",
        attr,
        ".")
    } else {
      // 对于 Conv2D 后处理检查
      TORCH_CHECK(
        attr == "none" || attr == "relu" || attr == "hardtanh" || attr == "hardswish" || attr == "swish",
        "none post_op or post_op relu/hardtanh/hardswish is supported for quantized pointwise conv2d. Got unary_post_op: ",
        attr,
        ".")
    }
    // 调用基于 OneDNN 的量化卷积运算
    return _quantized_convolution_onednn(
        act, act_scale, act_zero_point,
        weight, weight_scales, weight_zero_points,
        bias, stride, padding, dilation, /*transposed*/false,
        groups, output_scale, output_zero_point,
        /*accum*/c10::nullopt, /*accum_scale*/0.0, /*accum_zero_point*/0,
        /*output_dtype*/output_dtype, /*binary_attr*/c10::nullopt, /*binary_alpha*/c10::nullopt,
        /*unary_attr*/attr, /*unary_scalars*/scalars, /*unary_algorithm*/algorithm
    );
#else
    // 如果未启用 OneDNN，抛出未实现异常
    TORCH_CHECK(false, "Unimplemented as onednn is not available.")
#endif
  }
  static at::Tensor run_pointwise_binary(
      at::Tensor act, // 输入的激活张量，包含量化值但不是量化张量
      double act_scale, // 激活张量的量化比例因子
      int64_t act_zero_point, // 激活张量的零点
      at::Tensor accum, // 输入的累积张量，包含量化值但不是量化张量
      double accum_scale, // 累积张量的量化比例因子
      int64_t accum_zero_point, // 累积张量的零点
      at::Tensor weight, // 输入的权重张量，包含量化值但不是量化张量
      at::Tensor weight_scales, // 权重张量的量化比例因子
      at::Tensor weight_zero_points, // 权重张量的零点
      std::optional<at::Tensor> bias, // 可选的偏置张量
      torch::List<int64_t> stride, // 卷积的步长
      torch::List<int64_t> padding, // 卷积的填充
      torch::List<int64_t> dilation, // 卷积的扩张
      int64_t groups, // 卷积的分组数
      double output_scale, // 输出张量的量化比例因子
      int64_t output_zero_point, // 输出张量的零点
      std::optional<c10::ScalarType> output_dtype, // 可选的输出数据类型
      c10::string_view binary_attr, // 二元操作的属性名称
      std::optional<at::Scalar> alpha, // 可选的 alpha 参数
      std::optional<c10::string_view> unary_attr, // 单目操作的属性名称
      torch::List<std::optional<at::Scalar>> unary_scalars, // 单目操作的标量参数列表
      std::optional<c10::string_view> unary_algorithm) { // 可选的算法名称
#if AT_MKLDNN_ENABLED()
    // 对于 Conv2D 后处理检查

    // 对于 Conv2D 后处理检查
    TORCH_CHECK(
        unary_attr == "none" || unary_attr == "relu" || unary_attr == "hardtanh" || unary_attr == "hardswish" || unary_attr == "swish",
        "none post_op or post_op relu/hardtanh/hardswish is supported for quantized pointwise conv2d. Got unary_post_op: ",
        unary_attr,
        "."
    );
    // 调用基于 OneDNN 的二元量化卷积运算
    return _quantized_binary_convolution_onednn(
        act, act_scale, act_zero_point,
        accum, accum_scale, accum_zero_point,
        weight, weight_scales, weight_zero_points,
        bias, stride, padding, dilation, /*transposed*/false,
        groups, output_scale, output_zero_point,
        /*output_dtype*/output_dtype, binary_attr, alpha,
        unary_attr, unary_scalars, unary_algorithm
    );
#else
    // 如果未启用 OneDNN，抛出未实现异常
    TORCH_CHECK(false, "Unimplemented as onednn is not available.")
#endif
  }
};
    # 使用 TORCH_CHECK 宏来验证条件，确保以下条件成立：
    # - act 的维度为4
    # - binary_attr 为 "sum"
    # - 如果 unary_attr 存在，则其值为 "none" 或者 "relu"
    # 如果条件不成立，将会返回带有相应错误信息的消息字符串。
    TORCH_CHECK(
      act.dim() == 4 && binary_attr == "sum" && (
        !unary_attr.has_value() ||
        (unary_attr.has_value() &&
          (
            unary_attr.value() == "none" || unary_attr.value() == "relu"
          )
        )
      ),
      "post_op sum or post_op sum_relu is supported for quantized pointwise conv2d. Got binary_post_op: ",
      binary_attr,
      " unary_post_op: ",
      unary_attr.has_value() ? unary_attr.value() : "none",
      ".")
    # 如果验证通过，执行 _quantized_convolution_onednn 函数，进行量化的卷积操作。
    return _quantized_convolution_onednn(
        act, act_scale, act_zero_point,
        weight, weight_scales, weight_zero_points,
        bias, stride, padding, dilation, /*transposed*/false,
        groups, output_scale, output_zero_point,
        accum, accum_scale, accum_zero_point,
        /*output_dtype*/output_dtype, binary_attr, alpha,
        unary_attr, unary_scalars, unary_algorithm
    );
    
    
    这段代码通过 TORCH_CHECK 宏来验证输入条件是否满足，如果满足则执行 `_quantized_convolution_onednn` 函数进行量化的卷积操作，否则抛出错误消息。
#else
    TORCH_CHECK(false, "Unimplemented as onednn is not available.")
#endif
  }
};

// 定义 TORCH_LIBRARY_IMPL 宏，实现 quantized 库的 CPU 版本
TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
  // 实现 quantized::conv1d 操作，使用 QConv1dInt8<false>::run 函数
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv1d"),          QConv1dInt8<false>::run);
  // 实现 quantized::conv1d_relu 操作，使用 QConv1dInt8<true>::run 函数
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv1d_relu"),     QConv1dInt8<true>::run);
  // 实现 quantized::conv2d.new 操作，使用 QConvInt8<2, false>::run 函数
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv2d.new"),      QConvInt8<2, false>::run);
  // 实现 quantized::conv2d_relu.new 操作，使用 QConvInt8<2, true>::run 函数
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv2d_relu.new"), QConvInt8<2, true>::run);
  // 实现 quantized::conv2d_add 操作，使用 QConvAddInt8<2, false>::run 函数
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv2d_add"),      QConvAddInt8<2, false>::run);
  // 实现 quantized::conv2d_add_relu 操作，使用 QConvAddInt8<2, true>::run 函数
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv2d_add_relu"), QConvAddInt8<2, true>::run);
  // 实现 quantized::conv3d.new 操作，使用 QConvInt8<3, false>::run 函数
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv3d.new"),      QConvInt8<3, false>::run);
  // 实现 quantized::conv3d_relu.new 操作，使用 QConvInt8<3, true>::run 函数
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv3d_relu.new"), QConvInt8<3, true>::run);

  // 为了向后兼容性
  // 实现 quantized::conv2d 操作，使用 QConvInt8ForBC<2, false>::run 函数
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv2d"), QConvInt8ForBC<2, false>::run);
  // 实现 quantized::conv2d_relu 操作，使用 QConvInt8ForBC<2, true>::run 函数
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv2d_relu"), QConvInt8ForBC<2, true>::run);
  // 实现 quantized::conv3d 操作，使用 QConvInt8ForBC<3, false>::run 函数
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv3d"), QConvInt8ForBC<3, false>::run);
  // 实现 quantized::conv3d_relu 操作，使用 QConvInt8ForBC<3, true>::run 函数
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv3d_relu"), QConvInt8ForBC<3, true>::run);

  // transpose 操作
  // 实现 quantized::conv_transpose1d 操作，使用 QConv1dInt8<false>::run 函数
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose1d"),  QConv1dInt8<false>::run);
  // 实现 quantized::conv_transpose2d 操作，使用 QConvInt8<2, false>::run 函数
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose2d"),  QConvInt8<2, false>::run);
  // 实现 quantized::conv_transpose3d 操作，使用 QConvInt8<3, false>::run 函数
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::conv_transpose3d"),
      QConvInt8<3, false>::run);
}

// 定义 TORCH_LIBRARY_IMPL 宏，实现 _quantized 库的 CPU 版本
TORCH_LIBRARY_IMPL(_quantized, QuantizedCPU, m) {
  // 实现 _quantized::conv2d 操作，使用 QConvInt8<2, false>::run 函数
  m.impl(TORCH_SELECTIVE_NAME("_quantized::conv2d"),      QConvInt8<2, false>::run);
  // 实现 _quantized::conv2d_relu 操作，使用 QConvInt8<2, true>::run 函数
  m.impl(TORCH_SELECTIVE_NAME("_quantized::conv2d_relu"), QConvInt8<2, true>::run);

  // transpose 操作
  // 实现 _quantized::conv_transpose1d 操作，使用 QConv1dInt8<false>::run 函数
  m.impl(TORCH_SELECTIVE_NAME("_quantized::conv_transpose1d"),  QConv1dInt8<false>::run);
  // 实现 _quantized::conv_transpose2d 操作，使用 QConvInt8<2, false>::run 函数
  m.impl(TORCH_SELECTIVE_NAME("_quantized::conv_transpose2d"),  QConvInt8<2, false>::run);
}

// 定义 TORCH_LIBRARY_IMPL 宏，实现 onednn 库的 MkldnnCPU 版本
TORCH_LIBRARY_IMPL(onednn, MkldnnCPU, m) {
  // 实现 onednn::qconv1d_pointwise 操作，使用 QConvoneDNN::run_pointwise 函数
  // Conv1D/2D/3D with unary postop
  m.impl(TORCH_SELECTIVE_NAME("onednn::qconv1d_pointwise"), QConvoneDNN::run_pointwise);
  m.impl(TORCH_SELECTIVE_NAME("onednn::qconv2d_pointwise"), QConvoneDNN::run_pointwise);
  m.impl(TORCH_SELECTIVE_NAME("onednn::qconv3d_pointwise"), QConvoneDNN::run_pointwise);

  // 实现 onednn::qconv2d_pointwise.binary 操作，使用 QConvoneDNN::run_pointwise_binary 函数
  // Conv2D with binary postop
  m.impl(TORCH_SELECTIVE_NAME("onednn::qconv2d_pointwise.binary"), QConvoneDNN::run_pointwise_binary);
}

} // namespace
} // namespace at::native
```