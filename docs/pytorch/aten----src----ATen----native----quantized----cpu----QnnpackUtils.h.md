# `.\pytorch\aten\src\ATen\native\quantized\cpu\QnnpackUtils.h`

```py
#pragma once
// 如果定义了宏 USE_PYTORCH_QNNPACK，则包含以下头文件

#ifdef USE_PYTORCH_QNNPACK
#include <ATen/core/Tensor.h>  // 引入 ATen 库的 Tensor 类
#include <c10/util/irange.h>   // 引入 c10 库的 irange 工具
#include <pytorch_qnnpack.h>   // 引入 PyTorch QNNPACK 头文件
#include <qnnpack_func.h>      // 引入 QNNPACK 函数定义
#include <ATen/native/quantized/cpu/XnnpackUtils.h>  // 引入 ATen 库的 quantized CPU XNNPACK 工具
#include <ATen/native/quantized/PackedParams.h>     // 引入 ATen 库的 quantized PackedParams
#include <ATen/native/utils/Factory.h>              // 引入 ATen 库的 Factory 工具

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>  // 引入 ATen 库的 Functions 头文件
#else
#include <ATen/ops/empty.h>  // 引入 ATen 库的 empty 操作头文件
#endif

#include <utility>  // 引入 C++ 标准库的 utility 头文件

// 定义整数常量 kPaddingChannels，值为 8
inline int kPaddingChannels = 8;

// 定义结构体 QnnpackOperatorDeleter，用于管理 QNNPACK 操作符的销毁
struct QnnpackOperatorDeleter {
  void operator()(pytorch_qnnp_operator_t op) {
    pytorch_qnnp_delete_operator(op);  // 调用 QNNPACK 的操作符删除函数
  }
};

// PackedWeight 结构体用于存储 QNNPACK 的打包权重和偏置参数
// 在 PyTorch Mobile 中，一旦模型被脚本化和序列化，就不需要调用 unpack 函数，可以在打包后释放原始权重，节省内存
// 预打包步骤中输入比例设置为空。QNNPACK 需要将偏置量化与运行时的输入比例一起使用，这在 PyTorch 中是可用的。
// 在运行时，如果输入比例值发生变化，则重新量化偏置量化为更新的比例。对于推断，我们期望图形是静态的，因此输入比例在连续推断调用中不应更改。
struct PackedLinearWeightsQnnp : public LinearPackedParamsBase {
  PackedLinearWeightsQnnp(
      std::unique_ptr<qnnpack::PackBMatrix> w,  // 打包矩阵的唯一指针
      at::Tensor orig_weight,                   // 原始权重张量
      at::Tensor bias,                          // 偏置张量
      std::optional<double> input_scale,        // 可选的输入比例
      at::Tensor w_scales,                      // 权重比例张量
      std::vector<uint8_t>&& w_zps)            // 权重零点的移动语义向量
      : w(std::move(w)),
        orig_weight(std::move(orig_weight)),
        bias_(at::native::mobile::allocate_padded_contiguous_if_needed(
            bias, bias.suggest_memory_format())),  // 分配或复制填充的偏置张量
        per_channel_(this->orig_weight.qscheme() == at::kPerChannelAffine),  // 检查是否为通道间隙比
        input_scale(std::move(input_scale)),   // 移动语义输入比例
        w_scales(std::move(w_scales)),         // 移动语义权重比例
        w_zero_points(std::move(w_zps)),       // 移动语义权重零点
        q_scheme(this->orig_weight.qscheme()) {  // 原始权重的 QScheme
    weight_sizes = this->orig_weight.sizes().vec();  // 保存权重大小的向量
  }

  std::unique_ptr<qnnpack::PackBMatrix> w;    // 打包矩阵的唯一指针
  at::Tensor orig_weight;                     // 原始权重张量
  at::Tensor bias_;                           // 偏置张量
  bool per_channel_;                          // 是否为通道间隙比
  std::optional<double> input_scale;          // 可选的输入比例
  at::Tensor w_scales;                        // 权重比例张量
  std::vector<uint8_t> w_zero_points;         // 权重零点的向量
  std::vector<float> requantization_scales;   // 重新量化比例的向量
  std::vector<int64_t> weight_sizes;          // 权重大小的向量
  c10::QScheme q_scheme;                      // QScheme 类型

  // 应用函数，应用线性层操作
  at::Tensor apply(
      at::Tensor input,
      double output_scale,
      int64_t output_zero_point) override;

  // 应用函数，应用带 ReLU 的线性层操作
  at::Tensor apply_relu(
      at::Tensor input,
      double output_scale,
      int64_t output_zero_point) override;

  // 应用动态函数，应用动态线性层操作
  at::Tensor apply_dynamic(at::Tensor input, bool reduce_range=false) override;

  // 应用动态函数，应用带 ReLU 的动态线性层操作
  at::Tensor apply_dynamic_relu(at::Tensor input, bool reduce_range=false) override;

  // 解包函数，解包张量
  std::tuple<at::Tensor, std::optional<at::Tensor>> unpack() override;

  // 偏置函数，返回偏置张量的可选值
  std::optional<at::Tensor> bias() override {


这段代码主要定义了一些结构体和函数，用于在 PyTorch 中实现对 QNNPACK 的支持和优化。
    // 返回私有成员变量 bias_
    return bias_;
  }

  // 静态方法，用于预打包线性层参数
  static c10::intrusive_ptr<LinearPackedParamsBase> prepack(
      at::Tensor weight,
      std::optional<at::Tensor> bias);

  // 返回私有成员变量 per_channel_
  bool per_channel() const {
    return per_channel_;
  }

 private:
  // 用于保护 QNNPACK 相关操作的互斥量
  std::mutex qnnp_mutex_;
#ifdef USE_XNNPACK
  // 如果定义了 USE_XNNPACK 宏，则声明 XNNPACK 操作符变量
  xnnpack_operator xnnp_linear_op;

  // 定义一个模板函数 apply_impl_xnnp，用于在使用 XNNPACK 的情况下应用线性操作
  template <typename scalar_t, bool kReluFused>
  at::Tensor apply_impl_xnnp(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point);
#endif // USE_XNNPACK

// 定义一个模板函数 apply_impl，用于应用线性操作
template <bool ReluFused>
at::Tensor apply_impl(
    at::Tensor input,
    double output_scale,
    int64_t output_zero_point);

// 定义一个模板函数 apply_dynamic_impl，用于动态应用操作
template <bool ReluFused>
at::Tensor apply_dynamic_impl(at::Tensor input, bool reduce_range);
};

// 定义一个模板结构 PackedConvWeightsQnnp，继承自 ConvPackedParamsBase，用于包装 QNNPACK 的卷积权重
template <int kSpatialDim = 2>
struct PackedConvWeightsQnnp : public ConvPackedParamsBase<kSpatialDim> {
  // 构造函数，初始化 QNNPACK 的卷积权重包装器
  PackedConvWeightsQnnp(
      std::unique_ptr<qnnpack::PrePackConvWeights> w,
      at::Tensor orig_weight,
      at::Tensor bias,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> output_padding,
      torch::List<int64_t> dilation,
      int64_t groups,
      bool transpose,
      std::optional<double> input_scale,
      std::vector<int64_t> kernel,
      at::Tensor w_scale,
      std::vector<uint8_t>&& w_zps,
      bool is_per_channel)
      : w(std::move(w)),
        orig_weight(std::move(orig_weight)),
        bias(std::move(bias)),
        stride_(std::move(stride)),
        padding_(std::move(padding)),
        output_padding_(std::move(output_padding)),
        dilation_(std::move(dilation)),
        groups_(groups),
        transpose_(transpose),
        is_per_channel_(is_per_channel),
        input_scale(input_scale),
        kernel_(std::move(kernel)),
        w_scales(std::move(w_scale)),
        w_zero_points(std::move(w_zps)) {
    // 检查是否有任何填充值不为零
    const bool any_padding = std::any_of(
        padding_.begin(), padding_.end(), [](const auto& e) { return e != 0; });
    // 计算卷积核的大小
    const size_t kernel_size =
        std::accumulate(kernel_.begin(), kernel_.end(), 1, std::multiplies<>());

    // 计算每组的输入通道数和输出通道数
    const size_t group_input_channels = transpose
        ? this->orig_weight.size(0) / groups
        : this->orig_weight.size(1);
    const size_t group_output_channels = transpose
        ? this->orig_weight.size(1)
        : this->orig_weight.size(0) / groups;

    // 计算卷积核的深度、高度和宽度
    const size_t kernel_depth = kSpatialDim == 3 ? kernel_[0] : 1;
    const size_t kernel_height = kernel_[kSpatialDim - 2];
    const size_t kernel_width = kernel_[kSpatialDim - 1];

    // 设置 QNNPACK 的 ukernel 类型为卷积类型，如果是转置卷积，则设置为转置卷积类型
    pytorch_qnnp_ukernel_type ukernel_type;
    if (transpose_) {
      ukernel_type = pytorch_qnnp_ukernel_type_conv;
    } else {
      // 如果不满足任何特殊条件，使用标准卷积运算器类型
      ukernel_type = pytorch_qnnp_ukernel_type_none;

      // 检查是否满足深度可分离卷积的维度要求和分组要求
      const bool has_depthwise_dimensions =
          (kSpatialDim == 2 &&
           ((kernel_height == 3 && kernel_width == 3) ||
            (kernel_height == 5 && kernel_width == 5))) ||
          (kSpatialDim == 3 && kernel_height == 3 && kernel_width == 3 &&
           kernel_depth == 3);
      const bool has_depthwise_grouping =
          group_input_channels == 1 && group_output_channels == 1 && groups > 1;

      // 根据满足的条件设置深度可分离卷积运算器类型
      if (has_depthwise_dimensions && has_depthwise_grouping) {
        ukernel_type = pytorch_qnnp_ukernel_type_dwconv;
      } else if (
          // 检查是否满足零填充且卷积核大小为 1 的条件
          kernel_size == 1 &&
          std::all_of(
              stride_.begin(),
              stride_.end(),
              [](const auto& e) { return e == 1; }) &&
          !any_padding) {
        // 根据输入通道数设置卷积运算器类型为 gemm 或 xzp_gemm
        ukernel_type = group_input_channels >= SIZE_MAX
            ? pytorch_qnnp_ukernel_type_xzp_gemm
            : pytorch_qnnp_ukernel_type_gemm;
      } else {
        // 默认情况下使用标准卷积运算器类型
        ukernel_type = pytorch_qnnp_ukernel_type_conv;
      }
    }

    // 如果是按通道量化且卷积运算器类型为 xzp_gemm，则抛出错误
    if (is_per_channel && ukernel_type == pytorch_qnnp_ukernel_type_xzp_gemm) {
      TORCH_INTERNAL_ASSERT(
          false, "Per channel quantized weights are not supported for XZP kernels");
    }

    // 初始化卷积操作符，所有参数初始设为零
    pytorch_qnnp_operator_t convolution{nullptr};
    convolution = static_cast<pytorch_qnnp_operator_t>(
        calloc(1, sizeof(struct pytorch_qnnp_operator)));
    // 检查内存分配是否成功
    if (convolution == nullptr) {
      TORCH_INTERNAL_ASSERT(
          false, "failed to allocate %zu bytes for pytorch_qnnp_operator structure",
          sizeof(struct pytorch_qnnp_operator));
    }

    // 使用 unique_ptr 管理卷积操作符的生命周期
    convolution_op =
        std::unique_ptr<pytorch_qnnp_operator, QnnpackOperatorDeleter>(
            convolution);

    // 设置卷积操作符的各项参数
    convolution->ukernel_type = ukernel_type;
    convolution->groups = groups;
    convolution->group_input_channels = group_input_channels;
    convolution->group_output_channels = group_output_channels;
    convolution->kernel_depth = kernel_depth;
    convolution->kernel_height = kernel_height;
    convolution->kernel_width = kernel_width;
    convolution->stride_depth = kSpatialDim == 3 ? stride_[0] : 1;
    convolution->stride_height = stride_[kSpatialDim - 2];
    convolution->stride_width = stride_[kSpatialDim - 1];
    convolution->dilation_depth = kSpatialDim == 3 ? dilation_[0] : 1;
    convolution->dilation_height = dilation_[kSpatialDim - 2];
    convolution->dilation_width = dilation_[kSpatialDim - 1];
    convolution->input_padding_height = padding_[kSpatialDim - 2];
    convolution->input_padding_width = padding_[kSpatialDim - 1];
    convolution->input_padding_depth = kSpatialDim == 3 ? padding_[0] : 0;
    convolution->per_channel = is_per_channel_;
    convolution->transpose = transpose_;

    // 设置卷积操作符的 kr 参数
    const uint32_t kr = pytorch_qnnp_params.q8conv.kr;
    // 计算对齐后的步长，确保能够容纳所有输入通道
    const size_t k_stride = (group_input_channels + (kr - 1)) & -kr;

    // 计算零填充缓冲区的大小，以字节为单位
    size_t zero_size = sizeof(uint8_t) * k_stride;
    // 零填充缓冲区的起始偏移量
    size_t zero_offset = 0;

    // 如果需要转置操作
    if (transpose_) {
      // 设置卷积对象的宽度调整值为输出填充的宽度
      convolution->adjustment_width = output_padding_[1];
      // 设置卷积对象的高度调整值为输出填充的高度
      convolution->adjustment_height = output_padding_[0];
      // 如果输入通道数小于8，调整零填充缓冲区大小和偏移量
      if (group_input_channels < 8) {
        zero_size += 8;
        zero_offset = 8;
      }
    } else {
      // 对于非转置操作，不需要零填充缓冲区
      zero_buffer_size = 0;
      // 如果存在任何填充
      if (any_padding) {
        // 重置零填充缓冲区大小和偏移量
        zero_size = 0;
        zero_offset = 0;
        // 对于深度卷积类型，根据组数调整零填充缓冲区大小和偏移量
        if (ukernel_type == pytorch_qnnp_ukernel_type_dwconv) {
          const uint32_t cr = pytorch_qnnp_params.q8dw9.cr;
          const size_t group_stride = (groups + (cr - 1)) & -cr;
          if (groups >= 8) {
            zero_size = sizeof(uint8_t) * group_stride;
            zero_offset = 0;
          } else {
            zero_size = sizeof(uint8_t) * group_stride + 8;
            zero_offset = sizeof(uint8_t) * 8;
          }
        } else if (
            ukernel_type == pytorch_qnnp_ukernel_type_conv ||
            ukernel_type == pytorch_qnnp_ukernel_type_gemm) {
          // 对于普通卷积和矩阵乘法类型，根据输入通道数调整零填充缓冲区大小和偏移量
          if (group_input_channels >= 8) {
            zero_size = sizeof(uint8_t) * k_stride;
            zero_offset = 0;
          } else {
            zero_size = sizeof(uint8_t) * k_stride + 8;
            zero_offset = 8;
          }
        }
      }
    }

    // 分配零填充缓冲区的内存空间
    // NOLINTNEXTLINE(clang-analyzer-optin.portability.UnixAPI)
    void* zero_buffer = malloc(zero_size);
    // 如果内存分配失败，则删除卷积操作对象并抛出错误
    if (zero_buffer == nullptr) {
      pytorch_qnnp_delete_operator(convolution);
      TORCH_INTERNAL_ASSERT(
          false, "failed to allocate %zu bytes for zero padding",
          zero_size);
    }
    // 设置零填充缓冲区的大小
    zero_buffer_size = zero_size;
    // 将分配的零填充缓冲区的指针赋给卷积对象的零填充缓冲区字段
    convolution->zero_buffer = zero_buffer;
  convolution->zero_pointer = (void*)((uintptr_t)zero_buffer + zero_offset);

# 将 convolution 对象的 zero_pointer 设置为 zero_buffer 内存地址加上 zero_offset 的偏移量。

  std::unique_ptr<pytorch_qnnp_operator, QnnpackOperatorDeleter> convolution_op;

# 使用 std::unique_ptr 来管理 pytorch_qnnp_operator 对象，使用 QnnpackOperatorDeleter 自定义删除器。

#ifdef USE_XNNPACK
  xnnpack_operator xnnp_convolution_op;
#endif  // USE_XNNPACK

# 如果定义了 USE_XNNPACK 宏，则声明 xnnpack_operator 类型的 xnnp_convolution_op 对象。

  std::unique_ptr<qnnpack::PrePackConvWeights> w;

# 使用 std::unique_ptr 来管理 qnnpack::PrePackConvWeights 对象的所有权。

  at::Tensor orig_weight;
  at::Tensor bias;
  torch::List<int64_t> stride_;
  torch::List<int64_t> padding_;
  torch::List<int64_t> output_padding_;
  torch::List<int64_t> dilation_;
  int64_t groups_;
  bool transpose_;
  bool is_per_channel_;
  std::optional<double> input_scale;
  std::vector<int64_t> kernel_;
  at::Tensor w_scales;
  std::vector<uint8_t> w_zero_points;
  std::vector<float> requantization_scales;
  size_t zero_buffer_size;

# 定义多个成员变量，包括原始权重、偏置、步幅、填充、输出填充、扩张、分组数、是否转置、是否按通道量化、输入缩放（可选）、卷积核大小、权重缩放张量、权重零点向量、重量化缩放向量以及零缓冲区大小。

  at::Tensor apply(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point) override;

# 定义虚函数 apply，接受输入张量、输出缩放比例和输出零点，返回处理后的张量。

  at::Tensor apply_relu(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point) override;

# 定义虚函数 apply_relu，接受输入张量、输出缩放比例和输出零点，应用 ReLU 激活后返回处理后的张量。

  at::Tensor apply_dynamic(
      const at::Tensor& input,
      bool reduce_range=false) override;

# 定义虚函数 apply_dynamic，接受输入张量和是否减少范围的布尔值，默认为 false，返回动态处理后的张量。

  std::tuple<at::Tensor, std::optional<at::Tensor>> unpack() override;

# 定义虚函数 unpack，返回张量元组，其中第二个张量是可选的。

  static c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>> prepack(
      at::Tensor weight,
      std::optional<at::Tensor> bias,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> output_padding,
      torch::List<int64_t> dilation,
      int64_t groups,
      bool transpose);

# 定义静态函数 prepack，用于预打包卷积参数，返回包含卷积参数的特定类型的指针。

  torch::List<int64_t> stride() const override {
    return stride_;
  }

# 实现虚函数 stride，返回成员变量 stride_。

  torch::List<int64_t> padding() const override {
    return padding_;
  }

# 实现虚函数 padding，返回成员变量 padding_。

  torch::List<int64_t> output_padding() const override {
    return output_padding_;
  }

# 实现虚函数 output_padding，返回成员变量 output_padding_。

  torch::List<int64_t> dilation() const override {
    return dilation_;
  }

# 实现虚函数 dilation，返回成员变量 dilation_。

  int64_t groups() const override {
    return groups_;
  }

# 实现虚函数 groups，返回成员变量 groups_。

  bool transpose() const override {
    return transpose_;
  }

# 实现虚函数 transpose，返回成员变量 transpose_。

  bool per_channel() const {
    return is_per_channel_;
  }

# 成员函数 per_channel，返回成员变量 is_per_channel_。

private:
  std::mutex qnnp_mutex_;

# 声明私有成员变量 qnnp_mutex_，用于线程安全的互斥访问。

  template <bool ReluFused>
  at::Tensor apply_impl(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point);

# 声明模板函数 apply_impl，接受输入张量、输出缩放比例和输出零点，实现特定的卷积操作并返回处理后的张量。
#ifdef USE_XNNPACK
  template <typename scalar_t, bool ReluFused>
  // 如果定义了 USE_XNNPACK 宏，则使用 XNNPACK 库实现的模板函数
  at::Tensor apply_impl_xnnp(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point);
#endif // USE_XNNPACK
};

enum class Activation : uint8_t { NONE = 0, RELU = 1 };

#if defined(__ANDROID__) && !defined(__NDK_MAJOR__)
template <class T>
// 如果编译环境是 Android 且不是 NDK 主要版本，则定义浮点数舍入函数 Round
inline float Round(const float x) {
  return ::nearbyintf(x);
}
// 对双精度浮点数进行舍入
inline double Round(const double x) {
  return ::nearbyint(x);
}
#else
template <class T>
// 默认情况下，定义通用的舍入函数 Round
inline T Round(const T x) {
  return std::nearbyint(x);
}
#endif

template<typename T>
// 根据指定的缩放因子和零点对浮点数值进行量化
inline T QuantizeValue(float scale, int32_t zero_point, float value) {
  const int32_t qmin = std::numeric_limits<T>::min();
  const int32_t qmax = std::numeric_limits<T>::max();
  auto r = zero_point + static_cast<int32_t>(Round(value / scale));
  r = std::max(r, qmin);
  r = std::min(r, qmax);
  return static_cast<T>(r);
}

template<typename T>
// 根据激活函数类型和量化参数，返回量化后数值的上下限
inline std::pair<T, T> activationLimits(
    float scale,
    int32_t zero_point,
    Activation Ac) {
  switch (Ac) {
    case Activation::NONE:
      // 对于无激活函数，返回数据类型 T 的最小和最大值
      return {std::numeric_limits<T>::min(),
              std::numeric_limits<T>::max()};
    case Activation::RELU:
      // 对于 ReLU 激活函数，返回量化后的下限为 0，上限为数据类型 T 的最大值
      return {QuantizeValue<T>(scale, zero_point, 0.0),
              std::numeric_limits<T>::max()};
    default:
#ifdef _MSC_VER
      __assume(0);
#else
      // 对于不可能出现的情况，标记为不可达状态
      __builtin_unreachable();
#endif
  }
}

namespace at {
namespace native {
namespace qnnp_avgpool_helper {
// 在 qnnp_avgpool_helper 命名空间中定义 QNNPACK 实现的平均池化函数
Tensor qnnpack_avg_pool2d(
    Tensor input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    std::optional<int64_t> divisor_override);
} // qnnp_avgpool_helper
} // namespace native
} // namespace at

namespace {
// 在匿名命名空间中定义一个未使用的函数，生成重新量化的比例尺
C10_UNUSED std::vector<float> generate_requantization_scales(
    const at::Tensor& weight_scales,
    const float input_scale,
    const float output_scale,
    std::vector<float>& requant_scales) {
  // 由于权重比例带有填充，通过 weight_scales.numel() 获取填充后的元素数
  const auto num_output_channels_padded = weight_scales.numel();
  float *const weight_scales_data = weight_scales.data_ptr<float>();
  if (static_cast<int64_t>(requant_scales.size()) < num_output_channels_padded) {
    requant_scales.resize(num_output_channels_padded);
  }
  for (const auto i : c10::irange(num_output_channels_padded)) {
    const auto inverse_output_scale = 1.f /output_scale;
    // 计算重新量化比例尺
    requant_scales[i] = (weight_scales_data[i] * input_scale) * inverse_output_scale;
    TORCH_CHECK(
        (requant_scales[i] > 0.0f && std::isnormal(requant_scales[i])),
        // 检查重新量化比例尺是否有效
        "failed to create op with requantization scale: ",
        requant_scales[i],
        ": requantization scale must be finite and positive");
  }
  return requant_scales;
}

// 在匿名命名空间中定义一个未使用的函数，创建零点和比例尺张量
C10_UNUSED std::pair<std::vector<uint8_t>, at::Tensor> make_zero_points_and_scales_tensor(
    const at::Tensor& weight_contig,
    bool transpose = false,
    uint32_t groups = 1
  ) {
  // 确定输出通道索引，考虑是否需要转置
  const int out_ch_idx = transpose ? 1 : 0;
  // 计算输出通道的数量，考虑组数和是否转置，同时加上用于 QNNPACK 缓冲的额外通道数
  const auto num_output_channels = weight_contig.size(out_ch_idx) * (transpose ? groups : 1);
  // 添加 8 来适应 QNNPACK 需要的缓冲
  const auto num_output_channels_padded = num_output_channels + kPaddingChannels;
  // 获取权重的量化类型
  const auto qtype = weight_contig.qscheme();
  // 创建一个用于存储权重零点的向量，初始化为零
  std::vector<uint8_t> weight_zp(num_output_channels_padded, 0);
  // 调整权重的零点，与权重数据类似
  if (qtype == at::kPerTensorAffine) {
    // 对于每个输出通道，根据权重的量化零点进行调整
    for (const auto i : c10::irange(num_output_channels)) {
      weight_zp[i] = (uint8_t)(weight_contig.q_zero_point() + 128);
    }
  } else if (qtype == at::kPerChannelAffine) {
    // 检查每通道零点的数据类型必须为 long int
    TORCH_CHECK(
        weight_contig.q_per_channel_zero_points().scalar_type() == at::kLong,
        "Per channel zero points dtype must be long int.");
    // 获取每通道的零点数组指针
    const int64_t* per_channel_zero_points =
      weight_contig.q_per_channel_zero_points().data_ptr<int64_t>();
    // 对于每个输出通道，根据对应通道的零点进行调整
    for (const auto i : c10::irange(num_output_channels)) {
      weight_zp[i] = (uint8_t)(per_channel_zero_points[i] + 128);
    }
  } else {
    // 不支持的量化方案
    TORCH_INTERNAL_ASSERT(false, "Unsupported quantization scheme.");
  }
  // 创建存储权重缩放因子的张量，初始化为空
  at::Tensor weight_scales =
    at::empty(
        {num_output_channels_padded},
        at::device(at::kCPU).dtype(at::kFloat));
  // 获取权重缩放因子的数据指针
  float *const weight_scales_data = weight_scales.data_ptr<float>();
  // 根据量化类型设置权重缩放因子
  if (qtype == at::kPerTensorAffine) {
    // 对于每个输出通道，设置为权重的量化比例因子
    for (const auto i : c10::irange(num_output_channels)) {
      weight_scales_data[i] = weight_contig.q_scale();
    }
  } else if (qtype == at::kPerChannelAffine) {
    // 检查每通道缩放因子的数据类型必须为 double
    TORCH_CHECK(
        weight_contig.q_per_channel_scales().scalar_type() == at::kDouble,
        "Per channel scales dtype must be double.");
    // 获取每通道的缩放因子数组指针
    const double *const per_channel_scales =
      weight_contig.q_per_channel_scales().data_ptr<double>();
    // 对于每个输出通道，设置为对应通道的缩放因子
    for (const auto i : c10::irange(num_output_channels)) {
      weight_scales_data[i] = static_cast<float>(per_channel_scales[i]);
    }
  } else {
    // 不支持的量化方案
    TORCH_INTERNAL_ASSERT(false, "Unsupported quantization scheme.");
  }
  // 将多余的输出通道的权重缩放因子设置为 1.0
  for (const auto i : c10::irange(num_output_channels, num_output_channels_padded)) {
    weight_scales_data[i] = 1.f;
  }
  // 返回权重的零点和缩放因子
  return {weight_zp, weight_scales};
}
} // namespace
#endif

注释：

// 结束命名空间定义的闭合
}
// 结束 C++ 头文件的条件编译保护
#endif
```