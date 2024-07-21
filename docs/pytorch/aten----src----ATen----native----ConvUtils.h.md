# `.\pytorch\aten\src\ATen\native\ConvUtils.h`

```
#pragma once
// 引入 ATen 库中的 Tensor 类定义
#include <ATen/core/Tensor.h>
// 引入 ATen 库中的 TensorUtils 工具函数
#include <ATen/TensorUtils.h>
// 引入 ATen 库中的 CUDAHooksInterface 接口定义
#include <ATen/detail/CUDAHooksInterface.h>
// 引入 ATen 库中的 DispatchStub 接口定义
#include <ATen/native/DispatchStub.h>
// 引入 C10 库中的环境变量相关功能
#include <c10/util/env.h>
// 引入 C10 库中的整数范围工具函数
#include <c10/util/irange.h>

// ATen 库的命名空间 at::native
namespace at::native {

// 定义一个函数指针类型 conv_depthwise2d_backward_fn，用于处理深度可分离卷积反向传播
using conv_depthwise2d_backward_fn = std::tuple<at::Tensor,at::Tensor>(*)(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, at::IntArrayRef, at::IntArrayRef,
    at::IntArrayRef, at::IntArrayRef, std::array<bool, 2>);
// 声明 conv_depthwise2d_backward_fn 的分发函数
DECLARE_DISPATCH(conv_depthwise2d_backward_fn, conv_depthwise2d_backward_stub);

// 定义一个函数指针类型 conv_depthwise3d_backward_fn，用于处理三维深度可分离卷积反向传播
using conv_depthwise3d_backward_fn = std::tuple<at::Tensor,at::Tensor,at::Tensor>(*)(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, at::IntArrayRef, at::IntArrayRef,
    at::IntArrayRef, at::IntArrayRef, std::array<bool, 3>);
// 声明 conv_depthwise3d_backward_fn 的分发函数
DECLARE_DISPATCH(conv_depthwise3d_backward_fn, conv_depthwise3d_backward_stub);

// 定义一个函数指针类型 cudnn_convolution_backward_fn，用于处理 cudnn 卷积反向传播
using cudnn_convolution_backward_fn = std::tuple<at::Tensor,at::Tensor>(*)(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, at::IntArrayRef, at::IntArrayRef,
    at::IntArrayRef, int64_t, bool, bool, bool, std::array<bool,2>);
// 声明 cudnn_convolution_backward_fn 的分发函数
DECLARE_DISPATCH(cudnn_convolution_backward_fn, cudnn_convolution_backward_stub);

// 定义一个函数指针类型 mps_convolution_backward_fn，用于处理 mps 卷积反向传播
using mps_convolution_backward_fn = std::tuple<at::Tensor,at::Tensor,at::Tensor>(*)(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, at::IntArrayRef, at::IntArrayRef,
    at::IntArrayRef, int64_t, std::array<bool,3>);
// 声明 mps_convolution_backward_fn 的分发函数
DECLARE_DISPATCH(mps_convolution_backward_fn, mps_convolution_backward_stub);

// 定义一个函数指针类型 cudnn_convolution_transpose_backward_fn，用于处理 cudnn 转置卷积反向传播
using cudnn_convolution_transpose_backward_fn = std::tuple<at::Tensor,at::Tensor>(*)(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, at::IntArrayRef, at::IntArrayRef,
    at::IntArrayRef, at::IntArrayRef, int64_t, bool, bool, bool, std::array<bool,2>);
// 声明 cudnn_convolution_transpose_backward_fn 的分发函数
DECLARE_DISPATCH(cudnn_convolution_transpose_backward_fn, cudnn_convolution_transpose_backward_stub);

// 定义一个函数指针类型 miopen_convolution_backward_fn，用于处理 miopen 卷积反向传播
using miopen_convolution_backward_fn = std::tuple<at::Tensor,at::Tensor,at::Tensor>(*)(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, at::IntArrayRef, at::IntArrayRef,
    at::IntArrayRef, int64_t, bool, bool, std::array<bool,3>);
// 声明 miopen_convolution_backward_fn 的分发函数
DECLARE_DISPATCH(miopen_convolution_backward_fn, miopen_convolution_backward_stub);

// 定义一个函数指针类型 miopen_convolution_transpose_backward_fn，用于处理 miopen 转置卷积反向传播
using miopen_convolution_transpose_backward_fn = std::tuple<at::Tensor,at::Tensor,at::Tensor>(*)(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, at::IntArrayRef, at::IntArrayRef,
    at::IntArrayRef, at::IntArrayRef, int64_t, bool, bool, std::array<bool,3>);
// 声明 miopen_convolution_transpose_backward_fn 的分发函数
DECLARE_DISPATCH(miopen_convolution_transpose_backward_fn, miopen_convolution_transpose_backward_stub);

// 定义一个函数指针类型 miopen_depthwise_convolution_backward_fn，用于处理 miopen 深度可分离卷积反向传播
using miopen_depthwise_convolution_backward_fn = std::tuple<at::Tensor,at::Tensor,at::Tensor>(*)(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, at::IntArrayRef, at::IntArrayRef,
    at::IntArrayRef, int64_t, bool, bool, std::array<bool,3>);
// 声明 miopen_depthwise_convolution_backward_fn 的分发函数
DECLARE_DISPATCH(miopen_depthwise_convolution_backward_fn, miopen_depthwise_convolution_backward_stub);

} // namespace at::native
using mkldnn_convolution_backward_fn = std::tuple<at::Tensor,at::Tensor,at::Tensor>(*)(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, at::IntArrayRef, at::IntArrayRef,
    at::IntArrayRef, int64_t, std::array<bool,3>);
DECLARE_DISPATCH(mkldnn_convolution_backward_fn, mkldnn_convolution_backward_stub);


// 定义 mkldnn_convolution_backward_fn 类型，表示一个指向函数的指针，该函数接受多个参数并返回三个张量的元组
using mkldnn_convolution_backward_fn = std::tuple<at::Tensor,at::Tensor,at::Tensor>(*)(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, at::IntArrayRef, at::IntArrayRef,
    at::IntArrayRef, int64_t, std::array<bool,3>);

// 声明 mkldnn_convolution_backward_stub，用于分发 mkldnn_convolution_backward_fn 类型的函数
DECLARE_DISPATCH(mkldnn_convolution_backward_fn, mkldnn_convolution_backward_stub);

using mkldnn_convolution_transpose_fn = Tensor(*)(const Tensor&, const Tensor&, const std::optional<Tensor>&,
    IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, int64_t);
DECLARE_DISPATCH(mkldnn_convolution_transpose_fn, mkldnn_convolution_transpose_stub);

using mkldnn_convolution_transpose_backward_fn = std::tuple<at::Tensor,at::Tensor,at::Tensor>(*)(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, at::IntArrayRef, at::IntArrayRef,
    at::IntArrayRef, at::IntArrayRef, int64_t, std::array<bool,3>);
DECLARE_DISPATCH(mkldnn_convolution_transpose_backward_fn, mkldnn_convolution_transpose_backward_stub);

using slow_conv_dilated2d_backward_fn = std::tuple<at::Tensor,at::Tensor,at::Tensor>(*)(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, at::IntArrayRef, at::IntArrayRef,
    at::IntArrayRef, at::IntArrayRef, std::array<bool, 3>);
DECLARE_DISPATCH(slow_conv_dilated2d_backward_fn, slow_conv_dilated2d_backward_stub);

using slow_conv_dilated3d_backward_fn = std::tuple<at::Tensor,at::Tensor,at::Tensor>(*)(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, at::IntArrayRef, at::IntArrayRef,
    at::IntArrayRef, at::IntArrayRef, std::array<bool, 3>);
DECLARE_DISPATCH(slow_conv_dilated3d_backward_fn, slow_conv_dilated3d_backward_stub);

using slow_conv_transpose2d_backward_fn = std::tuple<at::Tensor,at::Tensor,at::Tensor>(*)(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, at::IntArrayRef, at::IntArrayRef,
    at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, std::array<bool,3>);
DECLARE_DISPATCH(slow_conv_transpose2d_backward_fn, slow_conv_transpose2d_backward_stub);

using slow_conv_transpose3d_backward_fn = std::tuple<at::Tensor,at::Tensor,at::Tensor>(*)(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, at::IntArrayRef, at::IntArrayRef,
    at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, std::array<bool,3>);
DECLARE_DISPATCH(slow_conv_transpose3d_backward_fn, slow_conv_transpose3d_backward_stub);

namespace {
  bool is_cudnnv8_heuristic_mode_b() {
    // 检查环境变量 TORCH_CUDNN_USE_HEURISTIC_MODE_B 是否为 true，用于启用 cudnnv8 的启发式模式 B
    static const bool is_cudnnv8_heuristic_mode_b = c10::utils::check_env("TORCH_CUDNN_USE_HEURISTIC_MODE_B") == true;
    return is_cudnnv8_heuristic_mode_b;
  }
}

inline bool cudnnv8_enabled_check_debug() {
  // 检查环境变量 TORCH_CUDNN_V8_API_DISABLED 是否不为 true，用于检查 cudnnv8 API 是否启用
  static bool cudnnv8_flag = c10::utils::check_env("TORCH_CUDNN_V8_API_DISABLED") != true;
  // 检查环境变量 TORCH_CUDNN_V8_API_DEBUG 是否为 true，用于启用 cudnnv8 的调试模式
  static bool cudnnv8_debug = c10::utils::check_env("TORCH_CUDNN_V8_API_DEBUG") == true;
  static uint8_t cudnnv8_debugcount = 0;
  if (cudnnv8_debug == 1 && cudnnv8_debugcount < 10) {
    // 如果开启了 cudnnv8 的调试模式并且计数未达到10次，则发出警告
    TORCH_WARN("TORCH_CUDNN_V8_DEBUG ON, V8 ON: ", cudnnv8_flag, " TORCH_CUDNN_USE_HEURISTIC_MODE B: ", is_cudnnv8_heuristic_mode_b());
    # 增加 cudnnv8_debugcount 变量的值，用于调试计数
    cudnnv8_debugcount++;
  }
  # 返回 cudnnv8_flag 变量是否等于 1 的布尔结果
  return cudnnv8_flag == 1;
}

// 返回一个布尔值，指示是否使用 CUDNNv8 的启发式模式 B
inline bool cudnnv8_use_heur_mode_b() {
  return is_cudnnv8_heuristic_mode_b();
}

// 与 Module.cpp 中的 py::enum_ 保持同步
// 枚举卷积后端类型
enum class ConvBackend {
  CudaDepthwise2d,      // CUDA 二维深度卷积
  CudaDepthwise3d,      // CUDA 三维深度卷积
  Cudnn,                // cuDNN 卷积
  CudnnTranspose,       // cuDNN 转置卷积
  Empty,                // 空卷积
  Miopen,               // MIOpen 卷积
  MiopenDepthwise,      // MIOpen 深度卷积
  MiopenTranspose,      // MIOpen 转置卷积
  Mkldnn,               // MKL-DNN 卷积
  MkldnnTranspose,      // MKL-DNN 转置卷积
  MkldnnEmpty,          // 空 MKL-DNN 卷积
  NnpackSpatial,        // NNPACK 空间卷积
  Overrideable,         // 可重写的卷积
  Slow2d,               // 缓慢的二维卷积
  Slow3d,               // 缓慢的三维卷积
  SlowDilated2d,        // 缓慢的二维膨胀卷积
  SlowDilated3d,        // 缓慢的三维膨胀卷积
  SlowTranspose2d,      // 缓慢的二维转置卷积
  SlowTranspose3d,      // 缓慢的三维转置卷积
  Winograd3x3Depthwise, // Winograd 3x3 深度卷积
  Xnnpack2d,            // XNNPACK 二维卷积
  Mps,                  // MPS 卷积
  MpsTranspose,         // MPS 转置卷积
};

// 从完整的卷积输入中选择卷积后端的重载函数
// 此重载函数暴露给 Python 用于测试等
TORCH_API ConvBackend select_conv_backend(
    const Tensor& input, const Tensor& weight, const std::optional<Tensor>& bias_opt,
    SymIntArrayRef stride, SymIntArrayRef padding, SymIntArrayRef dilation,
    bool transposed, SymIntArrayRef output_padding, c10::SymInt groups, const at::OptionalSymIntArrayRef bias_sizes_opt);

// 确定后端内存格式的函数
TORCH_API at::MemoryFormat _determine_backend_memory_format(const Tensor& input,
    const Tensor& weight,
    const ConvBackend backend);

// ---------------------------------------------------------------------
//
// Math
//
// ---------------------------------------------------------------------

// 输入批量大小维度和梯度输入维度
constexpr int input_batch_size_dim = 0;
constexpr int input_channels_dim = 1;
// 输出批量大小维度和梯度输出维度
constexpr int output_batch_size_dim = 0;
constexpr int output_channels_dim = 1;
// 权重输出通道维度和输入通道维度
constexpr int weight_output_channels_dim = 0;
constexpr int weight_input_channels_dim = 1;

// 常用写法为 2 + max_dim（用于批量大小和通道的额外维度）
constexpr int max_dim = 3;

// ---------------------------------------------------------------------
//
// Checking
//
// ---------------------------------------------------------------------

// 在 pad、stride 和 dilation 上使用的检查函数
static void check_args(CheckedFrom c, IntArrayRef args, size_t expected_size, const char* arg_name)
{
  // 检查参数个数是否符合预期
  TORCH_CHECK(args.size() <= expected_size,
           "Too many ", arg_name, " values (", args.size(), ") supplied, expecting ",
           expected_size, " (while checking arguments for ", c, ")");
  TORCH_CHECK(args.size() >= expected_size,
           "Not enough ", arg_name, " values (", args.size(), ") supplied, expecting ",
           expected_size, " (while checking arguments for ", c, ")");

  // 检查是否有负值
  auto num_negative_values = std::count_if(args.begin(), args.end(), [](int x){return x < 0;});
  if (num_negative_values > 0){
    std::stringstream ss;
    ss << arg_name << " should be greater than zero but got (";
    std::copy(args.begin(), args.end() - 1, std::ostream_iterator<int>(ss,", "));
    ss << args.back() <<  ")" << " (while checking arguments for " << c << ")";
    AT_ERROR(ss.str());
  }
}

// NOTE [ Convolution checks ]
//
// 注意：对于许多调用点，不严格需要检查所有这些关系（例如，对于前向卷积，我们计算
//
// 进行卷积操作前的形状检查，确保输入、权重和输出的尺寸匹配和有效性
inline void convolution_shape_check(
    CheckedFrom c,
    const TensorGeometryArg& input, const TensorGeometryArg& weight, const TensorGeometryArg& output,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups)
{
  // 检查填充尺寸是否合法
  check_args(c, padding, input->dim() - 2, "padding");
  // 检查步幅尺寸是否合法，与填充尺寸相匹配
  check_args(c, stride, padding.size(), "stride");
  // 检查膨胀尺寸是否合法，与填充尺寸相匹配
  check_args(c, dilation, padding.size(), "dilation");

  // 检查输入张量的维度范围是否在3到6之间（不包含6）
  checkDimRange(c, input, 3, 6 /* exclusive */);
  // 检查输入张量的通道维度是否与权重张量的通道维度相匹配
  checkSize_symint(c, input, input_channels_dim, weight->size(1) * groups);

  // 检查输入张量与权重张量的维度是否一致
  checkSameDim(c, input, weight);

  // TODO: 检查输出张量的尺寸是否与指定的output_sizes匹配
  // TODO: 检查权重张量是否与输出张量的尺寸匹配
  // 检查输入张量与输出张量的维度是否一致
  checkSameDim(c, input, output);
}

// 注意：conv_output_size和conv_input_size不是双射关系，
// 因为conv_output_size会丢失信息；这就是为什么conv_input_size
// 需要额外的output_padding参数来消除歧义。
// 计算卷积操作的输出尺寸，输入输出尺寸会根据填充、步幅和（可选的）膨胀进行计算
template <typename T>
inline std::vector<T> _conv_output_size(
    ArrayRef<T> input_size, ArrayRef<T> weight_size,
    ArrayRef<T> padding, ArrayRef<T> stride, ArrayRef<T> dilation = ArrayRef<T>()
) {
  // ASSERT(input_size.size() > 2)
  // ASSERT(input_size.size() == weight_size.size())
  bool has_dilation = !dilation.empty();
  auto dim = input_size.size();
  std::vector<T> output_size(dim);
  // 计算输出尺寸的每一个维度
  output_size[0] = input_size[input_batch_size_dim];
  output_size[1] = weight_size[weight_output_channels_dim];
  for (const auto d : c10::irange(2, dim)) {
    auto dilation_ = has_dilation ? dilation[d - 2] : 1;
    auto kernel = dilation_ * (weight_size[d] - 1) + 1;
    // 根据公式计算每个维度的输出大小
    output_size[d] = (input_size[d] + (2 * padding[d - 2]) - kernel) / stride[d - 2] + 1;
  }
  return output_size;
}

// 计算卷积操作的输出尺寸（整数类型）
inline std::vector<int64_t> conv_output_size(
    IntArrayRef input_size, IntArrayRef weight_size,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation = IntArrayRef()
) {
  return _conv_output_size(input_size, weight_size, padding, stride, dilation);
}

// 计算卷积操作的输出尺寸（符号整数类型）
inline std::vector<c10::SymInt> conv_output_size(
    SymIntArrayRef input_size, SymIntArrayRef weight_size,
    SymIntArrayRef padding, SymIntArrayRef stride, SymIntArrayRef dilation = SymIntArrayRef()
) {
  return _conv_output_size(input_size, weight_size, padding, stride, dilation);
}

template <typename T>
std::vector<T> _conv_input_size(
    // 定义模板类型为 T 的数组引用 output_size，用于存储输出尺寸
    ArrayRef<T> output_size,
    // 定义模板类型为 T 的数组引用 weight_size，用于存储权重尺寸
    ArrayRef<T> weight_size,
    // 定义模板类型为 T 的数组引用 padding，用于存储填充大小
    ArrayRef<T> padding,
    // 定义模板类型为 T 的数组引用 output_padding，用于存储输出填充大小
    ArrayRef<T> output_padding,
    // 定义模板类型为 T 的数组引用 stride，用于存储步长
    ArrayRef<T> stride,
    // 定义模板类型为 T 的数组引用 dilation，用于存储膨胀大小
    ArrayRef<T> dilation,
    // 定义类型为 T 的 groups，用于存储分组数
    T groups
// 定义一个函数，计算卷积层输入的大小
template <typename T>
std::vector<T> _conv_input_size(
    ArrayRef<T> output_size, ArrayRef<T> weight_size,
    ArrayRef<T> padding, ArrayRef<T> output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups
) {
  // 获取输入维度的数量
  auto dim = output_size.size();
  // 创建一个大小为 dim 的向量 input_size，用于存储计算得到的输入大小
  std::vector<T> input_size(dim);
  // 计算输入的 batch size 维度
  input_size[0] = output_size[output_batch_size_dim];
  // 计算输入的通道数维度
  input_size[1] = weight_size[weight_input_channels_dim] * groups;
  // 对于从第二维开始的每一个维度 d
  for (const auto d : c10::irange(2, dim)) {
    // 计算卷积核的大小
    auto kernel = (weight_size[d] - 1) * dilation[d - 2] + 1;
    // 计算输入在维度 d 上的大小
    input_size[d] = (output_size[d] - 1) * stride[d - 2] - (padding[d - 2] * 2) +
                     kernel + output_padding[d - 2];
  }
  // 返回计算得到的输入大小向量
  return input_size;
}

// 定义一个函数，以符号表示（symbolic representation）计算卷积层输入的大小
inline std::vector<c10::SymInt> conv_input_size(
    SymIntArrayRef output_size, SymIntArrayRef weight_size,
    SymIntArrayRef padding, SymIntArrayRef output_padding, SymIntArrayRef stride, SymIntArrayRef dilation, c10::SymInt groups
) {
  // 调用模板函数 _conv_input_size，以符号表示的参数
  return _conv_input_size(output_size, weight_size, padding, output_padding, stride, dilation, groups);
}

// 定义一个函数，以整数表示计算卷积层输入的大小
inline std::vector<int64_t> conv_input_size(
    IntArrayRef output_size, IntArrayRef weight_size,
    IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups
) {
  // 调用模板函数 _conv_input_size，以整数表示的参数
  return _conv_input_size(output_size, weight_size, padding, output_padding, stride, dilation, groups);
}

// 定义一个模板函数，计算卷积层权重的大小
template <typename T>
std::vector<T> _conv_weight_size(
    ArrayRef<T> input_size, ArrayRef<T> output_size,
    ArrayRef<T> padding, ArrayRef<T> output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups
) {
  // 获取输入维度的数量
  auto dim = input_size.size();
  // 创建一个大小为 dim 的向量 weight_size，用于存储计算得到的权重大小
  std::vector<T> weight_size(dim);
  // 计算权重的输出通道数维度
  weight_size[0] = output_size[1];
  // 计算权重的输入通道数维度
  weight_size[1] = input_size[1] / groups;
  // 对于从第二维开始的每一个维度 d
  for (const auto d : c10::irange(2, dim)) {
    // 计算卷积核的大小
    auto kernel = input_size[d] - (output_size[d] - 1) * stride[d - 2]
               + padding[d - 2] * 2 - output_padding[d - 2];
    // 计算权重在维度 d 上的大小
    weight_size[d] = (kernel - 1) / dilation[d - 2] + 1;
  }
  // 返回计算得到的权重大小向量
  return weight_size;
}

// 定义一个函数，以符号表示计算卷积层权重的大小
inline std::vector<c10::SymInt> conv_weight_size(
    SymIntArrayRef input_size, SymIntArrayRef output_size,
    SymIntArrayRef padding, SymIntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups
) {
  // 调用模板函数 _conv_weight_size，以符号表示的参数
  return _conv_weight_size(input_size, output_size, padding, output_padding, stride, dilation, groups);
}

// 定义一个函数，以整数表示计算卷积层权重的大小
inline std::vector<int64_t> conv_weight_size(
    IntArrayRef input_size, IntArrayRef output_size,
    IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups
) {
  // 调用模板函数 _conv_weight_size，以整数表示的参数
  return _conv_weight_size(input_size, output_size, padding, output_padding, stride, dilation, groups);
}

// 定义一个函数，重新调整偏置向量的形状
inline Tensor reshape_bias(int64_t dim, const Tensor& bias) {
  // 创建一个形状为 dim 的全一向量 shape，用于重新调整偏置的形状
  std::vector<int64_t> shape(dim, 1);
  // 将第二维的大小设置为 -1，保留原始形状
  shape[1] = -1;
  // 返回重新调整形状后的偏置向量
  return bias.reshape(shape);
}

// 定义一个函数，推荐 CUDNN 卷积操作的内存格式
inline at::MemoryFormat cudnn_conv_suggest_memory_format(const at::Tensor& input, const at::Tensor& weight) {
  // 如果未编译 CUDNN 或输入或权重是双精度类型，则禁用 NHWC 内存格式
  if (!at::detail::getCUDAHooks().compiledWithCuDNN() ||
      input.scalar_type() == at::kDouble ||
      weight.scalar_type() == at::kDouble) {
    // 返回一个指示张量存储顺序的内存格式，这里指的是连续内存格式
    return at::MemoryFormat::Contiguous;
      }
    // 获取当前运行的 cuDNN 版本号
      long cudnn_version = at::detail::getCUDAHooks().versionCuDNN();
    // 推测输入张量的内存格式
      auto input_memory_format = input.suggest_memory_format();
    // 推测权重张量的内存格式
      auto weight_memory_format = weight.suggest_memory_format();
    // 获取权重张量的维度数
      auto weight_ndim = weight.ndimension();
    
    // 检查是否可以使用 cuDNN 的二维通道最后内存格式
      bool can_use_cudnn_channels_last_2d = (cudnn_version >= 7603) && (weight_ndim == 4) && (
        (input_memory_format  == at::MemoryFormat::ChannelsLast) ||
        (weight_memory_format == at::MemoryFormat::ChannelsLast)
      );
    // 如果可以使用二维通道最后内存格式，则返回该内存格式
      if (can_use_cudnn_channels_last_2d) {
        return at::MemoryFormat::ChannelsLast;
      }
    
    // 检查是否可以使用 cuDNN 的三维通道最后内存格式
      bool can_use_cudnn_channels_last_3d = (cudnn_version >= 8005) && (weight_ndim == 5) && (
        (input_memory_format  == at::MemoryFormat::ChannelsLast3d) ||
        (weight_memory_format == at::MemoryFormat::ChannelsLast3d)
      );
    // 如果可以使用三维通道最后内存格式，则返回该内存格式
      if (can_use_cudnn_channels_last_3d) {
        return at::MemoryFormat::ChannelsLast3d;
      }
    
    // 默认情况下，返回连续内存格式
      return at::MemoryFormat::Contiguous;
}

// 结束一个C++语言的函数定义

// 设置是否在cudnn卷积基准测试后调用emptyCache
TORCH_API void _cudnn_set_conv_benchmark_empty_cache(bool enable);
// 获取当前是否在cudnn卷积基准测试后调用emptyCache的设置
TORCH_API bool _cudnn_get_conv_benchmark_empty_cache();

// 检查是否可以使用miopen的通道最后布局（NHWC）的2D版本
inline bool miopen_conv_use_channels_last(const at::Tensor& input, const at::Tensor& weight) {

  // 如果未编译进MIOpen，或者输入或权重是双精度类型，则禁用NHWC布局
  if (!at::detail::getCUDAHooks().compiledWithMIOpen() ||
      input.scalar_type() == at::kDouble ||
      weight.scalar_type() == at::kDouble) {
    return false;
  }

  bool can_use_miopen_channels_last_2d = false;
  // TODO: 在ROCm正式支持MIOpen中的NHWC时移除PYTORCH_MIOPEN_SUGGEST_NHWC
  // 参见issue #64427
  static std::optional<bool> PYTORCH_MIOPEN_SUGGEST_NHWC = c10::utils::check_env("PYTORCH_MIOPEN_SUGGEST_NHWC");

  auto input_memory_format = input.suggest_memory_format();
  auto weight_memory_format = weight.suggest_memory_format();

  // 检查是否可以使用miopen的通道最后布局的2D版本
  can_use_miopen_channels_last_2d = PYTORCH_MIOPEN_SUGGEST_NHWC && *PYTORCH_MIOPEN_SUGGEST_NHWC && (
            (input_memory_format  == at::MemoryFormat::ChannelsLast) ||
            (weight_memory_format == at::MemoryFormat::ChannelsLast)
        );

  bool can_use_miopen_channels_last_3d = false;

  return can_use_miopen_channels_last_2d || can_use_miopen_channels_last_3d;
}

// 检查是否可以使用Mkldnn的通道最后布局（NHWC）的版本
inline bool mkldnn_conv_use_channels_last(const at::Tensor& input, const at::Tensor& weight) {

  // 如果输入或权重是双精度类型，则禁用NHWC布局
  if (input.scalar_type() == at::kDouble ||
      weight.scalar_type() == at::kDouble) {
    return false;
  }

  // 如果输入或权重是MkldnnCPU tensor，则禁用NHWC布局
  if (input.is_mkldnn() || weight.is_mkldnn()) {
    return false;
  }

  auto input_memory_format = input.suggest_memory_format();
  auto weight_memory_format = weight.suggest_memory_format();

  // 检查是否可以使用Mkldnn的通道最后布局的2D版本
  bool can_use_mkldnn_channels_last_2d =
      (input_memory_format  == at::MemoryFormat::ChannelsLast) ||
      (weight_memory_format == at::MemoryFormat::ChannelsLast);

  // 检查是否可以使用Mkldnn的通道最后布局的3D版本
  bool can_use_mkldnn_channels_last_3d =
      (input_memory_format  == at::MemoryFormat::ChannelsLast3d) ||
      (weight_memory_format == at::MemoryFormat::ChannelsLast3d);

  return can_use_mkldnn_channels_last_2d || can_use_mkldnn_channels_last_3d;
}

// 检查是否可以使用THNN的通道最后布局（NHWC）的2D版本
inline bool thnn_conv_use_channels_last(const at::Tensor& input, const at::Tensor& weight) {

  auto input_memory_format = input.suggest_memory_format();
  auto weight_memory_format = weight.suggest_memory_format();

  // 只有在CPU设备上检查THNN的通道最后布局的2D版本
  bool can_use_thnn_channels_last_2d = input.device().is_cpu() && (
      (input_memory_format  == at::MemoryFormat::ChannelsLast) || (
       weight_memory_format == at::MemoryFormat::ChannelsLast));

  return can_use_thnn_channels_last_2d;
}

// 检查是否可以使用XPU的通道最后布局（NHWC）
inline bool xpu_conv_use_channels_last(const at::Tensor& input, const at::Tensor& weight) {

  // 只检查XPU tensor的布局
  if (!input.is_xpu() || !weight.is_xpu()) {
    return false;
  }

  // 如果输入或权重是双精度类型，则禁用NHWC布局
  if (input.scalar_type() == at::kDouble ||
      weight.scalar_type() == at::kDouble) {
    # 如果输入为空，直接返回 false
    return false;
  }

  # 推测输入张量的内存格式
  auto input_memory_format = input.suggest_memory_format();
  # 推测权重张量的内存格式
  auto weight_memory_format = weight.suggest_memory_format();

  # 检查是否可以在 XPU 上使用 ChannelsLast 的二维内存格式
  bool can_use_xpu_channels_last_2d =
      (input_memory_format  == at::MemoryFormat::ChannelsLast) ||
      (weight_memory_format == at::MemoryFormat::ChannelsLast);

  # 检查是否可以在 XPU 上使用 ChannelsLast 的三维内存格式
  bool can_use_xpu_channels_last_3d =
      (input_memory_format  == at::MemoryFormat::ChannelsLast3d) ||
      (weight_memory_format == at::MemoryFormat::ChannelsLast3d);

  # 返回是否可以在 XPU 上使用 ChannelsLast 的二维或三维内存格式
  return can_use_xpu_channels_last_2d || can_use_xpu_channels_last_3d;
}

} // namespace at::native
```