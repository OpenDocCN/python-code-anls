# `.\pytorch\aten\src\ATen\native\quantized\cpu\Normalization.cpp`

```
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 定义编译时宏，用于仅包含方法操作符

#include <ATen/core/Tensor.h>
// 包含 ATen 核心张量类头文件

#include <ATen/Parallel.h>
// 包含 ATen 并行处理的头文件

#include <torch/library.h>
// 包含 Torch 库的头文件

#include <ATen/native/quantized/cpu/QuantizedOps.h>
// 包含 ATen 量化操作的 CPU 实现头文件

#include <c10/util/irange.h>
// 包含 c10 实用工具中的迭代范围头文件

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_empty_affine_quantized.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/quantized_batch_norm_native.h>
#endif
// 根据编译时宏条件编译不同的 ATen 操作头文件

#include <algorithm>
// 包含算法标准库头文件

namespace at {
namespace native {

DEFINE_DISPATCH(qbatch_norm_stub);
// 定义量化批归一化分发函数调度器

DEFINE_DISPATCH(qbatch_norm_relu_stub);
// 定义量化带 ReLU 的批归一化分发函数调度器

namespace {
void compute_fused_params(
    const int64_t channels,
    const float* weight_data,
    const float* bias_data,
    const float* mean_data,
    const float* var_data,
    double eps,
    double input_scale,
    double output_scale,
    float* alpha_data,
    float* beta_data) {
  // 计算融合参数
  // 批归一化公式：
  // 输出(n, c, h, w) = (输入(n, c, h, w) - 均值(c)) / sqrt(方差(c) + eps) * 权重(c) + 偏置(c)
  // 我们将 inv_sigma(c) = 1 / sqrt(方差(c) + eps) 分解出来。
  for (const auto c : c10::irange(channels)) {
    // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
    float inv_sigma = 1.0 / std::sqrt(var_data[c] + static_cast<float>(eps));
    float weight_v = weight_data ? weight_data[c] : 1;
    float bias_v = bias_data ? bias_data[c] : 0;
    // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
    alpha_data[c] = inv_sigma * weight_v * (input_scale / output_scale);
    // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
    beta_data[c] = (bias_v - mean_data[c] * inv_sigma * weight_v) / output_scale;
  }
}

template <bool ReluFused>
Tensor q_batch_norm1d_impl(
    Tensor qx,
    std::optional<Tensor> mb_weight,
    std::optional<Tensor> mb_bias,
    Tensor mean,
    Tensor var,
    double eps,
    double output_scale,
    int64_t output_zero_point) {

  TORCH_CHECK(mb_weight.has_value(), "Weight must be provided");
  TORCH_CHECK(mb_bias.has_value(), "Bias must be provided");
  const auto& weight = *mb_weight;
  const auto& bias = *mb_bias;

  if (qx.numel() == 0) {
    auto out = qx.clone();
  return out;
}
// 获取输入张量的维度数
int64_t ndim = qx.dim();
// 检查张量的维度是否为2或3
TORCH_CHECK(ndim == 2 || ndim == 3, "Expecting the input tensor of rank 2 or 3.");
// 获取张量的大小
const int64_t N = qx.size(0);
const int64_t C = qx.size(1);
// 如果维度为3，则获取第三个维度的大小，否则设置为1
const int64_t H = ndim == 3 ? qx.size(2) : 1;

// 检查权重张量的元素数量是否与C相等
TORCH_CHECK(weight.numel() == C, "Expect weight size to match C");
// 检查偏置张量的元素数量是否与C相等
TORCH_CHECK(bias.numel() == C, "Expect weight size to match C");

// 获取权重张量和偏置张量的数据指针
const float* weight_data = weight.template const_data_ptr<float>();
const float* bias_data = bias.template const_data_ptr<float>();

// 检查均值张量的元素数量是否与C相等
TORCH_CHECK(mean.numel() == C, "Mean size must match channel dimension");
// 检查方差张量的元素数量是否与C相等
TORCH_CHECK(var.numel() == C, "Variance size must match channel dimension");

// 创建与均值张量相同大小的空张量alpha和beta
Tensor alpha = at::empty_like(mean, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
Tensor beta = at::empty_like(mean, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
// 获取alpha和beta的可变数据指针
float* alpha_data = alpha.mutable_data_ptr<float>();
float* beta_data = beta.data_ptr<float>();

// 获取均值和方差张量的数据指针
const float* mean_data = mean.template const_data_ptr<float>();
const float* var_data = var.template const_data_ptr<float>();

// 如果维度为2，则在最后两个维度上扩展张量qx，以便使用NHWC格式
if (ndim == 2) {
  qx = qx.unsqueeze(-1).unsqueeze(-1);
} else {
  // 如果维度为3，则在最后一个维度上扩展张量qx，以便使用NHWC格式
  qx = qx.unsqueeze(-1);
}

// 获取扩展后张量qx的尺寸
auto oSizes = qx.sizes();
// 将qx转换为ChannelsLast内存格式的连续张量qx_nhwc
auto qx_nhwc = qx.contiguous(MemoryFormat::ChannelsLast);
// 创建与qx_nhwc相同尺寸的量化空张量qy
Tensor qy = at::_empty_affine_quantized(
    oSizes,
    at::device(kCPU)
      .dtype(qx_nhwc.scalar_type())
      .memory_format(MemoryFormat::ChannelsLast),
    output_scale,
    output_zero_point,
    c10::nullopt);

// 计算融合参数
compute_fused_params(
    C,
    weight_data,
    bias_data,
    mean_data,
    var_data,
    eps,
    qx.q_scale(),
    output_scale,
    alpha_data,
    beta_data);

// 如果启用ReLU融合，则调用qbatch_norm_relu_stub；否则调用qbatch_norm_stub
if (ReluFused) {
  qbatch_norm_relu_stub(
      qx.device().type(),
      N,
      C,
      H,
      qx.q_zero_point(),
      output_zero_point,
      qx_nhwc,
      alpha,
      beta,
      qy);
} else {
  qbatch_norm_stub(
      qx.device().type(),
      N,
      C,
      H,
      qx.q_zero_point(),
      output_zero_point,
      qx_nhwc,
      alpha,
      beta,
      qy);
}

// 移除虚拟的维度，并返回到连续格式（因为没有第4个通道）。注意，这会带来性能成本。
// 如果维度为2，则在最后一个维度上挤压张量result
Tensor result = qy.contiguous(MemoryFormat::Contiguous).squeeze(-1);
if (ndim == 2) {
  result = result.squeeze(-1);
}
// 返回结果张量
return result;
}

template <bool ReluFused>
// 实现二维批量归一化的量化版本
Tensor q_batch_norm2d_impl(
    Tensor qx,  // 输入量化张量
    std::optional<Tensor> mb_weight,  // 可选的权重张量
    std::optional<Tensor> mb_bias,    // 可选的偏置张量
    Tensor mean,   // 均值张量
    Tensor var,    // 方差张量
    double eps,    // 用于数值稳定性的小值
    double output_scale,  // 输出量化的比例因子
    int64_t output_zero_point) {  // 输出量化的零点偏移

  TORCH_CHECK(mb_weight.has_value(), "Weight must be provided");  // 检查权重是否提供
  TORCH_CHECK(mb_bias.has_value(), "Bias must be provided");      // 检查偏置是否提供
  const auto& weight = *mb_weight;  // 获取权重引用
  const auto& bias = *mb_bias;      // 获取偏置引用

  if (qx.numel() == 0) {
    auto out = qx.clone();  // 若输入为空，克隆输入并返回
    return out;
  }
  int64_t ndim = qx.dim();  // 获取输入张量的维度数
  TORCH_CHECK(ndim == 4, "Expecting the input tensor of rank 4.");  // 检查输入张量是否为四维

  const int64_t N = qx.size(0);  // 批量大小
  const int64_t C = qx.size(1);  // 通道数
  const int64_t H = qx.size(2);  // 高度
  const int64_t W = qx.size(3);  // 宽度

  TORCH_CHECK(weight.numel() == C, "Expect weight size to match C");  // 检查权重大小是否与通道数匹配
  TORCH_CHECK(bias.numel() == C, "Expect weight size to match C");     // 检查偏置大小是否与通道数匹配

  // 获取权重和偏置的数据指针
  const float* weight_data = weight.template const_data_ptr<float>();
  const float* bias_data = bias.template const_data_ptr<float>();

  TORCH_CHECK(mean.numel() == C, "Mean size must match channel dimension");  // 检查均值大小是否与通道数匹配
  TORCH_CHECK(var.numel() == C, "Variance size must match channel dimension");  // 检查方差大小是否与通道数匹配

  // 创建与均值相同大小的空张量 alpha 和 beta
  Tensor alpha = at::empty_like(mean, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  Tensor beta = at::empty_like(mean, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  float* alpha_data = alpha.mutable_data_ptr<float>();  // 获取 alpha 的可变数据指针
  float* beta_data = beta.data_ptr<float>();            // 获取 beta 的数据指针

  // 获取均值和方差的数据指针
  const float* mean_data = mean.template const_data_ptr<float>();
  const float* var_data = var.template const_data_ptr<float>();

  auto oSizes = qx.sizes();  // 获取输出张量的大小
  auto qx_nhwc = qx.contiguous(MemoryFormat::ChannelsLast);  // 将输入张量转换为 Channels Last 格式
  // 创建一个空的仿射量化张量 qy
  Tensor qy = at::_empty_affine_quantized(
      oSizes,
      at::device(kCPU)
        .dtype(qx_nhwc.scalar_type())
        .memory_format(MemoryFormat::ChannelsLast),
      output_scale,
      output_zero_point,
      c10::nullopt);

  // 计算融合参数：alpha 和 beta
  compute_fused_params(
      C,
      weight_data,
      bias_data,
      mean_data,
      var_data,
      eps,
      qx.q_scale(),  // 输入量化的比例因子
      output_scale,
      alpha_data,
      beta_data);

  if (ReluFused) {
    // 如果融合了 ReLU 操作，调用带 ReLU 的量化批量归一化函数
    qbatch_norm_relu_stub(
        qx.device().type(),
        N,
        C,
        H * W,
        qx.q_zero_point(),  // 输入量化的零点偏移
        output_zero_point,
        qx_nhwc,
        alpha,
        beta,
        qy);
  } else {
    // 否则，调用普通的量化批量归一化函数
    qbatch_norm_stub(
        qx.device().type(),
        N,
        C,
        H * W,
        qx.q_zero_point(),  // 输入量化的零点偏移
        output_zero_point,
        qx_nhwc,
        alpha,
        beta,
        qy);
  }
  return qy;  // 返回量化后的输出张量
}

template <bool ReluFused>
// 实现三维批量归一化的量化版本
Tensor q_batch_norm3d_impl(
    Tensor qx,  // 输入量化张量
    std::optional<Tensor> mb_weight,  // 可选的权重张量
    std::optional<Tensor> mb_bias,    // 可选的偏置张量
    Tensor mean,   // 均值张量
    Tensor var,    // 方差张量
    double eps,    // 用于数值稳定性的小值
    double output_scale,  // 输出量化的比例因子
    int64_t output_zero_point) {  // 输出量化的零点偏移

  TORCH_CHECK(mb_weight.has_value(), "Weight must be provided");  // 检查权重是否提供
  TORCH_CHECK(mb_bias.has_value(), "Bias must be provided");      // 检查偏置是否提供

  const auto& weight = *mb_weight;  // 获取权重引用
  const auto& bias = *mb_bias;      // 获取偏置引用

  if (qx.numel() == 0) {
    // 若输入张量为空，直接返回一个克隆的空张量
    auto out = qx.clone();
    return out;
  }
    // 将输入张量 qx 克隆到 out 中
    auto out = qx.clone();
    // 返回克隆后的张量 out
    return out;
  }
  
  // 获取输入张量 qx 的维度数 ndim
  int64_t ndim = qx.dim();
  // 检查输入张量 qx 是否为五维，否则抛出错误信息
  TORCH_CHECK(ndim == 5, "Expecting the input tensor of rank 5.");
  // 获取输入张量 qx 的各个维度尺寸，并赋值给常量变量 N, C, D, H, W
  const int64_t N = qx.size(0);
  const int64_t C = qx.size(1);
  const int64_t D = qx.size(2);
  const int64_t H = qx.size(3);
  const int64_t W = qx.size(4);

  // 检查权重张量 weight 的元素数是否与 C 相等，否则抛出错误信息
  TORCH_CHECK(weight.numel() == C, "Expect weight size to match C");
  // 检查偏置张量 bias 的元素数是否与 C 相等，否则抛出错误信息
  TORCH_CHECK(bias.numel() == C, "Expect weight size to match C");

  // 获取权重张量 weight 的数据指针，并转换为 float 类型的指针
  const float* weight_data = weight.template const_data_ptr<float>();
  // 获取偏置张量 bias 的数据指针，并转换为 float 类型的指针
  const float* bias_data = bias.template const_data_ptr<float>();

  // 检查均值张量 mean 的元素数是否与 C 相等，否则抛出错误信息
  TORCH_CHECK(mean.numel() == C, "Mean size must match channel dimension");
  // 检查方差张量 var 的元素数是否与 C 相等，否则抛出错误信息
  TORCH_CHECK(var.numel() == C, "Variance size must match channel dimension");

  // 创建与均值张量 mean 相同形状的空张量 alpha
  Tensor alpha = at::empty_like(mean, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  // 创建与均值张量 mean 相同形状的空张量 beta
  Tensor beta = at::empty_like(mean, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  // 获取 alpha 张量的可变数据指针，并转换为 float 类型的指针
  float* alpha_data = alpha.mutable_data_ptr<float>();
  // 获取 beta 张量的数据指针，并转换为 float 类型的指针
  float* beta_data = beta.data_ptr<float>();

  // 获取均值张量 mean 的数据指针，并转换为 float 类型的指针
  const float* mean_data = mean.template const_data_ptr<float>();
  // 获取方差张量 var 的数据指针，并转换为 float 类型的指针
  const float* var_data = var.template const_data_ptr<float>();

  // 获取输入张量 qx 的大小
  auto oSizes = qx.sizes();
  // 将输入张量 qx 转换为 ChannelsLast3d 内存格式
  auto qx_nhwc = qx.contiguous(MemoryFormat::ChannelsLast3d);
  // 创建一个与 qx_nhwc 相同形状的空张量 qy，用于存储量化后的输出
  Tensor qy = at::_empty_affine_quantized(
      oSizes,
      at::device(kCPU)
        .dtype(qx_nhwc.scalar_type())
        .memory_format(MemoryFormat::ChannelsLast3d),
      output_scale,
      output_zero_point,
      c10::nullopt);

  // 计算融合的 BatchNorm 参数 alpha 和 beta
  compute_fused_params(
      C,
      weight_data,
      bias_data,
      mean_data,
      var_data,
      eps,
      qx.q_scale(),
      output_scale,
      alpha_data,
      beta_data);

  // 如果启用了 ReluFused，则调用 qbatch_norm_relu_stub 函数，否则调用 qbatch_norm_stub 函数
  if (ReluFused) {
    qbatch_norm_relu_stub(
        qx.device().type(),
        N,
        C,
        D * H * W,
        qx.q_zero_point(),
        output_zero_point,
        qx_nhwc,
        alpha,
        beta,
        qy);
  } else {
    qbatch_norm_stub(
        qx.device().type(),
        N,
        C,
        D * H * W,
        qx.q_zero_point(),
        output_zero_point,
        qx_nhwc,
        alpha,
        beta,
        qy);
  }
  // 返回量化后的输出张量 qy
  return qy;
} // namespace

template <bool ReluFused>
// 定义了一个模板函数 q_batch_norm_impl，用于进行量化批归一化操作
Tensor q_batch_norm_impl(
    Tensor qx, // 输入量化张量
    std::optional<Tensor> mb_weight, // 可选的批归一化权重张量
    std::optional<Tensor> mb_bias, // 可选的批归一化偏置张量
    Tensor mean, // 批次的均值张量
    Tensor var, // 批次的方差张量
    double eps, // 用于数值稳定性的 epsilon 值
    double output_scale, // 输出的量化缩放因子
    int64_t output_zero_point // 输出的量化零点
) {
  Tensor qy; // 定义量化后的输出张量
  int64_t dim = qx.dim(); // 获取输入张量的维度
  if (dim == 2 || dim == 3) { // 如果输入是二维或三维的
    // 调用一维量化批归一化实现函数 q_batch_norm1d_impl 进行处理
    qy = q_batch_norm1d_impl<ReluFused>(
        qx, mb_weight, mb_bias, mean, var, eps, output_scale, output_zero_point);
  } else if (dim == 4) { // 如果输入是四维的
    // 调用二维量化批归一化实现函数 q_batch_norm2d_impl 进行处理
    qy = q_batch_norm2d_impl<ReluFused>(
        qx, mb_weight, mb_bias, mean, var, eps, output_scale, output_zero_point);
  } else if (dim == 5) { // 如果输入是五维的
    // 调用三维量化批归一化实现函数 q_batch_norm3d_impl 进行处理
    qy = q_batch_norm3d_impl<ReluFused>(
        qx, mb_weight, mb_bias, mean, var, eps, output_scale, output_zero_point);
  } else {
    // 如果输入维度不支持，则抛出错误信息
    TORCH_CHECK(false, "quantized::batch_norm only support 2d, 3d, 4d or 5d inputs.");
  }
  return qy; // 返回量化后的输出张量
}

} // namespace

// 定义量化批归一化函数 quantized_batch_norm
Tensor quantized_batch_norm(
    const Tensor& qx, // 输入量化张量
    const std::optional<Tensor>& weight_opt /* optional */, // 可选的权重张量
    const std::optional<Tensor>& bias_opt /* optional */, // 可选的偏置张量
    const Tensor& mean /* optional */, // 批次的均值张量（可选）
    const Tensor& var /* optional */, // 批次的方差张量（可选）
    double eps, // 用于数值稳定性的 epsilon 值
    double output_scale, // 输出的量化缩放因子
    int64_t output_zero_point // 输出的量化零点
) {
  // See [Note: hacky wrapper removal for optional tensor]
  // 借用权重张量的可能所有权
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned; // 获取权重张量
  // 根据是否有定义来确定偏置张量
  const Tensor& bias = c10::value_or_else(bias_opt, [] {return Tensor();});

  Tensor qy; // 定义量化后的输出张量
  // 调用二维量化批归一化实现函数 q_batch_norm2d_impl 进行处理
  qy = q_batch_norm2d_impl<false>(
      qx,
      weight.defined() ? c10::make_optional(weight) : c10::nullopt,
      bias.defined() ? c10::make_optional(bias) : c10::nullopt,
      mean, var, eps, output_scale, output_zero_point);
  return qy; // 返回量化后的输出张量
}

// 注册量化批归一化的实现函数
TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("quantized::batch_norm"),        TORCH_FN(q_batch_norm_impl<false>)); // 非融合 ReLU 版本
  m.impl(TORCH_SELECTIVE_NAME("quantized::batch_norm_relu"),   TORCH_FN(q_batch_norm_impl<true>)); // 融合 ReLU 版本
  m.impl(TORCH_SELECTIVE_NAME("quantized::batch_norm1d"),      TORCH_FN(q_batch_norm1d_impl<false>));
  m.impl(TORCH_SELECTIVE_NAME("quantized::batch_norm1d_relu"), TORCH_FN(q_batch_norm1d_impl<true>));
  m.impl(TORCH_SELECTIVE_NAME("quantized::batch_norm2d"),      TORCH_FN(q_batch_norm2d_impl<false>));
  m.impl(TORCH_SELECTIVE_NAME("quantized::batch_norm2d_relu"), TORCH_FN(q_batch_norm2d_impl<true>));
  m.impl(TORCH_SELECTIVE_NAME("quantized::batch_norm3d"),      TORCH_FN(q_batch_norm3d_impl<false>));
  m.impl(TORCH_SELECTIVE_NAME("quantized::batch_norm3d_relu"), TORCH_FN(q_batch_norm3d_impl<true>));
}

} // namespace native
} // namespace at
```