# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnormalization.cpp`

```
// 引入 ATen 库中的头文件
#include <ATen/core/Tensor.h>
#include <ATen/native/layer_norm.h>
#include <ATen/native/quantized/cpu/QuantizedOps.h>
#include <ATen/Parallel.h>
#include <c10/util/accumulate.h>
#include <torch/library.h>

// 根据情况选择是否包含特定的 ATen 函数头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/_empty_affine_quantized.h>
#endif

// 引入标准库文件
#include <algorithm>
#include <vector>

// 定义 at 命名空间下的 native 命名空间
namespace at {
namespace native {

// 定义 quantized_normalize_stub 和 quantized_groupnorm_nhwc_stub 的分发器
DEFINE_DISPATCH(quantized_normalize_stub);
DEFINE_DISPATCH(quantized_groupnorm_nhwc_stub);

// 实现量化层归一化的函数，返回量化后的张量
static Tensor quantized_layer_norm_impl(
    const Tensor& input,
    IntArrayRef normalized_shape,
    const Tensor& weight /* optional */,
    const Tensor& bias /* optional */,
    double eps,
    double output_scale,
    int64_t output_zero_point) {

  // 检查并获取 LayerNorm 的输入维度
  auto M_N = _check_layer_norm_inputs(input, normalized_shape, weight, bias);
  auto M = M_N.first; // 获取 batch size
  auto N = M_N.second; // 获取特征维度
  auto X = input.expect_contiguous(); // 确保输入张量是连续的
  auto gamma = weight.expect_contiguous(); // 确保权重张量是连续的
  auto beta = bias.expect_contiguous(); // 确保偏置张量是连续的

  // 创建一个与输入张量相同大小的量化空张量 Y
  Tensor Y = at::_empty_affine_quantized(
    X->sizes(),
    X->scalar_type(),
    output_scale,
    output_zero_point,
    X->suggest_memory_format());

  // 如果 batch size M 大于 0，则执行量化归一化操作
  if (M > 0) {
    bool affine_per_channel = false;
    int num_channels = 1; // 对于 LayerNorm 不相关
    int num_groups = 1; // 对于 LayerNorm 不相关
    quantized_normalize_stub(kCPU, *X, *gamma, *beta, affine_per_channel,
        num_channels, num_groups, M, N, eps, &Y);
  }
  return Y; // 返回量化后的张量 Y
}

// 实现量化组归一化的函数，返回量化后的张量
static Tensor quantized_group_norm_impl(
    const Tensor& qx,
    int64_t num_groups,
    const Tensor& weight, // 可选的权重张量
    const Tensor& bias, // 可选的偏置张量
    double eps,
    double output_scale,
    int64_t output_zero_point) {

  // 检查输入是否按照 ChannelsLast 格式连续
  bool is_channels_last = qx.is_contiguous(c10::MemoryFormat::ChannelsLast);
  auto mem_layout = is_channels_last ? c10::MemoryFormat::ChannelsLast :
                                       c10::MemoryFormat::Contiguous;

  // 确保输入张量 qx、权重张量 weight 和偏置张量 bias 是连续的
  const auto& qx_contig = qx.contiguous(mem_layout);
  const auto& weight_contig = weight.contiguous();
  const auto& bias_contig = bias.contiguous();

  // 检查输入张量 qx 的维度是否符合预期
  const auto input_ndim = qx_contig.dim();
  TORCH_CHECK(
      input_ndim >= 3,
      "Expected normalized_shape to be at least 3-dimensional");
  TORCH_CHECK(num_groups > 0, "Expected num_groups to be positive");

  // 检查输入张量 qx 的形状，确保通道数能够被 num_groups 整除
  const auto input_shape = qx_contig.sizes();
  TORCH_CHECK(input_shape[1] % num_groups == 0,
      "Expected channels to be divisible by groups");

  // 计算批次数、通道数和每批次元素数
  const int64_t batches = input_shape[0];
  const int64_t num_channels = input_shape[1];
  const int64_t elements_per_batch =
      c10::multiply_integers(input_shape.cbegin() + 1, input_shape.cend());

  // 计算 M 和 N
  const int64_t M = batches * num_groups;
  const int64_t N = elements_per_batch / num_groups;

  // 创建一个与输入张量 qx 相同大小的量化空张量 Y
  Tensor Y = at::_empty_affine_quantized(
    qx_contig.sizes(),
    qx_contig.scalar_type(),
    output_scale,
    output_zero_point,
    qx_contig.suggest_memory_format());

  // 如果 M 大于 0，则执行量化组归一化操作
  if (M > 0) {
    bool affine_per_channel = true;
    // 调用 quantized_groupnorm_nhwc_stub 进行量化组归一化操作
    quantized_groupnorm_nhwc_stub(kCPU, qx_contig, weight_contig, bias_contig,
        affine_per_channel, num_channels, num_groups, M, N, eps, &Y);
  }
  return Y; // 返回量化后的张量 Y
}

} // namespace native
} // namespace at
    # 如果输入数据是按通道最后（channels_last）的顺序排列
    if (is_channels_last) {
      # 调用量化组归一化的函数，处理按通道最后排列的输入数据
      quantized_groupnorm_nhwc_stub(kCPU, qx_contig, weight_contig, bias_contig,
          affine_per_channel, num_channels, num_groups, M, N, eps, &Y);
    } else {
      # 如果输入数据不是按通道最后（channels_last）的顺序排列
      # 调用量化归一化的函数，处理其他排列方式的输入数据
      quantized_normalize_stub(kCPU, qx_contig, weight_contig, bias_contig,
          affine_per_channel, num_channels, num_groups, M, N, eps, &Y);
    }
  }
  # 返回处理后的输出 Y
  return Y;
// 闭合前一个命名空间 'native'，表示该命名空间的定义结束
} // namespace native

// 闭合前一个命名空间 'at'，表示该命名空间的定义结束
} // namespace at
```