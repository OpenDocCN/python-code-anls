# `.\pytorch\aten\src\ATen\native\transformers\transformer.cpp`

```py
// 定义预处理宏，仅使用方法运算符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 包含 ATen 库的 Tensor 类和调度功能
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/NestedTensorImpl.h>

// 引入 Torch 库
#include <torch/library.h>

// 引入 NestedTensor 的转换功能
#include <ATen/native/nested/NestedTensorTransformerFunctions.h>

// 根据是否定义了 AT_PER_OPERATOR_HEADERS，选择性地包含不同的 ATen 函数头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_addmm_activation.h>
#include <ATen/ops/_native_multi_head_attention.h>
#include <ATen/ops/_transformer_encoder_layer_fwd_native.h>
#include <ATen/ops/addmm.h>
#include <ATen/ops/layer_norm.h>
#endif

namespace at {

namespace native {

// 命名空间下的私有静态函数 linear_for_ffn
namespace {
// 实现 feed-forward 网络的线性变换部分
Tensor linear_for_ffn(
    const Tensor& bias,
    const Tensor& mat1,
    const Tensor& mat2,
    std::optional<bool> use_gelu) {
  // 如果输入 mat1 是嵌套张量，则调用 NestedTensor_times_Tensor_plus_Tensor_addmm 函数
  if (mat1.is_nested()) {
    return NestedTensor_times_Tensor_plus_Tensor_addmm(
        bias, mat1, mat2.t(), 1, 1, use_gelu);
  }

  // 将 mat1 展平成二维张量 mat1_
  auto mat1_ = mat1.view({mat1.sizes()[0] * mat1.sizes()[1], mat1.sizes()[2]});
  Tensor result;
  // 根据是否指定 use_gelu 进行不同的线性变换操作
  if (use_gelu.has_value()) {
    result = at::_addmm_activation(bias, mat1_, mat2.t(), 1, 1, *use_gelu);
  } else {
    result = at::addmm(bias, mat1_, mat2.t());
  }
  // 将结果张量重新视图回原始形状
  return result.view({mat1.sizes()[0], mat1.sizes()[1], -1});
}

// 实现 feed-forward 网络的前向传播
Tensor ffn(
    const Tensor& input,
    const Tensor& w1,
    const Tensor& b1,
    const Tensor& w2,
    const Tensor& b2,
    bool use_gelu,
    bool add_norm) {
  // 检查是否支持 add_norm，目前未实现的功能
  TORCH_CHECK(add_norm == false, "TODO add_norm to be supported in FFN");
  // 检查输入张量的维度是否符合要求
  TORCH_CHECK(input.dim() == 3, "batched input size should be 3");
  // 检查权重张量 w1 和 w2 是否为二维张量
  TORCH_CHECK(w1.dim() == 2, "2d weights expected");
  TORCH_CHECK(w2.dim() == 2, "2d weights expected");
  // 执行线性变换操作
  Tensor res = linear_for_ffn(b1, input, w1, use_gelu);
  // 对结果再次执行线性变换操作
  res = linear_for_ffn(b2, res, w2, c10::nullopt);
  return res;
}

// 实现层归一化操作
Tensor norm(
    const Tensor& input,
    const int64_t embed_dim,
    const double eps,
    const Tensor& weight,
    const Tensor& bias,
    const bool use_nested_tensor) {
  // 调用 ATen 库中的 layer_norm 函数进行归一化处理
  return at::layer_norm(input, {embed_dim}, weight, bias, eps, true);
}

} // namespace

// 实现 Transformer 编码层的前向传播
Tensor transformer_encoder_layer_forward(
    const Tensor& src,
    const int64_t embed_dim,
    const int64_t num_heads,
    const Tensor& qkv_weight,
    const Tensor& qkv_bias,
    const Tensor& proj_weight,
    const Tensor& proj_bias,
    const bool use_gelu,
    const bool norm_first,
    const double layer_norm_eps,
    const Tensor& layer_norm_weight_1,
    const Tensor& layer_norm_bias_1,
    const Tensor& layer_norm_weight_2,
    const Tensor& layer_norm_bias_2,
    const Tensor& ffn_weight_1,
    const Tensor& ffn_bias_1,
    const Tensor& ffn_weight_2,
    const Tensor& ffn_bias_2,
    const std::optional<Tensor>& mask,
    const std::optional<int64_t> mask_type) {
  {
    // 检查是否为嵌套张量，获取相应的内部缓冲区张量
    const Tensor& check_for_empty = src.is_nested() ? get_nested_tensor_impl(src)->get_buffer() : src;
    // 如果内部张量元素数量为零，根据情况返回嵌套张量或克隆的张量
    if (check_for_empty.numel() == 0) {
      return src.is_nested()
        ? at::detail::make_tensor<NestedTensorImpl>(check_for_empty, get_nested_tensor_impl(src)->get_nested_sizes())
        : src.clone();
  }
}
const bool use_nested_tensor = src.is_nested();
// 检查源张量是否是嵌套张量，返回布尔值

Tensor x = src;
// 将输入张量 src 赋值给 x

if (norm_first) {
  // 如果需要先进行归一化
  x = norm(x, embed_dim, layer_norm_eps, layer_norm_weight_1, layer_norm_bias_1, use_nested_tensor);
  // 对 x 进行归一化处理，使用给定的参数和是否使用嵌套张量的信息
}

// 执行多头注意力机制操作，并返回注意力权重
x = std::get<0>(at::_native_multi_head_attention(
    x,
    x,
    x,
    embed_dim,
    num_heads,
    qkv_weight,
    qkv_bias,
    proj_weight,
    proj_bias,
    mask,
    false /* need_weights */,
    true /* average_attn_weights */,
    mask_type));

// 将源张量 src 添加到 x 上
x.add_(src);

if (!norm_first) {
  // 如果不需要先进行归一化
  x = norm(x, embed_dim, layer_norm_eps, layer_norm_weight_1, layer_norm_bias_1, use_nested_tensor);
  // 对 x 进行归一化处理，使用给定的参数和是否使用嵌套张量的信息
}

auto pre_ffn_res = x;
// 将当前 x 的值保存为前馈网络操作之前的备份 pre_ffn_res

if (norm_first) {
  // 如果需要先进行归一化
  x = norm(x, embed_dim, layer_norm_eps, layer_norm_weight_2, layer_norm_bias_2, use_nested_tensor);
  // 对 x 进行归一化处理，使用给定的参数和是否使用嵌套张量的信息
}

// 执行前馈神经网络 (FFN) 操作
x = ffn(
    x,
    ffn_weight_1,
    ffn_bias_1,
    ffn_weight_2,
    ffn_bias_2,
    use_gelu,
    /* add_norm* */ false);
// 使用给定的权重和偏置以及是否使用 GELU 激活函数，执行前馈神经网络操作

// 将前馈网络操作之前的备份 pre_ffn_res 加到 x 上
x.add_(pre_ffn_res);

if (!norm_first) {
  // 如果不需要先进行归一化
  x = norm(x, embed_dim, layer_norm_eps, layer_norm_weight_2, layer_norm_bias_2, use_nested_tensor);
  // 对 x 进行归一化处理，使用给定的参数和是否使用嵌套张量的信息
}

return x;
// 返回处理后的张量 x
}

} // namespace native
} // namespace at
```