# `.\pytorch\aten\src\ATen\native\nested\NestedTensorMath.h`

```
#pragma once

#include <ATen/core/ATen_fwd.h>
#include <ATen/NestedTensorImpl.h>
#include <c10/macros/Macros.h>

namespace at {
namespace native {

// 定义了一个公共的 API 函数，将 NestedTensor 转换为填充的通用张量
TORCH_API Tensor NestedTensor_to_padded_tensor_generic(
    const Tensor& t,                    // 输入的 NestedTensor 对象
    double padding,                     // 填充值
    OptionalIntArrayRef output_size);   // 可选的输出大小参数

// 对 NestedTensor 进行映射操作的模板函数
template <typename Func>
Tensor map_nt(const Tensor& nt, Func f) {
  auto* nt_impl = get_nested_tensor_impl(nt);      // 获取 NestedTensorImpl 对象指针
  const auto& sizes = nt_impl->get_nested_sizes(); // 获取 NestedTensor 内部张量的大小信息
  return at::detail::make_tensor<NestedTensorImpl>(f(nt_impl->get_buffer()), sizes);
}

// 对两个 NestedTensor 进行二元映射操作的模板函数
template <typename Func>
Tensor map_nt_binary(const Tensor& nt_1, const Tensor& nt_2, Func f){
  auto* nt_impl_1 = get_nested_tensor_impl(nt_1);  // 获取第一个 NestedTensorImpl 对象指针
  auto* nt_impl_2 = get_nested_tensor_impl(nt_2);  // 获取第二个 NestedTensorImpl 对象指针
  const auto& sizes = nt_impl_1->get_nested_sizes(); // 获取第一个 NestedTensor 内部张量的大小信息
  return at::detail::make_tensor<NestedTensorImpl>(f(nt_impl_1->get_buffer(), nt_impl_2->get_buffer()), sizes);
}

// 内联函数，用于检查 NestedTensor 层归一化输入的有效性，并返回 M 和 N 值
C10_ALWAYS_INLINE std::pair<int64_t, int64_t> _check_nested_layer_norm_inputs(
    const NestedTensorImpl& input,    // 输入的 NestedTensorImpl 对象引用
    IntArrayRef normalized_shape,     // 归一化形状
    const Tensor& weight /* optional */,  // 可选的权重张量
    const Tensor& bias /* optional */) {  // 可选的偏置张量

  const size_t normalized_ndim = normalized_shape.size();  // 归一化形状的维度数
  TORCH_CHECK(
      normalized_ndim >= 1,
      "Expected normalized_shape to be at least 1-dimensional, i.e., ",
      "containing at least one element, but got normalized_shape = ",
      normalized_shape);
  TORCH_CHECK(
      !weight.defined() || weight.sizes().equals(normalized_shape),
      "Expected weight to be of same shape as normalized_shape, but got ",
      "weight of shape ",
      weight.sizes(),
      " and normalized_shape = ",
      normalized_shape);
  TORCH_CHECK(
      !bias.defined() || bias.sizes().equals(normalized_shape),
      "Expected bias to be of same shape as normalized_shape, but got ",
      "bias of shape ",
      bias.sizes(),
      " and normalized_shape = ",
      normalized_shape);

  // 检查归一化形状的每个维度与 NestedTensor 输入的最后几个维度是否完全匹配，并计算 M 和 N
  int64_t N = 1;
  for (const auto i: c10::irange(normalized_ndim)) {
    TORCH_CHECK(
      input.opt_size(-normalized_ndim + i) != c10::nullopt,
      "normalized_shape extends into irregular dimensions for the nested tensor"
    );
    TORCH_CHECK(
      normalized_shape[i] == *input.opt_size(-normalized_ndim + i),
      "The shape at dimension ",
      i,
      "of normalized_shape doesn't match the input"
    );
    N *= normalized_shape[i];
  }

  const int64_t M = input.numel() / N;  // 计算 M 值，表示 NestedTensor 的元素数除以 N

  return std::make_pair(M, N);  // 返回 M 和 N 的 std::pair 对象
}

// 对 NestedTensor 进行重塑操作的函数
Tensor reshape_nested(const Tensor& self, IntArrayRef proposed_shape);

} // namespace native
} // namespace at
```