# `.\pytorch\torch\csrc\api\include\torch\nn\functional\normalization.h`

```
```cpp`
#pragma once

#include <torch/nn/functional/padding.h>
#include <torch/nn/functional/pooling.h>
#include <torch/nn/options/normalization.h>
#include <torch/types.h>

// 命名空间 torch，nn，functional 的嵌套命名空间
namespace torch {
namespace nn {
namespace functional {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
// detail 命名空间包含实现细节的函数
namespace detail {
inline Tensor normalize(
    const Tensor& input,
    double p,
    int64_t dim,
    double eps,
    std::optional<Tensor> out) {
  // 如果 out 参数为 nullopt，则计算归一化值，返回 input 除以归一化值
  if (out == c10::nullopt) {
    auto denom = input.norm(p, dim, true).clamp_min(eps).expand_as(input);
    return input / denom;
  } else {
    // 如果 out 参数不为 nullopt，则使用 out 作为输出，计算归一化值
    auto denom = input.norm(p, dim, true).clamp_min(eps).expand_as(input);
    return torch::div_out(*out, input, denom);
  }
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

// 定义 normalize 函数，调用 detail 命名空间中的 normalize 函数
/// 详细信息参见 PyTorch 官方文档 https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.normalize
///
/// 查看 torch::nn::functional::NormalizeFuncOptions 类文档，了解该函数支持的可选参数。
///
/// 示例：
/// ```
/// namespace F = torch::nn::functional;
/// F::normalize(input, F::NormalizeFuncOptions().p(1).dim(-1));
/// ```
inline Tensor normalize(
    const Tensor& input,
    NormalizeFuncOptions options = {}) {
  return detail::normalize(
      input, options.p(), options.dim(), options.eps(), options.out());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
// detail 命名空间包含实现细节的函数
namespace detail {
inline Tensor layer_norm(
    const Tensor& input,
    const std::vector<int64_t>& normalized_shape,
    const Tensor& weight,
    const Tensor& bias,
    double eps) {
  // 调用 torch::layer_norm 函数进行层归一化
  return torch::layer_norm(input, normalized_shape, weight, bias, eps);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

// 定义 layer_norm 函数，调用 detail 命名空间中的 layer_norm 函数
/// 详细信息参见 PyTorch 官方文档 https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.layer_norm
///
/// 查看 torch::nn::functional::LayerNormFuncOptions 类文档，了解该函数支持的可选参数。
///
/// 示例：
/// ```
/// namespace F = torch::nn::functional;
/// F::layer_norm(input, F::LayerNormFuncOptions({2, 2}).eps(2e-5));
/// ```
inline Tensor layer_norm(
    const Tensor& input,
    const LayerNormFuncOptions& options) {
  return detail::layer_norm(
      input,
      options.normalized_shape(),
      options.weight(),
      options.bias(),
      options.eps());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
// detail 命名空间包含实现细节的函数
namespace detail {
inline Tensor local_response_norm(
    const Tensor& input,
    int64_t size,
    double alpha,
    double beta,
    double k) {
  auto dim = input.dim(); // 获取输入张量的维度
  TORCH_CHECK(
      dim >= 3,
      "Expected 3D or higher dimensionality input (got ",
      dim,
      " dimensions)"); // 检查输入张量的维度是否大于等于 3
  auto div = input.mul(input).unsqueeze(1); // 对输入张量的每个元素平方，并在第 1 维插入一个新的维度
  if (dim == 3) {
    div = detail::pad(
        div,
        /*pad=*/{0, 0, size / 2, (size - 1) / 2},  // 在第二维度上进行填充操作，左右各填充 size/2 和 (size-1)/2
        /*mode=*/torch::kConstant,  // 使用常数填充模式
        /*value=*/0);  // 填充值设定为0

    div = detail::avg_pool2d(
              div,
              /*kernel_size=*/{size, 1},  // 二维平均池化操作的卷积核大小为 {size, 1}
              /*stride=*/1,  // 步长设定为1
              /*padding=*/0,  // 无额外填充
              /*ceil_mode=*/false,  // 不使用向上取整模式
              /*count_include_pad=*/true,  // 池化计算中包含填充部分
              /*divisor_override=*/c10::nullopt)  // 使用默认的除数覆盖选项
              .squeeze(1);  // 对第二维度进行压缩

  } else {
    auto sizes = input.sizes();
    div = div.view({sizes[0], 1, sizes[1], sizes[2], -1});  // 重新视图化 div 张量的形状
    div = detail::pad(
        div,
        /*pad=*/{0, 0, 0, 0, size / 2, (size - 1) / 2},  // 在第五维度上进行填充操作，左右各填充 size/2 和 (size-1)/2
        /*mode=*/torch::kConstant,  // 使用常数填充模式
        /*value=*/0);  // 填充值设定为0

    div = detail::avg_pool3d(
              div,
              /*kernel_size=*/{size, 1, 1},  // 三维平均池化操作的卷积核大小为 {size, 1, 1}
              /*stride=*/1,  // 步长设定为1
              /*padding=*/0,  // 无额外填充
              /*ceil_mode=*/false,  // 不使用向上取整模式
              /*count_include_pad=*/true,  // 池化计算中包含填充部分
              /*divisor_override=*/c10::nullopt)  // 使用默认的除数覆盖选项
              .squeeze(1);  // 对第二维度进行压缩

    div = div.view(sizes);  // 恢复 div 张量的原始形状
  }

  div = div.mul(alpha).add(k).pow(beta);  // 对 div 张量执行逐元素乘法、加法和幂运算
  return input / div;  // 返回 input 张量与 div 张量逐元素相除的结果
/// 结束 detail 命名空间的定义
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// 查看此功能的确切行为，请参阅
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.local_response_norm
///
/// 查看 `torch::nn::functional::LocalResponseNormFuncOptions` 类的文档，了解此功能支持的可选参数。
///
/// 示例：
/// ```
/// namespace F = torch::nn::functional;
/// F::local_response_norm(x, F::LocalResponseNormFuncOptions(2));
/// ```
inline Tensor local_response_norm(
    const Tensor& input,
    const LocalResponseNormFuncOptions& options) {
  return detail::local_response_norm(
      input, options.size(), options.alpha(), options.beta(), options.k());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
/// 执行 group_norm 操作的内部函数
inline Tensor group_norm(
    const Tensor& input,
    int64_t num_groups,
    const Tensor& weight,
    const Tensor& bias,
    double eps) {
  return torch::group_norm(
      input,
      num_groups,
      weight,
      bias,
      eps,
      at::globalContext().userEnabledCuDNN());
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// 查看此功能的确切行为，请参阅
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.group_norm
///
/// 查看 `torch::nn::functional::GroupNormFuncOptions` 类的文档，了解此功能支持的可选参数。
///
/// 示例：
/// ```
/// namespace F = torch::nn::functional;
/// F::group_norm(input, F::GroupNormFuncOptions(2).eps(2e-5));
/// ```
inline Tensor group_norm(
    const Tensor& input,
    const GroupNormFuncOptions& options) {
  return detail::group_norm(
      input,
      options.num_groups(),
      options.weight(),
      options.bias(),
      options.eps());
}

} // namespace functional
} // namespace nn
} // namespace torch


这些注释为给定的 C++ 代码片段提供了详细的解释，描述了每个函数和命名空间的用途及其相关性。
```