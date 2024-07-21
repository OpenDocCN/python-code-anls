# `.\pytorch\torch\csrc\api\include\torch\nn\functional\batchnorm.h`

```
#pragma once
// 预处理指令，确保头文件只被编译一次

#include <c10/util/irange.h>
// 包含 C10 库中的 irange.h 头文件

#include <torch/nn/options/batchnorm.h>
// 包含 Torch 中神经网络模块的批标准化选项头文件

#include <torch/types.h>
// 包含 Torch 中的数据类型定义头文件

namespace torch {
namespace nn {
namespace functional {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor batch_norm(
    const Tensor& input,
    const Tensor& running_mean,
    const Tensor& running_var,
    Tensor weight,
    Tensor bias,
    bool training,
    std::optional<double> momentum,
    double eps) {
  TORCH_CHECK(
      input.dim() >= 2,
      "Expected at least 2 input dimensions, but got ",
      input.dim());
  // 检查输入张量维度是否至少为2

  if (training) {
    auto size = input.sizes();
    int64_t size_prods = size[0];
    for (const auto i : c10::irange(size.size() - 2)) {
      size_prods *= size[i + 2];
    }
    // 如果是训练模式，检查每个通道是否有多于一个值
    TORCH_CHECK(
        size_prods != 1,
        "Expected more than 1 value per channel when training, got input size ",
        size);
  }

  return torch::batch_norm(
      input,
      weight,
      bias,
      running_mean,
      running_var,
      training,
      momentum.value(),
      eps,
      at::globalContext().userEnabledCuDNN());
  // 调用 Torch 的批标准化函数进行批标准化操作，并返回结果张量
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.batch_norm
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::BatchNormFuncOptions`
/// class to learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::batch_norm(input, mean, variance,
/// F::BatchNormFuncOptions().weight(weight).bias(bias).momentum(0.1).eps(1e-05).training(false));
/// ```
inline Tensor batch_norm(
    const Tensor& input,
    const Tensor& running_mean,
    const Tensor& running_var,
    const BatchNormFuncOptions& options = {}) {
  return detail::batch_norm(
      input,
      running_mean,
      running_var,
      options.weight(),
      options.bias(),
      options.training(),
      options.momentum(),
      options.eps());
  // 调用 detail 命名空间中的 batch_norm 函数，传递批标准化的参数并返回结果张量
}

} // namespace functional
} // namespace nn
} // namespace torch
```