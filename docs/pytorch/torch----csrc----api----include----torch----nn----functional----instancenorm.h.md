# `.\pytorch\torch\csrc\api\include\torch\nn\functional\instancenorm.h`

```
#pragma once

#include <torch/nn/options/instancenorm.h>  // 引入 InstanceNormFuncOptions 相关的头文件

namespace torch {
namespace nn {
namespace functional {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
// 定义了 instance_norm 的具体实现细节函数
inline Tensor instance_norm(
    const Tensor& input,              // 输入张量
    const Tensor& running_mean,       // 运行时均值
    const Tensor& running_var,        // 运行时方差
    const Tensor& weight,             // 权重
    const Tensor& bias,               // 偏置
    bool use_input_stats,             // 是否使用输入的统计信息
    double momentum,                  // 动量
    double eps) {                     // epsilon 值
  return torch::instance_norm(
      input,
      weight,
      bias,
      running_mean,
      running_var,
      use_input_stats,
      momentum,
      eps,
      at::globalContext().userEnabledCuDNN());  // 调用 Torch 的 instance_norm 函数
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.instance_norm
/// about the exact behavior of this functional.
/// 查看官方文档了解此函数的精确行为描述。

/// See the documentation for `torch::nn::functional::InstanceNormFuncOptions`
/// class to learn what optional arguments are supported for this functional.
/// 查看 `torch::nn::functional::InstanceNormFuncOptions` 类的文档，了解此函数支持的可选参数。

/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::instance_norm(input,
/// F::InstanceNormFuncOptions().running_mean(mean).running_var(variance).weight(weight).bias(bias).momentum(0.1).eps(1e-5));
/// ```
/// 示例用法，展示了如何调用 instance_norm 函数并设置各种可选参数。

inline Tensor instance_norm(
    const Tensor& input,                          // 输入张量
    const InstanceNormFuncOptions& options = {}) { // 可选参数选项
  return detail::instance_norm(
      input,
      options.running_mean(),    // 获取选项中的运行时均值
      options.running_var(),     // 获取选项中的运行时方差
      options.weight(),          // 获取选项中的权重
      options.bias(),            // 获取选项中的偏置
      options.use_input_stats(), // 获取选项中的是否使用输入统计信息
      options.momentum(),        // 获取选项中的动量
      options.eps());            // 获取选项中的 epsilon 值
}

} // namespace functional
} // namespace nn
} // namespace torch
```