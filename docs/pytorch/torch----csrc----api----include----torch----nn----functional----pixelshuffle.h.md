# `.\pytorch\torch\csrc\api\include\torch\nn\functional\pixelshuffle.h`

```py
#pragma once
// 使用预处理指令#pragma once，确保头文件只被编译一次

#include <torch/nn/options/pixelshuffle.h>
// 包含像素重排功能相关的选项头文件

namespace torch {
namespace nn {
namespace functional {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
// 如果未启用Doxygen文档生成工具，则跳过此部分

namespace detail {
// 声明命名空间detail，用于实现像素重排的详细操作

inline Tensor pixel_shuffle(const Tensor& input, int64_t upscale_factor) {
// 定义内联函数pixel_shuffle，用于像素重排操作
  return torch::pixel_shuffle(input, upscale_factor);
  // 调用torch命名空间中的pixel_shuffle函数，进行像素重排操作
}

inline Tensor pixel_unshuffle(const Tensor& input, int64_t downscale_factor) {
// 定义内联函数pixel_unshuffle，用于反向像素重排操作
  return torch::pixel_unshuffle(input, downscale_factor);
  // 调用torch命名空间中的pixel_unshuffle函数，进行反向像素重排操作
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.pixel_shuffle
/// about the exact behavior of this functional.
/// 
/// 查看PyTorch官方文档中关于torch.nn.functional.pixel_shuffle功能的确切行为。
///
/// See the documentation for `torch::nn::functional::PixelShuffleFuncOptions`
/// class to learn what optional arguments are supported for this functional.
///
/// 查看torch::nn::functional::PixelShuffleFuncOptions类的文档，了解此功能支持的可选参数。
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::pixel_shuffle(x, F::PixelShuffleFuncOptions(2));
/// ```py
/// 示例：使用torch命名空间中的functional别名F来调用pixel_shuffle函数，传入PixelShuffleFuncOptions对象参数。
inline Tensor pixel_shuffle(
    const Tensor& input,
    const PixelShuffleFuncOptions& options) {
  return detail::pixel_shuffle(input, options.upscale_factor());
  // 调用detail命名空间中的pixel_shuffle函数，传入像素重排选项对象的放大因子参数，执行像素重排操作。
}

inline Tensor pixel_unshuffle(
    const Tensor& input,
    const PixelUnshuffleFuncOptions& options) {
  return detail::pixel_unshuffle(input, options.downscale_factor());
  // 调用detail命名空间中的pixel_unshuffle函数，传入像素反向重排选项对象的缩小因子参数，执行反向像素重排操作。
}

} // namespace functional
} // namespace nn
} // namespace torch
// 结束torch命名空间的定义
```