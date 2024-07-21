# `.\pytorch\torch\csrc\api\src\nn\modules\pixelshuffle.cpp`

```
// 引入 Torch 库中的像素重排头文件
#include <torch/nn/modules/pixelshuffle.h>

// 命名空间别名，简化调用 torch::nn::functional
namespace F = torch::nn::functional;

// Torch 库命名空间开始
namespace torch {
namespace nn {

// PixelShuffleImpl 类的构造函数，接受 PixelShuffleOptions 类的参数对象
PixelShuffleImpl::PixelShuffleImpl(const PixelShuffleOptions& options_)
    : options(options_) {}

// 输出 PixelShuffleImpl 类的信息到指定流，显示像素上采样因子
void PixelShuffleImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::PixelShuffle(upscale_factor="
         << options.upscale_factor() << ")";
}

// 重置 PixelShuffleImpl 类，但此处未实现任何具体操作
void PixelShuffleImpl::reset() {}

// PixelShuffleImpl 类的前向传播函数，调用 torch::nn::functional 中的像素重排函数
Tensor PixelShuffleImpl::forward(const Tensor& input) {
  return F::detail::pixel_shuffle(input, options.upscale_factor());
}

// PixelUnshuffleImpl 类的构造函数，接受 PixelUnshuffleOptions 类的参数对象
PixelUnshuffleImpl::PixelUnshuffleImpl(const PixelUnshuffleOptions& options_)
    : options(options_) {}

// 输出 PixelUnshuffleImpl 类的信息到指定流，显示像素下采样因子
void PixelUnshuffleImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::PixelUnshuffle(downscale_factor="
         << options.downscale_factor() << ")";
}

// 重置 PixelUnshuffleImpl 类，但此处未实现任何具体操作
void PixelUnshuffleImpl::reset() {}

// PixelUnshuffleImpl 类的前向传播函数，调用 torch::nn::functional 中的像素反重排函数
Tensor PixelUnshuffleImpl::forward(const Tensor& input) {
  return F::detail::pixel_unshuffle(input, options.downscale_factor());
}

} // namespace nn
} // namespace torch
```