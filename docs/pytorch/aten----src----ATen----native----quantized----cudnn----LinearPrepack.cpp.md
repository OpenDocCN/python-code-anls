# `.\pytorch\aten\src\ATen\native\quantized\cudnn\LinearPrepack.cpp`

```py
#ifdef USE_CUDA
#include <ATen/cuda/CUDAConfig.h>  // 用于包含 AT_CUDNN_ENABLED 的定义

#if AT_CUDNN_ENABLED()

#include <ATen/ATen.h>
#include <torch/library.h>
#include <ATen/native/quantized/cudnn/utils.h>
#include <ATen/native/quantized/PackedParams.h>
#include <ATen/quantized/Quantizer.h>
#include <c10/core/QScheme.h>
#include <c10/util/irange.h>
#include <torch/library.h>

// 注册线性参数函数声明
int register_linear_params();

// 实现线性权重打包的类 PackedLinearWeightCudnn
c10::intrusive_ptr<LinearPackedParamsBase> PackedLinearWeightCudnn::prepack(
        at::Tensor weight,
        std::optional<at::Tensor> bias) {
  // 检查权重的量化方案是否为 kPerTensorAffine
  TORCH_CHECK(weight.qscheme() == c10::kPerTensorAffine, "Unsupported qscheme: ", toString(weight.qscheme()));
  // 获取输出通道数
  const int output_channels = weight.size(0);
  // 获取权重的量化类型
  const auto qtype = weight.qscheme();
  // 如果存在偏置，检查偏置的维度是否为 1
  if (bias.has_value()) {
    TORCH_CHECK(bias.value().dim() == 1, "bias should be a vector (1D Tensor)");
    // 检查偏置的长度是否与输出通道数相等
    TORCH_CHECK(
        bias.value().size(0) == output_channels,
        "bias should have K elements: " + std::to_string(output_channels));
  }

  // 创建 PackedLinearWeightCudnn 对象并返回指针
  auto ret_ptr = c10::make_intrusive<PackedLinearWeightCudnn>(
          weight,
          bias,
          qtype);
  return ret_ptr;
}

namespace at {
namespace native {
namespace {

// 在 CUDA 环境下运行的 QLinearPackWeightInt8Cudnn 类
class QLinearPackWeightInt8Cudnn final {
 public:
  // 运行权重打包函数，并返回打包后的线性参数
  static c10::intrusive_ptr<LinearPackedParamsBase> run(
      at::Tensor weight,
      std::optional<Tensor> bias) {
      return PackedLinearWeightCudnn::prepack(std::move(weight), std::move(bias));
  }
};

// 注册量化 CUDA 库中的方法
TORCH_LIBRARY_IMPL(quantized, QuantizedCUDA, m) {
  // 注册线性参数函数
  register_linear_params();
  // 实现 quantized::linear_prepack 方法，调用 QLinearPackWeightInt8Cudnn::run
  m.impl(TORCH_SELECTIVE_NAME("quantized::linear_prepack"), TORCH_FN(QLinearPackWeightInt8Cudnn::run));
}

} // namespace
} // namespace native
} // namespace at

#endif  // AT_CUDNN_ENABLED
#endif  // USE_CUDA
```