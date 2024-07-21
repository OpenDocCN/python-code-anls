# `.\pytorch\aten\src\ATen\native\quantized\cudnn\LinearUnpackImpl.cpp`

```py
#ifdef USE_CUDA
#include <ATen/cuda/CUDAConfig.h>  // 包含 AT_CUDNN_ENABLED 的定义，用于检查是否启用了 CUDNN

#if AT_CUDNN_ENABLED()
#include <ATen/ATen.h>
#include <ATen/native/quantized/cudnn/utils.h>
#include <ATen/native/quantized/PackedParams.h>
#include <torch/library.h>

#include <tuple>

// 在 AT_CUDNN_ENABLED 宏定义为真时定义 PackedLinearWeightCudnn 类的方法
std::tuple<at::Tensor, std::optional<at::Tensor>> PackedLinearWeightCudnn::unpack() {
  // 返回一个包含原始权重和可选偏置的元组
  return std::tuple<at::Tensor, std::optional<at::Tensor>>{orig_weight, bias_};
}

#endif  // AT_CUDNN_ENABLED
#endif  // USE_CUDA
```