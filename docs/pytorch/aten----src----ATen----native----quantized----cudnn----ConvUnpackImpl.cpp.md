# `.\pytorch\aten\src\ATen\native\quantized\cudnn\ConvUnpackImpl.cpp`

```
#ifdef USE_CUDA
// 如果定义了 USE_CUDA 宏，则编译以下代码块

#include <ATen/cuda/CUDAConfig.h>  // 用于 AT_CUDNN_ENABLED 的定义

#if AT_CUDNN_ENABLED()
// 如果 AT_CUDNN_ENABLED 宏被定义

#include <ATen/ATen.h>
#include <ATen/native/quantized/cudnn/utils.h>
#include <ATen/native/quantized/PackedParams.h>
#include <torch/library.h>

#include <tuple>

template <int kSpatialDim>
std::tuple<at::Tensor, std::optional<at::Tensor>> PackedConvWeightCudnn<
    kSpatialDim>::unpack() {
  // 返回一个包含 maybe_padded_weight_ 和 bias_ 的 std::tuple 对象
  return std::tuple<at::Tensor, std::optional<at::Tensor>>{maybe_padded_weight_, bias_};
}

// 显式实例化 PackedConvWeightCudnn 类的 unpack 函数模板，针对二维卷积进行特化
template std::tuple<at::Tensor, std::optional<at::Tensor>> PackedConvWeightCudnn<
    2>::unpack();

#endif  // AT_CUDNN_ENABLED
#endif  // USE_CUDA
```