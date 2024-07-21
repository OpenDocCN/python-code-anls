# `.\pytorch\torch\csrc\distributed\c10d\quantization\quantization_utils.h`

```
// 使用#pragma once确保头文件只被包含一次，避免重复定义问题

#include <ATen/ATen.h>  // 引入ATen库，用于PyTorch张量操作

#include <typeinfo>  // 引入typeinfo库，用于获取类型信息

// 定义一个内联函数，返回给定张量的设备名称字符串
inline std::string torch_tensor_device_name(const at::Tensor& ten) {
  return c10::DeviceTypeName(ten.device().type());  // 调用c10库获取张量设备类型名称
}

// 定义一个宏，检查张量的维度是否与指定的维度相等
#define TENSOR_NDIM_EQUALS(ten, dims)      \
  TORCH_CHECK(                             \
      (ten).ndimension() == (dims),        \
      "Tensor '" #ten "' must have " #dims \
      " dimension(s). "                    \
      "Found ",                            \
      (ten).ndimension())  // 使用TORCH_CHECK宏检查张量维度是否符合预期

// 定义一个宏，检查张量是否在CPU上
#define TENSOR_ON_CPU(x)                                      \
  TORCH_CHECK(                                                \
      !x.is_cuda(),                                           \
      #x " must be a CPU tensor; it is currently on device ", \
      torch_tensor_device_name(x))  // 使用TORCH_CHECK宏检查张量是否在CPU上

// 定义一个宏，检查张量是否在CUDA设备上
#define TENSOR_ON_CUDA_GPU(x)                                  \
  TORCH_CHECK(                                                 \
      x.is_cuda(),                                             \
      #x " must be a CUDA tensor; it is currently on device ", \
      torch_tensor_device_name(x))  // 使用TORCH_CHECK宏检查张量是否在CUDA设备上
```