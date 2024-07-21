# `.\pytorch\torch\csrc\inductor\aoti_torch\mkldnn_tensor.h`

```py
#pragma once
// 预处理指令，确保本头文件只被编译一次

#include <ATen/Tensor.h>
// 包含 ATen 库中的 Tensor 类头文件

namespace torch {
namespace aot_inductor {

void* data_ptr_from_mkldnn(at::Tensor* mkldnn_tensor);
// 声明一个函数 data_ptr_from_mkldnn，接受一个指向 ATen Tensor 的指针，并返回一个 void* 指针

at::Tensor mkldnn_tensor_from_data_ptr(
    void* data_ptr,
    at::IntArrayRef dims,
    at::ScalarType dtype,
    at::Device device,
    const uint8_t* opaque_metadata,
    int64_t opaque_metadata_size);
// 声明一个函数 mkldnn_tensor_from_data_ptr，接受 void* 数据指针、Tensor 尺寸、数据类型、设备类型、以及一个不透明元数据指针和大小，并返回一个 ATen Tensor 对象

} // namespace aot_inductor
} // namespace torch
// 命名空间声明结束
```