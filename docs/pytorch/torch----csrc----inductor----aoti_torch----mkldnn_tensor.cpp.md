# `.\pytorch\torch\csrc\inductor\aoti_torch\mkldnn_tensor.cpp`

```py
// 包含 ATen 库的配置文件
#include <ATen/Config.h>
// 包含 MKLDNN 引擎的张量头文件
#include <torch/csrc/inductor/aoti_torch/mkldnn_tensor.h>

// 如果 MKLDNN 功能已启用，则包含相关的头文件
#if AT_MKLDNN_ENABLED()
#include <ATen/native/mkldnn/MKLDNNCommon.h>
#include <ideep.hpp>
#endif

// 定义 torch 命名空间下的 aot_inductor 命名空间
namespace torch {
namespace aot_inductor {

// 如果 MKLDNN 功能已启用，则定义将 MKLDNN 张量转换为数据指针的函数
#if AT_MKLDNN_ENABLED()
void* data_ptr_from_mkldnn(at::Tensor* mkldnn_tensor) {
  return reinterpret_cast<void*>(
      at::native::data_ptr_from_mkldnn(*mkldnn_tensor));
}

// 如果 MKLDNN 功能已启用，则定义根据数据指针创建 MKLDNN 张量的函数
at::Tensor mkldnn_tensor_from_data_ptr(
    void* data_ptr,
    at::IntArrayRef dims,
    at::ScalarType dtype,
    at::Device device,
    const uint8_t* opaque_metadata,
    int64_t opaque_metadata_size) {
  return at::native::mkldnn_tensor_from_data_ptr(
      data_ptr, dims, dtype, device, opaque_metadata, opaque_metadata_size);
}

// 如果 MKLDNN 功能未启用，则定义错误处理函数，提示 MKLDNN 构建被禁用
#else
void* data_ptr_from_mkldnn(at::Tensor* mkldnn_tensor) {
  TORCH_CHECK(false, "MKL-DNN build is disabled");
}

// 如果 MKLDNN 功能未启用，则定义错误处理函数，提示 MKLDNN 构建被禁用
at::Tensor mkldnn_tensor_from_data_ptr(
    void* data_ptr,
    at::IntArrayRef dims,
    at::ScalarType dtype,
    at::Device device,
    const uint8_t* opaque_metadata,
    int64_t opaque_metadata_size) {
  TORCH_CHECK(false, "MKL-DNN build is disabled");
}
#endif

} // namespace aot_inductor
} // namespace torch
```