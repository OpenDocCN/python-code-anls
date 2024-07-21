# `.\pytorch\aten\src\ATen\native\mkldnn\MKLDNNCommon.h`

```
#pragma once
// 预处理指令，确保头文件只被编译一次

#include <ATen/core/Tensor.h>
// 包含 ATen 库中的 Tensor 类定义头文件
#include <ATen/Config.h>
// 包含 ATen 库的配置信息头文件

#if AT_MKLDNN_ENABLED()
// 如果 MKLDNN 功能被启用，则编译以下内容

#include <ideep.hpp>
// 包含 Intel 的 MKL-DNN 库头文件

#ifndef IDEEP_PREREQ
// 如果未定义 IDEEP_PREREQ 宏，则定义该宏用于版本检查
// IDEEP_PREREQ 宏用于比较 MKL-DNN 版本号和指定的版本号

// 请在 ideep.hpp 中找到版本号的定义
#if defined(IDEEP_VERSION_MAJOR) && defined(IDEEP_VERSION_MINOR) && \
  defined(IDEEP_VERSION_PATCH) && defined(IDEEP_VERSION_REVISION)
#define IDEEP_PREREQ(major, minor, patch, revision) \
  (((IDEEP_VERSION_MAJOR << 16) + (IDEEP_VERSION_MINOR << 8) + \
   (IDEEP_VERSION_PATCH << 0)) >= \
   ((major << 16) + (minor << 8) + (patch << 0)) && \
   (IDEEP_VERSION_REVISION >= revision))
#else
#define IDEEP_PREREQ(major, minor, patch, revision) 0
#endif
#endif

namespace at { namespace native {

// ATen 库的 native 命名空间中定义以下内容

// 根据 ScalarType 映射到对应的 MKL-DNN tensor 数据类型
TORCH_API ideep::tensor::data_type get_mkldnn_dtype(ScalarType type);

// 内联函数：根据 Tensor 获取对应的 MKL-DNN 数据类型
static inline ideep::tensor::data_type get_mkldnn_dtype(const Tensor& t) {
  return get_mkldnn_dtype(t.scalar_type());
}

// 从 MKL-DNN tensor 获取数据指针
TORCH_API int64_t data_ptr_from_mkldnn(const Tensor& mkldnn_tensor);

// 根据给定数据指针、维度、数据类型、设备、元数据，构造 MKL-DNN tensor
TORCH_API at::Tensor mkldnn_tensor_from_data_ptr(
    void* data_ptr,
    at::IntArrayRef dims,
    at::ScalarType dtype,
    at::Device device,
    const uint8_t* opaque_metadata,
    int64_t opaque_metadata_size);

// 根据 ideep::tensor 构造新的 MKL-DNN tensor
TORCH_API Tensor new_with_itensor_mkldnn(ideep::tensor&& it, std::optional<ScalarType> dtype, std::optional<Device> device);

// 从 MKL-DNN tensor 获取 ideep::tensor 引用
TORCH_API ideep::tensor& itensor_from_mkldnn(const Tensor& mkldnn_tensor);

// 从 MKL-DNN tensor 获取字节数
TORCH_API int64_t nbytes_from_mkldnn(const Tensor& mkldnn_tensor);

// 根据 dense tensor 构造 ideep::tensor 视图，可以选择是否使用常量数据指针
TORCH_API ideep::tensor itensor_view_from_dense(const Tensor& tensor, bool from_const_data_ptr=false);

// 根据给定描述符和 dense tensor 构造 ideep::tensor 视图
TORCH_API ideep::tensor itensor_view_from_dense(
    const at::Tensor& tensor,
    const ideep::tensor::desc& desc);

// 从 aten Tensor 或 MKL-DNN tensor 获取 ideep tensor 的辅助函数
TORCH_API ideep::tensor itensor_from_tensor(const Tensor& tensor, bool from_const_data_ptr=false);

// 设置 MKLDNN 的详细输出级别
TORCH_API int set_verbose(int level);

}}

#endif // AT_MKLDNN_ENABLED
// 结束条件：如果 MKLDNN 功能未启用，则结束当前预处理指令的处理
```