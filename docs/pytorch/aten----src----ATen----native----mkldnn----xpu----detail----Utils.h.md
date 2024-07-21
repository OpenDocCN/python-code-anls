# `.\pytorch\aten\src\ATen\native\mkldnn\xpu\detail\Utils.h`

```
#pragma once
// 预处理指令：指示编译器在编译此文件时只包含一次当前头文件

#include <iostream>
// 包含标准输入输出流库

#include <ATen/ATen.h>
// 包含 PyTorch 的 ATen 头文件

#include <ATen/Tensor.h>
#include <ATen/core/Tensor.h>
// 包含 PyTorch 的 Tensor 类和核心 Tensor 功能的头文件

#include <ATen/core/grad_mode.h>
// 包含 ATen 的梯度模式管理功能的头文件

#include <c10/core/MemoryFormat.h>
// 包含 c10 的内存格式管理功能的头文件

#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <oneapi/dnnl/dnnl_version.h>
// 包含 OneDNN（原名 DNNL）的相关头文件，包括核心功能、SYCL 支持和版本信息

#define ONEDNN_SUPPORT_DETERMINISTIC (DNNL_VERSION_MAJOR >=3 && DNNL_VERSION_MINOR >=4)
// 定义宏：检查 OneDNN 是否支持确定性操作，需版本大于等于 3.4

namespace at::native::onednn {

dnnl::memory::format_tag get_dnnl_default_format(
    int ndims,
    bool is_channels_last = false,
    bool allow_undef = false);
// 声明函数：获取 OneDNN 默认内存格式标签，根据维度数、是否通道在后以及是否允许未定义情况

dnnl::memory::data_type get_onednn_dtype(
    const at::Tensor& tensor,
    bool allow_undef = false);
// 声明函数：获取给定 Tensor 对应的 OneDNN 数据类型，可选择是否允许未定义情况

dnnl::memory::data_type get_onednn_dtype_include_double(
    const at::Tensor& tensor,
    bool allow_undef = false);
// 声明函数：获取给定 Tensor 对应的 OneDNN 数据类型，包括双精度，可选择是否允许未定义情况

bool is_supported_onednn_dtype(const at::Tensor& tensor);
// 声明函数：检查给定 Tensor 是否支持的 OneDNN 数据类型

dnnl::memory::dims get_onednn_dims(const at::Tensor& tensor);
// 声明函数：获取给定 Tensor 的 OneDNN 内存维度

dnnl::memory::dims get_onednn_strides(const at::Tensor& tensor);
// 声明函数：获取给定 Tensor 的 OneDNN 步幅

dnnl::memory::desc get_onednn_md(const at::Tensor& tensor);
// 声明函数：获取给定 Tensor 的 OneDNN 内存描述符

bool onednn_strides_check(const at::Tensor& src);
// 声明函数：检查给定 Tensor 的 OneDNN 步幅是否符合要求

bool is_broadcast(const at::Tensor& t);
// 声明函数：检查给定 Tensor 是否支持广播操作

bool is_onednn_matmul_strides(
    const at::Tensor& tensor,
    bool is_dst = false);
// 声明函数：检查给定 Tensor 是否符合 OneDNN 矩阵乘法的步幅要求

bool is_broadcast_from_other_to_self(
    const at::Tensor& self,
    const at::Tensor& other);
// 声明函数：检查从其他 Tensor 到自身的广播操作是否支持

at::MemoryFormat get_cl_tag_by_ndim(const int64_t ndim);
// 声明函数：根据 Tensor 的维度数获取内存格式标签

bool binary_valid(
    const at::Tensor& self,
    const at::Tensor& other,
    bool is_fusion = false);
// 声明函数：检查两个 Tensor 是否可以进行二进制运算，可选择是否为融合操作

bool use_channels_last_for_conv(
    const at::Tensor& src,
    const at::Tensor& weight,
    bool is_transpose);
// 声明函数：检查是否应在卷积操作中使用通道在后的布局

} // namespace at::native::onednn
// 结束命名空间：at::native::onednn
```