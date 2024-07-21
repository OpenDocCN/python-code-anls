# `.\pytorch\aten\src\ATen\native\mkldnn\xpu\detail\oneDNN.h`

```
#pragma once
// 预处理指令，确保头文件只被包含一次

#include <ATen/ATen.h>
// 包含 ATen 库的主头文件

#include <ATen/native/mkldnn/xpu/detail/oneDNNContext.h>
#include <ATen/native/mkldnn/xpu/detail/Attr.h>
#include <ATen/native/mkldnn/xpu/detail/Utils.h>
// 包含一些 MKL-DNN XPU 相关的细节头文件

namespace at::native::onednn{

TORCH_API sycl::event matmul(
    at::Tensor& result,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Tensor& b_raw,
    bool m2_trans,
    Attr attr,
    const std::vector<sycl::event>& deps = {});
// 声明一个用于矩阵乘法的函数 matmul，返回一个 SYCL 事件

TORCH_API sycl::event convolution(
    at::Tensor& dst,
    const at::Tensor& src,
    const at::Tensor& weight,
    const at::Tensor& bia,
    IntArrayRef padding_front_top_left,
    IntArrayRef padding_back_bottom_right,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    Attr& attr,
    const std::vector<sycl::event>& deps = {});
// 声明一个用于卷积操作的函数 convolution，返回一个 SYCL 事件

TORCH_API sycl::event convolution_backward_weights(
    at::Tensor& diff_weight,
    at::Tensor& diff_bia,
    const at::Tensor& diff_dst,
    const at::Tensor& src,
    IntArrayRef diff_weight_aten_size,
    IntArrayRef padding_front_top_left,
    IntArrayRef padding_back_bottom_right,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    const std::vector<sycl::event>& deps = {});
// 声明一个用于卷积反向权重更新的函数 convolution_backward_weights，返回一个 SYCL 事件

TORCH_API sycl::event convolution_backward_data(
    at::Tensor& diff_src,
    const at::Tensor& diff_dst,
    const at::Tensor& weight,
    IntArrayRef padding_front_top_left,
    IntArrayRef padding_back_bottom_right,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool bias_defined,
    const std::vector<sycl::event>& deps = {});
// 声明一个用于卷积反向数据更新的函数 convolution_backward_data，返回一个 SYCL 事件

TORCH_API sycl::event deconvolution(
    at::Tensor& dst,
    const at::Tensor& src,
    const at::Tensor& weight,
    const at::Tensor& bia,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dst_padding,
    IntArrayRef dilation,
    int64_t groups,
    Attr& attr,
    const std::vector<sycl::event>& deps = {});
// 声明一个用于反卷积操作的函数 deconvolution，返回一个 SYCL 事件

TORCH_API sycl::event deconvolution_backward_data(
    at::Tensor& diff_src,
    const at::Tensor& diff_dst,
    const at::Tensor& weight,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    bool bias_defined,
    const std::vector<sycl::event>& deps = {});
// 声明一个用于反卷积反向数据更新的函数 deconvolution_backward_data，返回一个 SYCL 事件

TORCH_API sycl::event deconvolution_backward_weights(
    at::Tensor& diff_weight,
    at::Tensor& diff_bia,
    const at::Tensor& diff_dst,
    const at::Tensor& src,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    const std::vector<sycl::event>& deps = {});
// 声明一个用于反卷积反向权重更新的函数 deconvolution_backward_weights，返回一个 SYCL 事件

dnnl::memory::dims conv_dst_size(
    int64_t ndim,
    IntArrayRef src_tz,
    IntArrayRef wgh_tz,
    IntArrayRef padding_front_top_left,
    IntArrayRef padding_back_bottom_right,
    IntArrayRef stride,
    IntArrayRef dilation);
// 声明一个用于计算卷积输出尺寸的函数 conv_dst_size，返回一个 dnnl::memory::dims 结构

dnnl::memory::dims deconv_dst_size(
    IntArrayRef src_size,
    IntArrayRef wgh_size,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    IntArrayRef dst_padding,
    int64_t groups);
// 声明一个用于计算反卷积输出尺寸的函数 deconv_dst_size，返回一个 dnnl::memory::dims 结构

} // namespace at::native::onednn
// 命名空间结束
} // 结束命名空间 at::native::onednn
```