# `.\pytorch\aten\src\ATen\native\cudnn\MHA.h`

```
#pragma once
// 预处理指令，确保头文件只被包含一次

#include <ATen/core/Tensor.h>
// 包含 ATen 库的 Tensor 头文件

namespace at {
namespace native {

void run_cudnn_SDP_fprop(
    int64_t b,
    int64_t h,
    int64_t s_q,
    int64_t s_kv,
    int64_t d,
    float scaling_factor,
    bool isTraining,
    bool is_causal,
    double dropout_probability,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    Tensor& softmaxstats,
    Tensor& o,
    Tensor& dropoutseed,
    Tensor& dropoutoffset);
// 声明前向传播函数 run_cudnn_SDP_fprop，接受多个参数包括张量和标量，无返回值

void run_cudnn_SDP_bprop(
    int64_t b,
    int64_t h,
    int64_t s_q,
    int64_t s_kv,
    int64_t d,
    float scaling_factor,
    bool is_causal,
    float dropout_probability,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& o,
    const Tensor& dO,
    const Tensor& softmaxstats,
    Tensor& dQ,
    Tensor& dK,
    Tensor& dV,
    const Tensor& dropoutseed,
    const Tensor& dropoutoffset);
// 声明反向传播函数 run_cudnn_SDP_bprop，接受多个参数包括张量和标量，无返回值

} // namespace native
} // namespace at
// 命名空间结束声明，包含 ATen 库中的 native 功能
```