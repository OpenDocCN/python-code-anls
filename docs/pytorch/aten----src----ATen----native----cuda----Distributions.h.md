# `.\pytorch\aten\src\ATen\native\cuda\Distributions.h`

```
#pragma once

namespace at {
struct CUDAGeneratorImpl;  // 声明 CUDA 生成器实现的结构体

struct TensorIteratorBase;  // 声明张量迭代器基类的结构体

class TensorBase;  // 声明张量基类的类

namespace native {

// 启动 CUDA 内核，计算泊松分布的结果
void launch_poisson_cuda_kernel(
    const TensorBase &ret, const TensorBase &lambda, CUDAGeneratorImpl *gen);

// 启动内核，计算 Gamma 分布的结果
void launch_gamma_kernel(
    const TensorBase &ret, const TensorBase &alpha, CUDAGeneratorImpl *gen);

// 启动 CUDA 内核，计算二项分布的结果
void launch_binomial_cuda_kernel(
    TensorIteratorBase &iter, CUDAGeneratorImpl *gen);

// 启动内核，计算狄利克雷分布的结果
void launch_dirichlet_kernel(TensorIteratorBase &iter);

// 启动内核，计算标准 Gamma 分布的梯度
void launch_standard_gamma_grad_kernel(TensorIteratorBase &iter);

// 启动内核，计算狄利克雷分布的梯度
void launch_dirichlet_grad_kernel(TensorIteratorBase &iter);

}}  // namespace at::native
```