# `.\pytorch\aten\src\ATen\native\cuda\Distributions.cpp`

```py
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 引入 CUDA 分布相关的头文件和依赖项
#include <ATen/native/cuda/Distributions.h>
#include <ATen/TensorIterator.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>

#ifndef AT_PER_OPERATOR_HEADERS
// 引入通用的 ATen 函数和操作的头文件
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
// 引入特定操作的头文件，如 Dirichlet 和 Gamma 分布的梯度和采样相关操作
#include <ATen/ops/_dirichlet_grad_native.h>
#include <ATen/ops/_sample_dirichlet_native.h>
#include <ATen/ops/_standard_gamma_grad_native.h>
#include <ATen/ops/_standard_gamma_native.h>
#include <ATen/ops/binomial_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/poisson_native.h>
#endif

namespace at::native {

// CUDA 加速的泊松分布采样函数
Tensor _s_poisson_cuda(const Tensor& lambda, std::optional<Generator> gen_) {
  // 获取 CUDA 生成器或默认生成器
  auto gen = get_generator_or_default<CUDAGeneratorImpl>(gen_, cuda::detail::getDefaultCUDAGenerator());
  // 创建一个空的张量用于存放结果
  Tensor ret = at::empty(lambda.sizes(), lambda.options());
  // 调用 CUDA 内核函数进行泊松分布采样
  launch_poisson_cuda_kernel(ret, lambda, gen);
  // 返回结果张量
  return ret;
}

// CUDA 加速的二项分布采样函数
Tensor _s_binomial_cuda(const Tensor& count, const Tensor& prob, std::optional<Generator> gen_) {
  // 获取 CUDA 生成器或默认生成器
  auto gen = get_generator_or_default<CUDAGeneratorImpl>(gen_, cuda::detail::getDefaultCUDAGenerator());
  // 创建一个空的张量用于存放结果
  Tensor ret = at::empty(count.sizes(), count.options());
  // 构建张量迭代器，用于处理二项分布相关计算
  at::TensorIterator iter = at::TensorIteratorConfig()
      .add_output(ret)
      .add_input(count)
      .add_input(prob)
      .build();
  // 调用 CUDA 内核函数进行二项分布采样
  launch_binomial_cuda_kernel(iter, gen);
  // 返回结果张量
  return ret;
}

// CUDA 加速的 Gamma 分布采样函数
Tensor _s_gamma_cuda(const Tensor& alpha, std::optional<Generator> gen_) {
  // 获取 CUDA 生成器或默认生成器
  auto gen = get_generator_or_default<CUDAGeneratorImpl>(gen_, cuda::detail::getDefaultCUDAGenerator());
  // 创建一个空的张量用于存放结果
  Tensor ret = at::empty(alpha.sizes(), alpha.options());
  // 调用 CUDA 内核函数进行 Gamma 分布采样
  launch_gamma_kernel(ret, alpha, gen);
  // 返回结果张量
  return ret;
}

// CUDA 加速的 Dirichlet 分布采样函数
Tensor _s_dirichlet_cuda(const Tensor& alpha, std::optional<Generator> gen_) {
  // 获取 CUDA 生成器或默认生成器
  auto gen = get_generator_or_default<CUDAGeneratorImpl>(gen_, cuda::detail::getDefaultCUDAGenerator());
  // 创建一个空的张量用于存放结果
  Tensor ret = at::empty(alpha.sizes(), alpha.options());
  // 调用 CUDA 内核函数进行 Gamma 分布采样
  launch_gamma_kernel(ret, alpha, gen);
  // 计算 Dirichlet 分布中的总和并保持维度
  auto gamma_sum = ret.sum(/*dim=*/-1, /*keepdim=*/true);
  // 构建张量迭代器，用于处理 Dirichlet 分布相关计算
  at::TensorIterator iter = at::TensorIteratorConfig()
      .add_output(ret)
      .add_input(ret)
      .add_input(gamma_sum)
      .build();
  // 调用 CUDA 内核函数进行 Dirichlet 分布采样
  launch_dirichlet_kernel(iter);
  // 返回结果张量
  return ret;
}

// CUDA 加速的标准 Gamma 分布梯度函数
Tensor _standard_gamma_grad_cuda(const Tensor& self, const Tensor& output) {
  // 创建一个空的张量用于存放结果
  Tensor ret = at::empty(self.sizes(), self.options());
  // 构建张量迭代器，用于处理标准 Gamma 分布梯度相关计算
  TensorIterator iter = at::TensorIteratorConfig()
      .add_output(ret)
      .add_input(self)
      .add_input(output)
      .build();
  // 调用 CUDA 内核函数进行标准 Gamma 分布梯度计算
  launch_standard_gamma_grad_kernel(iter);
  // 返回结果张量
  return ret;
}

// CUDA 加速的 Dirichlet 分布梯度函数
Tensor _dirichlet_grad_cuda(const Tensor& x, const Tensor& alpha, const Tensor& total) {
  // 创建一个空的张量用于存放结果
  Tensor ret = at::empty(x.sizes(), x.options());
  // 构建张量迭代器，用于处理 Dirichlet 分布梯度相关计算
  TensorIterator iter = at::TensorIteratorConfig()
      .add_output(ret)
      .add_input(x)
      .add_input(alpha)
      .add_input(total)
      .build();
  // 调用 CUDA 内核函数进行 Dirichlet 分布梯度计算
  launch_dirichlet_grad_kernel(iter);
  // 返回结果张量
  return ret;
}

} // namespace at::native
```