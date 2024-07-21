# `.\pytorch\aten\src\ATen\native\cpu\SampledAddmmKernel.h`

```
#pragma once
// 预处理指令，用于确保头文件只被包含一次

#include <ATen/core/Tensor.h>
// 包含 ATen 库的 Tensor 类的头文件

#include <ATen/native/DispatchStub.h>
// 包含 ATen 库的 DispatchStub 头文件

namespace at { namespace native {

using sampled_addmm_sparse_csr_fn = void(*)(const Tensor&, const Tensor&, const Scalar&, const Scalar&, const Tensor&);
// 定义 sampled_addmm_sparse_csr_fn 类型别名，表示一个函数指针类型，接受五个参数：两个 Tensor 对象、两个 Scalar 对象和一个 Tensor 对象，并且返回 void。

DECLARE_DISPATCH(sampled_addmm_sparse_csr_fn, sampled_addmm_sparse_csr_stub);
// 声明 sampled_addmm_sparse_csr_stub 函数的分发函数，它是 sampled_addmm_sparse_csr_fn 类型的指针。

}} // at::native
// 命名空间结束
```