# `.\pytorch\aten\src\ATen\native\Distance.h`

```py
#pragma once


// 一次性预处理指令，确保头文件只被编译一次
#include <ATen/native/DispatchStub.h>

namespace at {
class Tensor;

namespace native {

// 定义函数指针类型 pdist_forward_fn，用于计算距离的前向传播
using pdist_forward_fn = void(*)(Tensor&, const Tensor&, const double p);
// 定义函数指针类型 pdist_backward_fn，用于计算距离的反向传播
using pdist_backward_fn = void(*)(Tensor&, const Tensor&, const Tensor&, const double p, const Tensor&);
// 定义函数指针类型 cdist_fn，用于计算距离的函数
using cdist_fn = void(*)(Tensor&, const Tensor&, const Tensor&, const double p);
// 定义函数指针类型 cdist_backward_fn，用于计算距离的反向传播
using cdist_backward_fn = void(*)(Tensor&, const Tensor&, const Tensor&, const Tensor&, const double p, const Tensor&);

// 声明分发函数的前向声明，用于距离计算的分发
DECLARE_DISPATCH(pdist_forward_fn, pdist_forward_stub);
// 声明分发函数的前向声明，用于距离计算反向传播的分发
DECLARE_DISPATCH(pdist_backward_fn, pdist_backward_stub);
// 声明距离计算的分发函数
DECLARE_DISPATCH(cdist_fn, cdist_stub);
// 声明距离计算反向传播的分发函数
DECLARE_DISPATCH(cdist_backward_fn, cdist_backward_stub);

}} // namespace at::native
```