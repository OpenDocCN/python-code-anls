# `.\pytorch\aten\src\ATen\native\cpu\WeightNormKernel.h`

```
#pragma once
// 预处理指令：确保本文件只被编译一次

#include <ATen/native/DispatchStub.h>
// 包含 ATen 库中的 DispatchStub.h 文件

#include <cstdint>
// 包含标准整数类型定义的头文件

namespace at {
// 命名空间 at，用于 ATen 库的组件

class TensorBase;
// 前置声明 TensorBase 类

}

namespace at { namespace native {
// 命名空间 at::native，用于 ATen 库的本地实现部分

using weight_norm_fn = void(*)(
    TensorBase&, TensorBase&, const TensorBase&, const TensorBase&, int64_t);
// 定义 weight_norm_fn 类型别名，表示一个函数指针，接受五个参数，返回 void

using weight_norm_backward_fn = void(*)(
    TensorBase&, TensorBase&, const TensorBase&, const TensorBase&,
    const TensorBase&, const TensorBase&, int64_t);
// 定义 weight_norm_backward_fn 类型别名，表示一个函数指针，接受六个参数，返回 void

DECLARE_DISPATCH(weight_norm_fn, weight_norm_stub);
// 声明 weight_norm_stub 函数的分发接口，用于权重归一化

DECLARE_DISPATCH(weight_norm_backward_fn, weight_norm_backward_stub);
// 声明 weight_norm_backward_stub 函数的分发接口，用于权重归一化的反向传播

}}  // namespace at::native
// 结束命名空间 at::native
```