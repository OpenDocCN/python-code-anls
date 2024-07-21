# `.\pytorch\aten\src\ATen\native\PointwiseOps.h`

```py
// 包含一次性预处理指令，确保头文件只被编译一次
#pragma once

// 引入分发存根头文件，用于实现分发的抽象接口
#include <ATen/native/DispatchStub.h>

// 声明命名空间 c10 下的 Scalar 类
namespace c10 {
    class Scalar;
}

// 声明命名空间 at 下的结构体 TensorIterator 和 TensorIteratorBase
namespace at {

    // 用于描述张量迭代器的结构体
    struct TensorIterator;
    // 用于描述张量迭代器基类的结构体
    struct TensorIteratorBase;

    // 声明命名空间 native 下的函数指针类型 pointwise_fn，用于指向点对点操作函数
    using pointwise_fn = void (*)(TensorIterator&, const Scalar& scalar);
    // 声明命名空间 native 下的函数指针类型 structured_pointwise_fn，用于指向结构化点对点操作函数
    using structured_pointwise_fn = void (*)(TensorIteratorBase&, const Scalar& scalar);
    // 声明命名空间 native 下的函数指针类型 pointwise_fn_double，用于指向双参数点对点操作函数
    using pointwise_fn_double = void (*)(TensorIterator&, const Scalar&, double);

    // 声明命名空间 native 下的结构化点对点操作的分发存根，用于加法乘积操作
    DECLARE_DISPATCH(structured_pointwise_fn, addcmul_stub);
    // 声明命名空间 native 下的结构化点对点操作的分发存根，用于加法除法操作
    DECLARE_DISPATCH(structured_pointwise_fn, addcdiv_stub);
    // 声明命名空间 native 下的双参数点对点操作的分发存根，用于 smooth L1 损失函数反向传播
    DECLARE_DISPATCH(pointwise_fn_double, smooth_l1_backward_stub);
    // 声明命名空间 native 下的双参数点对点操作的分发存根，用于 Huber 损失函数反向传播
    DECLARE_DISPATCH(pointwise_fn_double, huber_backward_stub);
    // 声明命名空间 native 下的点对点操作的分发存根，用于均方误差损失函数反向传播
    DECLARE_DISPATCH(pointwise_fn, mse_backward_stub);

} // namespace at
```