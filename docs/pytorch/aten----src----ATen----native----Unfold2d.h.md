# `.\pytorch\aten\src\ATen\native\Unfold2d.h`

```py
#pragma once
// 预处理命令，确保本头文件只被编译一次

#include <ATen/native/DispatchStub.h>
// 包含 ATen 库的 DispatchStub.h 头文件，用于声明分发函数相关内容

#include <c10/core/ScalarType.h>
// 包含 c10 库的 ScalarType.h 头文件，用于声明标量类型相关内容

#include <cstdint>
// 包含标准整数类型库，提供固定大小整数类型定义

namespace at::native {

using unfold2d_copy_fn = void (*)(
    ScalarType dtype,
    void *finput,
    const void *input,
    int64_t kH,
    int64_t kW,
    int64_t dH,
    int64_t dW,
    int64_t padH,
    int64_t padW,
    int64_t n_input_plane,
    int64_t input_height,
    int64_t input_width,
    int64_t output_height,
    int64_t output_width,
    bool is_channels_last
);
// 定义一个函数指针类型 unfold2d_copy_fn，用于表示 unfold2d 操作的复制函数签名

using unfold2d_acc_fn = void (*)(
    ScalarType dtype,
    void *finput,
    void *input,
    int64_t kH,
    int64_t kW,
    int64_t dH,
    int64_t dW,
    int64_t padH,
    int64_t padW,
    int64_t n_input_plane,
    int64_t input_height,
    int64_t input_width,
    int64_t output_height,
    int64_t output_width,
    bool is_channels_last
);
// 定义一个函数指针类型 unfold2d_acc_fn，用于表示 unfold2d 操作的累加函数签名

DECLARE_DISPATCH(unfold2d_copy_fn, unfolded2d_copy_stub);
// 声明宏 DECLARE_DISPATCH，将 unfold2d_copy_fn 函数指针类型声明为 unfolded2d_copy_stub 的分发接口

DECLARE_DISPATCH(unfold2d_acc_fn, unfolded2d_acc_stub);
// 声明宏 DECLARE_DISPATCH，将 unfold2d_acc_fn 函数指针类型声明为 unfolded2d_acc_stub 的分发接口

} // namespace at::native
// 命名空间结束，作用域限定符
```