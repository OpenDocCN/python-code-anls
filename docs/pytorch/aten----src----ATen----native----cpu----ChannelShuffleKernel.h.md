# `.\pytorch\aten\src\ATen\native\cpu\ChannelShuffleKernel.h`

```py
#pragma once
// 使用 #pragma once 指令确保此头文件只被编译一次，避免重复包含

#include <ATen/native/DispatchStub.h>
// 包含 ATen 库中的 DispatchStub.h 头文件，用于定义调度相关的功能

#include <cstdint>
// 包含标准整数类型头文件，提供标准整数类型的定义，如 int64_t

namespace at {
// 命名空间 at，用于包含 ATen 库中的相关内容

class TensorBase;
// 前置声明 TensorBase 类，表示这是一个类的声明，但不会实际定义其成员和方法

}

namespace at { namespace native {
// 命名空间 at::native，用于包含 ATen 库中的本地（native）实现相关内容

using channel_shuffle_fn = void(*)(TensorBase&, const TensorBase&, int64_t);
// 定义 channel_shuffle_fn 类型别名，表示指向函数的指针，该函数接受两个 TensorBase 类型的参数和一个 int64_t 类型的参数，并返回 void

DECLARE_DISPATCH(channel_shuffle_fn, channel_shuffle_kernel);
// 声明一个宏，用于声明并定义 channel_shuffle_kernel 函数的调度分发，该函数的类型为 channel_shuffle_fn

}} // at::native
// 命名空间 at::native 结束
```