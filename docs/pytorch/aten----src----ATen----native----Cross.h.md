# `.\pytorch\aten\src\ATen\native\Cross.h`

```py
#pragma once
// 预处理指令，确保头文件仅被包含一次

#include <ATen/native/DispatchStub.h>
// 包含 ATen 库的 DispatchStub.h 头文件

namespace at {
class Tensor;
// 声明 at 命名空间下的 Tensor 类

namespace native {
// 声明 native 命名空间

using cross_fn = void(*)(const Tensor&, const Tensor&, const Tensor&, const int64_t d);
// 定义 cross_fn 类型别名，表示一个指向函数的指针，该函数接受三个 Tensor 对象和一个 int64_t 参数

DECLARE_DISPATCH(cross_fn, cross_stub);
// 声明一个用于分发的 cross_stub 函数，其函数指针类型为 cross_fn

}} // namespace at::native
// 结束 at 和 native 命名空间的定义
```