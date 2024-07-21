# `.\pytorch\aten\src\ATen\native\Copy.h`

```
#pragma once
// 使用#pragma once确保头文件只被编译一次，防止重复包含

#include <ATen/native/DispatchStub.h>
// 包含ATen库中的DispatchStub.h头文件，用于声明和定义调度相关的接口和函数

namespace at {
// 命名空间at，包含了所有的ATen库相关的内容

class Tensor;
// 声明一个Tensor类，用于表示张量对象

struct TensorIterator;
// 声明一个TensorIterator结构体，用于迭代张量的迭代器对象

class TensorBase;
// 声明一个TensorBase类，作为Tensor的基类

namespace native {
// 命名空间native，用于包含ATen库中的原生实现相关内容

using copy_fn = void (*)(TensorIterator&, bool non_blocking);
// 定义了一个函数指针类型copy_fn，接受一个TensorIterator引用和一个bool参数

DECLARE_DISPATCH(copy_fn, copy_stub);
// 使用宏DECLARE_DISPATCH声明了一个名为copy_stub的调度函数，其类型为copy_fn

TORCH_API void copy_ignoring_overlaps(const TensorBase &dst, const TensorBase &src);
// 声明了一个名为copy_ignoring_overlaps的函数，用于在忽略重叠情况下从src复制到dst

} // namespace native
} // namespace at
```