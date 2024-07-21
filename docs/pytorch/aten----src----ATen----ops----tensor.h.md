# `.\pytorch\aten\src\ATen\ops\tensor.h`

```
#pragma once
// 使用 pragma once 指令确保头文件只被编译一次

#include <ATen/core/Tensor.h>
#include <c10/core/ScalarType.h>

namespace at {

// 在 at 命名空间内定义一系列函数，这些函数在 ATen/Utils.cpp 中实现

#define TENSOR(T, S)                                                          \
  // 宏 TENSOR 接受类型 T 和标量类型 S 作为参数

  // 创建一个张量对象，用给定的值数组和选项创建
  TORCH_API Tensor tensor(ArrayRef<T> values, const TensorOptions& options);

  // 使用初始化列表创建张量对象，给定值和选项
  inline Tensor tensor(
      std::initializer_list<T> values, const TensorOptions& options) {
    return at::tensor(ArrayRef<T>(values), options);
  }

  // 使用单个值和选项创建张量对象
  inline Tensor tensor(T value, const TensorOptions& options) {
    return at::tensor(ArrayRef<T>(value), options);
  }

  // 使用值数组创建张量对象，默认选项为 k##S 类型
  inline Tensor tensor(ArrayRef<T> values) {
    return at::tensor(std::move(values), at::dtype(k##S));
  }

  // 使用初始化列表创建张量对象
  inline Tensor tensor(std::initializer_list<T> values) {
    return at::tensor(ArrayRef<T>(values));
  }

  // 使用单个值创建张量对象
  inline Tensor tensor(T value) {
    return at::tensor(ArrayRef<T>(value));
  }
// 宏展开结束

// 将宏 TENSOR 应用于所有标量类型和 Bool、Half、BFloat16 类型
AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, TENSOR)

// 将宏 TENSOR 应用于所有复数类型
AT_FORALL_COMPLEX_TYPES(TENSOR)

#undef TENSOR
// 取消定义宏 TENSOR

}  // namespace at
// 命名空间 at 结束
```