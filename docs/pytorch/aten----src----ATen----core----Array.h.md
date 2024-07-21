# `.\pytorch\aten\src\ATen\core\Array.h`

```py
#pragma once
// 表示该头文件只被编译一次的预处理指令

// A fixed-size array type usable from both host and
// device code.
// 一个可用于主机和设备代码的固定大小数组类型

#include <c10/macros/Macros.h>
// 包含C10库的宏定义头文件
#include <c10/util/irange.h>
// 包含C10库的irange工具头文件

namespace at::detail {

template <typename T, int size_>
struct Array {
  // NOLINTNEXTLINE(*c-array*)
  // 禁止lint检查下一行的C数组警告（针对Lint工具的指令）

  T data[size_];
  // 固定大小的数组，存储类型为T，大小为size_

  C10_HOST_DEVICE T operator[](int i) const {
    return data[i];
  }
  // 返回索引为i的元素值的常量访问操作符重载

  C10_HOST_DEVICE T& operator[](int i) {
    return data[i];
  }
  // 返回索引为i的元素的引用访问操作符重载

#if defined(USE_ROCM)
  C10_HOST_DEVICE Array() = default;
  C10_HOST_DEVICE Array(const Array&) = default;
  C10_HOST_DEVICE Array& operator=(const Array&) = default;
#else
  Array() = default;
  Array(const Array&) = default;
  Array& operator=(const Array&) = default;
#endif
  // 根据定义的宏USE_ROCM来选择性地生成默认的构造函数、拷贝构造函数和赋值操作符重载

  static constexpr int size() {
    return size_;
  }
  // 返回数组的大小作为编译时常量的静态成员函数

  // Fill the array with x.
  // 用值x填充整个数组
  C10_HOST_DEVICE Array(T x) {
    for (int i = 0; i < size_; i++) {
      data[i] = x;
    }
  }
  // 构造函数，用值x填充数组中的每个元素

};

} // namespace at::detail
// 命名空间at::detail结束
```