# `.\pytorch\aten\src\ATen\native\xnnpack\Pooling.h`

```
#pragma once
// 预处理指令，确保头文件只包含一次

#ifdef USE_XNNPACK
// 如果定义了 USE_XNNPACK 宏，则包含以下内容

#include <ATen/Tensor.h>
// 包含 ATen 库的 Tensor 头文件

namespace at::native::xnnpack::internal::pooling {
// 进入命名空间 at::native::xnnpack::internal::pooling

struct Parameters final {
// 定义结构体 Parameters，使用 final 修饰，表示不可继承

  std::array<int64_t, 2> kernel;
  // 存储 kernel 大小的数组，大小为 2

  std::array<int64_t, 2> padding;
  // 存储 padding 大小的数组，大小为 2

  std::array<int64_t, 2> stride;
  // 存储 stride 大小的数组，大小为 2

  std::array<int64_t, 2> dilation;
  // 存储 dilation 大小的数组，大小为 2

  explicit Parameters(
      const IntArrayRef kernel_,
      const IntArrayRef padding_,
      const IntArrayRef stride_,
      const IntArrayRef dilation_)
  : kernel(normalize(kernel_)),
    padding(normalize(padding_)),
    stride(normalize(stride_)),
    dilation(normalize(dilation_)) {
  }
  // 参数化构造函数，使用参数列表初始化各成员变量

private:
  static std::array<int64_t, 2> normalize(const IntArrayRef parameter) {
    // 定义静态函数 normalize，接收 IntArrayRef 类型的参数

    TORCH_INTERNAL_ASSERT(
        !parameter.empty(),
        "Invalid usage!  Reason: normalize() was called on an empty parameter.");
    // 断言，如果 parameter 不为空，否则抛出异常信息

    return std::array<int64_t, 2>{
      parameter[0],
      (2 == parameter.size()) ? parameter[1] : parameter[0],
    };
    // 返回一个大小为 2 的 std::array，根据 parameter 的大小进行填充
  }
};

} // namespace at::native::xnnpack::internal::pooling
// 结束命名空间 at::native::xnnpack::internal::pooling

#endif /* USE_XNNPACK */
// 结束条件编译块，检查 USE_XNNPACK 宏是否定义
```