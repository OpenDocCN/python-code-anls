# `.\pytorch\aten\src\ATen\native\metal\ops\MetalCopy.h`

```
#ifndef MetalCopy_h
#define MetalCopy_h

// 如果 MetalCopy_h 还没有被定义，则进入条件编译，防止多次包含该头文件


#include <ATen/Tensor.h>

// 包含 ATen 库中的 Tensor 头文件，以便使用其中定义的 Tensor 类型和相关功能


namespace at::native::metal {

// 进入 at::native::metal 命名空间，定义下面的函数或者类型在这个命名空间中


Tensor copy_to_host(const Tensor& input);

// 声明一个名为 copy_to_host 的函数，其参数为一个常量引用类型的 Tensor 对象，并且函数的返回类型为 Tensor


} // namespace at::native::metal

// 结束 at::native::metal 命名空间的定义


#endif

// 结束条件编译指令，确保 MetalCopy_h 只被包含一次，防止重复定义的问题
```