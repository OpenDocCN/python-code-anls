# `.\pytorch\aten\src\ATen\native\TensorProperties.h`

```
#pragma once
// 一般用于头文件，表示此文件只需包含一次

// See NOTE: [Tensor vs. TensorBase]
// 这段代码注释是为了指向某个特定的注释或者文档部分，提供进一步的阅读说明

namespace at {
class TensorBase;
// 声明一个名为 TensorBase 的类
}

namespace at::native {

// 声明一个函数 cudnn_is_acceptable，用于判断是否可以使用 cudnn 运行
TORCH_API bool cudnn_is_acceptable(const TensorBase& self);

} // namespace at::native
// 结束 at::native 命名空间的定义
```