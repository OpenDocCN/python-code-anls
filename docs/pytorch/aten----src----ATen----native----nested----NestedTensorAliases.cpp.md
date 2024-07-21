# `.\pytorch\aten\src\ATen\native\nested\NestedTensorAliases.cpp`

```py
#include <ATen/ATen.h>  // 引入 ATen 库，用于张量操作

namespace at {
namespace native {

// 在本地命名空间中定义 nested_to_padded_tensor 的别名函数
Tensor nested_to_padded_tensor(
    const Tensor& t,  // 参数 t，类型为 Tensor 引用，表示输入张量
    double padding,   // 参数 padding，类型为 double，表示填充值
    OptionalIntArrayRef output_size) {  // 参数 output_size，可选的整数数组引用，表示输出大小

    // 调用张量 t 的 to_padded_tensor 方法，使用给定的填充值和输出大小
    return t.to_padded_tensor(padding, output_size);
}

} // namespace native
} // namespace at
```