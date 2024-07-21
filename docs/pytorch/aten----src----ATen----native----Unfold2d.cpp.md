# `.\pytorch\aten\src\ATen\native\Unfold2d.cpp`

```py
// 定义宏 TORCH_ASSERT_ONLY_METHOD_OPERATORS，用于在编译时启用特定功能
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 包含 ATen 库中的 Unfold2d.h 头文件，用于实现二维展开操作
#include <ATen/native/Unfold2d.h>

// 进入 at::native 命名空间
namespace at::native {

// 定义 unfolded2d_copy_stub 函数的调度器（分发器）
DEFINE_DISPATCH(unfolded2d_copy_stub);
// 定义 unfolded2d_acc_stub 函数的调度器（分发器）
DEFINE_DISPATCH(unfolded2d_acc_stub);

} // namespace at::native
```