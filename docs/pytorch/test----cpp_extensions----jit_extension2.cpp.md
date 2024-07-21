# `.\pytorch\test\cpp_extensions\jit_extension2.cpp`

```py
#include <torch/extension.h>

// 引入 Torch C++ 扩展库头文件

using namespace at;

// 使用 Torch 命名空间

// 定义函数 exp_add，接受两个 Tensor 类型参数 x 和 y，返回它们的指数函数之和
Tensor exp_add(Tensor x, Tensor y) {
    // 返回 x 和 y 的指数函数之和
    return x.exp() + y.exp();
}
```