# `.\pytorch\aten\src\ATen\native\FusedSGD.h`

```
// 引入 ATen 核心库中的 Tensor 头文件
#include <ATen/core/Tensor.h>
// 引入 ATen 中的分发存根头文件
#include <ATen/native/DispatchStub.h>

// 命名空间 at 开始
namespace at {

// 命名空间 native 开始，用于定义原生实现的函数和类型
namespace native {

// 定义一个函数指针类型 fused_sgd_fn，用于表示 fused_sgd 函数的类型
using fused_sgd_fn = void (*)(
    const at::Tensor& param,                   // 参数张量
    const at::Tensor& grad,                    // 梯度张量
    const at::Tensor& momentum_buffer,         // 动量缓存张量
    const double weight_decay,                 // 权重衰减参数
    const double momentum,                     // 动量参数
    const double lr,                           // 学习率参数
    const double dampening,                    // 阻尼参数
    const bool nesterov,                       // 是否使用 Nesterov 动量
    const bool maximize,                       // 是否最大化优化目标
    const bool is_first_step,                  // 是否第一步
    const float* grad_scale_ptr);              // 梯度缩放指针

// 声明 fused_sgd_stub 函数，其类型为 fused_sgd_fn，用于分发 fused_sgd 函数
DECLARE_DISPATCH(fused_sgd_fn, fused_sgd_stub);

// 命名空间 native 结束
}

// 命名空间 at 结束
}
```