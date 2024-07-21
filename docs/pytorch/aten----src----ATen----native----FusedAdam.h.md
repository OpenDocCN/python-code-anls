# `.\pytorch\aten\src\ATen\native\FusedAdam.h`

```
// 包含 ATen 库中的 Tensor 头文件
#include <ATen/core/Tensor.h>
// 包含 ATen 库中的分发函数声明头文件
#include <ATen/native/DispatchStub.h>

// 声明 at 命名空间
namespace at {

// 声明 native 命名空间，用于包含与本地操作相关的功能
namespace native {

// 定义 ADAM_MODE 枚举，表示 ADAM 优化器的两种模式：ORIGINAL 和 ADAMW
enum class ADAM_MODE : uint8_t { ORIGINAL = 0, ADAMW = 1 };

// 定义 fused_adam_fn 类型，是一个函数指针，用于执行融合的 Adam 优化操作
using fused_adam_fn = void (*)(
    const at::Tensor& param,                // 参数 Tensor
    const at::Tensor& grad,                 // 梯度 Tensor
    const at::Tensor& exp_avg,              // 指数加权平均 Tensor
    const at::Tensor& exp_avg_sq,           // 指数加权平均平方 Tensor
    const at::Tensor& max_exp_avg_sq,       // 最大指数加权平均平方 Tensor
    const at::Tensor& state_step,           // 状态步骤 Tensor
    const double lr,                        // 学习率
    const double beta1,                     // Adam 的 beta1 参数
    const double beta2,                     // Adam 的 beta2 参数
    const double weight_decay,              // 权重衰减
    const double eps,                       // 防止除零的小数
    const bool amsgrad,                     // 是否使用 AMSGrad 变种
    const bool maximize,                    // 是否最大化问题
    const float* grad_scale_ptr,            // 梯度缩放因子指针
    const ADAM_MODE);                      // Adam 优化器模式

// 声明 fused_adam_stub 函数，用于分发 fused_adam_fn 的实现
DECLARE_DISPATCH(fused_adam_fn, fused_adam_stub);

} // namespace native
} // namespace at
```