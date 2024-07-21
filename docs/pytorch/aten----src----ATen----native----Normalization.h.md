# `.\pytorch\aten\src\ATen\native\Normalization.h`

```
#pragma once
// 预处理指令，指示编译器仅包含本文件一次

#include <ATen/TensorIterator.h>
// 包含 ATen 库中的 TensorIterator 头文件

#include <ATen/native/DispatchStub.h>
// 包含 ATen 库中的 DispatchStub 头文件

namespace at::native {
// 进入 at::native 命名空间

using renorm_scale_factor_fn = void (*) (TensorIteratorBase& iter, double maxnorm);
// 定义 renorm_scale_factor_fn 类型别名，表示一个函数指针，接受 TensorIteratorBase 和 double 参数，无返回值
DECLARE_DISPATCH(renorm_scale_factor_fn, renorm_scale_factor_stub);
// 声明 renorm_scale_factor_stub 函数的调度分发，用于将 renorm_scale_factor_fn 函数指针与具体实现关联

enum class BatchNormBackend {
  Native,
  Cudnn,
  Miopen,
};
// 定义 BatchNormBackend 枚举类，表示批归一化的后端实现选择，可选值有 Native、Cudnn、Miopen

TORCH_API BatchNormBackend _select_batch_norm_backend(const Tensor& input, const Tensor& weight, const Tensor& bias, const Tensor& running_mean, const Tensor& running_var, bool training, double eps);
// 声明 _select_batch_norm_backend 函数，用于根据输入张量的特征选择批归一化的后端实现，并返回选择的后端类型

}  // namespace at::native
// 结束 at::native 命名空间
```