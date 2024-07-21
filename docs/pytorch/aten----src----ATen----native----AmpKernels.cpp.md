# `.\pytorch\aten\src\ATen\native\AmpKernels.cpp`

```
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/AmpKernels.h>
#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_amp_foreach_non_finite_check_and_unscale.h>
#include <ATen/ops/_amp_foreach_non_finite_check_and_unscale_native.h>
#include <ATen/ops/_amp_update_scale.h>
#include <ATen/ops/_amp_update_scale_native.h>
#endif

namespace at::native {

// 定义了用于 CPU 的非有限性检查和反缩放操作函数
void _amp_foreach_non_finite_check_and_unscale_cpu_(
    TensorList scaled_grads,
    at::Tensor& found_inf,
    const at::Tensor& inv_scale) {
    // 调用分发函数来执行 CPU 版本的非有限性检查和反缩放操作的具体实现
    _amp_foreach_non_finite_check_and_unscale_cpu_stub(
        found_inf.device().type(), scaled_grads, found_inf, inv_scale);
}

// 更新缩放比例函数的 CPU 实现
at::Tensor& _amp_update_scale_cpu_ (
    at::Tensor& current_scale,
    at::Tensor& growth_tracker,
    const at::Tensor& found_inf,
    double growth_factor,
    double backoff_factor,
    int64_t growth_interval) {
    // 调用分发函数来执行 CPU 版本的缩放比例更新操作的具体实现
    return _amp_update_scale_cpu_stub(
        growth_tracker.device().type(), current_scale, growth_tracker,
        found_inf, growth_factor, backoff_factor, growth_interval);
}

// 定义了用于 CPU 的非有限性检查和反缩放操作的分发器
DEFINE_DISPATCH(_amp_foreach_non_finite_check_and_unscale_cpu_stub);

// 定义了用于 CPU 的缩放比例更新操作的分发器
DEFINE_DISPATCH(_amp_update_scale_cpu_stub);

} // namespace at::native
```