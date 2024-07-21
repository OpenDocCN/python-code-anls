# `.\pytorch\aten\src\ATen\native\AmpKernels.h`

```
#pragma once
// 预处理命令，确保头文件只被编译一次

#include <ATen/native/DispatchStub.h>
// 包含 ATen 库中的 DispatchStub 头文件，用于声明调度分发相关的功能

#include <ATen/core/ATen_fwd.h>
// 包含 ATen 库中的 ATen_fwd 头文件，用于前向声明 ATen 核心类和函数

namespace at {
// 定义 at 命名空间，包含了 ATen 库的核心功能

class Tensor;
// 声明 Tensor 类，这是 ATen 库中的核心数据类型

namespace native {
// 定义 native 命名空间，包含了 ATen 库的本地实现细节

using _amp_foreach_non_finite_check_and_unscale_cpu__fn = void (*)(
    TensorList,
    Tensor&,
    const Tensor&);
// 定义 _amp_foreach_non_finite_check_and_unscale_cpu__fn 类型别名，表示一个函数指针，
// 该函数接受 TensorList、Tensor 和 const Tensor& 参数，并返回 void

using _amp_update_scale_cpu__fn = Tensor& (*)(
    Tensor&,
    Tensor&,
    const Tensor&,
    double,
    double,
    int64_t);
// 定义 _amp_update_scale_cpu__fn 类型别名，表示一个函数指针，
// 该函数接受 Tensor&、Tensor&、const Tensor&、double、double 和 int64_t 参数，并返回 Tensor&

DECLARE_DISPATCH(_amp_foreach_non_finite_check_and_unscale_cpu__fn, _amp_foreach_non_finite_check_and_unscale_cpu_stub);
// 声明 _amp_foreach_non_finite_check_and_unscale_cpu_stub，它是一个函数分发器，
// 接受 _amp_foreach_non_finite_check_and_unscale_cpu__fn 类型的函数指针作为调度目标

DECLARE_DISPATCH(_amp_update_scale_cpu__fn, _amp_update_scale_cpu_stub);
// 声明 _amp_update_scale_cpu_stub，它是一个函数分发器，
// 接受 _amp_update_scale_cpu__fn 类型的函数指针作为调度目标

} // namespace native
} // namespace at
```