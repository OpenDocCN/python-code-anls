# `.\pytorch\aten\src\ATen\native\cpu\GridSamplerKernel.h`

```py
#pragma once
// 使用预处理指令#pragma once确保此头文件只被编译一次

#include <ATen/native/DispatchStub.h>
// 包含ATen库中的DispatchStub.h头文件，用于处理调度相关的功能

#include <array>
// 包含标准库中的array头文件，用于定义数组类型

#include <cstdint>
// 包含标准库中的cstdint头文件，定义了固定宽度的整数类型

namespace at {
class TensorBase;
}
// 命名空间at中声明了一个名为TensorBase的类

namespace at { namespace native {

using forward_2d_fn = void (*) (
    const TensorBase &output,
    const TensorBase &input,
    const TensorBase &grid,
    int64_t interpolation_mode,
    int64_t padding_mode,
    bool align_corners);
// 定义了一个函数指针类型forward_2d_fn，指向一个函数，该函数接受多个参数，包括TensorBase类型的引用、整数类型和布尔类型

using backward_2d_fn = void (*) (
    const TensorBase &grad_input,
    const TensorBase &grad_grid,
    const TensorBase &grad_output,
    const TensorBase &input,
    const TensorBase &grid,
    int64_t interpolation_mode,
    int64_t padding_mode,
    bool align_corners,
    std::array<bool, 2> output_mask);
// 定义了一个函数指针类型backward_2d_fn，指向一个函数，该函数接受多个参数，包括TensorBase类型的引用、整数类型、布尔类型和std::array<bool, 2>类型

DECLARE_DISPATCH(forward_2d_fn, grid_sampler_2d_cpu_kernel);
// 使用宏DECLARE_DISPATCH声明了一个名称为grid_sampler_2d_cpu_kernel的函数，该函数类型为forward_2d_fn

DECLARE_DISPATCH(backward_2d_fn, grid_sampler_2d_backward_cpu_kernel);
// 使用宏DECLARE_DISPATCH声明了一个名称为grid_sampler_2d_backward_cpu_kernel的函数，该函数类型为backward_2d_fn

}}  // namespace at::native
// 结束命名空间at::native
```