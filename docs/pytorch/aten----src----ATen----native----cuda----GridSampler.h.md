# `.\pytorch\aten\src\ATen\native\cuda\GridSampler.h`

```
#pragma once
// 使用预处理指令#pragma once，确保此头文件只被编译一次

#include <array>
// 包含标准数组头文件<array>

#include <cstdint>
// 包含标准整数类型头文件<stdint.h>

namespace at {
// 命名空间at开始

class TensorBase;
// 声明TensorBase类，用于表示张量基类

}

namespace at {
namespace native {
// 命名空间at::native开始，用于包含本地（native）实现

void launch_grid_sampler_2d_forward_kernel(
    const TensorBase &output, const TensorBase &input, const TensorBase &grid,
    int64_t interpolation_mode, int64_t padding_mode, bool align_corners);
// 声明launch_grid_sampler_2d_forward_kernel函数，执行2D网格采样前向传播计算，
// 参数包括输出张量output、输入张量input、网格张量grid、插值模式interpolation_mode、填充模式padding_mode和是否对齐角点align_corners

void launch_grid_sampler_3d_forward_kernel(
    const TensorBase &output, const TensorBase &input, const TensorBase &grid,
    int64_t interpolation_mode, int64_t padding_mode, bool align_corners);
// 声明launch_grid_sampler_3d_forward_kernel函数，执行3D网格采样前向传播计算，
// 参数与上述相同

void launch_grid_sampler_2d_backward_kernel(
    const TensorBase &grad_input, const TensorBase &grad_grid,
    const TensorBase &grad_output, const TensorBase &input,
    const TensorBase &grid, int64_t interpolation_mode, int64_t padding_mode,
    bool align_corners, std::array<bool, 2> output_mask);
// 声明launch_grid_sampler_2d_backward_kernel函数，执行2D网格采样反向传播计算，
// 参数包括梯度输入grad_input、梯度网格grad_grid、梯度输出grad_output、输入张量input、网格张量grid、插值模式interpolation_mode、填充模式padding_mode、是否对齐角点align_corners和输出掩码output_mask

void launch_grid_sampler_3d_backward_kernel(
    const TensorBase &grad_input, const TensorBase &grad_grid,
    const TensorBase &grad_output, const TensorBase &input,
    const TensorBase &grid, int64_t interpolation_mode, int64_t padding_mode,
    bool align_corners, std::array<bool, 2> output_mask);
// 声明launch_grid_sampler_3d_backward_kernel函数，执行3D网格采样反向传播计算，
// 参数与上述相同

}}  // namespace at::native
// 命名空间at::native结束
```