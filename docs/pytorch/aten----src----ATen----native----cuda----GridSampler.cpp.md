# `.\pytorch\aten\src\ATen\native\cuda\GridSampler.cpp`

```
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/cuda/GridSampler.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/grid_sampler_2d_backward_native.h>
#include <ATen/ops/grid_sampler_2d_native.h>
#include <ATen/ops/grid_sampler_3d_backward_native.h>
#include <ATen/ops/grid_sampler_3d_native.h>
#include <ATen/ops/zeros_like.h>
#endif

namespace at::native {

// CUDA实现的二维网格采样函数，用于处理输入张量和网格张量
Tensor grid_sampler_2d_cuda(const Tensor& input, const Tensor& grid,
                            int64_t interpolation_mode, int64_t padding_mode,
                            bool align_corners) {
  // 获取输入张量的尺寸
  auto in_size = input.sizes();
  // 获取网格张量的尺寸
  auto grid_size = grid.sizes();
  // 创建一个空的输出张量，形状为 [输入通道数, 输出通道数, 网格高度, 网格宽度]
  auto output = at::empty(
      {in_size[0], in_size[1], grid_size[1], grid_size[2]}, input.options());
  // 调用 CUDA 内核函数执行二维网格采样前向计算
  launch_grid_sampler_2d_forward_kernel(
      output, input, grid, interpolation_mode, padding_mode, align_corners);
  // 返回计算结果的输出张量
  return output;
}

// CUDA实现的三维网格采样函数，用于处理输入张量和网格张量
Tensor grid_sampler_3d_cuda(const Tensor& input, const Tensor& grid,
                            int64_t interpolation_mode, int64_t padding_mode,
                            bool align_corners) {
  // 获取输入张量的尺寸
  auto in_size = input.sizes();
  // 获取网格张量的尺寸
  auto grid_size = grid.sizes();
  // 创建一个空的输出张量，形状为 [输入通道数, 输出通道数, 网格深度, 网格高度, 网格宽度]
  auto output = at::empty(
      {in_size[0], in_size[1], grid_size[1], grid_size[2], grid_size[3]},
      input.options());
  // 调用 CUDA 内核函数执行三维网格采样前向计算
  launch_grid_sampler_3d_forward_kernel(
      output, input, grid, interpolation_mode, padding_mode, align_corners);
  // 返回计算结果的输出张量
  return output;
}

// CUDA实现的二维网格采样反向传播函数，计算梯度
std::tuple<Tensor, Tensor>
grid_sampler_2d_backward_cuda(const Tensor& grad_output, const Tensor& input,
                              const Tensor& grid, int64_t interpolation_mode, int64_t padding_mode,
                              bool align_corners, std::array<bool, 2> output_mask) {
  // 根据是否需要梯度，创建对应的梯度输入张量
  auto input_requires_grad = output_mask[0];
  Tensor grad_input = ([&]() {
    if (input_requires_grad) {
      // 如果需要梯度，创建一个与输入张量形状相同的零张量
      return at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    } else {
      // 如果不需要梯度，返回一个空张量
      return Tensor();
    }
  })();
  // 创建一个与网格张量形状相同的空张量，用于存储网格的梯度
  auto grad_grid = at::empty_like(grid, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  // 调用 CUDA 内核函数执行二维网格采样反向传播计算
  launch_grid_sampler_2d_backward_kernel(
      grad_input, grad_grid, grad_output, input,
      grid, interpolation_mode, padding_mode, align_corners, output_mask);
  // 返回梯度计算结果，包括输入张量和网格张量的梯度
  return std::make_tuple(grad_input, grad_grid);
}

// CUDA实现的三维网格采样反向传播函数，计算梯度
std::tuple<Tensor, Tensor>
grid_sampler_3d_backward_cuda(const Tensor& grad_output, const Tensor& input,
                              const Tensor& grid, int64_t interpolation_mode, int64_t padding_mode,
                              bool align_corners, std::array<bool,2> output_mask) {
  // 根据是否需要梯度，创建对应的梯度输入张量
  auto input_requires_grad = output_mask[0];
  Tensor grad_input = ([&]() {
    if (input_requires_grad) {
      // 如果需要梯度，创建一个与输入张量形状相同的零张量
      return at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    } else {
      // 如果不需要梯度，返回一个空张量
      return Tensor();
    }
  })();
  // 创建一个与网格张量形状相同的空张量，用于存储网格的梯度
  auto grad_grid = at::empty_like(grid, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  // 调用 CUDA 内核函数执行三维网格采样反向传播计算
  launch_grid_sampler_3d_backward_kernel(
      grad_input, grad_grid, grad_output, input,
      grid, interpolation_mode, padding_mode, align_corners, output_mask);
  // 返回梯度计算结果，包括输入张量和网格张量的梯度
  return std::make_tuple(grad_input, grad_grid);
}

} // namespace at::native
    }
  })();
  // 创建一个与 'grid' 相同大小的空张量 'grad_grid'，用于存储梯度信息，使用传统的内存布局
  auto grad_grid = at::empty_like(grid, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  // 调用 CUDA 核函数 'launch_grid_sampler_3d_backward_kernel'，执行三维网格采样的反向传播
  // 传入参数为：输入梯度 'grad_input'，网格梯度 'grad_grid'，输出梯度 'grad_output'，输入数据 'input'，
  // 网格数据 'grid'，插值模式 'interpolation_mode'，填充模式 'padding_mode'，角落对齐 'align_corners'，输出掩码 'output_mask'
  launch_grid_sampler_3d_backward_kernel(
      grad_input, grad_grid, grad_output, input,
      grid, interpolation_mode, padding_mode, align_corners, output_mask);
  // 返回包含 'grad_input' 和 'grad_grid' 的元组，表示计算得到的输入和网格的梯度
  return std::make_tuple(grad_input, grad_grid);
}

}  // namespace at::native
```