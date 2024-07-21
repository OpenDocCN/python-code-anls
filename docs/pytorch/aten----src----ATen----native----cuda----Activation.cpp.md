# `.\pytorch\aten\src\ATen\native\cuda\Activation.cpp`

```
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/cuda/Activation.h>

#include <ATen/core/DimVector.h>
#include <ATen/core/Tensor.h>
#include <ATen/TensorIterator.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/native/Resize.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/gelu_backward_native.h>
#include <ATen/ops/gelu_native.h>
#include <ATen/ops/glu_backward_native.h>
#include <ATen/ops/log_sigmoid_forward_native.h>
#endif

namespace at::native {

// -----------------------------------
// glu backward
// -----------------------------------

// 在 CUDA 环境下计算 glu 反向传播，填充给定的梯度张量 grad_input
Tensor& glu_backward_cuda_out(const Tensor& grad_output, const Tensor& input,
                              int64_t dim, Tensor& grad_input) {
  TORCH_CHECK(input.dim() > 0, "glu does not support 0-dimensional tensors");
  auto wrap_dim = maybe_wrap_dim(dim, input.dim()); // 根据维度调整 dim 的范围
  auto input_sizes = input.sizes(); // 获取输入张量的尺寸
  const int64_t nIn = input_sizes[wrap_dim]; // 获取指定维度的尺寸
  TORCH_CHECK(nIn % 2 == 0, "Halving dimension must be even, but dimension ",
              wrap_dim, " is size ", nIn); // 检查是否能够将指定维度划分为偶数个元素

  resize_output(grad_input, input_sizes); // 调整 grad_input 的尺寸与 input_sizes 一致

  DimVector iter_shape(input_sizes); // 创建迭代形状，与输入张量尺寸相同
  const auto dim_size = nIn / 2; // 计算指定维度的一半尺寸
  iter_shape[wrap_dim] = dim_size; // 更新迭代形状中指定维度的尺寸
  TORCH_CHECK(grad_output.sizes() == IntArrayRef{iter_shape}); // 检查 grad_output 的尺寸是否与 iter_shape 相同

  const auto iter = at::TensorIteratorConfig()
    .add_output(grad_input)
    .add_const_input(input)
    .add_const_input(grad_output)
    .resize_outputs(false)
    .declare_static_shape(iter_shape)
    .build(); // 创建张量迭代器配置

  if (iter.numel() == 0) {
    return grad_input; // 如果迭代器元素数为零，直接返回 grad_input
  }

  const auto I_stride = input.strides()[wrap_dim] * dim_size; // 计算输入张量在指定维度上的步长
  const auto gI_stride = grad_input.strides()[wrap_dim] * dim_size; // 计算梯度张量在指定维度上的步长

  if (iter.can_use_32bit_indexing()) {
    launch_glu_backward_kernel(iter, gI_stride, I_stride); // 如果可以使用 32 位索引，则启动 glu 反向传播的核函数
  } else {
    for (const auto& sub_iter: iter.with_32bit_indexing()) {
      launch_glu_backward_kernel(sub_iter, gI_stride, I_stride); // 否则，使用 32 位索引启动 glu 反向传播的子迭代器核函数
    }
  }
  return grad_input; // 返回填充后的梯度张量 grad_input
}

// 在 CUDA 环境下计算 glu 反向传播，并返回结果张量
Tensor glu_backward_cuda(const Tensor& grad_output, const Tensor& input, int64_t dim) {
  auto grad_input = at::empty({0}, input.options()); // 创建空张量 grad_input，使用与 input 相同的选项
  return glu_backward_cuda_out(grad_output, input, dim, grad_input); // 调用 glu_backward_cuda_out 函数计算 glu 反向传播并返回结果
}

// -----------------------------------
// log_sigmoid forward
// -----------------------------------

// 在 CUDA 环境下计算 log_sigmoid 前向传播，并返回结果张量及缓冲区
std::tuple<Tensor&, Tensor&> log_sigmoid_forward_out_cuda(const Tensor& input, Tensor& result, Tensor& buffer) {
  // 注意：buffer 仅在 CPU 调度时使用，在此处我们忽略它
  auto iter = TensorIteratorConfig()
    .add_output(result)
    .add_const_input(input)
    .build(); // 创建张量迭代器配置

  launch_log_sigmoid_forward_kernel(iter); // 启动 log_sigmoid 前向传播的核函数
  return std::forward_as_tuple(result, buffer); // 返回结果张量及缓冲区的元组
}

} // namespace at::native
// 在 CUDA 上执行 log sigmoid 操作的前向传播，并返回结果和缓冲区
std::tuple<Tensor, Tensor> log_sigmoid_forward_cuda(const Tensor& input) {
  // 创建一个与输入张量相同大小的空张量来存储结果
  auto result = at::empty_like(input);
  // 创建一个空的缓冲区张量
  auto buffer = at::empty({0}, input.options());
  // 调用 log_sigmoid_forward_out_cuda 函数进行前向传播计算
  log_sigmoid_forward_out_cuda(input, result, buffer);
  // 返回结果张量和缓冲区张量的元组
  return std::forward_as_tuple(result, buffer);
}

// 实现在 CUDA 上的 gelu 操作，接受一个近似值参数和一个结果张量
TORCH_IMPL_FUNC(gelu_out_cuda) (
  const Tensor& /*self*/, c10::string_view approximate, const Tensor& /*result*/
) {
  // 调用 GeluCUDAKernelImpl 函数执行 gelu 操作的 CUDA 内核实现
  GeluCUDAKernelImpl(*this, get_gelutype_enum(approximate));
}

// 实现在 CUDA 上的 gelu 反向传播操作，接受梯度、自身张量、近似值参数和梯度输入张量
TORCH_IMPL_FUNC(gelu_backward_out_cuda) (
  const Tensor& /*grad*/, const Tensor& /*self*/, c10::string_view approximate, const Tensor& /*grad_input*/
) {
  // 调用 GeluBackwardCUDAKernelImpl 函数执行 gelu 反向传播的 CUDA 内核实现
  GeluBackwardCUDAKernelImpl(*this, get_gelutype_enum(approximate));
}

// 结束 at::native 命名空间的声明
}  // namespace at::native
```