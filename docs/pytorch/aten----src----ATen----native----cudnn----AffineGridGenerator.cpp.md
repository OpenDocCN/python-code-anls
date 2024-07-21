# `.\pytorch\aten\src\ATen\native\cudnn\AffineGridGenerator.cpp`

```py
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Config.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAConfig.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/cudnn_affine_grid_generator_backward_native.h>
#include <ATen/ops/cudnn_affine_grid_generator_native.h>
#include <ATen/ops/empty.h>
#endif

#if !AT_CUDNN_ENABLED()

namespace at {
namespace native {

// 如果未启用 cuDNN 支持，则定义在这个命名空间中的函数将抛出错误信息
Tensor cudnn_affine_grid_generator_forward(
    const Tensor& theta,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W) {
  AT_ERROR(
      "cudnn_affine_grid_generator_forward: ATen not compiled with cuDNN support");
}

Tensor cudnn_affine_grid_generator_backward(
    const Tensor& grad_theta,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W) {
  AT_ERROR(
      "cudnn_affine_grid_generator_backward: ATen not compiled with cuDNN support");
}

} // namespace native
} // namespace at

#else // AT_CUDNN_ENABLED()

#include <ATen/cuda/Exceptions.h>
#include <ATen/cudnn/Descriptors.h>
#include <ATen/cudnn/Handle.h>
#include <ATen/cudnn/Types.h>
#include <ATen/cudnn/Utils.h>
#include <ATen/cudnn/cudnn-wrapper.h>

#include <ATen/TensorUtils.h>

namespace at {
namespace native {

namespace {

// 定义一个用于设置空间变换器描述符的私有函数
void setSamplerDescriptor(
    SpatialTransformerDescriptor& desc,
    cudnnDataType_t dataType,
    int N,
    int C,
    int H,
    int W) {
  int inputSize[4] = {N, C, H, W};
  // 设置描述符，指定数据类型和输入尺寸
  desc.set(dataType, 4, inputSize);
}

} // namespace

// 实现 cuDNN affine grid generator 的前向计算函数
Tensor cudnn_affine_grid_generator_forward(
    const Tensor& theta_t,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W) {
  // 确保 theta 是连续存储的张量
  auto theta_t_contig = theta_t.contiguous();
  TensorArg theta{theta_t_contig, "theta", 1};
  CheckedFrom c = "cudnn_affine_grid_generator_forward";
  // 检查 theta 张量是否连续存储
  checkContiguous(c, theta);
  // 检查 theta 张量的尺寸是否为 [N, 2, 3]
  checkSize(c, theta, {N, 2, 3});

  // 创建一个空的 grid_t 张量，用于存储生成的网格
  auto grid_t = at::empty({0}, theta->options());
  grid_t.resize_({N, H, W, 2});

  // 获取 cuDNN 数据类型
  auto dataType = getCudnnDataType(*theta);
  // 创建空间变换器描述符对象，并设置描述符
  SpatialTransformerDescriptor desc;
  setSamplerDescriptor(desc, dataType, N, C, H, W);
  // 调用 cuDNN 的空间变换器前向计算函数
  AT_CUDNN_CHECK(cudnnSpatialTfGridGeneratorForward(
      getCudnnHandle(), desc.desc(), theta->data_ptr(), grid_t.data_ptr()));
  // 返回生成的 grid_t 张量
  return grid_t;
}

// 实现 cuDNN affine grid generator 的反向计算函数
Tensor cudnn_affine_grid_generator_backward(
    const Tensor& grad_grid_t,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W) {
    // 将输入的梯度张量转为连续存储
    auto grad_grid_contig = grad_grid_t.contiguous();
    // 定义梯度张量的参数
    TensorArg grad_grid{grad_grid_contig, "grad_grid", 1};
    CheckedFrom c = "cudnn_affine_grid_generator_backward";
    // 检查梯度张量是否是连续存储
    checkContiguous(c, grad_grid);
    // 检查梯度张量的大小是否符合预期
    checkSize(c, grad_grid, {N, H, W, 2});

    // 创建一个空的梯度变换张量，并设置其大小
    auto grad_theta_t = at::empty({0}, grad_grid->options());
    grad_theta_t.resize_({N, 2, 3});

    // 获取梯度变换张量的数据类型
    auto dataType = getCudnnDataType(grad_theta_t);
    SpatialTransformerDescriptor desc;
    // 设置空间变换器描述符的采样器描述符
    setSamplerDescriptor(desc, dataType, N, C, H, W);

    // 调用 cuDNN 函数计算空间变换器反向操作的梯度
    AT_CUDNN_CHECK(cudnnSpatialTfGridGeneratorBackward(
        getCudnnHandle(),
        desc.desc(),
        grad_grid->data_ptr(),
        grad_theta_t.data_ptr()));

    // 返回计算得到的梯度变换张量
    return grad_theta_t;
}

} // namespace native
} // namespace at

#endif // AT_CUDNN_ENABLED()


注释：


} // 结束 at 命名空间

} // 结束 native 命名空间

} // 结束 at 命名空间的条件编译部分，检查是否启用了 CUDNN
```