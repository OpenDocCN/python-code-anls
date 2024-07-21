# `.\pytorch\aten\src\ATen\native\AffineGridGenerator.cpp`

```py
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/TensorOperators.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/affine_grid_generator_backward_native.h>
#include <ATen/ops/affine_grid_generator_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/linspace.h>
#include <ATen/ops/tensor.h>
#endif

namespace at::native {

// 定义一个静态函数，生成一个从-1到1的均匀分布的数列张量，用于创建基础网格
static at::Tensor linspace_from_neg_one(const Tensor& grid, int64_t num_steps,
                                 bool align_corners) {
  // 如果步数小于等于1，返回一个零张量，使用与输入张量相同的选项
  if (num_steps <= 1) {
    return at::tensor(0, grid.options());
  }
  // 创建一个从-1到1的均匀分布的数列张量
  auto range = at::linspace(-1, 1, num_steps, grid.options());
  // 如果不对齐角落点，调整数列的值
  if (!align_corners) {
    range = range * (num_steps - 1) / num_steps;
  }
  return range;
}

// 创建一个四维基础网格张量
static Tensor make_base_grid_4D(
    const Tensor& theta,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    bool align_corners) {
  // 创建一个空的四维张量，形状为[N, H, W, 3]，使用与输入 theta 相同的选项
  auto base_grid = at::empty({N, H, W, 3}, theta.options());

  // 复制从-1到1的均匀分布的数列到基础网格的第一维度
  base_grid.select(-1, 0).copy_(linspace_from_neg_one(theta, W, align_corners));
  // 复制从-1到1的均匀分布的数列到基础网格的第二维度，并添加一个额外的维度
  base_grid.select(-1, 1).copy_(linspace_from_neg_one(theta, H, align_corners).unsqueeze_(-1));
  // 填充基础网格的第三维度为1
  base_grid.select(-1, 2).fill_(1);

  return base_grid;
}

// 创建一个五维基础网格张量
static Tensor make_base_grid_5D(
    const Tensor& theta,
    int64_t N,
    int64_t C,
    int64_t D,
    int64_t H,
    int64_t W,
    bool align_corners) {
  // 创建一个空的五维张量，形状为[N, D, H, W, 4]，使用与输入 theta 相同的选项
  auto base_grid = at::empty({N, D, H, W, 4}, theta.options());

  // 复制从-1到1的均匀分布的数列到基础网格的第一维度
  base_grid.select(-1, 0).copy_(linspace_from_neg_one(theta, W, align_corners));
  // 复制从-1到1的均匀分布的数列到基础网格的第二维度，并添加一个额外的维度
  base_grid.select(-1, 1).copy_(linspace_from_neg_one(theta, H, align_corners).unsqueeze_(-1));
  // 复制从-1到1的均匀分布的数列到基础网格的第三维度，并添加两个额外的维度
  base_grid.select(-1, 2).copy_(linspace_from_neg_one(theta, D, align_corners).unsqueeze_(-1).unsqueeze_(-1));
  // 填充基础网格的第四维度为1
  base_grid.select(-1, 3).fill_(1);

  return base_grid;
}

// 创建四维仿射变换生成器
static Tensor affine_grid_generator_4D(
    const Tensor& theta,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    bool align_corners) {
  // 使用 theta 创建基础网格张量
  Tensor base_grid = make_base_grid_4D(theta, N, C, H, W, align_corners);
  // 将基础网格张量视图变换为[N, H*W, 3]，然后与 theta 的转置矩阵进行批矩阵乘法
  auto grid = base_grid.view({N, H * W, 3}).bmm(theta.transpose(1, 2));
  // 将结果网格张量视图变换为[N, H, W, 2]
  return grid.view({N, H, W, 2});
}

// 创建五维仿射变换生成器
static Tensor affine_grid_generator_5D(
    const Tensor& theta,
    int64_t N,
    int64_t C,
    int64_t D,
    int64_t H,
    int64_t W,
    bool align_corners) {
  // 使用 theta 创建基础网格张量
  Tensor base_grid = make_base_grid_5D(theta, N, C, D, H, W, align_corners);
  // 将基础网格张量视图变换为[N, D*H*W, 4]，然后与 theta 的转置矩阵进行批矩阵乘法
  auto grid = base_grid.view({N, D * H * W, 4}).bmm(theta.transpose(1, 2));
  // 将结果网格张量视图变换为[N, D, H, W, 3]
  return grid.view({N, D, H, W, 3});
}

// 主要的仿射变换生成器函数，根据输入的大小和对齐角落点标志生成相应维度的仿射网格
Tensor affine_grid_generator(const Tensor& theta, IntArrayRef size, bool align_corners) {
  // 检查输入大小的维度是否为4或5
  TORCH_CHECK(
      size.size() == 4 || size.size() == 5,
      "AffineGridGenerator needs 4d (spatial) or 5d (volumetric) inputs.");
  // 如果是四维输入，调用四维仿射变换生成器
  if (size.size() == 4) {
    return affine_grid_generator_4D(
        theta, size[0], size[1], size[2], size[3], align_corners);
  } else {
    // 如果是五维输入，调用五维仿射变换生成器
    return affine_grid_generator_5D(
        theta, size[0], size[1], size[2], size[3], size[4], align_corners);
  }
}

} // namespace at::native
// 定义一个静态函数，用于计算 4 维情况下的仿射网格生成器的反向传播
static Tensor affine_grid_generator_4D_backward(
    const Tensor& grad_grid,  // 梯度网格张量
    int64_t N,                // 批次大小
    int64_t C,                // 通道数
    int64_t H,                // 高度
    int64_t W,                // 宽度
    bool align_corners) {     // 是否对齐角点

  // 使用 grad_grid 创建基础网格
  auto base_grid = make_base_grid_4D(grad_grid, N, C, H, W, align_corners);

  // 断言 grad_grid 的尺寸符合预期
  AT_ASSERT(grad_grid.sizes() == IntArrayRef({N, H, W, 2}));

  // 计算 grad_theta，将 base_grid 重塑后与 grad_grid 进行批次矩阵乘法
  auto grad_theta = base_grid.view({N, H * W, 3})
                        .transpose(1, 2)
                        .bmm(grad_grid.reshape({N, H * W, 2}));

  // 返回 grad_theta 的转置结果
  return grad_theta.transpose(1, 2);
}

// 定义一个静态函数，用于计算 5 维情况下的仿射网格生成器的反向传播
static Tensor affine_grid_generator_5D_backward(
    const Tensor& grad_grid,  // 梯度网格张量
    int64_t N,                // 批次大小
    int64_t C,                // 通道数
    int64_t D,                // 深度
    int64_t H,                // 高度
    int64_t W,                // 宽度
    bool align_corners) {     // 是否对齐角点

  // 使用 grad_grid 创建基础网格
  auto base_grid = make_base_grid_5D(grad_grid, N, C, D, H, W, align_corners);

  // 断言 grad_grid 的尺寸符合预期
  AT_ASSERT(grad_grid.sizes() == IntArrayRef({N, D, H, W, 3}));

  // 计算 grad_theta，将 base_grid 重塑后与 grad_grid 进行批次矩阵乘法
  auto grad_theta = base_grid.view({N, D * H * W, 4})
                        .transpose(1, 2)
                        .bmm(grad_grid.reshape({N, D * H * W, 3}));

  // 返回 grad_theta 的转置结果
  return grad_theta.transpose(1, 2);
}

// 定义一个函数，根据输入的大小和对齐角点信息，选择合适的反向传播方法
Tensor affine_grid_generator_backward(const Tensor& grad, IntArrayRef size, bool align_corners) {
  // 检查输入尺寸是否为 4 或 5 维
  TORCH_CHECK(
      size.size() == 4 || size.size() == 5,
      "AffineGridGenerator needs 4d (spatial) or 5d (volumetric) inputs.");

  // 根据输入尺寸的维度调用对应的反向传播函数
  if (size.size() == 4) {
    return affine_grid_generator_4D_backward(
        grad, size[0], size[1], size[2], size[3], align_corners);
  } else {
    return affine_grid_generator_5D_backward(
        grad, size[0], size[1], size[2], size[3], size[4], align_corners);
  }
}

}  // namespace at::native
```