# `.\pytorch\aten\src\ATen\native\im2col_shape_check.h`

```py
#pragma once
#include <ATen/core/Tensor.h>
#include <ATen/TensorUtils.h>
#include <ATen/div_rtn.h>

namespace at::native {

// 形状检查函数，用于 col2im 操作
inline void col2im_shape_check(
    const Tensor& input,                     // 输入张量
    const Tensor& grad_output,               // 梯度输出张量
    int64_t output_height,                   // 输出高度
    int64_t output_width,                    // 输出宽度
    int64_t kernel_height,                   // 卷积核高度
    int64_t kernel_width,                    // 卷积核宽度
    int64_t dilation_height,                 // 扩张高度
    int64_t dilation_width,                  // 扩张宽度
    int64_t pad_height,                      // 填充高度
    int64_t pad_width,                       // 填充宽度
    int64_t stride_height,                   // 步幅高度
    int64_t stride_width) {                  // 步幅宽度

  // 检查卷积核大小是否大于零
  TORCH_CHECK(
      kernel_width > 0 && kernel_height > 0,
      "kernel size should be greater than zero, but got kernel_height: ",
      kernel_height,
      " kernel_width: ",
      kernel_width);

  // 检查步幅是否大于零
  TORCH_CHECK(
      stride_width > 0 && stride_height > 0,
      "stride should be greater than zero, but got stride_height: ",
      stride_height,
      " stride_width: ",
      stride_width);

  // 检查扩张是否大于零
  TORCH_CHECK(
      dilation_width > 0 && dilation_height > 0,
      "dilation should be greater than zero, but got dilation_height: ",
      dilation_height,
      " dilation_width: ",
      dilation_width);

  // 检查填充是否非负
  TORCH_CHECK(
      pad_width >= 0 && pad_height >= 0,
      "padding should be non-negative, but got pad_height: ",
      pad_height,
      " pad_width: ",
      pad_width);

  int64_t ndim = input.ndimension();
  // 检查输入张量维度是否符合要求
  TORCH_CHECK(
      (ndim == 2 && input.size(0) != 0 && input.size(1) != 0) ||
      (ndim == 3 && input.size(1) != 0 && input.size(2) != 0),
      "Expected 2D or 3D (batch mode) tensor for input with possibly 0 batch size and non-zero dimensions for input, but got: ",
      input.sizes());

  int64_t batch_dim = (ndim == 3) ? 0 : -1;
  int64_t n_input_plane = input.size(batch_dim + 1);

  // 检查输入张量的第二个维度是否能被卷积核大小整除
  if (n_input_plane % (kernel_width * kernel_height) != 0) {
    AT_ERROR(
        "Expected size of input's dimension 1 to be divisible by the "
        "product of kernel_size, but got input.size(1)=",
        n_input_plane,
        " and kernel_size=(",
        kernel_height,
        ", ",
        kernel_width,
        ").");
  }

  int64_t input_length = input.size(batch_dim + 2);

  // 计算高度方向上的块数
  int64_t n_blocks_height =
      div_rtn<int64_t>(
          output_height + 2 * pad_height -
              dilation_height * (kernel_height - 1) - 1,
          stride_height) +
      1;

  // 计算宽度方向上的块数
  int64_t n_blocks_width = div_rtn<int64_t>(
                               output_width + 2 * pad_width -
                                   dilation_width * (kernel_width - 1) - 1,
                               stride_width) +
      1;

  // 检查输入长度是否与块的数量匹配
  if (input_length != (n_blocks_height * n_blocks_width)) {
    # 抛出错误，指示期望的输入维度2的大小与计算得到的滑动块数量不匹配
    AT_ERROR(
        "Given output_size=(",
        output_height,
        ", ",
        output_width,
        "), kernel_size=(",
        kernel_height,
        ", ",
        kernel_width,
        "), dilation=(",
        dilation_height,
        ", ",
        dilation_width,
        "), padding=(",
        pad_height,
        ", ",
        pad_width,
        "), stride=(",
        stride_height,
        ", ",
        stride_width,
        "), expected size of input's dimension 2 to match the calculated number of ",
        "sliding blocks ",
        n_blocks_height,
        " * ",
        n_blocks_width,
        " = ",
        (n_blocks_height * n_blocks_width),
        ", but got input.size(2)=",
        input_length,
        ".");
  }

  # 检查滑动块的数量是否大于等于1，否则抛出错误
  TORCH_CHECK(
    n_blocks_height >= 1 && n_blocks_width >= 1,
    "Given output_size=(", output_height, ", ", output_width, "), ",
    "kernel_size=(", kernel_height, ", ", kernel_width, "), ",
    "dilation=(", dilation_height, ", ", dilation_width, "), ",
    "padding=(", pad_height, ", ", pad_width, "), ",
    "stride=(", stride_height, ", ", stride_width, "), ",
    "calculated shape of the array of sliding blocks as ",
    "(", n_blocks_height, ", ", n_blocks_width, "), ",
    "which is too small (non-positive)");

  # 如果输出的宽度或高度小于1，则抛出错误
  if (output_width < 1 || output_height < 1) {
    AT_ERROR(
        "Expected output spatial size to be positive, but got: output_size=(",
        output_height,
        ", ",
        output_width,
        ").");
  }
// 结束 at::native 命名空间

inline void im2col_shape_check(
    const Tensor& input,  // 输入张量
    const Tensor& grad_output,  // 梯度输出张量（未使用）
    int64_t kernel_height,  // 卷积核高度
    int64_t kernel_width,  // 卷积核宽度
    int64_t dilation_height,  // 高度方向的膨胀率
    int64_t dilation_width,  // 宽度方向的膨胀率
    int64_t pad_height,  // 高度方向的填充大小
    int64_t pad_width,  // 宽度方向的填充大小
    int64_t stride_height,  // 高度方向的步长
    int64_t stride_width) {  // 宽度方向的步长

  // 检查卷积核大小是否合法
  TORCH_CHECK(
      kernel_width > 0 && kernel_height > 0,
      "kernel size should be greater than zero, but got kernel_height: ",
      kernel_height,
      " kernel_width: ",
      kernel_width);

  // 检查膨胀率是否合法
  TORCH_CHECK(
      dilation_width > 0 && dilation_height > 0,
      "dilation should be greater than zero, but got dilation_height: ",
      dilation_height,
      " dilation_width: ",
      dilation_width);

  // 检查填充大小是否合法
  TORCH_CHECK(
      pad_width >= 0 && pad_height >= 0,
      "padding should be non-negative, but got pad_height: ",
      pad_height,
      " pad_width: ",
      pad_width);

  // 检查步长是否合法
  TORCH_CHECK(
      stride_width > 0 && stride_height > 0,
      "stride should be greater than zero, but got stride_height: ",
      stride_height,
      " stride_width: ",
      stride_width);

  int64_t ndim = input.ndimension();  // 输入张量的维度数

  // 只允许 dim=0 是批量维度
  bool valid_dims = input.size(1) != 0 && input.size(2) != 0;
  TORCH_CHECK(
      (ndim == 3 && input.size(0) && valid_dims) ||  // 检查 3D 张量的维度
      (ndim == 4 && valid_dims && input.size(3) != 0),  // 检查 4D 张量的维度
      "Expected 3D or 4D (batch mode) tensor with possibly 0 batch size and other non-zero dimensions for input, but got: ",
      input.sizes());

  int64_t dim_batch = 0;

  if (ndim == 3) {
    dim_batch = -1;  // 如果是 3D 张量，批量维度设置为 -1
  }

  int64_t input_height = input.size(dim_batch + 2);  // 输入张量的高度
  int64_t input_width = input.size(dim_batch + 3);  // 输入张量的宽度

  // 计算输出的高度和宽度
  int64_t output_height = div_rtn<int64_t>(
                              input_height + 2 * pad_height -
                                  (dilation_height * (kernel_height - 1) + 1),
                              stride_height) +
      1;
  int64_t output_width = div_rtn<int64_t>(
                             input_width + 2 * pad_width -
                                 (dilation_width * (kernel_width - 1) + 1),
                             stride_width) +
      1;

  // 如果输出的高度或宽度小于 1，则抛出错误
  if (output_height < 1 || output_width < 1) {
    AT_ERROR(
        "Given input with spatial size (",
        input_height,
        ", ",
        input_height,
        "), kernel_size=(",
        kernel_height,
        ", ",
        kernel_width,
        "), dilation=(",
        dilation_height,
        ", ",
        dilation_width,
        "), padding=(",
        pad_height,
        ", ",
        pad_width,
        "), calculated shape of the array of sliding blocks as (",
        output_height,
        ", ",
        output_width,
        "), but its components must be at least one.");
  }
}
```