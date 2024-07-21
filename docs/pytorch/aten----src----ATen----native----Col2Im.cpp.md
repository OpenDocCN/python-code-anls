# `.\pytorch\aten\src\ATen\native\Col2Im.cpp`

```
// 定义 TORCH_ASSERT_ONLY_METHOD_OPERATORS 宏，用于某些特定的方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 引入 ATen 库中的 Tensor 类和相关的头文件
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorUtils.h>

// 引入 ATen 库中的 im2col 相关头文件和形状检查头文件
#include <ATen/native/im2col.h>
#include <ATen/native/im2col_shape_check.h>
// 引入 c10 库中的 irange.h 头文件，用于范围遍历

// 如果未定义 AT_PER_OPERATOR_HEADERS 宏，则引入 ATen 库中的所有函数
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
// 如果定义了 AT_PER_OPERATOR_HEADERS 宏，则引入特定的函数头文件
#else
#include <ATen/ops/col2im_native.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/im2col_native.h>
#endif

// 注释：im2col/col2im 输出填充
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 我们的 im2col 和 col2im 实现同时需要输入的高度/宽度和看似多余的输出的高度/宽度。
// 原则上，可以通过卷积形状公式计算输出的高度/宽度。那么，这是怎么回事呢？
//
// 当反向运行带有 output_padding >= stride 的转置卷积时会出现问题。（顺便说一下，output_padding 在 THNN 中称为 adj。）让我们考虑一个简单的情况，
// 输入大小为 4x4，kernel=2，dilation=2，stride=1，output_padding=1：
//
// 输入：  X
//
// 输出：  X.X.
//        ....
//        X.X.
//        ....
//
// 如果我们对输出进行标准卷积的反向计算，参数相同，那么我们得到一个 2x2 的 grad_input（因为可以将模板向右移动一次，向下移动一次）。但是，如果计算 1x1 输入的反向，那些计算就都超出边界了。
//
// “现在 Edward，”你可能会说，“真正的问题是你设置了 output_padding >= stride，肯定应该在这种情况下引发错误。” 要理解为什么处理这种情况有用，我们必须理解如何计算卷积的权重梯度。假设我们对一个 5x5 输入进行 kernel=2，stride=2 的卷积。让我们看看权重 weight[0][0]（我们标记为 w）在输出中的所有贡献：
//
// 输入：  a.b..  Weight: w.
//        .....          ..
//        c.d..
//        .....
//        .....
//
// 输出：  [ aw+...  bw+... ]
//        [ cw+...  dw+... ]
//
// 从这个图表中，很容易看出，我们可以通过在输入和输出梯度之间进行 kernel=2，dilation=2，stride=1 的 *扩展* 卷积来计算权重梯度。但是有一个问题：如果直接进行扩展卷积，我们将得到一个 3x3 的权重梯度，但我们明显希望得到 2x2 的梯度。那么，如何避免越界呢？我们可以为非转置卷积添加 'output_padding' 概念，但另一个简单有效的解决方法是直接接受所需的输出大小，并仅在这些界限内进行计算。

// 此外，还要考虑 vol2col

// 命名空间 at::native 内部定义
namespace at::native {
// 匿名命名空间开始

// 静态函数 col2im_out_cpu_template，用于处理 CPU 上的 col2im 操作
static void col2im_out_cpu_template(
    // 输出 Tensor 对象，用于存储结果
    Tensor& output,
    // 输入 Tensor 对象，即输入数据
    const Tensor& input_,
    // 输出的大小，作为一个整数数组的引用
    IntArrayRef output_size,
    // 卷积核大小，作为一个整数数组的引用
    IntArrayRef kernel_size,
    // 扩展率，作为一个整数数组的引用
    IntArrayRef dilation,
    // 填充大小，作为一个整数数组的引用
    IntArrayRef padding,
  // 检查输出大小是否为2，否则抛出错误信息
  TORCH_CHECK(
      output_size.size() == 2,
      "It is expected output_size equals to 2, but got size ",
      output_size.size());

  // 检查卷积核大小是否为2，否则抛出错误信息
  TORCH_CHECK(
      kernel_size.size() == 2,
      "It is expected kernel_size equals to 2, but got size ",
      kernel_size.size());

  // 检查扩张大小是否为2，否则抛出错误信息
  TORCH_CHECK(
      dilation.size() == 2,
      "It is expected dilation equals to 2, but got size ",
      dilation.size());

  // 检查填充大小是否为2，否则抛出错误信息
  TORCH_CHECK(
      padding.size() == 2,
      "It is expected padding equals to 2, but got size ",
      padding.size());

  // 检查步长大小是否为2，否则抛出错误信息
  TORCH_CHECK(
      stride.size() == 2,
      "It is expected stride equals to 2, but got size ",
      stride.size());

  // 提取输出高度和宽度
  int64_t output_height = output_size[0];
  int64_t output_width = output_size[1];
  // 提取卷积核高度和宽度
  int64_t kernel_height = kernel_size[0];
  int64_t kernel_width = kernel_size[1];
  // 提取扩张高度和宽度
  int64_t dilation_height = dilation[0];
  int64_t dilation_width = dilation[1];
  // 提取填充高度和宽度
  int64_t pad_height = padding[0];
  int64_t pad_width = padding[1];
  // 提取步长高度和宽度
  int64_t stride_height = stride[0];
  int64_t stride_width = stride[1];

  // 调用 col2im_shape_check 函数，检查输入张量形状是否符合要求
  col2im_shape_check(
      input_,
      Tensor(),
      output_height,
      output_width,
      kernel_height,
      kernel_width,
      dilation_height,
      dilation_width,
      pad_height,
      pad_width,
      stride_height,
      stride_width);

  // 将输入张量转为连续内存存储的张量
  Tensor input = input_.contiguous();

  // 检查是否是批处理输入，如果输入维度为2，则强制视为非批处理输入
  bool batched_input = true;
  if (input.dim() == 2) {
    // 强制批处理
    batched_input = false;
    input = input.view({1, input.size(0), input.size(1)});
  }



    // 将输入张量重塑为形状为 {1, input.size(0), input.size(1)} 的新张量
    input = input.view({1, input.size(0), input.size(1)});
  }



  int64_t batch_size = input.size(0);
  int64_t n_input_plane = input.size(1);
  int64_t n_output_plane = n_input_plane / (kernel_width * kernel_height);

  output.resize_({batch_size, n_output_plane, output_height, output_width});



  // 计算批量大小、输入平面数和输出平面数
  int64_t batch_size = input.size(0);
  int64_t n_input_plane = input.size(1);
  int64_t n_output_plane = n_input_plane / (kernel_width * kernel_height);

  // 重新调整输出张量的大小为 {batch_size, n_output_plane, output_height, output_width}
  output.resize_({batch_size, n_output_plane, output_height, output_width});



  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kBFloat16, kHalf,
      input.scalar_type(), "col2im_out_cpu", [&] {
        Tensor input_n = Tensor();
        Tensor output_n = Tensor();

        int64_t height_col = (output_height + 2 * pad_height -
                              (dilation_height * (kernel_height - 1) + 1)) /
                stride_height +
            1;
        int64_t width_col = (output_width + 2 * pad_width -
                             (dilation_width * (kernel_width - 1) + 1)) /
                stride_width +
            1;

        for (const auto elt : c10::irange(batch_size)) {
          input_n = input.select(0, elt);
          output_n = output.select(0, elt);

          col2im<scalar_t>(
              input_n.const_data_ptr<scalar_t>(),
              n_output_plane,
              output_height,
              output_width,
              height_col,
              width_col,
              kernel_height,
              kernel_width,
              pad_height,
              pad_width,
              stride_height,
              stride_width,
              dilation_height,
              dilation_width,
              output_n.mutable_data_ptr<scalar_t>());
        }

        if (!batched_input) {
          output.resize_({n_output_plane, output_height, output_width});
        }
      });



  // 使用 AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2 宏根据输入的标量类型执行函数 "col2im_out_cpu"
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kBFloat16, kHalf,
      input.scalar_type(), "col2im_out_cpu", [&] {
        // 定义局部变量 input_n 和 output_n 作为 Tensor
        Tensor input_n = Tensor();
        Tensor output_n = Tensor();

        // 计算输出列的高度和宽度
        int64_t height_col = (output_height + 2 * pad_height -
                              (dilation_height * (kernel_height - 1) + 1)) /
                stride_height +
            1;
        int64_t width_col = (output_width + 2 * pad_width -
                             (dilation_width * (kernel_width - 1) + 1)) /
                stride_width +
            1;

        // 遍历每个批次中的元素
        for (const auto elt : c10::irange(batch_size)) {
          // 选择当前批次的输入和输出张量
          input_n = input.select(0, elt);
          output_n = output.select(0, elt);

          // 调用 col2im 函数，将列数据转换回图像数据
          col2im<scalar_t>(
              input_n.const_data_ptr<scalar_t>(),
              n_output_plane,
              output_height,
              output_width,
              height_col,
              width_col,
              kernel_height,
              kernel_width,
              pad_height,
              pad_width,
              stride_height,
              stride_width,
              dilation_height,
              dilation_width,
              output_n.mutable_data_ptr<scalar_t>());
        }

        // 如果不是批处理输入，则重新调整输出张量的大小为 {n_output_plane, output_height, output_width}
        if (!batched_input) {
          output.resize_({n_output_plane, output_height, output_width});
        }
      });
} // namespace



} // namespace

这行注释指出了上述代码段的命名空间的结束。


Tensor& col2im_out_cpu(const Tensor& input,
    IntArrayRef output_size,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride,
    Tensor& output) {

定义了一个名为 `col2im_out_cpu` 的函数，接受多个参数，包括输入张量 `input`、输出尺寸 `output_size`、核大小 `kernel_size`、膨胀大小 `dilation`、填充大小 `padding`、步幅 `stride`，以及输出张量 `output` 的引用。函数返回类型为 `Tensor&`。


  col2im_out_cpu_template(
      output, input, output_size, kernel_size, dilation, padding, stride);

调用了名为 `col2im_out_cpu_template` 的模板函数，用于执行具体的计算操作，将结果存储在 `output` 引用的张量中。


  return output;

返回经过处理后的输出张量 `output` 的引用。


}

Tensor col2im_cpu(
    const Tensor& input,
    IntArrayRef output_size,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride) {

定义了一个名为 `col2im_cpu` 的函数，接受多个参数，包括输入张量 `input`、输出尺寸 `output_size`、核大小 `kernel_size`、膨胀大小 `dilation`、填充大小 `padding`、步幅 `stride`。函数返回类型为 `Tensor`。


  Tensor output = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

创建了一个新的张量 `output`，其形状和 `input` 相同，并且采用传统的连续内存布局。


  col2im_out_cpu_template(
      output, input, output_size, kernel_size, dilation, padding, stride);

再次调用了 `col2im_out_cpu_template` 模板函数，用于执行具体的计算操作，将结果存储在 `output` 张量中。


  return output;

返回经过处理后的输出张量 `output`。


} // namespace at::native

这行注释指出了代码段的命名空间 `at::native` 的结束。
```