# `.\pytorch\aten\src\ATen\native\ReplicationPadding.cpp`

```py
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/TensorMeta.h>
#include <ATen/native/Padding.h>
#include <c10/util/irange.h>
#include <algorithm>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/replication_pad1d_backward_native.h>
#include <ATen/ops/replication_pad1d_native.h>
#include <ATen/ops/replication_pad2d_backward_native.h>
#include <ATen/ops/replication_pad2d_native.h>
#include <ATen/ops/replication_pad3d_backward_native.h>
#include <ATen/ops/replication_pad3d_native.h>
#include <ATen/ops/zeros_like.h>
#endif

namespace at::meta {

// 定义 replication_pad1d 元函数，处理 1D 数据的复制填充
TORCH_META_FUNC(replication_pad1d) (
  const Tensor& input, IntArrayRef paddingSize  // 没有输出参数！
) {
  // 检查填充大小是否为 2
  TORCH_CHECK(paddingSize.size() == 2, "padding size is expected to be 2");

  // 初始化维度变量
  int64_t dimw = 1;
  int64_t dimslices = 0;
  int64_t nbatch = 1;

  // 获取左右填充大小
  int64_t pad_l = paddingSize[0];
  int64_t pad_r = paddingSize[1];

  // 检查输入合法性
  at::native::padding::check_valid_input<1>(input, paddingSize);

  // 如果输入是三维的，调整批次和维度
  if (input.ndimension() == 3) {
    nbatch = input.size(0);
    dimw++;
    dimslices++;
  }

  /* sizes */
  // 计算输入的尺寸
  int64_t nslices = input.size(dimslices);
  int64_t iwidth = input.size(dimw);
  int64_t owidth = iwidth + pad_l + pad_r;

  // 检查输出宽度是否合理
  TORCH_CHECK(owidth >= 1,
      "input (W: ", iwidth, ") is too small."
      " Calculated output W: ", owidth);

  // 根据输入维度设置输出形状
  if (input.ndimension() == 2) {
    set_output_raw_strided(0, {nslices, owidth}, {}, input.options());
  } else {
    set_output_raw_strided(0, {nbatch, nslices, owidth}, {}, input.options());
  }
}

// 定义 replication_pad1d_backward 元函数，处理 1D 数据的复制填充反向传播
TORCH_META_FUNC(replication_pad1d_backward) (
  const Tensor& gradOutput,
  const Tensor& input,
  IntArrayRef paddingSize
) {
  // 初始化维度变量
  int64_t dimw = 1;

  // 检查填充大小是否为 2
  TORCH_CHECK(paddingSize.size() == 2, "padding size is expected to be 2");
  int64_t pad_l = paddingSize[0];
  int64_t pad_r = paddingSize[1];

  // 如果输入是三维的，调整维度
  if (input.ndimension() == 3) {
    dimw++;
  }

  /* sizes */
  // 计算输入的尺寸
  int64_t iwidth = input.size(dimw);
  int64_t owidth  = iwidth + pad_l + pad_r;

  // 检查梯度输出宽度是否合理
  TORCH_CHECK(owidth == gradOutput.size(dimw),
      "gradOutput width unexpected. Expected: ", owidth,
      " Got: ", gradOutput.size(dimw));

  // 根据输入形状设置输出形状
  set_output_raw_strided(0, input.sizes(), {}, input.options());
}

// 定义 replication_pad2d 元函数，处理 2D 数据的复制填充
TORCH_META_FUNC(replication_pad2d) (
  const Tensor& input, IntArrayRef paddingSize
) {
  // 检查填充大小是否为 4
  TORCH_CHECK(paddingSize.size() == 4, "padding size is expected to be 4");
  int64_t pad_l = paddingSize[0];
  int64_t pad_r = paddingSize[1];
  int64_t pad_t = paddingSize[2];
  int64_t pad_b = paddingSize[3];
  int64_t dimw = 2;
  int64_t dimh = 1;
  int64_t dimslices = 0;
  int64_t nbatch = 1;

  // 检查输入合法性
  at::native::padding::check_valid_input<2>(input, paddingSize);

  // 如果输入是四维的，调整批次和维度
  if (input.dim() == 4) {
    nbatch = input.size(0);
    dimw++;
    dimh++;

    dimslices++;
  }

  /* sizes */
  // 计算输入的尺寸
  int64_t nslices = input.size(dimslices);
  int64_t iwidth = input.size(dimw);
  int64_t iheight = input.size(dimh);
  int64_t owidth = iwidth + pad_l + pad_r;
  int64_t oheight = iheight + pad_t + pad_b;

  // 设置输出形状
  set_output_raw_strided(0, {nbatch, nslices, oheight, owidth}, {}, input.options());
}
    dimslices++;

# 增加 `dimslices` 变量的值，用于迭代切片的维度编号。


  /* sizes */

# 下面的变量是关于输入和输出尺寸的定义。


  int64_t nslices = input.size(dimslices);

# 计算切片的数量，根据 `dimslices` 指定的维度。


  int64_t iheight = input.size(dimh);

# 计算输入张量在高度维度上的大小。


  int64_t iwidth = input.size(dimw);

# 计算输入张量在宽度维度上的大小。


  int64_t oheight = iheight + pad_t + pad_b;
  int64_t owidth  = iwidth + pad_l + pad_r;

# 计算输出张量的高度和宽度，考虑了填充的情况。


  TORCH_CHECK(owidth >= 1 || oheight >= 1,
      "input (H: ", iheight, ", W: ", iwidth, " ) is too small."
      " Calculated output H: ", oheight, " W: ", owidth);

# 检查输出张量的尺寸是否合理，确保输出尺寸不小于1。


  if (input.dim() == 3) {
    set_output_raw_strided(0, {nslices, oheight, owidth}, {}, input.options());
  } else {
    set_output_raw_strided(0, {nbatch, nslices, oheight, owidth}, {}, input.options());
  }

# 根据输入张量的维度数量设置输出张量的尺寸，使用指定的选项。
} // 结束 TORCH_META_FUNC(replication_pad3d) 函数的定义

TORCH_META_FUNC(replication_pad3d) (
  const Tensor& input, IntArrayRef paddingSize
) {
  // 检查 paddingSize 的长度是否为 6
  TORCH_CHECK(paddingSize.size() == 6, "padding size is expected to be 6");
  // 分别获取各个方向的填充大小
  int64_t pleft = paddingSize[0];
  int64_t pright = paddingSize[1];
  int64_t ptop = paddingSize[2];
  int64_t pbottom = paddingSize[3];
  int64_t pfront = paddingSize[4];
  int64_t pback = paddingSize[5];
  // 设置默认的维度索引
  int64_t dimw = 3;
  int64_t dimh = 2;
  int64_t dimd = 1;
  int64_t dimslices = 0;
  int64_t nbatch = 1;

  // 检查输入张量的有效性
  at::native::padding::check_valid_input<3>(input, paddingSize);

  // 如果输入张量的维度为 5
  if (input.dim() == 5) {
    // 更新批次大小
    nbatch = input.size(0);
    // 更新各维度索引
    dimw++;
    dimh++;
    dimd++;
    dimslices++;
  }

  /* sizes */
  // 计算输入张量的各维度大小
  int64_t nslices = input.size(dimslices);
  int64_t idepth = input.size(dimd);
  int64_t iheight = input.size(dimh);
  int64_t iwidth = input.size(dimw);
  // 计算输出张量的各维度大小
  int64_t odepth = idepth + pfront + pback;
  int64_t oheight = iheight + ptop + pbottom;
  int64_t owidth  = iwidth + pleft + pright;

  // 检查输出张量的维度是否合理
  TORCH_CHECK(owidth >= 1 || oheight >= 1 || odepth >= 1,
      "input (D: ", idepth, " H: ", iheight, ", W: ", iwidth,
      ") is too small."
      " Calculated output D: ", odepth, " H: ", oheight, " W: ", owidth);

  /* resize output */
  // 根据输入张量的维度情况设置输出张量的大小和选项
  if (input.dim() == 4) {
    set_output_raw_strided(0, {nslices, odepth, oheight, owidth}, {}, input.options());
  } else {
    set_output_raw_strided(0, {nbatch, nslices, odepth, oheight, owidth}, {}, input.options());
  }
}

} // namespace at::meta

namespace at::native {

namespace {

// 定义 replication_pad2d_backward_out_cpu_template 函数，处理 2D 反向填充的计算
void replication_pad2d_backward_out_cpu_template(
    Tensor& gradInput,
    const Tensor& gradOutput,
    const Tensor& input,
    IntArrayRef paddingSize)
{
  // 检查 paddingSize 的长度是否为 4
  TORCH_CHECK(paddingSize.size() == 4, "padding size is expected to be 4");
  // 分别获取各个方向的填充大小
  int pad_l = paddingSize[0];
  int pad_r = paddingSize[1];
  int pad_t = paddingSize[2];
  int pad_b = paddingSize[3];
  // 设置默认的维度索引
  int dimw = 2;
  int dimh = 1;

  // 如果输入张量的维度为 4
  if (input.dim() == 4) {
    // 更新各维度索引
    dimw++;
    dimh++;
  }

  /* sizes */
  // 计算输入张量的高度和宽度
  int64_t iheight = input.size(dimh);
  int64_t iwidth = input.size(dimw);
  // 计算输出张量的高度和宽度
  int64_t oheight = iheight + pad_t + pad_b;
  int64_t owidth  = iwidth + pad_l + pad_r;

  // 检查梯度输出张量的维度是否合理
  TORCH_CHECK(owidth == gradOutput.size(dimw),
      "gradOutput width unexpected. Expected: ", owidth, ", Got: ",
      gradOutput.size(dimw));
  TORCH_CHECK(oheight == gradOutput.size(dimh),
      "gradOutput height unexpected. Expected: ", oheight, ", Got: ",
      gradOutput.size(dimh));

  // 如果梯度输入张量的元素数为 0，则直接返回
  if (gradInput.numel() == 0) {
    return;
  }

  // 调用具体的 2D 反向填充的 CPU 实现函数
  replication_pad2d_backward_kernel(kCPU, gradInput, gradOutput, paddingSize);
}

void replication_pad3d_backward_out_cpu_template(
    Tensor& gradInput,
    const Tensor& gradOutput,
    const Tensor& input,
    IntArrayRef paddingSize)
{
  // 检查 paddingSize 是否为长度为 6，否则抛出异常
  TORCH_CHECK(paddingSize.size() == 6, "padding size is expected to be 6");
  // 初始化各个方向的 padding 大小
  int pleft = paddingSize[0];
  int pright = paddingSize[1];
  int ptop = paddingSize[2];
  int pbottom = paddingSize[3];
  int pfront = paddingSize[4];
  int pback = paddingSize[5];
  // 初始化默认的数据维度索引
  int dimw = 3;
  int dimh = 2;
  int dimd = 1;

  // 如果输入张量的维度为 5，则增加相应的维度索引
  if (input.dim() == 5) {
    dimw++;
    dimh++;
    dimd++;
  }

  /* sizes */
  // 计算输出张量的深度、高度和宽度
  int64_t idepth = input.size(dimd);
  int64_t iheight = input.size(dimh);
  int64_t iwidth = input.size(dimw);
  int64_t odepth = idepth + pfront + pback;
  int64_t oheight = iheight + ptop + pbottom;
  int64_t owidth  = iwidth + pleft + pright;

  // 检查输入张量是否满足 padding 的有效性要求
  at::native::padding::check_valid_input<3>(input, paddingSize);

  // 检查 gradOutput 的宽度、高度和深度是否符合预期
  TORCH_CHECK(owidth == gradOutput.size(dimw),
      "gradOutput width unexpected. Expected: ", owidth, ", Got: ",
      gradOutput.size(dimw));
  TORCH_CHECK(oheight == gradOutput.size(dimh),
      "gradOutput height unexpected. Expected: ", oheight, ", Got: ",
      gradOutput.size(dimh));
  TORCH_CHECK(odepth == gradOutput.size(dimd),
      "gradOutput depth unexpected. Expected: ", odepth, ", Got: ",
      gradOutput.size(dimd));

  // 如果 gradInput 的元素数量为 0，则直接返回
  if (gradInput.numel() == 0) {
    return;
  }

  // 调用 replication_pad3d_backward_kernel 函数进行反向传播计算
  replication_pad3d_backward_kernel(kCPU, gradInput, gradOutput, paddingSize);
}

} // 匿名命名空间结束

TORCH_IMPL_FUNC(replication_pad1d_out_cpu) (
  const Tensor& input, IntArrayRef paddingSize, const Tensor& output
) {
  // 调用 replication_pad1d_kernel 函数进行前向传播计算
  replication_pad1d_kernel(kCPU, output, input, paddingSize);
}

TORCH_IMPL_FUNC(replication_pad1d_backward_out_cpu) (
  const Tensor& gradOutput, const Tensor& input, IntArrayRef paddingSize, const Tensor& gradInput
) {
  // 如果 gradInput 的元素数量为 0，则直接返回
  if (gradInput.numel() == 0) {
    return;
  }
  // 将 gradInput 张量的所有元素置为 0
  gradInput.zero_();

  // 调用 replication_pad1d_backward_kernel 函数进行反向传播计算
  replication_pad1d_backward_kernel(kCPU, gradInput, gradOutput, paddingSize);
}

TORCH_IMPL_FUNC(replication_pad2d_out_cpu) (
  const Tensor& input, IntArrayRef paddingSize, const Tensor& output
) {
  // TODO: 在 CUDA 支持通道最后存储时，将此移至 TORCH_META_FUNC
  // 调整输出张量的大小和内存格式，以便与输入张量兼容
  output.resize_(output.sizes(), input.suggest_memory_format());

  // 调用 replication_pad2d_kernel 函数进行前向传播计算
  replication_pad2d_kernel(kCPU, output, input, paddingSize);
}

Tensor& replication_pad2d_backward_out_cpu(const Tensor& gradOutput,
    const Tensor& input,
    IntArrayRef paddingSize,
    Tensor& gradInput)
{
  // 调整 gradInput 张量的大小和内存格式，以便与输入张量兼容
  gradInput.resize_as_(input, input.suggest_memory_format());
  // 将 gradInput 张量的所有元素置为 0
  gradInput.zero_();
  // 调用 replication_pad2d_backward_out_cpu_template 函数进行反向传播计算
  replication_pad2d_backward_out_cpu_template(
      gradInput, gradOutput, input, paddingSize);
  return gradInput;
}

Tensor replication_pad2d_backward_cpu(
    const Tensor& gradOutput,
    const Tensor& input,
    IntArrayRef paddingSize)
{
  // 创建一个与 input 张量形状相同的全零张量
  auto gradInput = at::zeros_like(input, input.suggest_memory_format());
  // 调用 replication_pad2d_backward_out_cpu_template 函数进行反向传播计算
  replication_pad2d_backward_out_cpu_template(
      gradInput, gradOutput, input, paddingSize);
  return gradInput;
}

TORCH_IMPL_FUNC(replication_pad3d_out_cpu) (
  const Tensor& input, IntArrayRef paddingSize, const Tensor& output
) {
// TODO: move this to TORCH_META_FUNC when CUDA has channels last support
// 警告：当 CUDA 支持通道最后（channels last）时，将此功能迁移到 TORCH_META_FUNC
output.resize_(output.sizes(), input.suggest_memory_format());

replication_pad3d_kernel(kCPU, output, input, paddingSize);
// 调用 replication_pad3d_kernel 函数，对输出进行 3D 复制填充操作
// 参数 kCPU 表示在 CPU 上执行，output 是输出张量，input 是输入张量，paddingSize 是填充尺寸
// 函数用于在 3D 情况下执行复制填充操作

Tensor& replication_pad3d_backward_out_cpu(const Tensor& gradOutput,
    const Tensor& input,
    IntArrayRef paddingSize,
    Tensor& gradInput)
{
  // 根据输入张量的大小，调整梯度输入张量的大小，并根据推荐的内存格式进行调整
  gradInput.resize_as_(input, input.suggest_memory_format());
  // 将梯度输入张量的所有元素置为零
  gradInput.zero_();
  // 调用模板函数 replication_pad3d_backward_out_cpu_template 进行 3D 复制填充的反向传播
  replication_pad3d_backward_out_cpu_template(
      gradInput, gradOutput, input, paddingSize);
  return gradInput;
}
// 返回更新后的梯度输入张量

Tensor replication_pad3d_backward_cpu(
    const Tensor& gradOutput,
    const Tensor& input,
    IntArrayRef paddingSize)
{
  // 根据输入张量的大小创建一个全零张量，并使用推荐的内存格式
  auto gradInput = at::zeros_like(input, input.suggest_memory_format());
  // 调用模板函数 replication_pad3d_backward_out_cpu_template 进行 3D 复制填充的反向传播
  replication_pad3d_backward_out_cpu_template(
      gradInput, gradOutput, input, paddingSize);
  return gradInput;
}
// 返回更新后的梯度输入张量，该函数用于计算 3D 复制填充的反向传播

DEFINE_DISPATCH(replication_pad1d_kernel);
DEFINE_DISPATCH(replication_pad1d_backward_kernel);
DEFINE_DISPATCH(replication_pad2d_kernel);
DEFINE_DISPATCH(replication_pad2d_backward_kernel);
DEFINE_DISPATCH(replication_pad3d_kernel);
DEFINE_DISPATCH(replication_pad3d_backward_kernel);
// 定义了各个维度下复制填充操作的 CUDA 调度器分发器

} // namespace at::native
// 结束 at::native 命名空间
```