# `.\pytorch\aten\src\ATen\native\ReflectionPad.cpp`

```
// 定义宏，仅启用方法运算符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 包含 ATen 库的头文件
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/TensorMeta.h>
#include <ATen/quantized/Quantizer.h>
#include <ATen/native/Padding.h>
#include <c10/util/irange.h>

// 如果未定义 AT_PER_OPERATOR_HEADERS，则包含一组标准函数头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
// 否则，包含一组特定操作的头文件
#else
#include <ATen/ops/_empty_affine_quantized.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/reflection_pad1d_backward_native.h>
#include <ATen/ops/reflection_pad1d_native.h>
#include <ATen/ops/reflection_pad2d_backward_native.h>
#include <ATen/ops/reflection_pad2d_native.h>
#include <ATen/ops/reflection_pad3d_backward_native.h>
#include <ATen/ops/reflection_pad3d_native.h>
#include <ATen/ops/zeros_like.h>
#endif

// 定义命名空间 at::meta
namespace at::meta {

// TORCH_META_FUNC 是一个宏，用于定义元数据函数 reflection_pad1d
TORCH_META_FUNC(reflection_pad1d)(const Tensor& input, IntArrayRef padding) {
  int64_t dim_plane = 0; // 初始化平面维度索引为0
  int64_t dim_w = 1; // 初始化宽度维度索引为1
  int64_t nbatch = 1; // 初始化批次数为1

  // 如果输入张量维度为3
  if (input.ndimension() == 3) {
    nbatch = input.size(0); // 更新批次数为输入张量的第0维大小
    dim_w++; // 增加宽度维度索引
    dim_plane++; // 增加平面维度索引
  }

  // 调用 padding 模块的函数检查输入的有效性
  at::native::padding::check_valid_input<1>(input, padding);

  /* sizes */
  auto pad_l = padding[0]; // 获取左填充大小
  auto pad_r = padding[1]; // 获取右填充大小

  int64_t nplane = input.size(dim_plane); // 获取平面维度的大小
  int64_t input_w = input.size(dim_w); // 获取宽度维度的大小
  int64_t output_w = input_w + pad_l + pad_r; // 计算输出宽度

  // 使用 TORCH_CHECK 断言检查填充大小是否有效
  TORCH_CHECK(
      pad_l < input_w && pad_r < input_w,
      "Argument #4: Padding size "
      "should be less than the corresponding input dimension, but got: padding (",
      pad_l,
      ", ",
      pad_r,
      ") at dimension ",
      dim_w,
      " of input ",
      input.sizes());

  // 使用 TORCH_CHECK 断言检查输出宽度是否有效
  TORCH_CHECK(
      output_w >= 1,
      "input (W: ",
      input_w,
      ") is too small. Calculated output W: ",
      output_w);

  // 如果输入张量维度为2，则设置输出张量的大小为 [nplane, output_w]
  if (input.ndimension() == 2) {
    set_output_raw_strided(0, {nplane, output_w}, {}, input.options());
  } else { // 否则，设置输出张量的大小为 [nbatch, nplane, output_w]
    set_output_raw_strided(0, {nbatch, nplane, output_w}, {}, input.options());
  }
}

// TORCH_META_FUNC 是一个宏，用于定义元数据函数 reflection_pad1d_backward
TORCH_META_FUNC(reflection_pad1d_backward)(const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding) {
  int64_t dim_w = 1; // 初始化宽度维度索引为1
  // 如果输入张量维度为3
  if (input.ndimension() == 3) {
    dim_w++; // 增加宽度维度索引
  }

  /* sizes */
  auto pad_l = padding[0]; // 获取左填充大小
  auto pad_r = padding[1]; // 获取右填充大小
  int64_t input_w = input.size(dim_w); // 获取宽度维度的大小
  int64_t output_w  = input_w + pad_l + pad_r; // 计算输出宽度

  // 使用 TORCH_CHECK 断言检查填充大小是否有效
  TORCH_CHECK(
      pad_l < input_w && pad_r < input_w,
      "Argument #4: Padding size "
      "should be less than the corresponding input dimension, but got: padding (",
      pad_l,
      ", ",
      pad_r,
      ") at dimension ",
      dim_w,
      " of input ",
      input.sizes());

  // 使用 TORCH_CHECK 断言检查输出宽度是否匹配梯度输出张量的宽度
  TORCH_CHECK(output_w == grad_output.size(dim_w), "grad_output width unexpected."
    " Expected: ", output_w, ", Got: ", grad_output.size(dim_w));

  // 设置输出张量的大小与输入张量相同
  set_output_raw_strided(0, input.sizes(), {}, input.options());
}

} // 命名空间结束
// 定义反射填充函数 `reflection_pad3d`，接受输入张量 `input` 和填充数组 `padding`
TORCH_META_FUNC(reflection_pad3d)(const Tensor& input, IntArrayRef padding) {
  // 从填充数组中提取各个方向的填充值
  int64_t pad_left = padding[0];
  int64_t pad_right = padding[1];
  int64_t pad_top = padding[2];
  int64_t pad_bottom = padding[3];
  int64_t pad_front = padding[4];
  int64_t pad_back = padding[5];
  // 定义输入张量的维度索引
  int64_t dim_w = 3;
  int64_t dim_h = 2;
  int64_t dim_d = 1;
  int64_t dim_plane = 0;

  // 检查输入和填充的有效性，确保填充不超过输入张量的尺寸
  at::native::padding::check_valid_input<3>(input, padding);

  // 确定是否处于批处理模式
  bool batch_mode = (input.dim() == 5);
  if (batch_mode) {
    // 如果是批处理模式，调整维度索引
    dim_w++;
    dim_h++;
    dim_d++;
    dim_plane++;
  }

  // 获取输入张量的相关尺寸信息
  int64_t nplane = input.size(dim_plane);
  int64_t input_d = input.size(dim_d);
  int64_t input_h = input.size(dim_h);
  int64_t input_w = input.size(dim_w);
  // 计算输出张量的尺寸
  int64_t output_d = input_d + pad_front + pad_back;
  int64_t output_h = input_h + pad_top + pad_bottom;
  int64_t output_w = input_w + pad_left + pad_right;

  // 检查计算得到的输出尺寸是否合理
  TORCH_CHECK(
      pad_left < input_w && pad_right < input_w,
      "Argument #4: Padding size "
      "should be less than the corresponding input dimension, but got: padding (",
      pad_left, ", ", pad_right, ") at dimension ", dim_w, " of input ", input.sizes());
  TORCH_CHECK(
      pad_top < input_h && pad_bottom < input_h,
      "Argument #6: Padding size "
      "should be less than the corresponding input dimension, but got: padding (",
      pad_top, ", ", pad_bottom, ") at dimension ", dim_h, " of input ", input.sizes());
  TORCH_CHECK(
      pad_front < input_d && pad_back < input_d,
      "Argument #8: Padding size "
      "should be less than the corresponding input dimension, but got: padding (",
      pad_front, ", ", pad_back, ") at dimension ", dim_d, " of input ", input.sizes());

  // 检查输出尺寸是否有效，至少应为 1
  TORCH_CHECK(output_w >= 1 || output_h >= 1 || output_d >= 1,
      "input (D: ", input_d, " H: ", input_h, ", W: ", input_w,
      ") is too small."
      " Calculated output D: ", output_d, " H: ", output_h, " W: ", output_w);

  // 如果处于批处理模式，设置输出张量的尺寸和选项
  if (batch_mode) {
    set_output_raw_strided(0, {input.size(0), nplane, output_d, output_h, output_w}, {}, input.options());
  } else {
    set_output_raw_strided(0, {nplane, output_d, output_h, output_w}, {}, input.options());
  }
}

// 定义反向传播函数 `reflection_pad3d_backward`，接受梯度输出张量 `grad_output`、输入张量 `input` 和填充数组 `padding`
TORCH_META_FUNC(reflection_pad3d_backward)(
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding
) {
  // 检查填充数组的尺寸是否为 6
  TORCH_CHECK(padding.size() == 6, "padding size is expected to be 6");
  // 检查输入张量维度是否大于 3
  TORCH_CHECK(input.dim() > 3);
  // 检查梯度输出张量的维度与输入张量相同
  TORCH_CHECK(grad_output.dim() == input.dim());

  // 从填充数组中提取各个方向的填充值
  int64_t pad_left = padding[0];
  int64_t pad_right = padding[1];
  int64_t pad_top = padding[2];
  int64_t pad_bottom = padding[3];
  int64_t pad_front = padding[4];
  int64_t pad_back = padding[5];
  // 定义输入张量的维度索引
  int64_t dim_w = 3;
  int64_t dim_h = 2;
  int64_t dim_d = 1;

  // 如果输入张量处于批处理模式
  if (input.dim() == 5) {
    // 调整维度索引以匹配批处理模式
    dim_w++;
    dim_h++;
    dim_d++;

# 增加深度维度计数器，用于表示下一个深度维度的索引

  int64_t input_d = input.size(dim_d);

# 获取输入张量在当前深度维度上的大小

  int64_t input_h = input.size(dim_h);

# 获取输入张量在当前高度维度上的大小

  int64_t input_w = input.size(dim_w);

# 获取输入张量在当前宽度维度上的大小

  int64_t output_d = input_d + pad_front + pad_back;

# 计算输出张量在深度维度上的大小，包括前后填充

  int64_t output_h = input_h + pad_top + pad_bottom;

# 计算输出张量在高度维度上的大小，包括顶部和底部填充

  int64_t output_w = input_w + pad_left + pad_right;

# 计算输出张量在宽度维度上的大小，包括左侧和右侧填充

  TORCH_CHECK(output_w == grad_output.size(dim_w), "grad_output width unexpected."
    " Expected: ", output_w, ", Got: ", grad_output.size(dim_w));

# 检查梯度输出张量的宽度是否符合预期，若不符则抛出错误信息

  TORCH_CHECK(output_h == grad_output.size(dim_h), "grad_output height unexpected."
    " Expected: ", output_h, ", Got: ", grad_output.size(dim_h));

# 检查梯度输出张量的高度是否符合预期，若不符则抛出错误信息

  TORCH_CHECK(output_d == grad_output.size(dim_d), "grad_output depth unexpected."
    " Expected: ", output_d, ", Got: ", grad_output.size(dim_d));

# 检查梯度输出张量的深度是否符合预期，若不符则抛出错误信息

  set_output_raw_strided(0, input.sizes(), {}, input.options());

# 设置输出张量的原始步幅，从输入张量获取大小和选项
} // namespace at::meta
} // namespace at::native

namespace at::native {

namespace {

// 实现 reflection_pad2d_out_template 函数，用于在输出张量上执行 2D 反射填充操作
void reflection_pad2d_out_template(
    Tensor &output, const Tensor &input, IntArrayRef padding) {
  
  // 设置初始维度变量
  int dim_w = 2;
  int dim_h = 1;
  int dim_slices = 0;
  int64_t nbatch = 1;

  // 检查输入的有效性，确保填充参数与输入张量维度相符合
  at::native::padding::check_valid_input<2>(input, padding);

  // 获取输入张量的维度数
  int ndim = input.dim();
  if (ndim == 4) {
    // 如果输入张量是4维的，更新相关维度计数
    nbatch = input.size(0);
    dim_w++;
    dim_h++;
    dim_slices++;
  }

  /* sizes */
  // 获取填充的具体数值
  int64_t pad_l = padding[0];
  int64_t pad_r = padding[1];
  int64_t pad_t = padding[2];
  int64_t pad_b = padding[3];

  // 获取输入张量在各个维度上的大小
  int64_t nplane = input.size(dim_slices);
  int64_t input_h = input.size(dim_h);
  int64_t input_w = input.size(dim_w);
  int64_t output_h = input_h + pad_t + pad_b;
  int64_t output_w  = input_w + pad_l + pad_r;

  // 检查填充大小是否合理，不应超过对应的输入张量维度
  TORCH_CHECK(pad_l < input_w && pad_r < input_w,
    "Argument #4: Padding size should be less than the corresponding "
    "input dimension, but got: padding (", pad_l, ", ", pad_r,
    ") at dimension ", dim_w, " of input ", input.sizes());

  TORCH_CHECK(pad_t < input_h && pad_b < input_h,
    "Argument #6: Padding size should be less than the corresponding "
    "input dimension, but got: padding (", pad_t, ", ", pad_b,
    ") at dimension ", dim_h, " of input ", input.sizes());

  // 检查输出的尺寸是否合理，至少应为1
  TORCH_CHECK(output_w >= 1 || output_h >= 1,
    "input (H: ", input_h, ", W: ", input_w, ") is too small. Calculated "
    "output H: ", output_h, " W: ", output_w);

  /* resize output */
  // 根据输入张量的维度数进行输出张量的调整
  if (ndim == 3) {
    output.resize_({nplane, output_h, output_w});
  } else {
    if (input.is_quantized()) {
      // 如果输入张量是量化的，则不能使用 `memory_format` 参数来调整大小
      output.resize_({nbatch, nplane, output_h, output_w});
    } else {
      // 否则根据建议的内存格式调整输出张量的大小
      output.resize_({nbatch, nplane, output_h, output_w}, input.suggest_memory_format());
    }
  }
  // 调用 reflection_pad2d_kernel 函数执行具体的反射填充操作
  reflection_pad2d_kernel(kCPU, output, input, padding);
}

// 实现 reflection_pad2d_backward_out_template 函数，用于在反向传播时执行 2D 反射填充的梯度计算
void reflection_pad2d_backward_out_template(
    Tensor &grad_input, const Tensor &grad_output,
    const Tensor &input, IntArrayRef padding) {
  
  // 设置初始维度变量
  int dim_w = 2;
  int dim_h = 1;

  // 如果输入张量是4维的，更新相关维度计数
  if (input.ndimension() == 4) {
    dim_w++;
    dim_h++;
  }

  /* sizes */
  // 获取填充的具体数值
  int64_t pad_l = padding[0];
  int64_t pad_r = padding[1];
  int64_t pad_t = padding[2];
  int64_t pad_b = padding[3];

  // 获取输入张量在各个维度上的大小
  int64_t input_h = input.size(dim_h);
  int64_t input_w = input.size(dim_w);
  int64_t output_h = input_h + pad_t + pad_b;
  int64_t output_w  = input_w + pad_l + pad_r;

  // 检查梯度张量的尺寸是否与预期相符合
  TORCH_CHECK(output_w == grad_output.size(dim_w),
    "gradOutput width unexpected. Expected: ", output_w, ", Got: ",
    grad_output.size(dim_w));

  TORCH_CHECK(output_h == grad_output.size(dim_h),
    "gradOutput height unexpected. Expected: ", output_h, ", Got: ",
    grad_output.size(dim_h));

  // 调用 reflection_pad2d_backward_kernel 函数执行具体的反向传播计算
  reflection_pad2d_backward_kernel(kCPU, grad_input, grad_output, padding);
}

} // namespace

// 定义 reflection_pad1d_out_quantized_cpu 函数，用于在量化 CPU 上执行 1D 反射填充操作
Tensor& reflection_pad1d_out_quantized_cpu(const Tensor& input, IntArrayRef padding,
    // 检查输入张量的量化方案是否为每张量仿射量化，如果不是则抛出错误
    TORCH_CHECK(input.qscheme() == kPerTensorAffine, "Only per tensor quantization is supported");
    // 使用输入张量的量化参数创建一个每张量仿射量化器，并将其设置为输出张量的量化器
    set_quantizer_(output, make_per_tensor_affine_quantizer(input.q_scale(), input.q_zero_point(), input.scalar_type()));
    // 调用 CPU 上的 1 维反射填充卷积核函数，对输入张量进行反射填充操作，结果存入输出张量
    reflection_pad1d_kernel(kCPU, output, input, padding);
    // 返回经过反射填充后的输出张量
    return output;
# 定义 reflection_pad1d_out_cpu 函数，用于计算 1 维反射填充后的输出到指定的张量
TORCH_IMPL_FUNC(reflection_pad1d_out_cpu)
(const Tensor& input, IntArrayRef padding, const Tensor& output) {
  // 调用 CPU 上的 reflection_pad1d_kernel 函数执行反射填充操作
  reflection_pad1d_kernel(kCPU, output, input, padding);
}

# 定义 reflection_pad1d_backward_out_cpu 函数，用于计算 1 维反射填充的梯度到指定的张量
TORCH_IMPL_FUNC(reflection_pad1d_backward_out_cpu)(const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding,
    const Tensor& grad_input) {
  # 如果梯度输出的元素数为 0，则直接返回
  if (grad_output.numel() == 0) {
    return;
  }

  # 将梯度输入张量 grad_input 的所有元素置零
  grad_input.zero_();
  # 调用 CPU 上的 reflection_pad1d_backward_kernel 函数执行反射填充的反向传播操作
  reflection_pad1d_backward_kernel(kCPU, grad_input, grad_output, padding);
}

# 定义 reflection_pad2d_out_cpu 函数，用于计算 2 维反射填充后的输出到指定的张量
Tensor& reflection_pad2d_out_cpu(const Tensor& input, IntArrayRef padding,
    Tensor& output) {
  # 调用模板函数 reflection_pad2d_out_template 执行 2 维反射填充操作，并将结果写入 output 张量
  reflection_pad2d_out_template(output, input, padding);
  return output;
}

# 定义 reflection_pad2d_cpu 函数，用于计算 2 维反射填充后的输出张量
Tensor reflection_pad2d_cpu(const Tensor& input, IntArrayRef padding) {
  # 创建一个空的张量 output，与输入 input 具有相同的选项
  Tensor output = at::empty({0}, input.options());
  # 调用模板函数 reflection_pad2d_out_template 执行 2 维反射填充操作，并将结果写入 output 张量
  reflection_pad2d_out_template(output, input, padding);
  return output;
}

# 定义 reflection_pad2d_quantized_cpu 函数，用于计算量化的 2 维反射填充后的输出张量
Tensor reflection_pad2d_quantized_cpu(const Tensor& input, IntArrayRef padding) {
  # 检查输入张量 input 的量化方案是否为 kPerTensorAffine，若不是则报错
  TORCH_CHECK(input.qscheme() == kPerTensorAffine, "Only per tensor quantization is supported");
  # 使用输入张量 input 的量化参数创建一个空的量化张量 output
  Tensor output = at::_empty_affine_quantized({0}, input.options(),
                                           input.q_scale(),
                                           input.q_zero_point());
  # 调用模板函数 reflection_pad2d_out_template 执行 2 维反射填充操作，并将结果写入 output 张量
  reflection_pad2d_out_template(output, input, padding);
  return output;
}

# 定义 reflection_pad2d_backward_out_cpu 函数，用于计算 2 维反射填充的反向传播梯度到指定的张量
Tensor& reflection_pad2d_backward_out_cpu(const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding,
    Tensor& grad_input) {
  # 调整 grad_input 张量的大小为与输入 input 相同，并建议使用的内存格式
  grad_input.resize_as_(input, input.suggest_memory_format());
  # 将 grad_input 张量的所有元素置零
  grad_input.zero_();
  # 调用模板函数 reflection_pad2d_backward_out_template 执行 2 维反射填充的反向传播操作
  reflection_pad2d_backward_out_template(
    grad_input, grad_output, input, padding);
  return grad_input;
}

# 定义 reflection_pad2d_backward_cpu 函数，用于计算 2 维反射填充的反向传播梯度张量
Tensor reflection_pad2d_backward_cpu(
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding) {
  # 创建一个与输入 input 相同大小的零张量 grad_input，并使用建议的内存格式
  auto grad_input = at::zeros_like(input, input.suggest_memory_format());
  # 调用模板函数 reflection_pad2d_backward_out_template 执行 2 维反射填充的反向传播操作
  reflection_pad2d_backward_out_template(
    grad_input, grad_output, input, padding);
  return grad_input;
}

# 定义 reflection_pad3d_out_cpu 函数，用于计算 3 维反射填充后的输出到指定的张量
TORCH_IMPL_FUNC(reflection_pad3d_out_cpu)
(const Tensor& input, IntArrayRef padding, const Tensor& output) {
  // TODO: 在 CUDA 支持通道最后格式时将此移至 TORCH_META_FUNC
  // 调整 output 张量的大小为与输入 input 相同，并使用建议的内存格式
  output.resize_(output.sizes(), input.suggest_memory_format());

  // 调用 CPU 上的 reflection_pad3d_kernel 函数执行 3 维反射填充操作
  reflection_pad3d_kernel(kCPU, output, input, padding);
}

# 定义 reflection_pad3d_backward_out_cpu 函数，用于计算 3 维反射填充的反向传播梯度到指定的张量
TORCH_IMPL_FUNC(reflection_pad3d_backward_out_cpu)(const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding,
    const Tensor& grad_input) {
  # 如果梯度输出的元素数为 0，则直接返回
  if (grad_output.numel() == 0) {
    return;
  }

  // TODO: 在 CUDA 支持通道最后格式时将此移至 TORCH_META_FUNC
  // 调整 grad_input 张量的大小为与输入 input 相同，并使用建议的内存格式
  grad_input.resize_(input.sizes(), input.suggest_memory_format());

  # 将 grad_input 张量的所有元素置零
  grad_input.zero_();
  # 调用 CPU 上的 reflection_pad3d_backward_kernel 函数执行 3 维反射填充的反向传播操作
  reflection_pad3d_backward_kernel(kCPU, grad_input, grad_output, padding);
}

# 定义 reflection_pad1d_kernel 函数的分发器
DEFINE_DISPATCH(reflection_pad1d_kernel);

# 定义 reflection_pad1d_backward_kernel 函数的分发器
DEFINE_DISPATCH(reflection_pad1d_backward_kernel);

# 定义 reflection_pad2d_kernel 函数的分发器
DEFINE_DISPATCH(reflection_pad2d_kernel);

# 定义 reflection_pad2d_backward_kernel 函数的分发器
DEFINE_DISPATCH(reflection_pad2d_backward_kernel);

# 定义 reflection_pad3d_kernel 函数的分发器
DEFINE_DISPATCH(reflection_pad3d_kernel);
DEFINE_DISPATCH(reflection_pad3d_backward_kernel);


// 定义一个名为 reflection_pad3d_backward_kernel 的宏或函数，用于分发反射填充的三维反向操作的内核函数。



} // namespace at::native


// 结束 at::native 命名空间的定义
```