# `.\pytorch\aten\src\ATen\native\DilatedMaxPool3d.cpp`

```
// 定义宏，用于仅包含方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 包含张量相关的头文件
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/native/Pool.h>

// 如果未定义每个操作符的头文件，则包含以下功能相关的头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
// 如果定义了每个操作符的头文件，则包含以下特定操作符的头文件
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/max_pool3d_with_indices_backward_native.h>
#include <ATen/ops/max_pool3d_with_indices_native.h>
#endif

// 创建 at::native 命名空间
namespace at::native {

// 创建匿名命名空间，用于限定作用域
namespace {

// 定义 max_pool3d_with_indices_out_cpu_template 函数
void max_pool3d_with_indices_out_cpu_template(
          Tensor& output, // 输出张量，存储池化后的结果
          Tensor& indices, // 输出张量，存储池化过程中的索引
          const Tensor& input, // 输入张量，进行池化操作的数据源
          IntArrayRef kernel_size, // 池化核大小
          IntArrayRef stride, // 池化步长
          IntArrayRef padding, // 池化填充
          IntArrayRef dilation, // 池化膨胀率
          bool ceil_mode) // 是否使用向上取整模式
{
  // #20866, #22032: 保证对于官方的 C++ API？
  TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 3,
    "max_pool3d: kernel_size must either be a single int, or a tuple of three ints")
  const int kT = safe_downcast<int, int64_t>(kernel_size[0]); // 获取池化核时间维度大小
  const int kH = kernel_size.size() == 1 ? kT : safe_downcast<int, int64_t>(kernel_size[1]); // 获取池化核高度维度大小
  const int kW = kernel_size.size() == 1 ? kT : safe_downcast<int, int64_t>(kernel_size[2]); // 获取池化核宽度维度大小

  TORCH_CHECK(stride.empty() || stride.size() == 1 || stride.size() == 3,
    "max_pool3d: stride must either be omitted, a single int, or a tuple of three ints")
  const int dT = stride.empty() ? kT : safe_downcast<int, int64_t>(stride[0]); // 获取池化步长时间维度大小
  const int dH = stride.empty() ? kH :
                 stride.size() == 1 ? dT : safe_downcast<int, int64_t>(stride[1]); // 获取池化步长高度维度大小
  const int dW = stride.empty() ? kW :
                 stride.size() == 1 ? dT : safe_downcast<int, int64_t>(stride[2]); // 获取池化步长宽度维度大小

  TORCH_CHECK(padding.size() == 1 || padding.size() == 3,
    "max_pool3d: padding must either be a single int, or a tuple of three ints");
  const int pT = safe_downcast<int, int64_t>(padding[0]); // 获取池化填充时间维度大小
  const int pH = padding.size() == 1 ? pT : safe_downcast<int, int64_t>(padding[1]); // 获取池化填充高度维度大小
  const int pW = padding.size() == 1 ? pT : safe_downcast<int, int64_t>(padding[2]); // 获取池化填充宽度维度大小

  TORCH_CHECK(dilation.size() == 1 || dilation.size() == 3,
    "max_pool3d: dilation must be either a single int, or a tuple of three ints");
  const int dilationT = safe_downcast<int, int64_t>(dilation[0]); // 获取池化膨胀率时间维度大小
  const int dilationH = dilation.size() == 1 ? dilationT : safe_downcast<int, int64_t>(dilation[1]); // 获取池化膨胀率高度维度大小
  const int dilationW = dilation.size() == 1 ? dilationT : safe_downcast<int, int64_t>(dilation[2]); // 获取池化膨胀率宽度维度大小

  const auto memory_format = input.suggest_memory_format(); // 建议输入张量的内存格式
  if (memory_format == at::MemoryFormat::ChannelsLast3d) {
    TORCH_CHECK(input.ndimension() == 5,
      "non-empty 5D (batch mode) tensor expected for input with channels_last_3d layout");
  } else if (memory_format == at::MemoryFormat::Contiguous) {
    TORCH_CHECK((input.ndimension() == 4 || input.ndimension() == 5),
      "non-empty 4D or 5D (batch mode) tensor expected for input");
  } else {
    # 检查条件，如果为 false，则输出错误信息并中止程序运行
    TORCH_CHECK(false, "Unsupport memory format. Supports only ChannelsLast3d, Contiguous");
  }

  # 计算输入张量的各维度大小
  const int64_t nslices = input.size(-4);   // 输入张量的通道数
  const int64_t itime = input.size(-3);      // 输入张量的时间维度大小
  const int64_t iheight = input.size(-2);    // 输入张量的高度维度大小
  const int64_t iwidth = input.size(-1);     // 输入张量的宽度维度大小

  # 计算池化操作后输出张量的各维度大小
  const int64_t otime = pooling_output_shape<int64_t>(itime, kT, pT, dT, dilationT, ceil_mode);    // 池化后时间维度大小
  const int64_t oheight = pooling_output_shape<int64_t>(iheight, kH, pH, dH, dilationH, ceil_mode); // 池化后高度维度大小
  const int64_t owidth = pooling_output_shape<int64_t>(iwidth, kW, pW, dW, dilationW, ceil_mode);   // 池化后宽度维度大小

  # 对池化后的输出张量和输入张量的形状进行检查，确保可以进行池化操作
  pool3d_shape_check(
    input,
    nslices,
    kT, kH, kW,          // 池化核的时间、高度、宽度
    dT, dH, dW,          // 池化的步长
    pT, pH, pW,          // 池化的填充
    dilationT, dilationH, dilationW,  // 池化的膨胀系数
    itime, iheight, iwidth,           // 输入张量的时间、高度、宽度
    otime, oheight, owidth,           // 池化后的时间、高度、宽度
    "max_pool3d_with_indices_out_cpu_template()");  // 错误信息提示

  if (input.dim() == 4) { /* non-batch mode */
    # 如果输入张量维度为 4，表示非批处理模式

    /* resize output */
    output.resize_({nslices, otime, oheight, owidth});  // 调整输出张量的大小为池化后的尺寸

    /* indices will contain ti,i,j locations for each output point */
    indices.resize_({nslices, otime, oheight, owidth});  // 调整索引张量的大小为池化后的尺寸
  }
  else { /* batch mode */
    # 如果输入张量维度不为 4，表示批处理模式
    const int64_t nbatch = input.size(0);  // 获取批次大小

    /* resize output */
    output.resize_({nbatch, nslices, otime, oheight, owidth}, memory_format);  // 调整输出张量的大小为池化后的尺寸，并指定存储格式

    /* indices will contain ti,i,j locations for each output point */
    indices.resize_({nbatch, nslices, otime, oheight, owidth}, memory_format);  // 调整索引张量的大小为池化后的尺寸，并指定存储格式
  }

  # 调用池化的核心函数，执行池化操作
  max_pool3d_kernel(
      kCPU, output, indices, input,
      kW, kH, kT,
      dW, dH, dT,
      pW, pH, pT,
      dilationW, dilationH, dilationT);
  // 确保 kernel_size 是长度为1或3的元组
  TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 3,
    "max_pool3d: kernel_size must either be a single int, or a tuple of three ints")
  // 将 kernel_size 转换为对应的整数变量
  const int kT = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kH = kernel_size.size() == 1 ? kT : safe_downcast<int, int64_t>(kernel_size[1]);
  const int kW = kernel_size.size() == 1 ? kT : safe_downcast<int, int64_t>(kernel_size[2]);

  // 确保 stride 是空、长度为1或3的元组
  TORCH_CHECK(stride.empty() || stride.size() == 1 || stride.size() == 3,
    "max_pool3d: stride must either be omitted, a single int, or a tuple of three ints")
  // 将 stride 转换为对应的整数变量
  const int dT = stride.empty() ? kT : safe_downcast<int, int64_t>(stride[0]);
  const int dH = stride.empty() ? kH :
                 stride.size() == 1 ? dT : safe_downcast<int, int64_t>(stride[1]);
  const int dW = stride.empty() ? kW :
                 stride.size() == 1 ? dT : safe_downcast<int, int64_t>(stride[2]);

  // 确保 padding 是长度为1或3的元组
  TORCH_CHECK(padding.size() == 1 || padding.size() == 3,
    "max_pool3d: padding must either be a single int, or a tuple of three ints");
  // 将 padding 转换为对应的整数变量
  const int pT = safe_downcast<int, int64_t>(padding[0]);
  const int pH = padding.size() == 1 ? pT : safe_downcast<int, int64_t>(padding[1]);
  const int pW = padding.size() == 1 ? pT : safe_downcast<int, int64_t>(padding[2]);

  // 确保 dilation 是长度为1或3的元组
  TORCH_CHECK(dilation.size() == 1 || dilation.size() == 3,
    "max_pool3d: dilation must be either a single int, or a tuple of three ints");
  // 将 dilation 转换为对应的整数变量
  const int dilationT = safe_downcast<int, int64_t>(dilation[0]);
  const int dilationH = dilation.size() == 1 ? dilationT : safe_downcast<int, int64_t>(dilation[1]);
  const int dilationW = dilation.size() == 1 ? dilationT : safe_downcast<int, int64_t>(dilation[2]);

  // 检查输入和梯度输出的数据类型必须一致
  TORCH_CHECK(input.dtype() == gradOutput.dtype(),
    "expected dtype ", input.dtype(), " for `gradOutput` but got dtype ", gradOutput.dtype());

  // 建议的内存格式为 ChannelsLast3d 时，检查输入张量的维度是否为5
  const auto memory_format = input.suggest_memory_format();
  if (memory_format == at::MemoryFormat::ChannelsLast3d) {
    TORCH_CHECK(input.ndimension() == 5,
      "non-empty 5D (batch mode) tensor expected for input with channels_last_3d layout");
  } else if (memory_format == at::MemoryFormat::Contiguous) {
    // 建议的内存格式为 Contiguous 时，检查输入张量的维度是否为4或5
    TORCH_CHECK((input.ndimension() == 4 || input.ndimension() == 5),
      "non-empty 4D or 5D (batch mode) tensor expected for input");
  } else {
    TORCH_CHECK(false, "Unsupport memory format. Supports only ChannelsLast3d, Contiguous");

# 检查条件为 false 时，抛出错误信息，指示内存格式不支持，只支持 ChannelsLast3d 和 Contiguous

  const int64_t nslices = input.size(-4);
  const int64_t itime = input.size(-3);
  const int64_t iheight = input.size(-2);
  const int64_t iwidth = input.size(-1);

  /* resize */

# 根据输入张量的尺寸和内存格式，重新调整梯度输入张量的尺寸
  gradInput.resize_(input.sizes(), memory_format);

# 将梯度输入张量初始化为零
  gradInput.zero_();

  const int64_t otime = gradOutput.size(-3);
  const int64_t oheight = gradOutput.size(-2);
  const int64_t owidth = gradOutput.size(-1);

# 执行形状检查，确保输入张量、梯度输出张量和索引张量的尺寸满足最大池化反向传播的要求
  max_pool3d_backward_shape_check(
    input,
    gradOutput,
    indices,
    nslices,
    kT, kH, kW,
    dT, dH, dW,
    pT, pH, pW,
    dilationT, dilationH, dilationW,
    itime, iheight, iwidth,
    otime, oheight, owidth,
    "max_pool3d_with_indices_backward_out_cpu_template()");

# 调用 CPU 上的最大池化反向传播的核函数，计算梯度输入张量
  max_pool3d_backward_kernel(
      kCPU, gradInput,
      gradOutput, indices);

# 返回计算后的梯度输入张量
  return gradInput;
} // namespace

// 定义函数 max_pool3d_with_indices_out_cpu，用于在 CPU 上执行带索引的三维最大池化操作
std::tuple<Tensor&, Tensor&> max_pool3d_with_indices_out_cpu(const Tensor& input,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  IntArrayRef dilation,
  bool ceil_mode,
  Tensor& output,
  Tensor& indices)
{
  // 调用模板函数执行实际的池化操作
  max_pool3d_with_indices_out_cpu_template(
    output,
    indices,
    input,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode);
  // 返回输出张量和索引张量的引用
  return std::tuple<Tensor&, Tensor&>(output, indices);
}

// 定义函数 max_pool3d_with_indices_cpu，在 CPU 上执行带索引的三维最大池化操作
std::tuple<Tensor, Tensor> max_pool3d_with_indices_cpu(
  const Tensor& input,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  IntArrayRef dilation,
  bool ceil_mode)
{
  // 创建 NoNamesGuard 对象
  NoNamesGuard guard;

  // 根据输入张量的选项创建空的输出张量和索引张量
  Tensor output = at::empty({0}, input.options());
  Tensor indices = at::empty({0}, input.options().dtype(kLong));

  // 调用模板函数执行实际的池化操作
  max_pool3d_with_indices_out_cpu_template(
    output,
    indices,
    input,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode);

  // 重置 NoNamesGuard
  guard.reset();

  // 根据输入张量推断输出张量和索引张量的命名
  namedinference::propagate_names(output, input);
  namedinference::propagate_names(indices, input);

  // 返回输出张量和索引张量的元组
  return std::tuple<Tensor, Tensor>(output, indices);
}

// 定义函数 max_pool3d_with_indices_backward_out_cpu，用于在 CPU 上执行带索引的三维最大池化的反向传播
Tensor& max_pool3d_with_indices_backward_out_cpu(const Tensor& gradOutput_,
  const Tensor& input,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  IntArrayRef dilation,
  bool ceil_mode,
  const Tensor& indices,
  Tensor& gradInput)
{
  // 调用模板函数执行实际的反向传播操作
  max_pool3d_with_indices_backward_out_cpu_template(
    gradInput,
    gradOutput_,
    input,
    indices,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode);
  // 返回梯度输入张量的引用
  return gradInput;
}

// 定义函数 max_pool3d_with_indices_backward_cpu，在 CPU 上执行带索引的三维最大池化的反向传播
Tensor max_pool3d_with_indices_backward_cpu(
  const Tensor& gradOutput_,
  const Tensor& input,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  IntArrayRef dilation,
  bool ceil_mode,
  const Tensor& indices)
{
  // 根据输入张量的选项创建空的梯度输入张量
  auto gradInput = at::empty({0}, input.options());

  // 调用模板函数执行实际的反向传播操作
  max_pool3d_with_indices_backward_out_cpu_template(
    gradInput,
    gradOutput_,
    input,
    indices,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode);
  
  // 返回梯度输入张量
  return gradInput;
}

// 定义 max_pool3d_kernel 的分发器
DEFINE_DISPATCH(max_pool3d_kernel);

// 定义 max_pool3d_backward_kernel 的分发器
DEFINE_DISPATCH(max_pool3d_backward_kernel);
} // namespace at::native
```