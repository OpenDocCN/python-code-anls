# `.\pytorch\aten\src\ATen\native\NaiveConvolutionTranspose3d.cpp`

```
// 定义宏以仅使用方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 包含张量操作的头文件
#include <ATen/core/Tensor.h>
// 包含分发机制的头文件
#include <ATen/Dispatch.h>
// 包含张量工具函数的头文件
#include <ATen/TensorUtils.h>

// 包含卷积相关的实用函数和操作
#include <ATen/native/ConvUtils.h>
// 包含CPU BLAS的头文件
#include <ATen/native/CPUBlas.h>
// 包含vol2col操作的头文件
#include <ATen/native/vol2col.h>

// 根据编译宏的不同选择包含不同的操作和函数头文件
#ifndef AT_PER_OPERATOR_HEADERS
// 包含常规的操作和函数头文件
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
// 包含特定操作的头文件
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/slow_conv_transpose3d_native.h>
#include <ATen/ops/sum.h>
#endif

// 定义了 at::native 命名空间
namespace at::native {

// 模板函数，实现了 gemv 矩阵向量乘法
template<typename scalar_t>
void gemv(char trans, int64_t m, int64_t n, scalar_t alpha, scalar_t *a, int64_t lda, scalar_t *x, int64_t incx, scalar_t beta, scalar_t *y, int64_t incy);

// 匿名命名空间，用于实现一些内部函数或静态函数
namespace {

// 进行慢速卷积转置操作的形状检查函数
static inline void slow_conv_transpose3d_shape_check(
    const Tensor& input,
    const Tensor& grad_output,
    const Tensor& weight,
    const Tensor& bias,
    int kernel_depth,
    int kernel_width,
    int kernel_height,
    int stride_depth,
    int stride_width,
    int stride_height,
    int padding_depth,
    int padding_width,
    int padding_height,
    int dilation_depth,
    int dilation_width,
    int dilation_height,
    int output_padding_depth,
    int output_padding_width,
    int output_padding_height,
    int weight_nullable) {


// 输入参数包括一个 4D 或 5D 张量 input，还有几个整数参数，函数返回 void
TORCH_CHECK(
    input.numel() != 0 && (input.dim() == 4 || input.dim() == 5),
    "non-empty 4D or 5D (batch mode) tensor expected for input, but got: ",
    input.sizes());
// 检查输入张量 input 必须为非空的 4D 或 5D 张量

TORCH_CHECK(
    stride_depth > 0 && stride_width > 0 && stride_height > 0,
    "stride should be greater than zero, but got stride_depth: ",
    stride_depth,
    " stride_height: ",
    stride_height,
    " stride_width: ",
    stride_width);
// 检查步幅参数 stride_depth、stride_width、stride_height 必须大于零

TORCH_CHECK(
    dilation_depth > 0 && dilation_width > 0 && dilation_height > 0,
    "dilation should be greater than zero, but got dilation_depth: ",
    dilation_depth,
    ", dilation_height: ",
    dilation_height,
    ", dilation_width: ",
    dilation_width);
// 检查膨胀参数 dilation_depth、dilation_width、dilation_height 必须大于零

TORCH_CHECK(
    (output_padding_depth < stride_depth ||
     output_padding_depth < dilation_depth) &&
        (output_padding_width < stride_width ||
         output_padding_width < dilation_width) &&
        (output_padding_height < stride_height ||
         output_padding_height < dilation_height),
    "output padding must be smaller than either stride or dilation,",
    " but got output_padding_depth: ",
    output_padding_depth,
    " output_padding_height: ",
    output_padding_height,
    " output_padding_width: ",
    output_padding_width,
    " stride_depth: ",
    stride_depth,
    " stride_height: ",
    stride_height,
    " stride_width: ",
    stride_width,
    " dilation_depth: ",
    dilation_depth,
    " dilation_height: ",
    dilation_height,
    " dilation_width: ",
    dilation_width);
// 检查输出填充参数 output_padding 必须小于对应的步幅或膨胀参数

// number of input & output planes and kernel size is indirectly defined by
// the weight tensor
if (weight.defined()) {
    /* TODO: TORCH_CHECK just have 2 args: condition and message */
    TORCH_CHECK(
        weight.numel() != 0 && weight.dim() == 5,
        "non-empty 5D (n_output_plane x n_input_plane x kernel_depth",
        " x kernel_height x kernel_width) tensor ",
        "expected for weight, but got: ",
        weight.sizes());
    if (bias.defined()) {
        check_dim_size(bias, 1, 0, weight.size(1));
    }
} else if (!weight_nullable) {
    AT_ERROR("weight tensor is expected to be non-nullable");
}
// 如果 weight 已定义，则检查其是否为非空的 5D 张量，否则检查是否允许为空

int ndim = input.dim();
int dimf = 0;
int dimd = 1;
int dimh = 2;
int dimw = 3;

if (ndim == 5) {
    dimf++;
    dimd++;
    dimh++;
    dimw++;
}
// 确定输入张量的维度并设置对应的维度索引

if (weight.defined()) {
    const int64_t n_input_plane = weight.size(0);
  // 检查输入张量的指定维度是否与给定的维度大小相匹配
  check_dim_size(input, ndim, dimf, n_input_plane);
}

// 计算输入张量的宽度、高度和深度，以及卷积操作后的输出宽度、高度和深度
const int64_t input_width = input.size(dimw);
const int64_t input_height = input.size(dimh);
const int64_t input_depth = input.size(dimd);
const int64_t output_depth = (input_depth - 1) * stride_depth -
    2 * padding_depth + (dilation_depth * (kernel_depth - 1) + 1) +
    output_padding_depth;
const int64_t output_height = (input_height - 1) * stride_height -
    2 * padding_height + (dilation_height * (kernel_height - 1) + 1) +
    output_padding_height;
const int64_t output_width = (input_width - 1) * stride_width -
    2 * padding_width + (dilation_width * (kernel_width - 1) + 1) +
    output_padding_width;

// 如果计算得到的输出尺寸任一维度小于1，抛出错误信息
if (output_depth < 1 || output_width < 1 || output_height < 1) {
  AT_ERROR(
      "Given input size per channel: (",
      input_depth,
      " x ",
      input_height,
      " x ",
      input_width,
      "). "
      "Calculated output size per channel: (",
      output_depth,
      " x ",
      output_height,
      " x ",
      output_width,
      "). Output size is too small");
}

// 如果梯度输出张量已定义，则检查其与权重或偏置的维度匹配情况
if (grad_output.defined()) {
  if (weight.defined()) {
    // 如果权重张量已定义，获取输出平面数并检查梯度输出的维度
    const int64_t n_output_plane = weight.size(1);
    check_dim_size(grad_output, ndim, dimf, n_output_plane);
  } else if (bias.defined()) {
    // 如果偏置张量已定义，获取输出平面数并检查梯度输出的维度
    const int64_t n_output_plane = bias.size(0);
    check_dim_size(grad_output, ndim, dimf, n_output_plane);
  }
  // 检查梯度输出张量与计算得到的输出宽度、高度和深度是否匹配
  check_dim_size(grad_output, ndim, dimd, output_depth);
  check_dim_size(grad_output, ndim, dimh, output_height);
  check_dim_size(grad_output, ndim, dimw, output_width);
}
// 慢速 3D 转置卷积的 CPU 模板函数，用于计算输出的梯度
void slow_conv_transpose3d_out_cpu_template(
    Tensor& output,  // 输出张量，用于存储卷积操作的结果
    const Tensor& input_,  // 输入张量，可以是 4D 或 5D（带批处理）张量
    const Tensor& weight_,  // 权重张量，描述卷积核的参数（输入平面数 x 输出平面数 x 核深度 x 核高度 x 核宽度）
    IntArrayRef kernel_size,  // 卷积核的尺寸
    const Tensor& bias_,  // 偏置张量
    IntArrayRef stride,  // 步幅
    IntArrayRef padding,  // 填充
    IntArrayRef output_padding,  // 输出填充
    IntArrayRef dilation) {  // 膨胀

  TORCH_CHECK(
      kernel_size.size() == 3,
      "It is expected kernel_size equals to 3, but got size ",
      kernel_size.size());  // 检查卷积核尺寸的维度是否正确

  TORCH_CHECK(
      dilation.size() == 3,
      "It is expected dilation equals to 3, but got size ",
      dilation.size());  // 检查膨胀参数的维度是否正确

  TORCH_CHECK(
      padding.size() == 3,
      "It is expected padding equals to 3, but got size ",
      padding.size());  // 检查填充参数的维度是否正确

  TORCH_CHECK(
      stride.size() == 3,
      "It is expected stride equals to 3, but got size ",
      stride.size());  // 检查步幅参数的维度是否正确

  TORCH_CHECK(
      output_padding.size() == 3,
      "It is expected output_padding equals to 3, but got size ",
      output_padding.size());  // 检查输出填充参数的维度是否正确

  // 提取卷积核尺寸和膨胀参数的具体值
  int64_t kernel_depth = kernel_size[0];
  int64_t kernel_height = kernel_size[1];
  int64_t kernel_width = kernel_size[2];
  int64_t dilation_depth = dilation[0];
  int64_t dilation_height = dilation[1];
  int64_t dilation_width = dilation[2];
  // 提取填充参数的具体值
  int64_t padding_depth = padding[0];
  int64_t padding_height = padding[1];
  int64_t padding_width = padding[2];
  // 提取步幅参数的具体值
  int64_t stride_depth = stride[0];
  int64_t stride_height = stride[1];
  int64_t stride_width = stride[2];
  // 提取输出填充参数的具体值
  int64_t output_padding_depth = output_padding[0];
  int64_t output_padding_height = output_padding[1];
  int64_t output_padding_width = output_padding[2];

  // 执行形状检查，确保输入、权重和输出张量的维度和参数正确
  slow_conv_transpose3d_shape_check(
      input_,  // 输入张量
      Tensor(),  // 忽略的张量（在此函数中未使用）
      weight_,  // 权重张量
      bias_,  // 偏置张量
      kernel_depth,  // 卷积核深度
      kernel_width,  // 卷积核宽度
      kernel_height,  // 卷积核高度
      stride_depth,  // 步幅深度
      stride_width,  // 步幅宽度
      stride_height,  // 步幅高度
      padding_depth,  // 填充深度
      padding_width,  // 填充宽度
      padding_height,  // 填充高度
      dilation_depth,  // 膨胀深度
      dilation_width,  // 膨胀宽度
      dilation_height,  // 膨胀高度
      output_padding_depth,  // 输出填充深度
      output_padding_width,  // 输出填充宽度
      output_padding_height,  // 输出填充高度
      0);  // 不使用附加参数

  // 将输入张量转换为连续内存布局的张量
  Tensor input = input_.contiguous();
  // 将权重张量转换为连续内存布局的张量
  Tensor weight = weight_.contiguous();
  // 如果定义了偏置张量，则将其转换为连续内存布局的张量，否则保持原样
  Tensor bias = bias_.defined() ? bias_.contiguous() : bias_;

  // 提取权重张量的输入平面数和输出平面数
  const int n_input_plane = (int)weight.size(0);
  const int n_output_plane = (int)weight.size(1);

  // 检查是否为批处理模式，如果输入张量的维度为 4，则强制设置为批处理模式
  bool is_batch = false;
  if (input.dim() == 4) {
    // 强制启用批处理模式
    is_batch = true;
  }
}

// 慢速 3D 转置卷积的 CPU 模板函数，用于计算反向传播的梯度
void slow_conv_transpose3d_backward_out_cpu_template(
    const Tensor& input_,  // 输入张量
    const Tensor& grad_output_,  // 输出梯度张量
    Tensor& grad_input,  // 输入梯度张量
    const Tensor& weight_,  // 权重张量
    IntArrayRef kernel_size,  // 卷积核尺寸
    IntArrayRef stride,  // 步幅
    IntArrayRef padding,  // 填充
    IntArrayRef output_padding,  // 输出填充
  // 检查 kernel_size 是否为 3，如果不是则抛出错误信息
  TORCH_CHECK(
      kernel_size.size() == 3,
      "It is expected kernel_size equals to 3, but got size ",
      kernel_size.size());

  // 检查 dilation 是否为 3，如果不是则抛出错误信息
  TORCH_CHECK(
      dilation.size() == 3,
      "It is expected dilation equals to 3, but got size ",
      dilation.size());

  // 检查 padding 是否为 3，如果不是则抛出错误信息
  TORCH_CHECK(
      padding.size() == 3,
      "It is expected padding equals to 3, but got size ",
      padding.size());

  // 检查 stride 是否为 3，如果不是则抛出错误信息
  TORCH_CHECK(
      stride.size() == 3,
      "It is expected stride equals to 3, but got size ",
      stride.size());

  // 检查 output_padding 是否为 3，如果不是则抛出错误信息
  TORCH_CHECK(
      output_padding.size() == 3,
      "It is expected stride equals to 3, but got size ",
      output_padding.size());

  // 提取 kernel_size、dilation、padding、stride、output_padding 的各个维度值
  int64_t kernel_depth = kernel_size[0];
  int64_t kernel_height = kernel_size[1];
  int64_t kernel_width = kernel_size[2];
  int64_t dilation_depth = dilation[0];
  int64_t dilation_height = dilation[1];
  int64_t dilation_width = dilation[2];
  int64_t padding_depth = padding[0];
  int64_t padding_height = padding[1];
  int64_t padding_width = padding[2];
  int64_t stride_depth = stride[0];
  int64_t stride_height = stride[1];
  int64_t stride_width = stride[2];
  int64_t output_padding_depth = output_padding[0];
  int64_t output_padding_height = output_padding[1];
  int64_t output_padding_width = output_padding[2];

  // 使用 slow_conv_transpose3d_shape_check 函数检查输入参数的合法性
  slow_conv_transpose3d_shape_check(
      input_,
      grad_output_,
      weight_,
      Tensor(),
      kernel_depth,
      kernel_width,
      kernel_height,
      stride_depth,
      stride_width,
      stride_height,
      padding_depth,
      padding_width,
      padding_height,
      dilation_depth,
      dilation_width,
      dilation_height,
      output_padding_depth,
      output_padding_width,
      output_padding_height,
      0);

  // 对输入、权重和梯度输出进行内存连续性检查，并赋值给新的 Tensor 对象
  Tensor input = input_.contiguous();
  Tensor weight = weight_.contiguous();
  Tensor grad_output = grad_output_.contiguous();

  // 获取权重的输入和输出平面数量
  const int64_t n_input_plane = weight.size(0);
  const int64_t n_output_plane = weight.size(1);

  // 检查是否为批处理数据，如果是四维的，则强制设定为批处理数据
  bool is_batch = false;
  if (input.dim() == 4) {
    // 强制设定为批处理数据
    is_batch = true;
    // 重设 input 的维度，增加批处理维度
    input.resize_(
        {1, input.size(0), input.size(1), input.size(2), input.size(3)});
}

// 定义函数 `slow_conv_transpose3d_acc_grad_parameters_cpu`，用于计算 3D 转置卷积的梯度参数
void slow_conv_transpose3d_acc_grad_parameters_cpu(
    const Tensor& input_,                     // 输入张量
    const Tensor& grad_output_,               // 梯度输出张量
    Tensor& grad_weight,                      // 梯度权重张量
    Tensor& grad_bias,                        // 梯度偏置张量
    IntArrayRef kernel_size,                  // 卷积核大小
    IntArrayRef stride,                       // 步幅
    IntArrayRef padding,                      // 填充
    IntArrayRef output_padding,               // 输出填充
    IntArrayRef dilation,                     // 扩张
    int scale_) {                             // 缩放因子

  // 检查卷积核大小是否为3
  TORCH_CHECK(
      kernel_size.size() == 3,
      "It is expected kernel_size equals to 3, but got size ",
      kernel_size.size());

  // 检查扩张大小是否为3
  TORCH_CHECK(
      dilation.size() == 3,
      "It is expected dilation equals to 3, but got size ",
      dilation.size());

  // 检查填充大小是否为3
  TORCH_CHECK(
      padding.size() == 3,
      "It is expected padding equals to 3, but got size ",
      padding.size());

  // 检查步幅大小是否为3
  TORCH_CHECK(
      stride.size() == 3,
      "It is expected stride equals to 3, but got size ",
      stride.size());

  // 检查输出填充大小是否为3
  TORCH_CHECK(
      output_padding.size() == 3,
      "It is expected stride equals to 3, but got size ",
      output_padding.size());

  // 提取卷积核尺寸
  int64_t kernel_depth = kernel_size[0];
  int64_t kernel_height = kernel_size[1];
  int64_t kernel_width = kernel_size[2];

  // 提取扩张尺寸
  int64_t dilation_depth = dilation[0];
  int64_t dilation_height = dilation[1];
  int64_t dilation_width = dilation[2];

  // 提取填充尺寸
  int64_t padding_depth = padding[0];
  int64_t padding_height = padding[1];
  int64_t padding_width = padding[2];

  // 提取步幅尺寸
  int64_t stride_depth = stride[0];
  int64_t stride_height = stride[1];
  int64_t stride_width = stride[2];

  // 提取输出填充尺寸
  int64_t output_padding_depth = output_padding[0];
  int64_t output_padding_height = output_padding[1];
  int64_t output_padding_width = output_padding[2];

  // 调用函数检查输入输出形状是否匹配
  slow_conv_transpose3d_shape_check(
      input_,
      grad_output_,
      grad_weight,
      grad_bias,
      kernel_depth,
      kernel_width,
      kernel_height,
      stride_depth,
      stride_width,
      stride_height,
      padding_depth,
      padding_width,
      padding_height,
      dilation_depth,
      dilation_width,
      dilation_height,
      output_padding_depth,
      output_padding_width,
      output_padding_height,
      1);

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int64_t n_output_plane;

  // 如果定义了梯度权重，获取输出平面数目
  if (grad_weight.defined()) {
    n_output_plane = grad_weight.size(1);
  } else if (grad_bias.defined()) {
    // 否则，如果定义了梯度偏置，获取输出平面数目
    n_output_plane = grad_bias.size(0);
  } else {
    return;
  }

  // 使输入张量连续
  Tensor input = input_.contiguous();
  // 使梯度输出张量连续
  Tensor grad_output = grad_output_.contiguous();

  // 如果定义了梯度权重，检查其是否连续
  if (grad_weight.defined()) {
    TORCH_CHECK(grad_weight.is_contiguous(), "grad_weight needs to be contiguous");
  }
  // 如果定义了梯度偏置，检查其是否连续
  if (grad_bias.defined()) {
    TORCH_CHECK(grad_bias.is_contiguous(), "grad_bias needs to be contiguous");
  }

  bool is_batch = false;
  // 如果输入张量的维度为4，强制批处理
  if (input.dim() == 4) {
    is_batch = true;
    // 重置输入张量的尺寸，增加批次维度
    input.resize_(
        {1, input.size(0), input.size(1), input.size(2), input.size(3)});
  }
}

// namespace 结束
} // namespace
    // 定义一个函数，用于计算三维反卷积操作的CPU版本，并将结果输出到指定的张量中
    const Tensor& slow_conv_transpose3d_out_cpu(
        const Tensor& input,
        const Tensor& weight,
        IntArrayRef kernel_size,
        const std::optional<Tensor>& bias_opt,
        IntArrayRef stride,
        IntArrayRef padding,
        IntArrayRef output_padding,
        IntArrayRef dilation,
        Tensor& output) {
      // [注意：为可选张量的包装移除的 hack]
      // 从可能存在的可选张量中获取对权重的引用
      c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
      const Tensor& bias = *bias_maybe_owned;
    
      // 调用模板化的 CPU 版本的慢速三维反卷积操作
      slow_conv_transpose3d_out_cpu_template(
          output,           // 输出张量，接收计算结果
          input,            // 输入张量
          weight,           // 反卷积核权重张量
          kernel_size,      // 反卷积核大小
          bias,             // 可选的偏置张量
          stride,           // 反卷积步长
          padding,          // 反卷积填充
          output_padding,   // 输出填充
          dilation);        // 反卷积扩张系数
    
      // 返回计算结果的输出张量
      return output;
    }
// 定义一个名为 slow_conv_transpose3d_cpu 的函数，接受多个参数：
//   - input: 输入张量
//   - weight: 权重张量
//   - kernel_size: 卷积核大小
//   - bias_opt: 可选的偏置张量
//   - stride: 步长
//   - padding: 填充
//   - output_padding: 输出填充
//   - dilation: 膨胀率
Tensor slow_conv_transpose3d_cpu(
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size, const std::optional<Tensor>& bias_opt,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef dilation) {
  
  // 使用 borrow_from_optional_tensor 函数获取可能存在的偏置张量
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  // 创建一个与输入张量 input 类型相同、形状相同的空张量 output
  Tensor output = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

  // 调用 slow_conv_transpose3d_out_cpu_template 函数执行反卷积操作
  slow_conv_transpose3d_out_cpu_template(
      output,
      input,
      weight,
      kernel_size,
      bias,
      stride,
      padding,
      output_padding,
      dilation);

  // 返回输出张量 output
  return output;
}

// 定义一个静态函数 slow_conv_transpose3d_backward_cpu，返回一个包含三个张量的元组：
//   - grad_input: 输入张量的梯度
//   - grad_weight: 权重张量的梯度
//   - grad_bias: 偏置张量的梯度
static std::tuple<Tensor, Tensor, Tensor> slow_conv_transpose3d_backward_cpu(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef dilation,
    std::array<bool, 3> output_mask) {

  Tensor grad_input;
  Tensor grad_weight;
  Tensor grad_bias;

  // 根据 output_mask[0] 的值确定是否创建 grad_input 张量
  if (output_mask[0]) {
    grad_input = at::empty({0}, grad_output.options());
  } else {
    grad_input = Tensor();
  }

  // 根据 output_mask[1] 的值确定是否创建 grad_weight 张量
  if (output_mask[1]) {
    grad_weight = at::empty({0}, grad_output.options());
  } else {
    grad_weight = Tensor();
  }

  // 根据 output_mask[2] 的值确定是否创建 grad_bias 张量
  if (output_mask[2]) {
    grad_bias = at::empty({0}, grad_output.options());
  } else {
    grad_bias = Tensor();
  }

  // 如果 grad_input 已定义，调用 slow_conv_transpose3d_backward_out_cpu_template 函数计算梯度
  if (grad_input.defined()) {
    slow_conv_transpose3d_backward_out_cpu_template(
        input,
        grad_output,
        grad_input,
        weight,
        kernel_size,
        stride,
        padding,
        output_padding,
        dilation);
  }

  // 如果 grad_weight 已定义，将其形状调整为与 weight 相同并清零
  if (grad_weight.defined()) {
    grad_weight.resize_(weight.sizes());
    grad_weight.zero_();
  }

  // 如果 grad_bias 已定义，将其形状调整为与 weight 第二维相同并清零
  if (grad_bias.defined()) {
    grad_bias.resize_({weight.size(1)});
    grad_bias.zero_();
  }

  // 如果 grad_weight 或 grad_bias 已定义，则调用 slow_conv_transpose3d_acc_grad_parameters_cpu 累加参数的梯度
  if (grad_weight.defined() || grad_bias.defined()) {
    slow_conv_transpose3d_acc_grad_parameters_cpu(
        input,
        grad_output,
        grad_weight,
        grad_bias,
        kernel_size,
        stride,
        padding,
        output_padding,
        dilation,
        1);
  }

  // 返回包含 grad_input、grad_weight 和 grad_bias 的元组
  return std::tuple<Tensor, Tensor, Tensor>(grad_input, grad_weight, grad_bias);
}

// 注册 slow_conv_transpose3d_backward_stub 的 CPU 分发函数
REGISTER_ALL_CPU_DISPATCH(slow_conv_transpose3d_backward_stub, &slow_conv_transpose3d_backward_cpu);

// 结束命名空间 at::native
} // namespace at::native
```