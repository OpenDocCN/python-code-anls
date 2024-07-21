# `.\pytorch\aten\src\ATen\native\AveragePool2d.cpp`

```py
// 定义 TORCH_ASSERT_ONLY_METHOD_OPERATORS，用于在编译时启用特定的 Torch 断言
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 包含 ATen 核心 Tensor 类和标量操作头文件
#include <ATen/core/Tensor.h>
#include <ATen/ScalarOps.h>
// 包含 ATen 的池化操作相关头文件
#include <ATen/native/Pool.h>

// 如果未定义 AT_PER_OPERATOR_HEADERS，则包含一般的 ATen 函数和 Native 函数头文件；否则包含特定的池化操作头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/avg_pool2d_backward_native.h>
#include <ATen/ops/avg_pool2d_native.h>
#endif

// 命名空间定义：at::meta
namespace at::meta {
// 使用 at::native 命名空间
using namespace ::at::native;

// TORCH_PRECOMPUTE_META_FUNC 宏定义，用于预计算的 avg_pool2d 函数声明
TORCH_PRECOMPUTE_META_FUNC(avg_pool2d)
(const Tensor& input,
 IntArrayRef kernel_size,
 IntArrayRef stride,
 IntArrayRef padding,
 bool ceil_mode,
 bool count_include_pad,
 std::optional<int64_t> divisor_override) {
  // #20866, #22032: 对于官方的 C++ API，确保 kernel_size 是一个整数或两个整数的元组
  TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 2,
    "avg_pool2d: kernel_size must either be a single int, or a tuple of two ints");
  // 获取 kernel_size 的高度和宽度
  const int64_t kH = kernel_size[0];
  const int64_t kW = kernel_size.size() == 1 ? kH : kernel_size[1];

  // 检查 stride 参数的有效性
  TORCH_CHECK(stride.empty() || stride.size() == 1 || stride.size() == 2,
    "avg_pool2d: stride must either be omitted, a single int, or a tuple of two ints");
  // 获取 stride 的高度和宽度
  const int64_t dH = stride.empty() ? kH : stride[0];
  const int64_t dW = stride.empty() ? kW : stride.size() == 1 ? dH : stride[1];

  // 检查 padding 参数的有效性
  TORCH_CHECK(padding.size() == 1 || padding.size() == 2,
    "avg_pool2d: padding must either be a single int, or a tuple of two ints");
  // 获取 padding 的高度和宽度
  const int64_t padH = padding[0];
  const int64_t padW = padding.size() == 1 ? padH : padding[1];

  // 检查 divisor_override 是否有效
  TORCH_CHECK(!divisor_override.has_value() || divisor_override.value() != 0,
    "divisor must be not zero");

  // 获取输入张量的批次数、输入通道数、高度和宽度
  const int64_t nbatch = input.ndimension() == 4 ? input.size(-4) : 1;
  const int64_t nInputPlane = input.size(-3);
  const int64_t inputHeight = input.size(-2);
  const int64_t inputWidth = input.size(-1);

  // 计算池化操作后的输出高度和宽度
  const int64_t outputHeight = pooling_output_shape<int64_t>(
      inputHeight, kH, padH, dH, 1, ceil_mode);
  const int64_t outputWidth =
      pooling_output_shape<int64_t>(inputWidth, kW, padW, dW, 1, ceil_mode);

  // 推荐的内存格式
  auto memory_format = input.suggest_memory_format();
  // 执行池化形状检查
  pool2d_shape_check(
      input,
      kH,
      kW,
      dH,
      dW,
      padH,
      padW,
      1,
      1,
      nInputPlane,
      inputHeight,
      inputWidth,
      outputHeight,
      outputWidth,
      memory_format);

  /* resize output */
  // 如果输入张量维度为 3，则设置输出形状
  if (input.ndimension() == 3) {
    set_output_raw_strided(
        0,
        {nInputPlane,
         outputHeight,
         outputWidth},
        {},
        input.options());
  }
  // 否则设置批次维度的输出形状
  else {
    set_output_raw_strided(
        0,
        {nbatch,
         nInputPlane,
         outputHeight,
         outputWidth},
        {},
        input.options().memory_format(memory_format));
  }

  // 返回预计算的结构 avg_pool2d 的设置
  return TORCH_PRECOMPUTE_STRUCT(avg_pool2d)().set_kH(kH).set_kW(kW).set_dH(dH).set_dW(dW).set_padH(padH).set_padW(padW);
}
// 定义 avg_pool2d_backward_out_cpu 函数，用于计算平均池化操作的反向传播
TORCH_IMPL_FUNC(avg_pool2d_backward_out_cpu) (
  // 输入：梯度输出张量，反向传播的输入张量，池化核大小，步长，填充，是否向上取整，是否包含填充值，覆盖除数
  const Tensor& gradOutput,
  const Tensor& input,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  bool ceil_mode,
  bool count_include_pad,
  std::optional<int64_t> divisor_override,
  // 输出：梯度输入张量
  const Tensor& gradInput
) {
  // 检查：核大小只能是一个整数或两个整数的元组
  TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 2,
    "avg_pool2d: kernel_size must either be a single int, or a tuple of two ints");
  const int kH = safe_downcast<int, int64_t>(kernel_size[0]); // 安全转换池化核高度
  const int kW = kernel_size.size() == 1 ? kH : safe_downcast<int, int64_t>(kernel_size[1]); // 安全转换池化核宽度

  // 检查：步长可以省略、是一个整数或两个整数的元组
  TORCH_CHECK(stride.empty() || stride.size() == 1 || stride.size() == 2,
    "avg_pool2d: stride must either be omitted, a single int, or a tuple of two ints");
  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]); // 计算池化步长高度
  const int dW = stride.empty() ? kW :
                 stride.size() == 1 ? dH : safe_downcast<int, int64_t>(stride[1]); // 计算池化步长宽度

  // 检查：填充可以是一个整数或两个整数的元组
  TORCH_CHECK(padding.size() == 1 || padding.size() == 2,
    "avg_pool2d: padding must either be a single int, or a tuple of two ints");
  const int padH = safe_downcast<int, int64_t>(padding[0]); // 计算池化填充高度
  const int padW = padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]); // 计算池化填充宽度

  // 检查：如果覆盖除数存在，其值不能为零
  TORCH_CHECK(!divisor_override.has_value() || divisor_override.value() != 0, "divisor must be not zero");

  /* sizes */
  // 计算输入张量的批次大小、输入通道数、高度和宽度
  const int64_t nbatch = input.ndimension() == 4 ? input.size(-4) : 1; // 批次大小
  const int64_t nInputPlane = input.size(-3); // 输入通道数
  const int64_t inputHeight = input.size(-2); // 输入高度
  const int64_t inputWidth = input.size(-1); // 输入宽度

  // 计算输出的高度和宽度，通过池化输出形状函数计算
  const int64_t outputWidth = pooling_output_shape<int64_t>(inputWidth, kW, padW, dW, 1, ceil_mode);
  const int64_t outputHeight = pooling_output_shape<int64_t>(inputHeight, kH, padH, dH, 1, ceil_mode);

  auto memory_format = input.suggest_memory_format(); // 建议的内存格式
  // 执行平均池化反向传播的形状检查
  avg_pool2d_backward_shape_check(
    input,
    gradOutput,
    nbatch,
    kH, kW, dH, dW, padH, padW,
    nInputPlane,
    inputHeight, inputWidth,
    outputHeight, outputWidth,
    memory_format);

  /* resize output */
  // 设置输出张量的原始步进数组，并使用建议的内存格式设置选项
  set_output_raw_strided(0, input.sizes(), {}, input.options().memory_format(memory_format));
}
// 定义常量 kH 为 kernel_size 的第一个元素，使用 safe_downcast 将 int64_t 转换为 int 类型
const int kH = safe_downcast<int, int64_t>(kernel_size[0]);

// 定义常量 kW 为 kernel_size 的第二个元素（如果有），否则与 kH 相同，使用 safe_downcast 转换类型
const int kW = kernel_size.size() == 1 ? kH : safe_downcast<int, int64_t>(kernel_size[1]);

// 定义常量 dH 为 stride 的第一个元素（如果为空则使用 kH），使用 safe_downcast 转换类型
const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);

// 定义常量 dW 为 stride 的第二个元素（如果只有一个元素则使用 dH，否则使用 stride 的第二个元素），使用 safe_downcast 转换类型
const int dW = stride.empty() ? kW :
               stride.size() == 1 ? dH : safe_downcast<int, int64_t>(stride[1]);

// 定义常量 padH 为 padding 的第一个元素，使用 safe_downcast 转换类型
const int padH = safe_downcast<int, int64_t>(padding[0]);

// 定义常量 padW 为 padding 的第二个元素（如果只有一个元素则使用 padH），使用 safe_downcast 转换类型
const int padW = padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);

// 检查 divisor_override 不为空且其值不为零，否则抛出异常 "divisor must be not zero"
TORCH_CHECK(!divisor_override.has_value() || divisor_override.value() != 0, "divisor must be not zero");

// 检查 input 和 gradOutput 的数据类型是否一致，否则抛出异常，显示期望的 dtype 和实际得到的 dtype
TORCH_CHECK(input.dtype() == gradOutput.dtype(),
  "expected dtype ", input.dtype(), " for `gradOutput` but got dtype ", gradOutput.dtype());

/* 将 gradInput 的梯度置零 */
gradInput.zero_();

// 调用 avg_pool2d_backward_kernel 函数，传递相关参数进行反向平均池化操作
avg_pool2d_backward_kernel(
    kCPU, gradInput, gradOutput,
    kW, kH, dW, dH, padW, padH,
    count_include_pad, divisor_override);
```