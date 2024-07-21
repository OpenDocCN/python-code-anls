# `.\pytorch\aten\src\ATen\native\DilatedMaxPool2d.cpp`

```py
// 定义宏，限定仅使用方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 引入张量操作的头文件
#include <ATen/core/Tensor.h>
#include <ATen/core/NamedTensor.h>
#include <ATen/ScalarOps.h>
#include <ATen/TensorMeta.h>
#include <ATen/native/Pool.h>

// 如果未定义每个操作符的头文件，则引入通用操作符的头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
// 如果定义了每个操作符的头文件，则引入特定操作符的头文件
#else
#include <ATen/ops/max_pool2d_with_indices_backward_native.h>
#include <ATen/ops/max_pool2d_with_indices_native.h>
#endif

// 命名空间定义开始
namespace at::meta {
// 使用本地函数命名空间
using namespace at::native;

// 定义元函数：max_pool2d_with_indices
TORCH_META_FUNC(max_pool2d_with_indices)
// 函数签名：输入张量，核大小，步幅，填充，扩展模式
(const Tensor& input,
 IntArrayRef kernel_size,
 IntArrayRef stride,
 IntArrayRef padding,
 IntArrayRef dilation,
 bool ceil_mode) {

  // 检查核大小合法性：必须是单个整数或两个整数组成的元组
  TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 2,
    "max_pool2d: kernel_size must either be a single int, or a tuple of two ints")
  const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kW = kernel_size.size() == 1 ? kH : safe_downcast<int, int64_t>(kernel_size[1]);

  // 步幅合法性检查：可以省略，或者是单个整数，或者是两个整数组成的元组
  TORCH_CHECK(stride.empty() || stride.size() == 1 || stride.size() == 2,
    "max_pool2d: stride must either be omitted, a single int, or a tuple of two ints")
  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  const int dW = stride.empty() ? kW :
                 stride.size() == 1 ? dH : safe_downcast<int, int64_t>(stride[1]);

  // 填充合法性检查：必须是单个整数或两个整数组成的元组
  TORCH_CHECK(padding.size() == 1 || padding.size() == 2,
    "max_pool2d: padding must either be a single int, or a tuple of two ints");
  const int padH = safe_downcast<int, int64_t>(padding[0]);
  const int padW = padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);

  // 扩展合法性检查：必须是单个整数或两个整数组成的元组
  TORCH_CHECK(dilation.size() == 1 || dilation.size() == 2,
    "max_pool2d: dilation must be either a single int, or a tuple of two ints");
  const int dilationH = safe_downcast<int, int64_t>(dilation[0]);
  const int dilationW = dilation.size() == 1 ? dilationH : safe_downcast<int, int64_t>(dilation[1]);

  // 建议的内存格式检查
  const auto memory_format = input.suggest_memory_format();
  if (memory_format == at::MemoryFormat::ChannelsLast) {
    // 如果内存格式为ChannelsLast，要求输入张量维度必须为4（批处理模式）
    TORCH_CHECK(input.ndimension() == 4,
      "non-empty 4D (batch mode) tensor expected for input with channels_last layout");
  } else if (memory_format == at::MemoryFormat::Contiguous) {
    // 如果内存格式为Contiguous，要求输入张量维度必须为3或4（批处理模式）
    TORCH_CHECK((input.ndimension() == 3 || input.ndimension() == 4),
      "non-empty 3D or 4D (batch mode) tensor expected for input");
  } else {
  // 检查是否支持的内存格式，如果不支持则抛出错误信息
  TORCH_CHECK(false, "Unsupport memory format. Supports only ChannelsLast, Contiguous");
}

/* sizes */
// 计算输入张量的批次数、输入通道数、输入高度和宽度
const int64_t nbatch = input.ndimension() == 4 ? input.size(-4) : 1;
const int64_t nInputPlane = input.size(-3);
const int64_t inputHeight = input.size(-2);
const int64_t inputWidth = input.size(-1);

// 计算池化操作后的输出高度和宽度
const int64_t outputHeight = pooling_output_shape<int64_t>(inputHeight, kH, padH, dH, dilationH, ceil_mode);
const int64_t outputWidth = pooling_output_shape<int64_t>(inputWidth, kW, padW, dW, dilationW, ceil_mode);

// 检查池化操作的形状是否符合要求
pool2d_shape_check(
  input,
  kH, kW, dH, dW, padH, padW, dilationH, dilationW,
  nInputPlane,
  inputHeight, inputWidth,
  outputHeight, outputWidth, memory_format);

/* resize output and indices */
// 如果输入张量是三维的，设置输出张量的形状为 [nInputPlane, outputHeight, outputWidth]
set_output_raw_strided(0, {nInputPlane, outputHeight, outputWidth}, {}, input.options().memory_format(memory_format), maybe_names);
// indices 数组将包含每个输出点的位置信息
set_output_raw_strided(1, {nInputPlane, outputHeight, outputWidth}, {}, input.options().memory_format(memory_format).dtype(kLong), maybe_names);
// 如果输入张量不是三维的，设置输出张量的形状为 [nbatch, nInputPlane, outputHeight, outputWidth]
set_output_raw_strided(0, {nbatch, nInputPlane, outputHeight, outputWidth}, {}, input.options().memory_format(memory_format), maybe_names);
// indices 数组将包含每个输出点的位置信息
set_output_raw_strided(1, {nbatch, nInputPlane, outputHeight, outputWidth}, {}, input.options().memory_format(memory_format).dtype(kLong), maybe_names);
# 关闭之前的大括号，表示函数定义的结束
}

# 定义 TORCH_META_FUNC 函数，用于反向传播最大池化操作
TORCH_META_FUNC(max_pool2d_with_indices_backward)
(
    const Tensor& gradOutput,      // 输入参数：梯度输出张量
    const Tensor& input,           // 输入参数：输入张量
    IntArrayRef kernel_size,       // 输入参数：池化核大小
    IntArrayRef stride,            // 输入参数：步幅大小
    IntArrayRef padding,           // 输入参数：填充大小
    IntArrayRef dilation,          // 输入参数：膨胀大小
    bool ceil_mode,                // 输入参数：是否使用 ceil 模式
    const Tensor& indices          // 输入参数：最大值索引张量
) {
    // #20866, #22032: 为官方 C++ API 确保此操作
    // 检查 kernel_size 必须是单个整数或两个整数元组
    TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 2,
        "max_pool2d: kernel_size must either be a single int, or a tuple of two ints")
    const int kH = safe_downcast<int, int64_t>(kernel_size[0]);    // 获取池化核的高度
    const int kW = kernel_size.size() == 1 ? kH : safe_downcast<int, int64_t>(kernel_size[1]);  // 获取池化核的宽度

    // 注意事项：stride 默认值无法表示为整数常量，因此接受空的 stride
    // 检查 stride 必须省略、单个整数或两个整数元组
    TORCH_CHECK(stride.empty() || stride.size() == 1 || stride.size() == 2,
        "max_pool2d: stride must either be omitted, a single int, or a tuple of two ints")
    const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);  // 获取步幅的高度
    const int dW = stride.empty() ? kW :
                   stride.size() == 1 ? dH : safe_downcast<int, int64_t>(stride[1]);  // 获取步幅的宽度

    // 检查 padding 必须是单个整数或两个整数元组
    TORCH_CHECK(padding.size() == 1 || padding.size() == 2,
        "max_pool2d: padding must either be a single int, or a tuple of two ints");
    const int padH = safe_downcast<int, int64_t>(padding[0]);  // 获取填充的高度
    const int padW = padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);  // 获取填充的宽度

    // 检查 dilation 必须是单个整数或两个整数元组
    TORCH_CHECK(dilation.size() == 1 || dilation.size() == 2,
        "max_pool2d: dilation must be either a single int, or a tuple of two ints");
    const int dilationH = safe_downcast<int, int64_t>(dilation[0]);  // 获取膨胀的高度
    const int dilationW = dilation.size() == 1 ? dilationH : safe_downcast<int, int64_t>(dilation[1]);  // 获取膨胀的宽度

    // 检查输入和梯度输出张量的数据类型必须相同
    TORCH_CHECK(input.dtype() == gradOutput.dtype(),
        "expected dtype ", input.dtype(), " for `gradOutput` but got dtype ", gradOutput.dtype());

    const auto memory_format = input.suggest_memory_format();
    // 如果内存格式为 ChannelsLast
    if (memory_format == at::MemoryFormat::ChannelsLast) {
        // 检查输入张量必须是非空的 4D 张量（批处理模式），使用 channels_last 布局
        TORCH_CHECK(input.ndimension() == 4,
            "non-empty 4D (batch mode) tensor expected for input with channels_last layout");
    } else if (memory_format == at::MemoryFormat::Contiguous) {
        // 检查输入张量必须是非空的 3D 或 4D 张量（批处理模式）
        TORCH_CHECK((input.ndimension() == 3 || input.ndimension() == 4),
            "non-empty 3D or 4D (batch mode) tensor expected for input");
    } else {
        // 不支持的内存格式，仅支持 ChannelsLast 和 Contiguous
        TORCH_CHECK(false, "Unsupport memory format. Supports only ChannelsLast, Contiguous");
    }

    /* sizes */
    const int64_t nInputPlane = input.size(-3);       // 获取输入张量的通道数
    const int64_t inputHeight = input.size(-2);       // 获取输入张量的高度
    const int64_t inputWidth = input.size(-1);        // 获取输入张量的宽度

    /* XXX preserve the existing shape check behavior */
    // 获取池化操作后的输出高度和宽度（用于形状检查）
    const int64_t outputHeight_for_shape_check = pooling_output_shape<int64_t>(inputHeight, kH, padH, dH, dilationH, ceil_mode);
    const int64_t outputWidth_for_shape_check = pooling_output_shape<int64_t>(inputWidth, kW, padW, dW, dilationW, ceil_mode);

    // 执行反向池化形状检查
    max_pool2d_backward_shape_check(
        input,
        gradOutput,
        indices,
        kH, kW, dH, dW, padH, padW, dilationH, dilationW,
        nInputPlane,
    inputHeight, inputWidth,
    outputHeight_for_shape_check, outputWidth_for_shape_check,
    memory_format);


// 定义变量 inputHeight, inputWidth, outputHeight_for_shape_check, outputWidth_for_shape_check 和 memory_format，
// 这些变量可能用于后续的操作或函数调用。



  set_output_raw_strided(0, input.sizes(), {}, input.options().memory_format(memory_format),
             input.has_names() ? input.names() : DimnameList{});


// 调用 set_output_raw_strided 函数来设置第 0 个输出：
//   - 第一个参数是输出的索引号 0
//   - 第二个参数是 input 的大小 (sizes)，可能用于确定输出的尺寸
//   - 第三个参数是空的附加参数 ({} 表示空的字典)
//   - 第四个参数是通过 memory_format 构建的输入选项的内存格式
//   - 最后一个参数是根据 input 是否有名称来决定的 DimnameList，用于输出的命名
} // namespace at::meta



} // namespace at::meta
// 结束 at::meta 命名空间的定义

namespace at::native {
// 进入 at::native 命名空间的定义

TORCH_IMPL_FUNC(max_pool2d_with_indices_out_cpu)
(const Tensor& input,
 IntArrayRef kernel_size,
 IntArrayRef stride,
 IntArrayRef padding,
 IntArrayRef dilation,
 bool ceil_mode,
 const Tensor& output,
 const Tensor& indices) {
  NoNamesGuard guard;
  // 创建 NoNamesGuard 对象，用于临时禁用函数参数的名称

  const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
  // 将 kernel_size 的第一个元素转换为 int 类型，存储在 kH 中
  const int kW = kernel_size.size() == 1 ? kH : safe_downcast<int, int64_t>(kernel_size[1]);
  // 如果 kernel_size 只有一个元素，将 kH 赋给 kW；否则将第二个元素转换为 int 类型，存储在 kW 中

  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  // 如果 stride 为空，将 kH 赋给 dH；否则将 stride 的第一个元素转换为 int 类型，存储在 dH 中
  const int dW = stride.empty() ? kW :
                 stride.size() == 1 ? dH : safe_downcast<int, int64_t>(stride[1]);
  // 如果 stride 为空，将 kW 赋给 dW；如果 stride 只有一个元素，将 dH 赋给 dW；否则将 stride 的第二个元素转换为 int 类型，存储在 dW 中

  const int padH = safe_downcast<int, int64_t>(padding[0]);
  // 将 padding 的第一个元素转换为 int 类型，存储在 padH 中
  const int padW = padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);
  // 如果 padding 只有一个元素，将 padH 赋给 padW；否则将第二个元素转换为 int 类型，存储在 padW 中

  const int dilationH = safe_downcast<int, int64_t>(dilation[0]);
  // 将 dilation 的第一个元素转换为 int 类型，存储在 dilationH 中
  const int dilationW = dilation.size() == 1 ? dilationH : safe_downcast<int, int64_t>(dilation[1]);
  // 如果 dilation 只有一个元素，将 dilationH 赋给 dilationW；否则将第二个元素转换为 int 类型，存储在 dilationW 中

  max_pool2d_kernel(
      kCPU, output, indices, input,
      kW, kH,
      dW, dH,
      padW, padH,
      dilationW, dilationH);
  // 调用 max_pool2d_kernel 函数，执行最大池化操作，传递所需的参数

}

TORCH_IMPL_FUNC(max_pool2d_with_indices_backward_out_cpu)
(const Tensor& gradOutput,
 const Tensor& input,
 IntArrayRef kernel_size,
 IntArrayRef stride,
 IntArrayRef padding,
 IntArrayRef dilation,
 bool ceil_mode,
 const Tensor& indices,
 const Tensor& gradInput) {
  NoNamesGuard guard;
  // 创建 NoNamesGuard 对象，用于临时禁用函数参数的名称

  gradInput.zero_();
  // 将 gradInput 张量的所有元素清零

  max_pool2d_backward_kernel(
      kCPU, const_cast<Tensor&>(gradInput),
      gradOutput, indices);
  // 调用 max_pool2d_backward_kernel 函数，执行最大池化的反向传播操作，传递所需的参数
}

DEFINE_DISPATCH(max_pool2d_kernel);
DEFINE_DISPATCH(max_pool2d_backward_kernel);
// 定义 max_pool2d_kernel 和 max_pool2d_backward_kernel 的调度分发器

} // at::native
// 结束 at::native 命名空间的定义
```