# `.\pytorch\aten\src\ATen\native\Im2Col.cpp`

```py
// 定义宏，仅声明操作符方法
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 引入 PyTorch 的 Tensor 相关头文件
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorUtils.h>

// 引入 PyTorch 的 im2col 相关头文件和形状检查头文件
#include <ATen/native/im2col.h>
#include <ATen/native/im2col_shape_check.h>
#include <c10/util/irange.h>

// 根据预处理器宏选择是否引入具体操作函数的头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/col2im_native.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/im2col_native.h>
#endif

// 定义 at::native 命名空间
namespace at::native {

// 匿名命名空间，用于限定静态函数的作用域
namespace {

// 静态函数，实现 CPU 上的 im2col 操作，输出到指定的 Tensor
static void im2col_out_cpu_template(
    Tensor& output,                   // 输出 Tensor 的引用
    const Tensor& input_,             // 输入 Tensor 的常量引用
    IntArrayRef kernel_size,          // 卷积核大小
    IntArrayRef dilation,             // 扩展大小
    IntArrayRef padding,              // 填充大小
    IntArrayRef stride) {             // 步长大小

  // 检查卷积核大小是否为 2
  TORCH_CHECK(
      kernel_size.size() == 2,
      "It is expected kernel_size equals to 2, but got size ",
      kernel_size.size());

  // 检查扩展大小是否为 2
  TORCH_CHECK(
      dilation.size() == 2,
      "It is expected dilation equals to 2, but got size ",
      dilation.size());

  // 检查填充大小是否为 2
  TORCH_CHECK(
      padding.size() == 2,
      "It is expected padding equals to 2, but got size ",
      padding.size());

  // 检查步长大小是否为 2
  TORCH_CHECK(
      stride.size() == 2,
      "It is expected stride equals to 2, but got size ",
      stride.size());

  // 提取各个参数的具体数值
  int64_t kernel_height = kernel_size[0];
  int64_t kernel_width = kernel_size[1];
  int64_t dilation_height = dilation[0];
  int64_t dilation_width = dilation[1];
  int64_t pad_height = padding[0];
  int64_t pad_width = padding[1];
  int64_t stride_height = stride[0];
  int64_t stride_width = stride[1];

  // 进行输入 Tensor 的形状检查，确保可以进行 im2col 操作
  im2col_shape_check(
      input_,
      Tensor(),
      kernel_height,
      kernel_width,
      dilation_height,
      dilation_width,
      pad_height,
      pad_width,
      stride_height,
      stride_width);

  // 将输入 Tensor 进行内存连续化操作，以便进行后续计算
  Tensor input = input_.contiguous();

  // 判断是否为批量输入，如果输入 Tensor 的维度为 3，则为非批量输入
  bool batched_input = true;
  if (input.dim() == 3) {
    batched_input = false;
  }
    // 将输入张量视图变换为 {1, input.size(0), input.size(1), input.size(2)} 的形状
    input = input.view({1, input.size(0), input.size(1), input.size(2)});
  }

  // 获取输入张量的维度信息
  int64_t batch_size = input.size(0);          // 获取批量大小
  int64_t n_input_plane = input.size(1);       // 获取输入通道数
  int64_t input_height = input.size(2);        // 获取输入高度
  int64_t input_width = input.size(3);         // 获取输入宽度

  // 计算输出张量的尺寸
  int64_t output_height = (input_height + 2 * pad_height -
                           (dilation_height * (kernel_height - 1) + 1)) /
          stride_height + 1;                    // 计算输出高度
  int64_t output_width = (input_width + 2 * pad_width -
                          (dilation_width * (kernel_width - 1) + 1)) /
          stride_width + 1;                     // 计算输出宽度
  int64_t n_output_plane = n_input_plane * kernel_width * kernel_height;  // 计算输出通道数
  int64_t output_length = output_height * output_width;                   // 计算输出长度

  // 调整输出张量的尺寸
  output.resize_({batch_size, n_output_plane, output_length});

  // 根据输入张量的数据类型和函数名称进行分发计算
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kBFloat16, kHalf,
      input.scalar_type(), "im2col_out_cpu", [&] {
        Tensor input_n;    // 定义单个样本的输入张量
        Tensor output_n;   // 定义单个样本的输出张量

        // 遍历每个样本
        for (const auto elt : c10::irange(batch_size)) {
          input_n = input.select(0, elt);  // 选择当前样本的输入张量
          output_n = output.select(0, elt);  // 选择当前样本的输出张量

          // 调用 im2col 函数将输入张量转换为矩阵形式
          im2col<scalar_t>(
              input_n.const_data_ptr<scalar_t>(),
              n_input_plane,
              input_height,
              input_width,
              output_height,
              output_width,
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

        // 如果输入张量未经批量处理，则调整输出张量的尺寸
        if (!batched_input) {
          output.resize_({n_output_plane, output_length});
        }
      });
} // 结束当前命名空间

} // 结束整个命名空间

namespace at::native {

// 将输入张量转换为im2col格式的输出张量，并存储在给定的输出张量中
Tensor& im2col_out_cpu(const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride,
    Tensor& output) {
  // 调用模板函数，将输入张量转换为im2col格式的输出张量
  im2col_out_cpu_template(
      output, input, kernel_size, dilation, padding, stride);
  // 返回结果输出张量的引用
  return output;
}

// 将输入张量转换为im2col格式的输出张量，并返回结果张量
Tensor im2col_cpu(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride) {
  // 创建一个和输入张量相同形状的空张量作为输出
  Tensor output = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

  // 调用模板函数，将输入张量转换为im2col格式的输出张量
  im2col_out_cpu_template(
      output, input, kernel_size, dilation, padding, stride);
  // 返回结果张量
  return output;
}

} // 结束命名空间 at::native
```