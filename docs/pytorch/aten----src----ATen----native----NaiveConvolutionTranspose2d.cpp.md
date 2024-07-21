# `.\pytorch\aten\src\ATen\native\NaiveConvolutionTranspose2d.cpp`

```
// 定`
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS // 定义宏，禁止在方法中使用断言操作符
#include <ATen/Dispatch.h> // 包含 ATen 库的 Dispatch 头文件，用于分发函数调用
#include <ATen/TensorMeta.h> // 包含 ATen 库的 TensorMeta 头文件，定义张量元数据
#include <ATen/TensorUtils.h> // 包含 ATen 库的 TensorUtils 头文件，提供张量工具函数

#include <ATen/core/Tensor.h> // 包含 ATen 库的核心 Tensor 头文件，定义张量数据结构
#include <ATen/native/ConvUtils.h> // 包含 ATen 库的 ConvUtils 头文件，提供卷积操作的工具函数
#include <ATen/native/CPUBlas.h> // 包含 ATen 库的 CPUBlas 头文件，提供 CPU 版本的 BLAS 操作
#include <ATen/native/im2col.h> // 包含 ATen 库的 im2col 头文件，提供 im2col 操作的实现

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h> // 包含 ATen 库的 Functions 头文件，提供标准函数接口
#include <ATen/NativeFunctions.h> // 包含 ATen 库的 NativeFunctions 头文件，定义原生函数接口
#else
#include <ATen/ops/empty.h> // 包含 ATen 库的 empty 操作头文件，定义 empty 张量的创建
#include <ATen/ops/ones.h> // 包含 ATen 库的 ones 操作头文件，定义全1张量的创建
#include <ATen/ops/slow_conv_transpose2d_native.h> // 包含 ATen 库的 slow_conv_transpose2d_native 操作头文件，定义反卷积的原生操作
#include <ATen/ops/sum.h> // 包含 ATen 库的 sum 操作头文件，定义求和操作
#include <ATen/ops/zeros.h> // 包含 ATen 库的 zeros 操作头文件，定义全0张量的创建
#endif

#include <c10/core/TensorOptions.h> // 包含 c10 库的 TensorOptions 头文件，定义张量选项
#include <c10/util/irange.h> // 包含 c10 库的 irange 头文件，提供生成整数范围的工具

namespace at { // 开始 at 命名空间
namespace { // 匿名命名空间，内部的内容在其他文件中不可见

// 定义一个静态内联函数，执行反卷积运算的形状检查
static inline void slow_conv_transpose2d_shape_check(
    const Tensor& input, // 输入张量
    const Tensor& grad_output, // 反向传播时的梯度输出张量
    const Tensor& weight, // 卷积核权重张量
    const Tensor& bias, // 偏置张量
    int kernel_height, // 卷积核高度
    int kernel_width, // 卷积核宽度
    int stride_height, // 高度步幅
    int stride_width, // 宽度步幅
    int pad_height, // 高度填充
    int pad_width, // 宽度填充
    int output_padding_height, // 输出填充高度
    int output_padding_width, // 输出填充宽度
    int dilation_height, // 高度膨胀
    int dilation_width, // 宽度膨胀
    bool weight_nullable) { // 权重是否可为空

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
  // 检查膨胀系数是否大于零
  TORCH_CHECK(
      dilation_width > 0 && dilation_height > 0,
      "dilation should be greater than zero, but got dilation_height: ",
      dilation_height,
      ", dilation_width: ",
      dilation_width);
  // 检查输出填充是否小于步幅或膨胀值
  TORCH_CHECK(
      (output_padding_width < stride_width ||
       output_padding_width < dilation_width) &&
          (output_padding_height < stride_height ||
           output_padding_height < dilation_height),
      "output padding must be smaller than either stride or dilation, but got output_padding_height: ",
      output_padding_height,
      " output_padding_width: ",
      output_padding_width,
      " stride_height: ",
      stride_height,
      " stride_width: ",
      stride_width,
      " dilation_height: ",
      dilation_height,
      " dilation_width: ",
      dilation_width);

  // 如果权重张量已定义，则进一步检查其维度和非零性
  if (weight.defined()) {
    TORCH_CHECK(
        weight.numel() != 0 && (weight.dim() == 2 || weight.dim() == 4),
        "non-empty 2D or 4D weight tensor expected, but got: ",
        weight.sizes());
    //nullable");
  }

  // 输入张量的维度数
  int ndim = input.dim();
  int dimf = 0;
  int dimh = 1;
  int dimw = 2;

  // 如果输入张量是 4 维的
  if (ndim == 4) {
    dimf++;   // 维度 f 自增
    dimh++;   // 维度 h 自增


这里只展示了部分代码和对应的注释，代码较长，因此需要根据代码的结构和功能逐行添加注释。
    dimw++;



// 增加 dimw 的值，即使其自增 1
dimw++;



  TORCH_CHECK(
      input.numel() != 0 && (ndim == 3 || ndim == 4),
      "non-empty 3D or 4D input tensor expected but got a tensor with size ",
      input.sizes());



// 使用 Torch 的检查函数确保输入张量满足条件：非空且为 3D 或 4D 张量
TORCH_CHECK(
    input.numel() != 0 && (ndim == 3 || ndim == 4),
    "non-empty 3D or 4D input tensor expected but got a tensor with size ",
    input.sizes());



  int64_t input_height = input.size(dimh);
  int64_t input_width = input.size(dimw);
  int64_t output_height = (input_height - 1) * stride_height - 2 * pad_height +
      (dilation_height * (kernel_height - 1) + 1) + output_padding_height;
  int64_t output_width = (input_width - 1) * stride_width - 2 * pad_width +
      (dilation_width * (kernel_width - 1) + 1) + output_padding_width;



// 计算输入和输出的高度和宽度，基于给定的卷积参数和填充值
int64_t input_height = input.size(dimh);
int64_t input_width = input.size(dimw);
int64_t output_height = (input_height - 1) * stride_height - 2 * pad_height +
    (dilation_height * (kernel_height - 1) + 1) + output_padding_height;
int64_t output_width = (input_width - 1) * stride_width - 2 * pad_width +
    (dilation_width * (kernel_width - 1) + 1) + output_padding_width;



  if (output_width < 1 || output_height < 1) {
    AT_ERROR(
        "Given input size per channel: (",
        input_height,
        " x ",
        input_width,
        "). "
        "Calculated output size per channel: (",
        output_height,
        " x ",
        output_width,
        "). Output size is too small");
  }



// 如果计算得到的输出宽度或高度小于 1，则抛出错误
if (output_width < 1 || output_height < 1) {
  AT_ERROR(
      "Given input size per channel: (",
      input_height,
      " x ",
      input_width,
      "). "
      "Calculated output size per channel: (",
      output_height,
      " x ",
      output_width,
      "). Output size is too small");
}



  if (weight.defined()) {
    int64_t n_input_plane = weight.size(0);
    check_dim_size(input, ndim, dimf, n_input_plane);
  }



// 如果权重张量已定义，则检查输入张量在指定维度上的尺寸是否与权重张量匹配
if (weight.defined()) {
  int64_t n_input_plane = weight.size(0);
  check_dim_size(input, ndim, dimf, n_input_plane);
}



  if (grad_output.defined()) {
    if (weight.defined()) {
      int64_t n_output_plane = weight.size(1);
      check_dim_size(grad_output, ndim, dimf, n_output_plane);
    } else if (bias.defined()) {
      int64_t n_output_plane = bias.size(0);
      check_dim_size(grad_output, ndim, dimf, n_output_plane);
    }
    check_dim_size(grad_output, ndim, dimh, output_height);
    check_dim_size(grad_output, ndim, dimw, output_width);
  }



// 如果梯度输出张量已定义，则根据情况检查其在不同维度上的尺寸是否匹配
if (grad_output.defined()) {
  // 如果权重已定义，检查梯度输出张量在特定维度上的尺寸是否与权重张量匹配
  if (weight.defined()) {
    int64_t n_output_plane = weight.size(1);
    check_dim_size(grad_output, ndim, dimf, n_output_plane);
  } else if (bias.defined()) {
    // 如果偏置已定义，检查梯度输出张量在特定维度上的尺寸是否与偏置张量匹配
    int64_t n_output_plane = bias.size(0);
    check_dim_size(grad_output, ndim, dimf, n_output_plane);
  }
  // 检查梯度输出张量在高度和宽度维度上的尺寸是否与计算得到的输出尺寸匹配
  check_dim_size(grad_output, ndim, dimh, output_height);
  check_dim_size(grad_output, ndim, dimw, output_width);
}
} // 结束 meta 命名空间

namespace meta {
// 定义名为 slow_conv_transpose2d 的 Torch 元函数，实现转置卷积操作
TORCH_META_FUNC(slow_conv_transpose2d)
(const Tensor& input,
 const Tensor& weight,
 IntArrayRef kernel_size,
 OptionalTensorRef bias_opt,
 IntArrayRef stride,
 IntArrayRef padding,
 IntArrayRef output_padding,
 IntArrayRef dilation) {
  // 检查 kernel_size 是否为二维
  TORCH_CHECK(
      kernel_size.size() == 2,
      "期望 kernel_size 为 2，但其尺寸为 ",
      kernel_size.size());

  // 检查 dilation 是否为二维
  TORCH_CHECK(
      dilation.size() == 2,
      "期望 dilation 为 2，但其尺寸为 ",
      dilation.size());

  // 检查 padding 是否为二维
  TORCH_CHECK(
      padding.size() == 2,
      "期望 padding 为 2，但其尺寸为 ",
      padding.size());

  // 检查 stride 是否为二维
  TORCH_CHECK(
      stride.size() == 2,
      "期望 stride 为 2，但其尺寸为 ",
      stride.size());

  // 检查 output_padding 是否为二维
  TORCH_CHECK(
      output_padding.size() == 2,
      "期望 output_padding 为 2，但其尺寸为 ",
      output_padding.size());

  // 提取各个参数的具体值
  int64_t kernel_height = kernel_size[0];
  int64_t kernel_width = kernel_size[1];
  int64_t dilation_height = dilation[0];
  int64_t dilation_width = dilation[1];
  int64_t pad_height = padding[0];
  int64_t pad_width = padding[1];
  int64_t stride_height = stride[0];
  int64_t stride_width = stride[1];
  int64_t output_padding_height = output_padding[0];
  int64_t output_padding_width = output_padding[1];

  // 调用 slow_conv_transpose2d_shape_check 函数检查参数形状
  slow_conv_transpose2d_shape_check(
      input,
      Tensor(),
      weight,
      bias_opt.getTensorRef(),
      kernel_height,
      kernel_width,
      stride_height,
      stride_width,
      pad_height,
      pad_width,
      output_padding_height,
      output_padding_width,
      dilation_height,
      dilation_width,
      false);

  // 提取权重张量的输出平面数量
  int n_output_plane = weight.size(1);

  // 检查是否使用通道优先的内存布局
  bool use_channels_last = native::thnn_conv_use_channels_last(input, weight);
  auto memory_format = use_channels_last ? at::MemoryFormat::ChannelsLast : at::MemoryFormat::Contiguous;

  // 使输入张量连续化，根据内存格式
  Tensor input_ = input.contiguous(memory_format);

  // 如果输入张量的维度为 3，添加一个维度到批处理大小为 1
  if (input_.dim() == 3) {
    input_.resize_({1, input_.size(0), input_.size(1), input_.size(2)});
  }

  // 计算输入、输出和输出的高度和宽度
  int64_t input_height = input_.size(2);
  int64_t input_width = input_.size(3);
  int64_t output_height = (input_height - 1) * stride_height - 2 * pad_height +
      (dilation_height * (kernel_height - 1) + 1) + output_padding_height;
  int64_t output_width = (input_width - 1) * stride_width - 2 * pad_width +
      (dilation_width * (kernel_width - 1) + 1) + output_padding_width;

  // 批处理大小为输入张量的第一个维度
  int64_t batch_size = input_.size(0);

  // 设置输出张量的选项
  TensorOptions options(input.options());

  // 设置输出张量的原始步幅
  set_output_raw_strided(
      0,
      {batch_size, n_output_plane, output_height, output_width},
      {},
      options.memory_format(memory_format));
}
} // 结束 meta 命名空间

namespace native {

namespace {
// 定义 slow_conv_transpose2d_out_cpu_template 函数模板，处理 CPU 上的转置卷积
void slow_conv_transpose2d_out_cpu_template(
    const Tensor& output,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size,
    const Tensor& bias,
    IntArrayRef stride,


这样的注释保证了每行代码的功能和作用都得到了清晰的解释，帮助他人理解代码的每个细节和意图。
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef dilation) {
  int64_t kernel_height = kernel_size[0];  // 从 kernel_size 中获取卷积核的高度
  int64_t kernel_width = kernel_size[1];   // 从 kernel_size 中获取卷积核的宽度
  int64_t dilation_height = dilation[0];   // 从 dilation 中获取扩展的高度
  int64_t dilation_width = dilation[1];    // 从 dilation 中获取扩展的宽度
  int64_t pad_height = padding[0];         // 从 padding 中获取垫高度
  int64_t pad_width = padding[1];          // 从 padding 中获取垫宽度
  int64_t stride_height = stride[0];       // 从 stride 中获取步长的高度
  int64_t stride_width = stride[1];        // 从 stride 中获取步长的宽度
  int64_t output_padding_height = output_padding[0];  // 从 output_padding 中获取输出的高度填充
  int64_t output_padding_width = output_padding[1];    // 从 output_padding 中获取输出的宽度填充

  int n_input_plane = weight.size(0);      // 获取权重张量的输入平面数
  int n_output_plane = weight.size(1);     // 获取权重张量的输出平面数

  bool use_channels_last = thnn_conv_use_channels_last(input, weight);  // 检查是否使用通道最后的内存格式
  auto memory_format = use_channels_last ? at::MemoryFormat::ChannelsLast : at::MemoryFormat::Contiguous;  // 根据是否使用通道最后格式选择内存格式

  Tensor input_ = input.contiguous(memory_format);  // 使用选择的内存格式对输入张量进行连续化
  Tensor weight_ = weight.contiguous(memory_format);  // 使用选择的内存格式对权重张量进行连续化
  Tensor bias_ = bias.defined() ? bias.contiguous() : Tensor();  // 如果定义了偏置，则对其进行连续化，否则创建空张量

  bool is_batch = false;
  if (input_.dim() == 3) {
    // 强制批处理
    is_batch = true;  // 标记为批处理模式
    input_.resize_({1, input.size(0), input.size(1), input.size(2)});  // 将输入张量大小调整为批处理模式
  }

  int64_t input_height = input_.size(2);    // 获取调整后输入张量的高度
  int64_t input_width = input_.size(3);     // 获取调整后输入张量的宽度
  int64_t output_height = (input_height - 1) * stride_height - 2 * pad_height +
      (dilation_height * (kernel_height - 1) + 1) + output_padding_height;  // 计算输出的高度
  int64_t output_width = (input_width - 1) * stride_width - 2 * pad_width +
      (dilation_width * (kernel_width - 1) + 1) + output_padding_width;  // 计算输出的宽度

  // 批处理大小 + 输入平面数
  int64_t batch_size = input_.size(0);  // 获取批处理大小

  // 创建临时列
  Tensor columns = at::empty({0}, input.options());  // 使用与输入相同的选项创建空张量
  if (use_channels_last) {
    columns.resize_({batch_size, input_height * input_width, kernel_height * kernel_width * n_output_plane});  // 如果使用通道最后格式，则调整列张量的大小
  } else {
    columns.resize_({batch_size, n_output_plane * kernel_height * kernel_width, input_height * input_width});  // 否则调整列张量的大小
  }
  columns.zero_();  // 将列张量初始化为零

  // 如果是 COW，需要实例化，因为无法在 parallel_for 中执行此操作
  output.mutable_data_ptr();  // 获取输出张量的可变数据指针

  AT_DISPATCH_FLOATING_TYPES_AND3(at::ScalarType::Long, at::ScalarType::BFloat16,
      at::ScalarType::Half, input.scalar_type(), "slow_conv_transpose2d_out_cpu", [&] {
    // 使用并行处理，对每个批次中的元素进行操作
    at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
      // 对于每个批次中的元素 elt，执行以下操作：
      for (const auto elt : c10::irange(begin, end)) {
        // 根据批次中的索引 elt 选择输入张量的对应部分
        Tensor input_n = input_.select(0, elt);
        // 根据批次中的索引 elt 选择输出张量的对应部分
        Tensor output_n = output.select(0, elt);
        // 根据批次中的索引 elt 选择列张量的对应部分
        Tensor columns_n = columns.select(0, elt);

        // 如果使用 channels_last 布局
        if (use_channels_last) {
          // 计算矩阵乘法所需的参数
          int64_t m = kernel_height * kernel_width * n_output_plane;
          int64_t n = input_height * input_width;
          int64_t k = n_input_plane;

          // 利用 CPU 实现矩阵乘法
          // 注意：这里的 gemm 函数来自于 cpublas 库，执行不同的矩阵乘法操作
          cpublas::gemm(
              TransposeType::NoTranspose,         // 不转置输入矩阵 A
              TransposeType::NoTranspose,         // 不转置输入矩阵 B
              m,                                  // 矩阵 C 的行数
              n,                                  // 矩阵 C 的列数
              k,                                  // 矩阵 A 的列数和矩阵 B 的行数
              static_cast<scalar_t>(1),           // 系数 alpha
              weight_.const_data_ptr<scalar_t>(), // 矩阵 A 的数据指针
              m,                                  // 矩阵 A 的列数
              input_n.const_data_ptr<scalar_t>(), // 矩阵 B 的数据指针
              k,                                  // 矩阵 B 的列数
              static_cast<scalar_t>(0),           // 系数 beta
              columns_n.mutable_data_ptr<scalar_t>(), // 矩阵 C 的数据指针
              m);                                 // 矩阵 C 的列数
        } else {
          // 计算矩阵乘法所需的参数
          int64_t m = input_height * input_width;
          int64_t n = n_output_plane * kernel_height * kernel_width;
          int64_t k = n_input_plane;

          // 利用 CPU 实现矩阵乘法，但是输入矩阵 B 被转置
          cpublas::gemm(
              TransposeType::NoTranspose,         // 不转置输入矩阵 A
              TransposeType::Transpose,           // 转置输入矩阵 B
              m,                                  // 矩阵 C 的行数
              n,                                  // 矩阵 C 的列数
              k,                                  // 矩阵 A 的列数和矩阵 B 转置后的行数
              static_cast<scalar_t>(1),           // 系数 alpha
              input_n.const_data_ptr<scalar_t>(), // 矩阵 A 的数据指针
              m,                                  // 矩阵 A 的列数
              weight_.const_data_ptr<scalar_t>(), // 矩阵 B 的数据指针
              n,                                  // 矩阵 B 的列数（转置后的行数）
              static_cast<scalar_t>(0),           // 系数 beta
              columns_n.mutable_data_ptr<scalar_t>(), // 矩阵 C 的数据指针
              m);                                 // 矩阵 C 的列数
        }

        // 将列张量重新转换回输入张量的形式
        col2im<scalar_t>(
            columns_n.const_data_ptr<scalar_t>(), // 列张量的数据指针
            n_output_plane,                      // 输出平面的数量
            output_height,                       // 输出高度
            output_width,                        // 输出宽度
            input_height,                        // 输入高度
            input_width,                         // 输入宽度
            kernel_height,                       // 卷积核高度
            kernel_width,                        // 卷积核宽度
            pad_height,                          // 填充高度
            pad_width,                           // 填充宽度
            stride_height,                       // 步幅高度
            stride_width,                        // 步幅宽度
            dilation_height,                     // 膨胀高度
            dilation_width,                      // 膨胀宽度
            output_n.data_ptr<scalar_t>(),       // 输出张量的数据指针
            use_channels_last);                  // 是否使用 channels_last 布局
      }
    });
  });

  // 如果定义了偏置张量，将其加到输出上
  if (bias.defined()) {
    output.add_(bias_.reshape({-1, 1, 1}));
  }

  // 调整输出张量的大小
  // 如果是批处理模式，将输出的大小调整为 {n_output_plane, output_height, output_width}
  if (is_batch) {
    output.resize_({n_output_plane, output_height, output_width});
  }
  }

static void slow_conv_transpose2d_backward_out_cpu_template(
    const Tensor& input_,  // 输入张量，表示前向传播的输入
    const Tensor& grad_output_,  // 梯度张量，表示前向传播的输出的梯度
    Tensor& grad_input,  // 梯度张量，用于存储反向传播计算得到的输入的梯度
    const Tensor& weight_,  // 权重张量，卷积核的权重
    IntArrayRef kernel_size,  // 整数数组引用，卷积核的尺寸
    IntArrayRef stride,  // 整数数组引用，卷积核的步长
    IntArrayRef padding,  // 整数数组引用，卷积操作的填充
    IntArrayRef output_padding,  // 整数数组引用，转置卷积操作的输出填充
    IntArrayRef dilation) {  // 整数数组引用，卷积核的扩张率
  TORCH_CHECK(
      kernel_size.size() == 2,  // 检查卷积核尺寸是否为2
      "It is expected kernel_size equals to 2, but got size ",
      kernel_size.size());

  TORCH_CHECK(
      dilation.size() == 2,  // 检查卷积核扩张率是否为2
      "It is expected dilation equals to 2, but got size ",
      dilation.size());

  TORCH_CHECK(
      padding.size() == 2,  // 检查填充大小是否为2
      "It is expected padding equals to 2, but got size ",
      padding.size());

  TORCH_CHECK(
      stride.size() == 2,  // 检查步长是否为2
      "It is expected stride equals to 2, but got size ",
      stride.size());

  TORCH_CHECK(
      output_padding.size() == 2,  // 检查输出填充大小是否为2
      "It is expected stride equals to 2, but got size ",
      output_padding.size());

  int64_t kernel_height = kernel_size[0];  // 获取卷积核的高度
  int64_t kernel_width = kernel_size[1];  // 获取卷积核的宽度
  int64_t dilation_height = dilation[0];  // 获取卷积核的高度扩张率
  int64_t dilation_width = dilation[1];  // 获取卷积核的宽度扩张率
  int64_t pad_height = padding[0];  // 获取填充的高度
  int64_t pad_width = padding[1];  // 获取填充的宽度
  int64_t stride_height = stride[0];  // 获取步长的高度
  int64_t stride_width = stride[1];  // 获取步长的宽度
  int64_t output_padding_height = output_padding[0];  // 获取输出填充的高度
  int64_t output_padding_width = output_padding[1];  // 获取输出填充的宽度

  int64_t n_input_plane = weight_.size(0);  // 获取输入平面数
  int64_t n_output_plane = weight_.size(1);  // 获取输出平面数

  bool use_channels_last = thnn_conv_use_channels_last(input_, weight_);  // 检查是否使用通道最后的内存布局
  auto memory_format = use_channels_last ? at::MemoryFormat::ChannelsLast : at::MemoryFormat::Contiguous;  // 根据使用情况选择内存布局格式

  slow_conv_transpose2d_shape_check(
      input_,
      grad_output_,
      weight_,
      Tensor(),
      kernel_height,
      kernel_width,
      stride_height,
      stride_width,
      pad_height,
      pad_width,
      output_padding_height,
      output_padding_width,
      dilation_height,
      dilation_width,
      false);  // 执行反向传播形状检查

  Tensor input = input_.contiguous(memory_format);  // 获取连续的输入张量
  Tensor grad_output = grad_output_.contiguous(memory_format);  // 获取连续的梯度张量
  Tensor weight = weight_.contiguous(memory_format);  // 获取连续的权重张量

  bool is_batch = false;  // 是否为批处理
  if (input.dim() == 3) {
    // 强制批处理
    is_batch = true;
    input.resize_({1, input.size(0), input.size(1), input.size(2)});  // 调整输入张量的大小为批处理形式
  grad_output.resize_(
      {1, grad_output.size(0), grad_output.size(1), grad_output.size(2)});



// 调整梯度输出的大小，为后续计算做准备，将维度重新排列为 {1, grad_output.size(0), grad_output.size(1), grad_output.size(2)}
}



int64_t input_width = input.size(3);
int64_t input_height = input.size(2);
int64_t output_height = (input_height - 1) * stride_height - 2 * pad_height +
    (dilation_height * (kernel_height - 1) + 1) + output_padding_height;
int64_t output_width = (input_width - 1) * stride_width - 2 * pad_width +
    (dilation_width * (kernel_width - 1) + 1) + output_padding_width;



// 计算输出的高度和宽度，根据输入的尺寸、步幅、填充和扩张率进行计算得出
int64_t input_width = input.size(3);  // 输入张量的宽度
int64_t input_height = input.size(2); // 输入张量的高度
// 计算输出的高度，考虑了步幅、填充、扩张率以及输出填充
int64_t output_height = (input_height - 1) * stride_height - 2 * pad_height +
    (dilation_height * (kernel_height - 1) + 1) + output_padding_height;
// 计算输出的宽度，考虑了步幅、填充、扩张率以及输出填充
int64_t output_width = (input_width - 1) * stride_width - 2 * pad_width +
    (dilation_width * (kernel_width - 1) + 1) + output_padding_width;



// Batch size + input planes
int64_t batch_size = input.size(0);



// 计算批量大小（batch size），即输入张量的第一个维度大小，表示批量中包含的样本数
int64_t batch_size = input.size(0);



// Resize output
grad_input.resize_({batch_size, n_input_plane, input_height, input_width}, memory_format);
grad_input.zero_();



// 调整梯度输入的大小，将其重新设置为指定的维度 {batch_size, n_input_plane, input_height, input_width}，使用给定的内存格式
grad_input.zero_();
// 将梯度输入张量的所有元素清零，以便后续的计算



// Create temporary columns
bool need_columns = (kernel_height != 1 || kernel_width != 1 || stride_height != 1 ||
    stride_width != 1 || pad_height != 0 || pad_width != 0 ||
    dilation_height != 1 || dilation_width != 1);

Tensor grad_columns = at::empty({0}, input.options());
if (need_columns) {
  if (use_channels_last) {
    grad_columns.resize_({input_height * input_width, kernel_height * kernel_width * n_output_plane});
  } else {
    grad_columns.resize_({n_output_plane * kernel_height * kernel_width, input_height * input_width});
  }



// 创建临时列向量（temporary columns），用于存储卷积计算中的中间结果
bool need_columns = (kernel_height != 1 || kernel_width != 1 || stride_height != 1 ||
    stride_width != 1 || pad_height != 0 || pad_width != 0 ||
    dilation_height != 1 || dilation_width != 1);

// 根据是否需要临时列向量，决定是否创建并调整大小
Tensor grad_columns = at::empty({0}, input.options());
if (need_columns) {
  // 根据是否使用通道最后的存储顺序，调整临时列向量的大小
  if (use_channels_last) {
    grad_columns.resize_({input_height * input_width, kernel_height * kernel_width * n_output_plane});
  } else {
    grad_columns.resize_({n_output_plane * kernel_height * kernel_width, input_height * input_width});
  }
}
    }
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::BFloat16, at::ScalarType::Half,
      grad_output.scalar_type(), "slow_conv_transpose2d_backward_out_cpu", [&] {
        // Helpers

        // 定义梯度输入和梯度输出张量
        Tensor grad_input_n = Tensor();
        Tensor grad_output_n = Tensor();

        // 对于每个批次中的每个元素，执行以下操作：
        for (const auto elt : c10::irange(batch_size)) {

          // 矩阵乘法的每个样本：
          grad_input_n = grad_input.select(0, elt);  // 选择当前批次元素的梯度输入
          grad_output_n = grad_output.select(0, elt);  // 选择当前批次元素的梯度输出

          if (need_columns) {
            // 提取列：
            im2col<scalar_t>(
                  grad_output_n.const_data_ptr<scalar_t>(),
                  n_output_plane,
                  output_height,
                  output_width,
                  input_height,
                  input_width,
                  kernel_height,
                  kernel_width,
                  pad_height,
                  pad_width,
                  stride_height,
                  stride_width,
                  dilation_height,
                  dilation_width,
                  grad_columns.data_ptr<scalar_t>(),
                  use_channels_last);
          }

          auto gemm_in_ptr = need_columns ? grad_columns.const_data_ptr<scalar_t>()
              : grad_output_n.const_data_ptr<scalar_t>();

          if (use_channels_last) {
            int64_t m = n_input_plane;
            int64_t n = input_height * input_width;
            int64_t k = n_output_plane * kernel_height * kernel_width;

            // 列主序矩阵乘法
            cpublas::gemm(
                TransposeType::Transpose,
                TransposeType::NoTranspose,
                m,
                n,
                k,
                static_cast<scalar_t>(1),
                weight.const_data_ptr<scalar_t>(),
                k,
                gemm_in_ptr,
                k,
                static_cast<scalar_t>(0),
                grad_input_n.mutable_data_ptr<scalar_t>(),
                m);

          } else {
            int64_t m = input_height * input_width;
            int64_t n = n_input_plane;
            int64_t k = n_output_plane * kernel_height * kernel_width;

            // 列主序矩阵乘法
            cpublas::gemm(
                TransposeType::NoTranspose,
                TransposeType::NoTranspose,
                m,
                n,
                k,
                static_cast<scalar_t>(1),
                gemm_in_ptr,
                m,
                weight.const_data_ptr<scalar_t>(),
                k,
                static_cast<scalar_t>(0),
                grad_input_n.mutable_data_ptr<scalar_t>(),
                m);
          }
        }

        // 调整输出大小
        if (is_batch) {
          grad_input.resize_({n_input_plane, input_height, input_width});
        }
      });
  // 慢速反卷积层的梯度参数累积计算，CPU 版本

  // 检查卷积核大小是否为2
  TORCH_CHECK(
      kernel_size.size() == 2,
      "期望 kernel_size 等于 2，但得到大小为 ",
      kernel_size.size());

  // 检查扩展大小是否为2
  TORCH_CHECK(
      dilation.size() == 2,
      "期望 dilation 等于 2，但得到大小为 ",
      dilation.size());

  // 检查填充大小是否为2
  TORCH_CHECK(
      padding.size() == 2,
      "期望 padding 等于 2，但得到大小为 ",
      padding.size());

  // 检查步长大小是否为2
  TORCH_CHECK(
      stride.size() == 2,
      "期望 stride 等于 2，但得到大小为 ",
      stride.size());

  // 检查输出填充大小是否为2
  TORCH_CHECK(
      output_padding.size() == 2,
      "期望输出填充等于 2，但得到大小为 ",
      output_padding.size());

  // 提取各种参数
  int64_t kernel_height = kernel_size[0];
  int64_t kernel_width = kernel_size[1];
  int64_t dilation_height = dilation[0];
  int64_t dilation_width = dilation[1];
  int64_t pad_height = padding[0];
  int64_t pad_width = padding[1];
  int64_t stride_height = stride[0];
  int64_t stride_width = stride[1];
  int64_t output_padding_height = output_padding[0];
  int64_t output_padding_width = output_padding[1];

  // 检查是否使用通道最后格式
  bool use_channels_last = thnn_conv_use_channels_last(input_, weight_);
  auto memory_format = use_channels_last ? at::MemoryFormat::ChannelsLast : at::MemoryFormat::Contiguous;

  // 执行形状检查
  slow_conv_transpose2d_shape_check(
      input_,
      grad_output_,
      grad_weight,
      grad_bias,
      kernel_height,
      kernel_width,
      stride_height,
      stride_width,
      pad_height,
      pad_width,
      output_padding_height,
      output_padding_width,
      dilation_height,
      dilation_width,
      true);

  // 获取输入和梯度输出的通道数
  int n_input_plane = weight_.size(0);
  int n_output_plane = weight_.size(1);

  // 确保输入和梯度输出张量在指定的内存格式上连续
  Tensor input = input_.contiguous(memory_format);
  Tensor grad_output = grad_output_.contiguous(memory_format);
  TORCH_CHECK(grad_weight.is_contiguous(memory_format), "grad_weight 需要在指定的内存格式上连续");

  // 如果输入维度为3，则扩展为4维
  if (input.dim() == 3) {
    input.resize_({1, input.size(0), input.size(1), input.size(2)});
  // 调整梯度输出的大小，增加额外维度
  grad_output.resize_(
      {1, grad_output.size(0), grad_output.size(1), grad_output.size(2)});
}

// 计算输入张量的宽度和高度
int64_t input_width = input.size(3);
int64_t input_height = input.size(2);

// 计算输出张量的高度和宽度
int64_t output_height = (input_height - 1) * stride_height - 2 * pad_height +
    (dilation_height * (kernel_height - 1) + 1) + output_padding_height;
int64_t output_width = (input_width - 1) * stride_width - 2 * pad_width +
    (dilation_width * (kernel_width - 1) + 1) + output_padding_width;

// 计算批量大小
int64_t batch_size = input.size(0);

// 根据需要判断是否调整临时列的大小
bool need_columns = (kernel_height != 1 || kernel_width != 1 || stride_height != 1 ||
    stride_width != 1 || pad_height != 0 || pad_width != 0 ||
    dilation_height != 1 || dilation_width != 1);

// 创建空的张量 columns 用于存储临时数据
Tensor columns = at::empty({0}, input.options());

// 如果需要临时列
if (need_columns) {
  // 如果使用 channels_last 格式，调整 columns 张量的大小
  if (use_channels_last) {
    columns.resize_({input_height * input_width, kernel_height * kernel_width * n_output_plane});
  } else {
    // 否则，调整 columns 张量的大小
    columns.resize_({n_output_plane * kernel_height * kernel_width, input_height * input_width});
  }
}
    }
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::BFloat16, at::ScalarType::Half,
      input.scalar_type(), "slow_conv_transpose2d_acc_grad_parameters_cpu", [&] {
        // Helpers

        // 定义输入和梯度输出张量
        Tensor input_n = Tensor();
        Tensor grad_output_n = Tensor();

        // 将 scale_ 转换为对应的标量类型
        scalar_t scale = static_cast<scalar_t>(scale_);

        // 对于每个批次中的元素，执行以下操作:
        for (const auto elt : c10::irange(batch_size)) {
          // 矩阵乘法，针对每个输出:
          grad_output_n = grad_output.select(0, elt);

          // 处理权重:
          if (grad_weight.defined()) {
            // 矩阵乘法，针对每个输出:
            input_n = input.select(0, elt);

            if (need_columns) {
              // 提取列:
              im2col<scalar_t>(
                  grad_output_n.const_data_ptr<scalar_t>(),
                  n_output_plane,
                  output_height,
                  output_width,
                  input_height,
                  input_width,
                  kernel_height,
                  kernel_width,
                  pad_height,
                  pad_width,
                  stride_height,
                  stride_width,
                  dilation_height,
                  dilation_width,
                  columns.data_ptr<scalar_t>(),
                  use_channels_last);
            }

            auto gemm_in_ptr = need_columns ? columns.const_data_ptr<scalar_t>()
                : grad_output_n.const_data_ptr<scalar_t>();

            if (use_channels_last) {
              int64_t m = kernel_height * kernel_width * n_output_plane;
              int64_t n = n_input_plane;
              int64_t k = input_height * input_width;

              // 列主序矩阵
              // 使用 CPU BLAS 执行矩阵乘法
              cpublas::gemm(
                  TransposeType::NoTranspose,
                  TransposeType::Transpose,
                  m,
                  n,
                  k,
                  static_cast<scalar_t>(scale),
                  gemm_in_ptr,
                  m,
                  input_n.const_data_ptr<scalar_t>(),
                  n,
                  static_cast<scalar_t>(1),
                  grad_weight.mutable_data_ptr<scalar_t>(),
                  m);
            } else {
              int64_t m = n_output_plane * kernel_height * kernel_width;
              int64_t n = n_input_plane;
              int64_t k = input_height * input_width;

              // 列主序矩阵
              // 使用 CPU BLAS 执行矩阵乘法
              cpublas::gemm(
                  TransposeType::Transpose,
                  TransposeType::NoTranspose,
                  m,
                  n,
                  k,
                  static_cast<scalar_t>(scale),
                  gemm_in_ptr,
                  k,
                  input_n.const_data_ptr<scalar_t>(),
                  k,
                  static_cast<scalar_t>(1),
                  grad_weight.mutable_data_ptr<scalar_t>(),
                  m);
            }
          }
        }
      });
} // namespace

} // namespace

TORCH_IMPL_FUNC(slow_conv_transpose2d_structured_cpu)
(const Tensor& input,
 const Tensor& weight,
 IntArrayRef kernel_size,
 OptionalTensorRef bias_opt,
 IntArrayRef stride,
 IntArrayRef padding,
 IntArrayRef output_padding,
 IntArrayRef dilation,
 const Tensor& output){
  const Tensor& bias = bias_opt.getTensorRef();

  // 调用模板函数，执行慢速二维转置卷积操作，CPU 版本
  slow_conv_transpose2d_out_cpu_template(
      output,
      input,
      weight,
      kernel_size,
      bias,
      stride,
      padding,
      output_padding,
      dilation);
 }

static std::tuple<Tensor, Tensor, Tensor> slow_conv_transpose2d_backward_cpu(
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

  // 根据 output_mask 的设置，创建 grad_input, grad_weight, grad_bias 张量
  if (output_mask[0]) {
    grad_input = at::empty({0}, grad_output.options());
  } else {
    grad_input = Tensor();
  }

  if (output_mask[1]) {
    grad_weight = at::empty({0}, grad_output.options());
  } else {
    grad_weight = Tensor();
  }

  if (output_mask[2]) {
    grad_bias = at::empty({0}, grad_output.options());
  } else {
    grad_bias = Tensor();
  }

  // 如果 grad_input 已定义，调用模板函数计算慢速二维转置卷积的反向传播（CPU 版本）
  if (grad_input.defined()) {
    slow_conv_transpose2d_backward_out_cpu_template(
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

  // 如果 grad_bias 已定义，对 grad_output 沿指定维度求和
  if (grad_bias.defined()) {
    at::sum_out(grad_bias, grad_output, IntArrayRef{0, 2, 3});
  }

  // 如果 grad_weight 已定义，重置大小并清零，然后累积慢速二维转置卷积的梯度（CPU 版本）
  if (grad_weight.defined()) {
    grad_weight.resize_(weight.sizes(), weight.suggest_memory_format());
    grad_weight.zero_();
    slow_conv_transpose2d_acc_grad_parameters_cpu(
        input,
        weight,
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

  // 返回计算得到的梯度：grad_input, grad_weight, grad_bias 的元组
  return std::tuple<Tensor, Tensor, Tensor>(grad_input, grad_weight, grad_bias);
}

// 将慢速二维转置卷积的反向传播函数注册到所有 CPU 分发中心
REGISTER_ALL_CPU_DISPATCH(slow_conv_transpose2d_backward_stub, &slow_conv_transpose2d_backward_cpu);

} // namespace native
} // namespace at
```