# `.\pytorch\aten\src\ATen\native\cudnn\ConvShared.cpp`

```py
// 定义编译时仅使用方法操作符的宏，以便包含正确的头文件
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 包含 ATen 库的头文件，用于张量的上下文管理、几何结构和工具函数
#include <ATen/Context.h>
#include <ATen/TensorGeometry.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAConfig.h>
#include <ATen/cuda/EmptyTensor.h>
#include <ATen/native/ConvUtils.h>

// 如果启用了 cuDNN 支持，则包含相关的头文件和命名空间
#if AT_CUDNN_ENABLED()

// 包含 cuDNN 相关的共享代码头文件
#include <ATen/native/cudnn/ConvShared.h>

// 根据预处理器宏的不同选择是否包含所有操作的头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/cudnn_convolution_add_relu_native.h>
#include <ATen/ops/cudnn_convolution_native.h>
#include <ATen/ops/cudnn_convolution_relu_native.h>
#include <ATen/ops/cudnn_convolution_transpose_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like.h>
#endif

// 注释：cuDNN API 版本说明
//
// ConvPlaceholders.cpp 包含当 cuDNN 未启用时的卷积的占位符实现。这些操作符仅抛出错误，不进行实际计算。
// 这些操作符是使用当前操作符实现的。
//
// cuDNN v7 和 v8 有不同的 API。ConvShared.{cpp, h} 包含 v7 和 v8 共享的代码。
// Conv_v7.cpp 包含使用 cuDNN v7 API 的卷积实现。
// Conv_v8.cpp 包含使用 v8 API 的实现。
//
// 注释：卷积设计说明
//
// cuDNN 的卷积不处理偏置。偏置在外部处理。
//
// 一般策略:
// - cudnn_convolution (Tensor)
//   客户端的入口点
// - cudnn_convolution_forward (TensorArg)
//   入口点，可以在常规卷积和转置卷积之间重用
// - raw_cudnn_convolution_forward_out (Tensor)
//   具有在 Conv_v7.cpp 和 Conv_v8.cpp 中不同实现的函数
//
// 原始 API 直接调用 CuDNN，并且在 cuDNN v7 和 cuDNN v8 上实现不同。
//
// 这些原始函数不应该直接通过 ATen 暴露出去的几个原因:
// - 它接受输出作为参数（这应该是计算出来的！）
// - 它不进行输入检查
// - 它不调整输出大小（假定输出已正确大小）
//
// 参数检查在哪里发生？责任分工如下:
// - 发生在 at::Tensor 中的事情:
//   - TensorArg 分配
// - 发生在 TensorArg 中的事情:
//   - 检查参数（类型、GPU、形状）

// ATen 的命名空间
namespace at {
// native 命名空间，包含本地实现的函数和算法
namespace native {

// ---------------------------------------------------------------------
//
// ConvolutionParams
//
// ---------------------------------------------------------------------
// 重载输出流操作符，用于将 ConvolutionParams 结构体的信息输出到流 out 中
std::ostream& operator<<(std::ostream& out, const ConvolutionParams& params) {
  // 输出 ConvolutionParams 的头部信息
  out << "ConvolutionParams \n"
      // 输出内存格式信息
      << "    memory_format = " << params.memory_format << "\n"
      // 输出数据类型信息，转换为字符串表示
      << "    data_type = " << cudnnTypeToString(params.dataType) << "\n"
      // 输出填充信息
      << "    padding = " << ArrayRef<int>{params.padding} << "\n"
      // 输出步长信息
      << "    stride = " << ArrayRef<int>{params.stride} << "\n"
      // 输出扩展信息
      << "    dilation = " << ArrayRef<int>{params.dilation} << "\n"
      // 输出分组信息
      << "    groups = " << params.groups << "\n"
      // 输出确定性信息，转换为布尔值表示
      << "    deterministic = " << (params.deterministic ? "true" : "false")
      << "\n"
      // 输出是否允许 tf32
      << "    allow_tf32 = " << (params.allow_tf32 ? "true" : "false") << "\n";

  // 返回输出流对象
  return out;
}

// 注意：这不能是一个构造函数，因为这样 ConvolutionParams 就不再是 POD 类型了。
// TODO: 在这里使用 TensorGeometry 而不是整个 Tensor，因为我们实际上不需要整个 Tensor。（但我们可以随时传递 grad_input/grad_output，所以这并不是很紧急）
// 设置卷积参数
void setConvolutionParams(
    ConvolutionParams* params,          // 卷积参数指针
    const at::Tensor& input,            // 输入 Tensor
    const at::Tensor& weight,           // 权重 Tensor
    IntArrayRef padding,                // 填充数组
    IntArrayRef stride,                 // 步长数组
    IntArrayRef dilation,               // 扩展数组
    int64_t groups,                     // 分组数
    bool deterministic,                 // 是否确定性
    bool allow_tf32,                    // 是否允许 tf32
    at::MemoryFormat memory_format) {   // 内存格式

  // 获取输入 Tensor 的 CuDNN 数据类型
  cudnnDataType_t dataType = getCudnnDataType(input);
  // 将 params 内存清零
  memset(params, 0, sizeof(ConvolutionParams));
  // 设置设备 ID 为当前 CUDA 设备 ID
  params->device_id = at::cuda::current_device();
  // 设置数据类型
  params->dataType = dataType;
  // 断言：权重 Tensor 的维度应与输入 Tensor 的维度相同
  params->input_dim = input.dim();
  // 设置内存格式
  params->memory_format = memory_format;
  // 遍历设置输入 Tensor 和权重 Tensor 的尺寸信息
  for (int i = 0; i != params->input_dim; ++i) {
    params->input_size[i] = (int)input.sizes()[i];
    params->weight_size[i] = (int)weight.sizes()[i];
  }
  // 断言：填充、步长和扩展的数组大小应相等
  // 遍历设置填充、步长和扩展信息
  for (size_t i = 0; i != padding.size(); ++i) {
    params->padding[i] = padding[i];
    params->stride[i] = stride[i];
    params->dilation[i] = dilation[i];
  }
  // 一般情况下，对于旧版 CuDNN，我们不应该按分组进行参数化，但实际上这并不值得特别的努力。
  // 设置分组数
  params->groups = groups;
  // 设置是否确定性
  params->deterministic = deterministic;
  // 设置是否允许 tf32
  params->allow_tf32 = allow_tf32;
}

// 根据 ConvolutionParams 生成可重现的参数信息字符串
std::string repro_from_args(const ConvolutionParams& params) {
  // 匿名函数，用于返回布尔值的字符串表示
  auto pybool = [](bool b) -> const char* { return b ? "True" : "False"; };
  // 部分数据类型字符串
  std::string partial_dtype;
  // 根据数据类型设置部分数据类型字符串
  switch (params.dataType) {
    case CUDNN_DATA_FLOAT:
      partial_dtype = "float";
      break;
    case CUDNN_DATA_DOUBLE:
      partial_dtype = "double";
      break;
    case CUDNN_DATA_HALF:
      partial_dtype = "half";
      break;
  // 根据 params.error 选择部分数据类型
  switch (params.error) {
    // 默认情况下选择不支持的类型
    default:
      partial_dtype = "unsupported";
  }
  // 创建完整的数据类型字符串，以 torch. 开头
  const std::string full_dtype = "torch." + partial_dtype;
  // 输出通道数为权重张量的第一个维度大小
  const int out_channels = params.weight_size[0];
  // 输入通道数为权重张量的第二个维度大小乘以分组数
  const int in_channels = params.weight_size[1] * params.groups;
  // 输入数据的维度大小
  const size_t dim = params.input_dim;
  // 如果维度为4，则通道顺序为 channels_last，否则为 channels_last_3d
  const std::string channels_last_xd =
      dim == 4 ? "channels_last" : "channels_last_3d";
  // 如果内存格式为 ChannelsLast 或 ChannelsLast3d，则转换为对应的通道顺序
  const std::string to_channels_last =
      ((params.memory_format == at::MemoryFormat::ChannelsLast) ||
       (params.memory_format == at::MemoryFormat::ChannelsLast3d))
      ? ".to(memory_format=torch." + channels_last_xd + ")"
      : "";

  // 创建字符串流对象，用于构建详细的错误信息和代码示例
  std::ostringstream ss;
  ss << "You can try to repro this exception using the following code snippet. ";
  ss << "If that doesn't trigger the error, please include your original repro script when reporting this issue.\n\n";
  ss << "import torch\n";
  ss << "torch.backends.cuda.matmul.allow_tf32 = "
     << pybool(at::globalContext().allowTF32CuBLAS()) << "\n";
  ss << "torch.backends.cudnn.benchmark = "
     << pybool(at::globalContext().benchmarkCuDNN()) << "\n";
  ss << "torch.backends.cudnn.deterministic = " << pybool(params.deterministic)
     << "\n";
  ss << "torch.backends.cudnn.allow_tf32 = " << pybool(params.allow_tf32)
     << "\n";
  ss << "data = torch.randn(" << ArrayRef<int>(params.input_size, dim)
     << ", dtype=" << full_dtype << ", ";
  ss << "device='cuda', requires_grad=True)" << to_channels_last << "\n";
  ss << "net = torch.nn.Conv" << dim - 2 << "d(" << in_channels << ", "
     << out_channels << ", ";
  ss << "kernel_size=" << ArrayRef<int>(&params.weight_size[2], dim - 2)
     << ", ";
  ss << "padding=" << ArrayRef<int>(params.padding, dim - 2) << ", ";
  ss << "stride=" << ArrayRef<int>(params.stride, dim - 2) << ", ";
  ss << "dilation=" << ArrayRef<int>(params.dilation, dim - 2) << ", ";
  ss << "groups=" << params.groups << ")\n";
  ss << "net = net.cuda()." << partial_dtype << "()" << to_channels_last
     << "\n";
  ss << "out = net(data)\n";
  ss << "out.backward(torch.randn_like(out))\n";
  ss << "torch.cuda.synchronize()\n\n";

  // 返回构建好的完整错误信息字符串
  return ss.str();
}

// ---------------------------------------------------------------------
//
// Convolution forward / Transposed convolution backward
//
// ---------------------------------------------------------------------

// 执行 CUDNN 卷积前向传播或转置卷积反向传播操作的函数
void cudnn_convolution_forward_out(
    TensorArg& output,  // 输出张量参数（结果）
    CheckedFrom c,      // 错误检查来源
    const TensorArg& input,   // 输入张量参数
    const TensorArg& weight,  // 权重张量参数
    IntArrayRef padding,      // 填充
    IntArrayRef stride,       // 步幅
    IntArrayRef dilation,     // 膨胀
    int64_t groups,           // 分组数
    bool benchmark,           // 是否使用基准模式
    bool deterministic,       // 是否确定性计算
    bool allow_tf32) {        // 是否允许 TF32 模式
  checkAllSameType(c, {input, weight});  // 检查输入和权重张量的数据类型是否相同
  checkAllSameGPU(c, {input, weight});   // 检查输入和权重张量是否在同一 GPU 上

  auto memory_format = output->suggest_memory_format();  // 根据输出建议的内存格式
  convolution_shape_check(
      c, input, weight, output, padding, stride, dilation, groups);  // 执行卷积形状检查

  Tensor weight_contig = weight->contiguous(memory_format);  // 获取连续的权重张量
  Tensor input_contig = input->contiguous(memory_format);    // 获取连续的输入张量

  raw_cudnn_convolution_forward_out(
      *output,
      input_contig,
      weight_contig,
      padding,
      stride,
      dilation,
      groups,
      benchmark,
      deterministic,
      allow_tf32);  // 执行 CUDNN 原始卷积前向传播操作
}

// 执行 CUDNN 卷积操作的函数
Tensor cudnn_convolution(
    const Tensor& input_t,          // 输入张量
    const Tensor& weight_t,         // 权重张量
    IntArrayRef padding,            // 填充
    IntArrayRef stride,             // 步幅
    IntArrayRef dilation,           // 膨胀
    int64_t groups,                 // 分组数
    bool benchmark,                 // 是否使用基准模式
    bool deterministic,             // 是否确定性计算
    bool allow_tf32) {              // 是否允许 TF32 模式
  TensorArg input{input_t, "input", 1}, weight{weight_t, "weight", 2};  // 定义输入和权重张量参数
  CheckedFrom c = "cudnn_convolution";  // 错误检查来源
  auto memory_format = cudnn_conv_suggest_memory_format(input_t, weight_t);  // 建议的内存格式
  Tensor output_t = at::detail::empty_cuda(
      conv_output_size(
          input_t.sizes(), weight_t.sizes(), padding, stride, dilation),  // 计算卷积输出大小
      input->options().memory_format(memory_format));  // 根据内存格式创建空的 CUDA 张量
  if (output_t.numel() == 0) {
    return output_t;  // 如果输出张量元素数为零，则直接返回空张量
  }
  // 避免在反向传播时输出名称歧义
  TensorArg output{output_t, "result", 0};  // 定义输出张量参数
  cudnn_convolution_forward_out(
      output,
      c,
      input,
      weight,
      padding,
      stride,
      dilation,
      groups,
      benchmark,
      deterministic,
      allow_tf32);  // 调用 CUDNN 卷积前向传播函数
  return *output;  // 返回输出张量
}

// 执行 CUDNN 卷积操作，并将结果存入预分配的输出张量中
at::Tensor& cudnn_convolution_out(
    const Tensor& input_t,          // 输入张量
    const Tensor& weight_t,         // 权重张量
    IntArrayRef padding,            // 填充
    IntArrayRef stride,             // 步幅
    IntArrayRef dilation,           // 膨胀
    int64_t groups,                 // 分组数
    bool benchmark,                 // 是否使用基准模式
    bool deterministic,             // 是否确定性计算
    bool allow_tf32,                // 是否允许 TF32 模式
    Tensor& output_t) {             // 输出张量的引用
  TensorArg input{input_t, "input", 1}, weight{weight_t, "weight", 2};  // 定义输入和权重张量参数
  CheckedFrom c = "cudnn_convolution";  // 错误检查来源
  if (output_t.numel() == 0) {
    return output_t;  // 如果输出张量元素数为零，则直接返回空张量的引用
  }
  TensorArg output{output_t, "result", 0};  // 定义输出张量参数
  cudnn_convolution_forward_out(
      output,
      c,
      input,
      weight,
      padding,
      stride,
      dilation,
      groups,
      benchmark,
      deterministic,
      allow_tf32);  // 调用 CUDNN 卷积前向传播函数
  return output_t;  // 返回输出张量的引用
}

// 注意：这里不需要 output_padding，因为没有需要解决的歧义
// 执行 CUDNN 转置卷积反向输入操作的函数
Tensor cudnn_convolution_transpose_backward_input(
    const Tensor& grad_output_t,   // 梯度输出张量
    const Tensor& weight_t,        // 权重张量
    IntArrayRef padding,                              # 定义一个名为 padding 的整数数组引用参数
    IntArrayRef stride,                               # 定义一个名为 stride 的整数数组引用参数
    IntArrayRef dilation,                             # 定义一个名为 dilation 的整数数组引用参数
    int64_t groups,                                   # 定义一个名为 groups 的64位整数参数
    bool benchmark,                                   # 定义一个名为 benchmark 的布尔型参数
    bool deterministic,                               # 定义一个名为 deterministic 的布尔型参数
    bool allow_tf32) {                                # 定义一个名为 allow_tf32 的布尔型参数
  TensorArg grad_output{grad_output_t, "grad_output", 1},  # 定义一个名为 grad_output 的张量参数，指定名称和索引
      weight{weight_t, "weight", 2};                   # 定义一个名为 weight 的张量参数，指定名称和索引
  auto memory_format =                                # 根据输入张量和权重张量推荐适合的内存格式
      cudnn_conv_suggest_memory_format(grad_output_t, weight_t);
  Tensor output_t = at::detail::empty_cuda(            # 创建一个 CUDA 空张量，用于存储卷积输出
      conv_output_size(                               # 计算卷积输出的尺寸
          grad_output_t.sizes(), weight_t.sizes(), padding, stride, dilation),
      grad_output_t.options().memory_format(memory_format));

  if (output_t.numel() == 0) {                        # 如果输出张量元素数为0，直接返回空张量
    return output_t;
  }
  TensorArg output{output_t, "result", 0};            # 定义一个名为 output 的张量参数，指定名称和索引
  cudnn_convolution_forward_out(                      # 调用 cudnn 库进行卷积前向计算
      output,
      "cudnn_convolution_transpose_backward_input",   # 指定卷积算法的名称
      grad_output,                                    # 梯度输出张量
      weight,                                         # 权重张量
      padding,                                        # 填充数组
      stride,                                         # 步幅数组
      dilation,                                       # 膨胀数组
      groups,                                         # 分组数
      benchmark,                                      # 是否使用基准
      deterministic,                                  # 是否确定性计算
      allow_tf32);                                    # 是否允许 TF32 格式
  return *output;                                     # 返回计算结果张量
}

// ---------------------------------------------------------------------
//
// Convolution backward / Transposed convolution forward
//
// ---------------------------------------------------------------------

// NOTE [ Backward vs transpose convolutions ]
//
// Backward and transpose are algorithmically equivalent, but they
// compute their geometry differently.  In a backwards, you knew what
// the original size of the input tensor was, so you can cache that
// geometry and fill it directly.  In transposed convolution, it is
// more conventional to not explicitly specify the output (previously
// input) size, and compute it.  This, however, leaves a degree of
// freedom; this degree of freedom is resolved using the
// output_padding parameter.  Both of these interfaces are equivalent,
// but they are differently convenient depending on the use case.

// 定义函数 cudnn_convolution_backward_input，用于计算 CUDNN 后向卷积输入的梯度
Tensor cudnn_convolution_backward_input(
    CheckedFrom c,                            // 检查来源参数
    IntArrayRef input_size,                   // 输入尺寸数组
    const TensorArg& grad_output,             // 梯度输出张量参数
    const TensorArg& weight,                  // 权重张量参数
    IntArrayRef padding,                      // 填充数组
    IntArrayRef stride,                       // 步幅数组
    IntArrayRef dilation,                     // 膨胀数组
    int64_t groups,                           // 组数
    bool benchmark,                           // 是否使用基准模式
    bool deterministic,                       // 是否确定性计算
    bool allow_tf32) {                        // 是否允许 TF32 计算

  checkAllSameType(c, {grad_output, weight});  // 检查梯度输出和权重张量类型是否相同
  checkAllSameGPU(c, {grad_output, weight});   // 检查梯度输出和权重张量是否在同一 GPU 上

  auto memory_format = cudnn_conv_suggest_memory_format(*grad_output, *weight);  // 推荐内存格式
  Tensor grad_input_t = at::detail::empty_cuda(
      input_size, grad_output->options().memory_format(memory_format));  // 创建空的 CUDA 张量 grad_input_t

  // 避免在作为转置卷积时使用 "grad_input"
  TensorArg grad_input{grad_input_t, "result", 0};  // 定义梯度输入张量参数

  // 执行卷积形状检查
  convolution_shape_check(
      c, grad_input, weight, grad_output, padding, stride, dilation, groups);

  // 使权重张量连续
  Tensor weight_contig = weight->contiguous(memory_format);
  // 使梯度输出张量连续
  Tensor grad_output_contig = grad_output->contiguous(memory_format);

  // 调用原始 CUDNN 后向卷积输入计算函数
  raw_cudnn_convolution_backward_input_out(
      *grad_input,
      grad_output_contig,
      weight_contig,
      padding,
      stride,
      dilation,
      groups,
      benchmark,
      deterministic,
      allow_tf32);

  // 返回计算后的梯度输入张量
  return *grad_input;
}

// 定义函数 cudnn_convolution_transpose_forward，用于计算 CUDNN 转置卷积的前向传播
Tensor cudnn_convolution_transpose_forward(
    CheckedFrom c,                            // 检查来源参数
    const TensorArg& grad_output,             // 梯度输出张量参数
    const TensorArg& weight,                  // 权重张量参数
    IntArrayRef padding,                      // 填充数组
    IntArrayRef output_padding,               // 输出填充数组
    IntArrayRef stride,                       // 步幅数组
    IntArrayRef dilation,                     // 膨胀数组
    int64_t groups,                           // 组数
    bool benchmark,                           // 是否使用基准模式
    bool deterministic,                       // 是否确定性计算
    bool allow_tf32) {                        // 是否允许 TF32 计算

  auto input_size = conv_input_size(
      grad_output->sizes(),
      weight->sizes(),
      padding,
      output_padding,
      stride,
      dilation,
      groups);                                // 计算输入尺寸

  // 调用 cudnn_convolution_backward_input 函数计算 CUDNN 后向卷积输入
  return cudnn_convolution_backward_input(
      c,
      input_size,
      grad_output,
      weight,
      padding,
      stride,
      dilation,
      groups,
      benchmark,
      deterministic,
      allow_tf32);
}

// 定义函数 cudnn_convolution_backward_input，用于计算 CUDNN 后向卷积输入的梯度
Tensor cudnn_convolution_backward_input(
    IntArrayRef input_size,                    // 输入尺寸数组
    const Tensor& grad_output_t,               // 梯度输出张量
    const Tensor& weight_t,                    // 权重张量
    IntArrayRef padding,                       // 填充数组
    IntArrayRef stride,                        // 步幅数组
    IntArrayRef dilation,     # 输入参数：表示卷积操作中的扩张大小的数组引用
    int64_t groups,           # 输入参数：表示卷积操作中的分组数
    bool benchmark,           # 输入参数：指示是否在算法中使用性能基准的布尔值
    bool deterministic,       # 输入参数：指示是否使用确定性算法的布尔值
    bool allow_tf32) {        # 输入参数：指示是否允许使用TF32混合精度的布尔值

  TensorArg grad_output{grad_output_t, "grad_output", 1},  # 定义梯度输出张量的参数信息
      weight{weight_t, "weight", 2};                      # 定义权重张量的参数信息
  
  // 调用 CuDNN 库的反向卷积输入函数
  return cudnn_convolution_backward_input(
      "cudnn_convolution_backward_input",  # CuDNN 函数名称字符串
      input_size,                          # 输入大小参数，描述卷积输入的形状
      grad_output,                         # 梯度输出张量参数
      weight,                              # 权重张量参数
      padding,                             # 填充参数，描述卷积操作中的填充策略
      stride,                              # 步幅参数，描述卷积操作中的步幅大小
      dilation,                            # 扩张参数，描述卷积操作中的扩张大小
      groups,                              # 分组参数，描述卷积操作中的分组数
      benchmark,                           # 性能基准布尔值参数
      deterministic,                       # 确定性算法布尔值参数
      allow_tf32                           # TF32混合精度允许布尔值参数
  );
}

Tensor cudnn_convolution_transpose(
    const Tensor& input_t,  // 输入张量
    const Tensor& weight_t,  // 权重张量
    IntArrayRef padding,  // 填充数组
    IntArrayRef output_padding,  // 输出填充数组
    IntArrayRef stride,  // 步长数组
    IntArrayRef dilation,  // 膨胀数组
    int64_t groups,  // 分组数量
    bool benchmark,  // 是否基准测试
    bool deterministic,  // 是否确定性运算
    bool allow_tf32) {  // 是否允许 TF32 模式

  TensorArg input{input_t, "input", 1}, weight{weight_t, "weight", 2};  // 定义输入和权重张量参数
  CheckedFrom c = "cudnn_convolution_transpose";  // 检查来源
  auto output_t = cudnn_convolution_transpose_forward(  // 执行反卷积前向传播
      c,
      input,
      weight,
      padding,
      output_padding,
      stride,
      dilation,
      groups,
      benchmark,
      deterministic,
      allow_tf32);
  return output_t;  // 返回输出张量
}

// ---------------------------------------------------------------------
//
// Convolution backward (weight)
//
// ---------------------------------------------------------------------

Tensor cudnn_convolution_backward_weight(
    CheckedFrom c,  // 检查来源
    IntArrayRef weight_size,  // 权重尺寸数组
    const Tensor& grad_output_t,  // 梯度输出张量
    const Tensor& input_t,  // 输入张量
    IntArrayRef padding,  // 填充数组
    IntArrayRef stride,  // 步长数组
    IntArrayRef dilation,  // 膨胀数组
    int64_t groups,  // 分组数量
    bool benchmark,  // 是否基准测试
    bool deterministic,  // 是否确定性运算
    bool allow_tf32) {  // 是否允许 TF32 模式

  auto layout = cudnn_conv_suggest_memory_format(input_t, grad_output_t);  // 推荐内存格式布局

  Tensor grad_output_contig_t = grad_output_t.contiguous(layout);  // 使梯度输出张量连续
  TensorArg grad_output_contig{grad_output_contig_t, "grad_output", 1};  // 定义连续的梯度输出张量参数

  Tensor input_contig_t = input_t.contiguous(layout);  // 使输入张量连续
  TensorArg input{input_contig_t, "input", 2};  // 定义连续的输入张量参数

  checkAllSameType(c, {grad_output_contig, input});  // 检查所有张量是否为相同类型
  checkAllSameGPU(c, {grad_output_contig, input});  // 检查所有张量是否在相同的 GPU 上

  auto grad_weight_t =
      at::empty(weight_size, grad_output_contig->options(), layout);  // 创建空的梯度权重张量

  // For uniformity with everything else, although it seems grad_weight
  // would be unambiguous too.
  TensorArg grad_weight{grad_weight_t, "result", 0};  // 定义梯度权重张量参数
  convolution_shape_check(
      c,
      input,
      grad_weight,
      grad_output_contig,
      padding,
      stride,
      dilation,
      groups);  // 卷积形状检查

  raw_cudnn_convolution_backward_weight_out(
      *grad_weight,
      *grad_output_contig,
      *input,
      padding,
      stride,
      dilation,
      groups,
      benchmark,
      deterministic,
      allow_tf32);  // 执行原始的 cudnn 权重反卷积

  return grad_weight_t;  // 返回梯度权重张量
}

Tensor cudnn_convolution_backward_weight(
    IntArrayRef weight_size,  // 权重尺寸数组
    const Tensor& grad_output_t,  // 梯度输出张量
    const Tensor& input_t,  // 输入张量
    IntArrayRef padding,  // 填充数组
    IntArrayRef stride,  // 步长数组
    IntArrayRef dilation,  // 膨胀数组
    int64_t groups,  // 分组数量
    bool benchmark,  // 是否基准测试
    bool deterministic,  // 是否确定性运算
    bool allow_tf32) {  // 是否允许 TF32 模式

  return cudnn_convolution_backward_weight(
      "cudnn_convolution_backward_weight",  // 检查来源
      weight_size,
      grad_output_t,
      input_t,
      padding,
      stride,
      dilation,
      groups,
      benchmark,
      deterministic,
      allow_tf32);
}

std::tuple<at::Tensor, at::Tensor> cudnn_convolution_backward(
    const at::Tensor& input,  // 输入张量
    const at::Tensor& grad_output_t,  // 梯度输出张量
    const at::Tensor& weight,  // 权重张量
    IntArrayRef padding,  // 填充数组
    IntArrayRef stride,  // 步长数组
    // 定义函数的参数列表：梯度输入，卷积操作参数，是否使用cuDNN的benchmark模式，是否确定性计算，是否允许使用TF32加速，输出掩码数组
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool allow_tf32,
    std::array<bool, 2> output_mask) {
  // 将梯度输出张量转换为与输入张量推荐的内存格式
  Tensor grad_output = grad_output_t.to(input.suggest_memory_format());

  Tensor grad_input, grad_weight;
  // 如果输入张量的元素数量为0
  if (input.numel() == 0) {
    // 如果第一个输出掩码为true，则创建一个与输入张量相同大小的空张量（使用旧式连续内存格式）
    if (output_mask[0]) {
      grad_input = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    }
    // 如果第二个输出掩码为true，则创建一个与权重张量相同大小的零张量（使用旧式连续内存格式）
    if (output_mask[1]) {
      grad_weight = at::zeros_like(weight, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    }
  } else {
    // 如果第一个输出掩码为true，则使用cuDNN计算卷积操作的输入梯度
    if (output_mask[0]) {
      grad_input = cudnn_convolution_backward_input(
          input.sizes(),
          grad_output,
          weight,
          padding,
          stride,
          dilation,
          groups,
          benchmark,
          deterministic,
          allow_tf32);
    }
    // 如果第二个输出掩码为true，则使用cuDNN计算卷积操作的权重梯度
    if (output_mask[1]) {
      grad_weight = cudnn_convolution_backward_weight(
          weight.sizes(),
          grad_output,
          input,
          padding,
          stride,
          dilation,
          groups,
          benchmark,
          deterministic,
          allow_tf32);
    }
  }

  // 返回包含计算得到的输入梯度和权重梯度的元组
  return std::tuple<Tensor, Tensor>{grad_input, grad_weight};
}
// cudnn_convolution_transpose_backward_weight 函数实现了使用 cuDNN 进行转置卷积反向传播的权重更新。
Tensor cudnn_convolution_transpose_backward_weight(
    // weight_size: 权重张量的尺寸
    IntArrayRef weight_size,
    // grad_output_t: 输出梯度张量
    const Tensor& grad_output_t,
    // input_t: 输入张量
    const Tensor& input_t,
    // padding: 填充
    IntArrayRef padding,
    // stride: 步长
    IntArrayRef stride,
    // dilation: 膨胀率
    IntArrayRef dilation,
    // groups: 分组数
    int64_t groups,
    // benchmark: 是否使用基准模式
    bool benchmark,
    // deterministic: 是否使用确定性模式
    bool deterministic,
    // allow_tf32: 是否允许使用 TF32
    bool allow_tf32) {
  // 调用 cuDNN 的反向权重更新函数 cudnn_convolution_backward_weight
  return cudnn_convolution_backward_weight(
      "cudnn_convolution_backward_weight",  // 函数名
      weight_size,  // 权重尺寸
      input_t,  // 输入张量
      grad_output_t,  // 输出梯度张量
      padding,  // 填充
      stride,  // 步长
      dilation,  // 膨胀率
      groups,  // 分组数
      benchmark,  // 是否基准模式
      deterministic,  // 是否确定性模式
      allow_tf32);  // 是否允许 TF32
}

// cudnn_convolution_transpose_backward 函数实现了使用 cuDNN 进行转置卷积的反向传播。
std::tuple<at::Tensor, at::Tensor> cudnn_convolution_transpose_backward(
    // input: 输入张量
    const at::Tensor& input,
    // grad_output_t: 输出梯度张量
    const at::Tensor& grad_output_t,
    // weight: 权重张量
    const at::Tensor& weight,
    // padding: 填充
    IntArrayRef padding,
    // output_padding: 输出填充
    IntArrayRef output_padding,
    // stride: 步长
    IntArrayRef stride,
    // dilation: 膨胀率
    IntArrayRef dilation,
    // groups: 分组数
    int64_t groups,
    // benchmark: 是否使用基准模式
    bool benchmark,
    // deterministic: 是否使用确定性模式
    bool deterministic,
    // allow_tf32: 是否允许使用 TF32
    bool allow_tf32,
    // output_mask: 输出遮罩数组
    std::array<bool, 2> output_mask) {
  // 将 grad_output_t 转换为连续内存格式，并推荐内存格式为 input 的格式
  Tensor grad_output = grad_output_t.contiguous(input.suggest_memory_format());

  Tensor grad_input, grad_weight;
  // 如果 output_mask 的第一个元素为 true
  if (output_mask[0]) {
    // 调用 cudnn_convolution_transpose_backward_input 函数计算输入梯度
    grad_input = cudnn_convolution_transpose_backward_input(
        grad_output,
        weight,
        padding,
        stride,
        dilation,
        groups,
        benchmark,
        deterministic,
        allow_tf32);
  }
  // 如果 output_mask 的第二个元素为 true
  if (output_mask[1]) {
    // 调用 cudnn_convolution_transpose_backward_weight 函数计算权重梯度
    grad_weight = cudnn_convolution_transpose_backward_weight(
        weight.sizes(),
        grad_output,
        input,
        padding,
        stride,
        dilation,
        groups,
        benchmark,
        deterministic,
        allow_tf32);
  }

  // 返回计算得到的梯度张量 grad_input 和 grad_weight 的元组
  return std::tuple<Tensor, Tensor>{grad_input, grad_weight};
}

// cudnn_convolution_relu 函数实现了在输入上执行带有 ReLU 的 cuDNN 卷积操作。
Tensor cudnn_convolution_relu(
    // input_t: 输入张量
    const Tensor& input_t,
    // weight_t: 权重张量
    const Tensor& weight_t,
    // bias_t: 偏置张量（可选）
    const std::optional<Tensor>& bias_t,
    // stride: 步长
    IntArrayRef stride,
    // padding: 填充
    IntArrayRef padding,
    // dilation: 膨胀率
    IntArrayRef dilation,
    // groups: 分组数
    int64_t groups) {
  // 推荐的内存格式
  auto memory_format = cudnn_conv_suggest_memory_format(input_t, weight_t);
  // 将输入张量和权重张量转换为连续内存格式
  const Tensor input = input_t.contiguous(memory_format);
  const Tensor weight = weight_t.contiguous(memory_format);

  // 根据输入、权重、填充、步长和膨胀率计算输出的尺寸
  Tensor output_t = at::detail::empty_cuda(
      conv_output_size(
          input.sizes(), weight.sizes(), padding, stride, dilation),
      input.options().memory_format(memory_format));
  // 如果输出张量的元素数为 0
  if (output_t.numel() == 0) {
    return output_t;
  }

  // 获取全局的 ATen 上下文
  auto& ctx = at::globalContext();
  // 检查是否开启了 CuDNN 的性能基准测试
  bool benchmark = ctx.benchmarkCuDNN();
  // 检查是否允许使用 CuDNN 的 TF32 加速
  bool allow_tf32 = ctx.allowTF32CuDNN();
  // 如果存在偏置项，则使用给定的偏置值；否则创建一个与输出通道数相同的全零张量作为偏置
  auto _bias = bias_t.has_value()
      ? bias_t.value()
      : at::zeros(
            {output_t.size(1)},  // 与输出张量的通道数相同
            optTypeMetaToScalarType(output_t.options().dtype_opt()),  // 使用输出张量的数据类型
            output_t.options().layout_opt(),  // 使用输出张量的布局
            output_t.options().device_opt(),  // 使用输出张量的设备
            output_t.options().pinned_memory_opt());  // 是否使用固定内存

  // 调用底层的 CUDNN 卷积操作并加上 ReLU 激活函数
  raw_cudnn_convolution_add_relu_out(
      output_t,          // 输出张量
      input,             // 输入张量
      weight,            // 卷积核张量
      output_t,          // 将输出张量作为 z 参数，以满足 CUDNN API 要求
      0,                 // alpha 参数，这里为零
      _bias,             // 偏置张量
      stride,            // 步幅
      padding,           // 填充
      dilation,          // 空洞卷积扩展系数
      groups,            // 分组卷积参数
      benchmark,         // 是否进行性能基准测试
      false,             // 是否确定性执行
      allow_tf32         // 是否允许 TF32 加速
  );

  // 返回经过卷积操作后的输出张量
  return output_t;
} // namespace native
} // namespace at
```