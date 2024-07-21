# `.\pytorch\aten\src\ATen\native\quantized\cudnn\Pooling.cpp`

```py
// 包含 C10 异常处理工具的头文件
#include <c10/util/Exception.h>
#ifdef USE_CUDA
// 包含 CUDA 配置信息，例如 AT_CUDNN_ENABLED 的定义
#include <ATen/cuda/CUDAConfig.h>  // for the definition of AT_CUDNN_ENABLED

#if AT_CUDNN_ENABLED()
// 如果使用了 CUDNN，包含相关的异常处理、描述符、句柄和类型定义头文件
#include <ATen/cuda/Exceptions.h>
#include <ATen/cudnn/Descriptors.h>
#include <ATen/cudnn/Handle.h>
#include <ATen/cudnn/Types.h>
#endif // AT_CUDNN_ENABLED
#endif // USE_CUDA

// 包含 ATen 核心库和相关的池化操作和张量迭代器头文件
#include <ATen/ATen.h>
#include <ATen/native/Pool.h>
#include <ATen/native/TensorIterator.h>
#include <c10/core/QScheme.h>
#include <c10/core/ScalarType.h>
#include <c10/util/ArrayRef.h>
#include <torch/library.h>

#include <vector>

namespace at {
namespace native {
namespace {
// TODO: 此函数与 Pooling.cpp 中的函数相同。我们应将其重构到量化目录中，
// 以避免重复实现
// 检查 maxpool2d 的参数是否合法
void check_maxpool2d_params(
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation) {
  // 检查核大小应为 1 维或 2 维
  TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 2,
              "Expected 1d or 2d kernel size, got ", kernel_size.size());
  // 检查步幅应为空或者为 2 维
  TORCH_CHECK(stride.empty() || stride.size() == 2,
              "Expected no strides or 2d strides, got", stride.size());
  // 检查填充应为 1 维或 2 维
  TORCH_CHECK(padding.size() == 1 || padding.size() == 2,
              "Expected 1d or 2d padding, got ", padding.size());
  // 检查扩展应为 1 维或 2 维
  TORCH_CHECK(dilation.size() == 1 || dilation.size() == 2,
              "Expected 1d or 2d dilation, got ", dilation.size());
}
}

// 当前的量化 CUDA 自适应平均池化实现使用以下步骤：
// 解量化 -> fp32 自适应平均池化 -> 量化。数值上与量化自适应平均池化相同。
// 这不是理想的实现方式，我们希望直接操作量化值。
// 但目前受阻于等待 cudnn 8.5.0 发布，预计将支持自适应平均池化。
// 一旦支持可用，我们将直接使用它。TODO
// 在 CUDA 下执行量化自适应平均池化操作
Tensor adaptive_avg_pool2d_quantized_cuda(
    const at::Tensor& input,
    IntArrayRef output_size) {
// TODO: 当我们使用 cudnn 实现这个函数时，重新启用类似 quantized_max_pool2d_cudnn 的预处理器
#ifdef USE_CUDA
// #if AT_CUDNN_ENABLED()
    // TODO: 目前仅限于每张量量化张量，尽管适应为每通道量化张量应该也很容易
    // 检查输入张量是否是每张量仿射量化的
    TORCH_CHECK(input.qscheme() == at::kPerTensorAffine, "adaptive_avg_pool2d_quantized_cuda only supports per tensor quantized tensors");
    // 解量化输入张量为 float32 格式
    auto input_fp32 = at::dequantize(input);
    // 执行 float32 格式的自适应平均池化
    auto result_fp32 = at::adaptive_avg_pool2d(input_fp32, output_size);
    // 将结果量化为每张量的量化格式，保留输入张量的量化参数和标量类型
    return at::quantize_per_tensor(result_fp32, input.q_scale(), input.q_zero_point(), input.scalar_type());
#else // USE_CUDA
  // 如果未使用 CUDA，抛出错误
  AT_ERROR("at::native::adaptive_avg_pool2d_quantized_cuda: ATen not compiled with USE_CUDA support");
  return Tensor{}; // 不会执行到这里，用于安抚编译器
#endif
}
// 目前我们支持4D和3D输入（qx）张量，后者是为了兼容旧版本而支持的。
// 4D输入张量的第一个维度是批量大小。
// 对于3D张量，没有批量大小维度 -- 可以视为单个批量。
// cudnn的2D池化操作要求输入和输出都是4D张量，因此我们必须在使用cudnn之前将任何3D张量转换为4D。
// 此实现当前使用v7的cudnn API，因为v8的cudnn API对池化操作尚不可用。
// 请参阅https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnPoolingForward以获取API文档。
// 目前似乎cudnn不支持扩张池化 -- 我们将向cudnn提交此功能请求。
// TODO: 理想情况下，我们希望在这里使用结构化的内核支持，这样就不必重复输入检查，但这需要我们根据当前在native_functions.yaml中构建的调度表来实现max_pool2d_with_indices_out_quantized_cuda。
// 目前cudnn不支持生成最大池化的索引，因此在此功能可用之前，这是不可能的。
Tensor quantized_max_pool2d_cudnn(
    const Tensor& qx,                    // 输入张量qx
    IntArrayRef kernel_size,             // 池化核大小
    IntArrayRef stride,                  // 步幅
    IntArrayRef padding,                 // 填充
    IntArrayRef dilation,                // 扩张
    bool ceil_mode) {                    // 是否使用ceil模式进行池化
#ifdef USE_CUDA
#if AT_CUDNN_ENABLED()
  // 检查池化操作的参数是否合法
  check_maxpool2d_params(
      kernel_size,
      stride,
      padding,
      dilation);
  // 如果步幅为空，则将其设置为核大小
  if (stride.empty()) {
    stride = kernel_size;
  }
  // 获取输入张量的维度
  auto ndim = qx.dim();
  // 检查输入张量的维度是否为3或4
  TORCH_CHECK(
      ndim == 3 || ndim == 4, "Expecting the input tensor of rank 3 or 4.");
  // 检查核大小是否为2维
  TORCH_CHECK(
      kernel_size.size() == 2,
      "quantized_max_pool2d_cudnn(): Expected kernel_size to be 2-dimensional: got ",
      kernel_size.size());
  // 检查步幅是否为2维
  TORCH_CHECK(
      stride.size() == 2,
      "quantized_max_pool2d_cudnn(): Expected stride to be 2-dimensional: got ",
      stride.size());
  // 检查扩张是否为2维，并且扩张值都为1
  TORCH_CHECK(
      dilation.size() == 2,
      "quantized_max_pool2d_cudnn(): Expected dilation to be 2-dimensional: got ",
      dilation.size());
  TORCH_CHECK(
      dilation[0] == 1 && dilation[1] == 1,
      "quantized_max_pool2d_cudnn(): Expected dilation=[1, 1] (cudnn does not currently support dilation[i] != 1), got",
      dilation);
  // 检查填充是否为2维
  TORCH_CHECK(
      padding.size() == 2,
      "quantized_max_pool2d_cudnn(): Expected padding to be 2-dimensional: got ",
      padding.size());

  // 将输入设置为qx
  auto input = qx;
  // 如果输入张量为4D，则将其格式转换为ChannelsLast
  if (ndim == 4) {
    input = qx.to(MemoryFormat::ChannelsLast);
  } else { // 3D
    // 创建新的大小向量以将3D张量转换为4D
    std::vector<int64_t> new_sizes{1, qx.size(0), qx.size(1), qx.size(2)};
    // 获取输入张量的新尺寸
    input = qx.view(new_sizes);
  }
  // 获取批量大小
  int batch_size = input.size(0);
  // 获取输入张量的通道数、高度和宽度
  int64_t inC = input.size(1);
  int64_t inH = input.size(2);
  int64_t inW = input.size(3);
  // 检查输出的维度信息
  int64_t padH = padding[0];  // 垂直方向填充
  int64_t padW = padding[1];  // 水平方向填充
  int64_t kH = kernel_size[0];  // 卷积核高度
  int64_t kW = kernel_size[1];  // 卷积核宽度
  int64_t strideH = stride[0];  // 垂直方向步长
  int64_t strideW = stride[1];  // 水平方向步长
  // 检查卷积核和步长是否有效
  TORCH_CHECK(
      kH > 0 && kW > 0,
      "qnnpack_maxpool2d(): kernel_size should be greater than zero.");
  TORCH_CHECK(
      strideH > 0 && strideW > 0,
      "qnnpack_maxpool2d(): strides should be greater than zero.");
  int64_t dilationH = dilation[0];  // 垂直方向膨胀率
  int64_t dilationW = dilation[1];  // 水平方向膨胀率
  int64_t outC = inC;  // 输出通道数与输入通道数相同
  // 计算池化操作后的输出高度和宽度
  int64_t outH = pooling_output_shape(inH, kH, padH, strideH, dilationH, ceil_mode);
  int64_t outW = pooling_output_shape(inW, kW, padW, strideW, dilationW, ceil_mode);
  // 检查输出尺寸是否合适
  TORCH_CHECK(outH > 0 && outW > 0,
              "Given input size: (",
              inC, "x", inH, "x", inW,
              "). Calculated output size: (",
              outC, "x", outH, "x", outW,
              "). Output size is too small.");

  std::vector<int64_t> output_shape;
  if (ndim == 3) {
    // 对于3维输入，cudnn要求4维输入和输出，因此在前面添加一个虚拟维度（大小为1）
    output_shape = {1, outC, outH, outW};
  } else {
    output_shape = {batch_size, outC, outH, outW};
  }
  // 创建一个空的量化张量
  auto qy = at::_empty_affine_quantized(
      output_shape,
      at::device(at::kCUDA).dtype(at::ScalarType::QInt8),
      input.q_scale(),
      input.q_zero_point(),
      (ndim == 4 ? MemoryFormat::ChannelsLast : MemoryFormat::Contiguous));

  // 获取CUDNN句柄
  cudnnHandle_t handle = getCudnnHandle();
  cudnnPoolingDescriptor_t poolingDesc;
  // 创建CUDNN池化描述符
  AT_CUDNN_CHECK_WITH_SHAPES(cudnnCreatePoolingDescriptor(&poolingDesc));
  // 设置CUDNN池化描述符参数
  AT_CUDNN_CHECK_WITH_SHAPES(cudnnSetPooling2dDescriptor(
      poolingDesc,
      CUDNN_POOLING_MAX_DETERMINISTIC,
      CUDNN_NOT_PROPAGATE_NAN,
      kernel_size[0], // 卷积核高度
      kernel_size[1], // 卷积核宽度
      padding[0], // 垂直填充
      padding[1], // 水平填充
      stride[0], // 垂直步长
      stride[1])); // 水平步长

  float one{1};  // 浮点数1
  float zero{0.0};  // 浮点数0
  TensorDescriptor xDesc;
  // 根据输入的内存格式设置TensorDescriptor
  at::MemoryFormat memory_format = (ndim == 4 ? at::MemoryFormat::ChannelsLast : at::MemoryFormat::Contiguous);
  xDesc.set(input, memory_format);
  TensorDescriptor yDesc;
  // 根据输出的内存格式设置TensorDescriptor
  yDesc.set(qy, memory_format);
  // 执行CUDNN池化前向操作
  cudnnPoolingForward(handle,
                      poolingDesc,
                      &one,
                      xDesc.desc(),
                      input.data_ptr<int8_t>(),
                      &zero,
                      yDesc.desc(),
                      qy.data_ptr<int8_t>());

  // 如果输入是3维的，则返回时将结果张量重新转换为3维
  return (ndim == 3 ? qy.view(std::vector<int64_t>(output_shape.begin() + 1, output_shape.end())) : qy);
#else // AT_CUDNN_ENABLED()
  // 如果未启用 cuDNN 支持，则抛出错误信息
  AT_ERROR("at::native::quantized_max_pool2d_cudnn: ATen not compiled with cuDNN support");
  // 返回一个空的 Tensor，编译器永远不会到达这里，用于消除编译器警告
  return Tensor{}; // never reached, placates the compiler
#endif // AT_CUDNN_ENABLED()
#else // USE_CUDA
  // 如果未启用 CUDA 支持，则抛出错误信息
  AT_ERROR("at::native::quantized_max_pool2d_cudnn: ATen not compiled with USE_CUDA support");
  // 返回一个空的 Tensor，编译器永远不会到达这里，用于消除编译器警告
  return Tensor{}; // never reached, placates the compiler
#endif
}

// Keep the registry in the anonymous namespace.
namespace {
// 定义一个模板类 QMaxPool_arr_args，用于执行量化最大池化操作
template <uint32_t kSpatialDim>
class QMaxPool_arr_args final {
 public:
  // 静态成员函数 run，接受输入 Tensor qx 和池化操作的参数，执行量化最大池化
  static Tensor run(
      const Tensor& qx,
      std::vector<int64_t> kernel_size,
      std::vector<int64_t> stride,
      std::vector<int64_t> padding,
      std::vector<int64_t> dilation,
      bool ceil_mode) {
    // 检查空间维度是否为 2，否则抛出错误
    TORCH_CHECK(kSpatialDim == 2, "quantized max pool is only valid for 2D")
    // 调用 quantized_max_pool2d_cudnn 函数执行量化最大池化操作
    return quantized_max_pool2d_cudnn(qx, kernel_size, stride, padding,
                                    dilation, ceil_mode);
  }
};

// 在匿名命名空间中注册 TORCH_LIBRARY_IMPL
TORCH_LIBRARY_IMPL(quantized, QuantizedCUDA, m) {
  // 将 quantized::max_pool2d 的实现注册为 QMaxPool_arr_args<2>::run
  m.impl(TORCH_SELECTIVE_NAME("quantized::max_pool2d"), TORCH_FN(QMaxPool_arr_args<2>::run));
}

} // namespace
} // namespace native
} // namespace at
```