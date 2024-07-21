# `.\pytorch\aten\src\ATen\native\cudnn\GridSampler.cpp`

```py
// 定义宏，仅允许方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 包含 ATen 库的配置文件
#include <ATen/Config.h>
// 包含 ATen 库的核心张量类定义
#include <ATen/core/Tensor.h>
// 包含 ATen 库的 CUDA 配置相关文件
#include <ATen/cuda/CUDAConfig.h>
// 包含 ATen 库的 GridSamplerUtils 头文件
#include <ATen/native/GridSamplerUtils.h>

// 如果未定义 AT_PER_OPERATOR_HEADERS 宏，则包含以下头文件
#ifndef AT_PER_OPERATOR_HEADERS
// 包含 ATen 库的函数声明头文件
#include <ATen/Functions.h>
// 包含 ATen 库的 NativeFunctions 头文件
#include <ATen/NativeFunctions.h>
// 否则，包含以下头文件
#else
// 包含 ATen 库的 cudnn_grid_sampler_backward_native 头文件
#include <ATen/ops/cudnn_grid_sampler_backward_native.h>
// 包含 ATen 库的 cudnn_grid_sampler_native 头文件
#include <ATen/ops/cudnn_grid_sampler_native.h>
// 包含 ATen 库的 empty 头文件
#include <ATen/ops/empty.h>
#endif

// 如果未启用 cuDNN 支持，则进入下面的命名空间
#if !AT_CUDNN_ENABLED()

namespace at {
namespace native {

// ATen 未编译支持 cuDNN 时，抛出错误信息的前向传播函数
Tensor cudnn_grid_sampler_forward(const Tensor& input_t, const Tensor& grid_t) {
  AT_ERROR("cudnn_grid_sampler_forward: ATen not compiled with cuDNN support");
}

// ATen 未编译支持 cuDNN 时，抛出错误信息的反向传播函数
std::tuple<Tensor, Tensor> cudnn_grid_sampler_backward(
    const Tensor& input_t,
    const Tensor& grid_t,
    const Tensor& grad_output_t) {
  AT_ERROR("cudnn_grid_sampler_backward: ATen not compiled with cuDNN support");
}

} // namespace native
} // namespace at

// 否则，即 AT_CUDNN_ENABLED 宏已定义，则执行以下代码
#else // AT_CUDNN_ENABLED

// 包含 ATen 库的异常处理头文件
#include <ATen/cuda/Exceptions.h>
// 包含 ATen 库的 cudnn 描述符头文件
#include <ATen/cudnn/Descriptors.h>
// 包含 ATen 库的 cudnn 类型定义头文件
#include <ATen/cudnn/Types.h>
// 包含 ATen 库的 cudnn 实用函数头文件
#include <ATen/cudnn/Utils.h>

// 包含 ATen 库的张量工具头文件
#include <ATen/TensorUtils.h>
// 包含 C10 库的 irange 头文件
#include <c10/util/irange.h>

// TODO: 描述符检查

namespace at {
namespace native {

// 匿名命名空间，定义了设置空间变换器描述符的函数
namespace {

void setSamplerDescriptor(
    SpatialTransformerDescriptor& desc,
    cudnnDataType_t dataType,
    const at::Tensor& tensor) {
  // 定义并初始化输入张量的大小数组
  int inputSize[4] = {0};
  // 遍历输入张量的维度，将每个维度的大小存入 inputSize 数组
  for (const auto i : c10::irange(tensor.dim())) {
    inputSize[i] = (int)tensor.size(i);
  }
  // 使用给定的数据类型和尺寸数组设置空间变换器描述符
  desc.set(dataType, 4, inputSize);
}

// 检查网格张量的大小和连续性
void checkGridSize(CheckedFrom c, TensorArg grid, TensorArg input) {
  // 断言网格张量是连续的
  checkContiguous(c, grid);
  // 断言网格张量的维度为 4
  checkDim(c, grid, 4);
  // TODO: 或许更友好地报告期望的大小来自哪里
  // 检查网格张量在指定维度上的大小是否正确
  checkSize(c, grid, 0, input->size(0));
  // 检查网格张量在第三维上的大小是否为 2
  checkSize(c, grid, 3, 2);
}

} // namespace
// 定义 cudnn_grid_sampler_forward 函数，接受输入张量 input_t 和 grid_t，并返回输出张量
Tensor cudnn_grid_sampler_forward(const Tensor& input_t, const Tensor& grid_t) {
  // 查看注释 [ grid_sampler Native Functions ]。
  // 在此处添加检查，以防此函数被错误调用而不是 grid_sampler。
  check_grid_sampler_common(input_t, grid_t);
  // 检查输入张量和网格张量是否符合 cudnn_grid_sampler 的条件
  TORCH_CHECK(
      cond_cudnn_grid_sampler(input_t, grid_t),
      "Invalid arguments to cudnn_grid_sampler_forward");

  // 如果输入张量的步幅为零，创建其连续版本
  auto input_contig = contiguousIfZeroInStrides(input_t);
  // 创建网格张量的连续版本
  auto grid_contig = grid_t.contiguous();
  // 定义输入张量和网格张量的参数对象，用于后续的检查
  TensorArg input{input_contig, "input", 1}, grid{grid_contig, "grid", 2};
  // 检查输入张量和网格张量是否在同一个 GPU 上
  CheckedFrom c = "cudnn_grid_sampler_forward";
  checkAllSameGPU(c, {input, grid});
  // 检查输入张量和网格张量是否具有相同的数据类型
  checkAllSameType(c, {input, grid});
  // 检查网格张量的尺寸是否与输入张量的相符
  checkGridSize(c, grid, input);
  // 检查输入张量的维度是否为4
  checkDim(c, input, 4);

  // 创建一个空的输出张量，其数据类型与输入张量相同
  auto output_t = at::empty({0}, input->options());
  // 重新调整输出张量的尺寸以匹配输入张量和网格张量的维度
  output_t.resize_(
      {input->size(0), input->size(1), grid->size(1), grid->size(2)});

  // 创建输入张量的描述符
  TensorDescriptor idesc{*input}; // input descriptor
  // 创建输出张量的描述符
  TensorDescriptor odesc{output_t}; // output descriptor
  // 创建空间变换器的描述符
  SpatialTransformerDescriptor desc; // sampler descriptor

  // 获取 CuDNN 句柄
  auto handle = getCudnnHandle();
  // 获取输入张量的数据类型
  auto dataType = getCudnnDataType(*input);
  // 设置空间变换器的描述符
  setSamplerDescriptor(desc, dataType, output_t);

  // 创建数据类型为 dataType 的常量 one 和 zero
  Constant one(dataType, 1);
  Constant zero(dataType, 0);
  // 调用 CuDNN 函数 cudnnSpatialTfSamplerForward，执行空间变换采样操作
  AT_CUDNN_CHECK(cudnnSpatialTfSamplerForward(
      handle,
      desc.desc(),
      &one,
      idesc.desc(),
      input->const_data_ptr(),
      grid->const_data_ptr(),
      &zero,
      odesc.desc(),
      output_t.data_ptr()));

  // 返回输出张量
  return output_t;
}

// 注意：CuDNN 不支持输出掩码；你总是会得到梯度和输出。
// 定义 cudnn_grid_sampler_backward 函数，接受输入张量 input_t 和 grid_t，并返回梯度张量和网格张量的元组
std::tuple<Tensor, Tensor> cudnn_grid_sampler_backward(
    const Tensor& input_t,
    const Tensor& grid_t,
    // 定义函数 cudnn_grid_sampler_backward，接收输入、网格和梯度输出张量作为参数
    const Tensor& cudnn_grid_sampler_backward(
        const Tensor& input_t,
        const Tensor& grid_t,
        const Tensor& grad_output_t) {
      // 查看注释 [ grid_sampler Native Functions ]，用于说明此函数的作用
      // 添加检查以防止此函数被错误调用而非 grid_sampler
      check_grid_sampler_common(input_t, grid_t);
      // 使用 TORCH_CHECK 进行条件检查，确保输入和网格的有效性
      TORCH_CHECK(
          cond_cudnn_grid_sampler(input_t, grid_t),
          "Invalid arguments to cudnn_grid_sampler_backward");
    
      // 将输入张量进行零步长优化，使其连续
      auto input_contig = contiguousIfZeroInStrides(input_t);
      // 确保网格张量是连续的
      auto grid_contig = grid_t.contiguous();
      // 将梯度输出张量进行零步长优化，使其连续
      auto grad_output_contig = contiguousIfZeroInStrides(grad_output_t);
      // 定义输入、网格和梯度输出张量的 TensorArg，用于后续的检查
      TensorArg input{input_contig, "input", 1}, grid{grid_contig, "grid", 2},
          grad_output{grad_output_contig, "grad_output", 3};
      // 创建 CheckedFrom 对象 c，用于错误检查
      CheckedFrom c = "cudnn_grid_sampler_backward";
      // 检查所有张量是否在同一个 GPU 上
      checkAllSameGPU(c, {input, grad_output, grid});
      // 检查网格的大小是否与输入张量一致
      checkGridSize(c, grid, input);
      // 检查输入张量的维度是否为 4
      checkDim(c, input, 4);
      // 检查梯度输出张量的维度是否为 4
      checkDim(c, grad_output, 4);
    
      // 创建一个空的梯度输入张量，与输入张量具有相同的选项（数据类型、设备等）
      auto grad_input_t = at::empty({0}, input->options());
      // 调整梯度输入张量的大小，与输入张量相同
      grad_input_t.resize_(input->sizes());
      // 创建一个空的梯度网格张量，与网格张量具有相同的选项
      auto grad_grid_t = at::empty({0}, grid->options());
      // 调整梯度网格张量的大小，与网格张量相同
      grad_grid_t.resize_(grid->sizes());
    
      // 创建输入描述符 idesc，使用输入张量
      TensorDescriptor idesc{*input}; // input descriptor
      // 创建输出描述符 odesc，使用梯度输出张量
      TensorDescriptor odesc{*grad_output}; // grad_output descriptor
      // 创建梯度输入描述符 gdesc，使用梯度输入张量
      TensorDescriptor gdesc{grad_input_t}; // grad_input descriptor
      // 创建空间变换器描述符 desc，用于采样器操作的描述
      SpatialTransformerDescriptor desc; // sampler descriptor
    
      // 获取当前的 cuDNN 句柄
      auto handle = getCudnnHandle();
      // 获取输入张量的数据类型
      auto dataType = getCudnnDataType(*input);
      // 设置采样器描述符，用于梯度输出张量的数据类型
      setSamplerDescriptor(desc, dataType, *grad_output);
    
      // 创建常量 one 和 zero，分别表示数据类型 dataType 的 1 和 0
      Constant one(dataType, 1);
      Constant zero(dataType, 0);
      // 调用 cuDNN 的空间变换器反向采样函数
      AT_CUDNN_CHECK(cudnnSpatialTfSamplerBackward(
          handle,
          desc.desc(),
          &one,
          idesc.desc(),
          input->const_data_ptr(),
          &zero,
          gdesc.desc(),
          grad_input_t.data_ptr(),
          &one,
          odesc.desc(),
          grad_output->const_data_ptr(),
          // 有趣的是，输出张量的数据不需要描述符
          grid->const_data_ptr(),
          &zero,
          grad_grid_t.data_ptr()));
    
      // 返回梯度输入张量和梯度网格张量的元组
      return std::tuple<Tensor, Tensor>{grad_input_t, grad_grid_t};
    }
}

} // namespace native
} // namespace at

#endif
```