# `.\pytorch\aten\src\ATen\native\cudnn\ConvShared.h`

```py
#pragma once
#include <ATen/core/Tensor.h>

#include <ATen/cudnn/Descriptors.h>
#include <ATen/cudnn/Types.h>
#include <ATen/cudnn/cudnn-wrapper.h>
#include <ATen/native/ConvUtils.h>

namespace at {
namespace native {

// ---------------------------------------------------------------------
//
// Helper classes
//
// ---------------------------------------------------------------------

// This POD struct is used to let us easily compute hashes of the
// parameters
// 用于存储卷积操作参数的 POD 结构体
struct ConvolutionParams {
  c10::DeviceIndex device_id; // 设备索引
  cudnnDataType_t dataType; // 数据类型
  int input_size[2 + max_dim]; // 输入张量的尺寸
  uint8_t input_dim; // 输入张量的维度
  at::MemoryFormat memory_format; // 存储格式
  int weight_size[2 + max_dim]; // 权重张量的尺寸
  int padding[max_dim]; // 填充
  int stride[max_dim]; // 步幅
  int dilation[max_dim]; // 膨胀系数
  int64_t groups; // 分组卷积中的组数
  bool deterministic; // 是否确定性操作
  bool allow_tf32; // 是否允许 TF32 格式
  // NB: transposed purposely omitted: transposed just swaps
  // forward and backward, so you can reuse the benchmark entry,
  // transposed 参数故意省略：转置操作只是前向和反向的交换，因此可以重用基准条目，
};

// 重载流输出操作符，用于打印 ConvolutionParams 结构体内容
std::ostream& operator<<(std::ostream& out, const ConvolutionParams& params);

// NB: This can't be a constructor, because then ConvolutionParams
// would not be a POD anymore.
// TODO: Use TensorGeometry here instead of the entire Tensor, which we
// don't actually need.  (OTOH: We can always pass in
// grad_input/grad_output, so this is not very pressing)
// 设置卷积操作参数的函数，填充 ConvolutionParams 结构体
void setConvolutionParams(
    ConvolutionParams* params,
    const at::Tensor& input,
    const at::Tensor& weight,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool deterministic,
    bool allow_tf32,
    at::MemoryFormat memory_format);

// 从 ConvolutionParams 结构体生成可重现的字符串表示
std::string repro_from_args(const ConvolutionParams& args);

// ---------------------------------------------------------------------
//
// Raw functions
//
// ---------------------------------------------------------------------

// 使用 cuDNN 执行卷积前向传播，输出到预分配的 output 张量中
void raw_cudnn_convolution_forward_out(
    const Tensor& output,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool allow_tf32);

// 使用 cuDNN 执行卷积反向传播，计算输入梯度，输出到 grad_input 张量中
void raw_cudnn_convolution_backward_input_out(
    const at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool allow_tf32);

// 使用 cuDNN 执行卷积反向传播，计算权重梯度，输出到 grad_weight 张量中
void raw_cudnn_convolution_backward_weight_out(
    const Tensor& grad_weight,
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool allow_tf32);

// 使用 cuDNN 执行带 ReLU 的卷积加法操作，输出到预分配的 output 张量中
void raw_cudnn_convolution_add_relu_out(
    const Tensor& output,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& z,
    float alpha,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    // 参数1：确定性标志，指示是否进行确定性运算
    bool deterministic,
    // 参数2：允许使用 TF32 数据类型的标志，控制是否允许 TF32
    bool allow_tf32);
// 定义一个函数原型，用于执行原始的 CUDNN 卷积操作，带有 ReLU 激活的后备处理，输出到指定的张量
void raw_cudnn_convolution_add_relu_fallback_out(
    const Tensor& output,            // 输出张量
    const Tensor& input,             // 输入张量
    const Tensor& weight,            // 权重张量
    const Tensor& z,                 // z 张量
    float alpha,                     // alpha 参数
    const Tensor& bias,              // 偏置张量
    IntArrayRef stride,              // 步长数组
    IntArrayRef padding,             // 填充数组
    IntArrayRef dilation,            // 膨胀数组
    int64_t groups,                  // 分组数
    bool benchmark,                  // 是否进行基准测试
    bool deterministic,              // 是否确定性操作
    bool allow_tf32);                // 是否允许 TF32 操作

#if AT_CUDNN_ENABLED()

// 定义一个函数原型，用于执行原始的 CUDNN v7 卷积前向计算，输出到指定的张量
void raw_cudnn_convolution_forward_out_v7(
    const Tensor& output,            // 输出张量
    const Tensor& input,             // 输入张量
    const Tensor& weight,            // 权重张量
    IntArrayRef padding,             // 填充数组
    IntArrayRef stride,              // 步长数组
    IntArrayRef dilation,            // 膨胀数组
    int64_t groups,                  // 分组数
    bool benchmark,                  // 是否进行基准测试
    bool deterministic,              // 是否确定性操作
    bool allow_tf32);                // 是否允许 TF32 操作

// 定义一个函数原型，用于执行原始的 CUDNN v7 卷积反向输入计算，输出到指定的张量
void raw_cudnn_convolution_backward_input_out_v7(
    const at::Tensor& grad_input,    // 梯度输入张量
    const at::Tensor& grad_output,   // 梯度输出张量
    const at::Tensor& weight,        // 权重张量
    IntArrayRef padding,             // 填充数组
    IntArrayRef stride,              // 步长数组
    IntArrayRef dilation,            // 膨胀数组
    int64_t groups,                  // 分组数
    bool benchmark,                  // 是否进行基准测试
    bool deterministic,              // 是否确定性操作
    bool allow_tf32);                // 是否允许 TF32 操作

// 定义一个函数原型，用于执行原始的 CUDNN v7 卷积反向权重计算，输出到指定的张量
void raw_cudnn_convolution_backward_weight_out_v7(
    const Tensor& grad_weight,       // 梯度权重张量
    const Tensor& grad_output,       // 梯度输出张量
    const Tensor& input,             // 输入张量
    IntArrayRef padding,             // 填充数组
    IntArrayRef stride,              // 步长数组
    IntArrayRef dilation,            // 膨胀数组
    int64_t groups,                  // 分组数
    bool benchmark,                  // 是否进行基准测试
    bool deterministic,              // 是否确定性操作
    bool allow_tf32);                // 是否允许 TF32 操作

// 定义一个函数原型，用于执行原始的 CUDNN v7 卷积添加 ReLU 激活操作，输出到指定的张量
void raw_cudnn_convolution_add_relu_out_v7(
    const Tensor& output,            // 输出张量
    const Tensor& input,             // 输入张量
    const Tensor& weight,            // 权重张量
    const Tensor& z,                 // z 张量
    float alpha,                     // alpha 参数
    const Tensor& bias,              // 偏置张量
    IntArrayRef stride,              // 步长数组
    IntArrayRef padding,             // 填充数组
    IntArrayRef dilation,            // 膨胀数组
    int64_t groups,                  // 分组数
    bool benchmark,                  // 是否进行基准测试
    bool deterministic,              // 是否确定性操作
    bool allow_tf32);                // 是否允许 TF32 操作

#endif
} // namespace native
} // namespace at
```