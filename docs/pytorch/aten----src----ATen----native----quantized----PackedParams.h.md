# `.\pytorch\aten\src\ATen\native\quantized\PackedParams.h`

```
#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/core/ivalue.h>

// LinearPackedParamsBase 结构体，继承自 torch::jit::CustomClassHolder
struct LinearPackedParamsBase : public torch::jit::CustomClassHolder {
  // 纯虚函数，用于应用线性变换
  virtual at::Tensor apply(
      at::Tensor input,
      double output_scale,
      int64_t output_zero_point) = 0;

  // 纯虚函数，用于应用带 ReLU 的线性变换
  virtual at::Tensor apply_relu(
      at::Tensor input,
      double output_scale,
      int64_t output_zero_point) = 0;

  // out 变体的 apply 函数，未实现的情况下抛出异常
  virtual at::Tensor& apply_out(
      const at::Tensor& /*input*/,
      double /*output_scale*/,
      int64_t /*output_zero_point*/,
      at::Tensor& output) {
    throw std::runtime_error(
        "apply_out is not implemented for this packed "
        "parameter type");
    return output;
  }

  // out 变体的 apply_relu 函数，未实现的情况下抛出异常
  virtual at::Tensor& apply_relu_out(
      const at::Tensor& /*input*/,
      double /*output_scale*/,
      int64_t /*output_zero_point*/,
      at::Tensor& output) {
    throw std::runtime_error(
        "apply_relu_out is not implemented for this packed "
        "parameter type");
    return output;
  }

  // apply_with_input_q_dq_qweight_dq_output_fp32 函数，未实现的情况下抛出异常
  // 函数原理对应的是 quantized::linear_with_input_q_dq_qweight_dq_output_fp32
  // 参数:
  //    input: 输入的 float32 Tensor，在操作中将被量化为 quint8
  // 返回:
  //    Y: float32 Tensor
  virtual at::Tensor apply_with_input_q_dq_qweight_dq_output_fp32(
      at::Tensor input,
      double input_scale,
      int64_t input_zero_point) {
    throw std::runtime_error(
        "apply_with_input_q_dq_qweight_dq_output_fp32 is not implemented for this packed "
        "parameter type");
    return {};
  }

  // apply_with_input_q_dq_qweight_dq_relu_output_fp32 函数，未实现的情况下抛出异常
  // 函数原理对应的是 quantized::linear_with_input_q_dq_qweight_dq_relu_output_fp32
  // 参数:
  //    input: 输入的 float32 Tensor，在操作中将被量化为 quint8
  // 返回:
  //    float32 Tensor
  virtual at::Tensor apply_with_input_q_dq_qweight_dq_relu_output_fp32(
      at::Tensor input,
      double input_scale,
      int64_t input_zero_point) {
    throw std::runtime_error(
        "apply_with_input_q_dq_qweight_dq_relu_output_fp32 is not implemented for this packed "
        "parameter type");
    // 返回空的 Tensor
    return {};
  }
};
  // 返回一个空的字典
  return {};
}

virtual at::Tensor apply_dynamic(
    at::Tensor input,
    bool reduce_range = false) = 0;
// 定义一个纯虚函数，用于动态应用操作到输入张量上

virtual at::Tensor apply_dynamic_relu(
    at::Tensor input,
    bool reduce_range = false) = 0;
// 定义一个纯虚函数，用于动态应用ReLU操作到输入张量上

virtual at::Tensor& apply_dynamic_out(
    const at::Tensor& /* input */,
    at::Tensor& output,
    bool /* reduce_range */) {
  throw std::runtime_error(
      "apply_dynamic_out is not implemented for this packed "
      "parameter type");
  return output;
}
// 定义一个纯虚函数，用于将动态操作应用到输入张量，并将结果存入输出张量中

virtual at::Tensor& apply_dynamic_relu_out(
    const at::Tensor& /* input */,
    at::Tensor& output,
    bool /* reduce_range */) {
  throw std::runtime_error(
      "apply_dynamic_relu_out is not implemented for this packed "
      "parameter type");
  return output;
}
// 定义一个纯虚函数，用于将动态ReLU操作应用到输入张量，并将结果存入输出张量中

virtual std::tuple<at::Tensor, std::optional<at::Tensor>> unpack() = 0;
// 定义一个纯虚函数，用于解包操作，返回一个张量和一个可选的张量

virtual std::optional<at::Tensor> bias() = 0;
// 定义一个纯虚函数，用于获取偏置项，返回一个可选的张量

virtual void set_bias(std::optional<at::Tensor> /*bias*/) {
  throw std::runtime_error(
      "set_bias is not implemented for this packed "
      "parameter type");
}
// 定义一个虚函数，用于设置偏置项，但对于这种打包参数类型，抛出运行时错误
};

// 模板声明，定义一个具有默认空间维度为2的卷积参数包装基类
template <int kSpatialDim = 2>
struct ConvPackedParamsBase : public torch::jit::CustomClassHolder {
  
  // 纯虚函数，应用卷积操作到输入张量，并返回处理后的张量
  virtual at::Tensor apply(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point) = 0;
  
  // 纯虚函数，应用带ReLU激活函数的卷积操作到输入张量，并返回处理后的张量
  virtual at::Tensor apply_relu(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point) = 0;
  
  // 纯虚函数，应用动态范围调整的卷积操作到输入张量，并返回处理后的张量
  virtual at::Tensor apply_dynamic(
      const at::Tensor& input,
      bool reduce_range) = 0;

  // 纯虚函数，解包卷积参数并返回结果张量和可选的偏置张量
  virtual std::tuple<at::Tensor, std::optional<at::Tensor>> unpack() = 0;

  // 纯虚函数，返回卷积的步幅列表
  virtual torch::List<int64_t> stride() const = 0;
  
  // 纯虚函数，返回卷积的填充列表
  virtual torch::List<int64_t> padding() const = 0;
  
  // 纯虚函数，返回卷积的输出填充列表
  virtual torch::List<int64_t> output_padding() const = 0;
  
  // 纯虚函数，返回卷积的扩张列表
  virtual torch::List<int64_t> dilation() const = 0;
  
  // 纯虚函数，返回卷积的组数
  virtual int64_t groups() const = 0;
  
  // 纯虚函数，返回是否为转置卷积的布尔值
  virtual bool transpose() const = 0;
};


这段代码定义了一个C++结构体模板`ConvPackedParamsBase`，它是一个抽象基类，用于表示卷积操作的参数包装。包含了多个纯虚函数，用于描述不同类型卷积操作的行为和参数，以及获取这些参数的方法。
```