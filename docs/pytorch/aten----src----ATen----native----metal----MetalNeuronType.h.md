# `.\pytorch\aten\src\ATen\native\metal\MetalNeuronType.h`

```py
#ifndef MetalNeuronType_h
#define MetalNeuronType_h

// 导入 Metal 相关头文件
#import <ATen/native/metal/mpscnn/MPSCNNNeuronOp.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

// 导入 ATen 头文件
#include <ATen/ATen.h>

// 定义命名空间 at::native::metal
namespace at::native::metal {

// 定义神经元类型枚举
enum class NeuronType {
  None,
  Clamp,
  Relu,
  Sigmoid,
  HardSigmoid,
  Tanh,
};

// 根据输出最小和最大值返回相应的神经元类型
static inline NeuronType neuronType(
    std::optional<c10::Scalar> output_min,
    std::optional<c10::Scalar> output_max) {
  // 设置正负无穷大常量
  float inf_max = std::numeric_limits<float>::infinity();
  float inf_min = -std::numeric_limits<float>::infinity();
  // 获取实际输出最大和最小值
  float output_max_ =
      output_max.has_value() ? output_max.value().toFloat() : inf_max;
  float output_min_ =
      output_min.has_value() ? output_min.value().toFloat() : inf_min;
  // 根据输出值判断返回的神经元类型
  if (output_max_ == inf_max && output_min_ == 0) {
    return NeuronType::Relu;
  } else if (output_max_ < inf_max && output_min_ > inf_min) {
    return NeuronType::Clamp;
  } else {
    return NeuronType::None;
  }
}

// 根据神经元类型返回对应的 MPSCNNNeuron 对象
static inline MPSCNNNeuron* neuron(NeuronType type) {
  // 根据类型返回相应的 MPSCNNNeuron 对象
  if (type == NeuronType::Relu) {
    return [MPSCNNNeuronOp relu];
  } else if (type == NeuronType::Sigmoid) {
    return [MPSCNNNeuronOp sigmoid];
  } else if (type == NeuronType::Tanh) {
    return [MPSCNNNeuronOp tanh];
  } else if (type == NeuronType::HardSigmoid) {
    return [MPSCNNNeuronOp hardSigmoid];
  } else {
    return nil;
  }
}

// 根据神经元类型返回对应的 MPSNNNeuronDescriptor 对象
API_AVAILABLE(ios(11.3), macos(10.13), macCatalyst(13.0))
static inline MPSNNNeuronDescriptor* neuronDescriptor(NeuronType type) {
  // 根据类型返回相应的 MPSNNNeuronDescriptor 对象
  if (type == NeuronType::Relu) {
    return [MPSCNNNeuronOpDescriptor reluDescriptor];
  } else if (type == NeuronType::Sigmoid) {
    return [MPSCNNNeuronOpDescriptor sigmoidDescriptor];
  } else if (type == NeuronType::Tanh) {
    return [MPSCNNNeuronOpDescriptor tanhDescriptor];
  } else if (type == NeuronType::HardSigmoid) {
    return [MPSCNNNeuronOpDescriptor hardSigmoidDescriptor];
  } else {
    return [MPSNNNeuronDescriptor cnnNeuronDescriptorWithType:MPSCNNNeuronTypeNone];
  }
}

} // namespace at::native::metal

#endif /* MetalNeuronType_h */
```